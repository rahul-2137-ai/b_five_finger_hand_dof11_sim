"""Thread-safe DDS bridge between the simulation environment and Unitree SDK2.

This bridge does NOT touch mj_data/mj_model directly.
Instead the simulation environment pushes
observation dicts into PublishLowState() and pulls commanded positions via
GetAction().  All DDS command handlers are protected by per-channel locks.

FIX: Uses OdoState_ (unitree_hg) instead of Odometry_ (nav_msgs) so the WBC's
     BodyStateProcessor can actually receive odometry data on rt/odostate.
"""

import sys
import struct
import threading
from typing import Dict, List, Tuple

import numpy as np
import scipy.spatial.transform
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__WirelessController_,
    unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
    unitree_hg_msg_dds__HandState_ as HandState_default,
    sensor_msgs_msg_dds__LaserScan_ as LaserScan_default,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_, MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__MotorCmd_ as MotorCmd_default,
    unitree_go_msg_dds__MotorState_ as MotorState_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, OdoState_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import LaserScan_, PointCloud2_, PointField_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_.PointField_Constants import FLOAT32_
from unitree_sdk2py.idl.tf2_msgs.msg.dds_ import TFMessage_
from unitree_sdk2py.idl.rosgraph_msgs.msg.dds_ import Clock_
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import (
    TransformStamped_,
    Transform_,
    Vector3_,
    Quaternion_,
)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_
from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_


class UnitreeSDK2Bridge:
    """Observation-driven DDS bridge for the G1 / H1-2 humanoid.

    The sim environment calls ``PublishLowState(obs)`` each physics step and
    reads back motor commands via the ``low_cmd`` / hand-cmd attributes (set
    asynchronously by DDS callbacks).
    """

    def __init__(self, config: dict):
        robot_type = config.get("ROBOT_TYPE", "g1_29dof")

        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
                unitree_hg_msg_dds__LowCmd_,
                unitree_hg_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__OdoState_ as OdoState_default,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                IMUState_,
                LowCmd_,
                LowState_,
            )

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif robot_type in ("h1", "go2"):
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_,
                unitree_go_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
                IMUState_,
                LowCmd_,
                LowState_,
            )

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            OdoState_default = None  # Not used for h1/go2
        else:
            raise ValueError(
                f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'."
            )

        self.num_body_motor = config["NUM_MOTORS"]
        self.num_hand_motor = config.get("NUM_HAND_MOTORS", 0)
        self.use_sensor = config.get("USE_SENSOR", False)

        # "dex3" (default, 7-DoF HandCmd on rt/dex3/*), "inspire" (6-DoF Inspire
        # DFX: a single MotorCmds_ on rt/inspire/cmd, MotorStates_ on rt/inspire/state),
        # or "brainco" (BrainCo Revo2 5-finger hand: per-hand MotorCmds_ on
        # rt/brainco/{left,right}/cmd and MotorStates_ on rt/brainco/{left,right}/state).
        self.hand_type = config.get("HAND_TYPE", "dex3")
        # Inspire combined hand layout (matches teleop Inspire_Controller_DFX):
        #   idx 0-5  = RIGHT hand [pinky, ring, middle, index, thumb-bend, thumb-rotation]
        #   idx 6-11 = LEFT  hand [pinky, ring, middle, index, thumb-bend, thumb-rotation]
        self.inspire_num_motor = config.get("NUM_HAND_MOTORS", 6)
        # BrainCo Revo2 per-hand layout (matches teleop Brainco_Controller / the
        # official motor table): idx 0-5 = [thumb, thumb-aux, index, middle, ring,
        # pinky], normalized [0,1] where 0.0 = open and 1.0 = closed.
        self.brainco_num_motor = config.get("NUM_HAND_MOTORS", 6)

        # ── Publishers ──────────────────────────────────────────────────
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        # OdoState — must use unitree_hg OdoState_, NOT nav_msgs Odometry_!
        # The WBC's BodyStateProcessor subscribes with OdoState_ type on rt/odostate.
        # DDS does type-based matching: publishing Odometry_ on the same topic is silently ignored.
        if "g1" in robot_type or "h1-2" in robot_type:
            self.odo_state = OdoState_default()
            self.odo_state_puber = ChannelPublisher("rt/odostate", OdoState_)
            self.odo_state_puber.Init()
        else:
            self.odo_state = None
            self.odo_state_puber = None

        self.torso_imu_state = IMUState_default()
        self.torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.torso_imu_puber.Init()

        if self.hand_type == "inspire":
            # Single combined hand-state publisher (12 motors), Inspire DFX protocol.
            self.inspire_hand_state = MotorStates_()
            self.inspire_hand_state.states = [
                MotorState_default() for _ in range(2 * self.inspire_num_motor)
            ]
            self.inspire_hand_state_puber = ChannelPublisher("rt/inspire/state", MotorStates_)
            self.inspire_hand_state_puber.Init()
        elif self.hand_type == "brainco":
            # Per-hand state publishers (6 motors each), BrainCo Revo2 protocol.
            self.brainco_left_hand_state = MotorStates_()
            self.brainco_left_hand_state.states = [
                MotorState_default() for _ in range(self.brainco_num_motor)
            ]
            self.brainco_left_hand_state_puber = ChannelPublisher("rt/brainco/left/state", MotorStates_)
            self.brainco_left_hand_state_puber.Init()

            self.brainco_right_hand_state = MotorStates_()
            self.brainco_right_hand_state.states = [
                MotorState_default() for _ in range(self.brainco_num_motor)
            ]
            self.brainco_right_hand_state_puber = ChannelPublisher("rt/brainco/right/state", MotorStates_)
            self.brainco_right_hand_state_puber.Init()
        else:
            self.left_hand_state = HandState_default()
            self.left_hand_state_puber = ChannelPublisher("rt/dex3/left/state", HandState_)
            self.left_hand_state_puber.Init()

            self.right_hand_state = HandState_default()
            self.right_hand_state_puber = ChannelPublisher("rt/dex3/right/state", HandState_)
            self.right_hand_state_puber.Init()

        self.lidar_scan = LaserScan_default()
        self.lidar_scan_puber = ChannelPublisher("rt/lidar/scan", LaserScan_)
        self.lidar_scan_puber.Init()

        # TF publisher
        self.tf_puber = ChannelPublisher("rt/tf", TFMessage_)
        self.tf_puber.Init()

        # Clock publisher (for ROS2 use_sim_time support)
        self.clock_puber = ChannelPublisher("rt/clock", Clock_)
        self.clock_puber.Init()

        # PointCloud2 publisher (for 3D lidar)
        self.pointcloud_puber = ChannelPublisher("rt/lidar/pcl", PointCloud2_)
        self.pointcloud_puber.Init()

        # Debug PointCloud2 publisher (for debug mode)
        self.pointcloud_debug_puber = ChannelPublisher("rt/lidar/pcl_debug", PointCloud2_)
        self.pointcloud_debug_puber.Init()

        # Pre-define PointCloud2 fields for XYZ (reused each publish)
        self._pcl_fields = [
            PointField_(name="x", offset=0, datatype=FLOAT32_, count=1),
            PointField_(name="y", offset=4, datatype=FLOAT32_, count=1),
            PointField_(name="z", offset=8, datatype=FLOAT32_, count=1),
        ]

        # ── Subscribers ─────────────────────────────────────────────────
        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self._low_cmd_handler, 1)

        if self.hand_type == "inspire":
            # Normalized [0,1] command per motor (1.0 = open, 0.0 = closed),
            # 12 entries (right 0-5, left 6-11). Default to open.
            self.inspire_cmd_lock = threading.Lock()
            self.inspire_hand_cmd_q = np.full(2 * self.inspire_num_motor, 1.0)
            self.inspire_hand_cmd_received = False
            self.inspire_hand_cmd_suber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
            self.inspire_hand_cmd_suber.Init(self._inspire_hand_cmd_handler, 1)
        elif self.hand_type == "brainco":
            # Normalized [0,1] command per motor (0.0 = open, 1.0 = closed),
            # 6 entries per hand. Default to open (0.0).
            self.brainco_cmd_lock = threading.Lock()
            self.brainco_left_hand_cmd_q = np.zeros(self.brainco_num_motor)
            self.brainco_right_hand_cmd_q = np.zeros(self.brainco_num_motor)
            self.brainco_hand_cmd_received = False
            self.brainco_left_hand_cmd_suber = ChannelSubscriber("rt/brainco/left/cmd", MotorCmds_)
            self.brainco_left_hand_cmd_suber.Init(self._brainco_left_hand_cmd_handler, 1)
            self.brainco_right_hand_cmd_suber = ChannelSubscriber("rt/brainco/right/cmd", MotorCmds_)
            self.brainco_right_hand_cmd_suber.Init(self._brainco_right_hand_cmd_handler, 1)
        else:
            self.left_hand_cmd = HandCmd_default()
            self.left_hand_cmd_suber = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
            self.left_hand_cmd_suber.Init(self._left_hand_cmd_handler, 1)

            self.right_hand_cmd = HandCmd_default()
            self.right_hand_cmd_suber = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
            self.right_hand_cmd_suber.Init(self._right_hand_cmd_handler, 1)

        # ── Per-channel locks ───────────────────────────────────────────
        self.low_cmd_lock = threading.Lock()
        self.left_hand_cmd_lock = threading.Lock()
        self.right_hand_cmd_lock = threading.Lock()

        # ── Wireless controller ─────────────────────────────────────────
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            "rt/wirelesscontroller", WirelessController_
        )
        self.wireless_controller_puber.Init()

        self.joystick = None
        self.key_map = {
            "R1": 0, "L1": 1, "start": 2, "select": 3,
            "R2": 4, "L2": 5, "F1": 6, "F2": 7,
            "A": 8, "B": 9, "X": 10, "Y": 11,
            "up": 12, "right": 13, "down": 14, "left": 15,
        }

        self.reset()

    # ── Command handlers (called from DDS reader threads) ───────────────

    def reset(self):
        with self.low_cmd_lock:
            self.low_cmd_received = False
            self.new_low_cmd = False
        with self.left_hand_cmd_lock:
            self.left_hand_cmd_received = False
            self.new_left_hand_cmd = False
        with self.right_hand_cmd_lock:
            self.right_hand_cmd_received = False
            self.new_right_hand_cmd = False

    def _low_cmd_handler(self, msg):
        with self.low_cmd_lock:
            self.low_cmd = msg
            self.low_cmd_received = True
            self.new_low_cmd = True

    def _left_hand_cmd_handler(self, msg):
        with self.left_hand_cmd_lock:
            self.left_hand_cmd = msg
            self.left_hand_cmd_received = True
            self.new_left_hand_cmd = True

    def _right_hand_cmd_handler(self, msg):
        with self.right_hand_cmd_lock:
            self.right_hand_cmd = msg
            self.right_hand_cmd_received = True
            self.new_right_hand_cmd = True

    def _inspire_hand_cmd_handler(self, msg):
        """Store the latest Inspire DFX MotorCmds_ (12 normalized [0,1] targets)."""
        n = 2 * self.inspire_num_motor
        with self.inspire_cmd_lock:
            for i in range(min(n, len(msg.cmds))):
                self.inspire_hand_cmd_q[i] = float(msg.cmds[i].q)
            self.inspire_hand_cmd_received = True

    def get_inspire_hand_cmd(self):
        """Return the latest 12-element normalized [0,1] Inspire hand command,
        or None if nothing has been received yet."""
        with self.inspire_cmd_lock:
            if not self.inspire_hand_cmd_received:
                return None
            return self.inspire_hand_cmd_q.copy()

    def _brainco_left_hand_cmd_handler(self, msg):
        """Store the latest BrainCo left-hand MotorCmds_ (6 normalized [0,1])."""
        n = self.brainco_num_motor
        with self.brainco_cmd_lock:
            for i in range(min(n, len(msg.cmds))):
                self.brainco_left_hand_cmd_q[i] = float(msg.cmds[i].q)
            self.brainco_hand_cmd_received = True

    def _brainco_right_hand_cmd_handler(self, msg):
        """Store the latest BrainCo right-hand MotorCmds_ (6 normalized [0,1])."""
        n = self.brainco_num_motor
        with self.brainco_cmd_lock:
            for i in range(min(n, len(msg.cmds))):
                self.brainco_right_hand_cmd_q[i] = float(msg.cmds[i].q)
            self.brainco_hand_cmd_received = True

    def get_brainco_hand_cmd(self):
        """Return the latest (left6, right6) normalized [0,1] BrainCo hand
        commands (0 = open, 1 = closed), or None if nothing received yet."""
        with self.brainco_cmd_lock:
            if not self.brainco_hand_cmd_received:
                return None
            return (self.brainco_left_hand_cmd_q.copy(),
                    self.brainco_right_hand_cmd_q.copy())

    def PublishBraincoHandState(self, left_norm, right_norm):
        """Publish normalized [0,1] hand state (0 = open, 1 = closed) as per-hand
        MotorStates_ on rt/brainco/{left,right}/state, in BrainCo motor order
        [thumb, thumb-aux, index, middle, ring, pinky]."""
        n = self.brainco_num_motor
        for i in range(n):
            self.brainco_left_hand_state.states[i].q = float(left_norm[i])
            self.brainco_right_hand_state.states[i].q = float(right_norm[i])
        self.brainco_left_hand_state_puber.Write(self.brainco_left_hand_state)
        self.brainco_right_hand_state_puber.Write(self.brainco_right_hand_state)

    def PublishInspireHandState(self, right_norm, left_norm):
        """Publish normalized [0,1] hand state as MotorStates_ on rt/inspire/state.

        Layout matches Inspire_Controller_DFX: states[0:6] = right hand,
        states[6:12] = left hand, in order [pinky, ring, middle, index,
        thumb-bend, thumb-rotation].
        """
        n = self.inspire_num_motor
        for i in range(n):
            self.inspire_hand_state.states[i].q = float(right_norm[i])
            self.inspire_hand_state.states[n + i].q = float(left_norm[i])
        self.inspire_hand_state_puber.Write(self.inspire_hand_state)

    def cmd_received(self) -> bool:
        with self.low_cmd_lock:
            a = self.low_cmd_received
        with self.left_hand_cmd_lock:
            b = self.left_hand_cmd_received
        with self.right_hand_cmd_lock:
            c = self.right_hand_cmd_received
        return a or b or c

    # ── Publish simulation state ────────────────────────────────────────

    def PublishLowState(self, obs: Dict[str, np.ndarray]):
        """Publish body/hand/IMU/odo state from an observation dict.

        Uses OdoState_ (unitree_hg) with flat fields: .position, .linear_velocity,
        .orientation, .angular_velocity — matching exactly what the WBC's
        BodyStateProcessor reads.
        """

        # Body motor state
        for i in range(self.num_body_motor):
            self.low_state.motor_state[i].q = float(obs["body_q"][i])
            self.low_state.motor_state[i].dq = float(obs["body_dq"][i])
            self.low_state.motor_state[i].ddq = float(obs["body_ddq"][i])
            self.low_state.motor_state[i].tau_est = float(obs["body_tau_est"][i])

        # OdoState — flat fields, matches original bridge exactly
        if self.odo_state is not None:
            self.odo_state.position[:] = obs["floating_base_pose"][:3]
            self.odo_state.linear_velocity[:] = obs["floating_base_vel"][:3]
            self.odo_state.orientation[:] = obs["floating_base_pose"][3:7]
            self.odo_state.angular_velocity[:] = obs["floating_base_vel"][3:6]

        # Pelvis IMU (from floating base)
        self.low_state.imu_state.quaternion[:] = obs["floating_base_pose"][3:7]
        self.low_state.imu_state.gyroscope[:] = obs["floating_base_vel"][3:6]
        self.low_state.imu_state.accelerometer[:] = obs["floating_base_acc"][:3]

        # Secondary (torso) IMU
        self.torso_imu_state.quaternion[:] = obs["secondary_imu_quat"]
        self.torso_imu_state.gyroscope[:] = obs["secondary_imu_vel"][3:6]

        # Timestamps and publish
        self.low_state.tick = int(obs["time"] * 1e3)
        self.low_state_puber.Write(self.low_state)

        if self.odo_state is not None:
            self.odo_state.tick = int(obs["time"] * 1e3)
            self.odo_state_puber.Write(self.odo_state)

        self.torso_imu_puber.Write(self.torso_imu_state)

        # Dex3 hand states (Inspire/BrainCo hands publish separately via
        # PublishInspireHandState / PublishBraincoHandState, in normalized units).
        if self.hand_type == "dex3":
            for i in range(self.num_hand_motor):
                self.left_hand_state.motor_state[i].q = float(obs["left_hand_q"][i])
                self.left_hand_state.motor_state[i].dq = float(obs["left_hand_dq"][i])
            self.left_hand_state_puber.Write(self.left_hand_state)

            for i in range(self.num_hand_motor):
                self.right_hand_state.motor_state[i].q = float(obs["right_hand_q"][i])
                self.right_hand_state.motor_state[i].dq = float(obs["right_hand_dq"][i])
            self.right_hand_state_puber.Write(self.right_hand_state)

    def PublishLidar(self, lidar_msg: dict, sim_time: float):
        """Publish lidar data as either LaserScan (2D) or PointCloud2 (3D)."""
        if lidar_msg is None:
            return

        if "points" in lidar_msg:
            self.PublishPointCloud2(
                lidar_msg["points"],
                sim_time,
                lidar_msg.get("frame_id", "lidar_link")
            )
        elif "debug_up" in lidar_msg:
            self.PublishPointCloud2(
                lidar_msg["debug_up"],
                sim_time,
                lidar_msg.get("frame_id", "lidar_link")
            )
            self.PublishPointCloud2(
                lidar_msg["debug_p"],
                sim_time,
                lidar_msg.get("frame_id", "lidar_link"),
                publisher=self.pointcloud_debug_puber
            )
        else:
            self.lidar_scan.header.stamp.sec = int(sim_time)
            self.lidar_scan.header.stamp.nanosec = int((sim_time % 1) * 1e9)
            self.lidar_scan.header.frame_id = "lidar_link"

            self.lidar_scan.angle_min = lidar_msg["angle_min"]
            self.lidar_scan.angle_max = lidar_msg["angle_max"]
            self.lidar_scan.angle_increment = lidar_msg["angle_increment"]
            self.lidar_scan.range_min = lidar_msg["range_min"]
            self.lidar_scan.range_max = lidar_msg["range_max"]
            self.lidar_scan.time_increment = lidar_msg["time_increment"]
            self.lidar_scan.ranges = lidar_msg["ranges"]
            self.lidar_scan.scan_time = lidar_msg["scan_time"]
            self.lidar_scan.intensities = lidar_msg.get("intensities", [])

            self.lidar_scan_puber.Write(self.lidar_scan)

    def PublishTF(self, transforms: List[Dict], sim_time: float):
        """Publish TF tree from pre-computed transforms."""
        if not transforms:
            return

        stamp_sec = int(sim_time)
        stamp_nanosec = int((sim_time % 1) * 1e9)

        tf_transforms = []
        for t in transforms:
            transform = TransformStamped_(
                header=Header_(
                    stamp=Time_(sec=stamp_sec, nanosec=stamp_nanosec),
                    frame_id=t["parent_frame"]
                ),
                child_frame_id=t["child_frame"],
                transform=Transform_(
                    translation=Vector3_(
                        x=float(t["translation"][0]),
                        y=float(t["translation"][1]),
                        z=float(t["translation"][2])
                    ),
                    rotation=Quaternion_(
                        x=float(t["rotation"][0]),
                        y=float(t["rotation"][1]),
                        z=float(t["rotation"][2]),
                        w=float(t["rotation"][3])
                    )
                )
            )
            tf_transforms.append(transform)

        tf_msg = TFMessage_(transforms=tf_transforms)
        self.tf_puber.Write(tf_msg)

    def PublishClock(self, sim_time: float):
        """Publish simulation clock for ROS2 time synchronization."""
        stamp_sec = int(sim_time)
        stamp_nanosec = int((sim_time % 1) * 1e9)

        clock_msg = Clock_(clock=Time_(sec=stamp_sec, nanosec=stamp_nanosec))
        self.clock_puber.Write(clock_msg)

    def PublishPointCloud2(self, points: List, sim_time: float, frame_id: str = "lidar_link", publisher=None):
        """Publish 3D point cloud as PointCloud2 message."""
        if not points:
            return

        if publisher is None:
            publisher = self.pointcloud_puber

        num_points = len(points)
        point_step = 12  # 3 floats * 4 bytes each

        data = bytearray(num_points * point_step)
        for i, pt in enumerate(points):
            if isinstance(pt, dict):
                x, y, z = pt["x"], pt["y"], pt["z"]
            else:
                x, y, z = pt[0], pt[1], pt[2]
            struct.pack_into('fff', data, i * point_step, x, y, z)

        stamp_sec = int(sim_time)
        stamp_nanosec = int((sim_time % 1) * 1e9)

        pcl_msg = PointCloud2_(
            header=Header_(
                stamp=Time_(sec=stamp_sec, nanosec=stamp_nanosec),
                frame_id=frame_id
            ),
            height=1,
            width=num_points,
            fields=self._pcl_fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=num_points * point_step,
            data=list(data),
            is_dense=True
        )
        publisher.Write(pcl_msg)

    # ── Retrieve latest commanded positions ─────────────────────────────

    def GetAction(self) -> Tuple[np.ndarray, bool, bool]:
        """Return the latest commanded joint positions from the controller."""
        with self.low_cmd_lock:
            body_q = [self.low_cmd.motor_cmd[i].q for i in range(self.num_body_motor)]
        with self.left_hand_cmd_lock:
            left_hand_q = [
                self.left_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)
            ]
        with self.right_hand_cmd_lock:
            right_hand_q = [
                self.right_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)
            ]

        with self.low_cmd_lock:
            with self.left_hand_cmd_lock:
                with self.right_hand_cmd_lock:
                    is_new = (
                        self.new_low_cmd
                        and self.new_left_hand_cmd
                        and self.new_right_hand_cmd
                    )
                    if is_new:
                        self.new_low_cmd = False
                        self.new_left_hand_cmd = False
                        self.new_right_hand_cmd = False

        return (
            np.concatenate([body_q[:-7], left_hand_q, body_q[-7:], right_hand_q]),
            self.cmd_received(),
            is_new,
        )

    # ── Wireless controller / joystick ──────────────────────────────────

    def PublishWirelessController(self):
        import pygame

        if self.joystick is None:
            return

        pygame.event.get()
        if hasattr(self.joystick, "update"):
            self.joystick.update()

        key_state = [0] * 16
        key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
        key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
        key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
        key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
        key_state[self.key_map["R2"]] = int(self.joystick.get_axis(self.axis_id["RT"]) > 0)
        key_state[self.key_map["L2"]] = int(self.joystick.get_axis(self.axis_id["LT"]) > 0)
        key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
        key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
        key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
        key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
        key_state[self.key_map["up"]] = int(self.joystick.get_hat(0)[1] > 0)
        key_state[self.key_map["right"]] = int(self.joystick.get_hat(0)[0] > 0)
        key_state[self.key_map["down"]] = int(self.joystick.get_hat(0)[1] < 0)
        key_state[self.key_map["left"]] = int(self.joystick.get_hat(0)[0] < 0)

        key_value = 0
        for i in range(16):
            key_value += key_state[i] << i

        self.wireless_controller.keys = key_value
        self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
        self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
        self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
        self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

        self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 3, "RY": 4,
                    "LT": 2, "RT": 5, "DX": 6, "DY": 7,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 4, "RB": 5, "SELECT": 6, "START": 7,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                    "LT": 4, "RT": 5,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 9, "RB": 10, "SELECT": 4, "START": 6,
                }
        elif js_type == "switch":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0,
                "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }


class ElasticBand:
    """Virtual spring-damper band with angular torque.

    Applies both linear force and angular torque restoring the attached body
    to an upright orientation.
    """

    def __init__(self, point=None):
        self.kp_pos = 10000
        self.kd_pos = 1000
        self.kp_ang = 1000
        self.kd_ang = 10
        # Anchor the band above the spawn point. Default sits over the world
        # origin so an origin-spawned robot is held upright (override via the
        # ELASTIC_BAND_POINT config key for scenes spawned elsewhere).
        self.point = np.array(point if point is not None else [0.0, 0.0, 1.0], dtype=float)
        self.length = 0.0
        self.enable = True

    def Advance(self, pose: np.ndarray) -> np.ndarray:
        """Compute 6-DOF wrench [force, torque] for the attached body.

        Args:
            pose: 13-element array [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)]
        """
        pos = pose[0:3]
        quat = pose[3:7]
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        dx = self.point - pos
        f = self.kp_pos * (dx + np.array([0, 0, self.length])) + self.kd_pos * (0 - lin_vel)

        # MuJoCo quaternion is [w,x,y,z]; scipy expects [x,y,z,w]
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot = scipy.spatial.transform.Rotation.from_quat(quat_scipy)
        rotvec = rot.as_rotvec()
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        import glfw

        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable

    def handle_keyboard_button(self, key):
        if key == "9":
            self.enable = not self.enable
            print(f"ElasticBand enable: {self.enable}")