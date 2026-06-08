import time
import math
from typing import Dict, List, Tuple

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from unitree_sdk2py_bridge import ElasticBand, UnitreeSDK2Bridge
from sim_config import load_config


# ═══════════════════════════════════════════════════════════════════════════
#  Quaternion utilities (MuJoCo uses w,x,y,z format)
# ═══════════════════════════════════════════════════════════════════════════

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute conjugate (inverse for unit quaternion) of quaternion (w, x, y, z format)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (w, x, y, z format)."""
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, q_v), q_conj)
    return result[1:4]

def _mujoco_pad_width(count: int) -> int:
    """Replicate digit width MuJoCo uses for a given count.

    Empirical rule (verified for MuJoCo replicate naming):
        count   1- 10 →  1 digit
        count  11-100 →  2 digits
        count 101-1000 → 3 digits
        ...
    i.e. width = max(1, len(str(count - 1))).
    """
    if count <= 1:
        return 1
    return len(str(count - 1))


# ── 3D LiDAR layout constants ─────────────────────────────────────────────
# These MUST stay in sync with the <replicate> blocks in g1_29dof_3d.xml.
# If you change the XML counts/steps, change these too — nothing else
# auto-detects geometry.
LIDAR3D_N_YAW   = 144       # outer replicate count
LIDAR3D_N_PITCH = 60        # inner replicate count
LIDAR3D_YAW_STEP        = 0.0436332    # rad  (2.5° per yaw  index)
LIDAR3D_PITCH_STEP      = 0.0261799    # rad  (1.5° per pitch index)
LIDAR3D_SITE_BASE_PITCH = 1.0471976    # rad  (60° = π/2 − 20·PITCH_STEP)

# Pre-compute the digit widths MuJoCo uses for these counts
LIDAR3D_YAW_PAD   = _mujoco_pad_width(LIDAR3D_N_YAW)    # 3
LIDAR3D_PITCH_PAD = _mujoco_pad_width(LIDAR3D_N_PITCH)  # 2
LIDAR3D_SUFFIX_LEN = LIDAR3D_YAW_PAD + LIDAR3D_PITCH_PAD  # 5


# ═══════════════════════════════════════════════════════════════════════════
#  Canonical WBC joint orderings
# ═══════════════════════════════════════════════════════════════════════════
# These define the EXACT order the WBC (Rahul WBC / Unitree SDK) sends and
# expects motor commands in.  The simulation MUST translate from this order
# to the (possibly different) MuJoCo kinematic-tree order of the loaded XML.
#
# Bug we are fixing: in some G1 XMLs (e.g. g1_29dof_3d.xml) the right hand
# fingers appear in kinematic order [thumb, index, middle], but the WBC
# always sends them in order [thumb, middle, index].  Discovering joints
# by keyword scan picks them up in XML order — that scrambles right-hand
# commands and the right wrist (because everything after a misordered
# block is also off).  Explicit name-based lookup avoids the problem.

# 29-DoF body motor order (matches LowCmd.motor_cmd[i] for i in 0..28):
#   legs (12) → waist (3) → left arm (7) → right arm (7)
WBC_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",     # 0
    "left_hip_roll_joint",      # 1
    "left_hip_yaw_joint",       # 2
    "left_knee_joint",          # 3
    "left_ankle_pitch_joint",   # 4
    "left_ankle_roll_joint",    # 5
    "right_hip_pitch_joint",    # 6
    "right_hip_roll_joint",     # 7
    "right_hip_yaw_joint",      # 8
    "right_knee_joint",         # 9
    "right_ankle_pitch_joint",  # 10
    "right_ankle_roll_joint",   # 11
    "waist_yaw_joint",          # 12
    "waist_roll_joint",         # 13
    "waist_pitch_joint",        # 14
    "left_shoulder_pitch_joint",   # 15
    "left_shoulder_roll_joint",    # 16
    "left_shoulder_yaw_joint",     # 17
    "left_elbow_joint",            # 18
    "left_wrist_roll_joint",       # 19
    "left_wrist_pitch_joint",      # 20
    "left_wrist_yaw_joint",        # 21
    "right_shoulder_pitch_joint",  # 22
    "right_shoulder_roll_joint",   # 23
    "right_shoulder_yaw_joint",    # 24
    "right_elbow_joint",           # 25
    "right_wrist_roll_joint",      # 26
    "right_wrist_pitch_joint",     # 27
    "right_wrist_yaw_joint",       # 28
]

# 7-DoF Dex3 hand motor order (matches HandCmd.motor_cmd[i] for i in 0..6).
# Defined by HandCommandSender / HandStateProcessor in the Rahul WBC.
WBC_LEFT_HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint",   # 0
    "left_hand_thumb_1_joint",   # 1
    "left_hand_thumb_2_joint",   # 2
    "left_hand_middle_0_joint",  # 3
    "left_hand_middle_1_joint",  # 4
    "left_hand_index_0_joint",   # 5
    "left_hand_index_1_joint",   # 6
]
WBC_RIGHT_HAND_JOINT_NAMES = [
    "right_hand_thumb_0_joint",   # 0
    "right_hand_thumb_1_joint",   # 1
    "right_hand_thumb_2_joint",   # 2
    "right_hand_middle_0_joint",  # 3
    "right_hand_middle_1_joint",  # 4
    "right_hand_index_0_joint",   # 5
    "right_hand_index_1_joint",   # 6
]

# 6-DoF Inspire 5-finger hand motor order, matching the teleop
# Inspire_Controller_DFX (rt/inspire/cmd MotorCmds_):
#   idx 0 pinky, 1 ring, 2 middle, 3 index, 4 thumb-bend, 5 thumb-rotation
# mapped to the actuated driver joints of g1_29dof_inspire.xml.
INSPIRE_LEFT_HAND_JOINT_NAMES = [
    "L_pinky_proximal_joint",          # 0  pinky
    "L_ring_proximal_joint",           # 1  ring
    "L_middle_proximal_joint",         # 2  middle
    "L_index_proximal_joint",          # 3  index
    "L_thumb_proximal_pitch_joint",    # 4  thumb-bend
    "L_thumb_proximal_yaw_joint",      # 5  thumb-rotation
]
INSPIRE_RIGHT_HAND_JOINT_NAMES = [
    "R_pinky_proximal_joint",          # 0  pinky
    "R_ring_proximal_joint",           # 1  ring
    "R_middle_proximal_joint",         # 2  middle
    "R_index_proximal_joint",          # 3  index
    "R_thumb_proximal_pitch_joint",    # 4  thumb-bend
    "R_thumb_proximal_yaw_joint",      # 5  thumb-rotation
]

# 6-DoF BrainCo Revo2 5-finger hand motor order, matching the teleop
# Brainco_Controller and the official Revo2 motor table (rt/brainco/*/cmd
# MotorCmds_): idx 0 thumb (= metacarpal/opposition, range ~1.57),
#              idx 1 thumb-aux (= thumb proximal flexion, range ~1.03),
#              idx 2 index, 3 middle, 4 ring, 5 pinky (proximal flexions).
# Each finger's distal joint is coupled to its proximal in the scene XML, so
# only these 6 driver joints per hand are actuated.
REVO2_LEFT_HAND_JOINT_NAMES = [
    "left_thumb_metacarpal_joint",     # 0  thumb (opposition)
    "left_thumb_proximal_joint",       # 1  thumb-aux (flexion)
    "left_index_proximal_joint",       # 2  index
    "left_middle_proximal_joint",      # 3  middle
    "left_ring_proximal_joint",        # 4  ring
    "left_pinky_proximal_joint",       # 5  pinky
]
REVO2_RIGHT_HAND_JOINT_NAMES = [
    "right_thumb_metacarpal_joint",    # 0  thumb (opposition)
    "right_thumb_proximal_joint",      # 1  thumb-aux (flexion)
    "right_index_proximal_joint",      # 2  index
    "right_middle_proximal_joint",     # 3  middle
    "right_ring_proximal_joint",       # 4  ring
    "right_pinky_proximal_joint",      # 5  pinky
]

# 43-element CONFIG-FILE order used by motor_effort_limit_list, motor_pos_*_limit_list,
# motor_vel_limit_list, etc.  This is DIFFERENT from MuJoCo actuator order:
#
#   config order:  legs(12) → waist(3) → L-arm(7) → L-hand(7) → R-arm(7) → R-hand(7)
#   MuJoCo XML  :  legs(12) → waist(3) → L-arm(7) → R-arm(7)  → L-hand(7) → R-hand(7)
#
# Mixing these up was the root cause of the "right arm not moving" bug:
# the right-arm actuators (MuJoCo idx 22-28) were being clipped with the
# left-hand effort limits (~0.7 Nm) from the config.
WBC_CONFIG_LIMIT_ORDER_NAMES = (
    WBC_BODY_JOINT_NAMES[0:15]                  # legs(12) + waist(3) = 15
    + WBC_BODY_JOINT_NAMES[15:22]               # left arm (7)
    + WBC_LEFT_HAND_JOINT_NAMES                 # left hand (7)
    + WBC_BODY_JOINT_NAMES[22:29]               # right arm (7)
    + WBC_RIGHT_HAND_JOINT_NAMES                # right hand (7)
)
# Total = 15 + 7 + 7 + 7 + 7 = 43


def _resolve_joints_by_name(mj_model, names):
    """Return MuJoCo joint IDs for a list of joint names.

    Skips names that do not exist in the model (returns a shorter list).
    """
    ids = []
    for name in names:
        try:
            jid = mj_model.joint(name).id
        except Exception:
            # Joint missing in this XML — skip silently
            continue
        ids.append(jid)
    return ids


def extract_3d_lidar_id(s: str):
    """Parse a 3D lidar site/sensor name and return (yaw_idx, pitch_idx).

    Naming follows MuJoCo nested replicate: INNER index first, OUTER last.
    With INNER = pitch (count=60, pad=2) and OUTER = yaw (count=144, pad=3),
    the name suffix is:
        "{pitch_idx:02d}{yaw_idx:03d}"
    e.g.  "lidar05143"  →  pitch_idx=5, yaw_idx=143.
    """
    suf = s[-LIDAR3D_SUFFIX_LEN:]
    pitch_idx = int(suf[:LIDAR3D_PITCH_PAD])
    yaw_idx   = int(suf[LIDAR3D_PITCH_PAD:])
    return (yaw_idx, pitch_idx)



# ═══════════════════════════════════════════════════════════════════════════
#  GearSimEnv — owns MuJoCo model/data, computes PD torques, steps physics
# ═══════════════════════════════════════════════════════════════════════════

class ModinoidSimEnv:
    """MuJoCo simulation environment for Modinoid, with built-in PD control and elastic band support, DDS publisher"""
    def __init__(self, config: dict):
        self.config = config
        self.sim_dt = config["SIMULATE_DT"]

        self.num_body_dof = config["NUM_JOINTS"]
        self.num_hand_dof = config.get("NUM_HAND_JOINTS", 0)
        # "dex3" (7-DoF, PD-torque on rt/dex3/*) or "inspire" (6-DoF position
        # servos driven by the Inspire DFX command on rt/inspire/cmd).
        self.hand_type = config.get("HAND_TYPE", "dex3")
        # NOTE: torque_limit is built later in _init_scene() so that it can be
        # reindexed from the config's WBC-order list into MuJoCo actuator order.
        # The raw config list is in config order (legs, waist, L-arm, L-hand,
        # R-arm, R-hand), but MuJoCo actuators are typically ordered
        # (legs, waist, L-arm, R-arm, L-hand, R-hand).  Indexing the raw list
        # by MuJoCo actuator ID would clip the right arm with hand-finger
        # limits (~0.7 Nm) — which is the "right arm not moving" bug.
        self.torque_limit = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.have_lidar = False
        self.lidar_type = config.get("LIDAR_TYPE", "2d")

        self.bridge = None  # set later via set_bridge()
        self.elastic_band = None

        self._init_scene()

    ## Scene Initialization
    def _init_scene(self):
        
        xml_path = self.config["ROBOT_SCENE"]
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.torso_index = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        self.root_body_id = self.mj_model.body("pelvis").id

        # Floating / constrained base detection
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        self.use_floating_root = "floating_base_joint" in joint_names
        self.use_constrained_root = "constrained_base_joint" in joint_names

        if self.use_floating_root:
            self.qpos_offset = 7
            self.qvel_offset = 6
        elif self.use_constrained_root:
            self.qpos_offset = 1
            self.qvel_offset = 1
        else:
            raise ValueError("No root link found — simulation would be unstable.")
        
        # Discover joint indices by CANONICAL WBC NAME (not by keyword scan).
        # This guarantees that self.body_joint_index[i] / self.left_hand_index[i] /
        # self.right_hand_index[i] always corresponds to the i-th WBC motor_cmd,
        # regardless of how the joints are ordered in the XML kinematic tree.
        #
        # Why this matters:
        #   The Rahul WBC publishes LowCmd.motor_cmd[0..28] in a FIXED order
        #   (legs → waist → left arm → right arm), and Dex3 HandCmd.motor_cmd[0..6]
        #   in a FIXED order (thumb_0..2, middle_0/1, index_0/1).
        #
        #   Some G1 XMLs (e.g. g1_29dof_3d.xml) put the right-hand "index"
        #   fingers BEFORE the "middle" fingers in the kinematic tree, so a
        #   plain keyword scan would produce a different order than the WBC
        #   expects.  We avoid that by looking each joint up by name.
        if self.hand_type == "inspire":
            left_hand_names = INSPIRE_LEFT_HAND_JOINT_NAMES
            right_hand_names = INSPIRE_RIGHT_HAND_JOINT_NAMES
        elif self.hand_type == "brainco":
            left_hand_names = REVO2_LEFT_HAND_JOINT_NAMES
            right_hand_names = REVO2_RIGHT_HAND_JOINT_NAMES
        else:
            left_hand_names = WBC_LEFT_HAND_JOINT_NAMES
            right_hand_names = WBC_RIGHT_HAND_JOINT_NAMES

        body_ids = _resolve_joints_by_name(self.mj_model, WBC_BODY_JOINT_NAMES)
        left_hand_ids = _resolve_joints_by_name(self.mj_model, left_hand_names)
        right_hand_ids = _resolve_joints_by_name(self.mj_model, right_hand_names)

        if len(body_ids) != self.num_body_dof:
            print(
                f"[WARN] modinoid_sim: expected {self.num_body_dof} body joints in WBC order, "
                f"found {len(body_ids)} in the XML. Missing joints will be ignored.\n"
                f"       Expected names: {WBC_BODY_JOINT_NAMES}"
            )

        self.body_joint_index = np.array(body_ids, dtype=np.intp)
        self.left_hand_index = np.array(left_hand_ids, dtype=np.intp)
        self.right_hand_index = np.array(right_hand_ids, dtype=np.intp)

        # ── Build joint-to-actuator mapping ────────────────────────────────
        # MuJoCo joint IDs and actuator IDs can differ when non-body joints
        # (e.g. hand joints) are interleaved in the kinematic tree between
        # body joints. The actuator list in the XML has all 29 body motors
        # first (indices 0-28), then hand motors. But the joint list has
        # left-hand joints between left-arm and right-arm body joints.
        #
        # We build a mapping from each body joint's MuJoCo joint ID to its
        # corresponding actuator index, by matching the joint each actuator
        # drives.
        self._joint_id_to_actuator = {}
        for act_i in range(self.mj_model.nu):
            # Each actuator's trnid[0] gives the joint ID it drives
            jnt_id = self.mj_model.actuator_trnid[act_i, 0]
            self._joint_id_to_actuator[jnt_id] = act_i

        def _resolve_actuators(joint_ids, group_name):
            """Map a list of joint IDs to actuator IDs, skipping joints with no actuator."""
            kept_joint_ids = []
            actuator_ids = []
            for jid in joint_ids:
                if jid in self._joint_id_to_actuator:
                    kept_joint_ids.append(jid)
                    actuator_ids.append(self._joint_id_to_actuator[jid])
                else:
                    jname = self.mj_model.joint(jid).name
                    print(
                        f"[WARN] modinoid_sim: {group_name} joint '{jname}' has no actuator "
                        f"in this XML — it will be observed but not driven."
                    )
            return (
                np.array(kept_joint_ids, dtype=np.intp),
                np.array(actuator_ids, dtype=np.intp),
            )

        # body_actuator_index[i] = actuator index for the i-th body motor
        self.body_joint_index, self.body_actuator_index = _resolve_actuators(
            self.body_joint_index, "body"
        )
        # hand actuator indices (same approach)
        self.left_hand_index, self.left_hand_actuator_index = _resolve_actuators(
            self.left_hand_index, "left-hand"
        )
        self.right_hand_index, self.right_hand_actuator_index = _resolve_actuators(
            self.right_hand_index, "right-hand"
        )

        # ── Precompute per-joint qpos / qvel addresses ────────────────────
        # Use MuJoCo's own jnt_qposadr / jnt_dofadr instead of the linear
        # "joint_id + qpos_offset - 1" formula.  The linear formula happens
        # to work when every joint before this one is single-DoF, but it
        # silently breaks the moment a multi-DoF joint (or extra free joint
        # for cubes in the scene) is added.  jnt_qposadr is the canonical
        # MuJoCo answer, so use it everywhere.
        self.body_qposadr = np.array(
            [self.mj_model.jnt_qposadr[jid] for jid in self.body_joint_index],
            dtype=np.intp,
        )
        self.body_dofadr = np.array(
            [self.mj_model.jnt_dofadr[jid] for jid in self.body_joint_index],
            dtype=np.intp,
        )
        self.left_hand_qposadr = np.array(
            [self.mj_model.jnt_qposadr[jid] for jid in self.left_hand_index],
            dtype=np.intp,
        )
        self.left_hand_dofadr = np.array(
            [self.mj_model.jnt_dofadr[jid] for jid in self.left_hand_index],
            dtype=np.intp,
        )
        self.right_hand_qposadr = np.array(
            [self.mj_model.jnt_qposadr[jid] for jid in self.right_hand_index],
            dtype=np.intp,
        )
        self.right_hand_dofadr = np.array(
            [self.mj_model.jnt_dofadr[jid] for jid in self.right_hand_index],
            dtype=np.intp,
        )

        # ── Fixed-stand mode ───────────────────────────────────────────────
        # When enabled, the pelvis floating base and the lower body (12 leg +
        # 3 waist joints) are rigidly held at a nominal standing pose every
        # step, so the humanoid stays upright with no balance controller. The
        # arms (and hands) still respond to commands normally — useful for
        # arm/teleop testing without a WBC. See _apply_fix_stand().
        self.fix_stand = self.config.get("FIX_STAND", False)
        if self.fix_stand:
            # Lower body = first 15 WBC joints (legs 0-11, waist 12-14).
            n_lower = min(15, len(self.body_qposadr))
            self.fix_lower_qposadr = self.body_qposadr[:n_lower]
            self.fix_lower_dofadr = self.body_dofadr[:n_lower]
            default_angles = np.asarray(self.config["DEFAULT_DOF_ANGLES"], dtype=np.float64)
            self.fix_lower_qpos = default_angles[:n_lower].copy()
            # Standing base pose: pelvis spawn position + orientation, both
            # taken from the scene XML so the robot keeps whatever yaw/facing
            # direction the pelvis body's quat defines (e.g. facing a room).
            pelvis_pos = self.mj_model.body("pelvis").pos.copy()
            pelvis_quat = self.mj_model.body("pelvis").quat.copy()
            self.fix_base_qpos = np.concatenate([pelvis_pos, pelvis_quat])

        # ── Inspire hand: position-servo ctrl ranges ──────────────────────
        # The Inspire DFX command is normalized [0,1] (1=open, 0=closed). We
        # de-normalize to a joint-angle setpoint using each finger actuator's
        # own ctrlrange, and normalize state back the same way.
        if self.hand_type in ("inspire", "brainco"):
            self.left_hand_ctrlrange = self.mj_model.actuator_ctrlrange[
                self.left_hand_actuator_index
            ].copy()
            self.right_hand_ctrlrange = self.mj_model.actuator_ctrlrange[
                self.right_hand_actuator_index
            ].copy()

        # ── Build torque_limit indexed by MuJoCo actuator ID ─────────────
        # The config's motor_effort_limit_list is laid out in WBC config order
        # (legs, waist, L-arm, L-hand, R-arm, R-hand).  Self.torques is indexed
        # by MuJoCo actuator ID, which is typically a different order
        # (legs, waist, L-arm, R-arm, L-hand, R-hand).  Without this remap,
        # `np.clip(torques, -limit, limit)` clips the right arm to ~0.7 Nm
        # (left-hand finger limit) and the robot's right arm becomes dead.
        raw_limits = list(self.config["motor_effort_limit_list"])
        # Map joint NAME -> effort limit using the config's known ordering.
        name_to_limit = {}
        for name, lim in zip(WBC_CONFIG_LIMIT_ORDER_NAMES, raw_limits):
            name_to_limit[name] = float(lim)

        # Now fill torque_limit in MuJoCo actuator order (one entry per actuator).
        torque_limit_actuator_order = np.full(self.mj_model.nu, np.inf, dtype=np.float64)
        for act_i in range(self.mj_model.nu):
            jid = self.mj_model.actuator_trnid[act_i, 0]
            jname = self.mj_model.joint(jid).name
            if jname in name_to_limit:
                torque_limit_actuator_order[act_i] = name_to_limit[jname]
            else:
                # Joint not in config list — fall back to MuJoCo's own limit if any
                lo, hi = self.mj_model.actuator_forcerange[act_i]
                if hi > lo:
                    torque_limit_actuator_order[act_i] = float(max(abs(lo), abs(hi)))
                # else leave as +inf (no clipping)
        self.torque_limit = torque_limit_actuator_order

        # self.torques must match self.mj_data.ctrl shape (one per actuator)
        self.torques = np.zeros(self.mj_model.nu, dtype=np.float64)

        if self.config.get("PRINT_SCENE_INFORMATION", False):
            print(f"Body joints: {len(self.body_joint_index)}, "
                  f"Left hand: {len(self.left_hand_index)}, "
                  f"Right hand: {len(self.right_hand_index)}")
            for i, (jid, aid) in enumerate(zip(self.body_joint_index, self.body_actuator_index)):
                jname = self.mj_model.joint(jid).name
                aname = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                print(f"  motor[{i:2d}] joint={jid:2d} ({jname:35s}) -> actuator={aid:2d} ({aname:25s}) lim=±{self.torque_limit[aid]:.2f}")
            for i, (jid, aid) in enumerate(zip(self.left_hand_index, self.left_hand_actuator_index)):
                jname = self.mj_model.joint(jid).name
                aname = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                print(f"  L-hand[{i:2d}] joint={jid:2d} ({jname:35s}) -> actuator={aid:2d} ({aname:25s}) lim=±{self.torque_limit[aid]:.2f}")
            for i, (jid, aid) in enumerate(zip(self.right_hand_index, self.right_hand_actuator_index)):
                jname = self.mj_model.joint(jid).name
                aname = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
                print(f"  R-hand[{i:2d}] joint={jid:2d} ({jname:35s}) -> actuator={aid:2d} ({aname:25s}) lim=±{self.torque_limit[aid]:.2f}")

        # LIDAR

        ## Discover lidar sensors and cache their sensordata addresses
        self.lidar_sensors = []  # list of (sensor_id, name, sensordata_address, site_id)

        # For 3D the site suffix length depends on yaw/pitch padding (e.g. 5 for 144×60).
        # For 2D/debug there is no equivalent, so we keep the legacy 4-char slice.
        if self.lidar_type.startswith("3d"):
            _suf_len = LIDAR3D_SUFFIX_LEN
        else:
            _suf_len = 4

        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name and name.startswith("lidar"):
                addr = self.mj_model.sensor_adr[i]
                site_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SITE, f"rf{name[-_suf_len:]}"
                )
                self.lidar_sensors.append((i, name, addr, site_id))
            
        if self.lidar_type.startswith("3d"):
            self.lidar_sensors.sort(key=lambda x: extract_3d_lidar_id(x[1]))  # sort by id in name
        elif self.lidar_type in ("2d", "debug"):
            self.lidar_sensors.sort(key=lambda x: x[1])  # angular order
        self.num_lidar_rays = len(self.lidar_sensors)
        self._lidar_addrs = [s[2] for s in self.lidar_sensors]

        ## Pre-compute LaserScan geometry (computed once)
        if self.num_lidar_rays > 0:
            self.have_lidar = True
            self.lidar_angle_increment = 2.0 * math.pi / self.num_lidar_rays
            self.lidar_angle_min = 0.0
            self.lidar_angle_max = self.lidar_angle_min + self.lidar_angle_increment * (self.num_lidar_rays - 1)
            self.lidar_range_min = 0.05
            self.lidar_range_max = 50000.0
            print(f"Discovered {self.num_lidar_rays} lidar sensors")

        # if self.lidar_type == "3d":
        #     raise NotImplementedError("3D lidar not implemented yet")
        
        # TF tree body hierarchy
        self.tf_base_frame = self.config.get("TF_BASE_FRAME", "world")
        self._build_tf_hierarchy()

        # Elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root:
            self.elastic_band = ElasticBand(point=self.config.get("ELASTIC_BAND_POINT"))
            robot_type = self.config.get("ROBOT_TYPE", "g1")
            if "g1" in robot_type:
                if self.config.get("enable_waist", True):
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

        # Initialise the standing pose so the robot starts upright (before the
        # viewer opens) when fixed-stand mode is active.
        if self.fix_stand:
            self._apply_fix_stand()
            mujoco.mj_forward(self.mj_model, self.mj_data)

        # Viewer
        if self.config.get("ENABLE_ONSCREEN", True):
            if self.elastic_band is not None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.root_body_id
        else:
            self.viewer = None        

    # ── Bridge ──────────────────────────────────────────────────────────

    def set_bridge(self, bridge: UnitreeSDK2Bridge):
        self.bridge = bridge

    # ── TF tree building ─────────────────────────────────────────────────

    def _build_tf_hierarchy(self):
        """Build body hierarchy from MuJoCo model for TF publishing."""
        self.tf_body_info = []  # List of (body_id, body_name, parent_id, parent_name)

        for i in range(self.mj_model.nbody):
            body_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i
            )
            if body_name is None:
                body_name = f"body_{i}"

            parent_id = self.mj_model.body_parentid[i]

            if parent_id == 0 and i != 0:
                # Parent is world
                parent_name = self.tf_base_frame
            elif i == 0:
                # This is the world body itself, skip it
                continue
            else:
                parent_name = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, parent_id
                )
                if parent_name is None:
                    parent_name = f"body_{parent_id}"

            self.tf_body_info.append((i, body_name, parent_id, parent_name))

        print(f"TF: Built hierarchy with {len(self.tf_body_info)} bodies")

    def _compute_relative_transform(
        self,
        parent_pos: np.ndarray,
        parent_quat: np.ndarray,
        child_pos: np.ndarray,
        child_quat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute relative transform from parent to child frame.

        Args:
            parent_pos: Parent position in world frame (3,)
            parent_quat: Parent quaternion in world frame (w, x, y, z) (4,)
            child_pos: Child position in world frame (3,)
            child_quat: Child quaternion in world frame (w, x, y, z) (4,)

        Returns:
            (relative_pos, relative_quat): Transform from parent to child
        """
        # Compute position difference in world frame
        pos_diff = child_pos - parent_pos

        # Rotate position difference into parent frame
        parent_quat_conj = quat_conjugate(parent_quat)
        relative_pos = quat_rotate_vector(parent_quat_conj, pos_diff)

        # Compute relative rotation: q_rel = q_parent^-1 * q_child
        relative_quat = quat_multiply(parent_quat_conj, child_quat)

        # Normalize quaternion
        relative_quat = relative_quat / np.linalg.norm(relative_quat)

        return relative_pos, relative_quat

    def prepare_tf_transforms(self) -> List[Dict]:
        """Compute TF transforms from current MuJoCo state.

        Returns:
            List of transform dicts ready for bridge.PublishTF()
        """
        transforms = []

        for body_id, body_name, parent_id, parent_name in self.tf_body_info:
            # Get world poses
            child_pos = self.mj_data.xpos[body_id].copy()
            child_quat = self.mj_data.xquat[body_id].copy()  # MuJoCo: [w, x, y, z]

            if parent_id == 0:
                # Parent is world frame (identity transform)
                parent_pos = np.zeros(3)
                parent_quat = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                parent_pos = self.mj_data.xpos[parent_id].copy()
                parent_quat = self.mj_data.xquat[parent_id].copy()

            # Compute relative transform
            rel_pos, rel_quat = self._compute_relative_transform(
                parent_pos, parent_quat, child_pos, child_quat
            )

            # Build transform dict with ROS quaternion order (x, y, z, w)
            transforms.append({
                "parent_frame": parent_name,
                "child_frame": body_name,
                "translation": (float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2])),
                "rotation": (float(rel_quat[1]), float(rel_quat[2]), float(rel_quat[3]), float(rel_quat[0])),
            })

        return transforms

    # ── Observation gathering ───────────────────────────────────────────

    def prepare_obs(self) -> Dict[str, np.ndarray]:
        obs = {}

        if self.use_floating_root:
            obs["floating_base_pose"] = self.mj_data.qpos[:7].copy()
            obs["floating_base_vel"] = self.mj_data.qvel[:6].copy()
            obs["floating_base_acc"] = self.mj_data.qacc[:6].copy()
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index].copy()

        # Secondary IMU velocity (torso link)
        pose = np.zeros(13)
        torso_id = self.mj_model.body("torso_link").id
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_id,
            pose[7:13], 1,
        )
        # mj_objectVelocity returns [ang, lin]; swap to [lin, ang]
        pose[7:10], pose[10:13] = pose[10:13].copy(), pose[7:10].copy()
        obs["secondary_imu_vel"] = pose[7:13]

        # Body joints — index by the joint's own qpos/qvel address.
        # Pad to self.num_body_dof so the DDS bridge always sees the WBC's
        # expected motor count even if the XML omits some joints.
        def _take_and_pad(src, addr, size):
            out = np.zeros(size, dtype=src.dtype)
            n = min(len(addr), size)
            if n > 0:
                out[:n] = src[addr[:n]]
            return out

        obs["body_q"] = _take_and_pad(self.mj_data.qpos, self.body_qposadr, self.num_body_dof)
        obs["body_dq"] = _take_and_pad(self.mj_data.qvel, self.body_dofadr, self.num_body_dof)
        obs["body_ddq"] = _take_and_pad(self.mj_data.qacc, self.body_dofadr, self.num_body_dof)
        body_tau = np.zeros(self.num_body_dof)
        n_b = min(len(self.body_actuator_index), self.num_body_dof)
        if n_b > 0:
            body_tau[:n_b] = self.mj_data.actuator_force[self.body_actuator_index[:n_b]]
        obs["body_tau_est"] = body_tau

        # Hand joints (each hand is num_hand_dof joints in WBC order)
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = _take_and_pad(
                self.mj_data.qpos, self.left_hand_qposadr, self.num_hand_dof
            )
            obs["left_hand_dq"] = _take_and_pad(
                self.mj_data.qvel, self.left_hand_dofadr, self.num_hand_dof
            )
            obs["left_hand_ddq"] = _take_and_pad(
                self.mj_data.qacc, self.left_hand_dofadr, self.num_hand_dof
            )
            l_tau = np.zeros(self.num_hand_dof)
            n_l = min(len(self.left_hand_actuator_index), self.num_hand_dof)
            if n_l > 0:
                l_tau[:n_l] = self.mj_data.actuator_force[self.left_hand_actuator_index[:n_l]]
            obs["left_hand_tau_est"] = l_tau

            obs["right_hand_q"] = _take_and_pad(
                self.mj_data.qpos, self.right_hand_qposadr, self.num_hand_dof
            )
            obs["right_hand_dq"] = _take_and_pad(
                self.mj_data.qvel, self.right_hand_dofadr, self.num_hand_dof
            )
            obs["right_hand_ddq"] = _take_and_pad(
                self.mj_data.qacc, self.right_hand_dofadr, self.num_hand_dof
            )
            r_tau = np.zeros(self.num_hand_dof)
            n_r = min(len(self.right_hand_actuator_index), self.num_hand_dof)
            if n_r > 0:
                r_tau[:n_r] = self.mj_data.actuator_force[self.right_hand_actuator_index[:n_r]]
            obs["right_hand_tau_est"] = r_tau
        else:
            obs["left_hand_q"] = np.zeros(0)
            obs["left_hand_dq"] = np.zeros(0)
            obs["left_hand_ddq"] = np.zeros(0)
            obs["left_hand_tau_est"] = np.zeros(0)
            obs["right_hand_q"] = np.zeros(0)
            obs["right_hand_dq"] = np.zeros(0)
            obs["right_hand_ddq"] = np.zeros(0)
            obs["right_hand_tau_est"] = np.zeros(0)

        obs["time"] = self.mj_data.time
        
        return obs

    # ── PD torque computation ───────────────────────────────────────────

    def compute_body_torques(self) -> np.ndarray:
        """PD control for the 29 body motors: tau = tau_ff + kp*(q_des-q) + kd*(dq_des-dq)."""
        tau = np.zeros(self.num_body_dof)
        if self.bridge is None or not self.bridge.low_cmd_received:
            return tau

        with self.bridge.low_cmd_lock:
            cmd = self.bridge.low_cmd

        n = min(self.bridge.num_body_motor, len(self.body_joint_index))
        for i in range(n):
            q_act = self.mj_data.qpos[self.body_qposadr[i]]
            dq_act = self.mj_data.qvel[self.body_dofadr[i]]
            tau[i] = (
                cmd.motor_cmd[i].tau
                + cmd.motor_cmd[i].kp * (cmd.motor_cmd[i].q - q_act)
                + cmd.motor_cmd[i].kd * (cmd.motor_cmd[i].dq - dq_act)
            )
        return tau

    def compute_hand_torques(self) -> np.ndarray:
        """PD control for left (7) + right (7) hand motors."""
        left = np.zeros(self.num_hand_dof)
        right = np.zeros(self.num_hand_dof)
        if self.bridge is None or not self.bridge.low_cmd_received:
            return np.concatenate((left, right))

        with self.bridge.left_hand_cmd_lock:
            lcmd = self.bridge.left_hand_cmd
        with self.bridge.right_hand_cmd_lock:
            rcmd = self.bridge.right_hand_cmd

        n_left = min(self.bridge.num_hand_motor, len(self.left_hand_index))
        for i in range(n_left):
            q_l = self.mj_data.qpos[self.left_hand_qposadr[i]]
            dq_l = self.mj_data.qvel[self.left_hand_dofadr[i]]
            left[i] = (
                lcmd.motor_cmd[i].tau
                + lcmd.motor_cmd[i].kp * (lcmd.motor_cmd[i].q - q_l)
                + lcmd.motor_cmd[i].kd * (lcmd.motor_cmd[i].dq - dq_l)
            )

        n_right = min(self.bridge.num_hand_motor, len(self.right_hand_index))
        for i in range(n_right):
            q_r = self.mj_data.qpos[self.right_hand_qposadr[i]]
            dq_r = self.mj_data.qvel[self.right_hand_dofadr[i]]
            right[i] = (
                rcmd.motor_cmd[i].tau
                + rcmd.motor_cmd[i].kp * (rcmd.motor_cmd[i].q - q_r)
                + rcmd.motor_cmd[i].kd * (rcmd.motor_cmd[i].dq - dq_r)
            )
        return np.concatenate((left, right))

    # ── Inspire 5-finger hand: position-servo targets & state ────────────

    @staticmethod
    def _denorm(norm, ctrlrange):
        """Map normalized [0,1] (1=open, 0=closed) -> joint-angle setpoint.

        open  -> ctrlrange lo,  closed -> ctrlrange hi  (matches the URDF
        convention where 0 rad = open and the positive limit = fully curled).
        """
        k = min(len(norm), len(ctrlrange))
        lo = ctrlrange[:k, 0]
        hi = ctrlrange[:k, 1]
        n = np.clip(norm[:k], 0.0, 1.0)
        return lo + (1.0 - n) * (hi - lo)

    @staticmethod
    def _norm(q, ctrlrange):
        """Inverse of _denorm: joint angle -> normalized [0,1] (1=open)."""
        k = min(len(q), len(ctrlrange))
        lo = ctrlrange[:k, 0]
        hi = ctrlrange[:k, 1]
        rng = np.where((hi - lo) != 0.0, hi - lo, 1.0)
        return np.clip((hi - q[:k]) / rng, 0.0, 1.0)

    def compute_inspire_hand_targets(self):
        """De-normalize the latest Inspire DFX command into (left, right) finger
        position-servo setpoints, or None if no command has arrived yet."""
        if self.bridge is None:
            return None
        cmd = self.bridge.get_inspire_hand_cmd()  # 12 normalized [0,1] or None
        if cmd is None:
            return None
        n = self.num_hand_dof
        right_t = self._denorm(np.asarray(cmd[:n]), self.right_hand_ctrlrange)
        left_t = self._denorm(np.asarray(cmd[n:2 * n]), self.left_hand_ctrlrange)
        return left_t, right_t

    def inspire_hand_state_normalized(self):
        """Return (left_norm, right_norm) — current finger angles as [0,1]."""
        n = self.num_hand_dof
        left = np.ones(n)
        right = np.ones(n)
        lq = self.mj_data.qpos[self.left_hand_qposadr]
        rq = self.mj_data.qpos[self.right_hand_qposadr]
        ln = self._norm(lq, self.left_hand_ctrlrange)
        rn = self._norm(rq, self.right_hand_ctrlrange)
        left[:len(ln)] = ln
        right[:len(rn)] = rn
        return left, right

    # ── BrainCo Revo2 5-finger hand: position-servo targets & state ──────
    # BrainCo convention is the OPPOSITE of Inspire: normalized [0,1] with
    # 0.0 = fully open and 1.0 = fully closed. The Revo2 joints all run from
    # 0 rad (open) to their positive limit (closed), so the mapping is direct.

    @staticmethod
    def _denorm_brainco(norm, ctrlrange):
        """Map normalized [0,1] (0=open, 1=closed) -> joint-angle setpoint.

        open -> ctrlrange lo (0 rad), closed -> ctrlrange hi.
        """
        k = min(len(norm), len(ctrlrange))
        lo = ctrlrange[:k, 0]
        hi = ctrlrange[:k, 1]
        n = np.clip(norm[:k], 0.0, 1.0)
        return lo + n * (hi - lo)

    @staticmethod
    def _norm_brainco(q, ctrlrange):
        """Inverse of _denorm_brainco: joint angle -> normalized [0,1] (0=open)."""
        k = min(len(q), len(ctrlrange))
        lo = ctrlrange[:k, 0]
        hi = ctrlrange[:k, 1]
        rng = np.where((hi - lo) != 0.0, hi - lo, 1.0)
        return np.clip((q[:k] - lo) / rng, 0.0, 1.0)

    def compute_brainco_hand_targets(self):
        """De-normalize the latest BrainCo command into (left, right) finger
        position-servo setpoints, or None if no command has arrived yet."""
        if self.bridge is None:
            return None
        cmd = self.bridge.get_brainco_hand_cmd()  # (left6, right6) normalized or None
        if cmd is None:
            return None
        left_cmd, right_cmd = cmd
        left_t = self._denorm_brainco(np.asarray(left_cmd), self.left_hand_ctrlrange)
        right_t = self._denorm_brainco(np.asarray(right_cmd), self.right_hand_ctrlrange)
        return left_t, right_t

    def brainco_hand_state_normalized(self):
        """Return (left_norm, right_norm) — current finger angles as [0,1]
        (0=open, 1=closed), in BrainCo motor order."""
        n = self.num_hand_dof
        left = np.zeros(n)
        right = np.zeros(n)
        lq = self.mj_data.qpos[self.left_hand_qposadr]
        rq = self.mj_data.qpos[self.right_hand_qposadr]
        ln = self._norm_brainco(lq, self.left_hand_ctrlrange)
        rn = self._norm_brainco(rq, self.right_hand_ctrlrange)
        left[:len(ln)] = ln
        right[:len(rn)] = rn
        return left, right

    def build_lidar_message(self):
        """Build a laserscan dictionary from the current lidar sensor readings."""
        if not self.have_lidar:
            return None

        ranges = np.array([float(self.mj_data.sensordata[addr]) for addr in self._lidar_addrs])
        clean_ranges = []
        for r in ranges:
            if r < self.lidar_range_min:
                clean_ranges.append(float('nan'))  # too close
            elif r > self.lidar_range_max:
                clean_ranges.append(float('inf'))  # too far
            else:
                clean_ranges.append(r)

        if self.lidar_type == "2d":

            return {
                "angle_min": self.lidar_angle_min,
                "angle_max": self.lidar_angle_max,
                "angle_increment": self.lidar_angle_increment,
                "range_min": self.lidar_range_min,
                "range_max": self.lidar_range_max,
                "ranges": clean_ranges,
                "scan_time": float(self.sim_dt),
                "time_increment": float(0.0)
            }

        elif self.lidar_type == "3d_privileged":
            # Get lidar_link frame for transforming points to local frame
            try:
                lidar_link_id = self.mj_model.body("lidar_link").id
                lidar_pos = self.mj_data.xpos[lidar_link_id].copy()
                lidar_quat = self.mj_data.xquat[lidar_link_id].copy()  # [w,x,y,z]
            except KeyError:
                lidar_pos = np.zeros(3)
                lidar_quat = np.array([1.0, 0.0, 0.0, 0.0])

            # With data="dist point", sensordata layout is: [dist, x, y, z] per sensor
            points = []
            min_dist = self.config.get("LIDAR_MIN_DIST", self.lidar_range_min)
            max_dist = self.config.get("LIDAR_MAX_DIST", self.lidar_range_max)
            for sensor in self.lidar_sensors:
                addr = sensor[2]
                dist = self.mj_data.sensordata[addr]

                # Skip invalid readings (dist = -1 means no hit) or outside distance range
                if dist < 0 or dist < min_dist or dist > max_dist:
                    continue

                # Read hit point in global frame (already computed by MuJoCo)
                point_global = np.array([
                    self.mj_data.sensordata[addr + 1],
                    self.mj_data.sensordata[addr + 2],
                    self.mj_data.sensordata[addr + 3]
                ])

                # Transform to lidar_link frame
                point_local = quat_rotate_vector(
                    quat_conjugate(lidar_quat),
                    point_global - lidar_pos
                )
                points.append(point_local)

            # Truncate to max points limit (default = full grid size)
            default_max = LIDAR3D_N_YAW * LIDAR3D_N_PITCH
            max_points = self.config.get("LIDAR_MAX_POINTS", default_max)
            if len(points) > max_points:
                points = points[:max_points]

            return {
                "points": points,
                "frame_id": "lidar_link",
            }

        elif self.lidar_type == "3d_unprivileged":
            # Unprivileged 3D lidar: compute hit points from sensor name + distance.
            # All geometry constants live at module top so the extractor and
            # this branch stay in sync — see LIDAR3D_* near extract_3d_lidar_id().
            #
            # XML layout (g1_29dof_3d.xml):
            #   <replicate count=N_YAW   euler="0 0 YAW_STEP">    <- OUTER: yaw
            #     <replicate count=N_PITCH euler="0 PITCH_STEP 0"> <- INNER: pitch
            #       <site name="rf" euler="0 SITE_BASE_PITCH 0"/>
            #
            # Ray direction (site +Z axis) in lidar_link frame:
            #   d = [ cos(yaw)*sin(alpha), sin(yaw)*sin(alpha), cos(alpha) ]
            #   where alpha = SITE_BASE_PITCH + pitch_idx*PITCH_STEP
            #         yaw   = yaw_idx*YAW_STEP

            YAW_STEP        = LIDAR3D_YAW_STEP
            PITCH_STEP      = LIDAR3D_PITCH_STEP
            SITE_BASE_PITCH = LIDAR3D_SITE_BASE_PITCH

            points = []
            min_dist = self.config.get("LIDAR_MIN_DIST", self.lidar_range_min)
            max_dist = self.config.get("LIDAR_MAX_DIST", self.lidar_range_max)

            for sensor in self.lidar_sensors:
                name = sensor[1]
                addr = sensor[2]

                yaw_idx, pitch_idx = extract_3d_lidar_id(name)

                dist = self.mj_data.sensordata[addr]

                # Skip invalid (-1 = no hit) or out-of-range readings
                if dist < 0 or dist < min_dist or dist > max_dist:
                    continue

                yaw   = yaw_idx   * YAW_STEP
                alpha = SITE_BASE_PITCH + pitch_idx * PITCH_STEP

                sin_a = math.sin(alpha)
                cos_a = math.cos(alpha)

                # Hit point in lidar_link frame
                x = dist * math.cos(yaw) * sin_a
                y = dist * math.sin(yaw) * sin_a
                z = dist * cos_a

                points.append(np.array([x, y, z]))

            # Truncate to max points limit (default = full grid size)
            default_max = LIDAR3D_N_YAW * LIDAR3D_N_PITCH
            max_points = self.config.get("LIDAR_MAX_POINTS", default_max)
            if len(points) > max_points:
                points = points[:max_points]

            return {
                "points": points,
                "frame_id": "lidar_link",
            }

        elif self.lidar_type == "debug":
            # Debug lidar:

            YAW_STEP = 0.7854
            PITCH_STEP = 0.7854

            rot = R.from_euler('y', 1.57, degrees=False)

            points_up = []
            points_p = []

            for sensor in self.lidar_sensors:
                name = sensor[1]  # sensor name e.g., "lidar0500"
                addr = sensor[2]  # sensordata address

                # Extract yaw (xx) and pitch (yy) indices from sensor name
                yy, xx = int(name[-1]), int(name[-2])

                # Get distance reading
                dist = self.mj_data.sensordata[addr]

                # Skip invalid readings
                if dist < 0 or dist < self.lidar_range_min or dist > self.lidar_range_max:
                    continue

                # Calculate angles (matching MuJoCo's rotation composition order)
                # MuJoCo applies: R_outer(pitch) * R_inner(yaw) * R_base(90° pitch)
                # This gives direction: [cos(pitch)*cos(yaw), sin(yaw), -sin(pitch)*cos(yaw)]
                yaw = xx * YAW_STEP
                pitch = yy * PITCH_STEP

                print(f"Sensor {name}: dist={dist:.2f}, yaw={math.degrees(yaw):.1f}°, pitch={math.degrees(pitch):.1f}°")

                # Convert to Cartesian using MuJoCo's composed rotation formula
                x = dist * math.cos(pitch) * math.cos(yaw)
                y = dist * math.sin(yaw)
                z = -dist * math.sin(pitch) * math.cos(yaw)
                p = np.array([x, y, z])

                # rotate into the new axes system
                # p = rot.inv().apply(p)
                # p = rot.apply(p)
                points_up.append(p)
                
                # points.append(np.array([x, y, z]))
                
                # Read hit point in global frame (already computed by MuJoCo)
                try:
                    lidar_link_id = self.mj_model.body("lidar_link").id
                    lidar_pos = self.mj_data.xpos[lidar_link_id].copy()
                    lidar_quat = self.mj_data.xquat[lidar_link_id].copy()  # [w,x,y,z]
                except KeyError:
                    lidar_pos = np.zeros(3)
                    lidar_quat = np.array([1.0, 0.0, 0.0, 0.0])
            
                point_global = np.array([
                    self.mj_data.sensordata[addr + 1],
                    self.mj_data.sensordata[addr + 2],
                    self.mj_data.sensordata[addr + 3]
                ])

                # Transform to lidar_link frame
                point_local = quat_rotate_vector(
                    quat_conjugate(lidar_quat),
                    point_global - lidar_pos
                )
                points_p.append(point_local)

            return {
                "debug_up": points_up,
                "debug_p": points_p,
                "frame_id": "lidar_link",
            }          


    # ── Physics step ────────────────────────────────────────────────────

    def sim_step(self):
        """One full simulation step: observe → publish → torque → mj_step → fall check."""
        obs = self.prepare_obs()

        # Publish state to DDS
        if self.bridge is not None:
            self.bridge.PublishLowState(obs)
            if self.hand_type == "inspire":
                left_n, right_n = self.inspire_hand_state_normalized()
                self.bridge.PublishInspireHandState(right_n, left_n)
            elif self.hand_type == "brainco":
                left_n, right_n = self.brainco_hand_state_normalized()
                self.bridge.PublishBraincoHandState(left_n, right_n)
            if self.bridge.joystick is not None:
                self.bridge.PublishWirelessController()

        # Elastic band (skipped in fixed-stand mode, which pins the base rigidly)
        if self.elastic_band is not None and not self.fix_stand:
            if self.elastic_band.enable and self.use_floating_root:
                pose = np.concatenate([
                    self.mj_data.xpos[self.band_attached_link],
                    self.mj_data.xquat[self.band_attached_link],
                    np.zeros(6),
                ])
                mujoco.mj_objectVelocity(
                    self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link, pose[7:13], 0,
                )
                # swap [ang, lin] → [lin, ang]
                pose[7:10], pose[10:13] = pose[10:13].copy(), pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)

        # Compute body torques (WBC PD via rt/lowcmd).
        body_torques = self.compute_body_torques()

        # Reset and scatter torques into MuJoCo actuator-indexed array.
        # self.torques is indexed by MuJoCo actuator ID (size = mj_model.nu).
        # Each *_actuator_index array gives the actuator ID for the i-th WBC
        # motor, so this remaps cleanly from WBC order to actuator order.
        self.torques[:] = 0.0
        if len(self.body_actuator_index) > 0:
            self.torques[self.body_actuator_index] = body_torques[: len(self.body_actuator_index)]

        # Dex3 hands are torque-driven via PD; the Inspire and Revo2/BrainCo
        # hands are position servos (handled below), so only scatter hand
        # torques for dex3.
        if self.hand_type == "dex3" and self.num_hand_dof > 0:
            hand_torques = self.compute_hand_torques()
            if len(self.left_hand_actuator_index) > 0:
                n_l = len(self.left_hand_actuator_index)
                self.torques[self.left_hand_actuator_index] = hand_torques[:n_l]
            if len(self.right_hand_actuator_index) > 0:
                n_r = len(self.right_hand_actuator_index)
                self.torques[self.right_hand_actuator_index] = (
                    hand_torques[self.num_hand_dof : self.num_hand_dof + n_r]
                )

        # Clip against per-actuator torque limits (also indexed by actuator ID).
        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config.get("FREE_BASE", False):
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques

        # Inspire finger position servos: write the de-normalized angle setpoint
        # straight to ctrl (these actuators interpret ctrl as a target position,
        # not a torque). The DFX command carries no gains, so the servo kp/kv
        # in the XML provides the tracking.
        if self.hand_type in ("inspire", "brainco"):
            if self.hand_type == "inspire":
                targets = self.compute_inspire_hand_targets()
            else:
                targets = self.compute_brainco_hand_targets()
            if targets is not None:
                left_t, right_t = targets
                if len(self.left_hand_actuator_index) > 0:
                    k = len(left_t)
                    self.mj_data.ctrl[self.left_hand_actuator_index[:k]] = left_t
                if len(self.right_hand_actuator_index) > 0:
                    k = len(right_t)
                    self.mj_data.ctrl[self.right_hand_actuator_index[:k]] = right_t

        # Disable sensor computation to skip 2880 rangefinder raycasts (~18ms)
        # Sensors are re-enabled on-demand in the run() loop only on lidar publish steps
        self.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Rigidly hold base + lower body so the robot stays standing (arms free).
        if self.fix_stand:
            self._apply_fix_stand()

        self._check_fall()

    def _apply_fix_stand(self):
        """Pin the floating base + lower body (legs + waist) to the standing pose.

        Called after every mj_step so the robot stays upright with no balance
        controller. The base is frozen in the world and the leg/waist joints are
        snapped back to their nominal angles with zero velocity; the arms and
        hands are left untouched, so they still follow incoming commands.
        """
        if self.use_floating_root:
            self.mj_data.qpos[:7] = self.fix_base_qpos
            self.mj_data.qvel[:6] = 0.0
        self.mj_data.qpos[self.fix_lower_qposadr] = self.fix_lower_qpos
        self.mj_data.qvel[self.fix_lower_dofadr] = 0.0

    # ── Fall detection & reset ──────────────────────────────────────────

    def _check_fall(self):
        if self.mj_data.qpos[2] < 0.2:
            print(f"Warning: Robot fell (height={self.mj_data.qpos[2]:.3f} m) — resetting")
            self.reset()

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.fix_stand:
            self._apply_fix_stand()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # ── Viewer ──────────────────────────────────────────────────────────

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()


# ═══════════════════════════════════════════════════════════════════════════
#  ModinoidSimulator — main simulation loop
# ═══════════════════════════════════════════════════════════════════════════

class ModinoidSimulator:
    """Rate-limited simulation loop with DDS bridge and viewer."""

    def __init__(self, config: dict):
        self.config = config
        self.sim_dt = config["SIMULATE_DT"]
        self.viewer_dt = config.get("VIEWER_DT", 0.02)

        # Environment
        self.env = ModinoidSimEnv(config)

        # DDS initialisation
        interface = config.get("INTERFACE")
        if interface:
            ChannelFactoryInitialize(config["DOMAIN_ID"], interface)
        else:
            ChannelFactoryInitialize(config["DOMAIN_ID"])

        # Bridge
        self.bridge = UnitreeSDK2Bridge(config)
        self.env.set_bridge(self.bridge)

        if config.get("USE_JOYSTICK"):
            self.bridge.SetupJoystick(
                device_id=config.get("JOYSTICK_DEVICE", 0),
                js_type=config.get("JOYSTICK_TYPE", "xbox"),
            )

        # TF publish rate
        self.tf_publish_rate = config.get("TF_PUBLISH_RATE", 50.0)  # Hz
        self.tf_publish_interval = int(1.0 / (self.tf_publish_rate * self.sim_dt))

        # Clock publish rate (for ROS2 use_sim_time support)
        self.clock_publish_rate = config.get("CLOCK_PUBLISH_RATE", 100.0)  # Hz
        self.clock_publish_interval = max(1, int(1.0 / (self.clock_publish_rate * self.sim_dt)))

        # Lidar publish rate
        self.lidar_publish_rate = config.get("LIDAR_PUBLISH_RATE", 10.0)  # Hz
        self.lidar_publish_interval = max(1, int(1.0 / (self.lidar_publish_rate * self.sim_dt)))

        # Camera image server (ZMQ, same protocol as real robot)
        if config.get("CAMERA_ENABLED", False):
            from sim_camera_server import SimCameraServer
            self.camera_server = SimCameraServer(self.env.mj_model, config)
            camera_fps = config.get("CAMERA_FPS", 10)
            self.camera_render_interval = max(1, int(1.0 / (camera_fps * self.sim_dt)))
        else:
            self.camera_server = None

    def run(self):
        """Main loop — runs until the viewer is closed or Ctrl-C."""
        sim_cnt = 0
        viewer_step = max(1, int(self.viewer_dt / self.sim_dt))

        try:
            while (self.env.viewer is None) or self.env.viewer.is_running():
                step_start = time.monotonic()

                self.env.sim_step()

                # Get current simulation time
                sim_time = self.env.mj_data.time

                # Publish clock at configured rate
                if sim_cnt % self.clock_publish_interval == 0:
                    self.bridge.PublishClock(sim_time)

                # Publish TF tree at configured rate
                if sim_cnt % self.tf_publish_interval == 0:
                    transforms = self.env.prepare_tf_transforms()
                    self.bridge.PublishTF(transforms, sim_time)

                # Publish lidar at configured rate — recompute sensors on demand only
                if self.env.have_lidar and sim_cnt % self.lidar_publish_interval == 0:
                    # Cast enum to int before bitwise operations
                    sensor_flag = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)

                    self.env.mj_model.opt.disableflags &= ~sensor_flag
                    mujoco.mj_sensorPos(self.env.mj_model, self.env.mj_data)
                    self.env.mj_model.opt.disableflags |= sensor_flag

                    lidar_msg = self.env.build_lidar_message()
                    self.bridge.PublishLidar(lidar_msg, sim_time)

                # Render cameras for ZMQ image server
                if self.camera_server is not None and sim_cnt % self.camera_render_interval == 0:
                    self.camera_server.render_cameras(self.env.mj_data)

                if sim_cnt % viewer_step == 0:
                    self.env.update_viewer()

                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("\nSimulation interrupted.")
        finally:
            self.close()

    def close(self):
        if self.env.viewer is not None:
            try:
                self.env.viewer.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()
    sim = ModinoidSimulator(cfg)
    print(f"Starting simulation: scene={cfg['ROBOT_SCENE']}, "
          f"interface={cfg.get('INTERFACE', 'lo')}, dt={cfg['SIMULATE_DT']}s")
    sim.run()