"""Unified config loader for the gear_sim simulation runner.

Loads the WBC YAML config and merges with CLI overrides to produce
a single config dict used by both the simulation environment and the DDS bridge.
"""

import argparse
import os
from pathlib import Path

import yaml

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_YAML = str(_THIS_DIR / "config.yaml")
_DEFAULT_SCENE = str(_THIS_DIR.parent / "robot" / "g1" / "scene.xml")


def load_config(cli_args=None):
    """Load WBC YAML and merge with CLI overrides.

    Returns a plain dict with all simulation parameters.
    """
    parser = argparse.ArgumentParser(description="Gear-Sonic MuJoCo Simulation")
    parser.add_argument("--yaml", default=_DEFAULT_YAML, help="Path to WBC YAML config")
    parser.add_argument("--scene", default=None, help="Override ROBOT_SCENE path")
    parser.add_argument("--interface", default=None, help="DDS network interface (e.g. lo, eth0)")
    parser.add_argument("--domain-id", type=int, default=None, help="DDS domain ID")
    parser.add_argument("--dt", type=float, default=None, help="Simulation timestep (s)")
    parser.add_argument("--no-elastic-band", action="store_true", help="Disable elastic band")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless (no viewer)")
    parser.add_argument("--joystick", action="store_true", help="Enable joystick input")
    parser.add_argument("--joystick-type", default="xbox", help="Joystick type (xbox/switch)")
    parser.add_argument("--enable-camera", action="store_true", help="Enable camera image ZMQ server")
    parser.add_argument("--camera-port", type=int, default=None, help="ZMQ port for camera image server")
    args = parser.parse_args(cli_args)

    # Load YAML
    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)

    # Set the scene path
    if args.scene:
        cfg["ROBOT_SCENE"] = args.scene
    elif "ROBOT_SCENE" not in cfg or "gear_sonic" in cfg.get("ROBOT_SCENE", ""):
        # Default to the local copy; the YAML may point to rahul_sim's path
        cfg["ROBOT_SCENE"] = _DEFAULT_SCENE

    # CLI overrides
    if args.interface is not None:
        cfg["INTERFACE"] = args.interface
    if cfg.get("INTERFACE") is None:
        cfg["INTERFACE"] = "lo"

    if args.domain_id is not None:
        cfg["DOMAIN_ID"] = args.domain_id
    cfg.setdefault("DOMAIN_ID", 0)

    if args.dt is not None:
        cfg["SIMULATE_DT"] = args.dt
    cfg.setdefault("SIMULATE_DT", 0.005)

    cfg.setdefault("VIEWER_DT", 0.02)
    cfg.setdefault("REWARD_DT", 0.02)

    if args.no_elastic_band:
        cfg["ENABLE_ELASTIC_BAND"] = False
    cfg.setdefault("ENABLE_ELASTIC_BAND", True)

    cfg["ENABLE_ONSCREEN"] = not args.no_viewer
    cfg.setdefault("FREE_BASE", False)

    cfg["USE_JOYSTICK"] = 1 if args.joystick else cfg.get("USE_JOYSTICK", 0)
    cfg["JOYSTICK_TYPE"] = args.joystick_type
    cfg.setdefault("JOYSTICK_DEVICE", 0)

    cfg.setdefault("USE_SENSOR", False)
    cfg.setdefault("PRINT_SCENE_INFORMATION", True)

    # Ensure required motor config arrays exist
    for key in ("MOTOR_KP", "MOTOR_KD", "NUM_MOTORS", "NUM_JOINTS",
                "motor_effort_limit_list", "MOTOR2JOINT", "JOINT2MOTOR",
                "DEFAULT_DOF_ANGLES", "DEFAULT_MOTOR_ANGLES",
                "WeakMotorJointIndex", "UNITREE_LEGGED_CONST"):
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")

    cfg.setdefault("NUM_HAND_MOTORS", 7)
    cfg.setdefault("NUM_HAND_JOINTS", 7)

    # Camera image server
    if args.enable_camera:
        cfg["CAMERA_ENABLED"] = True
    cfg.setdefault("CAMERA_ENABLED", False)
    cfg.setdefault("CAMERA_FPS", 10)
    cfg.setdefault("CAMERA_ZMQ_PORT", 5555)
    cfg.setdefault("CAMERA_HEAD_NAME", "d435i_rgb")
    cfg.setdefault("CAMERA_HEAD_RESOLUTION", [480, 640])
    cfg.setdefault("CAMERA_WRIST_NAMES", ["left_hand_cam", "right_hand_cam"])
    cfg.setdefault("CAMERA_WRIST_RESOLUTION", [480, 640])
    cfg.setdefault("CAMERA_JPEG_QUALITY", 80)
    if args.camera_port is not None:
        cfg["CAMERA_ZMQ_PORT"] = args.camera_port

    return cfg
