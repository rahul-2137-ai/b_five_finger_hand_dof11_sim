"""ZMQ camera image server for MuJoCo simulation.

Renders camera images from the MuJoCo scene and publishes them over ZeroMQ
as a msgpack-encoded dict:

    {
        "images": {
            "<camera_name>": "<base64-encoded JPEG bytes>",
            ...
        },
        "timestamp": <float seconds>,
    }

This matches the wire format expected by `three_img_clinet.py`, which decodes
each camera independently and displays them side-by-side with labels.
"""

import base64
import socket as _socket
import time

import cv2
import msgpack
import mujoco
import numpy as np
import zmq


def _port_is_busy(port: int) -> bool:
    """Return True if another process is already bound to TCP `port`."""
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        s.close()
        return False
    except OSError:
        return True


class SimCameraServer:
    """Renders MuJoCo cameras and streams per-camera JPEG frames over ZMQ PUB."""

    def __init__(self, mj_model, config: dict):
        self._mj_model = mj_model

        # Config
        self._port = config.get("CAMERA_ZMQ_PORT", 5555)
        self._jpeg_quality = config.get("CAMERA_JPEG_QUALITY", 80)
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]

        self._head_name = config.get("CAMERA_HEAD_NAME", "d435i_rgb")
        self._head_resolution = config.get("CAMERA_HEAD_RESOLUTION", [480, 640])  # [H, W]

        self._wrist_names = config.get("CAMERA_WRIST_NAMES", ["left_hand_cam", "right_hand_cam"])
        self._wrist_resolution = config.get("CAMERA_WRIST_RESOLUTION", [480, 640])  # [H, W]

        # CRITICAL: refuse to start if another server already owns the port.
        # An old raw-JPEG server still bound to 5555 is what causes the
        # client's "msgpack.ExtraData" error.
        if _port_is_busy(self._port):
            raise RuntimeError(
                f"[SimCameraServer] Port {self._port} is already in use. "
                f"An old server is still running. Kill it first:\n"
                f"    sudo fuser -k {self._port}/tcp\n"
                f"or  sudo lsof -i :{self._port} -t | xargs -r kill -9"
            )

        # Validate cameras exist in model
        self._available_head = self._camera_exists(self._head_name)
        self._available_wrists = [n for n in self._wrist_names if self._camera_exists(n)]

        if not self._available_head:
            print(f"[SimCameraServer] WARNING: head camera '{self._head_name}' not found in model")
        for name in self._wrist_names:
            if name not in self._available_wrists:
                print(f"[SimCameraServer] WARNING: wrist camera '{name}' not found in model")

        if not self._available_head and not self._available_wrists:
            print("[SimCameraServer] WARNING: no cameras available, server will not publish")

        # Renderers — created lazily on first render call (OpenGL context safety)
        self._renderers = None

        # ZMQ publisher
        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{self._port}")

        # Self-test: pack+unpack a dummy frame and confirm the roundtrip works.
        # If this fails, msgpack itself is broken — fail loudly NOW.
        self._self_test()

        print(f"[SimCameraServer] ZMQ PUB on tcp://*:{self._port} (msgpack dict format)")
        if self._available_head:
            print(f"  head: {self._head_name} {self._head_resolution[1]}x{self._head_resolution[0]}")
        for name in self._available_wrists:
            print(f"  wrist: {name} {self._wrist_resolution[1]}x{self._wrist_resolution[0]}")

    def _self_test(self):
        """Pack and unpack a sample payload to prove the wire format is valid."""
        dummy = np.zeros((8, 8, 3), dtype=np.uint8)
        b64 = self._encode_jpeg_b64(dummy)
        payload = {"images": {"_test_": b64}, "timestamp": time.time()}
        packed = msgpack.packb(payload, use_bin_type=True)
        decoded = msgpack.unpackb(packed, raw=False)
        assert "images" in decoded and "_test_" in decoded["images"], "self-test failed"
        print(f"[SimCameraServer] self-test OK ({len(packed)} bytes, magic={packed[:2].hex()})")

    def _camera_exists(self, name: str) -> bool:
        try:
            cam_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            return cam_id >= 0
        except Exception:
            return False

    def _init_renderers(self):
        """Create MuJoCo Renderer instances. Must be called from main thread."""
        self._renderers = {}
        if self._available_head:
            h, w = self._head_resolution
            self._renderers[self._head_name] = mujoco.Renderer(self._mj_model, h, w)
        for name in self._available_wrists:
            h, w = self._wrist_resolution
            self._renderers[name] = mujoco.Renderer(self._mj_model, h, w)

        self._scene_option = mujoco.MjvOption()
        self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    def _render_one(self, mj_data, camera_name: str) -> np.ndarray:
        """Render a single camera and return BGR image.

        MuJoCo's Renderer outputs RGB; the wire format (cv2.imencode/imdecode)
        is BGR, so convert here to keep colors correct on the client.
        """
        renderer = self._renderers[camera_name]
        renderer.update_scene(mj_data, camera=camera_name, scene_option=self._scene_option)
        rgb = renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _encode_jpeg_b64(self, bgr_img: np.ndarray) -> str:
        """JPEG-encode a BGR image and return a base64 ASCII string."""
        ret, buffer = cv2.imencode(".jpg", bgr_img, self._encode_params)
        if not ret:
            return ""
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    def render_cameras(self, mj_data):
        """Render all cameras, JPEG+base64 encode them, and publish over ZMQ.

        Wire format (msgpack):
            {
                "images": {cam_name: base64_jpeg_str, ...},
                "timestamp": float,
            }
        """
        if self._renderers is None:
            self._init_renderers()

        if not self._renderers:
            return

        images = {}

        if self._available_head:
            bgr = self._render_one(mj_data, self._head_name)
            b64 = self._encode_jpeg_b64(bgr)
            if b64:
                images[self._head_name] = b64

        for name in self._available_wrists:
            bgr = self._render_one(mj_data, name)
            b64 = self._encode_jpeg_b64(bgr)
            if b64:
                images[name] = b64

        if not images:
            return

        payload = {
            "images": images,
            "timestamp": time.time(),
        }

        try:
            packed = msgpack.packb(payload, use_bin_type=True)
            # Single-part send. Each recv() on the client gets exactly this many bytes.
            self._socket.send(packed, flags=zmq.NOBLOCK, copy=True)
        except zmq.Again:
            pass

    def close(self):
        """Shut down ZMQ resources and renderers."""
        self._socket.close(linger=0)
        self._zmq_context.term()
        if self._renderers:
            for renderer in self._renderers.values():
                renderer.close()
        print("[SimCameraServer] closed")