"""High-res EQUIRECTANGULAR (VR180) stereo image server.

MuJoCo cameras are pinhole and cannot render a true ~180 deg fisheye in one shot,
so each eye has 5 cube-face cameras (stL_*/stR_*, 90 deg each) baked into the
model. This server renders those faces and warps them to an equirectangular image
per eye (default 180x180 deg) — the format a VR headset uses for 180 deg stereo —
then publishes both eyes over ZMQ in the same msgpack wire format as
SimCameraServer:

    {"images": {"stereo_left": <b64 jpeg>, "stereo_right": <b64 jpeg>},
     "timestamp": <float>}

So stereo_client.py works unchanged (it just shows two equirect images).
"""
import base64
import socket as _socket
import time

import cv2
import msgpack
import mujoco
import numpy as np
import zmq

from stereo_equirect import build_equirect_lut, warp_to_equirect, FACE_ORDER


def _port_is_busy(port: int) -> bool:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port)); s.close(); return False
    except OSError:
        return True


class StereoEquirectServer:
    """Renders per-eye cubemaps and streams equirectangular frames over ZMQ PUB."""

    def __init__(self, mj_model, config: dict):
        self._mj_model = mj_model
        self._port = config.get("STEREO_ZMQ_PORT", 5577)
        self._jpeg_quality = config.get("STEREO_JPEG_QUALITY", 90)
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]

        # output equirect size [H, W] and per-face render size (smaller = faster)
        self._out_h, self._out_w = config.get("STEREO_EQUIRECT_SIZE", [640, 640])
        self._face = config.get("STEREO_FACE_SIZE", 256)
        self._hfov, self._vfov = config.get("STEREO_FOV", [180.0, 180.0])
        # tilt the VR180 view down N deg so "ahead" looks at the bench (human eye line)
        self._pitch = config.get("STEREO_PITCH_DEG", 0.0)

        # published eye name -> cube-face camera prefix in the model
        self._eyes = list(zip(
            config.get("STEREO_CAMERAS", ["stereo_left", "stereo_right"]),
            config.get("STEREO_FACE_PREFIXES", ["stL", "stR"]),
        ))

        # validate the 5 face cameras exist for each eye
        for out_name, prefix in self._eyes:
            for face in FACE_ORDER:
                cam = f"{prefix}_{face}"
                if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                    raise RuntimeError(
                        f"[StereoEquirectServer] camera '{cam}' not found in the model. "
                        f"Re-run build_revo2.py so the stL_*/stR_* cube cameras exist.")

        if _port_is_busy(self._port):
            raise RuntimeError(
                f"[StereoEquirectServer] Port {self._port} already in use. "
                f"Kill the old server:  sudo fuser -k {self._port}/tcp")

        # equirect remap LUT (built once)
        self._lut = build_equirect_lut(self._out_h, self._out_w,
                                       self._face, self._face,
                                       self._hfov, self._vfov, self._pitch)

        self._renderer = None   # created lazily on the main thread (GL context)

        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{self._port}")

        print(f"[StereoEquirectServer] ZMQ PUB on tcp://*:{self._port} "
              f"(equirect {self._out_w}x{self._out_h} @ {self._hfov:.0f}x{self._vfov:.0f} deg, "
              f"faces {self._face}px)")
        for out_name, prefix in self._eyes:
            print(f"  eye '{out_name}' <- {prefix}_[{'/'.join(FACE_ORDER)}]")

    def _init_renderer(self):
        self._renderer = mujoco.Renderer(self._mj_model, self._face, self._face)
        self._opt = mujoco.MjvOption()

    def _encode(self, bgr) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, self._encode_params)
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    def render_cameras(self, mj_data):
        """Render both eyes' cubemaps, warp to equirect, publish over ZMQ."""
        if self._renderer is None:
            self._init_renderer()

        images = {}
        for out_name, prefix in self._eyes:
            faces = []
            for face in FACE_ORDER:
                self._renderer.update_scene(mj_data, camera=f"{prefix}_{face}",
                                            scene_option=self._opt)
                rgb = self._renderer.render()
                faces.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            eq = warp_to_equirect(faces, self._lut)
            b64 = self._encode(eq)
            if b64:
                images[out_name] = b64

        if not images:
            return
        payload = {"images": images, "timestamp": time.time()}
        try:
            self._socket.send(msgpack.packb(payload, use_bin_type=True),
                              flags=zmq.NOBLOCK, copy=True)
        except zmq.Again:
            pass

    def close(self):
        self._socket.close(linger=0)
        self._zmq_context.term()
        if self._renderer is not None:
            self._renderer.close()
        print("[StereoEquirectServer] closed")
