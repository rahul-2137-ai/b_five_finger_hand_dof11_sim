"""ZMQ stereo image client — views the VR180 stereo pair from the sim.

With STEREO_PROJECTION=equirect (default) each eye is an EQUIRECTANGULAR ~180 deg
image (warped from a cubemap by stereo_server.py) — the format a VR headset uses.
The server publishes a msgpack dict on its own port (default 5577):

    {
        "images": {"stereo_left": "<b64 jpeg>", "stereo_right": "<b64 jpeg>"},
        "timestamp": <float>,
    }

This client shows the two eyes side by side (LEFT | RIGHT). Press 'a' to toggle
a red/cyan anaglyph view (use red/cyan glasses), 'q' to quit.

Run:
    python stereo_client.py                  # localhost:5577
    python stereo_client.py --port 5577
    python stereo_client.py --host 192.168.1.10 --port 5577
"""

import argparse
import base64
import time

import cv2
import msgpack
import numpy as np
import zmq

LEFT = "stereo_left"
RIGHT = "stereo_right"


def decode(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR


def anaglyph(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """Red (left) / cyan (right) anaglyph for red-cyan 3D glasses."""
    h = min(left_bgr.shape[0], right_bgr.shape[0])
    w = min(left_bgr.shape[1], right_bgr.shape[1])
    l = cv2.resize(left_bgr, (w, h))
    r = cv2.resize(right_bgr, (w, h))
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :, 2] = l[:, :, 2]   # Red   from left eye
    out[:, :, 1] = r[:, :, 1]   # Green from right eye
    out[:, :, 0] = r[:, :, 0]   # Blue  from right eye
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5577, type=int)
    parser.add_argument("--height", default=480, type=int,
                        help="display height per eye (downscaled from the hi-res feed)")
    args = parser.parse_args()

    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.setsockopt(zmq.RCVHWM, 2)                 # keep only the latest frame
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(f"tcp://{args.host}:{args.port}")
    print(f"Connected to tcp://{args.host}:{args.port}  —  'a' anaglyph, 'q' quit")

    t0 = time.time()
    frames = 0
    use_anaglyph = False

    while True:
        try:
            packed = socket.recv()
        except zmq.Again:
            print("Waiting for stereo server...")
            continue

        data = msgpack.unpackb(packed, raw=False)
        images = data.get("images", {})
        left = decode(images[LEFT]) if isinstance(images.get(LEFT), str) else None
        right = decode(images[RIGHT]) if isinstance(images.get(RIGHT), str) else None
        if left is None or right is None:
            continue

        frames += 1
        fps = frames / (time.time() - t0)

        if use_anaglyph:
            view = anaglyph(left, right)
            scale = args.height / view.shape[0]
            view = cv2.resize(view, (int(view.shape[1] * scale), args.height))
            cv2.putText(view, "ANAGLYPH (red/cyan)", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
        else:
            strips = []
            for name, img in ((LEFT, left), (RIGHT, right)):
                scale = args.height / img.shape[0]
                thumb = cv2.resize(img, (int(img.shape[1] * scale), args.height))
                cv2.putText(thumb, "LEFT EYE" if name == LEFT else "RIGHT EYE",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2)
                div = np.zeros((args.height, 3, 3), dtype=np.uint8)
                strips += [thumb, div]
            view = np.hstack(strips[:-1])

        cv2.putText(view, f"FPS: {fps:.1f}", (10, view.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Stereo view (port %d)" % args.port, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            use_anaglyph = not use_anaglyph

    socket.close()
    ctx.term()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
