"""
Simple ZMQ image client — directly shows 3 camera feeds from the sim server.

Run:
    python img_client.py              # port 5555 default
    python img_client.py --port 5555
    python img_client.py --host 192.168.1.10 --port 5555

Press 'q' to quit.
"""

import argparse
import base64
import time

import cv2
import msgpack
import numpy as np
import zmq


def decode(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5555, type=int)
    args = parser.parse_args()

    ctx    = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.setsockopt(zmq.RCVHWM, 2)                  # always latest frame
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(f"tcp://{args.host}:{args.port}")
    print(f"Connected to tcp://{args.host}:{args.port}  —  press q to quit")

    t0 = time.time()
    frames = 0

    while True:
        # ---------- receive ----------
        try:
            packed = socket.recv()
        except zmq.Again:
            print("Waiting for server...")
            continue

        data = msgpack.unpackb(packed, raw=False)

        # ---------- decode all cameras from data["images"] ----------
        images_raw = data.get("images", {})          # {cam_name: b64_str}
        cam_images = {}
        for cam_name, b64 in images_raw.items():
            if isinstance(b64, str):
                img = decode(b64)
                if img is not None:
                    cam_images[cam_name] = img

        if not cam_images:
            continue

        frames += 1
        fps = frames / (time.time() - t0)

        # ---------- build display: stack all cameras side by side ----------
        TARGET_H = 320                               # display height per camera
        strips = []
        for cam_name in sorted(cam_images.keys()):
            img = cam_images[cam_name]
            h, w = img.shape[:2]
            scale = TARGET_H / h
            thumb = cv2.resize(img, (int(w * scale), TARGET_H))

            # camera label
            cv2.putText(thumb, cam_name, (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2)

            # thin vertical divider
            div = np.zeros((TARGET_H, 3, 3), dtype=np.uint8)
            strips += [thumb, div]

        grid = np.hstack(strips[:-1])               # drop last divider

        # fps overlay
        cv2.putText(grid, f"FPS: {fps:.1f}", (10, grid.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Camera Feed (3 cameras)", grid)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    socket.close()
    ctx.term()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()