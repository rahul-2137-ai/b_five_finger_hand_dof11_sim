"""Cubemap -> equirectangular warp for the wide-FOV (VR180) stereo cameras.

MuJoCo cameras are pinhole, so a true ~180 deg fisheye cannot be rendered by a
single camera (it degenerates / crashes past ~100 deg fovy). Instead each eye has
5 cube-face cameras (front/left/right/up/down, 90 deg each) defined in the model.
We render those faces and remap them into an equirectangular image per eye, which
is the format a VR headset consumes for 180 deg stereo.

The face orientations below MUST match the xyaxes of the stL_*/stR_* cameras in
r_robot.xml. Each face is (F=view dir, R=image right, U=image up) in the eye frame
(forward = +X, left = +Y, up = +Z).
"""
import numpy as np

# face name -> (F view dir, R right axis, U up axis) in the eye/torso frame.
FACES = {
    "front": (( 1, 0, 0), ( 0,-1, 0), (0, 0, 1)),
    "left":  (( 0, 1, 0), ( 1, 0, 0), (0, 0, 1)),
    "right": (( 0,-1, 0), (-1, 0, 0), (0, 0, 1)),
    "up":    (( 0, 0, 1), ( 0, 1, 0), (1, 0, 0)),
    "down":  (( 0, 0,-1), ( 0, 1, 0), (-1,0, 0)),
}
FACE_ORDER = ["front", "left", "right", "up", "down"]


def build_equirect_lut(out_h, out_w, face_h, face_w,
                       hfov_deg=180.0, vfov_deg=180.0, pitch_deg=0.0):
    """Precompute the remap from a 5-face cubemap to an equirectangular image.

    Returns (face_idx, src_y, src_x), each int array of shape (out_h, out_w),
    so the equirect output is  faces_stack[face_idx, src_y, src_x].

    pitch_deg > 0 tilts the whole view DOWN, so the equirect center ("straight
    ahead" in the headset) looks at the workspace/table instead of the horizon
    — i.e. a natural human-at-a-bench eye line. The cube faces are unchanged;
    only the sampling ray directions are rotated.
    """
    lon = (np.linspace(0, 1, out_w) - 0.5) * np.deg2rad(hfov_deg)   # +lon -> +Y (left)
    lat = (0.5 - np.linspace(0, 1, out_h)) * np.deg2rad(vfov_deg)   # +lat -> +Z (up)
    lon, lat = np.meshgrid(lon, lat)                                # (H, W)
    cl = np.cos(lat)
    # ray direction in eye frame: forward=+X, left=+Y, up=+Z
    D = np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)  # (H,W,3)

    # Pitch the view down by rotating ray dirs about the eye's +Y (left) axis:
    # +pitch tilts forward (+X) toward down (-Z). center ray (1,0,0) -> (cos,0,-sin).
    if pitch_deg:
        p = np.deg2rad(pitch_deg)
        cp, sp = np.cos(p), np.sin(p)
        x, y, z = D[..., 0], D[..., 1], D[..., 2]
        D = np.stack([x * cp + z * sp, y, -x * sp + z * cp], axis=-1)

    best_s = np.full((out_h, out_w), -1e9)
    face_idx = np.zeros((out_h, out_w), dtype=np.int64)
    sx = np.zeros((out_h, out_w), dtype=np.int64)
    sy = np.zeros((out_h, out_w), dtype=np.int64)

    for fi, name in enumerate(FACE_ORDER):
        F, R, U = (np.array(v, dtype=float) for v in FACES[name])
        s = D @ F                                  # alignment with this face's view dir
        sel = s > best_s
        # only meaningful where s>0 (in front of the face); pick the most-aligned face
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.nan_to_num((D @ R) / s, nan=0.0, posinf=0.0, neginf=0.0)
            v = np.nan_to_num((D @ U) / s, nan=0.0, posinf=0.0, neginf=0.0)
        px = ((u + 1.0) * 0.5 * (face_w - 1)).astype(np.int64)
        py = ((1.0 - (v + 1.0) * 0.5) * (face_h - 1)).astype(np.int64)
        valid = sel & (s > 1e-6)
        face_idx = np.where(valid, fi, face_idx)
        sx = np.where(valid, np.clip(px, 0, face_w - 1), sx)
        sy = np.where(valid, np.clip(py, 0, face_h - 1), sy)
        best_s = np.where(valid, s, best_s)

    return face_idx, sy, sx


def warp_to_equirect(faces, lut):
    """faces: list of 5 BGR images (front,left,right,up,down), same HxW.
    lut: (face_idx, sy, sx) from build_equirect_lut. Returns equirect BGR image."""
    face_idx, sy, sx = lut
    stack = np.stack(faces, axis=0)               # (5, fh, fw, 3)
    return stack[face_idx, sy, sx]                # (out_h, out_w, 3)
