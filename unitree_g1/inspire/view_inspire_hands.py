"""Unitree G1 with Inspire 5-finger hands in the MuJoCo viewer.

Run:  python3 view_inspire_hands.py
The viewer opens and both hands cycle open<->closed so you can see all five
fingers (thumb, index, middle, ring, pinky) responding to control. Body joints
are held at their assembled pose. Drag any actuator slider in the viewer's
Control panel to drive a finger manually.
"""
import time
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("scene_inspire.xml")
d = mujoco.MjData(m)

# hand actuators = the 12 finger drivers (names start with L_/R_)
hand = [a for a in range(m.nu)
        if (n := mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
        and (n.startswith("L_") or n.startswith("R_"))]
lo = m.actuator_ctrlrange[hand, 0]
hi = m.actuator_ctrlrange[hand, 1]
print(f"{len(hand)} finger actuators (6 per hand). Cycling open/close...")

with mujoco.viewer.launch_passive(m, d) as v:
    t0 = time.time()
    while v.is_running():
        phase = 0.5 * (1 - np.cos(2 * np.pi * (time.time() - t0) / 4.0))  # 0..1
        d.ctrl[hand] = lo + phase * (hi - lo)
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(max(0, m.opt.timestep))
