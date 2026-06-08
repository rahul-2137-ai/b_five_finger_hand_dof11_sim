"""Generate scene_inspire.xml: G1 + Inspire 5-finger hands with actuators and
the finger mimic-couplings (which MuJoCo drops on URDF import) re-added as
equality constraints."""
import mujoco

ROBOT = "g1_29dof_inspire.xml"
m = mujoco.MjModel.from_xml_path(ROBOT)

FINGER_KW = ("thumb", "index", "middle", "ring", "pinky")

# coupling: dependent_joint = multiplier * driver_joint  (from the URDF <mimic>)
COUPLINGS = []
for side in ("L", "R"):
    COUPLINGS += [
        (f"{side}_thumb_intermediate_joint", f"{side}_thumb_proximal_pitch_joint", 1.6),
        (f"{side}_thumb_distal_joint",       f"{side}_thumb_proximal_pitch_joint", 2.4),
        (f"{side}_index_intermediate_joint", f"{side}_index_proximal_joint",       1.0),
        (f"{side}_middle_intermediate_joint",f"{side}_middle_proximal_joint",      1.0),
        (f"{side}_ring_intermediate_joint",  f"{side}_ring_proximal_joint",        1.0),
        (f"{side}_pinky_intermediate_joint", f"{side}_pinky_proximal_joint",       1.0),
    ]
dependent = {d for d, _, _ in COUPLINGS}

# classify joints
hand_drivers, body_joints = [], []
for j in range(m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    if n is None:
        continue
    lo, hi = m.jnt_range[j]
    is_hand = any(k in n for k in FINGER_KW)
    if is_hand:
        if n not in dependent:          # only actuate the independent finger joints
            hand_drivers.append((n, lo, hi))
    else:
        body_joints.append((n, lo, hi))

acts = []
# body: stiff position servos so the robot holds its assembled pose
for n, lo, hi in body_joints:
    acts.append(f'    <position name="{n}" joint="{n}" kp="150" kv="8" '
                f'ctrlrange="{lo:.4f} {hi:.4f}" forcerange="-200 200"/>')
# hands: position servos; thumb needs more torque (it drives two coupled joints)
for n, lo, hi in hand_drivers:
    kp = 80 if "thumb" in n else 18
    acts.append(f'    <position name="{n}" joint="{n}" kp="{kp}" kv="1.0" '
                f'ctrlrange="{lo:.4f} {hi:.4f}" forcerange="-60 60"/>')

# stiff equality so the coupled joint tracks its driver tightly
eqs = [f'    <joint joint1="{d}" joint2="drv" polycoef="0 {k} 0 0 0" '
       f'solref="0.004 1" solimp="0.99 0.999 0.001"/>'.replace("drv", drv)
       for d, drv, k in COUPLINGS]

scene = f"""<mujoco model="g1_29dof_inspire scene">
  <include file="{ROBOT}"/>

  <!-- implicitfast handles the joint damping stably; plain Euler does not -->
  <option integrator="implicitfast"/>

  <statistic center="0 0 0.1" extent="0.8"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="1280"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
  </asset>

  <equality>
{chr(10).join(eqs)}
  </equality>

  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
"""
open("scene_inspire.xml", "w").write(scene)
print(f"wrote scene_inspire.xml: {len(body_joints)} body actuators, "
      f"{len(hand_drivers)} hand actuators, {len(eqs)} couplings")

# validate
ms = mujoco.MjModel.from_xml_path("scene_inspire.xml")
print(f"scene compiles: nu={ms.nu}  neq={ms.neq}")
