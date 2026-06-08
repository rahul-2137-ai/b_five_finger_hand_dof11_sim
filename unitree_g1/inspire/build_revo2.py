"""Generate g1_revo2.xml: the Unitree G1 body with the BrainCo Revo2 5-finger
hands mounted on the wrists, replacing the Inspire DFX hands.

Source of truth for the G1 body is r_robot.xml (G1 + Inspire). This script
strips the Inspire hand subtrees/meshes off each wrist_yaw_link and grafts on
the Revo2 hand bodies parsed from ../../urdf/revo2_{left,right}_hand.xml.

The Revo2 hands are standalone models (base at their own origin), so the
wrist->hand-base transform is NOT defined anywhere. The LEFT_MOUNT_* /
RIGHT_MOUNT_* constants below are a deliberate ESTIMATE — open the scene in the
MuJoCo viewer and nudge pos/quat until the palm sits correctly on the wrist,
then re-run this script.

Run:  python3 build_revo2.py      (from this directory)
"""
import copy
import xml.etree.ElementTree as ET

ROBOT_SRC = "r_robot.xml"                       # G1 + Inspire (donor body)
LEFT_HAND_SRC = "../../urdf/revo2_left_hand.xml"
RIGHT_HAND_SRC = "../../urdf/revo2_right_hand.xml"
OUT = "g1_revo2.xml"

# ── Wrist -> Revo2 hand-base mount transform (relative to wrist_yaw_link) ────
# The Revo2 hand frame has its fingers extending along +Z; the wrist's distal
# direction is +X, so we first rotate the hand +Z onto wrist +X (+90 deg about
# wrist Y). On top of that we roll the hand about the finger axis by ~+/-94 deg
# so the THUMB points up (thumb-up pose) instead of sideways. The two quats
# below are R_x(roll) * R_y(90deg); left rolls -93.8 deg, right +93.8 deg.
# (Tune in the viewer if you want a different palm orientation.)
LEFT_MOUNT_POS = "0.0415 0 0"
LEFT_MOUNT_QUAT = "0.483354 -0.516109 0.483354 -0.516109"
RIGHT_MOUNT_POS = "0.0415 0 0"
RIGHT_MOUNT_QUAT = "0.483364 0.516100 0.483364 0.516100"

# Floating-base spawn pose. Kept identical to the donor r_robot.xml so the robot
# stands inside the house in front of the jug+glass table (table_r1), facing the
# room. This is a pure yaw about Z, so the thumb-up wrist mount is preserved.
# (On the plain floor scene the robot simply stands pinned at this location.)
PELVIS_POS = "4.75 2.2 0.9"
PELVIS_QUAT = "0.747602 0 0 0.664147"

# Inspire meshes to drop from <asset> (no longer referenced once hands swapped).
INSPIRE_MESHES = {"L_hand_base_link", "R_hand_base_link"}
for side in ("L", "R"):
    for i in range(11, 23):
        INSPIRE_MESHES.add(f"Link{i}_{side}")

# Inspire finger root bodies hanging off each wrist_yaw_link.
LEFT_INSPIRE_FINGERS = {"L_thumb_proximal_base", "L_index_proximal",
                        "L_middle_proximal", "L_ring_proximal", "L_pinky_proximal"}
RIGHT_INSPIRE_FINGERS = {"R_thumb_proximal_base", "R_index_proximal",
                         "R_middle_proximal", "R_ring_proximal", "R_pinky_proximal"}


def find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None


def hand_worldbody_children(path):
    """Return a deep copy of the <worldbody> children of a standalone hand XML
    (the base geom + the five finger root bodies) and its <mesh> asset elems."""
    tree = ET.parse(path)
    root = tree.getroot()
    wb = root.find("worldbody")
    children = [copy.deepcopy(c) for c in list(wb)]
    meshes = [copy.deepcopy(m) for m in root.find("asset").findall("mesh")]
    return children, meshes


def graft_hand(wrist_body, inspire_fingers, hand_children, mount_name, pos, quat):
    """Strip Inspire hand geoms/finger bodies off wrist_body and append a mount
    body holding the Revo2 hand."""
    for child in list(wrist_body):
        if child.tag == "geom" and child.get("mesh") in INSPIRE_MESHES:
            wrist_body.remove(child)
        elif child.tag == "body" and child.get("name") in inspire_fingers:
            wrist_body.remove(child)

    mount = ET.SubElement(wrist_body, "body",
                          {"name": mount_name, "pos": pos, "quat": quat})
    for c in hand_children:
        mount.append(c)


def main():
    tree = ET.parse(ROBOT_SRC)
    root = tree.getroot()
    root.set("model", "g1_29dof_rev_1_0_with_revo2_hand")

    # ── Spawn pose ──
    pelvis = find_body(root, "pelvis")
    pelvis.set("pos", PELVIS_POS)
    pelvis.set("quat", PELVIS_QUAT)

    # ── Asset swap: drop Inspire finger meshes, add Revo2 hand meshes ──
    asset = root.find("asset")
    for m in asset.findall("mesh"):
        if m.get("name") in INSPIRE_MESHES:
            asset.remove(m)

    left_children, left_meshes = hand_worldbody_children(LEFT_HAND_SRC)
    right_children, right_meshes = hand_worldbody_children(RIGHT_HAND_SRC)
    for m in left_meshes + right_meshes:
        asset.append(m)

    # ── Graft hands onto the wrists ──
    graft_hand(find_body(root, "left_wrist_yaw_link"), LEFT_INSPIRE_FINGERS,
               left_children, "left_hand_mount", LEFT_MOUNT_POS, LEFT_MOUNT_QUAT)
    graft_hand(find_body(root, "right_wrist_yaw_link"), RIGHT_INSPIRE_FINGERS,
               right_children, "right_hand_mount", RIGHT_MOUNT_POS, RIGHT_MOUNT_QUAT)

    ET.indent(tree, space="  ")
    header = ("<!-- GENERATED by build_revo2.py - do not edit by hand.\n"
              "     G1 body from r_robot.xml + BrainCo Revo2 hands on the wrists.\n"
              "     Wrist mount transform is an estimate; tune LEFT/RIGHT_MOUNT_* there. -->\n")
    xml_bytes = ET.tostring(root, encoding="unicode")
    with open(OUT, "w") as f:
        f.write(header)
        f.write(xml_bytes)
        f.write("\n")
    print(f"wrote {OUT}")

    # validate
    import mujoco
    m = mujoco.MjModel.from_xml_path(OUT)
    print(f"compiles: nbody={m.nbody} njnt={m.njnt} nmesh={m.nmesh}")


if __name__ == "__main__":
    main()
