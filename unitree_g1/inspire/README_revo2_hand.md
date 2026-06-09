# BrainCo Revo2 Hand — Integration on the Unitree G1 (MuJoCo)

This documents the BrainCo **Revo2** 5-finger hands that replace the Inspire DFX
hands on the G1 in this simulation: the joint limits, the position-servo gains
(kp / kd), the under-actuation coupling, and the DDS command/state contract.

---

## 1. Files

| File | Role |
|------|------|
| `build_revo2.py` | Generates `g1_revo2.xml` (G1 body + Revo2 hands on wrists). **Re-run after editing the mount transform.** |
| `g1_revo2.xml` | Generated robot model (G1 + Revo2 hands). |
| `revo2_house_sim.xml` | Scene: 3-room house + jug/glass/water pour task (the active scene). |
| `revo2_sim.xml` | Scene: plain floor (lightweight bring-up variant). |
| `../../urdf/revo2_{left,right}_hand.xml` | Source standalone Revo2 hand models. |
| `../modinoid_sim/unitree_sdk2py_bridge.py` | DDS bridge (`brainco` hand type). |
| `../modinoid_sim/modinoid_sim.py` | Sim loop: maps commands → finger servos. |

---

## 2. Hand anatomy

Each hand has **5 fingers** and **11 joints**, driven by **6 motors**
(the distal joints are tendon-coupled, not independently driven).

### Top-down layout (looking into the palm)

```
                 finger tips
        index  middle  ring  pinky
          │      │      │      │
        ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐     distal  link  (coupled)
        │   │  │   │  │   │  │   │
        ├───┤  ├───┤  ├───┤  ├───┤  ← proximal joint (DRIVEN: motors 2,3,4,5)
        │   │  │   │  │   │  │   │
   ┌────┴───┴──┴───┴──┴───┴──┴───┴────┐
   │                                  │
   │   thumb                  PALM    │
   │   ┌───┐                          │
   │   │   │ ← thumb distal (coupled) │
   │   ├───┤ ← thumb proximal (motor 1 "thumb-aux", flexion)
   │   │   │
   │   ╲   ╱  ← thumb metacarpal (motor 0 "thumb", opposition/rotation)
   └────┴─┴───────────────────────────┘
              wrist  ↑
```

### Side view of one finger (flexion chain)

```
   open ───────────────────────────►  closed (curled)

   proximal joint           distal joint
        ●───────[ proximal ]───●────[ distal ]──►  fingertip
        │                      │
     DRIVEN by motor        COUPLED: distal = ratio × proximal
     (position servo)       (MuJoCo equality constraint)
```

### Motor table (BrainCo official order — matches `rt/brainco/*/cmd` indices)

```
┌──────┬───────┬────────────┬────────┬────────┬────────┬────────┐
│  Id  │   0   │     1      │   2    │   3    │   4    │   5    │
├──────┼───────┼────────────┼────────┼────────┼────────┼────────┤
│Joint │ thumb │ thumb-aux  │ index  │ middle │  ring  │ pinky  │
│ →MJ  │metacar│ thumb prox │idx prox│mid prox│ring pr.│pky pr. │
└──────┴───────┴────────────┴────────┴────────┴────────┴────────┘
   "thumb"     = metacarpal joint (opposition / rotation)
   "thumb-aux" = thumb proximal joint (bend/flexion)
```

---

## 3. Joint limits (radians)

All joints run **0 rad = open** → **positive = closed**.

| Finger  | Joint           | Driven? | Range (rad) | Joint torque limit (N·m)¹ |
|---------|-----------------|---------|-------------|---------------------------|
| Thumb   | metacarpal      | ✅ motor 0 | `0.00 … 1.57` | ±0.5 |
| Thumb   | proximal        | ✅ motor 1 | `0.00 … 1.03` | ±1.1 |
| Thumb   | distal          | coupled (×1.0) | `0.00 … 1.03` | ±1.1 |
| Index   | proximal        | ✅ motor 2 | `0.00 … 1.41` | ±2.0 |
| Index   | distal          | coupled (×1.156) | `0.00 … 1.63` | ±2.0 |
| Middle  | proximal        | ✅ motor 3 | `0.00 … 1.41` | ±2.0 |
| Middle  | distal          | coupled (×1.156) | `0.00 … 1.63` | ±2.0 |
| Ring    | proximal        | ✅ motor 4 | `0.00 … 1.41` | ±2.0 |
| Ring    | distal          | coupled (×1.156) | `0.00 … 1.63` | ±2.0 |
| Pinky   | proximal        | ✅ motor 5 | `0.00 … 1.41` | ±2.0 |
| Pinky   | distal          | coupled (×1.156) | `0.00 … 1.63` | ±2.0 |

¹ `actuatorfrcrange` on the joint (from the URDF) — caps the force MuJoCo applies
through that joint regardless of the servo output.

**Joint defaults** (applied to every finger joint, for numerical stability):
`armature = 0.02`, `damping = 0.4`, `frictionloss = 0.005`.

---

## 3b. Link masses (per hand)

From the URDF inertials (identical left/right). One hand = **base + 11 links**.

| Link | Mass |
|------|-----:|
| hand base (`*_hand_mount`, incl. `*_base_link`) | **221.6 g** |
| thumb metacarpal | 10 g |
| thumb proximal | 50 g |
| thumb distal | 12 g |
| index proximal / distal | 9 g / 11 g |
| middle proximal / distal | 9 g / 11 g |
| ring proximal / distal | 9 g / 11 g |
| pinky proximal / distal | 9 g / 11 g |
| **finger links subtotal** | **152 g** |
| **whole hand (base + fingers)** | **≈ 373.6 g** |

(The whole G1 + 2 hands robot is **34.0 kg**; see the scene README for the
graspable-object masses.)

---

## 4. Actuator gains (kp / kd)

The 6 driver joints per hand are MuJoCo **position servos**
(`<position kp="..." kv="...">`). In a position actuator:

- **kp** = position gain (stiffness)
- **kd** = velocity gain `kv` (damping)

| Motor | Joint (left/right) | **kp** | **kd** (`kv`) | ctrlrange (rad) | forcerange (N·m) |
|-------|--------------------|-------:|--------------:|-----------------|------------------|
| 0 thumb     | `*_thumb_metacarpal_joint` | **20** | **0.5** | `0.00 … 1.57` | ±10 |
| 1 thumb-aux | `*_thumb_proximal_joint`   | **20** | **0.5** | `0.00 … 1.03` | ±10 |
| 2 index     | `*_index_proximal_joint`   | **18** | **0.5** | `0.00 … 1.41` | ±10 |
| 3 middle    | `*_middle_proximal_joint`  | **18** | **0.5** | `0.00 … 1.41` | ±10 |
| 4 ring      | `*_ring_proximal_joint`    | **18** | **0.5** | `0.00 … 1.41` | ±10 |
| 5 pinky     | `*_pinky_proximal_joint`   | **18** | **0.5** | `0.00 … 1.41` | ±10 |

> The thumb motors are a touch stiffer (kp = 20) because the thumb also drags its
> coupled distal joint. `forcerange` ±10 is the servo's own output clamp; the
> tighter per-joint `actuatorfrcrange` (§3) clamps it further.
> To re-tune, edit these `<position>` lines in `revo2_house_sim.xml` /
> `revo2_sim.xml` (gains live in the **scene** file, not in `g1_revo2.xml`).

---

## 5. Distal coupling (under-actuation)

Each distal joint follows its proximal via a MuJoCo `<equality><joint>` with
`polycoef="0 ratio 0 0 0"`  ⇒  `distal = ratio × proximal`.

| Coupling | ratio | why |
|----------|------:|-----|
| thumb distal ← thumb proximal | **1.0** | equal ranges (1.03 / 1.03) |
| {index,middle,ring,pinky} distal ← proximal | **1.156** | ranges 1.63 / 1.41 → distal closes fully when proximal does |

---

## 6. DDS command / state contract

Per-hand topics (BrainCo Revo2 protocol). The **simulation** subscribes to the
`cmd` topics and publishes the `state` topics; the **teleop / controller** does
the opposite.

| Topic | Type | Dir (sim) | Payload |
|-------|------|-----------|---------|
| `rt/brainco/left/cmd`   | `MotorCmds_`   | subscribe | 6 × `q` |
| `rt/brainco/right/cmd`  | `MotorCmds_`   | subscribe | 6 × `q` |
| `rt/brainco/left/state` | `MotorStates_` | publish   | 6 × `q` |
| `rt/brainco/right/state`| `MotorStates_` | publish   | 6 × `q` |

- **Units:** normalized `q ∈ [0, 1]` per motor — **`0.0` = fully open, `1.0` = fully closed.**
- **Order:** index `0..5 = [thumb, thumb-aux, index, middle, ring, pinky]`.
- **Mapping in sim:** `setpoint = lo + q·(hi − lo)` with `(lo, hi)` = the joint's
  ctrlrange in §4 (since `lo = 0`, this is just `q · hi`). State is the inverse:
  `q = (angle − lo)/(hi − lo)`.

Config: `HAND_TYPE: brainco`, `NUM_HAND_MOTORS: 6` in
`../modinoid_sim/config.yaml`.

---

## 7. Wrist mount (thumb-up)

The Revo2 hand base is mounted on `{left,right}_wrist_yaw_link` via the
`LEFT/RIGHT_MOUNT_POS` and `LEFT/RIGHT_MOUNT_QUAT` constants in `build_revo2.py`.

- Position: `0.0415 0 0` (down the wrist +X, distal direction).
- Orientation: `R_x(roll) · R_y(90°)` — the `R_y(90°)` points the fingers down
  the forearm; the roll (`−93.8°` left, `+93.8°` right) puts the **thumb up**.
- Quats currently set: left `0.483354 -0.516109 0.483354 -0.516109`,
  right `0.483364 0.516100 0.483364 0.516100`.

To change the palm orientation, edit those constants and re-run `build_revo2.py`.

---

## 8. Run / regenerate

```bash
# regenerate the robot model after editing build_revo2.py
cd unitree_g1/inspire
python3 build_revo2.py

# run the full sim (use the py3.10 / b_hand_sim env that has cyclonedds)
cd ../modinoid_sim
python3 modinoid_sim.py
```
