# Revo2 House Scene — Object & Friction Changes

Changes made to **`revo2_house_sim.xml`**. The robot, hands, house, furniture and
the three "room" cubes are unchanged except where noted; only the Room-1 task
objects and the friction values were edited.

---

## 1. What changed (summary)

| # | Change | Before | After |
|---|--------|--------|-------|
| 1 | Jug → **big cube** | hollow jug (9 box geoms, handle + spout) | single box, 0.22 m, ~852 g — **pick with both hands** |
| 2 | Glass → **small cube** | cylinder + 8 walls | single box, 0.03 m, ~19 g — **pinch with two fingers** |
| 3 | Water | 12 sphere particles inside the jug | **removed** (no container to hold them) |
| 4 | Collision classes | `container` (3/3) + `water_particle` (2/0) | **removed** (no longer needed) |
| 5 | Object friction | torsional/rolling far too high (see §3) | realistic values (see §3) |
| 6 | `<option>`/`<default>` comments | referenced jug/glass/water | updated to the two-cube task |
| 7 | Spawn keyframe | ended with jug + glass poses | ends with big_cube + small_cube poses |

`nq` dropped from 177 → **93** (removing the 12 water free-joints = 84 qpos).

---

## 2. New objects (Room 1, on `table_r1`, top at z = 0.79)

| Object | Body / geom | Size (half-extent) | Full size | Density | Mass | Spawn pos | Grasp |
|--------|-------------|--------------------|-----------|---------|------|-----------|-------|
| **Big cube** | `big_cube` | `0.11 0.11 0.11` | 0.22 m cube | 80 kg/m³ | **852 g** | `4.77 2.74 0.90` | both hands |
| **Small cube** | `small_cube` | `0.015 0.015 0.015` | 0.03 m cube | 700 kg/m³ | **19 g** | `4.98 2.50 0.805` | thumb + index pinch |

Spawn positions are tuned for **easy grasping** from the robot's standing pose
(pelvis `4.75 2.2 0.9`, facing the table; resting wrists at x≈4.63 / 4.92):
- **big_cube** is centered in x between the two hands (faces at x≈4.66 / 4.88) and
  pushed to y=2.74 so it clears the relaxed hands at spawn (0 mm penetration) —
  both hands land on its left/right faces when the arms reach forward.
- **small_cube** sits just forward of and below the right hand (Δ≈ +0.06 x,
  +0.12 y, −0.19 z) for a clean one-hand pinch, with a 12 cm gap from the big cube.
- The viewer keyframe base pose was aligned to this same standing pose so the
  MuJoCo viewer and `modinoid_sim` show identical geometry.

Both use `solimp="0.99 0.995 0.001"`, `solref="0.004 1"` (firm rigid contact),
a free joint with `armature="0"` and `damping="0"` (required for correct
free-object physics — see §3b), and the realistic friction below.
The big cube density (80) is intentionally low so the G1 arms can lift it easily;
raise it if you want a heavier object.

---

## 3. Friction values — OLD vs NEW

MuJoCo friction is a 3-vector: **`[sliding, torsional, rolling]`**.
The old scene used very large torsional (0.3) and rolling (0.1) coefficients,
which make objects behave as if glued — they barely twist or roll, which is
**not** physically realistic for rigid household objects. Real rigid objects
have tiny torsional/rolling friction; MuJoCo's own defaults (`0.005`, `0.0001`)
are the realistic reference. Sliding stays high (`1.0`) so fingers get a firm
grip.

| Geom | OLD friction | NEW friction | Note |
|------|--------------|--------------|------|
| Jug (removed) | `1.0  0.3   0.1`   | — | replaced by big_cube |
| Glass (removed) | `1.0  0.3   0.1` | — | replaced by small_cube |
| Water (removed) | `0.2  0.05  0.05` | — | removed |
| **big_cube** | *(new)* | `1.0  0.005  0.0001` | realistic |
| **small_cube** | *(new)* | `1.0  0.005  0.0001` | realistic |
| `cube_r1 / r2 / r3` | `0.95 0.3   0.1` | `1.0  0.005  0.0001` | corrected |
| Default geoms (hands + body) | `1.0` ⇒ `[1.0, 0.005, 0.0001]` | `1.0 0.005 0.0001` | made explicit (effective value unchanged) |

### Why these numbers
- **Sliding = 1.0** → strong grip so the fingers don't slip off the cube
  (a contact's effective friction is the elementwise max of the two geoms, so
  hand `1.0` × cube `1.0` = `1.0`).
- **Torsional = 0.005** → the object can twist in the grip naturally instead of
  being rotationally locked (old `0.3` was ~60× too high).
- **Rolling = 0.0001** → the cube can roll/tip realistically on the table
  (old `0.1` was ~1000× too high and made cubes feel "sticky").

---

## 3b. Small-cube "slow motion" fix — root cause was **armature**, not friction

**Symptom:** the small cube moved/fell in slow motion (~half speed) and felt floaty.

**Diagnosis (measured):** with everything isolated in empty air it free-fell at
**−4.6 m/s²** instead of −9.81. Two compounding causes:

| Cause | Effect on small cube | Fix |
|-------|----------------------|-----|
| **`armature="0.02"`** inherited by the cube free-joints | adds 0.02 rotor-inertia to **every** DOF. On a 19 g cube that *doubles* the effective translational mass → falls at half g. (852 g big cube only +2%, so it looked fine.) | set **`armature="0"`** on all cube free joints |
| **`damping="0.0005"`** on the free-joints | vs the cube's tiny rotational inertia (1.6e‑6) this is a damp/inertia ratio of **~309 /s** → rotation glacially overdamped | set **`damping="0"`** |

The `armature="0.02"` comes from the joint `<default>` in `g1_revo2.xml`
(originally in `r_robot.xml`, there to stabilise the tiny finger joints). When the
scene `<include>`s that file, the default merges into the main class, so the
cube free-joints silently picked it up. Real free objects must have
`armature = 0` and `damping = 0` — only contact/gravity should act on them.

**After the fix (measured free-fall, 0.5 s in empty air):**

| Cube | armature | damping | free-fall drop (ideal 1.226 m) |
|------|---------:|--------:|--------------------------------|
| small_cube | 0 | 0 | 1.198 m ✅ |
| big_cube   | 0 | 0 | 1.230 m ✅ |
| cube_r1/2/3 | 0 | 0 | 1.213 m ✅ |

Settle test (2 s on the table): small cube rests at z = 0.805, big cube at
z = 0.900 — both exactly `table_top(0.79) + half_size`, no sinking, no NaN.

I also raised the small cube density **400 → 700 kg/m³** (≈ light wood, ~19 g)
so it is less skittish when pinched.

---

## 4. Hand friction (checked — left as-is)

The Revo2 hand geoms carry **no explicit friction**, so they inherit the scene
default `[1.0, 0.005, 0.0001]`. Verified on the fingertips:

```
left_index_distal_link  friction = [1.0, 0.005, 0.0001]
right_thumb_distal_link friction = [1.0, 0.005, 0.0001]
```

This is already realistic (firm 1.0 sliding grip, small spin/roll), so the hands
were **not changed**. If you want an even grippier "rubber fingertip" feel,
raise the hand sliding term toward ~1.2–1.5 — but note the contact takes the max
with the object, so raising only the cube's sliding has the same effect.

---

## 5. Unchanged

- Robot, Revo2 hands (thumb-up mount), 12 finger position servos and their
  kp/kd — see `README_revo2_hand.md`.
- House structure, furniture, lights, the three room cubes' geometry.
- `integrator="implicitfast"`, `cone="elliptic"`, DDS brainco hand protocol.

Validated: scene compiles (`nq=93, nu=41, neq=10`); big_cube 852 g, small_cube
19 g; jug/glass/water absent; all cubes free-fall at g and settle on the table.
