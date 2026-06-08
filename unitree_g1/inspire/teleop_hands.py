"""Perfect keyboard teleop for the Unitree G1 Inspire 5-finger hands (MuJoCo).

Run:  DISPLAY=:1 python3 teleop_hands.py

HOLD a key to curl that finger; RELEASE to open it again — exactly like
flexing a real hand. Each finger eases smoothly in/out (no snapping).

  RIGHT HAND                 LEFT HAND
    a : thumb swing            j : thumb swing
    s : thumb curl             k : thumb curl
    d : index                  l : index
    f : middle                 ; : middle
    z : ring                   m : ring
    x : pinky                  n : pinky

    SPACE : latch / unlatch a full fist on BOTH hands (hands-free close)
    r     : release everything (open)
    mouse : left-drag orbit, right-drag pan, scroll zoom
    ESC   : quit

This is a self-contained GLFW viewer (not launch_passive) so it can sense key
RELEASE, which the simple viewer cannot.
"""
import glfw
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("scene_inspire.xml")
d = mujoco.MjData(m)


def aid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


# fully-curled control target per DoF (open target is 0 for all)
CURL = {"thumb_proximal_yaw": 1.0, "thumb_proximal_pitch": 0.6,
        "index_proximal": 1.6, "middle_proximal": 1.6,
        "ring_proximal": 1.6, "pinky_proximal": 1.6}

# key char -> (actuator id, curl target).  cmd[key] in [0,1] is the curl amount.
def bind(side, ch_map):
    out = {}
    for ch, part in ch_map.items():
        out[ch] = (aid(f"{side}_{part}_joint"), CURL[part])
    return out

KEYS = {}
KEYS.update(bind("R", {"a": "thumb_proximal_yaw", "s": "thumb_proximal_pitch",
                       "d": "index_proximal", "f": "middle_proximal",
                       "z": "ring_proximal", "x": "pinky_proximal"}))
KEYS.update(bind("L", {"j": "thumb_proximal_yaw", "k": "thumb_proximal_pitch",
                       "l": "index_proximal", ";": "middle_proximal",
                       "m": "ring_proximal", "n": "pinky_proximal"}))

GLFW_KEY = {ch: getattr(glfw, "KEY_" + ch.upper()) if ch.isalpha()
            else glfw.KEY_SEMICOLON for ch in KEYS}

cmd = {ch: 0.0 for ch in KEYS}     # 0 = open, 1 = fully curled
RAMP = 0.30                         # seconds open<->closed
fist = False                       # space latch


def key_cb(window, key, scancode, action, mods):
    global fist
    if action != glfw.PRESS:
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_SPACE:
        fist = not fist
        print("FIST LATCH:", "CLOSED" if fist else "open")
    elif key == glfw.KEY_R:
        for ch in cmd:
            cmd[ch] = 0.0
        print("release all")


# ---- minimal GLFW + MuJoCo viewer with mouse camera ----
glfw.init()
window = glfw.create_window(1100, 900, "G1 Inspire hands - keyboard teleop", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
mujoco.mjv_defaultCamera(cam)
cam.azimuth, cam.elevation, cam.distance = 180, -12, 1.1
cam.lookat[:] = [0.0, 0.0, -0.05]
scene = mujoco.MjvScene(m, maxgeom=10000)
ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)

_mouse = {"lastx": 0.0, "lasty": 0.0, "L": False, "M": False, "R": False}


def mouse_button(win, button, act, mods):
    _mouse["L"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    _mouse["R"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    _mouse["M"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    _mouse["lastx"], _mouse["lasty"] = glfw.get_cursor_pos(win)


def mouse_move(win, xpos, ypos):
    dx, dy = xpos - _mouse["lastx"], ypos - _mouse["lasty"]
    _mouse["lastx"], _mouse["lasty"] = xpos, ypos
    if not (_mouse["L"] or _mouse["R"] or _mouse["M"]):
        return
    h = glfw.get_window_size(win)[1]
    if _mouse["L"]:
        act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
    elif _mouse["R"]:
        act = mujoco.mjtMouse.mjMOUSE_MOVE_V
    else:
        act = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(m, act, dx / h, dy / h, scene, cam)


def scroll(win, xoff, yoff):
    mujoco.mjv_moveCamera(m, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoff, scene, cam)


glfw.set_key_callback(window, key_cb)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_scroll_callback(window, scroll)

print(__doc__)

sim_dt = m.opt.timestep
last = glfw.get_time()
while not glfw.window_should_close(window):
    now = glfw.get_time()
    frame_dt = now - last
    last = now
    step = frame_dt / RAMP if RAMP > 0 else 1.0

    # update per-finger curl command from held keys (+ fist latch)
    for ch, (a, curl) in KEYS.items():
        held = glfw.get_key(window, GLFW_KEY[ch]) == glfw.PRESS
        target = 1.0 if (held or fist) else 0.0
        cmd[ch] += np.clip(target - cmd[ch], -step, step)
        d.ctrl[a] = cmd[ch] * curl

    # advance physics to keep up with wall clock
    n = max(1, int(round(frame_dt / sim_dt)))
    for _ in range(min(n, 40)):
        mujoco.mj_step(m, d)

    w, h = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, ctx)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
