import mujoco
import mujoco.viewer
import time

# 1. Load your newly created MuJoCo XML file
# Change this to 'revo2_right_hand.xml' if you want to see the right hand
model = mujoco.MjModel.from_xml_path('revo2_left_hand.xml')
data = mujoco.MjData(model)

# 2. Launch the interactive GUI viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("MuJoCo Simulation started. Close the window to exit.")
    
    # 3. Run the simulation loop
    while viewer.is_running():
        step_start = time.time()

        # Advance the physics simulation
        mujoco.mj_step(model, data)

        # Refresh the viewer graphics
        viewer.sync()

        # Maintain real-time simulation speed
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
