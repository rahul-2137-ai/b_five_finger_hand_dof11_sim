import mujoco
import mujoco.viewer

# Change to 'revo2_right_hand.xml' for the right hand
model = mujoco.MjModel.from_xml_path('revo2_left_hand.xml')
data = mujoco.MjData(model)

# Managed viewer: runs physics + rendering internally, blocks until you close the window
mujoco.viewer.launch(model, data)