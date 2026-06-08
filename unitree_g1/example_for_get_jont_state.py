# subscribe_lowstate.py

import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

# --- Set domain ID here (match your publisher) ---
ChannelFactoryInitialize(0,"lo")  # 0 = domain ID enp5s0

# For G1 / H1-2 robots
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default

low_state = LowState_default()
received = False

def LowStateHandler(msg):
    global low_state, received
    low_state = msg
    received = True

# Subscribe to the same topic the bridge publishes to
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(LowStateHandler, 1)

print("Waiting for LowState messages...")

while True:
    if received:
        print("=" * 50)
        print(f"Tick: {low_state.tick}")
        print(f"IMU quaternion (w,x,y,z): {low_state.imu_state.quaternion[:]}")
        print(f"IMU gyroscope:             {low_state.imu_state.gyroscope[:]}")
        print(f"IMU accelerometer:         {low_state.imu_state.accelerometer[:]}")
        print("\n--- Joint States ---")
        for i in range(35):  # Change 10 to NUM_MOTORS
            m = low_state.motor_state[i]
            print(f"  Joint [{i:2d}] q={m.q:8.4f}  dq={m.dq:8.4f}  tau={m.tau_est:8.4f}")
        received = False
    time.sleep(0.1)
