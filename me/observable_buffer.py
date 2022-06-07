import robosuite as suite
import numpy as np
from robosuite.utils.buffers import RingBuffer

# Create env instance
control_freq = 10
env = suite.make("Lift", robots="Panda", has_offscreen_renderer=False, use_camera_obs=False, control_freq=control_freq)

# Define a ringbuffer to store joint position values
buffer = RingBuffer(dim=env.robots[0].robot_model.dof, length=10)

# Create a function that we'll use as the "filter" for the joint position Observable
# This is a pass-through operation, but we record the value every time it gets called
# As per the Observables API, this should take in an arbitrary numeric and return the same type / shape
def filter_fcn(corrupted_value):
    # Record the inputted value
    buffer.push(corrupted_value)
    # Return this value (no-op performed)
    return corrupted_value

# Now, let's enable the joint position Observable with this filter function
env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="filter",
    modifier=filter_fcn,
)

# Let's also increase the sampling rate to showcase the Observable's ability to update multiple times per env step
obs_sampling_freq = control_freq * 4
env.modify_observable(
    observable_name="robot0_joint_pos",
    attribute="sampling_rate",
    modifier=obs_sampling_freq,
)

# Take a single environment step with positive joint velocity actions
arm_action = np.ones(env.robots[0].robot_model.dof) * 1.0
gripper_action = [1]
action = np.concatenate([arm_action, gripper_action])
env.step(action)

# Now we can analyze what values were recorded
np.set_printoptions(precision=2)
print(f"\nPolicy Frequency: {control_freq}, Observable Sampling Frequency: {obs_sampling_freq}")
print(f"Number of recorded samples after 1 policy step: {buffer._size}\n")
for i in range(buffer._size):
    print(f"Recorded value {i}: {buffer.buf[i]}")
