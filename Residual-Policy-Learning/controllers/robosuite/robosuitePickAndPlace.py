"""
    Robosuite environment for the pick and place task
    Has a position based controller
    NOTE: Does not have a gym type wrapper
"""
import gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

"""
[0] JOINT_VELOCITY - Joint Velocity
[1] JOINT_TORQUE - Joint Torque
[2] JOINT_POSITION - Joint Position
[3] OSC_POSITION - Operational Space Control (Position Only)
[4] OSC_POSE - Operational Space Control (Position + Orientation)
[5] IK_POSE - Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)
"""
options = {}
controller_name = 'OSC_POSITION'
options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

env = GymWrapper(
    suite.make(
        "PickPlaceMilk",                # pickPLaceMilk task
        robots="IIWA",                  # use IIWA robot
        **options,                      # controller options
        use_camera_obs=False,           # do not use pixel observations
        has_offscreen_renderer=False,   # not needed since not using pixel obs
        has_renderer=True,              # make sure we can render to the screen
        reward_shaping=True,            # use dense rewards
        control_freq=20,                # control should happen fast enough so that simulation looks smooth
    )
)

def controller(obs:dict, object_in_hand:bool=False):
    gripper_pos = obs['robot0_eef_pos']
    try:
        object_pos  = obs['Can0_pos']
    except:
        object_pos  = obs['Milk_pos']
    z_table = 0.86109826
    # print(object_pos)
    if not object_in_hand:
        action = 10 * (object_pos[:2] - gripper_pos[:2])
        action = np.hstack((action, [0,-1]))
        if np.linalg.norm((object_pos[:2] - gripper_pos[:2])) < 0.01:
            action = [0,0,-1,-1]
            if np.linalg.norm((object_pos[2] - gripper_pos[2])) < 0.01:
                action = [0,0,0,1]
                object_in_hand = True
    else:
        action = [0,0,1,1]
        if object_pos[2] - z_table > 0.1:
            action = 10 * (goal_pos[:2] - gripper_pos[:2])
            action = np.hstack((action,[0,1]))
            if np.linalg.norm((goal_pos[:2] - gripper_pos[:2])) < 0.01:
                action = [0,0,0,-1]
    return action, object_in_hand

for i_episode in range(20):
    observation = env.reset()
    # import pdb
    # pdb.set_trace()
    object_in_hand = False
    goal_pos = np.array(env.bin2_pos) - np.array(env.bin_size)/4    # robosuite doesn't have a goal definition in observation
    for t in range(500):
        env.render()
        # action = env.action_space.sample()
        action, object_in_hand = controller(env.observation_spec(), object_in_hand)
        observation, reward, done, info = env.step(action)
        if reward == 1:
            print("Episode finished after {} timesteps".format(t + 1))
            break