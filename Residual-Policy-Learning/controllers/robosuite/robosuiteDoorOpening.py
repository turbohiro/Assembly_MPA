"""
    Robosuite environment for the Door opening task with a controller
    NOTE: This is under construction
"""
import gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
import imageio

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

cameraName = "birdview"
skip_frame = 2
env = suite.make(
    "Door",
    robots="IIWA",
    **options,
    has_renderer=False,
    ignore_done=True,
    use_camera_obs=True,
    use_object_obs=True,
    camera_names=cameraName,
    camera_heights=512,
    camera_widths=512,
)


def controller(obs:dict, object_in_hand:bool=False):
    gripper_pos = obs['robot0_eef_pos']
    try:
        object_pos  = obs['Can0_pos']
    except:
        object_pos  = obs['Milk0_pos']
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

writer = imageio.get_writer('video/Door.mp4', fps=20)
frames = []

for i_episode in range(2):
    observation = env.reset()
    object_in_hand = False
    # goal_pos = np.array(env.bin2_pos) - np.array(env.bin_size)/4    # robosuite doesn't have a goal definition in observation
    for t in range(30):
        # action = env.action_space.sample()
        # Temporarily set action to all 0s in lieu of a real controller currently
        action = [0,0,0,0]
        observation, reward, done, info = env.step(action)
        #if t % skip_frame == 0:
        #    frame = observation[cameraName + "_image"][::-1]
        #    writer.append_data(frame)

        if reward == 1:
            print("Episode finished after {} timesteps".format(t + 1))
            break
writer.close()
