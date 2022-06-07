#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:00:07 2022

@author: wchen
"""

import numpy as np
import robosuite as suite
import pdb
from robosuite.controllers import load_controller_config
from robosuite.utils.observables import Observable

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="JOINT_POSITION")


# create environment instance
env5 = suite.make(
    env_name="Wipe", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    gripper_types="WipingGripper",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    has_renderer=True,
    render_camera='robot0_eye_in_hand',  #('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand').
    has_offscreen_renderer=False,
    render_visual_mesh = True,
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
    renderer="mujoco",
    renderer_config=None,
)


# create an environment to visualize on-screen
env2 = suite.make(
    "TwoArmLift",
    robots=["Sawyer", "Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=True,                      # on-screen rendering
    render_camera="frontview",              # visualize the "frontview" camera
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # no observations needed
    use_camera_obs=False,                   # no observations needed
)

# create an environment for policy learning from low-dimensional observations
env = suite.make(
    "TwoArmLift",
    robots=["Sawyer", "Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=True,                     # no on-screen rendering
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                    # provide object observations to agent
    use_camera_obs=False,                   # don't provide image observations to agent
    reward_shaping=True,                    # use a dense reward signal for learning
)

# create an environment for policy learning from pixels
env3 = suite.make(
    "TwoArmLift",
    robots=["Sawyer", "Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # each arm is controlled using OSC
    env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                   # provide image observations to agent
    camera_names="agentview",               # use "agentview" camera for observations
    camera_heights=84,                      # image height
    camera_widths=84,                       # image width
    reward_shaping=True,                    # use a dense reward signal for learning
)

# create an environment for policy learning from pixels
env4 = suite.make(
    "Wipe",
    robots='Panda',             # load a Sawyer robot and a Panda robot
    gripper_types="WipingGripper",                # use default grippers per robot arm
    initialization_noise="default",
    use_camera_obs=True,
    use_object_obs=True,
    reward_scale=1.0,
    reward_shaping=True,
    has_renderer=False,
    has_offscreen_renderer=True,
    render_camera="frontview",
    render_collision_mesh=False,
    render_visual_mesh=True,
    render_gpu_device_id=-1,
    control_freq=20,
    horizon=1000,
    ignore_done=False,
    hard_reset=True,
    camera_names="agentview",
    camera_heights=256,
    camera_widths=256,
    camera_depths=False,
    camera_segmentations=None,  # {None, instance, class, element}
    renderer="mujoco",
    renderer_config=None,
)




def get_policy_action(obs):
    # a trained policy could be used here, but we choose a random action
    low, high = env4.action_spec
    return np.random.uniform(low, high)

# reset the environment to prepare for a rollout
obs = env4.reset()

pdb.set_trace()

done = False
ret = 0.
while not done:
    action = get_policy_action(obs)         # use observation to decide on an action

    obs, reward, done, _ = env4.step(action) # play action
    env4.render()
    ret += reward
    print(env4)
    
    #env5.render()
print("rollout completed with return {}".format(ret))