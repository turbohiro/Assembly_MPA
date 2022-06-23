"""
    Robosuite environment for the nut assembly task with a controller
    Refer the report for more details on the controller
"""
import gym
from gym.utils import seeding
import numpy as np
from typing import Dict
import os

# Additional libraries needed for robosuite
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

import robosuite.utils.transform_utils as T

import platform

from math import pi

# for OpenMP error on MacOS with dylib files
# check https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
if 'Darwin' in platform.platform():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MultiPegAssembly(gym.Env):
    """
        NutAssembly: with no controller
        NutAssembly task from robosuite with no controller. Can be used for learning from scratch.
    """
    def __init__(self, *args, **kwargs):
        options = {}
        controller_name = 'OSC_POSE'
        options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)
        
        self.env = GymWrapper(
            suite.make(
                "PegAssemblySquare",            
                robots="LWR",                  # use LWR robot
                **options,                      # controller options
                use_object_obs = False,
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=True,              # make sure we can render to the screen
                reward_shaping=False,            # use sparse rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
            )
        )
        self.max_episode_steps = 500
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_type = 'sparse'
        self.distance_threshold = 0.065

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self.env.observation_spec()
        observation = {}
        info['is_success'] = reward
        return observation, reward, done, info

    def reset(self):
        ob = self.env.reset()
        #ob = self.env.observation_spec()
        return ob

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.env.seed(seed=seed)

    def render(self, mode="human", *args, **kwargs):
        return self.env.render()

    def close(self):
        return self.env.close()

    def goal_distance(self, achieved_goal,desired_goal):
        return np.linalg.norm(achieved_goal-desired_goal, axis = 1)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

import pdb
if __name__ == "__main__":
    env_name = 'MultiPegAssembly'
    env = globals()[env_name]() # this will initialise the class as per the string env_name
    #env = gym.wrappers.Monitor(env, 'video/' + env_name, force=True)
    successes = []
    # set the seeds
    env.seed(1)
    env.action_space.seed(1)
    env.observation_space.seed(1)
    failed_eps = []
    #action_0 = [0,0,0,0,0,0,-1]
    #env.step(action_0)
    for ep in range(100):
        success = np.zeros(env.max_episode_steps)
        # print('_'*50)
        obs = env.reset()
        action_0 = [0,0,0,0,0,0,-1]
        #env.step(action_0)
        # import pdb
        # pdb.set_trace()
        # print(obs.keys())
        #action = [0,0,0,0,0,0,-1]  # give zero action at first time step
 
        action = [0,0,0,0,0,0,-0.1]
        for i in (range(env.max_episode_steps)):

            env.render()            
            #obs, rew, done, info = env.step(action)
            #action = [0,0,0,0,0,0,0]
            obs, rew, done, info = env.step(action)
            success[i] = info['is_success']
        ep_success = info['is_success']
        if not ep_success:
            failed_eps.append(ep)
        successes.append(ep_success)
        print('this is successes ' + str(successes))
        # print(f'Episode:{ep} Success:{success}')
    print(f'Success Rate:{sum(successes)/len(successes)}')
    print(f'Failed Episodes:{failed_eps}')
    env.close()

