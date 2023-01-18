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
import pdb
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
                "PegAssembly",            
                robots="LWR",                  # use LWR robot
                **options,                      # controller options
                use_object_obs = True,
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=False,              # make sure we can render to the screen
                reward_shaping=False,            # use sparse rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
            )
        )
        self.max_episode_steps = 500
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_type = 'sparse'
        self.distance_threshold = 0.01

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self.env.observation_spec()
        observation = {}
        observation['observation'] = np.hstack((ob['robot0_eef_pos'], ob['robot0_eef_quat'],ob['robot0_eef_force'], ob['robot0_eef_torque'], ob['PegCircle_pos'], ob['PegCircle_quat']))
        observation['desired_goal'] = np.array(self.env.sim.data.body_xpos[self.env.hole_body_id]) 
        observation['achieved_goal'] = ob['robot0_eef_pos']
        info['is_success'] = reward
        #print(self.env.sim.data.qpos[:7])


       
        return observation, reward, done, info

    def reset(self):
        ob = self.env.reset()
        ob = self.env.observation_spec()
        observation = {}
        observation['observation'] = np.hstack((ob['robot0_eef_pos'], ob['robot0_eef_quat'],ob['robot0_eef_force'], ob['robot0_eef_torque'], ob['PegCircle_pos'], ob['PegCircle_quat']))
        observation['desired_goal'] = np.array(self.env.sim.data.body_xpos[self.env.hole_body_id]) 
        observation['achieved_goal'] = ob['robot0_eef_pos']
        
        return observation

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

class MultiPegAssemblyHand(gym.Env):
    """
        NutAssembly: with no controller
        NutAssembly task from robosuite with no controller. Can be used for learning from scratch.
    """
    def __init__(self, *args, **kwargs):
        options = {}
        controller_name = 'OSC_POSE'  #OSC_POSE
        options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)
        
        self.env = GymWrapper(
            suite.make(
                "PegAssembly",            
                robots="LWR",                  # use LWR robot
                **options,                      # controller options
                use_object_obs = True,
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=False,              # make sure we can render to the screen
                reward_shaping=False,            # use sparse rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
            )
        )
        self.max_episode_steps = 200
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_type = 'sparse'
        self.distance_threshold = 0.01
        self.sticky_prob = 0.5

    def step(self, residual_action:np.ndarray):

        controller_action = np.array(self.controller_action(self.last_observation))
        
        if (controller_action>1).any() or (controller_action<-1).any():
            print(controller_action)
        if controller_action[6] !=1:
            action = np.add(controller_action, residual_action)
            action = np.clip(action, -1, 1)
        else:
            action = [0,0,0,0,0,0,1]

        #print('current action is',action)
        
        ob, reward, done, info = self.env.step(action)
        ob = self.env.observation_spec()
        observation = {}
        observation['observation'] = np.hstack((ob['robot0_eef_pos'], ob['robot0_eef_quat'],ob['robot0_eef_force'], ob['robot0_eef_torque'], ob['PegCircle_pos'], ob['PegCircle_quat']))
        observation['desired_goal'] = np.array(self.env.sim.data.body_xpos[self.env.hole_body_id])
        observation['achieved_goal'] = ob['robot0_eef_pos'] #PegCircle_pos
        self.last_observation = observation.copy()
        info['is_success'] = reward
        #print(observation['achieved_goal'][2]-observation['desired_goal'][2])
        #print('achived goal position and goal position is', observation['achieved_goal'], observation['desired_goal'])
        return observation, reward, done, info

    def reset(self):
        ob = self.env.reset()
        ob = self.env.observation_spec()
        observation = {}
        observation['observation'] = np.hstack((ob['robot0_eef_pos'], ob['robot0_eef_quat'],ob['robot0_eef_force'], ob['robot0_eef_torque'], ob['PegCircle_pos'], ob['PegCircle_quat']))
        observation['desired_goal'] = np.array(self.env.sim.data.body_xpos[self.env.hole_body_id])
        observation['achieved_goal'] = ob['robot0_eef_pos']

        self.last_observation = observation.copy()
        self.object_in_hand = False
        self.object_below_hand = False
        self.gripper_reoriented = 0
        self.ori_adjust_num = 0
        self.prev_action = [0,0,0,0,0,0,0]
        return observation

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
    def controller_action(self, obs:dict, take_action:bool=True, DEBUG:bool=False):
        observation = obs['observation']
        goal_pos = obs['desired_goal']
        achieved_goal = obs['achieved_goal']

        gripper_pos = observation[:3]
        gripper_quat = observation[3:7]
        object_pos  = observation[7:10]
        object_quat = observation[10:]

        z_table = 0.8
        #action = [0,0,1,0,0,0,-1]
        
        #if achieved_goal[2] - z_table >= 0.06:
        #    action1 = 2 * (goal_pos[0] - achieved_goal[0])
        #    action2 = 2 * (goal_pos[1] - achieved_goal[1])
        #    action3 = 0.2 * (goal_pos[2] - achieved_goal[2])
        #    action = np.hstack(([action1,action2,action3],[0,0,0,-1]))
        #if 0.06>(achieved_goal[2] - z_table) and  (achieved_goal[2] - z_table)> 0.03:
        #    action3 = 0.5 * (goal_pos[2] - achieved_goal[2])
        #    action = np.hstack(([0,0,action3],[0,0,0,-1]))
        #    if np.linalg.norm((goal_pos - achieved_goal)) < 0.05:
        #        action = [0,0,0,0,0,0,-1] # Drop nut once it's close enough to the peg

        
            #action3 = 0.1 * (goal_pos[2] - achieved_goal[2])
        action3 = 0.5 * (goal_pos[2] - achieved_goal[2])
            #action0 = 1.0 * (goal_pos[0] - achieved_goal[0])
            #action1 = 1.0 * (goal_pos[1] - achieved_goal[1])
        action = np.hstack(([0,0,action3],[0,0,0,-1]))
            #print('final distance metric is',np.linalg.norm((goal_pos - achieved_goal)) )
            #if np.linalg.norm((goal_pos - achieved_goal)) < 0.3:
            #    action = [0,0,0,0,0,0,1] # Drop nut once it's close enough to the peg
        
        if (achieved_goal[2] -  goal_pos[2])< 0.26:   
            action = [0,0,0,0,0,0,1]
            
        action = np.clip(action, -1, 1)

        if np.random.random() < self.sticky_prob:
            action = np.array(self.prev_action)
        else:
            self.prev_action = action
        
        return action


if __name__ == "__main__":
    env_name = 'MultiPegAssemblyHand'
    env = globals()[env_name]() # this will initialise the class as per the string env_name
    #env = gym.wrappers.Monitor(env, 'video/' + env_name, force=True)
    successes = []
    # set the seeds
    env.seed(1)
    env.action_space.seed(1)
    env.observation_space.seed(1)
    failed_eps = []
    for ep in range(20):
        success = np.zeros(env.max_episode_steps)
        # print('_'*50)
        obs = env.reset()
        action = [0,0,0,0,0,0,-1]
        
        for i in (range(env.max_episode_steps)):

            #env.render()   
            #action = [0,0,0,0,0,0,-1]  
            #pdb.set_trace() 
            obs, rew, done, info = env.step(action)
            
            #print(action)
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

