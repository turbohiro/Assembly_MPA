"""
    Robosuite environment for the nut assembly task with a controller
    This is the same as robosuiteNutAssembly but with a dense reward structure
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

class NutAssemblyDense(gym.Env):
    """
        NutAssembly:
        NutAssembly task from robosuite with no controller. Can be used for learning from scratch.
    """
    def __init__(self, horizon=500, *args, **kwargs):
        options = {}
        controller_name = 'OSC_POSE'
        options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)
        
        self.env = GymWrapper(
            suite.make(
                "NutAssemblyRound",             # Nut Assembly task with the round peg
                robots="IIWA",                  # use IIWA robot
                **options,                      # controller options
                use_object_obs = True,
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=False,              # make sure we can render to the screen
                reward_shaping=True,            # use dense rewards
                reward_scale=1.0,
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
                horizon=horizon,                # number of timesteps for ending episode
            )
        )
        self.max_episode_steps = 500
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_type = 'dense'
        self.distance_threshold = 0.065

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self.env.observation_spec()
        observation = {}
        observation['observation'] = np.hstack((ob['robot0_eef_pos'], ob['robot0_eef_quat'], ob['RoundNut0_pos'], ob['RoundNut0_quat']))
        observation['desired_goal'] = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        observation['achieved_goal'] = ob['RoundNut0_pos']
        info['is_success'] = reward
        return observation, reward, done, info

    def reset(self):
        ob = self.env.reset()
        ob = self.env.observation_spec()
        observation = {}
        observation['observation'] = np.hstack((ob['robot0_eef_pos'], ob['robot0_eef_quat'], ob['RoundNut0_pos'], ob['RoundNut0_quat']))
        observation['desired_goal'] = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        observation['achieved_goal'] = ob['RoundNut0_pos']
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


class NutAssemblyDenseHand(gym.Env):
    """
        TODO: change obs and reward setting
        NutAssemblyHand:
            'FetchPickAndPlace-v1' with a perfect controller
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Parameters:
        -----------
        kp: float
            Scaling factor for position control
    """
    def __init__(self, kp:float=20, *args, **kwargs):
        options = {}
        controller_name = 'JOINT_POSITION'
        options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

        self.env = GymWrapper(
            suite.make(
                "NutAssemblyRound",             # Nut Assembly task with the round peg
                robots="IIWA",                  # use IIWA robot
                **options,                      # controller options
                use_object_obs = True,
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=True,              # make sure we can render to the screen
                reward_shaping=True,            # use dense rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
            )
        )

        self.max_episode_steps = 500
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.active_observables = self.env.active_observables
        self.modalities = self.env.observation_modalities
        self.reward_type = 'dense'
        self.distance_threshold = 0.065

    def step(self, residual_action:np.ndarray):
        controller_action = np.array(self.controller_action(self.last_observation))
        if (controller_action>1).any() or (controller_action<-1).any():
            print(controller_action)
        action = np.add(controller_action, residual_action)
        action = np.clip(action, -1, 1)
        ob, reward, done, info = self.env.step(action)
        ob = self.env.observation_spec()
        observation = {}
        observation['observation'] = np.hstack((ob['robot0_eef_pos'], ob['robot0_eef_quat'], ob['RoundNut_pos'], ob['RoundNut_quat']))
        observation['desired_goal'] = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        observation['achieved_goal'] = ob['RoundNut_pos']
        self.last_observation = observation.copy()
        info['is_success'] = reward
        return observation, reward, done, info

    def reset(self):
        self.env.reset() # reset according to task defaults
        ob = self.env.observation_spec()
        observation = {}
        observation['observation'] = np.hstack((ob['robot0_eef_pos'], ob['robot0_eef_quat'], ob['RoundNut_pos'], ob['RoundNut_quat']))
        observation['desired_goal'] = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        observation['achieved_goal'] = ob['RoundNut_pos']
        self.last_observation = observation.copy()
        self.object_in_hand = False
        self.object_below_hand = False
        self.gripper_reoriented = 0
        return observation

    def seed(self, seed:int=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.env.seed(seed=seed)

    def render(self, mode:str="human",**kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()

    def goal_distance(self, achieved_goal, desired_goal):
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

        z_table = 0.8610982

        object_axang = T.quat2axisangle(object_quat)
        if abs(object_axang[-1] - 1.24) < 0.2:
            object_axang_touse = [0,0,object_axang[-1]%(2*pi/8) + (2*pi/8)]
        else:
            object_axang_touse = [0,0,object_axang[-1]%(2*pi/8)]
        gripper_axang = T.quat2axisangle(gripper_quat)
        # print('object axang to use ' + str(object_axang_touse))

        if self.gripper_reoriented ==0:
            self.gripper_init_quat = gripper_quat
            self.gripper_reoriented = 1

        init_inv = T.quat_inverse(self.gripper_init_quat)
        changing_wf = T.quat_multiply(init_inv,gripper_quat)
        changing_wf_axang = T.quat2axisangle(changing_wf)

        # gripper_quat_inv = T.quat_inverse(gripper_quat)
        # changing_wf = T.quat_multiply(gripper_quat_inv,self.gripper_init_quat)
        # changing_wf_axang = T.quat2axisangle(changing_wf)

        # print('changing wf axis ' +str(changing_wf_axang))

        if not self.object_below_hand or self.gripper_reoriented < 5:
            self.nut_p = T.quat2axisangle(object_quat)[-1]
            # print(self.nut_p)
            if not self.object_below_hand:
                action = 20 * (object_pos[:2] - gripper_pos[:2])
            else:
                action = [0,0]

            action = 20 * (object_pos[:2] - gripper_pos[:2])

            # frac = 0.2 # Rate @ which to rotate gripper about z.
            # ang_goal = frac*self.nut_p # Nut p is the nut's random intial pertubation about z.
            
            # if self.gripper_reoriented == 0:
            #     self.gripper_init_quat = gripper_quat
            # if self.gripper_reoriented < 5: # Gripper should be aligned with nut after 5 action steps
            #     action_angle= [0,0,ang_goal]

            #     #print('object ' + str(object_axang))
            #     #print('gripper ' + str(gripper_axang))

            #     init_inv = T.quat_inverse(self.gripper_init_quat)
            #     init_inv_mat = T.quat2mat(init_inv)

            #     rel = T.quat_multiply(gripper_quat,init_inv)
            #     rel_axang = T.quat2axisangle(rel)
            #     #print('rel_axang ' +str(rel_axang))

            #     rel_axang_WF = np.matmul(init_inv_mat,rel_axang)

            #     #print('rel_axang_WF ' +str(rel_axang_WF))

            #     if take_action:
            #         self.gripper_reoriented+=1
            # else: # After 5 action steps, don't rotate gripper any more
            #     action_angle=[0,0,0]

            action_angle = 0.2*(object_axang_touse - changing_wf_axang)
            action_angle = [0,0,action_angle[-1]]
            #action_angle = [0,0,0]
            #print('action angle ' +str(action_angle))

            if np.linalg.norm(object_axang_touse - changing_wf_axang) <0.1:
                if take_action:
                    self.gripper_reoriented = 5

            action = np.hstack((action, [0], action_angle, [-1]))
            if np.linalg.norm((object_pos[:2] - gripper_pos[:2])) < 0.01:
                if take_action:
                    self.object_below_hand = True
                    #self.gripper_reoriented = 5

        elif not self.object_in_hand: # Close gripper
            action = [0,0,-1,0,0,0,-1]
            if np.linalg.norm((object_pos[2] - gripper_pos[2])) < 0.01:
                action = [0,0,0,0,0,0,1]
                if take_action:
                    self.object_in_hand = True
        
        else: # Move gripper up and toward goal position
            action = [0,0,1,0,0,0,1]
            if object_pos[2] - z_table > 0.1:
                action = 20 * (goal_pos[:2] - object_pos[:2])
                action = np.hstack((action,[0,0,0,0,1]))
                if np.linalg.norm((goal_pos[:2] - object_pos[:2])) < 0.0225:
                    action = [0,0,0,0,0,0,-1] # Drop nut once it's close enough to the peg

        action = np.clip(action, -1, 1)
        return action

if __name__ == "__main__":
    env_name = 'NutAssemblyDenseHand'
    env = globals()[env_name]() # this will initialise the class as per the string env_name
    # env = gym.wrappers.Monitor(env, 'video/' + env_name, force=True)
    successes = []
    # set the seeds
    env.seed(1)
    env.action_space.seed(1)
    env.observation_space.seed(1)
    failed_eps = []
    for ep in range(10):
        success = np.zeros(env.max_episode_steps)
        # print('_'*50)
        obs = env.reset()
        # print(obs.keys())
        action = [0,0,0,0,0,0,0]  # give zero action at first time step
        for i in (range(env.max_episode_steps)):
            env.render()
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
