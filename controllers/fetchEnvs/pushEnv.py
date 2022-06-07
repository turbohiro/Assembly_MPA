"""
    Collection of variations in the slide environment
    Contains the following:
    FetchPush:
        The vanilla 'FetchSlide-v1' without any controller
        Can be used for learning with RL from scratch
        Action taken as:
            Pi_theta(s) = f(s)
    FetchPushImperfect:
        'FetchPush-v1' with an imperfect controller
        This controller can at least push the puck but not perfectly
        Action taken as:
            Pi_theta(s) = f(s) + pi_theta(s)
    FetchPushSlippery:
        'FetchPush-v1' with the same imperfect controller as FetchPushImperfect
        The friction coefficient between puck and table is changed from 1.0 to 0.1
        Action taken as:
            Pi_theta(s) = f(s) + pi_theta(s)
"""
import gym
from gym.utils import seeding
import numpy as np
from typing import Dict

class FetchPush(gym.Env):
    """
        FetchPush:
        The vanilla 'FetchFetchPush-v1' without any controller
        Can be used for learning with RL from scratch
        No changes made to the original environment
    """
    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make('FetchPush-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space

    def step(self, action):
        observation, reward, done, info = self.fetch_env.step(action)
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
        return observation

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.fetch_env.seed(seed=seed)

    def render(self, mode="human", *args, **kwargs):
        return self.fetch_env.render(mode, *args, **kwargs)

    def close(self):
        return self.fetch_env.close()

    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

class FetchPushImperfect(gym.Env):
    """
        FetchPushImperfect:
            'FetchPush-v1' with an imperfect controller
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Parameters:
        -----------
        kp: float
            Scaling factor for position control (kind of)
        push: float
            Scaling factor to push the puck
    """
    def __init__(self, kp:float=5, push:float=5, *args, **kwargs):
        self.fetch_env = gym.make('FetchPush-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
        ############################
        self.kp = kp
        self.push = push
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False
        self.r = self.fetch_env.env.sim.model.geom_size[-1][0]      # height of the puck
        ############################

    def step(self, residual_action:np.ndarray):
        controller_action = np.array(self.controller_action(self.last_observation))
        if (controller_action>1).any() or (controller_action<-1).any():
            print(controller_action)
        action = np.add(controller_action, residual_action)
        action = np.clip(action, -1, 1)
        observation, reward, done, info = self.fetch_env.step(action)
        self.last_observation = observation.copy()
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
        self.last_observation = observation.copy()
        ############################
        # parameters for the imperfect controller
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False
        ############################
        return observation

    def seed(self, seed:int=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.fetch_env.seed(seed=seed)

    def render(self, mode:str="human", *args, **kwargs):
        return self.fetch_env.render(mode, *args, **kwargs)

    def close(self):
        return self.fetch_env.close()

    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

    def controller_action(self, obs:Dict, take_action:bool=True, DEBUG:bool=False):
        """
            Given an observation return actions according
            to an imperfect controller
            [grip_pos, object_pos, object_rel_pos, gripper_state, object_rot,
                     object_velp, object_velr, grip_velp, gripper_vel]
            take_action: bool
                Whether use this to take action in environment or just use for subtracting from rand actions
        """
        grip_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        object_rel_pos = obs['observation'][6:9]
        goal_pos = obs['desired_goal']
        # lift the hand little from the table vertically
        if not self.hand_higher:
            action = [0,0,1,0]
            if grip_pos[2]-object_pos[2] > 0.05:
                if take_action:
                    self.hand_higher = True
                if DEBUG:
                    print('Hand lifted from the table')
        # once above, move it behind
        if self.hand_higher and not self.hand_behind:
            goal_grip_pos = object_pos + (2*self.r) * (object_pos - goal_pos) / np.linalg.norm(object_pos - goal_pos)
            action_pos = list(self.kp * (goal_grip_pos - grip_pos))
            action = action_pos[:2] + [0,0]
            if np.linalg.norm(grip_pos[:2]-goal_grip_pos[:2]) < 0.0005:
                if take_action:
                    self.hand_behind = True
                if DEBUG:
                    print('Hand has moved behind')
        # now move the hand down
        if self.hand_behind and not self.hand_down:
            action = [0,0,-1,0]
            if grip_pos[2]-object_pos[2] < 0.005:
                if take_action:
                    self.hand_down = True
                if DEBUG:
                    print('Ready to HIT')
        # # now move the hand down
        # if self.hand_behind and not self.hand_down:
        #     action = [0,0,-1,0]
        #     if grip_pos[2]-object_pos[2]<0.01:
        #         if take_action:
        #             self.hand_down = True
        #         if DEBUG:
        #             print('Ready to HIT')
        # now give impulse
        if self.hand_down:
            action_pos = list(self.push * (goal_pos[:2]-grip_pos[:2]))
            action = action_pos + [0,0]
        action = np.array(action)
        # return action
        return np.clip(action,-1,1)

class FetchPushSlippery(gym.Env):
    """
        FetchPushSlippery:
            'FetchPush-v1' with an imperfect controller and lesser friction
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Parameters:
        -----------
        kp: float
            Scaling factor for position control (kind of)
        push: float
            Scaling factor to push the puck
    """
    def __init__(self, kp:float=10, push:float=2, new_mu:float=0.1, *args, **kwargs):
        self.fetch_env = gym.make('FetchPush-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
        num_contacts = len(self.fetch_env.env.sim.model.geom_friction)
        # change friction values
        for i in range(len(self.fetch_env.env.sim.model.geom_friction)):
            # only change for table and puck
            if i>num_contacts-3:
                self.fetch_env.env.sim.model.geom_friction[i] = [new_mu, 5.e-3, 1e-4]
        ############################
        self.kp = kp
        self.push = push
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False
        self.r = self.fetch_env.env.sim.model.geom_size[-1][0]      # height of the puck
        ############################

    def step(self, residual_action:np.ndarray):
        controller_action = np.array(self.controller_action(self.last_observation))
        if (controller_action>1).any() or (controller_action<-1).any():
            print(controller_action)
        action = np.add(controller_action, residual_action)
        action = np.clip(action, -1, 1)
        observation, reward, done, info = self.fetch_env.step(action)
        self.last_observation = observation.copy()
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
        self.last_observation = observation.copy()
        ############################
        # parameters for the imperfect controller
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False
        ############################
        return observation

    def seed(self, seed:int=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.fetch_env.seed(seed=seed)

    def render(self, mode:str="human", *args, **kwargs):
        return self.fetch_env.render(mode, *args, **kwargs)

    def close(self):
        return self.fetch_env.close()

    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

    def controller_action(self, obs:Dict, take_action:bool=True, DEBUG:bool=False):
        """
            Given an observation return actions according
            to an imperfect controller
            [grip_pos, object_pos, object_rel_pos, gripper_state, object_rot,
                     object_velp, object_velr, grip_velp, gripper_vel]
            take_action: bool
                Whether use this to take action in environment or just use for subtracting from rand actions
        """
        grip_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        object_rel_pos = obs['observation'][6:9]
        goal_pos = obs['desired_goal']
        # lift the hand little from the table vertically
        if not self.hand_higher:
            action = [0,0,1,0]
            if grip_pos[2]-object_pos[2] > 0.05:
                if take_action:
                    self.hand_higher = True
                if DEBUG:
                    print('Hand lifted from the table')
        # once above, move it behind
        if self.hand_higher and not self.hand_behind:
            goal_grip_pos = object_pos + (2*self.r) * (object_pos - goal_pos) / np.linalg.norm(object_pos - goal_pos)
            action_pos = list(self.kp * (goal_grip_pos - grip_pos))
            action = action_pos[:2] + [0,0]
            if np.linalg.norm(grip_pos[:2]-goal_grip_pos[:2]) < 0.0005:
                if take_action:
                    self.hand_behind = True
                if DEBUG:
                    print('Hand has moved behind')
        # now move the hand down
        if self.hand_behind and not self.hand_down:
            action = [0,0,-1,0]
            if grip_pos[2]-object_pos[2] < 0.005:
                if take_action:
                    self.hand_down = True
                if DEBUG:
                    print('Ready to HIT')
        # # now move the hand down
        # if self.hand_behind and not self.hand_down:
        #     action = [0,0,-1,0]
        #     if grip_pos[2]-object_pos[2]<0.01:
        #         if take_action:
        #             self.hand_down = True
        #         if DEBUG:
        #             print('Ready to HIT')
        # now give impulse
        if self.hand_down:
            action_pos = list(self.push * (goal_pos[:2]-grip_pos[:2]))
            action = action_pos + [0,0]
        action = np.array(action)
        # return action
        return np.clip(action,-1,1)

if __name__ == "__main__":
    # env_name = 'FetchPushImperfect'
    env_name = 'FetchPushSlippery'
    env = globals()[env_name]() # this will initialise the class as per the string env_nameå
    # env = gym.wrappers.Monitor(env, 'video/' + env_name, force=True) # save video
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
        action = [0,0,0,0]  # give zero action at first time step
        for i in (range(env.max_episode_steps)):
            env.render(mode='human')
            obs, rew, done, info = env.step(action)
            success[i] = info['is_success']
        ep_success = info['is_success']
        if not ep_success:
            failed_eps.append(ep)
        successes.append(ep_success)
        # print(f'Episode:{ep} Success:{success}')
    print(f'Success Rate:{sum(successes)/len(successes)}')
    print(f'Failed Episodes:{failed_eps}')
    env.close()
