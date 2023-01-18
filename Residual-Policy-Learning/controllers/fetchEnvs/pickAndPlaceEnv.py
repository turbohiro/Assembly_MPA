"""
    Collection of variations in the slide environment
    Contains the following:
    FetchPickAndPlace:
        The vanilla 'FetchPickAndPlace-v1' without any controller
        Can be used for learning with RL from scratch
        Action taken as:
            Pi_theta(s) = f(s)
    FetchPickAndPlacePerfect:
        'FetchPickAndPlace-v1' with a perfect controller
        The controller is a proportional controller.
        Action taken as:
            Pi_theta(s) = f(s) + pi_theta(s)
    FetchPickAndPlaceSticky:
        'FetchPickAndPlace-v1' with the same perfect controller as FetchPickAndPlacePerfect
        but the action taken is the same as the previous step with a probability of 0.5
        Action taken as:
            Pi_theta(s) = f(s) + pi_theta(s)
    FetchPickAndPlaceNoisy:
        'FetchPickAndPlace-v1' with the same perfect controller as FetchPickAndPlacePerfect
        but the action taken has some added random noise
        Action taken as:
            Pi_theta(s) = f(s) + pi_theta(s)
"""
import gym
from gym.utils import seeding
import numpy as np
from typing import Dict

class FetchPickAndPlace(gym.Env):
    """
        FetchPickAndPlace:
        The vanilla 'FetchPickAndPlace-v1' without any controller
        Can be used for learning with RL from scratch
        No changes made to the original environment
    """
    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make('FetchPickAndPlace-v1')
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

class FetchPickAndPlacePerfect(gym.Env):
    """
        FetchPickAndPlacePerfect:
            'FetchPickAndPlace-v1' with a perfect controller
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Parameters:
        -----------
        kp: float
            Scaling factor for position control
    """
    def __init__(self, kp:float=10, *args, **kwargs):
        self.fetch_env = gym.make('FetchPickAndPlace-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
        ############################
        self.kp = kp
        self.dist_threshold = 0.002
        self.height_threshold = 0.003
        self.object_in_hand = False
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
        self.object_in_hand = False
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

        action_pos = list(self.kp * object_rel_pos)
        if not self.object_in_hand:
            if np.linalg.norm(object_rel_pos[:2]) > self.dist_threshold:
                action = action_pos[:2] + [0,0]
            else:
                action = action_pos[:3] + [1]   # open the gripper
                # if we are close to the object close the gripper
                if np.linalg.norm(action_pos) < self.height_threshold:
                    action = [0,0,0] + [-1]  # close the gripper
                    if take_action:
                        self.object_in_hand = True
        # once object is in hand, move towards goal
        else:
            p_rel = obs['desired_goal'] - obs['achieved_goal']
            action_pos =  list(self.kp * p_rel)
            action = action_pos + [-1]
        action = np.array(action)
        # return action
        controller_action_out = np.clip(action,-1,1)
        return controller_action_out

class FetchPickAndPlaceSticky(gym.Env):
    """
        FetchPickAndPlaceSticky:
            'FetchPickAndPlace-v1' with a perfect controller
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Will take same action as previous one with probability `sticky_prob`
        Parameters:
        -----------
        kp: float
            Scaling factor for position control
    """
    def __init__(self, kp:float=10, *args, **kwargs):
        self.fetch_env = gym.make('FetchPickAndPlace-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
        ############################
        self.kp = kp
        self.dist_threshold = 0.002
        self.height_threshold = 0.003
        self.object_in_hand = False
        self.sticky_prob = 0.5
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
        self.object_in_hand = False
        self.prev_action = [0,0,0,0]
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

        action_pos = list(self.kp * object_rel_pos)
        if not self.object_in_hand:
            if np.linalg.norm(object_rel_pos[:2]) > self.dist_threshold:
                action = action_pos[:2] + [0,0]
            else:
                action = action_pos[:3] + [1]   # open the gripper
                # if we are close to the object close the gripper
                if np.linalg.norm(action_pos) < self.height_threshold:
                    action = action_pos[:3] + [-1]  # close the gripper
                    if take_action:
                        self.object_in_hand = True
        # once object is in hand, move towards goal
        else:
            p_rel = obs['desired_goal'] - obs['achieved_goal']
            action_pos =  list(self.kp * p_rel)
            action = action_pos + [-1]
        action = np.array(action)
        # return action
        if np.random.random() < self.sticky_prob:
            action = np.array(self.prev_action)
        else:
            self.prev_action = action
        return np.clip(action,-1,1)

class FetchPickAndPlaceNoisy(gym.Env):
    """
        FetchPickAndPlaceSticky:
            'FetchPickAndPlace-v1' with a perfect controller
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Will take same action as previous one with probability `sticky_prob`
        Parameters:
        -----------
        kp: float
            Scaling factor for position control
    """
    def __init__(self, kp:float=10, *args, **kwargs):
        self.fetch_env = gym.make('FetchPickAndPlace-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
        ############################
        self.kp = kp
        self.dist_threshold = 0.002
        self.height_threshold = 0.003
        self.object_in_hand = False
        self.noise_amplitude = 0.08
        self.grab_location = np.array([0,0,0])
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
        self.object_in_hand = False
        self.grab_location = np.array([0,0,0])
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
        # Add offset to estimate

        goal_pos = obs['desired_goal']

        # set grab location only once, at the first measurement, with some noise
        if np.linalg.norm(self.grab_location)<0.001:
            # only in-plane noise
            self.grab_location = object_pos + np.append(self.noise_amplitude*np.random.random_sample(2) - self.noise_amplitude/2,0)

        object_rel_pos = self.grab_location - grip_pos
        action_pos = list(self.kp * object_rel_pos)
        if not self.object_in_hand:
            # print(object_pos)
            if np.linalg.norm(object_rel_pos[:2]) > self.dist_threshold:
                action = action_pos[:2] + [0,0]
            else:
                action = action_pos[:3] + [1]   # open the gripper
                # if we are close to the object close the gripper
                if np.linalg.norm(action_pos) < self.height_threshold:
                    action = action_pos[:3] + [-1]  # close the gripper
                    if take_action:
                        self.object_in_hand = True
        # once object is in hand, move towards goal
        else:
            p_rel = obs['desired_goal'] - obs['achieved_goal']
            action_pos =  list(self.kp * p_rel)
            action = action_pos + [-1]
        action = np.array(action)
        # return action
        return np.clip(action,-1,1)

if __name__ == "__main__":
    # env_name = 'FetchPickAndPlacePerfect'
    # env_name = 'FetchPickAndPlaceSticky'
    env_name = 'FetchPickAndPlaceNoisy'
    env = globals()[env_name]() # this will initialise the class as per the string env_name
    # env = gym.wrappers.Monitor(env, 'video/' + env_name, force=True)
    successes = []
    # set the seeds
    env.seed(1)
    env.action_space.seed(1)
    env.observation_space.seed(1)
    failed_eps = []
    for ep in range(100):
        success = np.zeros(env.max_episode_steps)
        # print('_'*50)
        obs = env.reset()
        action = [0,0,0,0]  # give zero action at first time step
        for i in (range(env.max_episode_steps)):
            # env.render(mode='human')
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
