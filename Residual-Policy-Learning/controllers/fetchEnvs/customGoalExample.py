"""
    Example of setting goals at custom locations for Slide environment
"""
import gym
import numpy as np
import time
from tqdm import tqdm
from typing import Dict
from gym.utils import seeding


class FetchSlide(gym.Env):
    """
        FetchSlide:
    """
    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make('FetchSlide-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
    
    def step(self, action):
        observation, reward, done, info = self.fetch_env.step(action)
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
        self.fetch_env.env.goal = np.array([1,1, 0.41401894])   # change goal location HERE; z=0.414 because it is table surface
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

if __name__ == "__main__":
    
    env = FetchSlide()
    env_name = 'FetchSlideCustomGoal'
    obs = env.reset()

    action = [0,0,0,0]   # give zero action at first time step
    # time.sleep(5)
    for i in tqdm(range(50)):
        obs, rew, done, info = env.step(action)
        env.render(mode='human')
    print('Done')
