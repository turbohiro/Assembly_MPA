"""
    To evaluate the success rate with just the controller for baselines
    Tested only for fetchEnvs
    TODO: @nsidn98 add robosuite compatibility
"""
import copy
import gym
import argparse
import os
import numpy as np
import time

class Agent:
    def __init__(self, args:argparse.Namespace, env, writer=None):
        """
            Module for the DDPG agent along with HER
            Parameters:
            -----------
            args: argparse.Namespace
                args should contain the following:
            env: gym.Env
                OpenAI type gym environment
            writer: tensorboardX
                tensorboardX to log metrics like losses, rewards, success_rates, etc.
        """
        self.args = args
        self.env = env
        self.env_params = self.get_env_params(env)
        self.writer = writer

    def get_env_params(self, env):
        """
            Get the environment parameters
        """
        obs = env.reset()
        # close the environment
        params = {'obs': obs['observation'].shape[0],
                'goal': obs['desired_goal'].shape[0],
                'action': env.action_space.shape[0],
                'action_max': env.action_space.high[0],
                }
        try:
            params['max_timesteps'] = env._max_episode_steps
        # for custom envs
        except:
            params['max_timesteps'] = env.max_episode_steps
        return params
      
    def evaluate_controller(self):
        """
            Run the episodes with controller and evaluate
            Adding epochs, cycles as for loops so that the wandboard
            timestep matches with other plots
        """
        print('_'*50)
        print('Beginning the evaluation...')
        print('_'*50)
        num_cycles = 0
        for epoch in range(self.args.n_epochs):
            # evaluate the agent
            success_rate = self.eval_agent()
            print(f'Epoch:{epoch}\tSuccess Rate:{success_rate:.3f}')
            #time.sleep(2)
                 
    def eval_agent(self) -> float:
        """
            Evaluate the agent using just the controller
            performs n_test_rollouts in the environment
        """
        successes = []
        for _ in range(1):
            success = np.zeros(self.env_params['max_timesteps'])          
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for i in range(self.env_params['max_timesteps']):
                actions = np.array([0,0,0,0])
                observation_new, _, _, info = self.env.step(actions)
                time.sleep(0.001)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                success[i] = info['is_success']
            successes.append(success)
            self.env.render(mode='human')    ##show the rendering
            
        successes = np.array(successes)
        return np.mean(successes[:,-1]) # return mean of only final steps success


        


if __name__ == "__main__":

    from ddpg_config import args,load_checkpoint
    from utils import  make_env

    env = make_env(args.env_name)   # initialise the environment

    # set the random seeds for everything
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    #####################################

    # initialise the agent
    trainer = Agent(args, env,None)
    trainer.evaluate_controller()

    # close env and writer
    env.close()
