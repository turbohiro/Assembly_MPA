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
            for cycle in range(self.args.n_cycles):
                # for batch in range(self.args.n_batches):
                if self.writer:
                    self.writer.add_scalar('Dummy Losses/dummy loss', 1, num_cycles)
                num_cycles += 1
            # evaluate the agent
            success_rate = self.eval_agent()
            print(f'Epoch:{epoch}\tSuccess Rate:{success_rate:.3f}')
            if self.writer:
                self.writer.add_scalar('Success Rate/Success Rate', success_rate, epoch)
                 
    def eval_agent(self) -> float:
        """
            Evaluate the agent using just the controller
            performs n_test_rollouts in the environment
        """
        successes = []
        for _ in range(self.args.n_test_rollouts):
            success = np.zeros(self.env_params['max_timesteps'])
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for i in range(self.env_params['max_timesteps']):
                actions = np.array([0,0,0,0])
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                success[i] = info['is_success']
            successes.append(success)
        successes = np.array(successes)
        return np.mean(successes[:,-1]) # return mean of only final steps success


if __name__ == "__main__":
    # run from main folder as `python RL/baseline.py`
    import wandb
    from torch.utils.tensorboard import SummaryWriter

    from ddpg_config import args
    from utils import connected_to_internet, make_env

    env = make_env(args.env_name)   # initialise the environment

    # set the random seeds for everything
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    #####################################

    # book keeping to log stuff
    if args.dryrun:
        writer = None
    else:
        # check internet connection
        # for offline wandb. Will load everything on cloud afterwards
        if not connected_to_internet():
            import json
            # save a json file with your wandb api key in your home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on MIT Supercloud
            with open(os.path.expanduser('~')+'/keys.json') as json_file: 
                key = json.load(json_file)
                my_wandb_api_key = key['my_wandb_api_key'] # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key # my Wandb api key
            os.environ["WANDB_MODE"] = "dryrun"

        experiment_name = f"{args.exp_name}_{args.env_name}_{args.seed}"
        args.env_name = f"{args.exp_name}_{args.env_name}"  # changing baselines so that grouping becomes easier
            
        print('_'*50)
        print('Creating wandboard...')
        print('_'*50)
        wandb_save_dir = os.path.join(os.path.abspath(os.getcwd()),f"wandb_{args.env_name}")
        if not os.path.exists(wandb_save_dir):
            os.makedirs(wandb_save_dir)
        wandb.init(project='Residual Policy Learning', entity='6-881_project',\
                   sync_tensorboard=True, config=vars(args), name=experiment_name,\
                   save_code=True, dir=wandb_save_dir, group=f"{args.env_name}")
        writer = SummaryWriter(f"{wandb.run.dir}/{experiment_name}")
    # initialise the agent
    trainer = Agent(args, env, writer=writer)
    trainer.evaluate_controller()

    # close env and writer
    env.close()
    if writer:
        writer.close()