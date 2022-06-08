"""
    Implementation of Residual Policy Learning with DDPG+HER
    DDPG with HER Code inspired from https://github.com/TianhongDai/hindsight-experience-replay
"""
import copy
import gym
import argparse
import torch
from torch import nn
from torch import optim
import os
from datetime import datetime
import numpy as np
from typing import Tuple, List
from mpi4py import MPI

from RL.ddpg.models import actor, critic
from RL.ddpg.replay_buffer_mpi import replay_buffer
from her.her import her_sampler
from mpi.mpi_utils import sync_grads, sync_networks
from mpi.normalizer import normalizer

OPTIMIZERS = {
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
}

LOSS_FN = {
    'mse': nn.MSELoss(),
    'smooth_l1': nn.SmoothL1Loss(),
    'l1': nn.L1Loss()
}

class DDPG_Agent:
    def __init__(self, args:argparse.Namespace, env, save_dir: str, device:str, writer=None):
        """
            Module for the DDPG agent along with HER
            Parameters:
            -----------
            args: argparse.Namespace
                args should contain the following:
            env: gym.Env
                OpenAI type gym environment
            save_dir: str
                Path to save the network weights, checkpoints
            device: str
                device to run the training process on
                Choices: 'cpu', 'cuda'
            writer: tensorboardX
                tensorboardX to log metrics like losses, rewards, success_rates, etc.
        """
        self.args = args
        self.env = env
        self.env_params = self.get_env_params(env)
        self.save_dir = save_dir
        self.device = device
        self.writer = writer
        self.sim_steps = 0      # a count to keep track of number of simulation steps
        self.train_steps = 0    # a count to keep track of number of training steps

        # create the network
        self.actor_network = actor(args, self.env_params).to(device)
        self.critic_network = critic(args, self.env_params).to(device)

        # sync the networks across the cpu cores
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(args, self.env_params).to(device)
        self.critic_target_network = critic(args, self.env_params).to(device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim  = OPTIMIZERS[args.actor_optim](self.actor_network.parameters(),\
                                                         lr=self.args.actor_lr,\
                                                         weight_decay=self.args.weight_decay)
        self.critic_optim = OPTIMIZERS[args.critic_optim](self.critic_network.parameters(),
                                                         lr=self.args.critic_lr,\
                                                         weight_decay=self.args.weight_decay)
        self.burn_in_done = False   # to check if burn-in is done or not

        # loss function for DDPG
        self.criterion = LOSS_FN[args.loss_fn]

        # her sampler
        self.her_module = her_sampler(replay_strategy = self.args.replay_strategy, \
                                      replay_k = self.args.replay_k, \
                                      reward_func = self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, \
                                    self.args.buffer_size, \
                                    self.her_module.sample_her_transitions)
        
        # make the observation normalizer
        self.o_norm = normalizer(size=self.env_params['obs'],  default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)
    
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
        
    def set_actor_lr(self, loss:float, prev_loss:float ,verbose:bool=True):
        """
            Set the learning rate of the actor network
            to either the original learning rate
            or zero depending on the burn-in parameter
            diff_loss = |loss - prev_loss|
            if rl:
                actor_lr = original_actor_lr
            if residual learning and diff_loss < beta:
                actor_lr = original_actor_lr
            elif residual learning and diff_loss > beta:
                actor_lr = 0
            Parameters:
            -----------
            loss: float
                Mean of the losses till the current epoch
            prev_loss: float
                Mean of losses till the last epoch
            verbose: bool
                To print the change in learning rate
        """
        lr = self.args.actor_lr
        coin_flipping = False
        # loss is zero only in the first epoch hence do not change lr then
        # and that's why give a large value to diff_loss
        diff_loss = 100 if loss == 0 else abs(loss - prev_loss)
        if not self.burn_in_done:
            if self.args.exp_name == 'res' and diff_loss > self.args.beta:
                lr = 0.0
                coin_flipping = True
            elif self.args.exp_name == 'res' and diff_loss <= self.args.beta:
                if verbose:
                    print('_'*80)
                    print(f'Burn-in of the critic done. Changing actor_lr from 0.0 to {self.args.actor_lr}')
                    print('_'*80)
                self.burn_in_done = True

        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr
        return coin_flipping

    def train(self):
        """
            Run the episodes for training
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('_'*50)
            print('Beginning the training...')
            print('_'*50)
        num_cycles = 0
        actor_losses = [0.0]    # to store actor losses for burn-in
        critic_losses = [0.0]   # to store critic losses for burn-in
        prev_losses = [0.0]
        coin_flipping = False   # choose whether we want deterministic or not
        deterministic = False   # whether the whole episode should be noise and randomness free
        for epoch in range(self.args.n_epochs):
            
            # change the actor learning rate from zero to actor_lr by checking burn-in
            # check config.py for more information on args.beta_monitor
            if self.args.beta_monitor == 'actor':
                coin_flipping = self.set_actor_lr(np.mean(actor_losses), np.mean(prev_losses))
                prev_losses = actor_losses.copy()
            elif self.args.beta_monitor == 'critic':
                coin_flipping = self.set_actor_lr(np.mean(critic_losses), np.mean(prev_losses))
                prev_losses = critic_losses.copy()

            for cycle in range(self.args.n_cycles):
                mpi_obs, mpi_ag, mpi_g, mpi_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    observation_new = copy.deepcopy(observation)
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']

                    random_eps = self.args.random_eps
                    noise_eps  = self.args.noise_eps
                    if coin_flipping:
                        deterministic = np.random.random() < self.args.coin_flipping_prob  # NOTE/TODO change here
                    if deterministic:
                        random_eps = 0.0
                        noise_eps = 0.0
                    # collect the sample episode wise
                    for t in range(self.env_params['max_timesteps']):
                        # take actions
                        with torch.no_grad():
                            state = self.preprocess_inputs(obs, g)
                            pi = self.actor_network(state)
                            if self.args.exp_name == 'res':
                                controller_action = self.get_controller_actions(observation_new)
                                action = self.select_actions(pi, noise_eps=noise_eps, random_eps= random_eps, controller_action=controller_action)
                            else:
                                action = self.select_actions(pi, noise_eps=noise_eps, random_eps= random_eps, controller_action=None)
                        # give the action to the environment
                        observation_new, _, _, info = self.env.step(action)
                        self.sim_steps += 1                     # increase the simulation timestep by one
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']

                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    # append last states in the array
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())

                    # append everything to the mpi array
                    mpi_obs.append(ep_obs)
                    mpi_ag.append(ep_ag)
                    mpi_g.append(ep_g)
                    mpi_actions.append(ep_actions)
                # convert to np arrays
                mpi_obs     = np.array(mpi_obs)
                mpi_ag      = np.array(mpi_ag)
                mpi_g       = np.array(mpi_g)
                mpi_actions = np.array(mpi_actions)

                # store them in buffer
                self.buffer.store_episode([mpi_obs, mpi_ag, mpi_g, mpi_actions])
                self.update_normalizer([mpi_obs, mpi_ag, mpi_g, mpi_actions])

                actor_loss_cycle = 0; critic_loss_cycle = 0 
                for batch in range(self.args.n_batches):
                    # train the network with 'n_batches' number of batches
                    actor_loss, critic_loss = self.update_network(self.train_steps)
                    self.train_steps += 1
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    actor_loss_cycle += actor_loss
                    critic_loss_cycle += critic_loss
                if self.writer:
                    self.writer.add_scalar('Cycle Losses/Actor Loss', actor_loss_cycle, num_cycles)
                    self.writer.add_scalar('Cycle Losses/Critic Loss', critic_loss_cycle, num_cycles)
                self.polyak_update_networks(self.actor_target_network, self.actor_network)
                self.polyak_update_networks(self.critic_target_network, self.critic_network)
                num_cycles += 1
            # evaluate the agent
            success_rate = self.eval_agent()
            # print only once instead of the huge monstrosity!!!
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f'Epoch Critic: {np.mean(critic_losses):.3f} Epoch Actor:{np.mean(actor_losses):.3f}')
                print(f'Epoch:{epoch}\tSuccess Rate:{success_rate:.3f}')
                if self.writer:
                    self.writer.add_scalar('Success Rate/Success Rate', success_rate, self.sim_steps)
                    self.writer.add_scalar('Epoch Losses/Average Critic Loss', np.mean(critic_losses), epoch)
                    self.writer.add_scalar('Epoch Losses/Average Actor Loss', np.mean(actor_losses), epoch)
                # save checkpoints
                self.save_checkpoint(self.save_dir)

    def preprocess_inputs(self, obs:np.ndarray, g:np.ndarray) -> torch.Tensor:
        """
            Concatenate state and goal
            and convert them to torch tensors
            and then transfer them to either CPU of GPU
        """
        obs = self.o_norm.normalize(obs)
        g = self.g_norm.normalize(g)
        # concatenate the obs and goal
        inputs = np.concatenate([obs, g])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        inputs = inputs.to(self.device)
        return inputs
    
    def get_controller_actions(self, obs:dict):
        """
            Return the controller action if residual learning
        """
        return self.env.controller_action(obs, take_action=False)
    
    def select_actions(self, pi: torch.Tensor, noise_eps:float, random_eps:float, controller_action=None) -> np.ndarray:
        """
            Take action
            with a probability of self.args.random_eps, it will take random actions
            otherwise, this will add a gaussian noise to the action along with clipping
            pi: torch.Tensor
                The action given by the actor
            noise_eps: float
                The random gaussian noise added to actor output
            random_eps: float
                Probability of taking random action
            controller_action: None or np.ndarray
                To subtract from random action
        """
        # transfer action from CUDA to CPU if using GPU and make numpy array out of it
        action = pi.cpu().numpy().squeeze()

        # add the gaussian
        action += noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])

        # random actions
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # if residual learning, subtract the conroller action so that we don't add it twice
        if self.args.exp_name == 'res':
            random_actions = random_actions - controller_action
        # choose whether to take random actions or not
        rand = np.random.binomial(1, random_eps, 1)[0]
        action += rand * (random_actions - action)  # will be equal to either random_actions or action
        return action

    def preprocess_og(self, o:np.ndarray, g:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Perform observation clipping
        """
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def polyak_update_networks(self, target, source):
        """
            Polyak averaging of target and main networks; Also known as soft update of networks
            target_net_params = (1 - polyak) * main_net_params + polyak * target_net_params
            target and source are torch networks
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
    
    def update_normalizer(self, episode_batch:List):
        """
            Update the normalizer for both obs and goal
        """
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self.preprocess_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def update_network(self, step:int) -> Tuple[float, float]:
        """
            The actual DDPG training
        """

        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self.preprocess_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self.preprocess_og(o_next, g)

        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm   = self.g_norm.normalize(transitions['g'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm   = self.g_norm.normalize(transitions['g_next'])

        # concatenate obs and goal
        states = np.concatenate([obs_norm, g_norm], axis=1)
        next_states = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        # convert to tensor
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            actions_next = self.actor_target_network(next_states)
            q_next_value = self.critic_target_network(next_states, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = rewards + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the returns
            clip_return = 1 / (1-self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        
        # critic loss
        q_value = self.critic_network(states, actions)
        critic_loss = self.criterion(target_q_value, q_value)   # loss (mostly MSE)

        # actor loss
        actions_pred = self.actor_network(states)
        actor_loss = -self.critic_network(states, actions_pred).mean()
        actor_loss = actor_loss + self.args.action_l2 * (actions_pred / self.env_params['action_max']).pow(2).mean()

        # backpropagate
        self.actor_optim.zero_grad()    # zero the gradients
        actor_loss.backward()           # backward prop
        sync_grads(self.actor_network)
        self.actor_optim.step()         # take step towards gradient direction

        self.critic_optim.zero_grad()    # zero the gradients
        critic_loss.backward()           # backward prop
        sync_grads(self.critic_network)
        self.critic_optim.step()         # take step towards gradient directions

        return actor_loss.item(), critic_loss.item()
        
    def eval_agent(self) -> float:
        """
            Evaluate the agent using the trained policy
            performs n_test_rollouts in the environment
            and returns
        """
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self.preprocess_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()

    def save_checkpoint(self, path:str):
        """
            Saves the model in the wandb experiment run directory
            This will store the 
                * model state_dict
                * optimizer state_dict
                * args/hparams
            param:
                path: str
                    path to the wandb run directory
                    Example: os.path.join(wandb.run.dir, "model.ckpt")
        """
        checkpoint = {}
        checkpoint['args'] = vars(self.args)
        checkpoint['actor_state_dict'] = self.actor_network.state_dict()
        # save the normalizer mean and std
        checkpoint['normalizer_feats'] = [self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std]
        checkpoint['env_name'] = self.args.env_name
        torch.save(checkpoint, path)

    def load_checkpoint(self, path:str):
        """
            Load the trained model weights
            param:
                path: str
                    path to the saved weights file
        """
        use_cuda = torch.cuda.is_available()
        device   = torch.device("cuda" if use_cuda else "cpu")
        checkpoint_dict = torch.load(path, map_location=device)
        self.actor_network.load_state_dict(checkpoint_dict['actor_state_dict'])

if __name__ == "__main__":
    # run from main folder as `python RL/ddpg.py`
    import time
    import wandb
    from torch.utils.tensorboard import SummaryWriter

    from ddpg_config import args
    from utils import connected_to_internet, make_env, get_pretty_env_name

    # set parameters for threading
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # check whether GPU is available or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('_'*20)
        print('Device:',device)
        print('_'*20)
    args.device = device
    args.mpi = True         # running on mpi mode
    #####################################

    env = make_env(args.env_name)   # initialise the environment
    # env = gym.make(args.env_name)   # initialise the environment

    # set the random seeds for everything
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    #####################################

    # book keeping to log stuff
    if args.dryrun:
        writer = None
        weight_save_path = 'model_dryrun.ckpt'
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

        start_time = time.strftime("%H_%M_%S-%d_%m_%Y", time.localtime())
        pretty_env_name = get_pretty_env_name(args.env_name)
        experiment_name = f"{args.exp_name}_{pretty_env_name}_{args.seed}_{start_time}"
        
        # create only one wandb logger instead of 8/16!!!
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('_'*50)
            print('Creating wandboard...')
            print('_'*50)
            wandb_save_dir = os.path.join(os.path.abspath(os.getcwd()),f"wandb_{pretty_env_name}")
            if not os.path.exists(wandb_save_dir):
                os.makedirs(wandb_save_dir)
            wandb.init(project='Residual Policy Learning', entity='turbohiro',\
                       sync_tensorboard=True, config=vars(args), name=experiment_name,\
                       save_code=True, dir=wandb_save_dir, group=f"{pretty_env_name}")
            writer = SummaryWriter(f"{wandb.run.dir}/{experiment_name}")
            weight_save_path = os.path.join(wandb.run.dir, "model.ckpt")
        else:
            writer = None
            weight_save_path = "model_mpi.ckpt"
    ##########################################################################
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('_'*50)
        print('Arguments:')
        for arg in vars(args):
            print(f'{arg} = {getattr(args, arg)}')
        print('_'*50)
    # initialise the agent
    trainer = DDPG_Agent(args, env, save_dir=weight_save_path, device=device, writer=writer)
    # train the agent
    trainer.train()

    # close env and writer
    env.close()
    if writer:
        writer.close()
