"""
    The main file to run Soft Actor Critic 
"""
import os
import math
import numpy as np
import itertools
import argparse
import copy

import torch
import torch.nn.functional as F
from torch.optim import Adam

from sac_config import args
from RL.sac.replay_buffer import replay_buffer
from RL.sac.models import GaussianActor, DeterministicActor, Critic
import pdb


class SAC_Agent:
    def __init__(self, args:argparse.Namespace, env, save_dir:str, device:str, writer=None):
        """
            Module for the SAC agent
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
        self.writer = writer
        self.device = device
        self.env_params = self.get_env_params(env)
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.critic = Critic(self.env_params['obs'], self.env_params['action'], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = Critic(self.env_params['obs'], self.env_params['action'], args.hidden_size).to(self.device)
        self.hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.actor = GaussianActor(self.env_params['obs'], self.env_params['action'], args.hidden_size, self.env_params['action_space']).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicActor(self.env_params['obs'], self.env_params['action'], args.hidden_size, self.env_params['action_space']).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.burn_in_done = False   # to check if burn-in is done or not
        self.buffer = replay_buffer(self.args.buffer_size, self.args.seed)

    def get_env_params(self, env):
        """
            Get the environment parameters
        """
        obs = env.reset()
        # close the environment
        params = {'obs': obs.shape[0],
                'action': env.action_space.shape[0],
                'action_space': env.action_space,
                'action_max': env.action_space.high[0],
                }
        try:
            params['max_timesteps'] = env._max_episode_steps
        # for custom envs
        except:
            params['max_timesteps'] = env.max_episode_steps
        print('_'*50)
        print('Environment Parameters:')
        for key in params:
            print(f'{key}: {params[key]}')
        print('_'*50)
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
        lr = self.args.lr
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
                    print(f'Burn-in of the critic done. Changing actor_lr from 0.0 to {self.args.lr}')
                    print('_'*80)
                self.burn_in_done = True

        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr
        return coin_flipping

    def train(self):
        print('_'*50)
        print('Beginning Training')
        print('_'*50)
        # Training Loop
        total_numsteps = 0
        updates = 0

        actor_losses = [0.0]    # to store actor losses for burn-in
        critic_losses = [0.0]   # to store critic losses for burn-in
        prev_losses = [0.0]
        coin_flipping = False   # whether the whole episode should be noise and randomness free
        deterministic = False   # choose whether we want deterministic or not 

        for i_episode in itertools.count(1):

             # change the actor learning rate from zero to actor_lr by checking burn-in
            # check config.py for more information on args.beta_monitor
            if self.args.beta_monitor == 'actor':
                coin_flipping = self.set_actor_lr(np.mean(actor_losses), np.mean(prev_losses))
                prev_losses = actor_losses.copy()
            elif self.args.beta_monitor == 'critic':
                coin_flipping = self.set_actor_lr(np.mean(critic_losses), np.mean(prev_losses))
                prev_losses = critic_losses.copy()

            episode_reward = 0
            episode_steps = 0
            done = False
            state = self.env.reset()
            next_state = copy.deepcopy(state)
            random_eps = self.args.random_eps
            noise_eps  = self.args.noise_eps
            #if coin_flipping:
            #        deterministic = np.random.random() < self.args.coin_flipping_prob  # NOTE/TODO change here
            if coin_flipping:
                random_eps = 0.0
                noise_eps = 0.0

            while not done:                
                if self.args.start_steps > total_numsteps:
                    pdb.set_trace()                    
                    if self.args.exp_name == 'res':
                        controller_action = self.get_controller_actions(state)   
                        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                        pi, _, _ = self.actor.sample(state)                    
                        action = self.select_action(pi, noise_eps=noise_eps, random_eps= random_eps, controller_action=controller_action)
                    else:
                        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                        pi, _, _ = self.actor.sample(state)
                        action = self.select_action(pi, noise_eps=noise_eps, random_eps= random_eps, controller_action=None)

                if len(self.buffer) > self.args.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.args.updates_per_step):
                        critic_1_loss, critic_2_loss, actor_loss, ent_loss, alpha = self.update_parameters(self.buffer, self.args.batch_size, updates)
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_1_loss)

                        if self.writer:
                            self.writer.add_scalar('Loss/Critic_1', critic_1_loss, updates)
                            self.writer.add_scalar('Loss/Critic_2', critic_2_loss, updates)
                            self.writer.add_scalar('Loss/Actor', actor_loss, updates)
                            self.writer.add_scalar('Loss/Entropy Loss', ent_loss, updates)
                            self.writer.add_scalar('Entropy Temprature/Alpha', alpha, updates)
                        updates += 1
                # TODO add info success rate metric
                next_state, reward, done, info = self.env.step(action)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # ignore the done signal if we hit time horizon
                # Refer: # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == self.env_params['max_timesteps'] else float(not done)

                self.buffer.push(state, action, reward, next_state, mask) # Append transition to memory
                state = next_state

            if total_numsteps > self.args.num_steps:
                break

            if self.writer:
                self.writer.add_scalar('Reward/train', episode_reward, i_episode)
            print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {episode_reward:.3f}")

            if i_episode % self.args.eval_freq == 0:
                avg_reward = 0
                avg_success = 0
                for _ in range(self.args.num_eval_episodes):
                    state = self.env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                        pi, _, _ = self.actor.sample(state)  
                        action = pi.detach().cpu().numpy().squeeze()
                        next_state, reward, done, info = self.env.step(action)
                        episode_reward += reward

                        state = next_state
                    avg_success += info['is_success']
                    avg_reward += episode_reward
                avg_reward /= self.args.num_eval_episodes
                avg_success /= self.args.num_eval_episodes

                if self.writer:
                    self.writer.add_scalar('Reward/Test', avg_reward, i_episode)
                    self.writer.add_scalar('Success Rate/Success Rate', avg_success, i_episode)

                print("_"*50)
                print(f"Reward: {avg_reward:.2}, Success Rate: {avg_success:.3}")
                print("_"*50)


    def get_controller_actions(self, obs:dict):
        """
            Return the controller action if residual learning
        """
        return self.env.controller_action(obs, take_action=False)   ###!!!!!!!!!!!!!!!!!True/False

    def select_action(self, pi, noise_eps, random_eps, controller_action=None):
        # transfer action from CUDA to CPU if using GPU and make numpy array out of it
        action = pi.cpu().numpy().squeeze()
        # random actions
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # if residual learning, subtract the controller action so that we don't add it twice
        if self.args.exp_name == 'res':
            random_actions = random_actions - controller_action
        # choose whether to take random actions or not
        rand = np.random.binomial(1, random_eps, 1)[0]
        action += rand * (random_actions - action)  # will be equal to either random_actions or actio
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size:int, updates:int):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

if __name__ == "__main__":
    # run from main folder as `python RL/ddpg.py`
    import time
    import wandb
    from torch.utils.tensorboard import SummaryWriter

    from sac_config import args
    from utils import connected_to_internet, make_env, get_pretty_env_name

    # check whether GPU is available or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('_'*50)
    print('Device:',device)
    print('_'*50)
    args.device = device
    #####################################

    env = make_env(args.env_name)   # initialise the environment

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
    ##########################################################################
    print('_'*50)
    print('Arguments:')
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    print('_'*50)
    # initialise the agent
    trainer = SAC_Agent(args, env, save_dir=weight_save_path, device=device, writer=writer)
    # train the agent
    trainer.train()

    # close env and writer
    env.close()
    if writer:
        writer.close()