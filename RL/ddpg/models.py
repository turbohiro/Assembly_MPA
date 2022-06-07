"""
    Networks for actor and critic
    NOTE: Input should be [observation, goal]
"""
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpi4py import MPI

ACTS = {
    'relu': F.relu,
    'sigmoid':torch.sigmoid,
    'tanh':torch.tanh,
}

class actor(nn.Module):
    """
            Actor Network
            Parameters:
            ----------
            args: argparse.Namespace
                Should at least have the following arguments:
                hidden_dims: list
                    List of hidden layer dimensions
                activation: str
                    Which activation function to use
                    Choices: 'relu', 'sigmoid', 'tanh'
                exp_name : str
                    Whether it is learning residues or just learning from scratch
                    Choices: 'res', 'rl'
            env_params: Dict
                a dictionary containing the following:
                {
                    'obs': obs['observation'].shape[0],
                    'goal': obs['desired_goal'].shape[0],
                    'action': env.action_space.shape[0],
                    'action_max': env.action_space.high[0],
                    'max_timesteps': env._max_episode_steps #fetchGet 50
                }
        """
    def __init__(self, args:argparse.Namespace, env_params:Dict):
        super(actor, self).__init__()

        hidden_dims = args.hidden_dims
        hidden_dims = [env_params['obs'] + env_params['goal']] + hidden_dims
        self.activation = ACTS[args.activation]
        self.max_action = env_params['action_max']
        self.layers = nn.ModuleList()
        for i, j in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(nn.Linear(i,j))    # append linear layers
        self.mlp = nn.Sequential()
        self.action_out = nn.Linear(hidden_dims[-1], env_params['action'])
        # if learning residues, then make weights of last layer equal to zero
        if args.exp_name == 'res':
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('_'*50)
                print('Initialising actor final layer with zeros')
                print('_'*50)
            self.action_out.weight.data.fill_(0)
            self.action_out.bias.data.fill_(0)

    def forward(self, x:torch.Tensor):
        for layer in self.layers:
            x = self.activation(layer(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions   #actor: policy function---> predict the next action

class critic(nn.Module):
    def __init__(self, args:argparse.Namespace, env_params:Dict):
        """
            Critic Network
            Parameters:
            ----------
            args: argparse.Namespace
                Should at least have the following arguments:
                hidden_dims: list
                    List of hidden layer dimensions
                activation: str
                    Which activation function to use
                    Choices: 'relu', 'sigmoid', 'tanh'
            env_params: Dict
                a dictionary containing the following:
                {
                    'obs': obs['observation'].shape[0],
                    'goal': obs['desired_goal'].shape[0],
                    'action': env.action_space.shape[0],
                    'action_max': env.action_space.high[0],
                    'max_timesteps': env._max_episode_steps
                }
            NOTE: Normalisation of actions happens in self.forward()
        """
        super(critic, self).__init__()

        hidden_dims = args.hidden_dims
        hidden_dims = [env_params['obs'] + env_params['goal'] + env_params['action']] + hidden_dims
        self.activation = ACTS[args.activation]
        self.max_action = env_params['action_max']
        self.layers = nn.ModuleList()
        for i, j in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(nn.Linear(i,j))    # append linear layers
        self.q_out = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x:torch.Tensor, actions:torch.Tensor):

        x = torch.cat([x, actions / self.max_action], dim=1)
        for layer in self.layers:
            x = self.activation(layer(x))
        q_value = self.q_out(x)
        return q_value  #critic: Q (action-value) function ---> get the q_values  

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# """
# the input x in both networks should be [o, g], where o is the observation and g is the goal.
# """

# # define the actor network
# class actor(nn.Module):
#     def __init__(self, args, env_params):
#         super(actor, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.action_out = nn.Linear(256, env_params['action'])

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         actions = self.max_action * torch.tanh(self.action_out(x))

#         return actions

# class critic(nn.Module):
#     def __init__(self, args, env_params):
#         super(critic, self).__init__()
#         self.max_action = env_params['action_max']
#         self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.q_out = nn.Linear(256, 1)

#     def forward(self, x, actions):
#         x = torch.cat([x, actions / self.max_action], dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         q_value = self.q_out(x)

#         return q_value