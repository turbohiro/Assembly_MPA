from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
#from rlkit.torch.sac.image_network import CNN_output,_get_convolution_net,_get_linear_net
import torch
from torch import optim, nn
from torchvision import models, transforms
import torchvision
from rlkit.torch.sac.vanilla_vae import VanillaVAE
import yaml
from rlkit.torch.sac.experiment import VAEXperiment
from rlkit.torch.sac.network_utils import DenseBlock,PreNorm,FeedForward,Attention
from perceiver_pytorch import Perceiver

 

class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.flag = 1
        self.img_model = None
        self.model_name = 'shuttleV2'


        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.perceiver = Perceiver(
                input_channels = 1,          # number of channels for each token of the input
                input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
                num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
                max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
                depth = 2,                   # depth of net. The shape of the final attention mechanism will be:
                                            #   depth * (cross attention -> self_per_cross_attn * self attention)
                num_latents = 128,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = 256,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8
                cross_dim_head = 16,         # number of dimensions per cross attention head
                latent_dim_head = 16,        # number of dimensions per latent self attention head
                num_classes = 64,          # output number of classes
                attn_dropout = 0.,
                ff_dropout = 0.,
                weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
                self_per_cross_attn = 2      # number of self attention blocks per cross attention
            )
        self.proprio_preprocessing = DenseBlock(in_features=44,out_features = 64,norm = None, activation='relu').to('cuda')
        self.eef_preprocessing = torch.nn.Embedding(num_embeddings=4,embedding_dim=1).to('cuda')
    
        self.proprio_indices = torch.LongTensor(np.arange(0,34)).cuda()
        self.eef_indices = torch.LongTensor(np.arange(34,44)).cuda()

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        if self.flag ==0:
                self.img_model= torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x0_5', pretrained=True)         
                self.img_model.conv1=nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                self.img_model.state_dict()['conv1.weight'] = self.img_model.state_dict()['conv1.weight'].sum(dim=1, keepdim=True)
                in_ftr  = self.img_model.fc.in_features
                out_ftr = 256
                self.img_model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
                self.img_model.eval()
        else:
            with open('/homeL/wchen/Assembly_RPL/rlkit/rlkit/torch/sac/vae.yaml', 'r') as file:
                    try:
                        config = yaml.safe_load(file)
                    except yaml.YAMLError as exc:
                        print(exc)
            vae = VanillaVAE(**config['model_params'])
            PATH = '/homeL/wchen/Assembly_RPL/rlkit/rlkit/torch/sac/VanillaVAE/circle_robot/checkpoints/last.ckpt'
            checkpoint = torch.load(PATH)       
            experiment = VAEXperiment(vae,config['exp_params'])
            self.img_model = experiment.load_from_checkpoint(PATH,vae_model = vae, params=config['exp_params'])
            self.img_model.eval()
        
        batch_size,length = obs.shape
        image_batch = obs[:,:64*64*1].reshape((batch_size,1,64,64))
        if torch.cuda.is_available():
            image_batch = image_batch.to('cuda')
            self.img_model.to('cuda')
        proprio_batch = obs[:,64*64*1:]
        with torch.no_grad():
            if self.flag ==0:
                mu = self.img_model(image_batch)
            else:
                result,input,mu,std,features = self.img_model(image_batch)
        obs = torch.cat((mu,proprio_batch),dim=1)
        next_batch_size,length = next_obs.shape
        next_image_batch = next_obs[:,:64*64*1].reshape((next_batch_size,1,64,64))
        next_proprio_batch = next_obs[:,64*64*1:]
        if torch.cuda.is_available():
            next_image_batch = next_image_batch.to('cuda')
        with torch.no_grad():
            if self.flag ==0:
                next_mu = self.img_model(next_image_batch)
            else:
                result,input,next_mu,next_std,next_features = self.img_model(next_image_batch)
        next_obs = torch.cat((next_mu,next_proprio_batch),dim=1)

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )

