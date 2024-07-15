import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp
import torchvision
from rlkit.torch.sac.vanilla_vae import VanillaVAE
import yaml
from rlkit.torch.sac.experiment import VAEXperiment
from rlkit.torch.sac.network_utils import DenseBlock,PreNorm,FeedForward,Attention,Conv1DBlock
from perceiver_pytorch import Perceiver
import math
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=44):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        
        #encoder coss attention
        # self.cross_attend_blocks = nn.ModuleList([
        #     PreNorm(latent_dim, Attention(latent_dim, self.input_dim_before_seq, head = cross_heads, dim_head = cross_dim_head, dropout = input_dropout),
        #     contex_dim = self.input_dim_before_seq),
        #     PreNorm(latent_dim,FeedForward(latent_dim))
        # ])
        #encoder self attention
        self.layers = nn.ModuleList([])

        
        self.flag=1   #0 means make dataset ;1 means train the collected dataset
        self.conv_processing = Conv1DBlock(in_channels=44, out_channels=64, kernel_sizes=1, strides=1,norm = None, activation='relu').to('cuda')
        self.proprio_preprocessing3 = DenseBlock(in_features=44,out_features = 64,norm = None, activation='relu').to('cuda')
        self.image_preprocessing3 = DenseBlock(in_features=64,out_features = 64,norm = None, activation='relu').to('cuda')
        self.proprio_preprocessing = DenseBlock(in_features=34,out_features = 55,norm = None, activation='relu').to('cuda')
        self.proprio_preprocessing2 = DenseBlock(in_features=64,out_features = 128,norm = None, activation='relu').to('cuda')
        self.eef_preprocessing = torch.nn.Embedding(num_embeddings=4,embedding_dim=1).to('cuda')
        
        self.proprio_indices = torch.LongTensor(np.arange(64,108)).cuda()
        self.image_indices = torch.LongTensor(np.arange(0,64)).cuda()
        self.joint_indices = torch.LongTensor(np.arange(0,34)).cuda()
        self.eef_indices = torch.LongTensor(np.arange(34,44)).cuda()
        self.position_encoding = PositionalEncoding(64)

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



    def get_action(self, obs_np, deterministic=False):

        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        # if deterministic:
        #     import pdb
        #     pdb.set_trace()
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        #h = obs
        
    
        #image feature extraction network

        if obs.shape[1]== 4140:  #12331
            if self.flag ==0:
                model= torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x0_5', pretrained=True)         
                model.conv1=nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                model.state_dict()['conv1.weight'] = model.state_dict()['conv1.weight'].sum(dim=1, keepdim=True)
                in_ftr  = model.fc.in_features
                out_ftr = 256
                model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
                model.eval()
            else:
                with open('/homeL/wchen/Assembly_RPL/rlkit/rlkit/torch/sac/vae.yaml', 'r') as file:
                    try:
                        config = yaml.safe_load(file)
                    except yaml.YAMLError as exc:
                        print(exc)
                #config = yaml.safe_load('vae.yaml')
                vae = VanillaVAE(**config['model_params'])
                #self.model.build_layers()    
                PATH = '/homeL/wchen/Assembly_RPL/rlkit/rlkit/torch/sac/VanillaVAE/circle_robot/checkpoints/last.ckpt'
                checkpoint = torch.load(PATH)
            
                experiment = VAEXperiment(vae,config['exp_params'])
                model = experiment.load_from_checkpoint(PATH,vae_model = vae, params=config['exp_params'])
                model.eval()
            batch_size,length = obs.shape            
            image_batch = obs[:,:64*64*1].reshape((batch_size,1,64,64))

            if torch.cuda.is_available():
                image_batch = image_batch.to('cuda')
                model.to('cuda')
            proprio_batch = obs[:,64*64*1:]
            with torch.no_grad():
                if self.flag ==0:
                    mu = model(image_batch)
                else:
                    result,input,mu,std,features = model(image_batch)
                #features = torch.cat([mu, std],axis=-1)
            
            h = torch.cat((mu,proprio_batch),dim=1) #64+7*4+3*4+4'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_force', 'robot0_eef_torque', 'robot0_eef_velocity'

            # cros_attn, cross_ff = self.cross_attend_blocks
            # for it in range(self.iteration):
            #     #encoder cross attention
            #     x = cross_attn(x,context = ins, mask=None) + x
            #     x = cross_ff(x) + x

            #     #encoder self attention
            #     for self_attn,self_ff in self.layers:
            #         x = self_attn(x) + x
            #         x = self_ff(x) + x
        else:
            h=obs
            #print('enter into the training period')
        if self.flag ==1:
        ###feature fusing module with transformer
            proprio_batch2 = torch.index_select(h,1,self.proprio_indices)
            mu = torch.index_select(h,1,self.image_indices)
            proprio_features = self.proprio_preprocessing3(proprio_batch2) 
            h_final = torch.cat((mu,proprio_features),dim=1)  #
            encoder_layer = nn.TransformerEncoderLayer(d_model=h_final.shape[-1], nhead=4).to('cuda')
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
            out = transformer_encoder(torch.unsqueeze(h_final,dim = 0))
            h_final = torch.squeeze(out,dim = 0) 

        #single modaility--only pretrained vision information for policy learning
        #h_final = mu  

        #single modality-- proprioception and force/torque
        #h_final = proprio_features
        
        #
        # proprio_features = self.proprio_preprocessing(torch.index_select(proprio_batch2,1,self.joint_indices))
        # #pro_features = self.eef_preprocessing(torch.index_select(proprio_batch2,1,self.eef_indices).long())
        # pro_features = torch.index_select(proprio_batch2,1,self.eef_indices)
        # #pro_features = pro_features.squeeze(2)
        # h2 = torch.cat((proprio_features,pro_features),dim=1)
        # pro_features = self.proprio_preprocessing2(h2)
        
        # out = proprio_batch2.unsqueeze(dim=-1).permute(0, 2, 1)
        # pos = F.one_hot(torch.arange(0,43),43).expand(proprio_batch2.shape[0],-1,-1).to('cuda')
        # out = torch.cat((pos,out),dim=1)
        # proprio_features = self.conv_processing(out).permute(0,2,1)
        # mu = torch.unsqueeze(mu,dim = 1)  
        # h_final = torch.cat((mu,proprio_features),dim=1)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=h_final.shape[-1], nhead=4).to('cuda')
        # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # out = transformer_encoder(h_final)
        # h_final = nn.Flatten()(out)
        #
        if self.flag ==0:
            h_final = h

        for i, fc in enumerate(self.fcs):
            h_final = self.hidden_activation(fc(h_final))
        mean = self.last_fc(h_final)
        if self.std is None:
            log_std = self.last_fc_log_std(h_final)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        print('!!!!!!!!!!this is evaluation period!!')
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
