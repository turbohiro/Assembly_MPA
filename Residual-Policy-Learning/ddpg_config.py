"""
    Config for all experiments related to DDPG+HER (sparse reward settings)
"""
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='DDPG agent')
parser.add_argument('--exp_name', type=str, default='rl', choices=['res', 'rl', 'baseline'],
                    help='Type of experiment: either learn with rl from scratch or learn residues')
parser.add_argument('--env_name', type=str, default="FetchReach-v1",
                    help='the id of the gym environment')

# neural network parameters
parser.add_argument('--hidden_dims', type=list, default=[256,256,256],
                    help='Hidden layer dimensions')
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid'],
                    help='Activation function for networks')

# optmizer parameters
parser.add_argument('--actor_lr', type=float, default=1e-3,
                    help='the learning rate of the actor optimizer')
parser.add_argument('--critic_lr', type=float, default=1e-3,
                    help='the learning rate of the critic optimizer')
parser.add_argument('--actor_optim', type=str, default='adam', choices=['adam', 'adamax', 'rmsprop'],
                    help='Optimizer for actor network')
parser.add_argument('--critic_optim', type=str, default='adam', choices=['adam', 'adamax', 'rmsprop'],
                    help='Optimizer for critic network')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay for optimizers (actor and critic)')
parser.add_argument('--polyak', type=float, default=0.95,
                    help='Polyak averaging coefficient')

# experiment parameters (the ones given in the list in the paper in Appendix)
parser.add_argument('--n_test_rollouts', type=int, default=50,
                    help='Number of test rollouts per epoch')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='Number of epochs to train')
parser.add_argument('--n_cycles', type=int, default=50,
                    help='Number of cycles per epoch')
parser.add_argument('--n_batches', type=int, default=40,
                    help='Number of batches per cycle')
parser.add_argument('--num_rollouts_per_mpi', type=int, default=2, 
                    help='the rollouts per mpi')
parser.add_argument('--beta', type=float, default=1.0,
                    help='Burn-in parameter for training residues')
# check https://github.com/k-r-allen/residual-policy-learning/issues/14#issue-747756960
# turns out that the original authors flipped the actor and critic loss monitoring values
# so instead of critic-loss being less than beta it is actor-loss less than beta
# so the original author(@tomsilver) has suggested us to try out both
parser.add_argument('--beta_monitor', type=str, default='critic', choices=['actor', 'critic'],
                    help='Which loss to check for burn-in parameter tuning')

# random noise
parser.add_argument('--noise_eps', type=float, default=0.2,
                    help='Scale of additive Gaussian Noise')
parser.add_argument('--random_eps', type=float, default=0.3,
                    help='Probability of taking random actions')
parser.add_argument('--coin_flipping_prob', type=float, default=0.5,
                    help='Probability for coin flipping')

# clipping stuff
parser.add_argument('--clip_obs', type=float, default=200,
                    help='Observation clipping')
parser.add_argument('--action_l2', type=float, default=1.0,
                    help='Action L2 norm coefficient to be added in actor loss')
parser.add_argument('--clip_range', type=float, default=5,
                    help='Clip range for MPI normalizer')


# seed related stuff
parser.add_argument('--seed', type=int, default=66,
                    help='seed of the experiment')
parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True,
                    help='if toggled, `torch.backends.cudnn.deterministic=False`')

# learning related general params
parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'smooth_l1', 'l1'],
                    help='Loss function for DDPG loss')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch Size for training')


# HER/replay buffer
parser.add_argument('--replay_strategy', type=str, default='future',
                    help='the HER strategy')
parser.add_argument('--replay_k', type=int, default=4,
                    help='ratio of buffer to  replace')
parser.add_argument('--buffer_size', type=int, default=int(1e6),
                    help='the replay memory buffer size')


# general RL specific arguments
parser.add_argument('--gamma', type=float, default=0.98,
                    help='the discount factor gamma')
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='the maximum norm for the gradient clipping')

parser.add_argument('--dryrun', type=bool, default=False,
                    help='Whether to use wandb writer or not')
                    
args = parser.parse_args()