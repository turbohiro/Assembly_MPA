"""
    Config for all experiments related to SAC (dense reward settings)
"""
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='SAC Agent')
parser.add_argument('--env_name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment')
parser.add_argument('--exp_name', type=str, default='rl', choices=['res', 'rl', 'baseline'],
                    help='Type of experiment: either learn with rl from scratch or learn residues')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')


parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=66, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=int(1e7), metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')

parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--eval_freq', type=int, default=10,
                    help='Number of episodes after which we evaluate the policy')
parser.add_argument('--num_eval_episodes', type=int, default=10,
                    help='Number of episodes to evaluate the agent')
parser.add_argument('--buffer_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')

parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--dryrun', type=bool, default=False,
                    help='Whether to use wandb writer or not')
parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True,
                    help='if toggled, `torch.backends.cudnn.deterministic=False`')

parser.add_argument('--beta', type=float, default=1.0,
                    help='Burn-in parameter for training residues')
# check https://github.com/k-r-allen/residual-policy-learning/issues/14#issue-747756960
# turns out that the original authors flipped the actor and critic loss monitoring values
# so instead of critic-loss being less than beta it is actor-loss less than beta
# so the original author(@tomsilver) has suggested us to try out both
parser.add_argument('--beta_monitor', type=str, default='critic', choices=['actor', 'critic'],
                    help='Which loss to check for burn-in parameter tuning')
args = parser.parse_args()