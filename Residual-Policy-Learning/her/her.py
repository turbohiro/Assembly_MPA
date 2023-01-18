"""
    Sampler for HER
"""
import numpy as np
from typing import Dict

class her_sampler:
    def __init__(self, replay_strategy:str, replay_k:int, reward_func=None):
        """
            Parameters:
            -----------
            replay_strategy: str
                HER strategy
            replay_k: int
                fraction to replace
            reward_func: env.compute_reward()
                function to compute rewards
        """
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch:Dict, batch_size:int):
        """
            Sample HER transitions
            Parameters:
            -----------
            episode_batch: a buffer storing [s_t||g, at, rt, s_{t+1}||g]
                {  'obs'       : np.empty([buffer_size, max_steps + 1, obs['observation'].shape[0]),
                    'ag'        : np.empty([buffer_size, max_steps + 1, obs['desired_goal'].shape[0],
                    'g'         : np.empty([buffer_size, max_steps,     obs['desired_goal'].shape[0],
                    'actions'   : np.empty([buffer_size, max_steps,     env.action_space.shape[0]),
                    'obs_next'  : np.empty([buffer_size, max_steps,     obs['observation'].shape[0]),
                    'ag_next'   : np.empty([buffer_size, max_steps,     obs['desired_goal'].shape[0]
                    }
            batch_size: int
                batch_size for buffer

        """
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions