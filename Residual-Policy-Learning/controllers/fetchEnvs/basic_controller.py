"""
    Testing controllers for:
    1: pick and place: 'FetchPickAndPlace-v1'
    2: slide: 'FetchSlide-v1'
"""
import pdb
import gym
import numpy as np
import time
from tqdm import tqdm
from typing import Dict
from gym.utils import seeding
"""
    Observation Dictionary:
    -----------------------
    obs = {'observation':   array([ 1.02446872e+00,  7.79774721e-01,  4.44128144e-01,  9.20302186e-01,
                                    8.43710702e-01,  4.14022466e-01, -1.04166531e-01,  6.39359808e-02,
                                    -3.01056786e-02,  0.00000000e+00,  0.00000000e+00,  5.42335951e-03,
                                    -6.13467855e-04, -2.42502331e-02, -2.51540086e-02, -2.66176212e-02,
                                    -2.62985093e-02, -1.87473696e-02, -5.45233617e-04, -7.21666643e-03,
                                    2.51113042e-02,  2.66512371e-02,  2.63722934e-02,  3.97559364e-04,
                                    6.89072952e-04]),
    achieved_goal: array([0.92030219, 0.8437107 , 0.41402247]),
    desired_goal : array([1.41340195, 0.69852914, 0.41401894])}

    observation   = [grip_pos, object_pos, object_rel_pos, gripper_state, object_rot,
                     object_velp, object_velr, grip_velp, gripper_vel]
    achieved_goal = if object in the environment, then this is object_pos;
                        else this is just gripper position
    desired_goal  = final desired position of the object
    action = [gripper_position, gripper_state] # 4 dimensional vector within the range [-1,1]
    done = True after 50 timesteps which is the time limit set by gym, can change it in gym.envs.__init__.py
"""

def pick_place_controller(obs:Dict, object_in_hand:bool):
    """
        Pick and Place controller
        Will follow this trajectory:
            1: Move above the puck
            2: Move down while opening the gripper
            3: Close the gripper
            4: Pick the object and move towards the goal
        parameters:
        -----------
        obs: Dict
            Observation dictionary from the environment
        object_in_hand: bool
            Whether object has been picked up
    """
    dist_threshold = 0.002
    height_threshold = 0.003
    kp = 2
    action_pos =  list(kp * obs['observation'][6:9])  # vector joining gripper and object 
    print(obs['observation'][6:9])

    if not object_in_hand:
        # try to hover above the object
        if np.linalg.norm(action_pos[:2])>dist_threshold:
            action = action_pos[:2] + [0,0]     ##single-step action
        # once above the object, move down while opening the gripper
        else:
            print('the gripper is above the puck and the gripper is opening!!')
            action = action_pos[:3] + [1]   # open the gripper
            # if we are close to the object close the gripper
            if np.linalg.norm(action_pos) < height_threshold:
                print('the gripper is moving down and the gripper is closing!!')
                action = action_pos[:3] + [-1]  # close the gripper
                object_in_hand = True
    # once object is in hand, move towards goal
    else:
        print('the gripper is moving the object to the goal position!!')
        p_rel = obs['desired_goal'] - obs['achieved_goal']
        action_pos =  list(kp * p_rel)
        action = action_pos + [-1]  ###single-step action
    return action, object_in_hand

def imperfect_slide_controller(obs:Dict, hand_higher:bool, hand_above:bool, hand_behind:bool, hand_down:bool):
    """
        Imperfect slider controller
        Will follow this trajectory:
            1: Move vertically a little higher than the table
            2: Move (hover)above the puck
            3: Move behind the puck such that the puck, goal and hand are collinear
            4: Move down so that now gripper almost touches the table
            5: Give an impulse in the direction of the goal
        parameters:
        -----------
        obs: Dict
            Observation dictionary from the environment
        hand_higher: bool
            Whether hand is higher than the puck height
        hand_above: bool
            Whether hand is hovering above the puck
        hand_behind: bool
            Whether hand is behind the puck
        hand_down: bool
            Whether the hand is touching the table
        All these bools are suppposed to be used sequentially. 
        That is, once something becomes True, we do not care about it
    """
    kp = 1
    hit = 5

    # move the hand a little higher than the puck
    if not hand_higher:
        action = [0,0,1,0]
        # check if gripper is atleast `height_threshold` distance above
        if obs['observation'][2]-obs['observation'][5]>0.05:
            hand_higher = True
            print('Hand Higher')

    # once it is above, move it above the puck
    if hand_higher and not hand_above:
        action_pos = list(kp * obs['observation'][6:9])
        action = action_pos[:2] + [0,0]
        if np.linalg.norm(action_pos[:2])<0.01:
            hand_above = True
            print('Hand Above')

    # once hand is above the puck, move it behind the puck in the direction of the goal
    if hand_above and not hand_behind:
        goal_object_vec = obs['observation'][3:6] - obs['desired_goal'] # vector pointing towards object from goal
        action_pos = list(kp*goal_object_vec)
        action = action_pos + [0]
        if np.linalg.norm(obs['observation'][:2]-obs['observation'][3:5]) > 0.05:
        # if np.linalg.norm(obs['observation'][:3]-obs['observation'][3:6]) > 0.001:
            hand_behind = True
            print('Hand Behind')
    
    # now move the hand down
    if hand_behind and not hand_down:
        action = [0,0,-1,0]
        if obs['observation'][2]-obs['observation'][5] < 0.005:
            hand_down = True
            print('Hand Down')

    # now give impulse
    if hand_down:
        action_pos = list(-hit*(obs['observation'][:2] - obs['desired_goal'][:2]))
        action = action_pos + [0,0]

    return action, hand_higher, hand_above, hand_behind, hand_down

class FetchPickAndPlace(gym.Env):
    """
        FetchSlide:
    """
    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make('FetchPickAndPlace-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
    
    def step(self, action):
        observation, reward, done, info = self.fetch_env.step(action)
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
        self.fetch_env.env.goal = np.array([1,1, 0.41401894])   # change goal location HERE; z=0.414 because it is table surface
        return observation

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.fetch_env.seed(seed=seed)

    def render(self, mode="human", *args, **kwargs):
        return self.fetch_env.render(mode, *args, **kwargs)
    
    def close(self):
        return self.fetch_env.close()
    
    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

# time.sleep(5)

if __name__ == "__main__":
    #env_name = 'FetchPickAndPlace-v1'
    #env_name = 'FetchSlide-v1'
    env_name = 'FetchPush-v1'
    env = gym.make(env_name)
    #env = FetchPickAndPlace()
    # env = gym.wrappers.Monitor(env,'video/'+env_name, force=True) # can be used to record videos but this does not work properly
    obs = env.reset()

    
    hand_above = False
    hand_behind = False
    hand_higher = False
    hand_down = False
    object_in_hand = False
    action = [0,0,0,0]   # give zero action at first time step
    pdb.set_trace()
    # time.sleep(5)
    for i in tqdm(range(1000)):
        obs, rew, done, info = env.step(action)
        if env_name == 'FetchPickAndPlace-v1':
            action, object_in_hand = pick_place_controller(obs, object_in_hand)
        elif env_name == 'FetchSlide-v1':
            action, hand_higher, hand_above, hand_behind, hand_down = imperfect_slide_controller(obs, hand_higher, hand_above, hand_behind, hand_down)
        #env.sim.render(mode='window')
        env.render(mode='human')
        time.sleep(0.1)
        if info['is_success']:
             break

    print('Done')
