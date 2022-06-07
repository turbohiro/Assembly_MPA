"""
    Collection of variations in the slide environment
    Contains the following:
    FetchSlide:
        The vanilla 'FetchSlide-v1' without any controller
        Can be used for learning with RL from scratch
        Action taken as:
            Pi_theta(s) = f(s)
    FetchSlideImperfectControl:
        'FetchSlide-v1' with an imperfect controller
        This controller can at least slide the puck but not perfectly
        As in does push it in some direction
        Action taken as:
            Pi_theta(s) = f(s) + pi_theta(s)
    FetchSlideFrictionControl:
        'FetchSlide-v1' with a fairly good controller
        This controller cannot slide the puck perfectly to the goal
        As in can push in the direction of the goal
        Action taken as:
            Pi_theta(s) = f(s) + pi_theta(s)
"""
from pickle import TRUE
import gym
from gym.utils import seeding
import numpy as np
from typing import Dict


class FetchSlide(gym.Env):
    """
        FetchSlide:
        The vanilla 'FetchSlide-v1' without any controller
        Can be used for learning with RL from scratch
        No changes made to the original environment
        Just putting it here so that every variation is available in one file.
    """
    def __init__(self, *args, **kwargs):
        self.fetch_env = gym.make('FetchSlide-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space

    def step(self, action):
        observation, reward, done, info = self.fetch_env.step(action)
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
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

class FetchSlideImperfectControl(gym.Env):
    """
        FetchSlideImperfectControl:
            'FetchSlide-v1' with an imperfect controller
            This controller can at least slide the puck but not perfectly
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Parameters:
        -----------
        kp: float
            Scaling factor for position control (kind of)
        hit: float
            Scaling factor to hit the puck
    """
    def __init__(self, kp:float=10, hit:float=5, *args, **kwargs):
        self.fetch_env = gym.make('FetchSlide-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
        ############################
        self.kp = kp
        self.hit = hit
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False
        ############################

    def step(self, residual_action:np.ndarray):
        controller_action = self.controller_action(self.last_observation)
        action = np.add(controller_action, residual_action)
        action = np.clip(action, -1, 1)
        observation, reward, done, info = self.fetch_env.step(action)
        self.last_observation = observation.copy()
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
        self.last_observation = observation.copy()
        ############################
        # parameters for the imperfect controller
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False
        ############################
        return observation

    def seed(self, seed:int=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.fetch_env.seed(seed=seed)

    def render(self, mode:str="human", *args, **kwargs):
        return self.fetch_env.render(mode, *args, **kwargs)

    def close(self):
        return self.fetch_env.close()

    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

    def controller_action(self, obs:Dict, take_action:bool=True, DEBUG:bool=False):
        """
            Given an observation return actions according
            to an imperfect controller
            [grip_pos, object_pos, object_rel_pos, gripper_state, object_rot,
                     object_velp, object_velr, grip_velp, gripper_vel]
            take_action: bool
                Whether use this to take action in environment or just use for subtracting from rand actions
        """
        grip_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        object_rel_pos = obs['observation'][6:9]
        goal_pos = obs['desired_goal']
        # lift the hand little from the table vertically
        if not self.hand_higher:
            action = [0,0,1,0]
            if grip_pos[2]-object_pos[2] > 0.05:
                if take_action:
                    self.hand_higher = True
                if DEBUG:
                    print('Hand lifted from the table')
        # once above, move it above the puck
        if self.hand_higher and not self.hand_above:
            action_pos = list(self.kp * object_rel_pos)
            action = action_pos[:2] + [0,0] # only move in X-Y plane
            if np.linalg.norm(action_pos[:2])<0.01:
                if take_action:
                    self.hand_above = True
                if DEBUG:
                    print('Hand above the puck')
        # once it is above, move it behind the puck in the direction of the goal
        if self.hand_above and not self.hand_behind:
            goal_object_vec = object_pos - goal_pos # vector pointing towards object from goal
            action_pos = list(self.kp * goal_object_vec)
            action = action_pos + [0]
            if np.linalg.norm(grip_pos[:2]-object_pos[:2]) > 0.1:
                if take_action:
                    self.hand_behind = True
                if DEBUG:
                    print('Hand has moved behind')
        # now move the hand down
        if self.hand_behind and not self.hand_down:
            action = [0,0,-1,0]
            if grip_pos[2]-object_pos[2] <0.01:
                if take_action:
                    self.hand_down = True
                if DEBUG:
                    print('Ready to HIT')
        # now give impulse
        if self.hand_down:
            action_pos = list(self.hit * (goal_pos[:2]-grip_pos[:2]) )
            action = action_pos + [0,0]
        action = np.array(action)
        return np.clip(action, -1, 1)

class FetchSlideFrictionControl(gym.Env):
    """
        FetchSlideFrictionControl:
            'FetchSlide-v1' with a fairly good controller
            This controller can slide the puck perfectly to the goal
            for the original frictional coefficient
            But now the friction coefficients are changed and hence it cannot
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)

        Friction Information for MuJoCo:
            http://mujoco.org/book/modeling.html
            Recall that contacts can have up to 5 friction coefficients:
            two tangential, one torsional, two rolling.
            Each contact in mjData.contact actually has all 5 of them,
            even if condim is less than 6 and not all coefficients are used.
            In contrast, geoms have only 3 friction coefficients:
            tangential (same for both axes), torsional, rolling (same for both axes).
            Each of these 3D vectors of friction coefficients is expanded into a 5D vector
            of friction coefficients by replicating the tangetial and rolling components.
            The contact friction coefficients are then computed according to the following rule:
            if one of the two geoms has higher priority, its friction coefficients are used.
            Otherwise the element-wise maximum of each friction coefficient over the two geoms is used.
            The rationale is similar to taking the maximum over condim: we want the more frictional geom to win.

            The reason for having 5 coefficients per contact and only 3 per geom is as follows.
            For a contact pair, we want to allow the most flexible model our solver can handle.
            As mentioned earlier, anisotropic friction can be exploited to model effects such as skating.
            This however requires knowing how the two axes of the contact tangent plane are oriented.
            For a predefined contact pair we know the two geom types in advance, and the corresponding
            collision function always generates contact frames oriented in the same way -
            which we do not describe here but it can be seen in the visualizer.
            For individual geoms however, we do not know which other geoms they might collide with
            and what their geom types might be, so there is no way to know how the contact
            tangent plane will be oriented when specifying an individual geom.
            This is why MuJoCo does now allow anisotropic friction in the individual geom specifications,
            but only in the explicit contact pair specifications.

        Friction for FetchEnv:
            Can get frictions for each body with:
                env.sim.model.geom_friction
                Use env.env.sim.model.geom_names to get body names
                Default friction values are:
                    NOTE: @rhjiang and @tonibronars please verify this
                    array([ [1.e+00, 5.e-03, 1.e-04],   <-  'floor0'
                            [1.e+00, 5.e-02, 1.e-02],   <-  'Unknown'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'Unknown'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'Unknown'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'Unknown'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:base_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:torso_lift_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:head_pan_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:head_tilt_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:shoulder_pan_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:shoulder_lift_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:upperarm_roll_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:elbow_flex_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:forearm_roll_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:wrist_flex_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:wrist_roll_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:gripper_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:r_gripper_finger_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:l_gripper_finger_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:estop_link'
                            [1.e+00, 5.e-03, 1.e-04],   <-  'robot0:laser_link'
                            [1.e+00, 5.e-02, 1.e-02],   <-  'robot0:torso_fixed_link'
                            [1.e-01, 5.e-03, 1.e-04],   <-  'table0'
                            [1.e-01, 5.e-03, 1.e-04]])  <-  'object0'
    """
    def __init__(self, kp:float=10, hit:float=5,*args, **kwargs):
        self.fetch_env = gym.make('FetchSlide-v1')
        self.metadata = self.fetch_env.metadata
        # change friction between all possible contacts
        # NOTE can only change to the last two indices to only change object and table friction
        # as of now we'll work with the original friction
        #for i in range(len(self.fetch_env.env.sim.model.geom_friction)):
        #     self.fetch_env.env.sim.model.geom_friction[i] = [18e-2, 5.e-3, 1e-4]
        ###############################################

        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
        self.kp = kp
        self.hit = hit
        ############################
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False
        # taking only table and puck friction coeffs
        self.mu_table = self.fetch_env.env.sim.model.geom_friction[-2][0]
        self.mu_object = self.fetch_env.env.sim.model.geom_friction[-1][0]
        self.mu = np.max(np.array([self.mu_table, self.mu_object])) # take max of the friction coeffs
        self.r = self.fetch_env.env.sim.model.geom_size[-1][0]
        self.dt = self.fetch_env.env.sim.model.opt.timestep
        self.g = 9.81
        ############################

    def step(self, residual_action:np.ndarray):
        controller_action = self.controller_action(self.last_observation)
        action = np.add(controller_action, residual_action)
        action = np.clip(action, -1, 1)
        observation, reward, done, info = self.fetch_env.step(action)
        self.last_observation = observation.copy()
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
        self.last_observation = observation.copy()
        #object_pos = observation['observation'][3:6]
        #goal_pos = observation['desired_goal']
        ############################
        # parameters for the imperfect controller
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False
        # choose d1 according to the dist between goal and object
        # do not want it to be bigger than the distance we want to move the puck by
        # d1 is the distance the gripper will move to slide the puck
        # self.d1 = np.linalg.norm(goal_pos - object_pos)/5
        ############################
        return observation

    def seed(self, seed:int=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.fetch_env.seed(seed=seed)

    def render(self, mode:str="human", *args, **kwargs):
        return self.fetch_env.render(mode, *args, **kwargs)

    def close(self):
        return self.fetch_env.close()

    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

    def controller_action(self, obs:Dict, take_action:bool=True, DEBUG:bool=False):
        """
            Given an observation return actions according
            to an imperfect controller
            [grip_pos, object_pos, object_rel_pos, gripper_state, object_rot,
                     object_velp, object_velr, grip_velp, gripper_vel]
            take_action: bool
                Whether use this to take action in environment or just use for subtracting from rand actions
        """
        grip_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        object_rel_pos = obs['observation'][6:9]
        goal_pos = obs['desired_goal']
        # lift the hand little from the table vertically
        if not self.hand_higher:
            action = [0,0,1,0]
            if grip_pos[2]-object_pos[2] > 0.05:
                if take_action:
                    self.hand_higher = True
                if DEBUG:
                    print('Hand lifted from the table')
        # once above, move it above the puck
        if self.hand_higher and not self.hand_behind:
            goal_grip_pos = object_pos + (0.025 + self.r)*(object_pos - goal_pos)/np.linalg.norm(object_pos - goal_pos)
            # goal_object_vec = object_pos - goal_pos # vector pointing towards object from goal
            # action_pos = list(self.kp * goal_object_vec)
            action_pos = list(self.kp*(goal_grip_pos - grip_pos))
            action = action_pos[:2] + [0,0]
            if np.linalg.norm(grip_pos[:2]-goal_grip_pos[:2]) < 0.001:
                if take_action:
                    self.hand_behind = True
                if DEBUG:
                    print('Hand has moved behind')
        # now move the hand down
        if self.hand_behind and not self.hand_down:
            action = [0,0,-1,0]
            if grip_pos[2]-object_pos[2] <0.01:
                self.start_time = self.fetch_env.env.sim.data.time  # start the time once we are ready to hit
                self.prev_time = self.start_time
                self.d1 = np.linalg.norm(goal_pos[:-1] - object_pos[:-1])/5 # Define d1 wrt the initial gripper pose rather than the object pose
                self.d2 = (np.linalg.norm(goal_pos[:-1] - object_pos[:-1]) - self.d1)
                self.f = self.d2 * self.mu * self.g / self.d1

                if take_action:
                    self.hand_down = True

                v1 = np.sqrt(2*self.d2*self.mu*self.g)
                a = v1**2/(2*self.d1)

                if DEBUG:
                    print('d2 = ' + str(self.d2))
                    print('mu = ' + str(self.mu))
                    print('v1 = ' +str(v1))
                    print('d1 = ' + str(self.d1))
                    print('a = '+str(a))
                    print('Ready to HIT')
        # slide the puck
        if self.hand_down:
            v1 = np.sqrt(2*self.d2*self.mu*self.g)
            a = v1**2/(2*self.d1)
            if np.linalg.norm(goal_pos[:-1] - grip_pos[:-1]) > self.d2:
                if DEBUG:
                    print('this is the distance ' + str(np.linalg.norm(goal_pos[:-1] - grip_pos[:-1])))
                cur_time = self.fetch_env.env.sim.data.time
                # delta s = sdot * dt, where sdot = f*t and s is measured along direction from puck to goal
                action_pos = list((goal_pos - grip_pos)/np.linalg.norm(goal_pos - grip_pos) * self.f * (cur_time - self.start_time)*(cur_time - self.prev_time))
                self.prev_time = cur_time
                #print('current speed = ' + str(a*(cur_time-self.start_time)))
            else:
                #print('no push')
                action_pos = [0,0]
            action = action_pos[:2] + [0,0]
            if DEBUG:
                print('commanded action = ' + str(np.linalg.norm(action[0:2])))
        # added clipping here
        #return action
        return np.clip(action, -1, 1)

class FetchSlideSlapControl(gym.Env):
    """
        FetchSlideImperfectControl:
            'FetchSlide-v1' with an imperfect controller
            This controller can at least slide the puck but not perfectly
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Parameters:
        -----------
        kp: float
            Scaling factor for position control (kind of)
        hit: float
            Scaling factor to hit the puck
    """
    def __init__(self, kp:float=10, hit:float=5, *args, **kwargs):
        self.fetch_env = gym.make('FetchSlide-v1')
        self.metadata = self.fetch_env.metadata
        self.max_episode_steps = self.fetch_env._max_episode_steps
        self.action_space = self.fetch_env.action_space
        self.observation_space = self.fetch_env.observation_space
        ############################
        self.kp = kp
        self.hit = hit
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False

        self.set_goal_speed = False
        self.dist_to_goal = 0
        self.goal_speed = 0
        self.prev_speed = 0
        self.a = 0
        self.prev_time = 0

        # taking only table and puck friction coeffs
        self.mu_table = self.fetch_env.env.sim.model.geom_friction[-2][0]
        self.mu_object = self.fetch_env.env.sim.model.geom_friction[-1][0]
        self.mu = np.max(np.array([self.mu_table, self.mu_object])) # take max of the friction coeffs
        self.r = self.fetch_env.env.sim.model.geom_size[-1][0]
        self.dt = self.fetch_env.env.sim.model.opt.timestep
        self.g = 9.81
        ############################

    def step(self, residual_action:np.ndarray):
        controller_action = self.controller_action(self.last_observation)
        action = np.add(controller_action, residual_action)
        action = np.clip(action, -1, 1)
        observation, reward, done, info = self.fetch_env.step(action)
        self.last_observation = observation.copy()
        return observation, reward, done, info

    def reset(self):
        observation = self.fetch_env.reset()
        self.last_observation = observation.copy()
        ############################
        # parameters for the imperfect controller
        self.hand_above = False
        self.hand_higher = False
        self.hand_down = False
        self.hand_behind = False

        self.set_goal_speed = False
        self.dist_to_goal = 0
        self.goal_speed = 0
        self.prev_speed = 0
        self.a = 0
        self.prev_time = 0

        ############################
        return observation

    def seed(self, seed:int=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.fetch_env.seed(seed=seed)

    def render(self, mode:str="human", *args, **kwargs):
        return self.fetch_env.render(mode, *args, **kwargs)

    def close(self):
        return self.fetch_env.close()

    def compute_reward(self, *args, **kwargs):
        return self.fetch_env.compute_reward(*args, **kwargs)

    def controller_action(self, obs:Dict, take_action:bool=True, DEBUG:bool=False):
        """
            Given an observation return actions according
            to an imperfect controller
            [grip_pos, object_pos, object_rel_pos, gripper_state, object_rot,
                     object_velp, object_velr, grip_velp, gripper_vel]
            take_action: bool
                Whether use this to take action in environment or just use for subtracting from rand actions
        """
        grip_pos = obs['observation'][:3]
        object_pos = obs['observation'][3:6]
        object_rel_pos = obs['observation'][6:9]
        goal_pos = obs['desired_goal']

        disp = 0.005 # extra distance to accellerate
        gripper_dim = 0.029 # maximum planar distance across the gripper

        # lift the hand little from the table vertically
        if not self.hand_higher:
            action = [0,0,1,0]
            if grip_pos[2]-object_pos[2] > 0.05:
                if take_action:
                    self.hand_higher = True
                if DEBUG:
                    print('Hand lifted from the table')
        # once above, move it above the puck
        if self.hand_higher and not self.hand_behind:
            goal_grip_pos = object_pos + (disp + gripper_dim + self.r)*(object_pos - goal_pos)/np.linalg.norm(object_pos - goal_pos)
            action_pos = list(self.kp*(goal_grip_pos - grip_pos))
            action = action_pos[:2] + [0,0]
            if np.linalg.norm(grip_pos[:2]-goal_grip_pos[:2]) < 0.001:
                if take_action:
                    self.hand_behind = True
                if DEBUG:
                    print('Hand has moved behind')
        # now move the hand down
        if self.hand_behind and not self.hand_down:
            action = [0,0,-1,0]
            if grip_pos[2]-object_pos[2] <0.01:
                self.start_time = self.fetch_env.env.sim.data.time  # start the time once we are ready to hit
                if take_action:
                    self.hand_down = True
                if DEBUG:
                    print('Ready to HIT')

        # define helper functions
        def calc_direction(pos, goal):
            # calculates unit vector direction from position to goal
            pos = pos[:-1]; goal = goal[:-1]
            return (goal - pos)/np.linalg.norm(goal - pos)

        # set the goal speed
        if self.hand_down and not self.set_goal_speed:
            self.dist_to_goal = np.linalg.norm(object_pos[:-1]-goal_pos[:-1])
            self.goal_speed = np.sqrt(2*self.mu*self.g*self.dist_to_goal)
            self.a = (self.goal_speed**2)/(2*disp)
            #print('this is dist to goal ' +str(self.dist_to_goal))
            #print('this is mu ' +str(self.mu))
            #print('this is the goal speed ' + str(self.goal_speed))
            #print('this is the timestep ' + str(self.dt))
            #print('this is a ' + str(self.a))
            #print('this is 0.025+self.r ' +str(0.025+self.r))
            #print('this is the distance between gripper pos and object pos ' +str(np.linalg.norm(object_pos-grip_pos)))
            self.prev_time = self.start_time
            if take_action:
                self.set_goal_speed = True

        # slap the puck
        if self.hand_down and self.set_goal_speed:
            time = self.fetch_env.env.sim.data.time
            dtime = time - self.prev_time
            if np.linalg.norm(goal_pos[:-1] - grip_pos[:-1]) > self.dist_to_goal:
                if DEBUG:
                    print('this is the distance ' + str(np.linalg.norm(goal_pos[:-1] - grip_pos[:-1])))
                #print(self.prev_speed)
                #print(next_speed)
                next_speed = self.prev_speed + (self.a * dtime)
                action_pos = list((dtime*(next_speed+self.prev_speed)/2)*calc_direction(object_pos,goal_pos))
                self.prev_speed = next_speed
                self.prev_time = time
            else:
                action_pos = [0,0]
            action = action_pos[:2] + [0,0]
            if DEBUG:
                print('commanded action = ' + str(np.linalg.norm(action[0:2])))
        import pdb
        pdb.set_trace()
        # added clipping here
        return np.clip(action, -1, 1)

# if __name__ == "__main__":
#     #env_name = 'FetchSlideSlapControl'
#     env_name = 'FetchSlideFrictionControl'
#     #env_name = 'FetchSlideImperfectControl'
#     env = globals()[env_name]() # this will initialise the class as per the string env_name
#     #env = gym.wrappers.Monitor(env, 'video/' + env_name, force=True)
#     successes = []
#     # set the seeds
#     env.seed(1)
#     env.action_space.seed(1)
#     env.observation_space.seed(1)
#     for ep in range(10):
#         success = np.zeros(env.max_episode_steps)
#         obs = env.reset()
#         action = [0,0,0,0]  # give zero action at first time step
#         for i in (range(env.max_episode_steps)):
#             env.render(mode='human')
#             obs, rew, done, info = env.step(action)
#             success[i] = info['is_success']
#         successes.append(success)
#     successes = np.array(successes)
#     env.close()

import tqdm
if __name__ == "__main__":
    env_name = 'FetchSlideSlapControl'
    #env_name = 'FetchSlideFrictionControl'
    #env_name = 'FetchSlideImperfectControl'
    env = globals()[env_name]() # this will initialise the class as per the string env_name
    # env = gym.wrappers.Monitor(env,'video/'+env_name, force=True) # can be used to record videos but this does not work properly
    obs = env.reset()

    
    hand_above = False
    hand_behind = False
    hand_higher = False
    hand_down = False
    object_in_hand = False
    action = [0,0,0,0]   # give zero action at first time step
    # time.sleep(5)
    for i in range(300):
        obs, rew, done, info = env.step(action)
        #env.sim.render(mode='window')
        env.render(mode='human')
        #time.sleep(0.1)
        if info['is_success']:
             break

    print('Done')
