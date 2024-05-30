import random
from collections import OrderedDict

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.arenas import AssemblyArena,PegsArena,EmptyArena
from robosuite.models.objects import SquareObject,CircleObject,PegSquareObject,PegCircleObject,RoundNutObject, SquareNutObject
from robosuite.environments.manipulation.lwr_arm_env import LwrArmEnv
from robosuite.utils.observables import Observable, sensor
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import mujoco_py
from dm_control.mujoco.testing import assets
from dm_control.utils import inverse_kinematics as ik
from dm_control import mujoco
from robosuite.utils.transform_utils import mat2quat
import pdb
import copy

class MultiPegAssembly(LwrArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        initialization_noise="default",
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1,
        reward_shaping=True,
        placement_initializer=None,
        object_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera='agentview',
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=60,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=540,
        camera_widths=960,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # task settings
        self.object_to_id = {"circle": 0, "pegcircle": 1}
        #self.object_id = self.object_to_id[object_type]  # use for convenient indexing
        self.object_id_to_sensors = {}  # Maps nut id to sensor names for that nut
        self.fixed = False
        self.nut_id_to_sensors = {}  # Maps nut id to sensor names for that nut

        self.obj_to_use = None

         # settings for table top
        self.table_friction = table_friction

         # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        self.initial_distance = None
        self.prev_distance_to_goal = None
        self.left_hole_pos = None
        self.right_hole_pos = None
        self.hole_pos = None
        self.site1_pos = None
        self.site2_pos = None
        self.x_random = 0.0
        self.y_random = 0.0
        self.rot_angle = 0.0
        self.seed_object  = 0

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=None,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
    
      
    


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        #self.seed_object = random.randint(0,3)
        self.seed_object = 0
        mujoco_arena = AssemblyArena(self.seed_object)
        mujoco_arena.set_origin([0, 0, 0])
        
        # define target object
        self.target_objects = PegCircleObject("PegCircle",self.seed_object)
       

        #task includes arena, robot, and objects of interest
        self.model = ManipulationTask(mujoco_arena=mujoco_arena, mujoco_robots=[robot.robot_model for robot in self.robots]
                                        , mujoco_objects=self.target_objects,)
           
    def _setup_references(self):

        super()._setup_references()

        self.table_body_id = self.sim.model.body_name2id("table")
        self.peg_body_id = self.sim.model.body_name2id("PegCircle_main")
        self.hole_body_id = self.sim.model.body_name2id("circle")
        

        nut = self.target_objects
        self.peg_geom_id = [self.sim.model.geom_name2id(g) for g in nut.contact_geoms]
        
        # information of objects
        self.left_site_ids = [self.sim.model.site_name2id(nut.important_site1["handle1"])]
        self.right_site_ids = [self.sim.model.site_name2id(nut.important_site2["handle2"])]   

        self.left_hole_ids = [self.sim.model.site_name2id('hole_left_site')]
        self.right_hole_ids = [self.sim.model.site_name2id('hole_right_site')]
        self.peg_top_ids = [self.sim.model.site_name2id('PegCircle_peg_top_site')]
        

        self.objects_on_pegs = np.zeros(1)    

    def _setup_observables(self):

        observables = super()._setup_observables()
        #self.show_model_info() 
        
        #print('22',mat2quat(np.array(self.sim.data.get_site_xmat('PegCircle_default_site').reshape(3,3))))
         # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            self.nut_id_to_sensors = {}

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return (
                    T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"])))
                    if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                    else np.eye(4)
                )

            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            enableds = [True]
            actives = [False]

            peg = self.target_objects
            using_nut = True
            nut_sensors, nut_sensor_names = self._create_peg_sensors(peg_name=peg.name, modality=modality)
            sensors += nut_sensors
            names += nut_sensor_names
            enableds += [using_nut] * 4
            actives += [using_nut] * 4
            self.nut_id_to_sensors = nut_sensor_names  #['PegCircle_pos', 'PegCircle_quat']
          
         # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )  
        return observables

    def _reset_internal(self):
        super()._reset_internal()
        #init_qpos = self.printNamedJointsWithMultipleDOFs()
        #pdb.set_trace()
        self.grip_site_pos = self.sim.data.get_site_xpos('robot0_grip_site')     
        obj = self.target_objects    

        obj_pos2 = np.array(self.grip_site_pos)
        obj_pos2[1] = obj_pos2[1]
        obj_pos2[2] = obj_pos2[2]-[0.065]##0.07


        self.x_random = np.random.uniform(-0.015,0.015)
        self.y_random = np.random.uniform(-0.015,0.015)
        #print('reset_hole_before',self.sim.model.body_pos[2])
        self.sim.model.body_pos[2] =[0.46 +self.x_random,0+self.y_random,0.8]
        self.rot_angle = 0.01*np.random.uniform(-1, 1)
        self.sim.model.body_quat[2] = np.array([0.707107, 0.707107, 0, 0])+np.array([0, 0, self.rot_angle, 0])
        self.x_random = 0.0
        self.x_random = 0.0
        self.rot_angle = 0.0
           
            
        obj_quat = [0,0,0,1]  #fixed orientation
        self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([obj_pos2, np.array(obj_quat)]))

        
        object_names = {obj.name}        
        object_names.remove(obj.name)
        self.clear_objects(list(object_names))
        ##update peg_sensor
        
        for name in self.nut_id_to_sensors:
            # Set all of these sensors to be enabled and active if this is the active nut, else False
            self._observables[name].set_enabled(True)
            self._observables[name].set_active(True)
        
        # f = open("assembly_task.xml", "w", encoding="utf-8")
        # self.sim.save(f,format='xml')
        



        
    def reward(self,action=None):
        
        reward = 0
        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards_me(action)
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return float(reward)
    
    def staged_rewards_me(self,action):
        r_reset = 0
        self._check_success()
        if (self._check_success()):
            print('this trail is successful')
        success_reward = self.objects_on_pegs
        distance_reward = 1 - np.tanh(15.0 * self.distance_to_goal())
        #print(self.initial_distance, self.prev_distance_to_goal,self.distance_to_goal() )
        # distance_reward = (self.prev_distance_to_goal - self.distance_to_goal()) / self.initial_distance
        # self.prev_distance_to_goal = self.distance_to_goal().copy()
        reward = distance_reward + success_reward
       
        return r_reset,reward
    
    # def distance_to_goal(self):
    #     grip_site_pos = self.sim.data.get_site_xpos('robot0_grip_site')
    #     hole_pos = self.hole_pos.copy()
    #     hole_pos[2] = hole_pos[2] +0.06+0.09
        
    #     distance = np.linalg.norm(hole_pos - grip_site_pos)
    #     return distance

    def _check_success(self):
        
        self.left_hole_pos = np.array(self.sim.data.site_xpos[self.left_hole_ids])
        self.right_hole_pos = np.array(self.sim.data.site_xpos[self.right_hole_ids])
        self.hole_pos = self.sim.data.body_xpos[self.hole_body_id]
      

        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].grip_site_id]
        self.site1_pos = self.sim.data.site_xpos[ self.left_site_ids]
        self.site2_pos = self.sim.data.site_xpos[self.right_site_ids]
        res = False
        if (
            abs(self.left_hole_pos[0][0]- self.site1_pos[0][0]) < 0.015
            and abs(self.left_hole_pos[0][1]- self.site1_pos[0][1]) < 0.015          
            and self.site1_pos[0][2] < self.left_hole_pos[0][2] + 0.065
            and abs(self.right_hole_pos[0][0]- self.site2_pos[0][0]) < 0.015
            and abs(self.right_hole_pos[0][1]- self.site2_pos[0][1]) < 0.015           
            and self.site2_pos[0][2] < self.right_hole_pos[0][2] + 0.065
        ):
            res = True
        if res:
            self.objects_on_pegs = 1

        return  res
        
    
    def distance_to_goal(self):
       
        hole_pos1 =copy.deepcopy(self.left_hole_pos[0])
        hole_pos1[2] = hole_pos1[2] +0.06
        hole_pos2 = copy.deepcopy(self.right_hole_pos[0])
        hole_pos2[2] = hole_pos2[2] +0.06
        site1_pos = copy.deepcopy(self.site1_pos[0])
        site2_pos = copy.deepcopy(self.site2_pos[0])
        #print('late',hole_pos1)
       
        distance = (np.linalg.norm(hole_pos1 - site1_pos) + np.linalg.norm(hole_pos2 - site2_pos))/2
        return distance

    def staged_rewards(self,action):
        grasp_mult = 0.0
        hover_mult = 0.4
        assembly_mult = 5.4

        id_left = self.left_site_ids
        site1_pos = self.sim.data.site_xpos[id_left]
        id_right = self.right_site_ids
        site2_pos = self.sim.data.site_xpos[id_right]

        left_hole_pos = self.sim.data.site_xpos[self.left_hole_ids]
        left_hole_pos[0][2] = left_hole_pos[0][2] +0.082 #0.07
        right_hole_pos = self.sim.data.site_xpos[self.right_hole_ids]
        right_hole_pos[0][2] = right_hole_pos[0][2] +0.082
        peg_top_pos = self.sim.data.site_xpos[self.peg_top_ids]
        
        
        obj_pos_x = (left_hole_pos[0][0] + right_hole_pos[0][0])/2
        obj_pos_y = (left_hole_pos[0][1] + right_hole_pos[0][1])/2
        obj_pos_z = (left_hole_pos[0][2] + right_hole_pos[0][2])/2+0.09
        target_top_pos = np.array([obj_pos_x,obj_pos_y,obj_pos_z])

        r_grasp = 0
        # 2.reaching reward governed by distance to closest object
        r_hover = 0.0
            # reaching reward via minimum distance to the handles of the objects
        

        dist1_xy = np.linalg.norm(left_hole_pos[0][:2] - site1_pos[0][:2])
        dist2_xy = np.linalg.norm(right_hole_pos[0][:2] - site2_pos[0][:2])
        dist3_xy = np.linalg.norm(peg_top_pos[0] - target_top_pos)
  
        #print('distance of reward',abs(left_hole_pos[0][0]- site1_pos[0][0]),abs(left_hole_pos[0][1]- site1_pos[0][1])) 
        #r_hover = r_grasp + ((1 - np.tanh(20*dist1_xy)) + (1 - np.tanh(20*dist2_xy)) + (1 - np.tanh(20*dist3_xy)))/3 * (hover_mult-grasp_mult)
        r_hover = r_grasp + ((1 - np.tanh(10*dist1_xy)) + (1 - np.tanh(10*dist2_xy)))/2 * (hover_mult-grasp_mult)
        r_assembly = 0.0
        #dist_z2 = 0.0

        penalty =0
        if abs(left_hole_pos[0][0]- site1_pos[0][0]) < 0.01 and abs(left_hole_pos[0][1]- site1_pos[0][1]) < 0.01 and abs(right_hole_pos[0][0]- site2_pos[0][0]) < 0.01 and abs(right_hole_pos[0][1]- site2_pos[0][1]) < 0.01:
            #penalty = -(abs(action[0]) + abs(action[1]) +10*(abs(action[3])+abs(action[4])+abs(action[5])))/5
            dist_z2 = site1_pos[0][2] - 0.85
            #r_assembly = 1+r_hover+(1 - np.tanh((dist_z2))) * (assembly_mult-hover_mult)+penalty*0.4
            r_assembly = 1+(1- np.tanh((10*dist_z2))) * (assembly_mult-hover_mult)
            if dist_z2<0.03:
                r_assembly = 2+(1- np.tanh((10*dist_z2))) * (assembly_mult-hover_mult)
        #if 0<dist_z2<0.03:
        #    print('assembly_score is',r_hover,r_assembly,dist_z2)
        
        #print(left_hole_pos,site1_pos)
        return r_grasp,r_hover,r_assembly
    


    
    
    
    def _create_peg_sensors(self,  peg_name, modality="object"):
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def peg_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.peg_body_id])

        @sensor(modality=modality)
        def peg_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to="xyzw")
        
        @sensor(modality=modality)
        def hole_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.hole_body_id])

        sensors = [peg_pos, peg_quat,hole_pos]
        names = [f"{peg_name}_pos", f"{peg_name}_quat",f"{peg_name}_hole_pos"]

        return sensors, names

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """

        print("\nNumber of bodies: {}".format(self.sim.model.nbody))
        for i in range(self.sim.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.sim.model.body_id2name(i)))

        print("\nNumber of joints: {}".format(self.sim.model.njnt))
        for i in range(self.sim.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}".format(
                    i, self.sim.model.joint_id2name(i), self.sim.model.jnt_range[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.sim.data.ctrl)))
        for i in range(len(self.sim.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Control Range: {}".format(
                    i,
                    self.sim.model.actuator_id2name(i),
                    #self.actuators[i][3],
                    self.sim.model.actuator_ctrlrange[i],
                )
            )

        #print("\nJoints in kinematic chain: {}".format([i.name for i in self.ee_chain.links]))

        print("\n Camera Info: \n")
        for i in range(self.sim.model.ncam):
            print(
                "Camera ID: {}, Camera Name: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}".format(
                    i,
                    self.sim.model.camera_id2name(i),
                    self.sim.model.cam_fovy[i],
                    self.sim.model.cam_pos0[i],
                    self.sim.model.cam_mat0[i],
                )
            )
    
    def printNamedJointsWithMultipleDOFs(self):
       
        """Regression test for b/77506142."""
        _ARM_XML = "/data/robosuite_manipulation/robosuite/robosuite/models/assets/robots/lwr/robot.xml"
        physics = mujoco.Physics.from_xml_path(_ARM_XML)
        #pdb.set_trace()
        site_name = 'ft_frame'  
        #'grip_site' 
        joint_names = ['joint_1', 'joint_2','joint_3','joint_4','joint_5','joint_6','joint_7']
        _TOL = 1.2e-14
        _MAX_STEPS = 100
        
        # This target position can only be achieved by rotating both ball joints. If
        # DOFs are incorrectly indexed by joint index then only the first two DOFs
        # in the first ball joint can change, and IK will fail to find a solution.
        target_pos = np.array([ 0.46, 0,  0.894])+[0,0,0.25]
        target_quat = np.array([0,0, 1,0])
        #pdb.set_trace()
        result = ik.qpos_from_site_pose(
            physics=physics,
            site_name=site_name,
            target_pos=target_pos,
            target_quat = target_quat,
            joint_names=joint_names,
            tol=_TOL,
            max_steps=_MAX_STEPS,
            inplace=True)
        
        return result
    
    def get_camera_matrices(self, camera_name, w, h):
        camera_id = self.sim.model.camera_name2id(camera_name)
        fovy = self.sim.model.cam_fovy[camera_id]
        f = 0.5 * h / math.tan(fovy * math.pi / 360)
        camera_matrix = np.array(((f, 0, w / 2), (0, f, h / 2), (0, 0, 1)))
        xmat = self.sim.data.get_camera_xmat(camera_name)
        xpos = self.sim.data.get_camera_xpos(camera_name)

        camera_transformation = np.eye(4)
        camera_transformation[:3, :3] = xmat
        camera_transformation[:3, 3] = xpos
        camera_transformation = np.linalg.inv(camera_transformation)[:3, :]

        return camera_matrix, camera_transformation
        
     
class PegAssembly(MultiPegAssembly):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "nut_type" not in kwargs, "invalid set of arguments"
        super().__init__(object_type="pegcircle", **kwargs)