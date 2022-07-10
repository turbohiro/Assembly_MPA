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
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        object_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=None,
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # task settings
        self.object_to_id = {"circle": 0, "pegcircle": 1}
        self.object_id = self.object_to_id[object_type]  # use for convenient indexing
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
    
    def on_hole(self, site1_pos, site2_pos):

        hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
        res = False
        obj_pos_x = (site1_pos[0][0] + site2_pos[0][0])/2
        obj_pos_y = (site1_pos[0][1] + site2_pos[0][1])/2
        obj_pos_z = (site1_pos[0][2] + site2_pos[0][2])/2
        if (
            abs(obj_pos_x- hole_pos[0]) < 0.03
            and abs(obj_pos_y - hole_pos[1]) < 0.03
            and obj_pos_z < hole_pos[2] + 0.033
        ):
            res = True
        return res

    def peg_to_hole(self,return_distance = False):
        hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
        id_left = self.left_site_ids
        site1_pos = self.sim.data.site_xpos[id_left]
        id_right = self.right_site_ids
        site2_pos = self.sim.data.site_xpos[id_right]

        obj_pos_x = (site1_pos[0][0] + site2_pos[0][0])/2
        obj_pos_y = (site1_pos[0][1] + site2_pos[0][1])/2
        obj_pos_z = (site1_pos[0][2] + site2_pos[0][2])/2
        peg_pos = np.array([obj_pos_x,obj_pos_y,obj_pos_z])
         # Calculate distance
        diff = hole_pos - peg_pos
        # Return appropriate value
        return np.linalg.norm(diff) if return_distance else diff

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        mujoco_arena = AssemblyArena()
        mujoco_arena.set_origin([0, 0, 0])
        
        # define target object
        self.target_objects = PegCircleObject("PegCircle")
       

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
        
        grip_site_pos = self.sim.data.get_site_xpos('robot0_grip_site')     
        obj = self.target_objects    
        if self.fixed:
            obj_pos = np.array(grip_site_pos)
            obj_pos[2] = np.array([0.82])
        else:
            obj_pos2 = np.array(grip_site_pos)
            obj_pos2[1] = obj_pos2[1]
            obj_pos2[2] = obj_pos2[2]-[0.065]##0.07

            self.sim.model.body_pos[2] =[obj_pos2[0],obj_pos2[1],0.8]

            
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
        # pdb.set_trace()
        


        
    def reward(self,action=None):
        # compute sparse rewards
        self._check_success()
        reward = self.objects_on_pegs
        #print(reward)

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward
    
    def staged_rewards(self):
        grasp_mult = 0.35
        reach_mult = 0.7

         # 1. grasping reward for touching any objects of interest
        r_grasp = (
            int(
                self._check_grasp(
                    gripper=self.robots[0],
                    object_geoms=[self.target_objects.contact_geoms],
                )
            )
            * grasp_mult
        )
        # 2.reaching reward governed by distance to closest object
        r_reach = 0.0
            # reaching reward via minimum distance to the handles of the objects
        if r_grasp > 0.0:
            dists = [self.peg_to_hole]
            r_reach = (1 - np.tanh(10.0 * min(dists))) * reach_mult

        return r_grasp,r_reach

    
    def _check_success(self):
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].grip_site_id]
        id_left = self.left_site_ids
        peg_left_pos1 = self.sim.data.site_xpos[id_left]
        id_right = self.right_site_ids
        peg_right_pos2 = self.sim.data.site_xpos[id_right]

        peg_pos = self.sim.data.body_xpos[self.peg_body_id]

        distance_peg_gripper_y = abs(peg_pos[1] - gripper_site_pos[1])
        
        if self.on_hole(peg_left_pos1,peg_right_pos2) and distance_peg_gripper_y<0.1:
            self.objects_on_pegs = 1
        
        return  self.objects_on_pegs==1
    
    def _create_peg_sensors(self,  peg_name, modality="object"):
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def peg_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.peg_body_id])

        @sensor(modality=modality)
        def peg_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to="xyzw")

        sensors = [peg_pos, peg_quat]
        names = [f"{peg_name}_pos", f"{peg_name}_quat"]

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
        
     
class PegAssembly(MultiPegAssembly):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "nut_type" not in kwargs, "invalid set of arguments"
        super().__init__(object_type="pegcircle", **kwargs)