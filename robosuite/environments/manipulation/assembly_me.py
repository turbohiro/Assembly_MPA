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
        horizon=500,
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

        #self.nut_id = self.nut_to_id[nut_type]  # use for convenient indexing
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
    
    def on_hole(self, obj_pos, peg_id):

        hole_pos = np.array(self.sim.data.body_xpos[self.peg2_body_id])
        res = False
        if (
            abs(obj_pos[0] - hole_pos[0]) < 0.03
            and abs(obj_pos[1] - hole_pos[1]) < 0.03
            and obj_pos[2] < self.table_offset[2] + 0.05
        ):
            res = True
        return res
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()


        # load model for table top workspace
        mujoco_arena = AssemblyArena()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

       

        # define target objects
        self.target_objects = []
        object_names = ("Circle", "Pegcircle")

         #peg object
        #peg_object = PegCircleObject("PegCircle")
        #pdb.set_trace()
        #peg_object.set("pos", "0.46 0 0.84")
        
        #self.target_objects.append(peg_object)

        self.table_offset = np.array((0, 0, 0.8))
        # Create default (SequentialCompositeSampler) sampler if it has not already been specified
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            for object_name, default_range in zip(object_names, ([[0.45, 0.46],[-0.02, 0.0]],[[0.45, 0.46],[-0.01, 0.01]])):
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=f"{object_name}Sampler",
                        x_range=default_range[0],
                        y_range=default_range[1],
                        rotation=None,
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.02,
                    )
                )
        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        for i, (object_cls, object_name) in enumerate(
            zip(
                (CircleObject,PegCircleObject),
                object_names,
            )
        ):
            object = object_cls(name=object_name)
            #if object_name =='Circle':
            self.target_objects.append(object)
            # Add this nut to the placement initializer
            if isinstance(self.placement_initializer, SequentialCompositeSampler):
            #if object_name =='Circle':
                # assumes we have two samplers so we add nuts to them
                self.placement_initializer.add_objects_to_sampler(sampler_name=f"{object_name}Sampler", mujoco_objects=object)
            else:
                # This is assumed to be a flat sampler, so we just add all nuts to this sampler
                self.placement_initializer.add_objects(object)
                #break
        
        #task includes arena, robot, and objects of interest
        self.model = ManipulationTask(mujoco_arena=mujoco_arena, mujoco_robots=[robot.robot_model for robot in self.robots]
                                        , mujoco_objects=self.target_objects,)
        
        
         

    def _setup_observables(self):

        observables = super()._setup_observables()
        #self.show_model_info() 
        #print('******',self.sim.data.get_body_xpos('robot0_eef_gripper'),self.sim.data.get_body_xpos('Circle_main'),self.sim.data.get_body_xpos('Pegcircle_main'))
        #print('******',mat2quat(np.array(self.sim.data.site_xmat('2').reshape(3,3))),mat2quat(np.array(self.sim.data.site_xmat('7').reshape(3,3))),mat2quat(np.array(self.sim.data.site_xmat('8').reshape(3,3))))
        print('11',mat2quat(np.array(self.sim.data.get_site_xmat('Circle_default_site').reshape(3,3))))
        print('22',mat2quat(np.array(self.sim.data.get_site_xmat('Pegcircle_default_site').reshape(3,3))))
        

        return observables


    def _reset_internal(self):

        super()._reset_internal()
        init_qpos = self.printNamedJointsWithMultipleDOFs()
        #pdb.set_trace()
        #peg_object = PegCircleObject("PegCircle")
        grip_site_pos = self.sim.data.get_site_xpos('robot0_grip_site')
        #obj_pos = grip_site_pos 
        #obj_quat = [1,0,0,0]
        #self.sim.data.set_joint_qpos(peg_object.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

  
         # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():                
                if obj.joints[0]=='Circle_joint0':                
                    if self.fixed:
                        obj_pos = np.array(grip_site_pos)
                        obj_pos[2] = np.array([0.82])
                    else:
                        obj_pos2 = np.array(grip_site_pos)
                        obj_pos2[1] = obj_pos2[1]+np.array(np.random.uniform(-0.01,0.01))
                        obj_pos2[2] = obj_pos[2]-[0.03]
                    #obj_quat = [1,0,0,0]  #fixed orientation
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([obj_pos2, np.array(obj_quat)]))
                else:

                    obj_pos = np.array(grip_site_pos)
                    obj_pos[2] = obj_pos[2]-[0.05]
                    obj_quat = [1,0,0,0]
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([obj_pos, np.array(obj_quat)]))
        
        #self.sim.data.qpos[7:9] = [-0.1,0.1]           
        object_names = {object.name for object in self.target_objects}
        for i in range(len(object_names)):
            object_names.remove(self.target_objects[i].name)

        self.clear_objects(list(object_names))

        
        
           
    

    def reward(self, action=None):

        # compute sparse rewards
        self._check_success()
        reward = 1

        # add in shaped rewards
        if self.reward_shaping:
            staged_rewards = self.staged_rewards()
            reward += max(staged_rewards)
        if self.reward_scale is not None:
            reward *= self.reward_scale
            reward /= 4.0
        return reward
    
    def _check_success(self):


        return True

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
        site_name = 'grip_site'
        joint_names = ['joint_1', 'joint_2','joint_3','joint_4','joint_5','joint_6','joint_7']
        _TOL = 1.2e-14
        _MAX_STEPS = 100
        
        # This target position can only be achieved by rotating both ball joints. If
        # DOFs are incorrectly indexed by joint index then only the first two DOFs
        # in the first ball joint can change, and IK will fail to find a solution.
        target_pos = np.array([ 0.44510334, -0.02574342,  0.89427983])+[0,0,0.1]
        #pdb.set_trace()
        result = ik.qpos_from_site_pose(
            physics=physics,
            site_name=site_name,
            target_pos=target_pos,
            joint_names=joint_names,
            tol=_TOL,
            max_steps=_MAX_STEPS,
            inplace=True)
        
        return result
        
     
class PegAssemblySquare(MultiPegAssembly):

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "nut_type" not in kwargs, "invalid set of arguments"
        super().__init__(object_type="pegcircle", **kwargs)