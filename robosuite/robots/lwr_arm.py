import copy
import os
from collections import OrderedDict

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.robots.manipulator_me import ManipulatorMe
from robosuite.controllers import controller_factory, load_controller_config
from robosuite.utils.buffers import DeltaBuffer, RingBuffer
from robosuite.utils.observables import Observable, sensor
import pdb

class LwrSingleArm(ManipulatorMe):

    def __init__(
        self,
        robot_type: str,
        idn=0,
        controller_config=None,
        initial_qpos=None,
        initialization_noise=None,
        control_freq=20,
    ):

        self.controller = None
        self.controller_config = copy.deepcopy(controller_config)
        self.gripper = None  # Gripper class

        self.eef_rot_offset = None  # rotation offsets from final arm link to gripper (quat)
        self.eef_site_id = None  # xml element id for eef in mjsim
        self.eef_cylinder_id = None  # xml element id for eef cylinder in mjsim
        self.torques = None  # Current torques being applied

        self.recent_ee_forcetorques = None  # Current and last forces / torques sensed at eef
        self.recent_ee_pose = None  # Current and last eef pose (pos + ori (quat))
        self.recent_ee_vel = None  # Current and last eef velocity
        self.recent_ee_vel_buffer = None  # RingBuffer holding prior 10 values of velocity values
        self.recent_ee_acc = None  # Current and last eef acceleration


        super().__init__(
            robot_type=robot_type,
            idn=idn,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            control_freq=control_freq,
        )

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        if not self.controller_config:
            # Need to update default for a single agent
            controller_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "controllers/config/{}.json".format(self.robot_model.default_controller_config),
            )
            self.controller_config = load_controller_config(custom_fpath=controller_path)
        assert (
            type(self.controller_config) == dict
        ), "Inputted controller config must be a dict! Instead, got type: {}".format(type(self.controller_config))

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self.controller_config["robot_name"] = self.name
        self.controller_config["sim"] = self.sim
        self.controller_config["eef_name"] = self.robot_model.important_sites["ft_frame"]
        self.controller_config["eef_rot_offset"] = None
        self.controller_config["joint_indexes"] = {
            "joints": self.joint_indexes[0:7],
            "qpos": self._ref_joint_pos_indexes[0:7],
            "qvel": self._ref_joint_vel_indexes[0:7],
        }
        self.controller_config["actuator_range"] = self.torque_limits   ###???ONLY ROBOT LIMIT OR ADD GRIPPER LIMITS
        self.controller_config["policy_freq"] = self.control_freq
        self.controller_config["ndim"] = len(self.robot_joints[0:7])

        # Instantiate the relevant controller
        self.controller = controller_factory(self.controller_config["type"], self.controller_config)

        

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super().load_model()

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides gripper joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim
        """
        # First, run the superclass method to reset the position and controller
        super().reset(deterministic)

        if not deterministic:           
            self.sim.data.qpos[7:9] = self.robot_model.init_qpos[7:9]

        # Update base pos / ori references in controller
        self.controller.update_base_pose(self.base_pos, self.base_ori)

        # # Setup buffers to hold recent values
        self.recent_ee_forcetorques = DeltaBuffer(dim=6)
        self.recent_ee_pose = DeltaBuffer(dim=7)
        self.recent_ee_vel = DeltaBuffer(dim=6)
        self.recent_ee_vel_buffer = RingBuffer(dim=6, length=10)
        self.recent_ee_acc = DeltaBuffer(dim=6)
    
    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.

        Note that this should get called during every reset from the environment
        """
        super().setup_references()

        # IDs of sites for eef visualization
        self.grip_site_id = self.sim.model.site_name2id(self.robot_model.important_sites["grip_site"])
        
        self.eef_site_id = self.sim.model.site_name2id(self.robot_model.important_sites["ft_frame"])
        self.eef_cylinder_id = self.sim.model.site_name2id(self.robot_model.important_sites["grip_site_cylinder"])
    
    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (np.array): The control to apply to the robot. The first @self.robot_model.dof dimensions should be
                the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof
                dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """

        # clip actions into valid range
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )

       
        gripper_action = action[self.controller.control_dim:]  # all indexes past controller dimension indexes
        arm_action = action[: self.controller.control_dim]


        # Update the controller goal if this is a new policy step
        if policy_step:
            self.controller.set_goal(arm_action)

        # Now run the controller for a step
        torques = self.controller.run_controller()

        # Clip the torques
        low, high = self.torque_limits
     
        self.torques = np.clip(torques, low, high)

        # Get gripper action, if applicable
        self.grip_action(robot=self.robot_model, gripper_action=gripper_action)

        # Apply joint torque control
        self.sim.data.ctrl[self._ref_joint_actuator_indexes[:7]] = self.torques   ##Note:only for the arm

        #print(arm_action,self._joint_positions[:7])

        # If this is a policy step, also update buffers holding recent values of interest
        if policy_step:
            # Update proprioceptive values
            self.recent_qpos.push(self._joint_positions[:7])   ##Note:only the arm position
            self.recent_actions.push(action)
            self.recent_torques.push(self.torques)
            self.recent_ee_forcetorques.push(np.concatenate((self.ee_force, self.ee_torque)))
            self.recent_ee_pose.push(np.concatenate((self.controller.ee_pos, T.mat2quat(self.controller.ee_ori_mat))))
            self.recent_ee_vel.push(np.concatenate((self.controller.ee_pos_vel, self.controller.ee_ori_vel)))

            # Estimation of eef acceleration (averaged derivative of recent velocities)
            self.recent_ee_vel_buffer.push(np.concatenate((self.controller.ee_pos_vel, self.controller.ee_ori_vel)))
            diffs = np.vstack(
                [self.recent_ee_acc.current, self.control_freq * np.diff(self.recent_ee_vel_buffer.buf, axis=0)]
            )
            ee_acc = np.array([np.convolve(col, np.ones(10) / 10.0, mode="valid")[0] for col in diffs.transpose()])
            self.recent_ee_acc.push(ee_acc)
    

################################################################################

###############################################################################
    def setup_observables(self):
        """
        Sets up observables to be used for this robot

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # Get general robot observables first
        observables = super().setup_observables()
        
        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        pf = self.robot_model.naming_prefix
        modality = f"{pf}proprio"

        # eef features
        @sensor(modality=modality)
        def eef_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.eef_site_id])

        @sensor(modality=modality)
        def eef_quat(obs_cache):
            return T.convert_quat(self.sim.data.get_body_xquat('robot0_eef_gripper'), to="xyzw")
                
        @sensor(modality=modality)
        def eef_force(obs_cache):
            return np.array(self.sim.data.sensordata[0:3])
        
        @sensor(modality=modality)
        def eef_torque(obs_cache):
            return np.array(self.sim.data.sensordata[3:])
        
        @sensor(modality=modality)
        def eef_velocity(obs_cache):
            return self._hand_vel

        sensors = [eef_pos, eef_quat, eef_force, eef_torque, eef_velocity]
        names = [f"{pf}eef_pos", f"{pf}eef_quat", f"{pf}eef_force", f"{pf}eef_torque",f"{pf}eef_velocity"]

        # Create observables for this robot
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables
    
    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        # Action limits based on controller limits
        low, high = ([-1] * 1, [1] * 1)
        low_c, high_c = self.controller.control_limits
        low = np.concatenate([low_c, low])
        high = np.concatenate([high_c, high])

        return low, high
    
    @property
    def ee_ft_integral(self):
        return np.abs((1.0 / self.control_freq) * self.recent_ee_forcetorques.average)

    @property
    def ee_force(self):
        return self.get_sensor_measurement(self.robot_model.important_sensors["force_ee"])

    @property
    def ee_torque(self):
        return self.get_sensor_measurement(self.robot_model.important_sensors["torque_ee"])

    @property
    def _hand_pose(self):
        """
        Returns:
            np.array: (4,4) array corresponding to the eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name(self.robot_model.eef_name)

    @property
    def _hand_quat(self):
        """
        Returns:
            np.array: (x,y,z,w) eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._hand_orn)

    @property
    def _hand_total_velocity(self):
        """
        Returns:
            np.array: 6-array representing the total eef velocity (linear + angular) in the base frame
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp('robot0_eef_gripper').reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes[:7]]    ###:Note use the index of arm

        Jr = self.sim.data.get_body_jacr('robot0_eef_gripper').reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes[:7]]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities[:7])
        eef_rot_vel = Jr_joint.dot(self._joint_velocities[:7])
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _hand_pos(self):
        """
        Returns:
            np.array: 3-array representing the position of eef in base frame of robot.
        """
        eef_pose_in_base = self._hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _hand_orn(self):
        """
        Returns:
            np.array: (3,3) array representing the orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _hand_vel(self):
        """
        Returns:
            np.array: (x,y,z) velocity of eef in base frame of robot.
        """
        return self._hand_total_velocity[:3]

    @property
    def _hand_ang_vel(self):
        """
        Returns:
            np.array: (ax,ay,az) angular velocity of eef in base frame of robot.
        """
        return self._hand_total_velocity[3:]