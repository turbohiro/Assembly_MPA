import numpy as np

from robosuite.environments.robot_env import RobotEnv
from robosuite.models.base import MujocoModel
from robosuite.robots import ROBOT_CLASS_MAPPING, ManipulatorMe


class ManipulationEnvMe(RobotEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        mount_types=None,
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
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
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        # Robot info
        robots = list(robots) if type(robots) is list or type(robots) is tuple else [robots]
        num_robots = len(robots)


        # Robot configurations to pass to super call
        robot_configs = None

        # Run superclass init
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=mount_types,
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
            robot_configs=robot_configs,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        vis_set = super()._visualizations
        return vis_set

    def _check_grasp(self, robot, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.

        By default, this will return True if at least one geom in both the "left_fingerpad" and "right_fingerpad" geom
        groups are in contact with any geom specified by @object_geoms. Custom gripper geom groups can be
        specified with @gripper as well.

        Args:
            gripper (GripperModel or str or list of str or list of list of str): If a MujocoModel, this is specific
            gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms). At least one geom from each group
                must be in contact with any geom in @object_geoms for this method to return True.
            object_geoms (str or list of str or MujocoModel): If a MujocoModel is inputted, will check for any
                collisions with the model's contact_geoms. Otherwise, this should be specific geom name(s) composing
                the object to check for contact.

        Returns:
            bool: True if the gripper is grasping the given object
        """
        # Convert object, gripper geoms into standardized form
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        g_geoms = [robot.important_geoms["left_fingerpad"], robot.important_geoms["right_fingerpad"]]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.check_contact(g_group, o_geoms):
                return False
        return True

    def _gripper_to_target(self, robot, target, target_type="body", return_distance=False):
        """
        Calculates the (x,y,z) Cartesian distance (target_pos - gripper_pos) from the specified @gripper to the
        specified @target. If @return_distance is set, will return the Euclidean (scalar) distance instead.

        Args:
            gripper (MujocoModel): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
            return_distance (bool): If set, will return Euclidean distance instead of Cartesian distance

        Returns:
            np.array or float: (Cartesian or Euclidean) distance from gripper to target
        """
        # Get gripper and target positions
        gripper_pos = self.sim.data.get_site_xpos(robot.important_sites["grip_site"])
        # If target is MujocoModel, grab the correct body as the target and find the target position
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # Calculate distance
        diff = target_pos - gripper_pos
        # Return appropriate value
        return np.linalg.norm(diff) if return_distance else diff

    def _visualize_gripper_to_target(self, robot, target, target_type="body"):
        """
        Colors the grip visualization site proportional to the Euclidean distance to the specified @target.
        Colors go from red --> green as the gripper gets closer.

        Args:
            gripper (MujocoModel): Gripper model to update grip site rgb
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
        """
        # Get gripper and target positions
        gripper_pos = self.sim.data.get_site_xpos(robot.important_sites["grip_site"])
        # If target is MujocoModel, grab the correct body as the target and find the target position
        if isinstance(target, MujocoModel):
            target_pos = self.sim.data.get_body_xpos(target.root_body)
        elif target_type == "body":
            target_pos = self.sim.data.get_body_xpos(target)
        elif target_type == "site":
            target_pos = self.sim.data.get_site_xpos(target)
        else:
            target_pos = self.sim.data.get_geom_xpos(target)
        # color the gripper site appropriately based on (squared) distance to target
        dist = np.sum(np.square((target_pos - gripper_pos)))
        max_dist = 0.1
        scaled = (1.0 - min(dist / max_dist, 1.0)) ** 15
        rgba = np.zeros(3)
        rgba[0] = 1 - scaled
        rgba[1] = scaled
        self.sim.model.site_rgba[self.sim.model.site_name2id(robot.important_sites["grip_site"])][:3] = rgba

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure inputted robots and the corresponding requested task/configuration combo is legal.
        Should be implemented in every specific task module

        Args:
            robots (str or list of str): Inputted requested robots at the task-level environment
        """
        # Make sure all inputted robots are a manipulation robot
        if type(robots) is str:
            robots = [robots]
        for robot in robots:
            assert issubclass(
                ROBOT_CLASS_MAPPING[robot], ManipulatorMe
            ), "Only manipulator robots supported for manipulation environment!"
