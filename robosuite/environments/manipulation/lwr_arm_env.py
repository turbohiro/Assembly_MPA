import numpy as np

from robosuite.environments.manipulation.manipulation_env_me import ManipulationEnvMe
from robosuite.robots import LwrSingleArm
from robosuite.utils.transform_utils import mat2quat

class LwrArmEnv(ManipulationEnvMe):
    """
    A manipulation environment intended for lwr robot arm.
    """
    
    def _load_model(self):
        """
        Verifies correct robot model is loaded
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(
            self.robots[0], LwrSingleArm
        ), "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))
    
    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        super()._check_robot_configuration(robots)
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"

    @property
    def _eef_xpos(self):
        """
        Grabs End Effector position

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    
    @property
    def _eef_xmat(self):
        """
        End Effector orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (3,3) End Effector orientation matrix
        """
        pf = self.robots[0].robot_model.naming_prefix
        #return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "ee")]).reshape(3, 3)
        return np.array(self.sim.data.site_xmat[self.robots[0].eef_site_id]).reshape(3, 3)
    @property
    def _eef_xquat(self):
        """
        End Effector orientation as a (x,y,z,w) quaternion
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (x,y,z,w) End Effector quaternion
        """
        return mat2quat(self._eef_xmat)