"""
Gripper with 140mm Jaw width from Robotiq (has two fingers).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Wsg50GripperBase(GripperModel):
    """
    Gripper with 50mm Jaw width from WSG (has two fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance

    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/wsg_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([-0.0, 0.0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_finger_collision",
                "left_finger_pad",
            ],
            "right_finger": [
                "right_finger_collision",
                "right_finger_pad",
            ],
        }


class Wsg50Gripper(Wsg50GripperBase):
    """
    Modifies Wsg50GripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.1

    @property
    def dof(self):
        return 1
