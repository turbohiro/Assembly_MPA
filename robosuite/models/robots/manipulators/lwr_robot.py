import numpy as np

from robosuite.models.robots.manipulators.manipulator_model_me import ManipulatorModelMe
from robosuite.utils.mjcf_utils import xml_path_completion


class LWR(ManipulatorModelMe):
    """
    LWR is a bright and spunky robot created by KUKA

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/lwr/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "nullMount"

    @property
    def default_gripper(self):
        return "WSG50Gripper"

    @property
    def default_controller_config(self):
        return "default_lwr"

    @property
    def init_qpos(self):
        #return np.array([0.000, 0.650, 0.000, -1.89, 0.000, 0.60, 0.000, -0.03, 0.03])  #-1.89
        return np.array([0.0,  0.44512, 0, -1.866547, 0,  0.8299238, 0, -0.025, 0.025   ])
        #return np.array([0.00487328,  0.43207766,  0.00637799, -1.88706874,  0.02917416, 0.79099126,  0.03468684,  -0.025, 0.025    ])
    
    @property
    def _important_geoms(self):
        return {
            "left_fingerpad": ["left_finger_pad"],
            "right_fingerpad": ["right_finger_pad"],
        }

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
    
    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed
        """
        speed = 0.5
        self.current_action = np.clip(self.current_action + np.array([-1.0, 1.0]) * speed * np.sign(action), -1.0, 1.0 )
        return self.current_action
