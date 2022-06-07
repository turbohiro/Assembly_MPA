"""
    Robosuite environment for the peg in hole task.  Using trajectory optimization
    to define keyframes to move through in order to complete task.
    NOTE: Example does not achieve successful results, as we ran into bugs with
    the absolute position controller 'OSC_POSE' in the robosuite environment
"""
import gym
import sys
import numpy as np
import robosuite as suite
import os

from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
import imageio

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
import robosuite.utils.transform_utils as T


"""
[0] JOINT_VELOCITY - Joint Velocity
[1] JOINT_TORQUE - Joint Torque
[2] JOINT_POSITION - Joint Position
[3] OSC_POSITION - Operational Space Control (Position Only)
[4] OSC_POSE - Operational Space Control (Position + Orientation)
[5] IK_POSE - Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)
"""


class keypointOptPegInHole(gym.Env):

    def __init__(self, *args, **kwargs):
        options = {}
        # Absolute pose control
        controller_name = 'OSC_POSE'
        options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

        self.cameraName = "frontview"
        skip_frame = 2
        self.peg_env = suite.make(
            "TwoArmPegInHole",
            robots=["IIWA","IIWA"],
            **options,
            has_renderer=False,
            ignore_done=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=self.cameraName,
            camera_heights=512,
            camera_widths=512,
        )

        # Tolerances on position and rotation of peg relative to hole
        posTol = 0.005
        rotTol = 0.05

        observation = self.peg_env.reset()

        # observation["peg_to_hole"] is the vector FROM the hole TO the peg
        peg_pos0 = observation["hole_pos"] + observation["peg_to_hole"]
        self.peg_pos0 = peg_pos0
        self.hole_pos0 = observation["hole_pos"]

        # Positions of robots 0 and 1 rel peg and hole, in peg and hole frames.  Constant forever.
        pRob0RelPeg = np.matmul(T.quat2mat(T.quat_inverse(observation["peg_quat"])) ,  observation["robot0_eef_pos"] - (peg_pos0))
        pRob1RelHole = np.matmul(T.quat2mat(T.quat_inverse(observation["hole_quat"])) ,  observation["robot1_eef_pos"] - observation["hole_pos"])
        qRob0RelPeg = T.quat_multiply(T.quat_inverse(observation["robot0_eef_quat"]),observation["peg_quat"])
        qRob1RelHole = T.quat_multiply(T.quat_inverse(observation["robot1_eef_quat"]),observation["hole_quat"])

        # Store geometric constants
        model = self.peg_env.model.get_model()
        pegDim = model.geom_size[15]
        rPeg = pegDim[0]
        lPeg = pegDim[2]
        distSlack = 2*lPeg

        # Set up optimization problem.
        # Define 3 keyframes: peg higher than hole, peg centered above hole with hole facing up, and peg in hole.
        # One constraint for each keyframe, and one constraint for unit quaternions
        nonlinear_constraint_1 = NonlinearConstraint(self.cons_1, distSlack, np.inf, jac='2-point', hess=BFGS())
        nonlinear_constraint_2 = NonlinearConstraint(self.cons_2, np.array([-posTol, -posTol, distSlack,-rotTol]), np.array([posTol,posTol,np.inf,rotTol]), jac='2-point', hess=BFGS())
        nonlinear_constraint_3 = NonlinearConstraint(self.cons_3, np.array([-posTol, -posTol, distSlack]), np.array([posTol,posTol,np.inf]), jac='2-point', hess=BFGS())
        nonlinear_constraint_4 = NonlinearConstraint(self.cons_unit_quat, np.array([1,1,1,1]), np.array([1,1,1,1]), jac='2-point', hess=BFGS())

        # Initial guess for optimizer
        x0 = np.tile(np.hstack((peg_pos0,observation["hole_pos"],observation["peg_quat"],observation["hole_quat"])),3)
        # Cut out quat from last chunk
        x0 = x0[0:34]

        # Solve optimization problem
        res = minimize(self.traj_obj, x0, method='trust-constr', jac='2-point', hess=BFGS(),
                   constraints=[nonlinear_constraint_1, nonlinear_constraint_2,nonlinear_constraint_3,nonlinear_constraint_4],
                   options={'verbose': 1})
                   # 'xtol': 1e-6,
        x = res.x
        # Extract optimization results
        # x = [p_peg_1, p_hole_1, q_peg_1, q_hole_1, p_peg_2, ... q_peg_3, q_hole_3]
        ind_offset_1 = 14
        ind_offset_2 = 28

        p_peg_1 = x[0:3]
        p_hole_1 = x[3:6]
        p_peg_2 = x[ind_offset_1 + 0:ind_offset_1 + 3]
        p_hole_2 = x[ind_offset_1 + 3:ind_offset_1 + 6]
        p_peg_3 = x[ind_offset_2 + 0:ind_offset_2 + 3]
        p_hole_3 = x[ind_offset_2 + 3:ind_offset_2 + 6]

        q_peg_1 = x[6:10]
        q_hole_1 = x[10:14]
        q_peg_2 = x[ind_offset_1 + 6:ind_offset_1 + 10]
        q_hole_2 = x[ind_offset_1 + 10:ind_offset_1 + 14]

        # Use the same orientations as in pose 2
        q_peg_3 = q_peg_2
        q_hole_3 = q_hole_2

        # Robot rel world = peg rel world + robot rel peg
        # Robot rel peg in world frame = (q world frame rel peg frame) * (robot rel peg in peg frame)
        q_rob0_1 = T.quat_inverse(T.quat_multiply(qRob0RelPeg,T.quat_inverse(q_peg_1)))
        q_rob1_1 = T.quat_inverse(T.quat_multiply(qRob1RelHole,T.quat_inverse(q_hole_1)))
        q_rob0_2 = T.quat_inverse(T.quat_multiply(qRob0RelPeg,T.quat_inverse(q_peg_2)))
        q_rob1_2 = T.quat_inverse(T.quat_multiply(qRob1RelHole,T.quat_inverse(q_hole_2)))
        q_rob0_3 = T.quat_inverse(T.quat_multiply(qRob0RelPeg,T.quat_inverse(q_peg_3)))
        q_rob1_3 = T.quat_inverse(T.quat_multiply(qRob1RelHole,T.quat_inverse(q_hole_3)))

        self.p_rob0_1 = p_peg_1 + np.matmul(T.quat2mat(q_peg_1),pRob0RelPeg)
        self.p_rob1_1 = p_hole_1 + np.matmul(T.quat2mat(q_hole_1),pRob1RelHole)
        self.p_rob0_2 = p_peg_2 + np.matmul(T.quat2mat(q_peg_2),pRob0RelPeg)
        self.p_rob1_2 = p_hole_2 + np.matmul(T.quat2mat(q_hole_2),pRob1RelHole)
        self.p_rob0_3 = p_peg_3 + np.matmul(T.quat2mat(q_peg_3),pRob0RelPeg)
        self.p_rob1_3 = p_hole_3 + np.matmul(T.quat2mat(q_hole_3),pRob1RelHole)

        self.axang_rob0_1 = T.quat2axisangle(q_rob0_1)
        self.axang_rob1_1 = T.quat2axisangle(q_rob1_1)
        self.axang_rob0_2 = T.quat2axisangle(q_rob0_2)
        self.axang_rob1_2 = T.quat2axisangle(q_rob1_2)
        self.axang_rob0_3 = T.quat2axisangle(q_rob0_3)
        self.axang_rob1_3 = T.quat2axisangle(q_rob1_3)


        self.max_episode_steps = 200
        # Gains for rotation and position error compensation rates
        self.kpp = 4
        self.kpr = 0.1

        # Initial pose Information
        self.rob0quat0 = observation["robot0_eef_quat"]
        self.rob1quat0 = observation["robot1_eef_quat"]
        self.rob0pos0 = observation["robot0_eef_pos"]
        self.rob1pos0 = observation["robot1_eef_pos"]

    def reset(self):
        self.poseNum = 0
    def controller(self,obs:dict):
        # Tolerances on reaching keyframes
        posePosTol = 0.01
        poseAxangTol = 0.09

        # Which keyframe are we in?
        if self.poseNum == 0:
            rob0GoalPos = self.p_rob0_1
            rob1GoalPos = self.p_rob1_1
            rob0GoalAxAng = self.axang_rob0_1
            rob1GoalAxAng = self.axang_rob1_1
        elif self.poseNum == 1:
            rob0GoalPos = self.p_rob0_2
            rob1GoalPos = self.p_rob1_2
            rob0GoalAxAng = self.axang_rob0_2
            rob1GoalAxAng = self.axang_rob1_2
        elif self.poseNum >= 2:
            rob0GoalPos = self.p_rob0_3
            rob1GoalPos = self.p_rob1_3
            rob0GoalAxAng = self.axang_rob0_3
            rob1GoalAxAng = self.axang_rob1_3

        rob0OrientationError = T.get_orientation_error(obs["robot0_eef_quat"],T.axisangle2quat(rob0GoalAxAng))
        rob1OrientationError = T.get_orientation_error(obs["robot1_eef_quat"],T.axisangle2quat(rob1GoalAxAng))

        # Absolute pose control, so actions are just the poses we want to get to
        posActionRob0 = self.p_rob0_1
        posActionRob1 = self.p_rob1_1
        axangActionRob0 = self.axang_rob0_1
        axangActionRob1 = self.axang_rob1_1

        # If we have reached the next keyframe, increment pose counter
        if np.linalg.norm(posActionRob0/self.kpp)<posePosTol and np.linalg.norm(rob0OrientationError)<poseAxangTol and np.linalg.norm(posActionRob1/self.kpp)<posePosTol and np.linalg.norm(rob1OrientationError)<poseAxangTol:
            if self.poseNum<=1:
                self.poseNum = self.poseNum + 1
            elif self.poseNum == 2:
                print("done!")
                self.poseNum = 3

        return np.hstack((posActionRob0,axangActionRob0,posActionRob1,axangActionRob1)).tolist()


    def cons_1(self,x):
        # First pose: constrain peg further than lPeg in z direction relative to hole
        p_peg = x[0:3]
        p_hole = x[3:6]
        q_hole = x[10:14]
        p_peg_in_hole_frame = np.matmul(T.quat2mat(T.quat_inverse(q_hole)),p_peg - p_hole)

        return p_peg_in_hole_frame[2]

    def cons_2(self,x):
        # Second  pose: constrain peg further than lPeg in z direction relative to hole
        # also constrain peg x and y in hole frame to be below a tolerance
        # also constrain rotation error
        ind_offset = 14 # ind at which to start looking for pose 2 info
        p_peg = x[ind_offset + 0:ind_offset + 3]
        p_hole = x[ind_offset + 3:ind_offset + 6]
        q_peg = x[ind_offset + 6:ind_offset + 10]
        q_hole = x[ind_offset + 10:ind_offset + 14]
        p_peg_in_hole_frame = np.matmul(T.quat2mat(T.quat_inverse(q_hole)),p_peg - p_hole)

        z_hole = np.matmul(T.quat2mat(q_hole),np.array([0,0,1])) # hole z in world frame
        z_peg = np.matmul(T.quat2mat(q_peg),np.array([0,0,1])) # peg z in world frames

        # Want the z axes to be antiparallel
        z_defect = np.linalg.norm(z_hole + z_peg)

        return np.hstack((p_peg_in_hole_frame,z_defect))

    def cons_3(self,x):
        # Third  pose: constrain peg less than lPeg/2 in z direction relative to hole
        # also constrain peg x and y in hole frame to be below a tolerance
        # also constrain same orientations as in pose 2
        last_ind_offset = 14 # ind at which to start looking for pose 2 info
        ind_offset = 28 # ind at which to start looking for pose 3 info
        p_peg = x[ind_offset + 0:ind_offset + 3]
        p_hole = x[ind_offset + 3:ind_offset + 6]
        # Using the quats from pose 2 because assuming they will stay the same
        q_hole = x[last_ind_offset + 10:ind_offset + 14]
        p_peg_in_hole_frame = np.matmul(T.quat2mat(T.quat_inverse(q_hole)),p_peg - p_hole)

        return np.hstack(p_peg_in_hole_frame)

    def cons_unit_quat(self,x):
        # Constrain quaternions to be unit
        ind_offset_1 = 14
        ind_offset_2 = 28

        q_peg_1 = x[6:10]
        q_hole_1 = x[10:14]
        q_peg_2 = x[ind_offset_1 + 6:ind_offset_1 + 10]
        q_hole_2 = x[ind_offset_1 + 10:ind_offset_1 + 14]

        return np.array([np.linalg.norm(q_peg_1), np.linalg.norm(q_hole_1), np.linalg.norm(q_peg_2), np.linalg.norm(q_hole_2)])

    def traj_obj(self,x):
        # Cost on motion between adjacent poses

        peg_pos0 = self.peg_pos0
        hole_pos0 = self.hole_pos0
        ind_offset_1 = 14
        ind_offset_2 = 28
        p_peg_1 = x[0:3]
        p_hole_1 = x[3:6]
        p_peg_2 = x[ind_offset_1 + 0:ind_offset_1 + 3]
        p_hole_2 = x[ind_offset_1 + 3:ind_offset_1 + 6]
        p_peg_3 = x[ind_offset_2 + 0:ind_offset_2 + 3]
        p_hole_3 = x[ind_offset_2 + 3:ind_offset_2 + 6]

        return np.linalg.norm(p_peg_1 - peg_pos0) + np.linalg.norm(p_peg_2 - p_peg_1) + np.linalg.norm(p_peg_3 - p_peg_2) + np.linalg.norm(p_hole_1 - hole_pos0) + np.linalg.norm(p_hole_2 - p_hole_1) + np.linalg.norm(p_hole_3 - p_hole_2)

if __name__ == "__main__":
    env_name = 'keypointOptPegInHole'
    env = globals()[env_name]()
    successes = []

    failed_eps = []
    # Save video of simulation
    writer = imageio.get_writer('video/TwoArmPegInHole.mp4', fps=20)
    frames = []
    skip_frame = 2
    for i_episode in range(1):
        success = np.zeros(env.max_episode_steps)
        obs = env.reset()
        action = ([0,0,0,0,0,0],[0,0,0,0,0,0])  # Give zero action at first time step
        for t in range(100):
            action = env.controller(env.peg_env._get_observation())
            observation, reward, done, info = env.peg_env.step(action)
            if t % skip_frame == 0:
                frame = observation[env.cameraName + "_image"][::-1]
                writer.append_data(frame)

            if reward == 1:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.peg_env.close()
    writer.close()
