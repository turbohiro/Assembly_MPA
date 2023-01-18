"""Gripper interaction demo.
This script illustrates the process of importing grippers into a scene and making it interact
with the objects with actuators. It also shows how to procedurally generate a scene with the
APIs of the MJCF utility functions.
Example:
    $ python run_gripper_test.py
"""

import xml.etree.ElementTree as ET

from mujoco_py import MjSim, MjViewer

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.assembly_arena import AssemblyArena
from robosuite.models.robots import LWR
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_actuator, new_joint
import numpy as np
import pdb

def _apply_gripper_action(gripper, action):
    """
    Applies binary gripper action

    Args:
        action (int): Action to apply. Should be -1 (open) or 1 (closed)
    """
    actuator_ids = [sim.model.actuator_name2id(x) for x in gripper.actuators]
    gripper_actuator_ids = actuator_ids[7:]
    gripper_action_actual = gripper.format_action(np.array([action]))
    # rescale normalized gripper action to control ranges
    ctrl_range = sim.model.actuator_ctrlrange[gripper_actuator_ids]
    bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
    weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
    applied_gripper_action = bias + weight * gripper_action_actual
    sim.data.ctrl[gripper_actuator_ids] = applied_gripper_action

if __name__ == "__main__":

    # start with an empty world
    world = MujocoWorldBase()

    # add a table
    arena = AssemblyArena()
    world.merge(arena)

    # add a gripper
    gripper = LWR()
    # Create another body with a slider joint to which we'll add this gripper
    #gripper_body = ET.Element("body", name="gripper_base")
    #gripper_body.set("pos", "0.01 0 0.3")
    #gripper_body.set("quat", "0 0 1 0")  # flip z
    #gripper_body.append(new_joint(name="gripper_z_joint", type="slide", axis="0 0 1", damping="30"))
    # Add the dummy body with the joint to the global worldbody
   # world.worldbody.append(gripper_body)
    # Merge the actual gripper as a child of the dummy body
    world.merge(gripper)
    # Create a new actuator to control our slider joint
    #world.actuator.append(new_actuator(joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"))

    # add an object for grasping
    mujoco_object = BoxObject(
        name="box", size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], friction=[1, 0.005, 0.0001]
    ).get_obj()
    # Set the position of this object
    mujoco_object.set("pos", "0 0 0.9")
    # Add our object to the world body
    world.worldbody.append(mujoco_object)

    # add reference objects for x and y axes
    x_ref = BoxObject(
        name="x_ref", size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 1], obj_type="visual", joints=None
    ).get_obj()
    x_ref.set("pos", "0.2 0 1.105")
    world.worldbody.append(x_ref)
    y_ref = BoxObject(
        name="y_ref", size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1], obj_type="visual", joints=None
    ).get_obj()
    y_ref.set("pos", "0 0.2 1.105")
    world.worldbody.append(y_ref)

    # start simulation
    model = world.get_model(mode="mujoco_py")

    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim_state = sim.get_state()

    # for gravity correction
    #gravity_corrected = ["gripper_z_joint"]
    #_ref_joint_vel_indexes = [sim.model.get_joint_qvel_addr(x) for x in gravity_corrected]

    # Set gripper parameters
    #gripper_z_id = sim.model.actuator_name2id("gripper_z")
    #gripper_z_low = 0.07
    #gripper_z_high = -0.01
    #gripper_z_is_low = False

    gripper_jaw_ids = [sim.model.actuator_name2id(x) for x in gripper.actuators]
    gripper_jaw_ids = gripper_jaw_ids[7:]
    gripper_open = [-0.04, 0.04]
    gripper_closed = [-0.001, 0.001]
    gripper_is_closed = True

    # hardcode sequence for gripper looping trajectory
    seq = [(False, False), (True, False), (True, True), (False, True)]
    seq2 = [(False), (True)]
    sim.set_state(sim_state)
    step = 0
    T = 500
    
    #close the gripper
    while True:

        if step % T == 0:
            plan = seq2[int(step / T) % len(seq2)]
            gripper_is_closed = plan
        
        if gripper_is_closed:
            sim.data.ctrl[gripper_jaw_ids] = gripper_closed
        else:
            sim.data.ctrl[gripper_jaw_ids] = gripper_open

         # Step through sim
        sim.step()
        #sim.data.qfrc_applied[_ref_joint_vel_indexes] = sim.data.qfrc_bias[_ref_joint_vel_indexes]
        viewer.render()
        step += 1
        viewer.render()
    
    while False:
        if step % 100 == 0:
            print("step: {}".format(step))

            # Get contact information
            for contact in sim.data.contact[0 : sim.data.ncon]:

                geom_name1 = sim.model.geom_id2name(contact.geom1)
                geom_name2 = sim.model.geom_id2name(contact.geom2)
                if geom_name1 == "floor" and geom_name2 == "floor":
                    continue
                geom2_body = sim.model.geom_bodyid[contact.geom2]
                #print(' Contact force on geom2 body', sim.data.cfrc_int[geom2_body])
                print('sensor data force is:',sim.data.sensordata)

                print("geom1: {}, geom2: {}".format(geom_name1, geom_name2))
                print('catact',contact)
                print("contact id {}".format(id(contact)))
                print("friction: {}".format(contact.friction))
                print("normal: {}".format(contact.frame[0:3]))

        # Iterate through gripping trajectory
        if step % T == 0:
            plan = seq[int(step / T) % len(seq)]
            gripper_z_is_low, gripper_is_closed = plan
            print("changing plan: gripper low: {}, gripper closed {}".format(gripper_z_is_low, gripper_is_closed))

        # Control gripper
        if gripper_z_is_low:
            sim.data.ctrl[gripper_z_id] = gripper_z_low
        else:
            sim.data.ctrl[gripper_z_id] = gripper_z_high
        if gripper_is_closed:
            sim.data.ctrl[gripper_jaw_ids] = gripper_closed
        else:
            sim.data.ctrl[gripper_jaw_ids] = gripper_open

        # Step through sim
        sim.step()
        sim.data.qfrc_applied[_ref_joint_vel_indexes] = sim.data.qfrc_bias[_ref_joint_vel_indexes]
        viewer.render()
        step += 1