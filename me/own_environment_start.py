
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:57:02 2022

@author: wchen
"""
from robosuite.models import MujocoWorldBase

world = MujocoWorldBase()   #1. create the mujoco  world

from robosuite.models.robots import Panda

mujoco_robot = Panda()     #2. create the robot

from robosuite.models.grippers import gripper_factory

gripper = gripper_factory('PandaGripper')
mujoco_robot.add_gripper(gripper)  ##3. add a gripper to the robot

mujoco_robot.set_base_xpos([-0.8,0,1.0])  ##4. merge th object to the world
world.merge(mujoco_robot)

from robosuite.models.arenas import WipeArena

mujoco_arena = WipeArena(table_full_size=(2.0, 2.0, 0.01),table_offset=(0, 0, 1.0))
mujoco_arena.set_origin([0,0,0])
world.merge(mujoco_arena)         ##5. create and merge a table area to the world

from robosuite.models.objects import HingedBoxObject,SquareNutObject,CanObject

from robosuite.utils.mjcf_utils import new_joint

#sphere = HammerObject(name = 'hammer',size = [0.04], rgba = [0,0.5,0.5,1]).get_obj()
#sphere = CanObject(name = 'can').get_obj()
#sphere.set('pos','0.7 0 2.0')
#world.worldbody.append(sphere)     ##6. create and add an object to the world


model = world.get_model(mode='mujoco_py')   ##7. finally get the MjModel instance


from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(1000000):
    sim.data.ctrl[:] = 0
    sim.step()
    viewer.render()