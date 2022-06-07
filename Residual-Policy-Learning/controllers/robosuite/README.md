## Contents of this folder:

This folder contains all the robosuite environments used in our experiments.
* `robosuitePickAndPlace.py`: First stab at controlling envs in robosuite for pick and place task.
* `robosuiteDoorOpening.py`: First stab at controlling envs in robosuite for door opening task.
* `robosuiteNutAssembly.py`: Contains the controller and vanilla env for nut assembly task. This one has sparse rewards.
* `robosuiteNutAssemblyDense.py`: Contains the controller and vanilla env for nut assembly task. This one has dense rewards. (Use with SAC)
* `robosuitePegInHole.py`: Contains the optimisation-based controller for peg-in-hole task. Although this doesn't work as intended due to a bug in robosuite as mentioned in the bugs section in the report.

The environments can be initialised just like normal classes after importing them.
