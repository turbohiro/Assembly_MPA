## Contents of this folder:

This folder contains all fetch the environments used in our experiments.
* `pushEnv.py` contains three environments:
    * `FetchPush`: Vanilla FetchPush
    * `FetchPushImperfect`: FetchPush with an imperfect controller
    * `FetchPushSlippery`: FetchPush with an imperfect controller and friction coefficient changed from 1 to 0.1
* `slideEnv.py` contains three environments:
    * `FetchSlide`: Vanilla FetchSlide
    * `FetchSlideFrictionControl`: FetchSlide with a friction-based controller as described in the report
    * `FetchSlideSlapControl`: FetchSlide with a slap controller as described in the report
* `pickAndPlaceEnv.py` contains three environments:
    * `FetchPickAndPlace`: Vanilla FetchPickAndPlace
    * `FetchPickAndPlacePerfect`: FetchPickAndPlace with a perfect controller
    * `FetchPickAndPlaceSticky`: FetchPickAndPlace with the same controller but takes same action as previous one with probability 0.5
* `basic_controller.py`: Contains our initial stab at controlling fetchEnvs. Contains the docs for state space and action space for fetchEnvs.
* `customGoalExample.py`: An example to change the goal of the environment manually.

The environments can be initialised just like normal classes after importing them.
