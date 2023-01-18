"""
Tests panda gripper on grabbing task
"""
from robosuite.models.grippers import GripperTester, Wsg50Gripper


def test_wsg_gripper():
    wsg_gripper_tester(False)


def wsg_gripper_tester(render, total_iters=1, test_y=True):
    gripper = Wsg50Gripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0.01 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=0.010,
        gripper_high_pos=0.1,
        render=render,
    )
    tester.start_simulation()
    tester.loop(total_iters=total_iters, test_y=test_y)
    tester.close()


if __name__ == "__main__":
    wsg_gripper_tester(True, 20, True)
    wsg_gripper_tester(True, 20, True)
