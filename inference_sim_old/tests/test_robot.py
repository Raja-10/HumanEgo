"""
test_robot.py

Test RobosuiteRobot wrapper.

Checks:
1. Read current EE pose
2. Command an absolute target
3. Print position/orientation error
4. Verify convergence
"""

import time
import numpy as np
import robosuite as suite
from scipy.spatial.transform import Rotation

from inference_sim.robot import RobosuiteRobot


# -------------------------------------------------------
# Environment
# -------------------------------------------------------

env = suite.make(
    env_name="ServeBread",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

env.reset()

robot = RobosuiteRobot(env)

# -------------------------------------------------------
# Current Pose
# -------------------------------------------------------

T_start = robot.get_T_ee()

print("\n==============================")
print("Initial EE Pose")
print("==============================")
print(T_start)

# -------------------------------------------------------
# Target Pose
# -------------------------------------------------------

T_goal = T_start.copy()

# Move 5 cm upward (BASE FRAME)
T_goal[2, 3] += 0.05

print("\n==============================")
print("Goal Pose")
print("==============================")
print(T_goal)

# -------------------------------------------------------
# Servo Loop
# -------------------------------------------------------

print("\nMoving...\n")

for i in range(150):

    robot.move_ee(T_goal)

    env.render()

    if i % 10 == 0:

        T_now = robot.get_T_ee()

        pos_err = T_goal[:3, 3] - T_now[:3, 3]

        R_err = (
            Rotation.from_matrix(
                T_goal[:3, :3] @ T_now[:3, :3].T
            )
            .as_rotvec()
        )

        print("=" * 60)
        print(f"Iteration {i}")

        print("\nCurrent Position")
        print(T_now[:3, 3])

        print("\nTarget Position")
        print(T_goal[:3, 3])

        print("\nPosition Error (m)")
        print(pos_err)
        print("Norm :", np.linalg.norm(pos_err))

        print("\nOrientation Error (rotvec)")
        print(R_err)
        print("Norm :", np.linalg.norm(R_err))

# -------------------------------------------------------
# Final Pose
# -------------------------------------------------------

T_final = robot.get_T_ee()

print("\n==============================")
print("Final Pose")
print("==============================")
print(T_final)

print("\nFinal Position Error")
print(T_goal[:3, 3] - T_final[:3, 3])

print("\nFinal Orientation Error")

R_err = Rotation.from_matrix(
    T_goal[:3, :3] @ T_final[:3, :3].T
).as_rotvec()

print(R_err)

# -------------------------------------------------------
# Gripper Test
# -------------------------------------------------------

print("\nOpening Gripper")

robot.open_gripper()

for _ in range(20):
    robot.move_ee(T_goal)
    env.render()

print("\nClosing Gripper")

robot.close_gripper()

for _ in range(20):
    robot.move_ee(T_goal)
    env.render()

env.close()