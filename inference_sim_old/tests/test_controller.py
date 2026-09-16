"""
test_controller.py

Test absolute OSC controller.

Move the end-effector down by 5 cm and verify convergence.
"""

import numpy as np
from scipy.spatial.transform import Rotation

import robosuite as suite

from inference_sim.robot import RobosuiteRobot
from inference_sim.controller import InferenceController


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

env = suite.make(
    env_name="Lift",
    robots="Nero7",          # Change to Nero7 later
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

env.reset()

robot = RobosuiteRobot(env)
controller = InferenceController(robot)

# ---------------------------------------------------------
# Print controller information
# ---------------------------------------------------------

cc = env.robots[0].composite_controller
osc = cc.get_controller("right")

print("=" * 80)
print("Controller")
print("=" * 80)

print("Input Type :", osc.input_type)
print("Reference  :", osc.input_ref_frame)

origin_pos, origin_ori = cc.get_controller_base_pose("right")

print("\nController Origin")
print(origin_pos)

print("\nController Orientation")
print(origin_ori)

# ---------------------------------------------------------
# Current pose
# ---------------------------------------------------------

T_start = robot.get_T_ee_world()

print("\nCurrent EE Pose")
print(T_start)

print("\nCurrent Position")
print(T_start[:3, 3])

# ---------------------------------------------------------
# Goal
# ---------------------------------------------------------

T_goal = T_start.copy()

# Move DOWN by 5 cm
T_goal[2, 3] -= 0.05

print("\nGoal Position")
print(T_goal[:3, 3])

print("\nExpected Translation")
print(T_goal[:3, 3] - T_start[:3, 3])

# ---------------------------------------------------------
# Execute
# ---------------------------------------------------------

print("\nMoving...\n")

success = controller.move_to_pose(
    T_goal,
    max_steps=300,
    render=True,
)

print("\nSuccess :", success)

# ---------------------------------------------------------
# Final pose
# ---------------------------------------------------------

T_final = robot.get_T_ee_world()

print("\nFinal Pose")
print(T_final)

print("\nFinal Position")
print(T_final[:3, 3])

pos_error = T_goal[:3, 3] - T_final[:3, 3]

R_err = T_goal[:3, :3] @ T_final[:3, :3].T
rot_error = Rotation.from_matrix(R_err).as_rotvec()

print("\nFinal Position Error")
print(pos_error)
print("Norm :", np.linalg.norm(pos_error))

print("\nFinal Orientation Error")
print(rot_error)
print("Norm :", np.linalg.norm(rot_error))

# ---------------------------------------------------------
# Gripper
# ---------------------------------------------------------

print("\nOpening Gripper")

robot.set_gripper(0.0)
robot.step(
    np.concatenate([
        np.zeros(6),
        [robot.get_gripper()],
    ])
)

for _ in range(50):
    env.render()

print("\nClosing Gripper")

robot.set_gripper(1.0)
robot.step(
    np.concatenate([
        np.zeros(6),
        [robot.get_gripper()],
    ])
)

for _ in range(50):
    env.render()

env.close()