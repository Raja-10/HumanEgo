"""
controller.py

High-level controller for HumanEgo.

Policy outputs:
    EE pose in CAMERA frame

Robosuite expects:
    EE pose in CONTROLLER BASE frame

Pipeline

Camera
   │
   ▼
World (robot.cam_to_world)
   │
   ▼
Controller Base
   │
   ▼
OSC Absolute
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class InferenceController:

    def __init__(
        self,
        robot,
        position_tol=0.005,
        orientation_tol=np.deg2rad(2),
    ):

        self.robot = robot

        self.position_tol = position_tol
        self.orientation_tol = orientation_tol

    # -----------------------------------------------------

    def reached(self, T_now_cam, T_goal_cam):

        dp = T_goal_cam[:3, 3] - T_now_cam[:3, 3]

        Rerr = T_goal_cam[:3, :3] @ T_now_cam[:3, :3].T

        dr = Rotation.from_matrix(Rerr).as_rotvec()

        return (
            np.linalg.norm(dp) < self.position_tol
            and np.linalg.norm(dr) < self.orientation_tol
        )

    # -----------------------------------------------------

    def world_to_controller(self, T_world):

        cc = self.robot.env.robots[0].composite_controller

        origin_pos, origin_rot = cc.get_controller_base_pose("right")

        T_origin = np.eye(4)
        T_origin[:3, :3] = origin_rot
        T_origin[:3, 3] = origin_pos

        return np.linalg.inv(T_origin) @ T_world

    # -----------------------------------------------------

    def move_to_pose(
        self,
        T_goal_cam,
        max_steps=200,
        render=True,
    ):

        for i in range(max_steps):

            # -------------------------------------------------
            # Current EE pose (camera frame)
            # -------------------------------------------------

            T_now_cam = self.robot.get_T_ee_in_cam()

            if self.reached(T_now_cam, T_goal_cam):
                print(f"Reached target in {i} iterations.")
                return True

            # -------------------------------------------------
            # Camera -> World
            # -------------------------------------------------

            T_goal_world = self.robot.cam_to_world(T_goal_cam)

            # -------------------------------------------------
            # World -> Controller Base
            # -------------------------------------------------

            T_goal_base = self.world_to_controller(T_goal_world)

            # -------------------------------------------------
            # Build OSC action
            # -------------------------------------------------

            pos = T_goal_base[:3, 3]

            rotvec = Rotation.from_matrix(
                T_goal_base[:3, :3]
            ).as_rotvec()

            action = np.concatenate(
                [
                    pos,
                    rotvec,
                    [self.robot.get_gripper()],
                ]
            )

            # -------------------------------------------------
            # Debug
            # -------------------------------------------------

            if i % 20 == 0:

                print("=" * 60)
                print("Goal (Camera)")
                print(T_goal_cam[:3, 3])

                print("\nGoal (World)")
                print(T_goal_world[:3, 3])

                print("\nGoal (Controller)")
                print(pos)

                print("\nAction")
                print(action)

                print("=" * 60)

            self.robot.step(action)

            if render:
                self.robot.render()

        print("Controller timeout.")
        return False