"""
robot.py

Robosuite Robot wrapper implementing the HumanEgo RobotArm interface.

Frames
------
Camera frame : HumanEgo "world"
World frame  : MuJoCo / Robosuite world

The policy always talks in CAMERA coordinates.
This wrapper performs the required frame conversions.
"""

from __future__ import annotations

import numpy as np


class RobosuiteRobot:

    def __init__(self, env, T_base_in_cam=None):

        self.env = env
        self.robot = env.robots[0]
        self.sim = env.sim

        self.ee_site = self.robot.eef_site_id[
            self.robot.arms[0]
        ]

        # ------------------------------------------------------------------
        # Hand-eye calibration
        #
        # Robot Base ---> Camera
        #
        # Replace this with the actual calibration when needed.
        # For simulation, identity is perfectly fine if camera==world.
        # ------------------------------------------------------------------

        if T_base_in_cam is None:
            T_base_in_cam = np.eye(4)

        self.T_base_in_cam = T_base_in_cam

        # Camera -> Base
        self.T_cam_in_base = np.linalg.inv(self.T_base_in_cam)

        self.gripper = 0.0

    # ==========================================================
    # Low-level utilities
    # ==========================================================

    def _get_T_ee_world(self):

        T = np.eye(4)

        T[:3, 3] = self.sim.data.site_xpos[self.ee_site]

        T[:3, :3] = (
            self.sim.data.site_xmat[self.ee_site]
            .reshape(3, 3)
        )

        return T

    # ==========================================================
    # HumanEgo API
    # ==========================================================

    def get_T_ee_in_cam(self):
        """
        End-effector pose expressed in camera frame.
        """

        T_world = self._get_T_ee_world()

        return self.T_cam_in_base @ T_world

    def cam_to_world(self, T_cam):

        return self.T_base_in_cam @ T_cam

    def world_to_cam(self, T_world):

        return self.T_cam_in_base @ T_world

    # ==========================================================
    # Simulation
    # ==========================================================

    def step(self, action):

        self.env.step(action)

    def render(self):

        self.env.render()

    # ==========================================================
    # Gripper
    # ==========================================================

    def get_gripper(self):

        return self.gripper

    def set_gripper(self, value):

        self.gripper = float(np.clip(value, 0.0, 1.0))

    # ==========================================================
    # Home
    # ==========================================================

    def go_home(self):

        self.env.reset()