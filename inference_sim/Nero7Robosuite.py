# -*- coding: utf-8 -*-
"""
Nero7Robosuite.py

Simulation equivalent of RobotArmTrossen.py.

Responsibilities
----------------
- Read EE pose
- Move EE
- Open/close gripper
- Return camera-frame EE pose
- Convert between SE(3) and pose vectors

No networking.
No hardware driver.
No GUI.
No teleoperation.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.spatial.transform import Rotation as R


class Nero7Robosuite:

    def __init__(self, env):

        self.env = env

        # -------------------------------------------------
        # Robot / Mujoco handles
        # -------------------------------------------------

        self.robot = env.robots[0]

        self.model = env.sim.model
        self.data = env.sim.data

        # -------------------------------------------------
        # End-effector site
        # -------------------------------------------------

        eef_site = self.robot.eef_site_id

        # Robosuite >=1.5 stores EEF sites in a dict
        if isinstance(eef_site, dict):
            eef_site = eef_site["right"]

        # Convert name -> id if necessary
        if isinstance(eef_site, str):
            self.ee_site = self.model.site_name2id(eef_site)
        else:
            self.ee_site = eef_site

        # -------------------------------------------------
        # Camera -> Base transform
        #
        # In simulation this is exact.
        # If your camera is fixed, compute once.
        # -------------------------------------------------

        self._T_base_in_cam = np.eye(4)

    # =====================================================
    # Utilities
    # =====================================================

    @property
    def T_base_in_cam(self):

        return self._T_base_in_cam

    # -----------------------------------------------------

    @staticmethod
    def T_to_p(T):

        p = np.zeros(6)

        p[:3] = T[:3, 3]

        p[3:] = R.from_matrix(T[:3, :3]).as_rotvec()

        return p

    # -----------------------------------------------------

    @staticmethod
    def p_to_T(p):

        T = np.eye(4)

        T[:3, :3] = R.from_rotvec(p[3:]).as_matrix()

        T[:3, 3] = p[:3]

        return T

    # =====================================================
    # EE Pose
    # =====================================================

    def get_T_ee_in_base(self):

        pos = self.data.site_xpos[self.ee_site]

        rot = self.data.site_xmat[self.ee_site].reshape(3, 3)

        T = np.eye(4)

        T[:3, :3] = rot

        T[:3, 3] = pos

        return T

    # -----------------------------------------------------

    def get_T_ee_in_cam(self):

        return self.T_base_in_cam @ self.get_T_ee_in_base()

    # =====================================================
    # Cartesian Motion
    # =====================================================

    def move_p_in_cam(
        self,
        pose,
        duration=2.0,
        blocking=False,
    ):

        target_pos = pose[:3]

        target_rot = pose[3:]

        action = np.zeros(self.env.action_dim)

        action[:3] = target_pos

        action[3:6] = target_rot

        steps = max(1, int(duration * self.env.control_freq))

        for _ in range(steps):

            self.env.step(action)

            if blocking:

                self.env.render()

        return True

    # =====================================================
    # Gripper
    # =====================================================

    # def get_gripper_q(self):

    #     gripper = self.robot.gripper["right"]

    #     return np.clip(
    #         gripper.current_action[0],
    #         0.0,
    #         1.0,
    #     )

    def get_gripper_q(self):


        obs = self.env._get_observations()
        # print(obs["robot0_gripper_qpos"])

        q = float(obs["robot0_gripper_qpos"][0])

        return q

    # -----------------------------------------------------

    def open_gripper(self, blocking=True):

        action = np.zeros(self.env.action_dim)

        action[-1] = -1

        self.env.step(action)

    # -----------------------------------------------------

    def close_gripper(self, blocking=True):

        action = np.zeros(self.env.action_dim)

        action[-1] = 1

        self.env.step(action)

    # =====================================================
    # Home
    # =====================================================

    def go_home(
        self,
        blocking=True,
    ):

        self.env.reset()

    # =====================================================
    # Shutdown
    # =====================================================

    def close(self):

        self.env.close()