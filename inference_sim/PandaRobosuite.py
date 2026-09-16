# -*- coding: utf-8 -*-
"""
PandaRobosuite.py

Simulation equivalent of RobotArmTrossen.py, for the robosuite Panda arm.

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


class PandaRobosuite:

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
        # get_T_ee_in_base() below actually reads MuJoCo's site_xpos/site_xmat,
        # which are WORLD-frame values (despite the name) — and SimPerception
        # expresses object poses in the REAL camera frame the same way
        # (T_in_cam = inv(T_cam_in_world) @ T_obj_in_world). So this transform
        # must be inv(T_cam_in_world), applied to those world-frame EE poses,
        # to land hands and objects in the same frame. Leaving this as
        # identity (the previous behavior) put the EE in world frame while
        # objects were in camera frame — two different frames silently
        # combined, which is why the arm didn't track objects at all.
        # -------------------------------------------------

        camera_name = (getattr(env, "camera_names", None) or ["agentview"])[0]
        cam_id = self.model.camera_name2id(camera_name)
        cam_pos = self.data.cam_xpos[cam_id]
        # GL -> CV camera-axis convention (see CamRobosuite.get_T_cam_in_world
        # for why): must match exactly, or the EE and objects end up in two
        # different "camera frames" again.
        cam_rot = self.data.cam_xmat[cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T_cam_in_world = np.eye(4)
        T_cam_in_world[:3, :3] = cam_rot
        T_cam_in_world[:3, 3] = cam_pos

        self._T_base_in_cam = np.linalg.inv(T_cam_in_world)

        # -------------------------------------------------
        # World <-> robot-base transform.
        #
        # The low-level OSC_POSE controller for this robot has
        # input_ref_frame="base": in absolute mode it takes goal_pos/goal_ori
        # expressed in the ROBOT'S OWN BASE frame, not world and not camera
        # frame. move_p_in_cam() below receives targets in camera frame (to
        # match the policy's convention), so it must convert cam -> world ->
        # base before it can be sent to the controller as an absolute action.
        # -------------------------------------------------

        base_body = f"{self.robot.robot_model.naming_prefix}base"
        base_id = self.model.body_name2id(base_body)
        base_pos = self.data.body_xpos[base_id]
        base_rot = self.data.body_xmat[base_id].reshape(3, 3)
        T_base_in_world = np.eye(4)
        T_base_in_world[:3, :3] = base_rot
        T_base_in_world[:3, 3] = base_pos

        self._T_world_in_base = np.linalg.inv(T_base_in_world)
        self._T_cam_in_base = self._T_world_in_base @ T_cam_in_world

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

        # Re-fetch env.sim.data every call rather than using the self.data
        # cached at __init__ time: robosuite's default "hard reset" (used by
        # go_home() -> env.reset()) rebuilds the MuJoCo model and allocates a
        # brand new sim/data object, so a cached reference goes stale and
        # silently stops updating — every subsequent read would report
        # whatever the pose happened to be at that reset, forever (this is
        # exactly what produced "moved=0.00000" every step after homing).
        data = self.env.sim.data

        pos = data.site_xpos[self.ee_site]

        rot = data.site_xmat[self.ee_site].reshape(3, 3)

        T = np.eye(4)

        T[:3, :3] = rot

        T[:3, 3] = pos

        return T

    # -----------------------------------------------------

    def get_T_ee_in_cam(self):

        return self.T_base_in_cam @ self.get_T_ee_in_base()

    # -----------------------------------------------------

    def get_T_ee_in_base_frame(self):
        """EE pose expressed in the robot's own base frame (what the
        absolute-mode OSC_POSE controller actually expects), as opposed to
        get_T_ee_in_base() above which — despite its name — returns the raw
        MuJoCo world-frame pose."""

        return self._T_world_in_base @ self.get_T_ee_in_base()

    # -----------------------------------------------------

    def _gripper_hold_action(self, gripper_cmd):
        """A zero-motion action for gripper-only commands.

        In absolute mode, action[:6] IS the goal pose — an all-zero action
        (as used pre-fix) would command the arm to the base origin. Instead,
        hold the arm at its current pose and only set the gripper channel.
        """

        T_ee_in_base = self.get_T_ee_in_base_frame()

        action = np.zeros(self.env.action_dim)

        action[:6] = self.T_to_p(T_ee_in_base)

        action[-1] = gripper_cmd

        return action

    # =====================================================
    # Cartesian Motion
    # =====================================================

    def move_p_in_cam(
        self,
        pose,
        duration=2.0,
        blocking=False,
    ):

        # `pose` is a target EE pose in CAMERA frame (the policy's
        # convention). The controller's absolute input is in BASE frame, so
        # convert: cam -> base.
        T_cmd_cam = self.p_to_T(pose)
        T_cmd_base = self._T_cam_in_base @ T_cmd_cam

        target_pos = T_cmd_base[:3, 3]
        target_rot = R.from_matrix(T_cmd_base[:3, :3]).as_rotvec()

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

    # Panda's finger_joint1 travels [0, 0.04] m (0=closed, 0.04=open) — see
    # gripper0_right_finger_joint1 in the MuJoCo model. get_gripper_q() must
    # return a normalized [0,1] fraction ("1=open", per RobosuiteArm.get_gripper's
    # own comment), NOT the raw meters value: raw qpos is always near 0
    # (~0-0.02), so `1.0 - raw_qpos` (the old code) was always ~0.98-1.0 —
    # i.e. every caller was told the gripper is basically always closed,
    # regardless of its true state. That fed a constant, wrong "closed"
    # signal into both the ICT (build_ict) and the clean-image grasp-state
    # render on every single step.
    GRIPPER_OPEN_Q = 0.04

    def get_gripper_q(self):

        obs = self.env._get_observations()

        q_raw = float(obs["robot0_gripper_qpos"][0])

        return float(np.clip(q_raw / self.GRIPPER_OPEN_Q, 0.0, 1.0))

    # -----------------------------------------------------

    def open_gripper(self, blocking=True):

        self.env.step(self._gripper_hold_action(-1))

    # -----------------------------------------------------

    def close_gripper(self, blocking=True):

        self.env.step(self._gripper_hold_action(1))

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
