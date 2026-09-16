"""
camera.py

Simple Robosuite camera wrapper.

Responsibilities
----------------
- Capture RGB image
- Capture depth image
- Return camera intrinsics

Nothing else.

No perception.
No robot state.
No preprocessing.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Frame:
    """One synchronized RGB-D observation."""

    rgb: np.ndarray          # (H,W,3) uint8
    depth: np.ndarray        # (H,W) float32 (meters)
    K: np.ndarray            # (3,3)


class RobosuiteCamera:
    """
    Wrapper around a Robosuite camera.

    Works exactly like Camera.get_frame() in HumanEgo.
    """

    def __init__(
        self,
        env,
        camera_name="agentview",
        width=640,
        height=480,
    ):

        self.env = env

        self.camera_name = camera_name

        self.width = width
        self.height = height

    # -----------------------------------------------------

    def get_frame(self):

        obs = self.env._get_observations()

        rgb = obs[f"{self.camera_name}_image"]

        depth = obs[f"{self.camera_name}_depth"]

        rgb = np.flipud(rgb).copy()
        depth = np.flipud(depth).copy()

        K = self.get_intrinsics()

        return Frame(
            rgb=rgb,
            depth=depth.astype(np.float32),
            K=K,
        )

    # -----------------------------------------------------

    def get_intrinsics(self):
        """
        Compute camera intrinsic matrix from
        MuJoCo camera FOV.
        """

        model = self.env.sim.model

        cam_id = model.camera_name2id(self.camera_name)

        fovy = model.cam_fovy[cam_id]

        fovy = np.deg2rad(fovy)

        fy = self.height / (2 * np.tan(fovy / 2))

        fx = fy

        cx = self.width / 2

        cy = self.height / 2

        K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        return K

    # -----------------------------------------------------

    def close(self):
        pass