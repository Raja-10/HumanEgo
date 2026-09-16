

from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass
class CamRobosuiteData:
    rgb: np.ndarray
    depth_m: np.ndarray


class CamRobosuite:
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
        fovy_deg=None,
    ):

        self.env = env

        self.camera_name = camera_name

        self.width = width
        self.height = height

        if fovy_deg is not None:
            # Override MuJoCo's default agentview fovy (45 deg) to match the real
            # Aria camera's vertical FOV (~67.4 deg, from training_data.json's
            # metadata.k). Must happen before get_intrinsics() reads cam_fovy.
            # Robosuite's default 45 deg fovy is far narrower than the real camera
            # (HFOV 57.9 deg sim vs 90.0 deg real) -- this was feeding
            # ICTPolicy.compute_anchor_uv() a systematically different pixel
            # projection than the model was trained on.
            cam_id = env.sim.model.camera_name2id(camera_name)
            env.sim.model.cam_fovy[cam_id] = fovy_deg

        self.k_rgb = self.get_intrinsics()
        self.k_depth = self.k_rgb.copy()

        self.depth_scale = 1.0

        self.rgb_key = f"{camera_name}_image"
        self.depth_key = f"{camera_name}_depth"

        self._robot_geom_ids = None

    # -----------------------------------------------------

    def _get_robot_geom_ids(self):
        """Geom ids belonging to the robot arm/gripper/mount, computed once
        and cached: these are what get_clean_bg() hides."""

        if self._robot_geom_ids is None:
            model = self.env.sim.model
            prefixes = ("robot0_", "gripper0_", "fixed_mount0_")
            self._robot_geom_ids = {
                i for i in range(model.ngeom)
                if (nm := model.geom_id2name(i)) and nm.startswith(prefixes)
            }
        return self._robot_geom_ids

    # -----------------------------------------------------

    def get_clean_bg(self) -> np.ndarray:
        """Re-render the current camera view with the robot arm/gripper/mount
        hidden, everything else (table, objects) rendered normally in their
        CURRENT pose.

        This is simulation's oracle substitute for the real pipeline's
        SAM2-segment + LaMa-inpaint arm removal (see preprocess/DINOSAM.py,
        preprocess/Lama.py): since we have exact scene geometry, we can just
        not draw the robot instead of learning to paint over it. Patches the
        already-resolved MjvScene geoms (post mjv_updateScene, pre mjr_render)
        rather than model.geom_rgba, because geom_rgba alpha is not respected
        once a geom has a bound material (empirically verified — zeroing it
        left the arm fully opaque). Scene-patching is also transient: scn is
        rebuilt from scratch by mjv_updateScene on the next call, so nothing
        needs to be restored afterward.
        """

        model = self.env.sim.model
        data = self.env.sim.data
        robot_geom_ids = self._get_robot_geom_ids()

        ctx = self.env.sim._render_context_offscreen
        mujoco.mjv_updateScene(
            model._model, data._data, ctx.vopt, ctx.pert, ctx.cam,
            mujoco.mjtCatBit.mjCAT_ALL, ctx.scn,
        )
        for i in range(ctx.scn.ngeom):
            g = ctx.scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in robot_geom_ids:
                g.rgba[3] = 0.0

        mujoco.mjr_render(
            viewport=mujoco.MjrRect(0, 0, self.width, self.height),
            scn=ctx.scn, con=ctx.con,
        )
        rgb = ctx.read_pixels(self.width, self.height, depth=False)

        return rgb[::-1].copy()

    # -----------------------------------------------------

    def get_rgbd(self) -> CamRobosuiteData:

        obs = self.env._get_observations()

        # print(obs.keys())
        rgb = obs[self.rgb_key]
        depth = obs[self.depth_key]

        return CamRobosuiteData(
            rgb=rgb,
            depth_m=depth.astype(np.float32),
            )

    # -----------------------------------------------------

    def get_intrinsics(self) -> np.ndarray:
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

    def get_T_cam_in_world(self):

        model = self.env.sim.model
        data = self.env.sim.data

        cam_id = model.camera_name2id(self.camera_name)

        pos = data.cam_xpos[cam_id]

        # MuJoCo's cam_xmat is in OpenGL convention (camera looks down its own
        # -Z, Y up). The policy was trained on real RGB-D camera data, which
        # uses the standard computer-vision convention (X right, Y down, Z
        # forward into the scene). Flipping Y/Z here converts the camera's
        # local axes from GL to CV convention so every "in_cam" pose computed
        # downstream (objects AND the EE) matches what the policy expects.
        rot = data.cam_xmat[cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])

        T = np.eye(4)

        T[:3, :3] = rot
        T[:3, 3] = pos

        return T
    
    # -----------------------------------------------------

    def close(self):
        pass