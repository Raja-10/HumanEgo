# -*- coding: utf-8 -*-
# @FileName: run_inference.py
"""
HumanEgo dual-arm real-world inference — reference loop.

    camera ─▶ perception ─▶ ICT + clean image ─▶ policy ─▶ EE trajectory ─▶ robot
       ▲                                                                      │
       └──────────────────────── close the loop ◀────────────────────────────┘

This is a TEMPLATE: it shows the standard structure of a HumanEgo inference
stack. It will not run as-is — you must supply working hardware drivers and a
perception module (the example uses Intel RealSense + Trossen + DINO-SAM/LaMa).
Everything hardware/perception-specific is isolated in the three adapters at the
top and clearly marked with `TODO`; the policy + control logic below is generic.

Pipeline
--------
ONE-TIME (episode start):
    1. estimate object 6DoF poses from a few RGB-D frames  -> object-centric frame
    2. home the arms, open grippers

EVERY STEP (closed loop, ~5-10 Hz):
    3. grab RGB-D
    4. read each arm's EE pose (FK) + gripper state
    5. "latch" grasped objects so their pose tracks the gripper
    6. build the clean, embodiment-agnostic image (inpaint arm + render gripper)
    7. build the ICT from hand + object poses
    8. policy.infer -> future EE trajectory (+ done probability) for both arms
    9. decode trajectory to camera-frame EE targets, execute the first few steps
   10. stop when the policy reports "done"
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import yaml

import robosuite as suite




# Make BOTH this folder and the repo root importable regardless of CWD, so the
# flat imports here (interfaces/policy/controller, CamRS/RobotArmTrossen) and the
# package imports inside policy.py (training.*, utils.*) all resolve when you run
# `python inference/run_inference.py cfg/inference/example_dualarm.yaml` from root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interfaces import Camera, Frame, ObjectState, Perception, RobotArm
from policy import ICTPolicy
from controller import TrajectoryController


# =====================================================================
# 1. HARDWARE ADAPTERS  — replace the bodies with your own hardware.
#    These wrap the shipped RealSense / Trossen drivers to satisfy the
#    interfaces in interfaces.py. The mapping is the whole point: copy
#    this pattern for your camera and arm.
# =====================================================================

class RobosuiteCamera(Camera):
    """Adapter over the shipped CamRS (Intel RealSense) driver."""
    def __init__(self, env, width=640, height=480, fovy_deg=None):
        # from CamRS import CamRS                       # TODO: your camera SDK
        # self.cam = CamRS(cam_cfg_path)
        from CamRobosuite import CamRobosuite
        self.cam = CamRobosuite(env, width=width, height=height, fovy_deg=fovy_deg)
        
    def get_frame(self) -> Frame:
        d = self.cam.get_rgbd()                       # CamRSData(rgb, depth_m, ...)
        return Frame(rgb=d.rgb, depth_m=d.depth_m, K=self.cam.k_rgb)

    def close(self) -> None:
        self.cam.close()

    def get_T_cam_in_world(self):
            return self.cam.get_T_cam_in_world()

class RobosuiteArm(RobotArm):
    """Adapter over the shipped RobotArmTrossen driver."""
    def __init__(self, arm_cfg_path: str):
        # from RobotArmTrossen import RobotArmTrossen   # TODO: your robot SDK
        # self.arm = RobotArmTrossen(arm_cfg_path)

        from PandaRobosuite import PandaRobosuite
        self.arm = PandaRobosuite(arm_cfg_path)
        
        self.T_base_in_cam = self.arm.T_base_in_cam   # from hand-eye calibration

    def get_T_ee_in_cam(self) -> np.ndarray:
        return self.arm.get_T_ee_in_cam()
    def move_ee_in_cam(self, T, duration, blocking=False) -> bool:
        return self.arm.move_p_in_cam(self.arm.T_to_p(T), duration=duration, blocking=blocking)
    def get_gripper(self) -> float:
        return 1.0 - self.arm.get_gripper_q()         # driver: 1=open -> ours: 0=open
    def set_gripper(self, value: float, blocking: bool = False) -> None:
        (self.arm.close_gripper if value > 0.5 else self.arm.open_gripper)(blocking=blocking)
    def go_home(self, blocking: bool = True) -> None:
        self.arm.go_home(blocking=blocking)
    def close(self) -> None:
        self.arm.close()


# =====================================================================
# 2. PERCEPTION  — the heaviest part to port. Reference impl uses the
#    shipped preprocess/ engines (DINO-SAM detect+segment, LaMa inpaint).
#    Swap in ANY detector/pose-estimator that returns object poses.
# =====================================================================

class ReferencePerception(Perception):
    """Object 6DoF poses + clean image, mirroring the training preprocessing.

    The exact same detect→segment→lift→pose / inpaint→render pipeline is used
    at training time (see preprocess/), which is why train and test images match.
    """
    def __init__(self, cam: RealSenseCamera, cfg: dict):
        self.cam = cam
        self.prompts = cfg["object_prompts"]          # {"obj1": "a green cup .", ...}
        self.erase_prompt = cfg["erase_prompt"]       # e.g. "a robot arm . a gripper ."
        self.anchor_key = cfg.get("anchor_key", "obj1")
        # Heavy models — load once. (These imports/weights are the engineering cost.)
        # from preprocess.DINOSAM import DINOSAM
        # from preprocess.Lama   import LamaEngine
        # self.detector = DINOSAM(cfg["dinosam_cfg_path"])
        # self.inpainter = LamaEngine(cfg["lama_cfg_path"])
        # self.renderer  = VisualKptsEngine(cfg["visualkpts_cfg_path"])

    def estimate_objects(self, frames: List[Frame]) -> Dict[str, ObjectState]:
        objs: Dict[str, ObjectState] = {}
        for key, prompt in self.prompts.items():
            # (a) detect + segment the object on the first frame
            #     mask = self.detector.process_single(frames[0].rgb, prompt)
            # (b) lift the masked pixels to 3D across frames (robust to depth noise)
            #     pts3d_cam = self.cam.cam.lift3d_for_multi_frames(uv, depths, mask, K)
            # (c) fit a 6DoF pose to the 3D points (PCA / your favorite estimator)
            #     T_in_cam, _ = estimate_frame_pca2(pts3d_cam, is_anchor=(key==self.anchor_key))
            # (d) keypoints in the object's own frame (only needed for PCD features)
            #     kpts_local = (inv(T_in_cam) @ homogeneous(pts3d_cam))
            raise NotImplementedError(
                "Plug in your detector + pose estimator here. See preprocess/ for the "
                "DINO-SAM + depth-lift + PCA reference, or return poses from any source "
                "(AprilTag, FoundationPose, known CAD + ICP, ...)."
            )
        return objs

    def make_clean_image(self, frame, ee_poses_in_cam, grippers) -> np.ndarray:
        # (a) inpaint the real arm out:   mask = detector(rgb, erase_prompt);
        #                                 clean = inpainter.inpaint(rgb, mask)
        # (b) render a virtual gripper at each EE pose onto `clean`, matching the
        #     training visualization:     renderer.process_single_gripper(clean, T_ee, grasp, K)
        # If the model was trained state-only (no image), just return a black frame.
        return frame.rgb.copy()



class SimPerception(Perception):

    def __init__(self, env, cam, cfg, grasp_threshold=0.5):

        self.env = env
        self.cam = cam
        self.grasp_threshold = grasp_threshold

        self.anchor_key = cfg.get("anchor_key", "obj1")
        self.object_names = cfg["object_names"]

        # Sim's oracle substitute for the real pipeline's clean-image step
        # (preprocess/DINOSAM.py detect+segment -> preprocess/Lama.py inpaint
        # -> preprocess/VisualKpts.py render). We skip the learned detect/
        # inpaint models entirely (CamRobosuite.get_clean_bg() removes the arm
        # exactly, via scene geometry, since sim has no need to *guess* where
        # the arm is) but reuse the real VisualKptsEngine renderer verbatim so
        # the gripper/keypoint visualization matches training pixel-for-pixel.
        from preprocess.VisualKpts import VisualKptsEngine
        visualkpts_cfg_path = cfg.get(
            "visualkpts_cfg_path",
            os.path.join(_ROOT, "cfg", "preprocess", "base", "VisualKpts.yaml"),
        )
        self.visualkpts = VisualKptsEngine(cfg_path=visualkpts_cfg_path)
        self._obj_colors = {
            key: self.visualkpts.cfg.obj_colors[i % len(self.visualkpts.cfg.obj_colors)]
            for i, key in enumerate(sorted(self.object_names.keys()))
        }

    def estimate_objects(self, frames):

        objs = {}

        T_cam_world = self.cam.get_T_cam_in_world()

        for key, body_name in self.object_names.items():

            body_id = self.env.sim.model.body_name2id(body_name)

            pos = self.env.sim.data.body_xpos[body_id]

            rot = self.env.sim.data.body_xmat[body_id].reshape(3,3)

            T_world = np.eye(4)

            T_world[:3,:3] = rot
            T_world[:3,3] = pos

            T_in_cam = np.linalg.inv(T_cam_world) @ T_world

            objs[key] = ObjectState(
                T_in_cam=T_in_cam,
                kpts_local=np.zeros((0,3), np.float32),
            )

        return objs

    def make_clean_image(self, frame, ee_poses_in_cam, grippers, objs=None):
        # (a) arm removal: re-render with the robot hidden (oracle, exact —
        #     see CamRobosuite.get_clean_bg) instead of detect+inpaint.
        clean = self.cam.cam.get_clean_bg()

        # (b) virtual gripper per hand, exactly as at training time.
        for side, T_ee in ee_poses_in_cam.items():
            is_grasping = grippers.get(side, 0.0) > self.grasp_threshold
            clean = self.visualkpts.process_single_gripper(clean, T_ee, is_grasping, frame.K)

        # (c) object keypoints: sim has no tracked point cloud per object
        # (kpts_local is always empty here), so render one stable-colored dot
        # at each object's projected center as a lightweight stand-in for the
        # real pipeline's CoTracker keypoint clouds.
        if objs:
            for key, obj in objs.items():
                p = obj.T_in_cam[:3, 3]
                if p[2] <= 1e-3:
                    continue
                uv = frame.K @ p
                u, v = uv[0] / uv[2], uv[1] / uv[2]
                color = self._obj_colors.get(key)
                clean = self.visualkpts.process_single_obj(clean, np.array([[u, v]]), force_color=color)

        return clean

# =====================================================================
# 3. Small helpers
# =====================================================================

def latch_objects(
    static_objs: Dict[str, ObjectState],
    hands_in_cam: Dict[str, np.ndarray],
    grippers: Dict[str, float],
    latches: dict,
    anchor_key: str,
    grasp_threshold: float = 0.5,
) -> Dict[str, ObjectState]:
    """Make a grasped object's pose follow the gripper.

    When an arm closes, we lock the relative transform gripper->nearest-object;
    while it stays closed, the object's pose is driven by the gripper. This keeps
    the object token in the ICT consistent with what's physically happening even
    if vision can't see the occluded object. (Simplified single-object latch.)

    The ANCHOR object is never latched — it defines the fixed object-centric
    reference frame for the episode and must stay put.
    """
    dynamic = {k: ObjectState(v.T_in_cam.copy(), v.kpts_local) for k, v in static_objs.items()}
    latchable = {k: v for k, v in static_objs.items() if k != anchor_key}
    if not latchable:
        return dynamic
    for side, T_h in hands_in_cam.items():
        if T_h is None:
            continue
        closed = grippers.get(side, 0.0) > grasp_threshold
        if closed and latches.get(side) is None:
            # latch onto the nearest (non-anchor) object at the moment of grasp
            nearest = min(latchable, key=lambda k: np.linalg.norm(
                latchable[k].T_in_cam[:3, 3] - T_h[:3, 3]))
            latches[side] = (nearest, np.linalg.inv(T_h) @ latchable[nearest].T_in_cam)
        elif not closed:
            latches[side] = None
        if latches.get(side) is not None:
            key, T_lock = latches[side]
            dynamic[key] = ObjectState(T_h @ T_lock, static_objs[key].kpts_local)
    return dynamic


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =====================================================================
# 4. Main loop
# =====================================================================

def run(cfg_path: str, device: str = "cuda") -> None:
    cfg = load_cfg(cfg_path)

    # print("Config path:", cfg_path)
    # print("Config keys:", cfg.keys())

    cfg_sides = cfg["robot"]["sides"]                  # physical arms present, e.g. ["left","right"]
    T_align = np.array(cfg["robot"].get("T_align", np.eye(4).tolist()), dtype=np.float32)
    anchor_key = cfg["perception"].get("anchor_key", "obj1")
    exec_horizon = cfg["control"].get("exec_horizon", 8)   # steps run before re-planning
    dt = 1.0 / cfg["control"].get("control_hz", 10.0)
    done_threshold = cfg["control"].get("done_threshold", 0.8)


    # move_p_in_cam() below sends absolute EE poses (from the smoothed policy
    # trajectory), not small normalized deltas — so the arm controller must be
    # configured for absolute input, not the robosuite-default delta input
    # (which would clip every command to a tiny fixed-size step in a fixed
    # direction, producing the "snaps/doesn't track the target" behavior of
    # the default delta config).
    from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config
    controller_config = load_composite_controller_config(robot=cfg["simulation"]["robot"])
    controller_config["body_parts"]["right"]["input_type"] = "absolute"

    # Create simulation environment
    env = suite.make(
        env_name=cfg["simulation"]["env_name"],
        robots=cfg["simulation"]["robot"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cfg["camera"]["camera_name"],
        camera_heights=cfg["camera"]["h"],
        camera_widths=cfg["camera"]["w"],
        camera_depths=[True],
        ignore_done=True,  # the policy's own done_prob is the stopping condition, not robosuite's step horizon
    )

    env.reset()


    # ---- build the stack ----
    # cam = RealSenseCamera(cfg["camera"]["cfg_path"])
    # arms: Dict[str, RobotArm] = {s: TrossenArm(cfg["robot"]["cfg_paths"][s]) for s in cfg_sides}
    cam = RobosuiteCamera(env, width=cfg["camera"]["w"], height=cfg["camera"]["h"],
                          fovy_deg=cfg["camera"].get("fovy_deg"))
    arms: Dict[str, RobotArm] = {"right": RobosuiteArm(env)}
    policy = ICTPolicy(cfg["policy"], device=device)
    controller = TrajectoryController(arms, cfg["control"])
    perception = SimPerception(env, cam, cfg["perception"],
                                grasp_threshold=cfg["control"].get("grasp_threshold", 0.5))

    # The policy decides how many hands it drives (read from the checkpoint).
    sides = policy.sides                               # e.g. ["left","right"] or ["right"]
    missing = [s for s in sides if s not in arms]
    if missing:
        raise ValueError(f"checkpoint predicts {sides} but no robot arm configured for {missing}. "
                         f"Set robot.sides to include them.")

    # ---- one-time: homing + object-centric setup ----
    # Homing MUST happen before estimate_objects(), not after: go_home() calls
    # env.reset(), and ServeBread's placement_initializer (UniformRandomSampler)
    # re-samples object positions on every reset. Capturing the anchor before
    # this reset snapshots a placement that gets thrown away the moment we
    # home the arm — the ICT is then built around stale object poses for the
    # entire episode while the real objects sit somewhere else (confirmed
    # empirically: anchor world Z read 0.93 right after the "before" reset,
    # but the true resting Z, only visible once physics had actually stepped
    # under the post-home placement, was 0.81 — a ~12cm, not merely a
    # "still settling", discrepancy).
    for arm in arms.values():
        arm.set_gripper(0.0, blocking=True)            # open
        arm.go_home(blocking=True)

    print("[run] estimating object poses...")
    n = cfg["perception"].get("n_init_frames", 10)
    static_objs = perception.estimate_objects([cam.get_frame() for _ in range(n)])
    anchor = static_objs.get(anchor_key)               # fixed object-centric reference for the episode

    latches: dict = {s: None for s in sides}
    done = {s: False for s in sides}
    print("[run] entering closed-loop control. Ctrl-C to stop.")

    DEBUG = os.environ.get("HUMANEGO_DEBUG", "0") == "1"
    _dbg_i = 0
    if DEBUG:
        for k, v in static_objs.items():
            print(f"[dbg] static_objs[{k}].T_in_cam pos = {v.T_in_cam[:3, 3]}")

    try:
        while not all(done.values()):
            frame = cam.get_frame()

            # read robot state -> HAND-frame poses (EE pose bridged by T_align)
            hands_in_cam = {s: arms[s].get_T_ee_in_cam() @ T_align for s in sides}
            grippers = {s: arms[s].get_gripper() for s in sides}

            # grasped (non-anchor) objects follow the gripper
            objs = latch_objects(static_objs, hands_in_cam, grippers, latches,
                                 anchor_key, controller.grasp_threshold)

            # build the policy inputs: clean image, ICT, and (if region-attn) anchor UV
            clean = perception.make_clean_image(frame, hands_in_cam, grippers, objs)
            x_ict, ict_mask = policy.build_ict(hands_in_cam, grippers, objs, anchor_key)
            x_rgb = policy.prepare_image(clean)
            anchor_uv = policy.compute_anchor_uv(anchor, frame.K,
                                                 frame.rgb.shape[1], frame.rgb.shape[0])

            # predict the future trajectory for every hand the policy drives
            traj, done_prob = policy.infer(x_rgb, x_ict, ict_mask, anchor_uv)

            # decode reference-frame predictions -> camera-frame EE targets
            ee_targets: Dict[str, List[np.ndarray]] = {}
            grasp_cmds: Dict[str, np.ndarray] = {}
            for s in sides:
                pos, o6d, grasp = traj[s]
                ee_targets[s] = [policy.decode_ee_in_cam(pos[k], o6d[k], anchor, T_align)
                                 for k in range(len(pos))]
                grasp_cmds[s] = grasp

                for side in grasp_cmds:
                    grasp_cmds[side] = np.squeeze(grasp_cmds[side], axis=-1)
               
                # print("\n===== DEBUG =====")
                # for side in grasp_cmds:
                #     print(f"{side}:")
                #     print("type :", type(grasp_cmds[side]))
                #     print("shape:", np.asarray(grasp_cmds[side]).shape)
                #     print("first:", grasp_cmds[side][0])
                #     print("=================")
            if DEBUG and _dbg_i < 20:
                ee_pos = hands_in_cam["right"][:3, 3]
                obj_pos = objs["obj2"].T_in_cam[:3, 3]
                anchor_pos = objs[anchor_key].T_in_cam[:3, 3]
                tgt0 = ee_targets["right"][0][:3, 3]
                tgtN = ee_targets["right"][-1][:3, 3]
                print(f"[dbg] i={_dbg_i} ee={ee_pos} croissant={obj_pos} plate={anchor_pos} "
                      f"dist_ee_to_croissant={np.linalg.norm(ee_pos - obj_pos):.3f} "
                      f"tgt0={tgt0} tgtN={tgtN} grip={grippers['right']:.2f} "
                      f"grasp_cmd0={grasp_cmds['right'][0]:.2f} done_prob={done_prob:.3f}")

                # ---- WORLD-FRAME ground truth cross-check ----
                T_cam_world = cam.get_T_cam_in_world()
                anchor_body = cfg["perception"]["object_names"][anchor_key]
                anchor_world_gt = env.sim.data.body_xpos[env.sim.model.body_name2id(anchor_body)].copy()
                ee_world = arms["right"].arm.get_T_ee_in_base()[:3, 3]
                tgt0_world = (T_cam_world @ np.append(tgt0, 1.0))[:3]
                tgtN_world = (T_cam_world @ np.append(tgtN, 1.0))[:3]

                # ---- ORIENTATION check: is the target a top-down grasp? ----
                # VisualKpts' gripper convention: local Z = approach axis,
                # origin at the fingertip plane. For "pick up from above," that
                # axis should point roughly straight down in world (~[0,0,-1])
                # once the gripper is at the object.
                R_cam_world = T_cam_world[:3, :3]
                tgt0_R_world = R_cam_world @ ee_targets["right"][0][:3, :3]
                tgt_k7_R_world = R_cam_world @ ee_targets["right"][exec_horizon - 1][:3, :3]
                ee_R_world = arms["right"].arm.get_T_ee_in_base()[:3, :3]
                down = np.array([0.0, 0.0, -1.0])
                def approach_angle_deg(R):
                    z_axis = R[:, 2]
                    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
                    cos_a = np.clip(np.dot(z_axis, down), -1.0, 1.0)
                    return np.degrees(np.arccos(cos_a)), z_axis
                ang_ee, zax_ee = approach_angle_deg(ee_R_world)
                ang_tgt0, zax_tgt0 = approach_angle_deg(tgt0_R_world)
                ang_k7, zax_k7 = approach_angle_deg(tgt_k7_R_world)
                print(f"[dbg]   ORIENT: approach-axis-vs-straight-down angle (0=top-down grasp): "
                      f"current_ee={ang_ee:.1f}deg (z_world={zax_ee}) "
                      f"tgt0={ang_tgt0:.1f}deg (z_world={zax_tgt0}) "
                      f"tgt_k{exec_horizon-1}={ang_k7:.1f}deg (z_world={zax_k7})")
                tgt_exec_last = ee_targets["right"][exec_horizon - 1][:3, 3]
                tgt_exec_last_world = (T_cam_world @ np.append(tgt_exec_last, 1.0))[:3]
                print(f"[dbg]   WORLD: ee={ee_world} anchor_gt({anchor_body})={anchor_world_gt} "
                      f"tgt0_world={tgt0_world} tgtN_world={tgtN_world} "
                      f"tgt_k{exec_horizon-1}_world={tgt_exec_last_world} "
                      f"dist(ee,anchor_gt)={np.linalg.norm(ee_world - anchor_world_gt):.3f} "
                      f"dist(tgt0_world,anchor_gt)={np.linalg.norm(tgt0_world - anchor_world_gt):.3f} "
                      f"dist(tgtN_world,anchor_gt)={np.linalg.norm(tgtN_world - anchor_world_gt):.3f} "
                      f"dist(tgt0_world,ee)={np.linalg.norm(tgt0_world - ee_world):.3f}")
                if _dbg_i == 0:
                    np.save("/tmp/claude-1000/-home-cobot-Desktop-Raja-Projects-HumanEgo/98890c8d-458d-4550-8107-191e1d93c8f7/scratchpad/real_ee_target.npy",
                            ee_targets["right"][0])
                    np.save("/tmp/claude-1000/-home-cobot-Desktop-Raja-Projects-HumanEgo/98890c8d-458d-4550-8107-191e1d93c8f7/scratchpad/real_ee_target_traj.npy",
                            np.stack(ee_targets["right"]))
                    np.save("/tmp/claude-1000/-home-cobot-Desktop-Raja-Projects-HumanEgo/98890c8d-458d-4550-8107-191e1d93c8f7/scratchpad/real_grasp_traj.npy",
                            grasp_cmds["right"])
                    np.save("/tmp/claude-1000/-home-cobot-Desktop-Raja-Projects-HumanEgo/98890c8d-458d-4550-8107-191e1d93c8f7/scratchpad/real_hand_in_cam.npy",
                            hands_in_cam["right"])
                    print("[dbg] saved real T_cmd + trajectory + current hand pose to /tmp scratchpad")
                _dbg_i += 1

            # execute the first few steps, then re-plan (receding horizon)
            if DEBUG and _dbg_i < 20:
                before = arms["right"].arm.get_T_ee_in_base()[:3, 3].copy()
            controller.execute_chunk(ee_targets, grasp_cmds, dt=dt, n_steps=exec_horizon)
            if DEBUG and _dbg_i < 20:
                after = arms["right"].arm.get_T_ee_in_base()[:3, 3].copy()
                print(f"[dbg]   world-pos before-chunk={before} after-chunk={after} "
                      f"moved={np.linalg.norm(after - before):.5f} control_freq={env.control_freq} "
                      f"action_dim={env.action_dim}")

            if done_prob > done_threshold:
                print(f"[run] policy reports done (p={done_prob:.2f}).")
                done = {s: True for s in sides}

    except KeyboardInterrupt:
        print("\n[run] interrupted.")
    finally:
        controller.home()
        for arm in arms.values():
            arm.close()
        cam.close()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "./cfg/inference/robosuite.yaml"
    run(cfg_path)
