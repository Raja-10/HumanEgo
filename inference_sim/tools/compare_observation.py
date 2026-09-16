# -*- coding: utf-8 -*-
# @FileName: compare_observation.py
"""
Observation-to-policy-input parity check: real recorded data vs robosuite sim.

Scope: everything up to and including building the policy's two inputs (the
clean RGB tensor and the ICT tensor) at EPISODE START — arm/hand home, gripper
open, nothing grasped, done=0 — on both the real (Aria) side and the sim
(robosuite) side. No flow-matching inference, no decode, no control.

Why episode start: real data is one human episode on Aria glasses, sim is a
fresh robosuite episode with a different embodiment and randomized object
placement — they can never be frame-identical. Episode start is the one
semantic phase that's unambiguous and reproducible on both sides.

Real-data frame gotcha (see plan): `inference_sim/policy.py`'s ICTPolicy
treats whatever pose you pass into build_ict() as already being "in the
reference frame" for this checkpoint's frame_mode ("camera_frame" ->
identity T_cam_in_ref). Training's actual camera_frame reference is
`metadata.world_transforms.cam0` -- the camera pose FROZEN at session start,
NOT the live/current camera pose (`metadata.c2w`, which drifts as the Aria
headset moves). So real poses must be pre-multiplied by
`T_w2ref = FlowMatchingDataloader._get_T_w2ref(d0)` (reusing the dataloader's
own method) before being handed to `policy.build_ict()` -- passing raw
world-frame poses directly would be wrong for any frame beyond the first few.
Grasp is also binarized (>0.5 -> 1.0) by the dataloader before it builds a
token, but NOT by `ICTPolicy.build_ict()` -- binarize on the real side before
calling it, or the ICT diff below will show a spurious mismatch in the last
token dim.

Usage:
    python inference_sim/tools/compare_observation.py cfg/inference/robosuite.yaml
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch

# Make inference_sim/ and the repo root importable, mirroring run_inference.py.
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../inference_sim/tools
_SIM = os.path.dirname(_HERE)                                  # .../inference_sim
_ROOT = os.path.dirname(_SIM)                                   # repo root
for _p in (_SIM, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interfaces import ObjectState                              # noqa: E402
from policy import ICTPolicy                                    # noqa: E402
from run_inference import RobosuiteCamera, RobosuiteArm, SimPerception, load_cfg  # noqa: E402

from training.FlowMatchingDataloader import FlowMatchingDataloader, MPSSessions   # noqa: E402
from utils.utils_io import read_json                             # noqa: E402
from utils.utils_math import unnormalize_pos                     # noqa: E402


TYPE_NAMES = {0.0: "PAD", 1.0: "HAND_L", 2.0: "HAND_R", 3.0: "OBJ_ANCHOR", 4.0: "OBJ_OTHER"}


def build_dataloader_for_session(mps_path: str, tcfg: dict, stats: dict) -> FlowMatchingDataloader:
    """One-session FlowMatchingDataloader, config pulled from the checkpoint's own
    config.json so it's byte-for-byte the same construction training used."""
    return FlowMatchingDataloader(
        sessions=[MPSSessions(mps_path)],
        image_size=tuple(tcfg.get("image_size", [240, 320])),
        pred_horizon=tcfg.get("pred_horizon", 50),
        single_hand=tcfg.get("single_hand", True),
        single_hand_side=tcfg.get("single_hand_side", "right"),
        max_ict=tcfg.get("max_ict", 8),
        centric_mode=tcfg.get("centric_mode", "object_centric"),
        frame_mode=tcfg.get("frame_mode", "anchor_frame"),
        action_mode=tcfg.get("action_mode", "absolute"),
        use_pcd_features=tcfg.get("use_pcd_features", False),
        use_aux_obj_dynamics=tcfg.get("use_aux_obj_dynamics", False),
        use_aux_visual_foresight=tcfg.get("use_aux_visual_foresight", False),
        use_aux_temporal_contrastive=tcfg.get("use_aux_temporal_contrastive", False),
        enable_augmentation=False,               # deterministic, teacher-forced comparison
        stats=stats,
        hand_tracking_method=tcfg.get("hand_tracking_method", "aria_mps"),
    )


def real_frame_to_policy_inputs(d0: dict, ds: FlowMatchingDataloader, T_w2ref: np.ndarray):
    """One training_data.json dict -> exactly what ICTPolicy.build_ict() expects.

    Applies T_w2ref (dataloader's own frozen-reference transform) to the raw
    world-frame poses, and binarizes grasp -- both required for the result to
    be comparable to ds._build_ict()'s ground truth. See module docstring.
    """
    hands = d0.get("entities", {}).get(ds.hand_entity_key, {}) or {}
    objs = d0.get("entities", {}).get("objects", {})
    hand_sides = [ds.single_hand_side] if ds.single_hand else ["left", "right"]

    ee_poses_in_cam, grippers = {}, {}
    for side in hand_sides:
        if side in hands:
            T_h_w = np.array(hands[side]["T_hand_to_world"], dtype=np.float32)
            ee_poses_in_cam[side] = T_w2ref @ T_h_w
            raw_grasp = float(hands[side]["grasp"])
            grippers[side] = 1.0 if raw_grasp > 0.5 else 0.0     # match dataloader's binarization

    obj_states = {
        k: ObjectState(
            T_in_cam=(T_w2ref @ np.array(v["T_obj_to_world"], dtype=np.float32)),
            kpts_local=np.zeros((0, 3), np.float32),
        )
        for k, v in objs.items()
    }
    anchor_key = d0["metadata"].get("anchor_key", "obj1")
    return ee_poses_in_cam, grippers, obj_states, anchor_key


def describe_image(name: str, x: torch.Tensor):
    arr = x.detach().cpu().numpy()
    print(f"  {name:<10s} shape={tuple(arr.shape)} dtype={arr.dtype} "
          f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")


def describe_ict(name: str, state: np.ndarray, mask: np.ndarray, pos_mean, pos_std):
    print(f"  {name} tokens (mask -> valid):")
    for i in range(state.shape[0]):
        if not mask[i]:
            continue
        row = state[i]
        type_name = TYPE_NAMES.get(float(row[0]), f"?{row[0]}")
        pos_norm = row[1:4]
        pos_m = unnormalize_pos(pos_norm, pos_mean, pos_std)
        grasp = row[-1]
        print(f"    [{i}] {type_name:<10s} pos_m={np.round(pos_m, 3)} "
              f"|pos_norm|={np.linalg.norm(pos_norm):.3f} grasp/flag={grasp:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cfg_path", nargs="?", default="cfg/inference/robosuite.yaml")
    ap.add_argument("--mps_path", default="data/serve_bread/aria/mps_serve_bread_000_vrs",
                     help="Real session to pull the episode-start step from.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol_ict", type=float, default=1e-3)
    ap.add_argument("--out_dir", default=os.path.join(_ROOT, "inference_sim", "tools", "_compare_out"))
    args = ap.parse_args()

    cfg = load_cfg(args.cfg_path)
    ckpt_path = cfg["policy"]["ckpt"]
    ckpt_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        tcfg = json.load(f)
    with open(os.path.join(ckpt_dir, "dataset_stats.json")) as f:
        stats = json.load(f)

    print(f"=== compare_observation.py | ckpt={ckpt_path} | frame_mode={tcfg.get('frame_mode')} "
          f"centric_mode={tcfg.get('centric_mode')} hand_tracking_method={tcfg.get('hand_tracking_method')} ===\n")

    policy = ICTPolicy(cfg["policy"], device=args.device)

    # =================================================================
    # REAL side
    # =================================================================
    print(f"[real] session = {args.mps_path}")
    ds = build_dataloader_for_session(args.mps_path, tcfg, stats)
    if len(ds.samples) == 0:
        raise RuntimeError(f"No samples found under {args.mps_path}/preprocess/all_data/ "
                            f"-- has this session been preprocessed?")
    json_path0 = ds.samples[0]                          # episode start = lowest-numbered step
    d0 = read_json(json_path0)
    print(f"[real] step   = {json_path0}")

    T_w2ref = ds._get_T_w2ref(d0)
    ee_poses_in_cam, grippers, obj_states, anchor_key = real_frame_to_policy_inputs(d0, ds, T_w2ref)

    # inference_sim's own ICT construction
    x_ict_real, mask_real = policy.build_ict(ee_poses_in_cam, grippers, obj_states, anchor_key)

    # training's ground-truth ICT construction, same JSON, same T_w2ref
    state_gt, _pcd_gt, mask_gt = ds._build_ict(d0, T_w2ref)

    # real clean image, same file the dataloader trains on
    frame_dir = os.path.dirname(d0["obs"]["rgb_path"])
    real_img_path = os.path.join(frame_dir, ds.img_name)
    clean_real = cv2.imread(real_img_path)
    if clean_real is None:
        raise RuntimeError(f"Could not read real clean image at {real_img_path}")
    x_rgb_real = policy.prepare_image(clean_real)

    # =================================================================
    # SIM side  (mirrors run_inference.py: run()'s one-time setup, verbatim)
    # =================================================================
    print(f"\n[sim] env = {cfg['simulation']['env_name']} robot = {cfg['simulation']['robot']}")
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

    controller_config = load_composite_controller_config(robot=cfg["simulation"]["robot"])
    controller_config["body_parts"]["right"]["input_type"] = "absolute"

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
        ignore_done=True,
    )
    env.reset()

    cam = RobosuiteCamera(env, width=cfg["camera"]["w"], height=cfg["camera"]["h"],
                          fovy_deg=cfg["camera"].get("fovy_deg"))
    arm = RobosuiteArm(env)
    arms = {"right": arm}
    perception = SimPerception(env, cam, cfg["perception"],
                                grasp_threshold=cfg["control"].get("grasp_threshold", 0.5))

    arm.set_gripper(0.0, blocking=True)
    arm.go_home(blocking=True)

    n = cfg["perception"].get("n_init_frames", 5)
    static_objs = perception.estimate_objects([cam.get_frame() for _ in range(n)])
    anchor_key_sim = cfg["perception"].get("anchor_key", "obj1")

    T_align = np.array(cfg["robot"].get("T_align", np.eye(4).tolist()), dtype=np.float32)
    frame = cam.get_frame()
    hands_in_cam_sim = {s: arms[s].get_T_ee_in_cam() @ T_align for s in policy.sides if s in arms}
    grippers_sim = {s: arms[s].get_gripper() for s in policy.sides if s in arms}

    x_ict_sim, mask_sim = policy.build_ict(hands_in_cam_sim, grippers_sim, static_objs, anchor_key_sim)
    clean_sim = perception.make_clean_image(frame, hands_in_cam_sim, grippers_sim, static_objs)
    x_rgb_sim = policy.prepare_image(clean_sim)

    arm.close()
    cam.close()

    # =================================================================
    # Report
    # =================================================================
    print("\n--- Image tensors (policy.prepare_image output) ---")
    describe_image("real", x_rgb_real)
    describe_image("sim", x_rgb_sim)

    print("\n--- ICT tensors (policy.build_ict output) ---")
    print(f"  shapes: real={tuple(x_ict_real.shape)} sim={tuple(x_ict_sim.shape)} "
          f"(expected (1, {policy.max_ict}, {policy.ict_dim}) both)")
    describe_ict("real", x_ict_real.cpu().numpy()[0], mask_real.cpu().numpy()[0],
                 policy.pos_mean, policy.pos_std)
    describe_ict("sim", x_ict_sim.cpu().numpy()[0], mask_sim.cpu().numpy()[0],
                 policy.pos_mean, policy.pos_std)

    print("\n--- Cross-check: ICTPolicy.build_ict() vs FlowMatchingDataloader._build_ict() on REAL data ---")
    mask_match = np.array_equal(mask_real.cpu().numpy()[0], mask_gt)
    diff = np.abs(x_ict_real.cpu().numpy()[0] - state_gt)
    max_diff = float(diff.max())
    print(f"  mask match: {mask_match}")
    print(f"  max abs diff: {max_diff:.6f} (tol={args.tol_ict})")
    verdict = "PASS" if (mask_match and max_diff <= args.tol_ict) else "FAIL"
    if verdict == "FAIL":
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  worst token/dim: token={worst[0]} dim={worst[1]} "
              f"real={x_ict_real.cpu().numpy()[0][worst]:.5f} gt={state_gt[worst]:.5f}")
    print(f"  VERDICT: {verdict}")
    if verdict == "FAIL":
        print("  -> bug likely lives in inference_sim/policy.py's build_ict() port, "
              "independent of the sim.")
    else:
        print("  -> inference_sim's ICT construction reproduces training's on real data; "
              "if sim behaves badly, look at SimPerception / T_align / object_names next.")

    os.makedirs(args.out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.out_dir, "real_clean.png"), clean_real)
    cv2.imwrite(os.path.join(args.out_dir, "sim_clean.png"), clean_sim)
    print(f"\nSaved real_clean.png / sim_clean.png to {args.out_dir}")


if __name__ == "__main__":
    main()
