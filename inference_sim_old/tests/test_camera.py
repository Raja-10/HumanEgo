"""
Test script for RobosuiteCamera.

Checks:
1. RGB acquisition
2. Depth acquisition
3. Camera intrinsics
4. Display RGB + Depth
"""

import cv2
import numpy as np
import robosuite as suite

from inference_sim.camera import RobosuiteCamera


def main():

    # ---------------------------------------------------------
    # Create Environment
    # ---------------------------------------------------------

    env = suite.make(

        env_name="ServeBread",

        robots="Nero7",

        has_renderer=True,

        has_offscreen_renderer=True,

        use_camera_obs=True,

        camera_names="agentview",

        camera_widths=640,

        camera_heights=480,

        camera_depths=True,

        control_freq=20,
    )

    env.reset()

    # ---------------------------------------------------------
    # Create Camera Wrapper
    # ---------------------------------------------------------

    camera = RobosuiteCamera(
        env,
        camera_name="agentview",
        width=640,
        height=480,
    )

    print("=" * 60)
    print("Testing Camera")
    print("=" * 60)

    while True:

        # Step simulation
        action = np.zeros(env.action_dim)
        env.step(action)

        frame = camera.get_frame()

        rgb = frame.rgb
        depth = frame.depth
        K = frame.K

        # -------------------------------------------------
        # Print information
        # -------------------------------------------------

        print()

        print("RGB Shape      :", rgb.shape)
        print("RGB dtype      :", rgb.dtype)

        print()

        print("Depth Shape    :", depth.shape)
        print("Depth dtype    :", depth.dtype)

        print()

        print("Depth Range")
        print("  Min :", depth.min())
        print("  Max :", depth.max())

        print()

        print("Intrinsic Matrix")
        print(K)

        # -------------------------------------------------
        # Normalize depth for visualization
        # -------------------------------------------------

        depth_vis = depth.copy()

        depth_vis -= depth_vis.min()

        depth_vis /= (depth_vis.max() + 1e-8)

        depth_vis = (255 * depth_vis).astype(np.uint8)

        depth_vis = cv2.applyColorMap(
            depth_vis,
            cv2.COLORMAP_TURBO,
        )

        # RGB comes as RGB -> convert for OpenCV display
        rgb_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("RGB", rgb_vis)

        cv2.imshow("Depth", depth_vis)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    camera.close()

    env.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()