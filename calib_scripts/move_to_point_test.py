import argparse
import copy
import pprint
import numpy as np
import torch
import pyrallis
import signal
import sys
import threading
from PIL import Image
import open3d as o3d

from envs.common_real_env_cfg import RealEnvConfig
from teleop.policies import TeleopPolicy
from scipy.spatial.transform import Rotation as R
import common_utils

import os
import time
from itertools import count

from envs.utils.camera_utils import pcl_from_obs, deproject_pixel_to_3d

import matplotlib.pyplot as plt
import numpy as np

# Global vars
env = None
cleanup_lock = threading.Lock()
cleanup_done = False

def select_pixel_from_numpy_image(np_image):
    """
    Display a NumPy image and let the user click a pixel.
    Returns (x, y, (R, G, B)) after the window is closed.
    """

    coords = {"x": None, "y": None, "rgb": None}

    def onclick(event):
        """Mouse click callback function"""
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)

            # Get pixel color at the clicked position
            pixel = np_image[y, x]

            # Convert to 0–255 range if needed
            if pixel.max() <= 1.0:
                rgb = tuple(int(c * 255) for c in pixel[:3])
            else:
                rgb = tuple(int(c) for c in pixel[:3])

            print(f"Coordinates: ({x}, {y}) | RGB: {rgb}")

            # Mark the clicked point
            ax.plot(x, y, 'go', markersize=5)
            plt.draw()

            # Save the selection
            coords["x"], coords["y"], coords["rgb"] = x, y, rgb

    print("Click on the image to get pixel coordinates.")
    print("Close the window to return.\n")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(np_image)
    ax.set_title("Click on image to get pixel coordinates")
    ax.axis('on')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Return the last clicked coordinates
    return (coords["x"], coords["y"])

# Signal handler for Ctrl+C
def handle_signal(signum, frame):
    global cleanup_done, env
    with cleanup_lock:
        if cleanup_done:
            print("[Force Exit] Cleanup already started. Forcing exit.")
            sys.exit(1)
        print("\n[Signal] Ctrl+C or Ctrl+\ received. Cleaning up...")
        cleanup_done = True
        if env is not None:
            try:
                env.close()
                print("Closed env.")
            except Exception as e:
                print(f"[Error] Failed to close env cleanly: {e}")
        sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGQUIT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def move_to_point(env):
    env.reset()
    obs = env.get_obs()
    quat = obs["arm_quat"]
    rgb_image = obs["base2_image"]
    x, y = coords = select_pixel_from_numpy_image(rgb_image)

    points, colors = pcl_from_obs(obs, env.cfg)

    target_xyz = deproject_pixel_to_3d(obs, (x, y), "base2", env.cfg)
    target_xyz[2] += 0.02

    visualize_salient = True
    if visualize_salient:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color([1, 0, 0])  # red
        sphere.translate(target_xyz)
        o3d.visualization.draw_geometries([pcd, sphere])

    reached, err, interrupt = env.move_to_arm_waypoint(target_xyz, quat, 1.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_cfg", type=str, default="envs/cfgs/real_base_arm.yaml")
    args = parser.parse_args()

    try:
        env_cfg = pyrallis.load(RealEnvConfig, open(args.env_cfg, "r"))
        from envs.real_env_base_arm import RealEnv

        env = RealEnv(env_cfg)
        env.reset()
        print('here1')
        move_to_point(env)


    except Exception as e:
        print(f"[Error] Unhandled exception: {e}")

    finally:
        if not cleanup_done:
            try:
                if env is not None:
                    env.close()
                    print("Closed env.")
            except Exception as e:
                print(f"[Error] Cleanup failed in finally block: {e}")
