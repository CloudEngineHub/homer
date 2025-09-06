import argparse
import copy
import numpy as np
import torch
import os
import pyrallis
import mujoco as mj

import time
from itertools import count
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from constants import POLICY_CONTROL_PERIOD
from envs.common_mj_env import MujocoEnvConfig
from envs.utils.camera_utils import pcl_from_obs
from interactive_scripts.dataset_recorder import ActMode
from models.waypoint_transformer_cond import WaypointTransformer
from scipy.spatial.transform import Rotation as R
import common_utils
import open3d as o3d
from common_utils.eval_utils import (
    check_for_interrupt,
)
from contextlib import nullcontext
from PIL import Image
import pdb
import sys
import os

from pointing_utils.gemini_wrapper import GeminiWrapper
gemini = GeminiWrapper("YOUR API KEY")

def eval_waypoint(
    policy: WaypointTransformer,
    env,
    seed: int,
    num_pass: int,
    save_dir,
    record: bool,
):
    assert not policy.training
    env.seed(seed)
    goal = env.reset()
    obs = env.get_obs()

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)

    curr_mode = ActMode.ArmWaypoint.value if env.cfg.wbc else ActMode.BaseWaypoint.value
    start_time = time.time()

    target_xyz = None

    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        obs = env.get_obs()
        recorder.add_numpy(obs, ["viewer_image"])

        if check_for_interrupt() or env.num_step > env.max_num_step or obs['reward']:
            break

        points, colors = pcl_from_obs(obs, env.cfg)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        proprio = obs["proprio"]

        if target_xyz is None:
        # pdb.set_trace()
            item = "center of the %s cube"%goal
            rgb_image = obs["base1_image"]
            depth_image = obs["base1_depth"]
            K = obs["base1_K"]

            # Save image
            image = Image.fromarray(rgb_image)
            os.makedirs("salient_auto", exist_ok=True)  # ensure directory exists
            image_path = "salient_auto/image.jpg"
            image.save(image_path)
            coords = gemini.point_to_object("salient_auto/image.jpg", prompt=item)
            # pdb.set_trace()
            print("COORDS: ", coords)
            # Parse 2D coordinates
            x_px = int(coords["cx"])
            y_px = int(coords["cy"])

            # Get depth at that point
            depth = depth_image[y_px, x_px]
            # (4) Deproject to camera frame
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x_cam = (x_px - cx) * depth / fx
            y_cam = (y_px - cy) * depth / fy
            z_cam = depth
            pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])  # homogeneous

            # (5) Transform to world frame
            T_base1 = obs["base1_T"]
            pt_world = T_base1 @ pt_cam
            pt_world = pt_world[:3]
            target_xyz = pt_world
        
        visualize_salient = False
        if visualize_salient:
            pcd = o3d.geometry.PointCloud()
            points, colors = pcl_from_obs(obs, env.cfg)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.paint_uniform_color([1, 0, 0])  # red
            sphere.translate(pt_world)
            o3d.visualization.draw_geometries([pcd, sphere])

        # === Generate soft click_probs ===
        radius = 0.05
        distances = np.linalg.norm(points - target_xyz[None], axis=1)
        click_probs = np.clip((radius - distances), 0, None)
        if click_probs.max() > 0:
            click_probs /= click_probs.max()
        else:
            click_probs[:] = 1.0 / len(click_probs)

        with torch.no_grad():
            _, pos_cmd, rot_cmd, gripper_cmd, mode = policy.inference(
                torch.from_numpy(points).float(),
                torch.from_numpy(colors).float(),
                torch.from_numpy(click_probs).float(),
                torch.from_numpy(proprio).float(),
                num_pass=num_pass,
            )
            if policy.cfg.use_euler:
                quat_cmd = R.from_euler('xyz', rot_cmd).as_quat()
            else:
                quat_cmd = rot_cmd

        if curr_mode == ActMode.ArmWaypoint.value:
            if quat_cmd[3] < 0:
                np.negative(quat_cmd, out=quat_cmd)
            reached, err = env.move_to_arm_waypoint(pos_cmd, quat_cmd, gripper_cmd, recorder=recorder)

        elif curr_mode == ActMode.BaseWaypoint.value:
            base_pose = np.array([pos_cmd[0], pos_cmd[1], R.from_quat(quat_cmd).as_euler('xyz')[2]])
            reached, err = env.move_to_base_waypoint(base_pose, recorder=recorder)

        else:
            break

        curr_mode = mode

    if recorder is not None:
        recorder.save(f"s{seed}", fps=10)

    return obs['reward'], env.num_step

def _eval_waypoint_multi_episode(
    policy: WaypointTransformer,
    num_pass: int,
    env_cfg: MujocoEnvConfig,
    seed: int,
    num_episode: int,
    save_dir,
    record: int,
):
    scores = []
    num_steps = []
    for seed in range(seed, seed + num_episode):
        score, num_step = eval_waypoint(
            policy,
            env_cfg,
            seed=seed,
            num_pass=num_pass,
            save_dir=save_dir,
            record=record,
        )
        scores.append(score)
        num_steps.append(num_step)
    return np.mean(scores), np.mean(num_steps)

def eval_waypoint_policy(
    policy: WaypointTransformer,
    env_cfg_path: str,
    num_pass: int,
    num_eval_episode: int,
    stat: common_utils.MultiCounter,
    prefix: str = "",
    save_dir = None,
    record: int = 0
):
    assert os.path.exists(env_cfg_path), f"cannot locate env config {env_cfg_path}"
    env_cfg = pyrallis.load(MujocoEnvConfig, open(env_cfg, "r"))
    score, num_step = _eval_waypoint_multi_episode(
        policy, num_pass, env_cfg, 99999, num_eval_episode, save_dir=save_dir, record=record
    )
    stat[f"eval/{prefix}score"].append(score)
    stat[f"eval/{prefix}num_step"].append(num_step)
    return score

def main():
    import os
    import sys
    from scripts.train_waypoint_cond import load_waypoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=20)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='rollouts')
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube.yaml")
    parser.add_argument("--headless", action='store_true')
    args = parser.parse_args()

    print(f">>>>>>>>>>{args.model}<<<<<<<<<<")

    policy = load_waypoint(args.model, device='cuda')
    policy.train(False)
    policy = policy.cuda()

    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))
    if env_cfg.wbc:
        from envs.mj_env_wbc import MujocoEnv
    else:
        from envs.mj_env_base_arm import MujocoEnv

    if args.topk > 0:
        print(f"Overriding topk_eval to be {args.topk}")
        policy.cfg.topk_eval = args.topk
    else:
        print(f"Eval with original topk_eval {policy.cfg.topk_eval}")

    scores = []

    idx = 0
    if os.path.exists('rollouts'):
        idx = len([fn for fn in os.listdir('rollouts') if 'mp4' in fn])
    args.seed += idx

    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, "eval.log")
        if os.path.exists(log_path):
            sys.stdout = common_utils.Logger(log_path, mode="a", print_to_stdout=True)
        else:
            sys.stdout = common_utils.Logger(log_path, mode="w", print_to_stdout=True)

    for idx, seed in enumerate(range(args.seed, args.seed + args.num_episode)):
        env = MujocoEnv(env_cfg, show_viewer=not args.headless, show_images=False)
        score, num_step = eval_waypoint(
            policy,
            env,
            seed=seed,
            num_pass=args.num_pass,
            save_dir=args.save_dir,
            record=args.record,
        )
        scores.append(score)
        print(f"[{idx+1}/{args.num_episode}], % Success: {np.mean(scores):.4f}, # Success: {int(np.sum(scores))}")
        print(common_utils.wrap_ruler("", max_len=80))
        env.close()

if __name__ == "__main__":
    ### Example commands

    ## On a workstation with a display
    ## HOMER-Cond
    # python scripts/eval_waypoint_cond.py --model exps/waypoint/cube_wbc_cond/latest.pt --env_cfg envs/cfgs/cube_wbc.yaml 
    # python scripts/eval_waypoint_cond.py --model exps/waypoint/cube_wbc_cond/latest.pt --env_cfg envs/cfgs/cube_wbc_size.yaml 
    # python scripts/eval_waypoint_cond.py --model exps/waypoint/cube_wbc_cond/latest.pt --env_cfg envs/cfgs/cube_wbc_distractor.yaml 
    # python scripts/eval_waypoint_cond.py --model exps/waypoint/cube_wbc_cond/latest.pt --env_cfg envs/cfgs/cube_wbc_specified.yaml 
    
    ## HOMER-Cond-NoAugs
    # python scripts/eval_waypoint_cond.py --model exps/waypoint/cube_wbc_cond_noaugs/latest.pt --env_cfg envs/cfgs/cube_wbc.yaml 
    # python scripts/eval_waypoint_cond.py --model exps/waypoint/cube_wbc_cond_noaugs/latest.pt --env_cfg envs/cfgs/cube_wbc_size.yaml 
    # python scripts/eval_waypoint_cond.py --model exps/waypoint/cube_wbc_cond_noaugs/latest.pt --env_cfg envs/cfgs/cube_wbc_distractor.yaml 
    # python scripts/eval_waypoint_cond.py --model exps/waypoint/cube_wbc_cond_noaugs/latest.pt --env_cfg envs/cfgs/cube_wbc_specified.yaml 

    ## NOTE: For headless eval
    # xvfb-run -s "-screen 0 1920x1080x24" python scripts/eval_waypoint.py ... --headless
    main()
