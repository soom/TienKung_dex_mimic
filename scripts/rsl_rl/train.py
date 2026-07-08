# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Override episode length in seconds after motion-based auto inference.",
)
parser.add_argument(
    "--episode_length_cap_s",
    type=float,
    default=None,
    help="Cap auto-inferred episode length to at most this many seconds.",
)
parser.add_argument(
    "--episode_length_scale",
    type=float,
    default=1.0,
    help="Scale factor applied to auto-inferred episode length before cap/override.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--curriculum_initial_stage",
    type=int,
    default=None,
    help="Override the starting curriculum stage (0-based). Useful when resuming to roll back to an earlier stage.",
)
parser.add_argument(
    "--registry_name", type=str, default=None,
    help="WandB registry artifact name. Mutually exclusive with --motion_file.",
)
parser.add_argument(
    "--load_ckpt", type=str, default=None,
    help="Absolute path to a checkpoint .pt file for cross-task weight initialization.",
)
parser.add_argument(
    "--motion_file", type=str, default=None,
    help="Local path to motion .npz file. Mutually exclusive with --registry_name.",
)


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_MIN_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_MIN_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_MIN_VERSION}"]
    else:
        cmd = ["python", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_MIN_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_MIN_VERSION}' or newer.\nTo install a compatible version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.tracking_env_cfg import set_episode_length_from_motion_file
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        env_cfg.device = f"cuda:{app_launcher.local_rank}"

        base_seed = agent_cfg.seed if agent_cfg.seed is not None else 42
        seed = base_seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # load the motion file — local path or WandB registry
    import pathlib

    if args_cli.motion_file is not None:
        motion_path = str(pathlib.Path(args_cli.motion_file).resolve())
        env_cfg.commands.motion.motion_file = motion_path
        registry_name = "local"
        # For episode length: if directory, pick the longest NPZ; if single file, use it directly.
        if os.path.isdir(motion_path):
            import glob as _glob
            npz_files = sorted(_glob.glob(os.path.join(motion_path, "*.npz")))
            npz_files = [f for f in npz_files
                         if not os.path.basename(f).endswith("_ik.npz")
                         and "_all" not in os.path.basename(f)]
            if not npz_files:
                raise FileNotFoundError(f"No .npz files found in directory: {motion_path}")
            best_file = max(npz_files, key=lambda f: np.load(f)["joint_pos"].shape[0])
            episode_length_ref_file = best_file
            print(f"[INFO]: Directory mode — {len(npz_files)} NPZ files. Episode length from: {os.path.basename(best_file)}")
        else:
            episode_length_ref_file = motion_path
            print(f"[INFO]: Using local motion file: {motion_path}")
    elif args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        raise ValueError("Either --motion_file or --registry_name must be provided.")

    episode_length_s = set_episode_length_from_motion_file(env_cfg, episode_length_ref_file)
    if args_cli.episode_length_scale <= 0.0:
        raise ValueError(f"`--episode_length_scale` must be positive, got {args_cli.episode_length_scale}")
    if args_cli.episode_length_cap_s is not None and args_cli.episode_length_cap_s <= 0.0:
        raise ValueError(f"`--episode_length_cap_s` must be positive, got {args_cli.episode_length_cap_s}")
    if args_cli.episode_length_s is not None and args_cli.episode_length_s <= 0.0:
        raise ValueError(f"`--episode_length_s` must be positive, got {args_cli.episode_length_s}")

    adjusted_episode_length_s = episode_length_s * args_cli.episode_length_scale
    if args_cli.episode_length_cap_s is not None:
        adjusted_episode_length_s = min(adjusted_episode_length_s, args_cli.episode_length_cap_s)
    if args_cli.episode_length_s is not None:
        adjusted_episode_length_s = args_cli.episode_length_s

    env_cfg.episode_length_s = adjusted_episode_length_s
    if args_cli.curriculum_initial_stage is not None:
        env_cfg.commands.motion.curriculum_initial_stage = args_cli.curriculum_initial_stage
        print(f"[INFO]: curriculum_initial_stage overridden to {args_cli.curriculum_initial_stage}")
    print(
        "[INFO]: Episode length set from motion file: "
        f"{episode_length_s:.2f}s ({int(round(episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))} steps)"
    )
    if abs(adjusted_episode_length_s - episode_length_s) > 1e-9:
        print(
            "[INFO]: Episode length adjusted: "
            f"{adjusted_episode_length_s:.2f}s "
            f"({int(round(adjusted_episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))} steps)"
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = os.getenv("WBT_RUN_TIMESTAMP", datetime.now().strftime("%m%d_%H%M"))
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if args_cli.distributed:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        print(
            f"[INFO] Distributed training enabled: rank={app_launcher.global_rank}, "
            f"local_rank={app_launcher.local_rank}, world_size={world_size}, device={agent_cfg.device}"
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # Cross-task checkpoint loading (e.g. standing model from tracking weights)
    if getattr(args_cli, "load_ckpt", None):
        load_path = pathlib.Path(args_cli.load_ckpt).expanduser().resolve()
        runner.load(load_path)
        print(f"[INFO]: Loading model checkpoint from: {load_path}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
