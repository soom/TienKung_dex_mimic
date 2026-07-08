"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import re
import sys
import types

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of environment steps to play.")
parser.add_argument(
    "--start_first_frame",
    action="store_true",
    default=False,
    help="Force playback to start from reference frame 0 instead of the reset sampler.",
)
parser.add_argument(
    "--export_rollout_npz",
    type=str,
    default=None,
    help="If set, export the played rollout and aligned reference data to this NPZ file.",
)
parser.add_argument(
    "--standing_load_run", type=str, default=None,
    help="Run directory for standing head checkpoint (merged ONNX export).",
)
parser.add_argument(
    "--standing_checkpoint", type=str, default=None,
    help="Standing head checkpoint filename (merged ONNX export).",
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

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.tracking_env_cfg import set_episode_length_from_motion_file
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx, export_motion_policy_merged_as_onnx


def sync_policy_observations_from_saved_run(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, resume_path: str):
    """Restore policy observation terms that differ across checkpoints.

    We only patch terms that affect the policy network input dimension. This keeps
    playback compatible when the current workspace config has drifted from the
    config used to train the checkpoint.
    """

    params_env_path = pathlib.Path(resume_path).resolve().parent / "params" / "env.yaml"
    if not params_env_path.is_file():
        return

    saved_text = params_env_path.read_text(encoding="utf-8")
    saved_has_lookahead = re.search(r"^\s*motion_joint_pos_lookahead:\s*$", saved_text, flags=re.MULTILINE) is not None
    current_has_lookahead = hasattr(env_cfg.observations.policy, "motion_joint_pos_lookahead")

    if saved_has_lookahead and not current_has_lookahead:
        lookahead_steps_match = re.search(
            r"motion_joint_pos_lookahead:\n(?:.*\n)*?\s+lookahead_steps:\s*(\d+)", saved_text
        )
        lookahead_steps = int(lookahead_steps_match.group(1)) if lookahead_steps_match else 2
        env_cfg.observations.policy.motion_joint_pos_lookahead = ObsTerm(
            func=mdp.motion_joint_pos_lookahead,
            params={"command_name": "motion", "lookahead_steps": lookahead_steps},
        )
        print(
            "[INFO]: Restored saved policy observation term 'motion_joint_pos_lookahead' "
            f"(lookahead_steps={lookahead_steps}) from {params_env_path}."
        )
    elif not saved_has_lookahead and current_has_lookahead:
        delattr(env_cfg.observations.policy, "motion_joint_pos_lookahead")
        print(
            "[INFO]: Removed workspace-only policy observation term 'motion_joint_pos_lookahead' "
            f"to match saved run config from {params_env_path}."
        )


def initialize_to_motion_frame(env, frame_idx: int = 0):
    command: MotionCommand = env.command_manager.get_term("motion")
    robot = env.scene[command.cfg.asset_name]
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    reset_envs_to_motion_frame(env, env_ids=env_ids, frame_idx=frame_idx)
    env.scene.write_data_to_sim()
    env.sim.render()
    env.scene.update(dt=env.physics_dt)


def reset_envs_to_motion_frame(env, env_ids: torch.Tensor, frame_idx: int = 0):
    command: MotionCommand = env.command_manager.get_term("motion")
    robot = env.scene[command.cfg.asset_name]

    if len(env_ids) == 0:
        return

    frame_idx = max(0, min(frame_idx, command.motion.time_step_total - 1))
    env_ids = env_ids.to(device=env.device, dtype=torch.long)

    command.time_steps[env_ids] = frame_idx

    root_states = robot.data.default_root_state[env_ids].clone()
    root_states[:, :3] = command.motion.body_pos_w[frame_idx, 0] + env.scene.env_origins[env_ids]
    root_states[:, 3:7] = command.motion.body_quat_w[frame_idx, 0]
    root_states[:, 7:10] = command.motion.body_lin_vel_w[frame_idx, 0]
    root_states[:, 10:13] = command.motion.body_ang_vel_w[frame_idx, 0]

    joint_pos = command.motion.joint_pos[frame_idx].unsqueeze(0).repeat(len(env_ids), 1)
    joint_vel = command.motion.joint_vel[frame_idx].unsqueeze(0).repeat(len(env_ids), 1)

    robot.write_root_state_to_sim(root_states, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def force_resets_to_first_frame(env):
    command: MotionCommand = env.command_manager.get_term("motion")

    def _resample_command_from_first_frame(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        reset_envs_to_motion_frame(env, env_ids=env_ids_tensor, frame_idx=0)
        # Reset phase state for playback so resets start fresh
        if command._phase_enabled:
            command.phase_mode[env_ids_tensor] = 0
            command._has_tracked[env_ids_tensor] = False
            command._phase_step_count[env_ids_tensor] = 0
            command._phase_clip_start[env_ids_tensor] = 0
            command._total_step_count[env_ids_tensor] = 0
            command.phase_timeout[env_ids_tensor] = False

    command._resample_command = types.MethodType(_resample_command_from_first_frame, command)
    command.time_steps.zero_()
    # Reset phase state for all envs immediately so _effective_time_steps
    # uses _phase_clip_start=0 (matching time_steps=0) during initial STAND
    if command._phase_enabled:
        command.phase_mode.zero_()
        command._has_tracked.zero_()
        command._phase_step_count.zero_()
        command._phase_clip_start.zero_()
        command._total_step_count.zero_()
        command.phase_timeout.zero_()
    print("[INFO]: Playback resets are pinned to reference frame 0.")


def record_rollout_frame(env, action_tensor: torch.Tensor | None = None) -> dict[str, np.ndarray | int]:
    command: MotionCommand = env.command_manager.get_term("motion")
    robot = env.scene[command.cfg.asset_name]

    ref_frame = int(command.time_steps[0].item())
    body_index = 0

    frame = {
        "ref_time_step": ref_frame,
        "joint_pos_ref": command.joint_pos[0].detach().cpu().numpy().copy(),
        "joint_vel_ref": command.joint_vel[0].detach().cpu().numpy().copy(),
        "joint_pos_policy": robot.data.joint_pos[0].detach().cpu().numpy().copy(),
        "joint_vel_policy": robot.data.joint_vel[0].detach().cpu().numpy().copy(),
        "body_pos_ref": command.body_pos_w[0].detach().cpu().numpy().copy(),
        "body_quat_ref": command.body_quat_w[0].detach().cpu().numpy().copy(),
        "body_lin_vel_ref": command.body_lin_vel_w[0].detach().cpu().numpy().copy(),
        "body_ang_vel_ref": command.body_ang_vel_w[0].detach().cpu().numpy().copy(),
        "body_pos_policy": command.robot_body_pos_w[0].detach().cpu().numpy().copy(),
        "body_quat_policy": command.robot_body_quat_w[0].detach().cpu().numpy().copy(),
        "body_lin_vel_policy": command.robot_body_lin_vel_w[0].detach().cpu().numpy().copy(),
        "body_ang_vel_policy": command.robot_body_ang_vel_w[0].detach().cpu().numpy().copy(),
        "anchor_pos_ref": command.anchor_pos_w[0].detach().cpu().numpy().copy(),
        "anchor_quat_ref": command.anchor_quat_w[0].detach().cpu().numpy().copy(),
        "anchor_lin_vel_ref": command.anchor_lin_vel_w[0].detach().cpu().numpy().copy(),
        "anchor_ang_vel_ref": command.anchor_ang_vel_w[0].detach().cpu().numpy().copy(),
        "anchor_pos_policy": command.robot_anchor_pos_w[0].detach().cpu().numpy().copy(),
        "anchor_quat_policy": command.robot_anchor_quat_w[0].detach().cpu().numpy().copy(),
        "anchor_lin_vel_policy": command.robot_anchor_lin_vel_w[0].detach().cpu().numpy().copy(),
        "anchor_ang_vel_policy": command.robot_anchor_ang_vel_w[0].detach().cpu().numpy().copy(),
    }
    if action_tensor is not None:
        frame["action"] = action_tensor[0].detach().cpu().numpy().copy()
    return frame


def save_rollout_npz(export_path: pathlib.Path, env, motion_file: str, checkpoint_path: str, frames: list[dict]):
    command: MotionCommand = env.command_manager.get_term("motion")
    robot = env.scene[command.cfg.asset_name]

    export_path.parent.mkdir(parents=True, exist_ok=True)

    rollout = {
        "fps": np.array(1.0 / env.step_dt, dtype=np.float32),
        "step_dt": np.array(env.step_dt, dtype=np.float32),
        "motion_file": np.array(str(pathlib.Path(motion_file).resolve())),
        "checkpoint_path": np.array(str(pathlib.Path(checkpoint_path).resolve())),
        "body_names": np.array(command.cfg.body_names),
        "joint_names": np.array(robot.joint_names),
    }

    for key in frames[0].keys():
        rollout[key] = np.stack([frame[key] for frame in frames], axis=0)

    np.savez(export_path, **rollout)
    print(f"[INFO]: Exported rollout comparison data to: {export_path}")


def print_rollout_summary(export_path: pathlib.Path):
    data = np.load(export_path)

    joint_pos_rmse = float(np.sqrt(np.mean(np.square(data["joint_pos_policy"] - data["joint_pos_ref"]))))
    body_pos_rmse = float(np.sqrt(np.mean(np.square(data["body_pos_policy"] - data["body_pos_ref"]))))
    body_lin_vel_rmse = float(np.sqrt(np.mean(np.square(data["body_lin_vel_policy"] - data["body_lin_vel_ref"]))))
    anchor_pos_rmse = float(np.sqrt(np.mean(np.square(data["anchor_pos_policy"] - data["anchor_pos_ref"]))))

    body_rot_err = quat_error_magnitude(
        torch.from_numpy(data["body_quat_ref"]), torch.from_numpy(data["body_quat_policy"])
    ).mean().item()
    anchor_rot_err = quat_error_magnitude(
        torch.from_numpy(data["anchor_quat_ref"]), torch.from_numpy(data["anchor_quat_policy"])
    ).mean().item()

    print("tracking_summary:")
    print(f"  joint_pos_rmse={joint_pos_rmse:.4f}")
    print(f"  body_pos_rmse={body_pos_rmse:.4f}")
    print(f"  body_lin_vel_rmse={body_lin_vel_rmse:.4f}")
    print(f"  anchor_pos_rmse={anchor_pos_rmse:.4f}")
    print(f"  body_rot_mean_rad={body_rot_err:.4f}")
    print(f"  anchor_rot_mean_rad={anchor_rot_err:.4f}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = str(pathlib.Path(args_cli.motion_file).resolve())
        print(f"[INFO]: Using motion file from CLI: {env_cfg.commands.motion.motion_file}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    sync_policy_observations_from_saved_run(env_cfg, resume_path)

    episode_length_s = set_episode_length_from_motion_file(env_cfg, env_cfg.commands.motion.motion_file)
    print(
        "[INFO]: Episode length set from motion file: "
        f"{episode_length_s:.2f}s ({int(round(episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))} steps)"
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    if args_cli.start_first_frame:
        force_resets_to_first_frame(env.unwrapped)

    # load previously trained model
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as _MotionOnPolicyRunner
    ppo_runner = _MotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    # Merged standing + tracking ONNX export
    if args_cli.standing_checkpoint:
        standing_ckpt = args_cli.standing_checkpoint
        if os.path.isfile(standing_ckpt):
            print(f"[INFO] Merging standing head from: {standing_ckpt}")
            export_motion_policy_merged_as_onnx(
                env.unwrapped,
                ppo_runner.alg.policy,
                tracking_normalizer=ppo_runner.obs_normalizer,
                standing_ckpt_path=standing_ckpt,
                path=export_model_dir,
                filename="policy_merged.onnx",
            )
            attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none",
                                export_model_dir, filename="policy_merged.onnx")
        else:
            print(f"[WARN] Standing checkpoint not found: {standing_ckpt}")

    if args_cli.start_first_frame:
        initialize_to_motion_frame(env.unwrapped, frame_idx=0)

    stop_after_steps = args_cli.max_steps
    if args_cli.export_rollout_npz is not None and stop_after_steps is None:
        motion_command: MotionCommand = env.unwrapped.command_manager.get_term("motion")
        stop_after_steps = motion_command.motion.time_step_total

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    rollout_frames: list[dict] = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if args_cli.export_rollout_npz is not None:
                rollout_frames.append(record_rollout_frame(env.unwrapped, actions))
            # env stepping
            obs, _, _, _ = env.step(actions)
        timestep += 1

        if timestep == 1 or timestep % 50 == 0:
            motion_command = env.unwrapped.command_manager.get_term("motion")
            base_pos = motion_command.robot_anchor_pos_w[0].cpu().numpy()
            print(f"step={timestep:04d} base_pos=[{base_pos[0]:.6f} {base_pos[1]:.6f} {base_pos[2]:.6f}]")

        if stop_after_steps is not None and timestep >= stop_after_steps:
            break

        if args_cli.video and timestep == args_cli.video_length:
            break

    if args_cli.export_rollout_npz is not None and rollout_frames:
        export_path = pathlib.Path(args_cli.export_rollout_npz).expanduser().resolve()
        save_rollout_npz(export_path, env.unwrapped, env_cfg.commands.motion.motion_file, resume_path, rollout_frames)
        print_rollout_summary(export_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
