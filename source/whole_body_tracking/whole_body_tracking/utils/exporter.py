# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
import torch.nn as nn

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    print("[exporter] Starting ONNX export...")
    policy_exporter.export(path, filename)
    print("[exporter] ONNX export complete.")


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.eval()
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        with torch.no_grad():
            torch.onnx.export(
                self,
                (obs, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "time_step"],
                output_names=[
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
                dynamic_axes={},
            )

def export_motion_policy_merged_as_onnx(
    env: ManagerBasedRLEnv,
    tracking_actor_critic,
    tracking_normalizer,
    standing_ckpt_path: str,
    path: str,
    filename="policy_merged.onnx",
    verbose=False,
):
    """Export a merged ONNX that routes between standing and tracking models via phase_mode.

    Inputs:  obs (policy dims), phase_mode (2D onehot [STAND, TRACKING]), time_step (int)
    Outputs: actions, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

    Routing:  action = standing_action * phase_mode[0] + tracking_action * phase_mode[1]
    """
    import copy
    from rsl_rl.modules import ActorCritic

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # ── Load standing model ──────────────────────────────────────────────────
    ckpt = torch.load(standing_ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    obs_dim = tracking_actor_critic.actor[0].in_features
    act_dim = tracking_actor_critic.actor[-1].out_features
    # Infer critic obs dim from checkpoint (standing has privileged obs, ~612 vs policy ~214)
    critic_obs_dim = sd.get("critic.0.weight", torch.zeros(1, obs_dim)).shape[1]

    standing_ac = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=critic_obs_dim,
        num_actions=act_dim,
        actor_hidden_dims=[768, 384, 192],
        critic_hidden_dims=[768, 384, 192],
        activation="elu",
        init_noise_std=0.5,
    )
    standing_ac.load_state_dict(sd, strict=False)
    standing_ac.eval()

    # Standing normalizer
    standing_norm = None
    if "obs_norm_state_dict" in ckpt:
        from rsl_rl.modules import EmpiricalNormalization
        norm_state = ckpt["obs_norm_state_dict"]
        standing_norm = EmpiricalNormalization(shape=(obs_dim,))
        standing_norm.load_state_dict(norm_state)
    else:
        standing_norm = copy.deepcopy(tracking_normalizer) if tracking_normalizer is not None else nn.Identity()

    # ── Build merged exporter ────────────────────────────────────────────────
    policy_exporter = _OnnxMergedPolicyExporter(
        env, tracking_actor_critic, standing_ac, tracking_normalizer, standing_norm, verbose
    )
    policy_exporter.export(path, filename)
    print(f"[exporter] Merged ONNX export complete → {os.path.join(path, filename)}")


class _OnnxMergedPolicyExporter(nn.Module):
    """Merged standing + tracking ONNX exporter with phase_mode embedded in obs.

    Input:  obs (raw_dim + 2, last 2 = phase_mode [STAND, TRACKING] onehot), time_step
    Output: actions, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

    Compatible with the same (obs, time_step) interface as the single-head ONNX.
    sim2sim / FSM append phase_mode to the raw observation, same as before.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        tracking_ac,
        standing_ac,
        tracking_norm,
        standing_norm,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.tracking_actor = copy.deepcopy(tracking_ac.actor)
        self.standing_actor = copy.deepcopy(standing_ac.actor)
        self.tracking_norm = copy.deepcopy(tracking_norm) if tracking_norm is not None else nn.Identity()
        self.standing_norm = copy.deepcopy(standing_norm) if standing_norm is not None else nn.Identity()

        cmd: MotionCommand = env.command_manager.get_term("motion")
        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, obs: torch.Tensor, time_step: torch.Tensor):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)

        # Split: last 2 dims = phase_mode onehot [STAND, TRACKING]
        raw_obs = obs[..., :-2]
        phase_mode = obs[..., -2:]

        t_normed = self.tracking_norm(raw_obs)
        s_normed = self.standing_norm(raw_obs)

        t_action = self.tracking_actor(t_normed)
        s_action = self.standing_actor(s_normed)

        # Route: s_action when phase_mode[0]=1, t_action when phase_mode[1]=1
        action = s_action * phase_mode[:, 0:1] + t_action * phase_mode[:, 1:2]

        return (
            action,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path: str, filename: str):
        self.eval()
        self.to("cpu")
        raw_dim = self.tracking_actor[0].in_features
        obs = torch.zeros(1, raw_dim + 2)  # raw obs + phase_mode onehot
        obs[:, -1] = 1.0  # default: TRACKING [0, 1]
        time_step = torch.zeros(1, 1)
        with torch.no_grad():
            torch.onnx.export(
                self,
                (obs, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "time_step"],
                output_names=[
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
                dynamic_axes={},
            )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    # Always read kp/kd from the nominal actuator config — NOT from data.joint_stiffness,
    # which carries domain-randomization noise that breaks sim2sim fidelity.
    import re
    joint_names_list = env.scene["robot"].data.joint_names
    kp_map: dict[str, float] = {}
    kd_map: dict[str, float] = {}
    effort_map: dict[str, float] = {}
    for actuator in env.scene["robot"].cfg.actuators.values():
        stiff = actuator.stiffness if isinstance(actuator.stiffness, dict) else {p: actuator.stiffness for p in actuator.joint_names_expr}
        damp = actuator.damping if isinstance(actuator.damping, dict) else {p: actuator.damping for p in actuator.joint_names_expr}
        effort = actuator.effort_limit_sim if isinstance(actuator.effort_limit_sim, dict) else {
            p: actuator.effort_limit_sim for p in actuator.joint_names_expr
        }
        for jname in joint_names_list:
            for pattern, val in stiff.items():
                if re.fullmatch(pattern, jname):
                    kp_map[jname] = float(val)
            for pattern, val in damp.items():
                if re.fullmatch(pattern, jname):
                    kd_map[jname] = float(val)
            for pattern, val in effort.items():
                if re.fullmatch(pattern, jname):
                    effort_map[jname] = float(val)
    joint_stiffness = [kp_map.get(jname, 0.0) for jname in joint_names_list]
    joint_damping = [kd_map.get(jname, 0.0) for jname in joint_names_list]
    joint_effort_limit = [effort_map.get(jname, 0.0) for jname in joint_names_list]

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": joint_stiffness,
        "joint_damping": joint_damping,
        "joint_effort_limit": joint_effort_limit,
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
