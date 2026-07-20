from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

try:
    from isaaclab.utils.math import quat_apply_inverse
except:
    from isaaclab.utils.math import quat_apply,quat_inv
    def quat_apply_inverse(a,b):
        return quat_apply(quat_inv(a),b)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_planar_drift_static(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    ref_speed_threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when planar anchor drift grows too large during near-static reference phases.

    If the command has curriculum_drift_termination_threshold configured, the per-stage
    threshold overrides the fixed threshold argument.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    curriculum_threshold = command.current_drift_threshold
    if curriculum_threshold is not None:
        threshold = curriculum_threshold
    ref_planar_speed = torch.linalg.vector_norm(command.anchor_lin_vel_w[:, :2], dim=-1)
    planar_error = torch.linalg.vector_norm(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
    static_mask = ref_planar_speed < ref_speed_threshold
    return static_mask & (planar_error > threshold)


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_anchor_ori_full(
    env: ManagerBasedRLEnv, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate when full anchor orientation error (including yaw) exceeds threshold (rad)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = math_utils.quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return error > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def phase_mode_timeout(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate (time_out) when TRANS_OUT phase completes successfully."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    timeout = getattr(command, "phase_timeout", None)
    if timeout is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return timeout


def _tracking_phase_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    phase = getattr(command, "phase_mode", None)
    if phase is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return phase == 1  # TRACKING only


def bad_anchor_pos_tracking_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """bad_anchor_pos_z_only gated to TRACKING phase."""
    return bad_anchor_pos_z_only(env, command_name, threshold) & _tracking_phase_mask(env, command_name)


def bad_anchor_ori_tracking_only(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """bad_anchor_ori gated to TRACKING phase."""
    return bad_anchor_ori(env, asset_cfg, command_name, threshold) & _tracking_phase_mask(env, command_name)


def bad_anchor_ori_full_tracking_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """bad_anchor_ori_full gated to TRACKING phase."""
    return bad_anchor_ori_full(env, command_name, threshold) & _tracking_phase_mask(env, command_name)


def bad_motion_body_pos_z_only_tracking_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """bad_motion_body_pos_z_only gated to TRACKING phase."""
    return bad_motion_body_pos_z_only(env, command_name, threshold, body_names) & _tracking_phase_mask(env, command_name)
