from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_joint_indexes(command: MotionCommand, joint_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.robot.joint_names) if (joint_names is None) or (name in joint_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_linear(
    env: ManagerBasedRLEnv, command_name: str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.linalg.vector_norm(
        command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes],
        dim=-1,
    )
    return error.mean(-1)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, joint_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    error = torch.mean(torch.square(command.joint_pos[:, joint_indexes] - command.robot_joint_pos[:, joint_indexes]), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_body_angular_velocity_error_linear(
    env: ManagerBasedRLEnv, command_name: str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.linalg.vector_norm(
        command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes],
        dim=-1,
    )
    return error.mean(-1)


def motion_joint_position_error_linear(
    env: ManagerBasedRLEnv, command_name: str, std: float, joint_names: list[str] | None = None
) -> torch.Tensor:
    """Bounded linear joint position reward using mean absolute error.

    Returns max(0, 1 - mae / std) so the term stays positive and linear near the target
    without creating the "terminate early to avoid penalty" failure mode of a pure negative cost.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    error = torch.mean(
        torch.abs(command.joint_pos[:, joint_indexes] - command.robot_joint_pos[:, joint_indexes]), dim=-1
    )
    return torch.clamp(1.0 - error / std, min=0.0)


def motion_joint_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, joint_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    error = torch.mean(
        torch.square(command.joint_vel[:, joint_indexes] - command.robot_joint_vel[:, joint_indexes]), dim=-1
    )
    return torch.exp(-error / std**2)


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_force_threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    in_contact = (contact_force > contact_force_threshold).float()

    body_lin_vel_w = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    planar_speed_sq = torch.sum(torch.square(body_lin_vel_w), dim=-1)

    contact_count = torch.sum(in_contact, dim=-1).clamp(min=1.0)
    return torch.sum(planar_speed_sq * in_contact, dim=-1) / contact_count


def anchor_planar_drift_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    ref_speed_threshold: float = 0.2,
    drift_tolerance: float = 0.03,
) -> torch.Tensor:
    """Penalize planar anchor drift when the reference anchor is nearly static.

    This targets the failure mode where the policy keeps balance by taking
    forward recovery steps and gradually walking away from the reference.
    During motions with meaningful reference translation, the term fades out.
    A small dead-zone avoids punishing tiny stationary jitter, while larger
    drift receives a stronger quadratic penalty.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_planar_speed = torch.linalg.vector_norm(command.anchor_lin_vel_w[:, :2], dim=-1)
    static_gate = torch.clamp(1.0 - ref_planar_speed / max(ref_speed_threshold, 1.0e-6), min=0.0, max=1.0)
    planar_error = torch.linalg.vector_norm(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)
    drift_excess = torch.clamp(planar_error - drift_tolerance, min=0.0)
    return torch.square(drift_excess) * static_gate


def anchor_static_planar_velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    ref_speed_threshold: float = 0.1,
    vel_tolerance: float = 0.03,
) -> torch.Tensor:
    """Penalize robot anchor planar velocity when the reference anchor is nearly static.

    anchor_planar_drift_penalty acts on accumulated displacement. This term
    acts earlier by discouraging the policy from initiating recovery steps or
    slow walking during near-static phases.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_planar_speed = torch.linalg.vector_norm(command.anchor_lin_vel_w[:, :2], dim=-1)
    static_gate = torch.clamp(1.0 - ref_planar_speed / max(ref_speed_threshold, 1.0e-6), min=0.0, max=1.0)
    robot_planar_speed = torch.linalg.vector_norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    vel_excess = torch.clamp(robot_planar_speed - vel_tolerance, min=0.0)
    return torch.square(vel_excess) * static_gate


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (policy output smoothness)."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def undesired_contacts_standing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize contacts on undesired bodies (knees, elbows, pelvis, etc.)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return torch.sum((force > threshold).float(), dim=-1)
