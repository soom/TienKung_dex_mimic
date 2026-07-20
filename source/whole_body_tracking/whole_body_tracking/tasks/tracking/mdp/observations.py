from __future__ import annotations

from collections.abc import Sequence

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat,  subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

try:
    from isaaclab.utils.math import quat_apply_inverse
except:
    from isaaclab.utils.math import quat_apply,quat_inv
    def quat_apply_inverse(a,b):
        return quat_apply(quat_inv(a),b)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_lin_vel_w.view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_ang_vel_w.view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def motion_ref_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference body positions expressed in the robot's anchor frame.

    Privileged critic observation: gives the critic the reference target positions
    so it can directly evaluate body-level tracking error without inferring it from
    robot_body_pos_b + command. Not available on real robot (requires reference anchor
    global position), so must only be used in the critic observation group.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.body_pos_relative_w,
        command.body_quat_relative_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def joint_pos_ref_residual(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current joint positions minus reference motion joint positions."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return env.scene["robot"].data.joint_pos - command.joint_pos


def joint_vel_ref_residual(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current joint velocities minus reference motion joint velocities."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return env.scene["robot"].data.joint_vel - command.joint_vel


def motion_joint_pos_lookahead(
    env: ManagerBasedEnv, command_name: str, lookahead_steps: int = 2
) -> torch.Tensor:
    """Future reference joint positions k steps ahead (anticipatory obs for velocity tracking)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos_lookahead(lookahead_steps)


def motion_joint_vel_lookahead(
    env: ManagerBasedEnv, command_name: str, lookahead_steps: int = 4
) -> torch.Tensor:
    """Future reference joint velocities k steps ahead — tells policy where velocity is heading."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_vel_lookahead(lookahead_steps)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_phase(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Normalized motion phase in [0, 1] encoded as (sin(2π·φ), cos(2π·φ)).

    Gives the policy a continuous, periodic signal of where it is in the
    reference trajectory, enabling anticipation of upcoming motion phases.
    Shape: (num_envs, 2).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    total = max(command.motion.time_step_total - 1, 1)
    phi = command.time_steps.float() / total  # [0, 1]
    angle = 2.0 * torch.pi * phi
    return torch.stack([angle.sin(), angle.cos()], dim=-1)


def wrist_ee_pos_error_b(
    env: ManagerBasedEnv,
    command_name: str,
    wrist_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Wrist EE position error in anchor-local frame: robot_wrist_pos - ref_wrist_pos.

    Shape: (num_envs, 3 * len(wrist_body_names)).
    Provides the policy a direct signal of how far each wrist is from its reference.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if wrist_body_names is None:
        wrist_body_names = ["wrist_roll_l_link", "wrist_roll_r_link"]
    body_indexes = [i for i, name in enumerate(command.cfg.body_names) if name in wrist_body_names]
    error = command.robot_body_pos_w[:, body_indexes] - command.body_pos_relative_w[:, body_indexes]
    return error.reshape(env.num_envs, -1)


def wrist_ee_ori_error_b(
    env: ManagerBasedEnv,
    command_name: str,
    wrist_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Wrist EE orientation error magnitude (scalar per wrist).

    Shape: (num_envs, len(wrist_body_names)).
    """
    from isaaclab.utils.math import quat_error_magnitude

    command: MotionCommand = env.command_manager.get_term(command_name)
    if wrist_body_names is None:
        wrist_body_names = ["wrist_roll_l_link", "wrist_roll_r_link"]
    body_indexes = [i for i, name in enumerate(command.cfg.body_names) if name in wrist_body_names]
    error = quat_error_magnitude(
        command.body_quat_relative_w[:, body_indexes],
        command.robot_body_quat_w[:, body_indexes],
    )
    return error.reshape(env.num_envs, -1)


def motion_anchor_lookahead(
    env: ManagerBasedEnv,
    command_name: str,
    lookahead_steps: Sequence[int] = (8,),
    use_obs_pre_shift: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_obs: list[torch.Tensor] = []
    env_origin_z = env.scene.env_origins[:, 2:3]

    for step in lookahead_steps:
        anchor_pos_w = command.anchor_pos_lookahead(step, use_obs_pre_shift=use_obs_pre_shift)
        anchor_quat_w = command.anchor_quat_lookahead(step, use_obs_pre_shift=use_obs_pre_shift)
        anchor_lin_vel_w = command.anchor_lin_vel_lookahead(step, use_obs_pre_shift=use_obs_pre_shift)
        anchor_ang_vel_w = command.anchor_ang_vel_lookahead(step, use_obs_pre_shift=use_obs_pre_shift)

        _, anchor_quat_b = subtract_frame_transforms(
            command.robot_anchor_pos_w,
            command.robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        anchor_ori_6d = matrix_from_quat(anchor_quat_b)[..., :2].reshape(env.num_envs, -1)
        anchor_lin_vel_b = quat_apply_inverse(command.robot_anchor_quat_w, anchor_lin_vel_w)
        anchor_ang_vel_b = quat_apply_inverse(command.robot_anchor_quat_w, anchor_ang_vel_w)
        anchor_height = anchor_pos_w[:, 2:3] - env_origin_z

        future_obs.extend([anchor_height, anchor_ori_6d, anchor_lin_vel_b, anchor_ang_vel_b])

    return torch.cat(future_obs, dim=-1)


def phase_mode_onehot(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return a (num_envs, 2) one-hot encoding of the current phase mode.

    Phases (indices): 0=STAND, 1=TRACKING.
    If the `MotionCommand` does not provide `phase_mode`, default to STAND.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_envs = env.num_envs

    mode = getattr(command, "phase_mode", None)
    if mode is None:
        idx = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    else:
        idx = torch.as_tensor(mode, dtype=torch.long, device=env.device).view(-1).clamp(min=0, max=1)

    onehot = torch.nn.functional.one_hot(idx, num_classes=2).to(torch.float32)
    return onehot
