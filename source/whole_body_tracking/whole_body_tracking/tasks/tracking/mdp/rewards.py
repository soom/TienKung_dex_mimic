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


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, joint_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    error = torch.mean(torch.square(command.joint_pos[:, joint_indexes] - command.robot_joint_pos[:, joint_indexes]), dim=-1)
    return torch.exp(-error / std**2)


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


def motion_joint_velocity_error_linear(
    env: ManagerBasedRLEnv, command_name: str, joint_names: list[str] | None = None
) -> torch.Tensor:
    """Mean absolute joint velocity error (positive scalar). Use with negative weight as a penalty.

    Unlike the exp kernel which saturates when error is large, this provides a constant gradient
    regardless of error magnitude — critical when error_joint_vel >> std of the exp version.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_indexes = _get_joint_indexes(command, joint_names)
    error = torch.mean(
        torch.abs(command.joint_vel[:, joint_indexes] - command.robot_joint_vel[:, joint_indexes]), dim=-1
    )
    return error


def action_jerk_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action differences (jerk): ||a_t - 2*a_{t-1} + a_{t-2}||^2.

    Targets true trajectory discontinuities rather than velocity, so it doesn't
    penalize smooth ramps. Requires caching a_{t-2} on the env object.
    """
    a_t = env.action_manager.action
    a_t1 = env.action_manager.prev_action
    if not hasattr(env, "_jerk_prev_prev_action"):
        env._jerk_prev_prev_action = torch.zeros_like(a_t)
    a_t2 = env._jerk_prev_prev_action
    env._jerk_prev_prev_action = a_t1.clone()
    return torch.sum(torch.square(a_t - 2.0 * a_t1 + a_t2), dim=1)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def expected_foot_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    foot_names: list[str],
    contact_force_threshold: float,
    ref_height_threshold: float = 0.05,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    ref_contact = command.foot_contact_label(foot_names, height_threshold=ref_height_threshold)
    actual_contact_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    actual_contact = actual_contact_force > contact_force_threshold
    return torch.sum(ref_contact * actual_contact, dim=-1)


def unexpected_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    foot_names: list[str],
    contact_force_threshold: float,
    ref_height_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise foot contact that occurs when the reference foot is in the air.

    This directly targets spurious steps: after a stepping motion the policy
    sometimes lifts and re-plants the foot even though the reference is already
    stationary or airborne. expected_foot_contact_reward only rewards correct
    contact; this term penalises the opposite case (actual contact when
    reference says air), closing the asymmetry.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    ref_contact = command.foot_contact_label(foot_names, height_threshold=ref_height_threshold)
    actual_contact_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    actual_contact = (actual_contact_force > contact_force_threshold).float()
    ref_air = (1.0 - ref_contact.float())  # 1 where reference foot is in the air
    return torch.sum(ref_air * actual_contact, dim=-1)


def motion_wrist_pos_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    wrist_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Wrist EE position tracking reward using relative body position (anchor-aligned).

    Uses the same relative-body-position metric as motion_relative_body_position_error_exp
    but with a tighter std and restricted to wrist links only, providing a stronger
    learning signal for distal arm joints whose joint-space reference is all-zero.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if wrist_body_names is None:
        wrist_body_names = ["wrist_roll_l_link", "wrist_roll_r_link"]
    body_indexes = _get_body_indexes(command, wrist_body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_wrist_ori_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    wrist_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Wrist EE orientation tracking reward."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if wrist_body_names is None:
        wrist_body_names = ["wrist_roll_l_link", "wrist_roll_r_link"]
    body_indexes = _get_body_indexes(command, wrist_body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def body_pair_clearance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    left_body_names: list[str],
    right_body_names: list[str],
    min_distance: float = 0.08,
) -> torch.Tensor:
    """Penalize selected left/right body pairs when they get too close.

    This is a lightweight proxy for self-collision clearance. It intentionally
    uses body origins instead of geom distances so it remains cheap in RL.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    left_indexes = _get_body_indexes(command, left_body_names)
    right_indexes = _get_body_indexes(command, right_body_names)
    if not left_indexes or not right_indexes:
        return torch.zeros(env.num_envs, device=env.device)
    left_pos = command.robot_body_pos_w[:, left_indexes]
    right_pos = command.robot_body_pos_w[:, right_indexes]
    distances = torch.linalg.vector_norm(left_pos[:, :, None, :] - right_pos[:, None, :, :], dim=-1)
    violation = torch.clamp(min_distance - distances, min=0.0)
    return torch.mean(torch.square(violation), dim=(1, 2))


def action_smoothness_l2(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
) -> torch.Tensor:
    """Action rate penalty restricted to the specified joints."""
    robot = env.scene["robot"]
    joint_ids = robot.find_joints(joint_names, preserve_order=True)[0]
    a_t = env.action_manager.action[:, joint_ids]
    a_t1 = env.action_manager.prev_action[:, joint_ids]
    return torch.sum(torch.square(a_t - a_t1), dim=-1)


def wrist_action_smoothness_l2(
    env: ManagerBasedRLEnv,
    wrist_joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Action rate penalty restricted to wrist/elbow-yaw joints (low stiffness, prone to oscillation)."""
    if wrist_joint_names is None:
        wrist_joint_names = [
            "elbow_yaw_l_joint", "elbow_yaw_r_joint",
            "wrist_pitch_l_joint", "wrist_pitch_r_joint",
            "wrist_roll_l_joint", "wrist_roll_r_joint",
        ]
    return action_smoothness_l2(env, wrist_joint_names)


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


def pelvis_lateral_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["pelvis"]),
) -> torch.Tensor:
    """Penalize absolute lateral (Y-axis world frame) velocity of the pelvis.

    Targets left-right swaying during standing/slow motion. Uses L2 on the
    lateral component only so forward locomotion is not penalised.
    """
    body_lin_vel_w = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (N, 1, 3)
    lateral_vel = body_lin_vel_w[:, :, 1]  # Y axis
    return torch.mean(torch.square(lateral_vel), dim=-1)


def pelvis_lateral_oscillation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["pelvis"]),
    window: int = 40,
    min_crossings: int = 2,
    min_vel_threshold: float = 0.03,
    penalty: float = 10000.0,
) -> torch.Tensor:
    """Massive penalty triggered when periodic lateral sway is detected.

    Detects periodic left-right oscillation by counting zero-crossings of the
    pelvis lateral (Y-axis) velocity within a sliding window. If the number of
    sign flips exceeds min_crossings within `window` steps, the env is judged
    to be in a periodic sway cycle and receives a flat `penalty`.

    At 50 Hz control, window=40 covers 0.8 s. min_crossings=2 catches
    oscillations >= 1.25 Hz (typical standing sway is 1-3 Hz).

    min_vel_threshold filters out near-zero noise crossings — only crossings
    where the velocity magnitude exceeds this value are counted.

    The penalty is a fixed scalar (not L2) so the gradient signal is a hard
    discrete trigger rather than a soft continuous push.
    """
    body_lin_vel_w = env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    lateral_vel = body_lin_vel_w[:, 0, 1]  # (N,) Y-axis velocity of pelvis

    # Initialise circular buffer on first call
    if not hasattr(env, "_lateral_vel_buf"):
        env._lateral_vel_buf = torch.zeros(env.num_envs, window, device=env.device)
        env._lateral_vel_buf_idx = 0

    # Write current velocity into buffer
    env._lateral_vel_buf[:, env._lateral_vel_buf_idx] = lateral_vel
    env._lateral_vel_buf_idx = (env._lateral_vel_buf_idx + 1) % window

    # Count zero-crossings: consecutive samples with opposite signs,
    # both exceeding the noise threshold
    buf = env._lateral_vel_buf  # (N, window)
    above = buf.abs() > min_vel_threshold  # mask: large enough to count
    sign = buf.sign()                       # -1, 0, +1

    # Shift by 1 along time axis to get consecutive pairs
    sign_cur  = sign[:, 1:]   # (N, window-1)
    sign_prev = sign[:, :-1]
    above_cur  = above[:, 1:]
    above_prev = above[:, :-1]

    crossing = (sign_cur * sign_prev < 0) & above_cur & above_prev  # (N, window-1)
    num_crossings = crossing.sum(dim=-1).float()  # (N,)

    triggered = (num_crossings >= min_crossings).float()
    return triggered * penalty


def pelvis_lateral_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Penalize lateral (Y-axis) displacement of pelvis relative to reference.

    Complements pelvis_lateral_vel_penalty: the velocity penalty suppresses fast
    swaying, but slow large-amplitude sway can still accumulate. This penalty
    directly penalises the Y-axis offset between the robot pelvis and the
    reference pelvis position (both expressed in the anchor-relative frame used
    by body_pos_relative_w), so it applies regardless of sway speed.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    pelvis_idx = [i for i, name in enumerate(command.cfg.body_names) if name == "pelvis"]
    if not pelvis_idx:
        return torch.zeros(env.num_envs, device=env.device)
    idx = pelvis_idx[0]
    ref_y = command.body_pos_relative_w[:, idx, 1]   # reference pelvis Y
    rob_y = command.robot_body_pos_w[:, idx, 1]      # robot pelvis Y
    return torch.square(rob_y - ref_y)


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


def wrist_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    wrist_joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize absolute joint velocity of wrist/elbow-yaw joints.

    action_smoothness penalises action *changes* (policy output rate), but
    wrist rotation oscillation also shows up as high joint *velocity* even when
    the policy output is smooth (driven by low stiffness + inertia). This
    penalty directly suppresses joint-level spinning.
    """
    if wrist_joint_names is None:
        wrist_joint_names = [
            "elbow_yaw_l_joint", "elbow_yaw_r_joint",
            "wrist_pitch_l_joint", "wrist_pitch_r_joint",
            "wrist_roll_l_joint", "wrist_roll_r_joint",
        ]
    robot = env.scene["robot"]
    joint_ids = robot.find_joints(wrist_joint_names, preserve_order=True)[0]
    joint_vel = robot.data.joint_vel[:, joint_ids]
    return torch.sum(torch.square(joint_vel), dim=-1)


def standing_feet_planar_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_body_names: list[str] | None = None,
    drift_tolerance: float = 0.01,
) -> torch.Tensor:
    """Penalize foot body XY drift from reference during standing.

    Unlike anchor_planar_drift_penalty (which tracks pelvis XY drift), this
    penalizes each foot's absolute XY position deviation from its reference.
    This is the strongest anti-stepping signal: if the robot lifts or slides
    a foot, the penalty spikes immediately.

    Uses L1 (MAE) for constant gradient at all drift magnitudes.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if foot_body_names is None:
        foot_body_names = ["ankle_roll_l_link", "ankle_roll_r_link"]

    # Get body indices for the named foot bodies
    foot_indices = [i for i, name in enumerate(command.cfg.body_names) if name in foot_body_names]
    if not foot_indices:
        return torch.zeros(env.num_envs, device=env.device)

    # Global XY positions: reference vs actual
    ref_xy = command.body_pos_w[:, foot_indices, :2]      # (N, n_feet, 2)
    robot_xy = command.robot_body_pos_w[:, foot_indices, :2]  # (N, n_feet, 2)

    # L1 drift per foot, averaged across feet
    drift = torch.linalg.vector_norm(robot_xy - ref_xy, dim=-1)  # (N, n_feet)
    drift_excess = torch.clamp(drift - drift_tolerance, min=0.0)
    return torch.mean(drift_excess, dim=-1)  # average over feet


def standing_foot_height_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_body_names: list[str] | None = None,
    height_tolerance: float = 0.01,
) -> torch.Tensor:
    """Penalize foot body Z position ABOVE reference (foot lifting = stepping).

    Only penalizes upward deviation — the foot going below reference (into
    ground) is not penalized (that's just contact deformation).

    Uses L1 for constant gradient — every mm of lift hurts proportionally.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if foot_body_names is None:
        foot_body_names = ["ankle_roll_l_link", "ankle_roll_r_link"]

    foot_indices = [i for i, name in enumerate(command.cfg.body_names) if name in foot_body_names]
    if not foot_indices:
        return torch.zeros(env.num_envs, device=env.device)

    ref_z = command.body_pos_w[:, foot_indices, 2]      # (N, n_feet)
    robot_z = command.robot_body_pos_w[:, foot_indices, 2]  # (N, n_feet)
    lift = torch.clamp(robot_z - ref_z - height_tolerance, min=0.0)
    return torch.mean(lift, dim=-1)  # average L1 lift over feet


def standing_foot_planar_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_body_names: list[str] | None = None,
    vel_tolerance: float = 0.02,
) -> torch.Tensor:
    """Penalize foot body XY velocity regardless of contact state.

    Unlike foot_slip_penalty (which requires contact force > threshold),
    this is always active.  ANY foot XY velocity is penalized — this
    catches both sliding (foot on ground) and stepping (foot in air).

    Uses L1 for constant gradient at all velocity levels.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if foot_body_names is None:
        foot_body_names = ["ankle_roll_l_link", "ankle_roll_r_link"]

    foot_indices = [i for i, name in enumerate(command.cfg.body_names) if name in foot_body_names]
    if not foot_indices:
        return torch.zeros(env.num_envs, device=env.device)

    # robot_body_lin_vel_w is in global frame
    robot = env.scene["robot"]
    body_lin_vel_w = robot.data.body_lin_vel_w[:, foot_indices, :2]  # (N, n_feet, 2)
    foot_speed = torch.linalg.vector_norm(body_lin_vel_w, dim=-1)    # (N, n_feet)
    vel_excess = torch.clamp(foot_speed - vel_tolerance, min=0.0)
    return torch.mean(vel_excess, dim=-1)  # average over feet


def standing_pelvis_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.08,
) -> torch.Tensor:
    """Reward pelvis Z height matching reference. Ignores XY drift.

    Standing is fundamentally about keeping the pelvis at the right HEIGHT.
    XY position is irrelevant — the robot can drift in XY as long as it
    stays upright at the correct height.

    Uses exp(-error²/std²) so near-zero error has strong gradient.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
    return torch.exp(-torch.square(error) / (std * std))


def standing_torso_upright_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.15,
) -> torch.Tensor:
    """Reward pelvis orientation matching reference (upright).

    The pelvis quaternion must stay close to identity (flat, not tilted).
    This is the single most important standing constraint — if the pelvis
    tilts, the robot will fall.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return torch.exp(-torch.square(error) / (std * std))


# ═══════════════════════════════════════════════════════════════════════════════
# Standing quality rewards (velocity-based, no position tracking)
# ═══════════════════════════════════════════════════════════════════════════════


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward XY linear velocity being close to zero. Standing = not moving."""
    robot = env.scene["robot"]
    vel_xy = robot.data.root_lin_vel_w[:, :2]  # (N, 2)
    error_sq = torch.sum(torch.square(vel_xy), dim=-1)
    return torch.exp(-error_sq / (std * std))


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward Z angular velocity being close to zero. Standing = not rotating."""
    robot = env.scene["robot"]
    ang_vel_z = robot.data.root_ang_vel_w[:, 2]  # (N,)
    return torch.exp(-torch.square(ang_vel_z) / (std * std))


def flat_orientation_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize pelvis tilt. Gravity in body frame should be [0,0,-1].

    The deviation of the projected gravity vector from the vertical indicates
    how much the pelvis is tilted.  Uses L2 — strong gradient for large tilts.
    """
    robot = env.scene["robot"]
    proj_grav = robot.data.projected_gravity_b  # (N, 3)
    # target: gravity pointing straight down in body frame = [0, 0, -1]
    tilt_error = torch.sum(torch.square(proj_grav[:, :2]), dim=-1)  # XY deviation
    return tilt_error


def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion",
) -> torch.Tensor:
    """L1 penalty for joint position deviation from the reference pose.

    Uses the motion command's joint reference (which is the default standing
    pose in the NPZ) as the zero point.  Penalizes ANY deviation.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    joint_ids = robot.find_joints(asset_cfg.joint_names, preserve_order=True)[0]
    joint_pos = robot.data.joint_pos[:, joint_ids]     # current
    joint_ref = command.joint_pos[:, joint_ids]         # NPZ reference
    return torch.sum(torch.abs(joint_pos - joint_ref), dim=-1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (policy output smoothness)."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def joint_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize joint acceleration (L2 on joint velocity change)."""
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.joint_acc), dim=-1)


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for episode termination (the termination buffer)."""
    return env.termination_manager.terminated.float()


def energy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize energy consumption = sum(|torque * velocity|)."""
    robot = env.scene["robot"]
    return torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=-1)


def undesired_contacts_standing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize contacts on undesired bodies (knees, elbows, pelvis, etc.)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return torch.sum((force > threshold).float(), dim=-1)
