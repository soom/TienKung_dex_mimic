"""Simple full-body constrained rewards env config for Walker C1 (standalone).

独立配置，直接继承 TrackingEnvCfg，不依赖 flat_env_cfg.py。
"""

from whole_body_tracking.tasks.tracking.mdp.actuators import DelayedImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.walker_c1 import WALKER_C1_ACTION_SCALE, WALKER_C1_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

# EE links for termination (ankles + wrists)
_EE_LINKS = ["L_ankle_roll_link", "R_ankle_roll_link", "L_wrist_roll_link", "R_wrist_roll_link"]
_UNDESIRED_CONTACT_EXCLUDE = ["L_ankle_roll_link", "R_ankle_roll_link"]
_UNDESIRED_CONTACT_REGEX = r"^(?!(?:" + "|".join(_UNDESIRED_CONTACT_EXCLUDE) + r")$).+"


@configclass
class WalkerC1SimpleEnvCfg(TrackingEnvCfg):
    """Simple Walker C1 env with full-body constrained joint rewards (standalone)."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot ──────────────────────────────────────────────────────────
        self.scene.robot = WALKER_C1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Actuators: wrap with position-only delay ───────────────────────
        for group_name, actuator_cfg in self.scene.robot.actuators.items():
            self.scene.robot.actuators[group_name] = DelayedImplicitActuatorCfg(
                joint_names_expr=actuator_cfg.joint_names_expr,
                effort_limit_sim=actuator_cfg.effort_limit_sim,
                velocity_limit_sim=actuator_cfg.velocity_limit_sim,
                stiffness=actuator_cfg.stiffness,
                damping=actuator_cfg.damping,
                min_delay=0,
                max_delay=3,
            )

        # ── Action: residual joint position ────────────────────────────────
        _scale = {k: v * 0.75 for k, v in WALKER_C1_ACTION_SCALE.items()}
        _scale["L_shoulder_roll_joint"] *= 1.0
        _scale["R_shoulder_roll_joint"] *= 1.0
        _scale["L_shoulder_yaw_joint"] *= 0.85
        _scale["R_shoulder_yaw_joint"] *= 0.85
        _scale["L_elbow_pitch_joint"] *= 0.65
        _scale["R_elbow_pitch_joint"] *= 0.65
        _scale["L_elbow_yaw_joint"] *= 0.
        _scale["R_elbow_yaw_joint"] *= 0.
        _scale["L_wrist_pitch_joint"] *= 0.
        _scale["R_wrist_pitch_joint"] *= 0.
        _scale["L_wrist_roll_joint"] *= 0.
        _scale["R_wrist_roll_joint"] *= 0.
        _scale["waist_yaw_joint"] *= 0.5
        _scale["waist_roll_joint"] *= 0.5
        _scale["waist_pitch_joint"] *= 0.35
        self.actions.joint_pos = mdp.ResidualJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=_scale,
            command_name="motion",
        )

        # ── Observations: policy ───────────────────────────────────────────
        self.observations.policy.motion_phase = ObsTerm(
            func=mdp.motion_phase,
            params={"command_name": "motion"},
        )
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_ref_residual,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_ref_residual,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-1.0, n_max=1.0),
        )
        self.observations.policy.base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        self.observations.policy.vel_lookahead_near = ObsTerm(
            func=mdp.motion_joint_vel_lookahead,
            params={"command_name": "motion", "lookahead_steps": 2},
        )
        self.observations.policy.pos_lookahead_near = ObsTerm(
            func=mdp.motion_joint_pos_lookahead,
            params={"command_name": "motion", "lookahead_steps": 2},
        )

        # ── Observations: critic (privileged) ──────────────────────────────
        self.observations.critic.motion_phase = ObsTerm(
            func=mdp.motion_phase,
            params={"command_name": "motion"},
        )
        self.observations.critic.joint_pos = ObsTerm(
            func=mdp.joint_pos_ref_residual,
            params={"command_name": "motion"},
        )
        self.observations.critic.joint_vel = ObsTerm(
            func=mdp.joint_vel_ref_residual,
            params={"command_name": "motion"},
        )
        self.observations.critic.vel_lookahead_near = ObsTerm(
            func=mdp.motion_joint_vel_lookahead,
            params={"command_name": "motion", "lookahead_steps": 2},
        )
        self.observations.critic.pos_lookahead_near = ObsTerm(
            func=mdp.motion_joint_pos_lookahead,
            params={"command_name": "motion", "lookahead_steps": 2},
        )
        self.observations.critic.anchor_pos_error = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
        )
        self.observations.critic.anchor_lin_vel = ObsTerm(
            func=mdp.robot_anchor_lin_vel_w,
            params={"command_name": "motion"},
        )
        self.observations.critic.anchor_ang_vel = ObsTerm(
            func=mdp.robot_anchor_ang_vel_w,
            params={"command_name": "motion"},
        )
        self.observations.critic.anchor_lookahead = ObsTerm(
            func=mdp.motion_anchor_lookahead,
            params={"command_name": "motion", "lookahead_steps": (4, 8)},
        )
        self.observations.critic.ref_body_pos = ObsTerm(
            func=mdp.motion_ref_body_pos_b,
            params={"command_name": "motion"},
        )

        # ── Commands ───────────────────────────────────────────────────────
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.joint_position_range = (-0.3, 0.3)
        self.commands.motion.start_from_first_frame = True
        self.commands.motion.body_names = [
            "base_link",                 #  0
            "L_hip_pitch_link",          #  1
            "R_hip_pitch_link",          #  2
            "waist_yaw_link",            #  3
            "L_hip_roll_link",           #  4
            "R_hip_roll_link",           #  5
            "waist_pitch_link",          #  6
            "L_hip_yaw_link",            #  7
            "R_hip_yaw_link",            #  8
            "waist_roll_link",           #  9
            "L_knee_pitch_link",         # 10
            "R_knee_pitch_link",         # 11
            "L_shoulder_pitch_link",     # 12
            "R_shoulder_pitch_link",     # 13
            "L_ankle_pitch_link",        # 14
            "R_ankle_pitch_link",        # 15
            "L_shoulder_roll_link",      # 16
            "R_shoulder_roll_link",      # 17
            "L_ankle_roll_link",         # 18
            "R_ankle_roll_link",         # 19
            "L_shoulder_yaw_link",       # 20
            "R_shoulder_yaw_link",       # 21
            "L_elbow_pitch_link",        # 22
            "R_elbow_pitch_link",        # 23
            "L_elbow_yaw_link",          # 24
            "R_elbow_yaw_link",          # 25
            "L_wrist_pitch_link",        # 26
            "R_wrist_pitch_link",        # 27
            "L_wrist_roll_link",         # 28
            "R_wrist_roll_link",         # 29
        ]

        # ── Domain randomization ───────────────────────────────────────────
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-0.04, 0.04), "y": (-0.08, 0.08), "z": (-0.08, 0.08)},
            },
        )
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.85, 1.15),
                "damping_distribution_params": (0.85, 1.15),
                "operation": "scale",
            },
        )
        self.events.physics_material.params["static_friction_range"] = (0.4, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.5)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.005)
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.6, 1.1),
                "velocity_range": (-0.3, 0.3),
            },
        )
        self.events.randomize_rigid_body_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )

        # ═══════════════════════════════════════════════════════════════════════
        # Rewards — 全身约束 (full-body constrained)
        # ═══════════════════════════════════════════════════════════════════════

        # ── Contact penalties ───────────────────────────────────────────
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[_UNDESIRED_CONTACT_REGEX]),
                "threshold": 1.0,
            },
        )
        self.rewards.wrist_contact_penalty = RewTerm(
            func=mdp.undesired_contacts,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["L_wrist_roll_link", "R_wrist_roll_link"],
                ),
                "threshold": 0.5,
            },
        )
        self.rewards.foot_slip = RewTerm(
            func=mdp.foot_slip_penalty,
            weight=-5.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["L_ankle_roll_link", "R_ankle_roll_link"],
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["L_ankle_roll_link", "R_ankle_roll_link"],
                ),
                "contact_force_threshold": 0.5,
            },
        )

        # ── Anchor tracking ─────────────────────────────────────────────
        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.motion_global_anchor_position_error_exp,
            weight=3.0,
            params={"command_name": "motion", "std": 0.4},
        )
        self.rewards.anchor_planar_drift = RewTerm(
            func=mdp.anchor_planar_drift_penalty,
            weight=-8.0,
            params={
                "command_name": "motion",
                "ref_speed_threshold": 0.05,
                "drift_tolerance": 0.02,
            },
        )
        self.rewards.anchor_planar_drift_ungated = RewTerm(
            func=mdp.anchor_planar_drift_penalty,
            weight=-5.0,
            params={
                "command_name": "motion",
                "ref_speed_threshold": 999.0,
                "drift_tolerance": 0.05,
            },
        )
        self.rewards.anchor_static_planar_vel = RewTerm(
            func=mdp.anchor_static_planar_velocity_penalty,
            weight=-4.0,
            params={
                "command_name": "motion",
                "ref_speed_threshold": 0.1,
                "vel_tolerance": 0.05,
            },
        )
        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=2.0,
            params={"command_name": "motion", "std": 0.6},
        )

        # ── Body-level tracking ────────────────────────────────────────
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        )
        self.rewards.motion_body_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        )
        self.rewards.motion_body_lin_vel = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.5,
            params={"command_name": "motion", "std": 0.6},
        )
        self.rewards.motion_body_ang_vel = RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=0.0,
            params={"command_name": "motion", "std": 3.0},
        )
        self.rewards.motion_torso_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=1.5,
            params={
                "command_name": "motion",
                "std": 0.25,
                "body_names": [
                    "base_link",
                    "waist_yaw_link",
                    "waist_roll_link",
                    "waist_pitch_link",
                ],
            },
        )

        # ── Full-body joint rewards ────────────────────────────────────
        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.motion_joint_position_error_linear,
            weight=5.0,
            params={"command_name": "motion", "std": 0.35},
        )
        self.rewards.motion_joint_vel = RewTerm(
            func=mdp.motion_joint_velocity_error_exp,
            weight=2.5,
            params={"command_name": "motion", "std": 3.0},
        )

        # ── Regularization ──────────────────────────────────────────────
        self.rewards.upper_body_lin_vel_penalty = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_linear,
            weight=-0.25,
            params={
                "command_name": "motion",
                "body_names": [
                    "base_link",
                    "waist_yaw_link",
                    "waist_roll_link",
                    "waist_pitch_link",
                ],
            },
        )
        self.rewards.upper_body_ang_vel_penalty = RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_linear,
            weight=-0.3,
            params={
                "command_name": "motion",
                "body_names": [
                    "base_link",
                    "waist_yaw_link",
                    "waist_roll_link",
                    "waist_pitch_link",
                ],
            },
        )
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.75)

        # ── Terminations ───────────────────────────────────────────────────
        self.terminations.anchor_pos = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.5},
        )
        self.terminations.anchor_ori = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                "threshold": 0.8,
            },
        )
        self.terminations.anchor_ori_full = DoneTerm(
            func=mdp.bad_anchor_ori_full,
            params={"command_name": "motion", "threshold": 2.0},
        )
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.9,
                "body_names": _EE_LINKS,
            },
        )
