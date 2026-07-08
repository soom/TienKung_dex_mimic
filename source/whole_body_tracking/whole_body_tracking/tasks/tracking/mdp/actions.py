from __future__ import annotations

from dataclasses import MISSING
import re
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResidualJointPositionAction(JointAction):
    """Joint position action that applies residuals on top of reference motion joint positions.

    target_joint_pos = ref_joint_pos + raw_action * scale
    """

    cfg: ResidualJointPositionActionCfg

    def __init__(self, cfg: ResidualJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.command_name
        if isinstance(cfg.ema_alpha, (float, int)):
            self._ema_alpha = float(cfg.ema_alpha)
        elif isinstance(cfg.ema_alpha, dict):
            self._ema_alpha = torch.ones(self.num_envs, self.action_dim, device=self.device)
            for pattern, value in cfg.ema_alpha.items():
                joint_ids = [index for index, name in enumerate(self._joint_names) if re.fullmatch(pattern, name)]
                if joint_ids:
                    self._ema_alpha[:, joint_ids] = float(value)
        else:
            raise ValueError(f"Unsupported ema_alpha type: {type(cfg.ema_alpha)}. Supported types are float and dict.")
        if torch.is_tensor(self._ema_alpha):
            self._ema_alpha = torch.clamp(self._ema_alpha, 0.0, 1.0)
        else:
            self._ema_alpha = max(0.0, min(1.0, self._ema_alpha))
        self._filtered_processed_actions = torch.zeros_like(self._processed_actions)
        self._has_filtered_actions = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if (not torch.is_tensor(self._ema_alpha)) and self._ema_alpha >= 1.0:
            self._filtered_processed_actions[:] = self._processed_actions
            self._has_filtered_actions[:] = True
            return

        first = ~self._has_filtered_actions
        if first.any():
            self._filtered_processed_actions[first] = self._processed_actions[first]
            self._has_filtered_actions[first] = True

        alpha = self._ema_alpha
        self._filtered_processed_actions[:] = (
            alpha * self._processed_actions + (1.0 - alpha) * self._filtered_processed_actions
        )

    def  apply_actions(self):
        command: MotionCommand = self._env.command_manager.get_term(self._command_name)
        ref_joint_pos = command.joint_pos  # (num_envs, num_joints)
        target = ref_joint_pos + self._filtered_processed_actions
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        # self._asset.set_joint_velocity_target(command.joint_vel, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._filtered_processed_actions[:] = 0.0
            self._has_filtered_actions[:] = False
        else:
            self._filtered_processed_actions[env_ids] = 0.0
            self._has_filtered_actions[env_ids] = False



@configclass
class ResidualJointPositionActionCfg(JointActionCfg):
    """Configuration for residual joint position action.

    The policy outputs a residual delta on top of the reference motion joint positions.
    """

    class_type: type[ActionTerm] = ResidualJointPositionAction

    joint_names: list[str] = MISSING
    command_name: str = "motion"
    ema_alpha: float | dict[str, float] = 1.0
