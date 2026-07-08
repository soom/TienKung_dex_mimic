from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators.actuator_pd import ImplicitActuator
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer


class DelayedImplicitActuator(ImplicitActuator):
    """隐式执行器 + 指令延迟。

    继承 ImplicitActuator，Isaac Lab 仍将 stiffness/damping 写入 PhysX，
    velocity_limit_sim 完全有效。仅在 compute() 里对目标位置加 delay buffer。
    """

    cfg: "DelayedImplicitActuatorCfg"

    def __init__(self, cfg: "DelayedImplicitActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.positions_delay_buffer.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """DelayedImplicitActuator 的配置类。"""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """最小延迟步数（physics steps）。"""

    max_delay: int = 1
    """最大延迟步数（physics steps）。"""

