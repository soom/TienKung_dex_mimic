from __future__ import annotations

import glob as _glob
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_motion_files(motion_file: str) -> list[str]:
    """Resolve motion_file to a sorted list of NPZ paths.

    Accepts:
    - A single .npz file path
    - A directory path  → all *.npz files inside (sorted)
    - A comma-separated list of .npz paths
    """
    if os.path.isdir(motion_file):
        all_files = sorted(_glob.glob(os.path.join(motion_file, "*.npz")))
        filtered_files = [
            f
            for f in all_files
            if not os.path.basename(f).endswith("_ik.npz") and "_all" not in os.path.basename(f)
        ]
        if not filtered_files:
            raise FileNotFoundError(f"No .npz files found in directory: {motion_file}")

        preferred_files: dict[str, str] = {}
        for file_path in filtered_files:
            base_name = os.path.basename(file_path)
            stem = os.path.splitext(base_name)[0]
            canonical_stem = stem.removesuffix("_with_transition")
            current = preferred_files.get(canonical_stem)
            if current is None or stem.endswith("_with_transition"):
                preferred_files[canonical_stem] = file_path

        files = [preferred_files[key] for key in sorted(preferred_files)]
        transition_count = sum(os.path.basename(f).endswith("_with_transition.npz") for f in files)
        print(
            f"[MotionCommand] Directory mode: selected {len(files)} NPZ files in {motion_file} "
            f"({transition_count} with transition)."
        )
        return files
    if "," in motion_file:
        files = [f.strip() for f in motion_file.split(",") if f.strip()]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f"Motion files not found: {missing}")
        return files
    # Single file
    if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    return [motion_file]


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        target_joint_names: Sequence[str] | None = None,
        target_body_names: Sequence[str] | None = None,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = float(np.asarray(data["fps"]).item())

        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)

        if target_joint_names is not None and "joint_names" in data:
            stored_joint_names = [str(name) for name in data["joint_names"].tolist()]
            if stored_joint_names != list(target_joint_names):
                name_to_index = {name: idx for idx, name in enumerate(stored_joint_names)}
                missing = [name for name in target_joint_names if name not in name_to_index]
                if missing:
                    raise KeyError(f"Motion file missing joints: {missing}")
                order = [name_to_index[name] for name in target_joint_names]
                joint_pos = joint_pos[:, order]
                joint_vel = joint_vel[:, order]

        self.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)

        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

        # Remap body columns by name so NPZ data matches the target (robot) body order.
        # Without this, body_indexes (which are robot body indices from find_bodies)
        # would index into NPZ columns that use a different ordering (e.g. MuJoCo
        # depth-first vs IsaacLab breadth-first).
        if target_body_names is not None and "body_names" in data:
            stored_body_names = [str(name) for name in data["body_names"].tolist()]
            if stored_body_names != list(target_body_names):
                name_to_index = {name: idx for idx, name in enumerate(stored_body_names)}
                missing = [name for name in target_body_names if name not in name_to_index]
                if missing:
                    raise KeyError(f"Motion file missing bodies: {missing}")
                order = [name_to_index[name] for name in target_body_names]
                body_pos_w = body_pos_w[:, order]
                body_quat_w = body_quat_w[:, order]
                body_lin_vel_w = body_lin_vel_w[:, order]
                body_ang_vel_w = body_ang_vel_w[:, order]

        self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)
        self._body_indexes = body_indexes

        # Load pre-computed transition frames (offline FK, see scripts/add_transition_to_npz.py)
        self.stand_to_track: torch.Tensor | None = None
        self.track_to_stand: torch.Tensor | None = None
        self.stand_to_track_body_pos: torch.Tensor | None = None
        self.stand_to_track_body_quat: torch.Tensor | None = None
        self.track_to_stand_body_pos: torch.Tensor | None = None
        self.track_to_stand_body_quat: torch.Tensor | None = None
        for key, attr in [("stand_to_track_joints", "stand_to_track"),
                          ("track_to_stand_joints", "track_to_stand"),
                          ("stand_to_track_body_pos", "stand_to_track_body_pos"),
                          ("stand_to_track_body_quat", "stand_to_track_body_quat"),
                          ("track_to_stand_body_pos", "track_to_stand_body_pos"),
                          ("track_to_stand_body_quat", "track_to_stand_body_quat")]:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
                setattr(self, attr, torch.tensor(arr, dtype=torch.float32, device=device))

        self.time_step_total = self.joint_pos.shape[0]
        # clip_boundaries: list of (start, end) frame indices, one per source file/clip.
        # Set to the full range by default; overridden by from_files().
        self.clip_boundaries: list[tuple[int, int]] = [(0, self.time_step_total)]

    @classmethod
    def from_files(
        cls,
        motion_files: list[str],
        body_indexes: Sequence[int],
        device: str = "cpu",
        target_joint_names: Sequence[str] | None = None,
        target_body_names: Sequence[str] | None = None,
    ) -> "MotionLoader":
        """Load and concatenate multiple NPZ files into one MotionLoader.

        Each file becomes one or more clips in clip_boundaries (split by
        adaptive_max_clip_duration_s later in MotionCommand). Files are
        concatenated along the time axis; clip_boundaries records the
        per-file frame ranges so adaptive sampling can treat each file
        as a separate clip.
        """
        if len(motion_files) == 1:
            loader = cls(motion_files[0], body_indexes, device, target_joint_names, target_body_names)
            return loader

        loaders = [cls(f, body_indexes, device, target_joint_names, target_body_names) for f in motion_files]
        fps = loaders[0].fps
        for i, l in enumerate(loaders[1:], 1):
            if abs(l.fps - fps) > 0.5:
                raise ValueError(f"FPS mismatch: {motion_files[0]} fps={fps}, {motion_files[i]} fps={l.fps}")

        # Concatenate all tensors in original coordinates.
        merged = cls.__new__(cls)
        merged.fps = fps
        merged._body_indexes = body_indexes
        merged.joint_pos = torch.cat([l.joint_pos for l in loaders], dim=0)
        merged.joint_vel = torch.cat([l.joint_vel for l in loaders], dim=0)
        merged._body_pos_w = torch.cat([l._body_pos_w for l in loaders], dim=0)
        merged._body_quat_w = torch.cat([l._body_quat_w for l in loaders], dim=0)
        merged._body_lin_vel_w = torch.cat([l._body_lin_vel_w for l in loaders], dim=0)
        merged._body_ang_vel_w = torch.cat([l._body_ang_vel_w for l in loaders], dim=0)
        merged.time_step_total = merged.joint_pos.shape[0]

        # Record per-file boundaries
        boundaries = []
        offset = 0
        for l in loaders:
            boundaries.append((offset, offset + l.time_step_total))
            offset += l.time_step_total
        merged.clip_boundaries = boundaries
        return merged

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Resolve motion files: single file, comma-separated list, or directory
        motion_files = _resolve_motion_files(self.cfg.motion_file)
        if len(motion_files) == 1:
            self.motion = MotionLoader(
                motion_files[0],
                self.body_indexes,
                device=self.device,
                target_joint_names=self.robot.joint_names,
                target_body_names=self.robot.body_names,
            )
        else:
            self.motion = MotionLoader.from_files(
                motion_files,
                self.body_indexes,
                device=self.device,
                target_joint_names=self.robot.joint_names,
                target_body_names=self.robot.body_names,
            )
        print(f"[MotionCommand] Loaded {len(motion_files)} file(s), {self.motion.time_step_total} frames total.")

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_duration_steps = max(1, int(round(self.cfg.adaptive_max_clip_duration_s * self.motion.fps)))

        clip_starts, clip_ends = [], []
        if self.cfg.start_from_first_frame:
            # One clip per source file: every reset starts from each file's first frame.
            for file_start, file_end in self.motion.clip_boundaries:
                clip_starts.append(file_start)
                clip_ends.append(file_end)
        else:
            # Build clip boundaries: subdivide each file's frame range by clip_duration_steps
            for file_start, file_end in self.motion.clip_boundaries:
                for s in range(file_start, file_end, self.clip_duration_steps):
                    e = min(s + self.clip_duration_steps, file_end)
                    clip_starts.append(s)
                    clip_ends.append(e)
        self.clip_start_steps = torch.tensor(clip_starts, dtype=torch.long, device=self.device)
        self.clip_end_steps = torch.tensor(clip_ends, dtype=torch.long, device=self.device)
        self.clip_count = int(self.clip_start_steps.numel())
        self.current_clip_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_clip_end_step = torch.full(
            (self.num_envs,), self.motion.time_step_total, dtype=torch.long, device=self.device
        )
        self.clip_completion_level = torch.full(
            (self.clip_count,), self.cfg.adaptive_completion_init, dtype=torch.float32, device=self.device
        )
        self.clip_error_ema = torch.full(
            (self.clip_count,), self.cfg.adaptive_error_threshold, dtype=torch.float32, device=self.device
        )
        # Per-clip failure tracking / blacklist
        self.clip_failure_count = torch.zeros(self.clip_count, dtype=torch.long, device=self.device)
        self.clip_total_count = torch.zeros(self.clip_count, dtype=torch.long, device=self.device)
        self.clip_blacklisted = torch.zeros(self.clip_count, dtype=torch.bool, device=self.device)
        self._load_blacklist_log()  # restore blacklist flags from previous run (counters start fresh)
        self._blacklist_warmup_iters = getattr(self.cfg, "curriculum_blacklist_warmup_iters", 0)
        self._blacklist_iter_count = 0
        self._global_env_step = 0  # monotonic env-step counter for blacklist warmup gating
        self._episode_key_body_error_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_key_body_error_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.observation_pre_shift_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_pos_stand"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_pos_track"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["segment_transition_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["curriculum_clip_fraction"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["curriculum_reset_scale"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["curriculum_upper_body_action_scale"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["raw_action_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["blacklisted_clip_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["full_episode_length"] = torch.zeros(self.num_envs, device=self.device)

        self._curriculum_stage = max(0, min(int(self.cfg.curriculum_initial_stage), len(self.cfg.curriculum_clip_fraction) - 1))
        self._curriculum_hold_count = 0
        self._curriculum_stage_step_count = 0
        self._curriculum_metric_ema: dict[str, torch.Tensor] = {}
        self._active_clip_fraction = 1.0
        self._active_reset_scale = 1.0
        self._active_obs_pre_shift_prob = self.cfg.observation_pre_shift_prob
        self._active_obs_pre_shift_max_steps = self.cfg.observation_pre_shift_max_steps
        self._active_uniform_ratio = self.cfg.adaptive_uniform_ratio
        self._active_upper_body_action_scale = 1.0
        self._action_term = None
        self._action_base_scale = None
        self._upper_body_joint_ids = None

        self._clip_difficulty = self._build_clip_difficulty()
        self._clip_difficulty_order = torch.argsort(self._clip_difficulty)
        self._apply_curriculum_stage(force=True)
        self._sample_observation_pre_shift()

        # ── Phase state machine (dual-head) ──────────────────────────────────
        self._phase_enabled = bool(getattr(self.cfg, "phase_mode_enabled", False))
        if self._phase_enabled:
            control_fps = 1.0 / (self._env.step_dt * self._env.cfg.decimation)
            self._stand_steps = max(1, int(round(self.cfg.stand_duration_s * control_fps)))
            # 0=STAND, 1=TRACKING
            self.phase_mode        = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._phase_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._phase_clip_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._phase_clip_end   = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._stand_joint_pos  = self._build_stand_joint_pos()
            self._build_stand_body_fk()
            # True when TRACKING clip ends → triggers time_out termination
            self.phase_timeout     = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # True after an env has completed TRACKING at least once (distinguishes return STAND)
            self._has_tracked      = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # True at the step when STAND→TRACKING transition occurs
            self._phase_boundary   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._total_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            print(f"[MotionCommand] Phase mode enabled: STAND={self._stand_steps} steps → TRACKING=clip")

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def _build_stand_joint_pos(self) -> torch.Tensor:
        """Build stand reference joint pos tensor ordered by robot joint names."""
        cfg_map: dict[str, float] = getattr(self.cfg, "stand_joint_pos", {})
        if cfg_map:
            pos = torch.zeros(len(self.robot.joint_names), device=self.device)
            for i, name in enumerate(self.robot.joint_names):
                if name in cfg_map:
                    pos[i] = cfg_map[name]
            return pos
        return self.robot.data.default_joint_pos[0].clone()

    def _build_stand_body_fk(self):
        """Load pre-computed standing body FK from config."""
        body_pos = getattr(self.cfg, "stand_body_pos", [])
        body_quat = getattr(self.cfg, "stand_body_quat", [])
        if not body_pos or not body_quat:
            self._stand_body_pos = None
            self._stand_body_quat = None
            return
        self._stand_body_pos = torch.tensor(body_pos, dtype=torch.float32, device=self.device)
        self._stand_body_quat = torch.tensor(body_quat, dtype=torch.float32, device=self.device)
        print(f"[MotionCommand] Loaded stand body FK: {len(body_pos)} bodies.")

    @property
    def _effective_time_steps(self) -> torch.Tensor:
        """Time steps for motion indexing, frozen during STAND phase."""
        if not self._phase_enabled:
            return self.time_steps
        t = self.time_steps.clone()
        # Initial STAND (before first tracking): freeze at clip start
        init_stand = (self.phase_mode == 0) & ~self._has_tracked
        t[init_stand] = self._phase_clip_start[init_stand]
        # Return STAND (after tracking): freeze at current time_steps (last tracking frame)
        ret_stand = (self.phase_mode == 0) & self._has_tracked
        # — time_steps already at last frame, no change needed
        # SAFETY: clamp to valid range to prevent CUDA index-out-of-bounds
        t = t.clamp(min=0, max=self.motion.time_step_total - 1)
        return t

    @property
    def joint_pos(self) -> torch.Tensor:
        if not self._phase_enabled:
            return self.motion.joint_pos[self.time_steps]
        t = self._effective_time_steps
        result = self.motion.joint_pos[t]
        # STAND: pure standing reference — no blend/transition
        stand_mask = self.phase_mode == 0
        if stand_mask.any():
            result[stand_mask] = self._stand_joint_pos.unsqueeze(0).expand(int(stand_mask.sum()), -1)
        return result

    def joint_pos_lookahead(self, k: int) -> torch.Tensor:
        """Reference joint positions k steps ahead, clamped to motion end."""
        if not self._phase_enabled:
            future_steps = (self.time_steps + k).clamp(max=self.motion.time_step_total - 1)
            return self.motion.joint_pos[future_steps]
        # During non-TRACKING phases, lookahead is frozen at effective time step
        base = self._effective_time_steps
        tracking = self.phase_mode == 1
        future_steps = base.clone()
        future_steps[tracking] = (self.time_steps[tracking] + k).clamp(max=self.motion.time_step_total - 1)
        return self.motion.joint_pos[future_steps]

    def joint_vel_lookahead(self, k: int) -> torch.Tensor:
        """Reference joint velocities k steps ahead, clamped to motion end."""
        if not self._phase_enabled:
            future_steps = (self.time_steps + k).clamp(max=self.motion.time_step_total - 1)
            return self.motion.joint_vel[future_steps]
        base = self._effective_time_steps
        tracking = self.phase_mode == 1
        future_steps = base.clone()
        future_steps[tracking] = (self.time_steps[tracking] + k).clamp(max=self.motion.time_step_total - 1)
        return self.motion.joint_vel[future_steps]

    def _future_time_steps(self, k: int, use_obs_pre_shift: bool = False) -> torch.Tensor:
        future_steps = self.time_steps + k
        if use_obs_pre_shift:
            future_steps = future_steps + self.observation_pre_shift_steps
        return torch.minimum(future_steps, self.current_clip_end_step - 1).clamp(max=self.motion.time_step_total - 1)

    @property
    def joint_vel(self) -> torch.Tensor:
        vel = self.motion.joint_vel[self._effective_time_steps]
        if self._phase_enabled:
            vel[self.phase_mode == 0] = 0.0  # STAND: zero velocity
        return vel

    def foot_contact_label(self, foot_names: Sequence[str] | str, height_threshold: float = 0.05) -> torch.Tensor:
        if isinstance(foot_names, str):
            foot_names = [foot_names]
        foot_indexes = [self.cfg.body_names.index(name) for name in foot_names]
        return self.motion.body_pos_w[self._effective_time_steps][:, foot_indexes, 2] <= height_threshold

    @property
    def body_pos_w(self) -> torch.Tensor:
        base = self.motion.body_pos_w[self._effective_time_steps]
        if self._phase_enabled and self._stand_body_pos is not None:
            stand_mask = self.phase_mode == 0
            if stand_mask.any():
                stand_body = self._stand_body_pos.unsqueeze(0).expand(self.num_envs, -1, -1)
                base[stand_mask] = stand_body[stand_mask]
        return base + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        base = self.motion.body_quat_w[self._effective_time_steps]
        if self._phase_enabled and self._stand_body_quat is not None:
            stand_mask = self.phase_mode == 0
            if stand_mask.any():
                stand_quat = self._stand_body_quat.unsqueeze(0).expand(self.num_envs, -1, -1)
                base[stand_mask] = stand_quat[stand_mask]
        return base

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        vel = self.motion.body_lin_vel_w[self._effective_time_steps]
        if self._phase_enabled:
            vel[self.phase_mode == 0] = 0.0
        return vel

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        vel = self.motion.body_ang_vel_w[self._effective_time_steps]
        if self._phase_enabled:
            vel[self.phase_mode == 0] = 0.0
        return vel

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        if self._phase_enabled and self._stand_body_pos is not None:
            stand_mask = self.phase_mode == 0
            if stand_mask.any():
                stand_anchor = self._stand_body_pos[0].unsqueeze(0).expand(self.num_envs, -1)
                return stand_anchor + self._env.scene.env_origins
        return self.motion.body_pos_w[self._effective_time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        if self._phase_enabled and self._stand_body_quat is not None:
            stand_mask = self.phase_mode == 0
            if stand_mask.any():
                stand_anchor = self._stand_body_quat[0].unsqueeze(0).expand(self.num_envs, -1)
                return stand_anchor
        return self.motion.body_quat_w[self._effective_time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._effective_time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._effective_time_steps, self.motion_anchor_body_index]

    def anchor_pos_lookahead(self, k: int, use_obs_pre_shift: bool = False) -> torch.Tensor:
        future_steps = self._future_time_steps(k, use_obs_pre_shift=use_obs_pre_shift)
        return self.motion.body_pos_w[future_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    def anchor_quat_lookahead(self, k: int, use_obs_pre_shift: bool = False) -> torch.Tensor:
        future_steps = self._future_time_steps(k, use_obs_pre_shift=use_obs_pre_shift)
        return self.motion.body_quat_w[future_steps, self.motion_anchor_body_index]

    def anchor_lin_vel_lookahead(self, k: int, use_obs_pre_shift: bool = False) -> torch.Tensor:
        future_steps = self._future_time_steps(k, use_obs_pre_shift=use_obs_pre_shift)
        return self.motion.body_lin_vel_w[future_steps, self.motion_anchor_body_index]

    def anchor_ang_vel_lookahead(self, k: int, use_obs_pre_shift: bool = False) -> torch.Tensor:
        future_steps = self._future_time_steps(k, use_obs_pre_shift=use_obs_pre_shift)
        return self.motion.body_ang_vel_w[future_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        err = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_pos"] = err
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        # Per-phase breakdown: zero out envs not in current phase
        if self._phase_enabled:
            stand = self.phase_mode == 0
            track = self.phase_mode == 1
            self.metrics["error_anchor_pos_stand"] = torch.where(stand, err, torch.zeros_like(err))
            self.metrics["error_anchor_pos_track"] = torch.where(track, err, torch.zeros_like(err))
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        self.metrics["raw_action_rate"] = torch.sum(
            torch.square(self._env.action_manager.action - self._env.action_manager.prev_action), dim=-1
        )
        self.metrics["curriculum_stage"] = torch.full(
            (self.num_envs,), float(self._curriculum_stage), dtype=torch.float32, device=self.device
        )
        self.metrics["curriculum_clip_fraction"] = torch.full(
            (self.num_envs,), self._active_clip_fraction, dtype=torch.float32, device=self.device
        )
        self.metrics["curriculum_reset_scale"] = torch.full(
            (self.num_envs,), self._active_reset_scale, dtype=torch.float32, device=self.device
        )
        self.metrics["curriculum_upper_body_action_scale"] = torch.full(
            (self.num_envs,), self._active_upper_body_action_scale, dtype=torch.float32, device=self.device
        )

        key_error = self.metrics[self.cfg.adaptive_key_error_metric]
        self._episode_key_body_error_sum += key_error
        self._episode_key_body_error_count += 1.0
        self._update_curriculum_state()

    def _clip_sampling_probabilities(self) -> torch.Tensor:
        threshold = max(self.cfg.adaptive_error_threshold, 1e-6)
        normalized_error = torch.clamp(self.clip_error_ema / threshold, max=1.0).pow(5)
        clip_scores = torch.where(self.clip_completion_level > 1.0, self.clip_completion_level, normalized_error)
        clip_scores = clip_scores + self._active_uniform_ratio / float(self.clip_count)
        allowed_mask = self._allowed_clip_mask()
        clip_scores = torch.where(allowed_mask, clip_scores, torch.zeros_like(clip_scores))
        hard_focus_prob = self._curriculum_value(self.cfg.curriculum_hard_focus_prob)
        if hard_focus_prob > 0.0:
            hard_mask = self._hard_clip_mask()
            if torch.any(hard_mask & allowed_mask):
                hard_scores = torch.where(hard_mask & allowed_mask, clip_scores, torch.zeros_like(clip_scores))
                hard_sum = hard_scores.sum()
                if hard_sum > 0:
                    mixed_scores = clip_scores * (1.0 - hard_focus_prob)
                    mixed_scores = mixed_scores + hard_scores / hard_sum * clip_scores.sum() * hard_focus_prob
                    clip_scores = mixed_scores
        if clip_scores.sum() <= 0:
            clip_scores = torch.ones_like(clip_scores)
        return clip_scores / clip_scores.sum()

    def _update_clip_statistics(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        has_history = self._episode_key_body_error_count[env_ids] > 0
        if not torch.any(has_history):
            return

        completed_env_ids = env_ids[has_history]
        clip_ids = self.current_clip_index[completed_env_ids]
        episode_failed = self._env.termination_manager.terminated[completed_env_ids]
        episode_error = self._episode_key_body_error_sum[completed_env_ids] / self._episode_key_body_error_count[
            completed_env_ids
        ].clamp_min(1.0)

        for clip_id in torch.unique(clip_ids):
            clip_mask = clip_ids == clip_id
            n_total = int(clip_mask.sum().item())
            n_failed = int(episode_failed[clip_mask].sum().item())

            # Accumulate per-clip failure statistics
            self.clip_total_count[clip_id] += n_total
            self.clip_failure_count[clip_id] += n_failed

            clip_error_mean = episode_error[clip_mask].mean()
            self.clip_error_ema[clip_id] = (
                self.cfg.adaptive_alpha * clip_error_mean + (1.0 - self.cfg.adaptive_alpha) * self.clip_error_ema[clip_id]
            )

            success_count = n_total - n_failed
            if success_count > 0:
                decay_factor = self.cfg.adaptive_completion_decay ** success_count
                self.clip_completion_level[clip_id] = torch.maximum(
                    self.clip_completion_level.new_tensor(1.0),
                    self.clip_completion_level[clip_id] * decay_factor,
                )

        self._update_blacklist()

    def _update_blacklist(self, log_dir: str | None = None) -> None:
        """Blacklist clips whose failure rate exceeds the configured threshold."""
        threshold = getattr(self.cfg, "curriculum_blacklist_failure_rate", 0.0)
        min_samples = getattr(self.cfg, "curriculum_blacklist_min_samples", 50)
        if threshold <= 0.0:
            return

        # Warmup: defer blacklist evaluation for N PPO *iterations* after init/resume.
        # _update_blacklist is called once per episode termination (~400×/PPO iter),
        # so we gate on monotonic env-step count instead of call count.
        steps_per_iter = int(getattr(self.cfg, "curriculum_steps_per_iter", 32))
        warmup_steps = int(self._blacklist_warmup_iters) * steps_per_iter
        if self._global_env_step < warmup_steps:
            return
        if self._blacklist_iter_count == 0:  # first time past warmup
            print(f"[MotionCommands] Blacklist warmup complete ({self._global_env_step}/{warmup_steps} steps).")
            self.clip_total_count.zero_()
            self.clip_failure_count.zero_()
            self.clip_blacklisted.zero_()
            self.metrics["blacklisted_clip_count"][:] = 0.0
            self._blacklist_iter_count = -1  # mark as done

        total = self.clip_total_count.float()
        failed = self.clip_failure_count.float()
        failure_rate = failed / total.clamp_min(1)
        newly_blacklisted = (total >= min_samples) & (failure_rate >= threshold) & ~self.clip_blacklisted
        self.clip_blacklisted |= newly_blacklisted

        if torch.any(newly_blacklisted):
            new_count = int(newly_blacklisted.sum().item())
            total_blacklisted = int(self.clip_blacklisted.sum().item())
            total_clips = int(self.clip_count)
            print(
                f"[MotionCommands] Blacklisted {new_count} new clip(s) "
                f"({total_blacklisted}/{total_clips} total blacklisted)."
            )

        # Emit metric
        self.metrics["blacklisted_clip_count"][:] = float(self.clip_blacklisted.sum().item())

        # Write per-clip failure stats to log file
        self._write_blacklist_log(log_dir)

    def _load_blacklist_log(self) -> None:
        """Restore blacklist state from a previous run's CSV log file."""
        import os as _os

        log_dir = getattr(self.cfg, "blacklist_log_dir", None)
        if log_dir is None:
            return

        log_path = _os.path.join(log_dir, "clip_blacklist.csv")
        if not _os.path.isfile(log_path):
            return

        try:
            import csv
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = int(row["clip_id"])
                    if cid < 0 or cid >= self.clip_count:
                        continue
                    self.clip_total_count[cid] = 0  # counters start fresh each run
                    self.clip_failure_count[cid] = 0
                    # Do NOT restore blacklist flags — let each run build its own
                    # based on current thresholds and performance.
            total_blacklisted = int(self.clip_blacklisted.sum().item())
            if total_blacklisted > 0:
                print(
                    f"[MotionCommands] Loaded blacklist from {log_path}: "
                    f"{total_blacklisted}/{self.clip_count} clips blacklisted."
                )
        except Exception:
            pass  # best-effort, don't crash on corrupt file

    def _write_blacklist_log(self, log_dir: str | None = None) -> None:
        """Dump per-clip failure statistics and blacklist to a CSV file."""
        import os as _os

        if log_dir is None:
            log_dir = getattr(self.cfg, "blacklist_log_dir", None)
        if log_dir is None:
            # fallback: write alongside the training log dir
            log_dir = getattr(self.cfg, "log_dir", None)
        if log_dir is None:
            return  # can't determine where to write

        log_path = _os.path.join(log_dir, "clip_blacklist.csv")
        _os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("clip_id,clip_start,clip_end,total_episodes,failed,failure_rate,blacklisted\n")
            for clip_id in range(self.clip_count):
                start = int(self.clip_start_steps[clip_id].item())
                end = int(self.clip_end_steps[clip_id].item())
                n_total = int(self.clip_total_count[clip_id].item())
                n_failed = int(self.clip_failure_count[clip_id].item())
                rate = n_failed / max(n_total, 1)
                bl = int(self.clip_blacklisted[clip_id].item())
                f.write(f"{clip_id},{start},{end},{n_total},{n_failed},{rate:.4f},{bl}\n")

    def _sample_observation_pre_shift(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        self.observation_pre_shift_steps[env_ids] = 0
        max_steps = int(self._active_obs_pre_shift_max_steps)
        if self._active_obs_pre_shift_prob <= 0.0 or max_steps <= 0:
            return

        shifted_mask = torch.rand(env_ids.numel(), device=self.device) < self._active_obs_pre_shift_prob
        if not torch.any(shifted_mask):
            return

        shifted_env_ids = env_ids[shifted_mask]
        self.observation_pre_shift_steps[shifted_env_ids] = torch.randint(
            1, max_steps + 1, (shifted_env_ids.numel(),), device=self.device
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._update_clip_statistics(env_ids)

        sampling_probabilities = self._clip_sampling_probabilities()
        sampled_clips = torch.multinomial(sampling_probabilities, env_ids.numel(), replacement=True)
        clip_starts = self.clip_start_steps[sampled_clips]
        clip_ends = self.clip_end_steps[sampled_clips]
        if self.cfg.start_from_first_frame:
            sampled_steps = clip_starts
        else:
            clip_lengths = torch.clamp(clip_ends - clip_starts, min=1)
            clip_offsets = (
                sample_uniform(0.0, 1.0, (env_ids.numel(),), device=self.device) * clip_lengths.float()
            ).long()
            sampled_steps = torch.minimum(clip_starts + clip_offsets, clip_ends - 1)

        self.time_steps[env_ids] = sampled_steps
        self.current_clip_index[env_ids] = sampled_clips
        self.current_clip_end_step[env_ids] = clip_ends

        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / max(self.clip_count, 1)

        self._episode_key_body_error_sum[env_ids] = 0.0
        self._episode_key_body_error_count[env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        # Phase mode: record clip boundaries and reset to STAND
        if self._phase_enabled:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self._phase_clip_start[env_ids_t] = self.time_steps[env_ids_t]
            self._phase_clip_end[env_ids_t]   = self.current_clip_end_step[env_ids_t]
            self.phase_mode[env_ids_t]         = 0  # → STAND
            self._phase_step_count[env_ids_t]  = 0
            self.phase_timeout[env_ids_t]      = False
            self._has_tracked[env_ids_t]       = False
            self._total_step_count[env_ids_t]   = 0
            # Freeze time_steps at clip start during STAND; clip_end stays as-is
            self.time_steps[env_ids_t] = self._phase_clip_start[env_ids_t]

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device) * self._active_reset_scale
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device) * self._active_reset_scale
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_low, joint_high = self.cfg.joint_position_range
        joint_pos += sample_uniform(
            joint_low * self._active_reset_scale,
            joint_high * self._active_reset_scale,
            joint_pos.shape,
            joint_pos.device,
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _reset_transition_buffers(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0 or not self.cfg.reset_state_buffers_on_clip_transition:
            return

        self._env.observation_manager.reset(env_ids)
        self._env.action_manager.reset(env_ids)

        if hasattr(self._env, "_jerk_prev_prev_action"):
            self._env._jerk_prev_prev_action[env_ids] = 0.0

        for sensor in self._env.scene.sensors.values():
            sensor.reset(env_ids)

        for actuator in self.robot.actuators.values():
            for buffer_name in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
                delay_buffer = getattr(actuator, buffer_name, None)
                if delay_buffer is not None:
                    delay_buffer.reset(env_ids)

        self.metrics["segment_transition_count"][env_ids] += 1.0
        self._env.sim.forward()

    def _build_clip_difficulty(self) -> torch.Tensor:
        difficulty = torch.zeros(self.clip_count, dtype=torch.float32, device=self.device)
        if self.clip_count == 0:
            return difficulty

        body_indexes: list[int] | None = None
        if self.cfg.curriculum_difficulty_body_names:
            body_indexes = [self.cfg.body_names.index(name) for name in self.cfg.curriculum_difficulty_body_names]

        for clip_id, (start_step, end_step) in enumerate(zip(self.clip_start_steps.tolist(), self.clip_end_steps.tolist())):
            joint_vel = self.motion.joint_vel[start_step:end_step]
            body_ang_vel = self.motion.body_ang_vel_w[start_step:end_step]
            if body_indexes is not None:
                body_ang_vel = body_ang_vel[:, body_indexes]
            joint_vel_score = torch.norm(joint_vel, dim=-1).mean()
            body_ang_vel_score = torch.norm(body_ang_vel, dim=-1).mean()
            difficulty[clip_id] = 0.6 * body_ang_vel_score + 0.4 * joint_vel_score
        return difficulty

    def _curriculum_value(self, values: tuple, stage: int | None = None):
        current_stage = self._curriculum_stage if stage is None else stage
        index = min(current_stage, len(values) - 1)
        return values[index]

    @property
    def current_drift_threshold(self) -> float | None:
        """Current drift termination threshold based on curriculum stage. None if not configured."""
        if not self.cfg.curriculum_drift_termination_threshold:
            return None
        return float(self._curriculum_value(self.cfg.curriculum_drift_termination_threshold))

    def _allowed_clip_mask(self) -> torch.Tensor:
        fraction = float(self._active_clip_fraction)
        # Compute keep_count from non-blacklisted pool so blacklisted clips
        # don't steal difficulty-rank slots from usable clips.
        usable_count = int((~self.clip_blacklisted).sum().item())
        keep_count = max(1, int(round(usable_count * fraction)))
        # Filter difficulty order to non-blacklisted only
        usable_order = self._clip_difficulty_order[~self.clip_blacklisted[self._clip_difficulty_order]]
        allowed = torch.zeros(self.clip_count, dtype=torch.bool, device=self.device)
        allowed[usable_order[:keep_count]] = True
        if not torch.any(allowed):
            allowed[:] = True  # fallback: allow all
        return allowed

    def _hard_clip_mask(self) -> torch.Tensor:
        hard_fraction = float(self.cfg.curriculum_hard_clip_fraction)
        usable_count = int((~self.clip_blacklisted).sum().item())
        hard_count = max(1, int(round(usable_count * hard_fraction)))
        usable_order = self._clip_difficulty_order[~self.clip_blacklisted[self._clip_difficulty_order]]
        hard = torch.zeros(self.clip_count, dtype=torch.bool, device=self.device)
        hard[usable_order[-hard_count:]] = True
        return hard

    def _try_initialize_action_curriculum(self):
        if self._action_term is not None:
            return
        action_manager = getattr(self._env, "action_manager", None)
        if action_manager is None:
            return
        action_term = action_manager.get_term("joint_pos")
        scale = getattr(action_term, "_scale", None)
        if scale is None or not isinstance(scale, torch.Tensor):
            return
        if not self.cfg.curriculum_upper_body_joint_names:
            return
        self._action_term = action_term
        self._action_base_scale = scale.clone()
        joint_ids = self.robot.find_joints(self.cfg.curriculum_upper_body_joint_names, preserve_order=True)[0]
        self._upper_body_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)

    def _apply_curriculum_stage(self, force: bool = False):
        if not self.cfg.curriculum_enabled:
            return

        self._active_clip_fraction = float(self._curriculum_value(self.cfg.curriculum_clip_fraction))
        self._active_reset_scale = float(self._curriculum_value(self.cfg.curriculum_reset_scale))
        self._active_obs_pre_shift_prob = float(self._curriculum_value(self.cfg.curriculum_observation_pre_shift_prob))
        self._active_obs_pre_shift_max_steps = int(
            self._curriculum_value(self.cfg.curriculum_observation_pre_shift_max_steps)
        )
        self._active_uniform_ratio = float(self._curriculum_value(self.cfg.curriculum_uniform_ratio))
        self._active_upper_body_action_scale = float(self._curriculum_value(self.cfg.curriculum_upper_body_action_scale))

        self._try_initialize_action_curriculum()
        if self._action_term is not None and self._action_base_scale is not None and self._upper_body_joint_ids is not None:
            self._action_term._scale.copy_(self._action_base_scale)
            self._action_term._scale[:, self._upper_body_joint_ids] *= self._active_upper_body_action_scale

        self.metrics["curriculum_stage"][:] = float(self._curriculum_stage)
        self.metrics["curriculum_clip_fraction"][:] = self._active_clip_fraction
        self.metrics["curriculum_reset_scale"][:] = self._active_reset_scale
        self.metrics["curriculum_upper_body_action_scale"][:] = self._active_upper_body_action_scale
        if force:
            print(
                "[MotionCommand] Curriculum stage "
                f"{self._curriculum_stage}: clip_fraction={self._active_clip_fraction:.2f}, "
                f"reset_scale={self._active_reset_scale:.2f}, "
                f"pre_shift_prob={self._active_obs_pre_shift_prob:.2f}, "
                f"upper_body_action_scale={self._active_upper_body_action_scale:.2f}"
            )

    def _update_curriculum_state(self):
        if not self.cfg.curriculum_enabled:
            return

        alpha = float(self.cfg.curriculum_metric_ema_alpha)
        tracked_means = {
            "error_anchor_pos": self.metrics["error_anchor_pos"].mean(),
            "error_anchor_lin_vel": self.metrics["error_anchor_lin_vel"].mean(),
            "error_body_pos": self.metrics["error_body_pos"].mean(),
            "error_body_ang_vel": self.metrics["error_body_ang_vel"].mean(),
            "error_joint_vel": self.metrics["error_joint_vel"].mean(),
            "raw_action_rate": self.metrics["raw_action_rate"].mean(),
        }
        for name, value in tracked_means.items():
            if name not in self._curriculum_metric_ema:
                self._curriculum_metric_ema[name] = value.detach().clone()
            else:
                self._curriculum_metric_ema[name] = (
                    alpha * value.detach() + (1.0 - alpha) * self._curriculum_metric_ema[name]
                )

        # Count in control steps internally; compare against iteration-based thresholds
        # by converting: threshold_steps = threshold_iters * steps_per_iter
        self._curriculum_stage_step_count += 1
        steps_per_iter = max(1, int(self.cfg.curriculum_steps_per_iter))

        if self._curriculum_stage > 0 and self.cfg.curriculum_demotion_hold_steps:
            demotion_iters = int(self._curriculum_value(self.cfg.curriculum_demotion_hold_steps, self._curriculum_stage - 1))
            anchor_pos_bad = False
            if self.cfg.curriculum_demote_error_anchor_pos:
                anchor_pos_bad = (
                    self._curriculum_metric_ema["error_anchor_pos"]
                    > self.cfg.curriculum_demote_error_anchor_pos[self._curriculum_stage - 1]
                )
            anchor_lin_vel_bad = False
            if self.cfg.curriculum_demote_error_anchor_lin_vel:
                anchor_lin_vel_bad = (
                    self._curriculum_metric_ema["error_anchor_lin_vel"]
                    > self.cfg.curriculum_demote_error_anchor_lin_vel[self._curriculum_stage - 1]
                )
            if (
                self._curriculum_stage_step_count >= demotion_iters * steps_per_iter
                and (anchor_pos_bad or anchor_lin_vel_bad)
            ):
                self._curriculum_stage -= 1
                self._curriculum_hold_count = 0
                self._curriculum_stage_step_count = 0
                self._apply_curriculum_stage(force=True)
                return

        next_stage = self._curriculum_stage + 1
        if next_stage >= len(self.cfg.curriculum_clip_fraction):
            return

        min_iters = int(self._curriculum_value(self.cfg.curriculum_min_stage_steps, self._curriculum_stage))
        if self._curriculum_stage_step_count < min_iters * steps_per_iter:
            return

        promote = (
            (not self.cfg.curriculum_promote_error_anchor_pos or self._curriculum_metric_ema["error_anchor_pos"]
             <= self.cfg.curriculum_promote_error_anchor_pos[self._curriculum_stage])
            and (not self.cfg.curriculum_promote_error_anchor_lin_vel or self._curriculum_metric_ema["error_anchor_lin_vel"]
                 <= self.cfg.curriculum_promote_error_anchor_lin_vel[self._curriculum_stage])
            and
            self._curriculum_metric_ema["error_body_pos"] <= self.cfg.curriculum_promote_error_body_pos[self._curriculum_stage]
            and self._curriculum_metric_ema["error_body_ang_vel"]
            <= self.cfg.curriculum_promote_error_body_ang_vel[self._curriculum_stage]
            and self._curriculum_metric_ema["error_joint_vel"]
            <= self.cfg.curriculum_promote_error_joint_vel[self._curriculum_stage]
            and self._curriculum_metric_ema["raw_action_rate"]
            <= self.cfg.curriculum_promote_raw_action_rate[self._curriculum_stage]
        )

        fallback_iters = int(self._curriculum_value(self.cfg.curriculum_fallback_min_stage_steps, self._curriculum_stage))
        fallback_ang_vel_ok = True
        if self.cfg.curriculum_fallback_error_body_ang_vel:
            fallback_ang_vel_ok = (
                self._curriculum_metric_ema["error_body_ang_vel"]
                <= self.cfg.curriculum_fallback_error_body_ang_vel[self._curriculum_stage]
            )
        fallback_anchor_pos_ok = True
        if self.cfg.curriculum_fallback_error_anchor_pos:
            fallback_anchor_pos_ok = (
                self._curriculum_metric_ema["error_anchor_pos"]
                <= self.cfg.curriculum_fallback_error_anchor_pos[self._curriculum_stage]
            )
        fallback_anchor_lin_vel_ok = True
        if self.cfg.curriculum_fallback_error_anchor_lin_vel:
            fallback_anchor_lin_vel_ok = (
                self._curriculum_metric_ema["error_anchor_lin_vel"]
                <= self.cfg.curriculum_fallback_error_anchor_lin_vel[self._curriculum_stage]
            )
        fallback_promote = (
            self._curriculum_stage_step_count >= fallback_iters * steps_per_iter
            and fallback_anchor_pos_ok
            and fallback_anchor_lin_vel_ok
            and self._curriculum_metric_ema["error_body_pos"]
            <= self.cfg.curriculum_fallback_error_body_pos[self._curriculum_stage]
            and self._curriculum_metric_ema["raw_action_rate"]
            <= self.cfg.curriculum_fallback_raw_action_rate[self._curriculum_stage]
            and fallback_ang_vel_ok
        )

        if promote:
            self._curriculum_hold_count += 1
        else:
            self._curriculum_hold_count = 0

        if fallback_promote:
            self._curriculum_stage = next_stage
            self._curriculum_hold_count = 0
            self._curriculum_stage_step_count = 0
            self._apply_curriculum_stage(force=True)
            return

        required_hold = int(self._curriculum_value(self.cfg.curriculum_promotion_hold_steps, self._curriculum_stage))
        if self._curriculum_hold_count < required_hold * steps_per_iter:
            return

        self._curriculum_stage = next_stage
        self._curriculum_hold_count = 0
        self._curriculum_stage_step_count = 0
        self._apply_curriculum_stage(force=True)

    def _update_phase_state(self):
        """Advance per-env phase state machine by one step."""
        self._phase_step_count += 1
        self._total_step_count += 1

        # STAND → TRACKING (first stand at episode start)
        # STAND after TRACKING → phase_timeout (return stand complete)
        stand_done = (self.phase_mode == 0) & (self._phase_step_count >= self._stand_steps)
        if stand_done.any():
            # Use _has_tracked flag to distinguish first vs return STAND
            already_tracked = stand_done & self._has_tracked
            first_stand = stand_done & ~self._has_tracked
            if already_tracked.any():
                # Loop: resample clip → TRACKING, repeat until standard time_out
                self._resample_command(already_tracked.nonzero(as_tuple=True)[0])
            if first_stand.any():
                env_ids = first_stand.nonzero(as_tuple=True)[0]
                self.phase_mode[first_stand] = 1
                self._phase_step_count[first_stand] = 0
                self.time_steps[first_stand] = self._phase_clip_start[first_stand]
                self.current_clip_end_step[first_stand] = self._phase_clip_end[first_stand]
                self._has_tracked[first_stand] = True  # mark as entered TRACKING
                # Reset robot joint state to tracking reference first frame.
                # Without this, the robot enters TRACKING from the standing pose,
                # causing inflated initial tracking error (e_anc_pos drifts up
                # as standing head stiffens).
                self.robot.write_joint_state_to_sim(
                    self.joint_pos[env_ids],   # tracking reference (phase→1 now)
                    self.joint_vel[env_ids],   # tracking velocity
                    env_ids=env_ids,
                )

    def _update_command(self):
        self._global_env_step += 1  # monotonic counter, used for blacklist warmup

        if self.cfg.curriculum_enabled and self._action_term is None:
            self._apply_curriculum_stage(force=False)

        if self._phase_enabled:
            self._update_phase_state()
            # Only advance time for TRACKING envs; STAND is frozen
            tracking = self.phase_mode == 1
            self.time_steps[tracking] += 1
        else:
            self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.current_clip_end_step)[0]

        if self._phase_enabled and env_ids.numel() > 0:
            # TRACKING clip ended → return to STAND; STAND will set phase_timeout when done
            tracking_done = env_ids[self.phase_mode[env_ids] == 1]
            if tracking_done.numel() > 0:
                self.phase_mode[tracking_done] = 0
                self._phase_step_count[tracking_done] = 0
                # return STAND — no GAE cut needed (Pro approach)
                # Extend clip_end to cover the return STAND duration
                self.current_clip_end_step[tracking_done] = (
                    self.time_steps[tracking_done] + self._stand_steps + 2
                )
            # Only resample envs that are not in a phase-managed state
            env_ids = env_ids[self.phase_mode[env_ids] != 0]

        self._resample_command(env_ids)
        self._reset_transition_buffers(env_ids)
        self._sample_observation_pre_shift()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_max_clip_duration_s: float = 10.0
    adaptive_completion_init: float = 10.0
    adaptive_completion_decay: float = 0.99
    adaptive_error_threshold: float = 0.15
    adaptive_key_error_metric: str = "error_anchor_pos"
    start_from_first_frame: bool = False
    reset_state_buffers_on_clip_transition: bool = False
    observation_pre_shift_prob: float = 0.0
    observation_pre_shift_max_steps: int = 0
    curriculum_enabled: bool = False
    curriculum_initial_stage: int = 0
    curriculum_clip_fraction: tuple[float, ...] = (1.0,)
    curriculum_hard_focus_prob: tuple[float, ...] = (0.0,)
    curriculum_hard_clip_fraction: float = 0.2
    curriculum_reset_scale: tuple[float, ...] = (1.0,)
    curriculum_uniform_ratio: tuple[float, ...] = (0.1,)
    curriculum_observation_pre_shift_prob: tuple[float, ...] = (0.0,)
    curriculum_observation_pre_shift_max_steps: tuple[int, ...] = (0,)
    curriculum_upper_body_action_scale: tuple[float, ...] = (1.0,)
    curriculum_upper_body_joint_names: list[str] = []
    curriculum_difficulty_body_names: list[str] = []
    curriculum_metric_ema_alpha: float = 0.01
    curriculum_min_stage_steps: tuple[int, ...] = (0,)
    curriculum_promotion_hold_steps: tuple[int, ...] = (1,)
    curriculum_promote_error_body_pos: tuple[float, ...] = ()
    curriculum_promote_error_body_ang_vel: tuple[float, ...] = ()
    curriculum_promote_error_joint_vel: tuple[float, ...] = ()
    curriculum_promote_raw_action_rate: tuple[float, ...] = ()
    curriculum_promote_error_anchor_pos: tuple[float, ...] = ()
    curriculum_promote_error_anchor_lin_vel: tuple[float, ...] = ()
    curriculum_fallback_min_stage_steps: tuple[int, ...] = ()
    curriculum_fallback_error_body_pos: tuple[float, ...] = ()
    curriculum_fallback_raw_action_rate: tuple[float, ...] = ()
    curriculum_fallback_error_body_ang_vel: tuple[float, ...] = ()
    curriculum_fallback_error_anchor_pos: tuple[float, ...] = ()
    curriculum_fallback_error_anchor_lin_vel: tuple[float, ...] = ()
    curriculum_demotion_hold_steps: tuple[int, ...] = ()
    curriculum_demote_error_anchor_pos: tuple[float, ...] = ()
    curriculum_demote_error_anchor_lin_vel: tuple[float, ...] = ()
    # Blacklist: clips whose failure rate (terminated episodes / total episodes)
    # exceeds this threshold are excluded from sampling.
    # Set to 0.0 to disable.  Recommended: 0.5 (50% failure rate).
    curriculum_blacklist_failure_rate: float = 0.0
    # Minimum episode samples before a clip is eligible for blacklisting.
    curriculum_blacklist_min_samples: int = 50
    # Number of _update_blacklist calls to skip after init/resume before
    # blacklist evaluation begins.  Prevents mass-blacklisting from resume shock.
    # ~500 calls ≈ 20-30 PPO iters with 8192 envs.
    curriculum_blacklist_warmup_iters: int = 0
    # Number of control steps per PPO iteration (num_steps_per_env).
    # curriculum_min_stage_steps and curriculum_fallback_min_stage_steps are
    # expressed in PPO iterations; multiply by this value to get control steps.
    curriculum_steps_per_iter: int = 32
    # Per-stage drift termination thresholds (m). If non-empty, overrides the fixed
    # threshold in bad_anchor_planar_drift_static. Indexed by curriculum stage;
    # last value is reused for all higher stages.
    curriculum_drift_termination_threshold: tuple[float, ...] = ()

    # ── Phase mode (dual-head) ────────────────────────────────────────────────
    # When enabled, each episode runs: STAND → TRACKING → STAND → time_out
    phase_mode_enabled: bool = False
    stand_duration_s: float = 1.0
    stand_body_pos: list[list[float]] = []  # FK of default standing pose (N_bodies × 3)
    stand_body_quat: list[list[float]] = []  # FK of default standing pose (N_bodies × 4)
    # joint_name → angle_rad for STAND reference. Empty = use robot default_joint_pos_nominal.
    stand_joint_pos: dict[str, float] = {}

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
