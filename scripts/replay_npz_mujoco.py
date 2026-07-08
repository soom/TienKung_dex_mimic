#!/usr/bin/env python3
"""Pure MuJoCo NPZ playlist replay.

This viewer does not run physics. It only writes qpos/qvel from NPZ frames,
calls mj_forward, and lets you switch motions with keyboard shortcuts.

Viewer keys:
  ]  next motion
  [  previous motion
  B  next motion
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

try:
    import mujoco.viewer as mujoco_viewer
except Exception:
    mujoco_viewer = None


DEFAULT_MJCF = Path("ros_deploy_pro/src/tienkung2_pro/assets/mjcf/tienkung_new.xml")
NEXT_MOTION_KEYS = {93, ord("]"), 66, ord("b"), ord("B")}
PREVIOUS_MOTION_KEYS = {91, ord("[")}
ROOT_BODY_ALIASES = ("pelvis", "Base_link")


@dataclass(frozen=True)
class MotionClipInfo:
    name: str
    path: Path


class NpzMotionClip:
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=True)
        self.path = npz_path
        self.name = npz_path.stem
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
        self.joint_names = [str(name) for name in data["joint_names"].tolist()]
        self.body_names = [str(name) for name in data["body_names"].tolist()]
        self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        self.joint_vel = (
            np.asarray(data["joint_vel"], dtype=np.float64)
            if "joint_vel" in data.files
            else np.zeros_like(self.joint_pos)
        )
        self.body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
        self.body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)
        self.body_lin_vel_w = (
            np.asarray(data["body_lin_vel_w"], dtype=np.float64)
            if "body_lin_vel_w" in data.files
            else np.zeros_like(self.body_pos_w)
        )
        self.body_ang_vel_w = (
            np.asarray(data["body_ang_vel_w"], dtype=np.float64)
            if "body_ang_vel_w" in data.files
            else np.zeros_like(self.body_pos_w)
        )
        self.total_frames = int(self.joint_pos.shape[0])
        self.root_body_index = self._resolve_root_body_index()

    def _resolve_root_body_index(self) -> int:
        for body_name in ROOT_BODY_ALIASES:
            if body_name in self.body_names:
                return self.body_names.index(body_name)
        return 0


class MotionLibrary:
    def __init__(self, motion_paths: list[Path]):
        if not motion_paths:
            raise RuntimeError("No NPZ motions available.")
        self._motions = [MotionClipInfo(name=path.stem, path=path) for path in motion_paths]
        self._cache: dict[int, NpzMotionClip] = {}

    @classmethod
    def from_directory(cls, npz_dir: Path) -> "MotionLibrary":
        motion_paths = sorted(path for path in npz_dir.glob("*.npz") if path.is_file())
        if not motion_paths:
            raise RuntimeError(f"No NPZ motions found in directory: {npz_dir}")
        return cls(motion_paths)

    @property
    def motion_count(self) -> int:
        return len(self._motions)

    @property
    def motion_names(self) -> list[str]:
        return [motion.name for motion in self._motions]

    def get_clip(self, motion_index: int) -> NpzMotionClip:
        index = motion_index % self.motion_count
        if index not in self._cache:
            self._cache[index] = NpzMotionClip(self._motions[index].path)
        return self._cache[index]


class KinematicReplay:
    def __init__(self, mjcf_path: Path, motion_library: MotionLibrary, start_index: int = 0):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.motion_library = motion_library
        self.default_qpos = self.data.qpos.copy()
        self.default_qvel = self.data.qvel.copy()
        self.root_qpos_adr, self.root_qvel_adr = self._find_root_addresses()
        self.current_motion_index = 0
        self.current_motion: NpzMotionClip | None = None
        self.current_frame = 0
        self._last_frame_time: float | None = None
        self._joint_qpos_addrs: list[int] = []
        self._joint_qvel_addrs: list[int] = []
        self.set_motion(start_index)

    def _find_root_addresses(self) -> tuple[int | None, int | None]:
        for joint_id in range(self.model.njnt):
            if int(self.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])
        return None, None

    def _resolve_joint_addresses(self, joint_names: list[str]) -> tuple[list[int], list[int]]:
        qpos_addrs: list[int] = []
        qvel_addrs: list[int] = []
        missing: list[str] = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                missing.append(joint_name)
                continue
            qpos_addrs.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_addrs.append(int(self.model.jnt_dofadr[joint_id]))
        if missing:
            raise RuntimeError(f"Joint(s) not found in MJCF: {missing}")
        return qpos_addrs, qvel_addrs

    def set_motion(self, motion_index: int) -> None:
        self.current_motion_index = motion_index % self.motion_library.motion_count
        self.current_motion = self.motion_library.get_clip(self.current_motion_index)
        self._joint_qpos_addrs, self._joint_qvel_addrs = self._resolve_joint_addresses(self.current_motion.joint_names)
        self.current_frame = 0
        self._last_frame_time = None
        self.apply_frame(0)
        print(
            f"[playlist] motion {self.current_motion_index + 1}/{self.motion_library.motion_count}: "
            f"{self.current_motion.name}  frames={self.current_motion.total_frames} fps={self.current_motion.fps:.2f}"
        )

    def next_motion(self) -> None:
        self.set_motion(self.current_motion_index + 1)

    def previous_motion(self) -> None:
        self.set_motion(self.current_motion_index - 1)

    def apply_frame(self, frame_index: int) -> None:
        if self.current_motion is None:
            return
        clip = self.current_motion
        frame = frame_index % clip.total_frames
        root_body_index = clip.root_body_index

        self.data.qpos[:] = self.default_qpos
        self.data.qvel[:] = self.default_qvel

        if self.root_qpos_adr is not None:
            self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = clip.body_pos_w[frame, root_body_index]
            self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = clip.body_quat_w[frame, root_body_index]
        if self.root_qvel_adr is not None:
            self.data.qvel[self.root_qvel_adr : self.root_qvel_adr + 3] = clip.body_ang_vel_w[frame, root_body_index]
            self.data.qvel[self.root_qvel_adr + 3 : self.root_qvel_adr + 6] = clip.body_lin_vel_w[frame, root_body_index]

        for joint_idx, qpos_adr in enumerate(self._joint_qpos_addrs):
            self.data.qpos[qpos_adr] = clip.joint_pos[frame, joint_idx]
        for joint_idx, qvel_adr in enumerate(self._joint_qvel_addrs):
            self.data.qvel[qvel_adr] = clip.joint_vel[frame, joint_idx]

        mujoco.mj_forward(self.model, self.data)
        self.current_frame = frame

    def maybe_advance(self, playback_rate: float) -> None:
        if self.current_motion is None:
            return
        now = time.monotonic()
        if self._last_frame_time is None:
            self._last_frame_time = now
            return
        frame_dt = 1.0 / max(self.current_motion.fps * playback_rate, 1e-6)
        elapsed = now - self._last_frame_time
        if elapsed < frame_dt:
            return
        frames_to_advance = max(1, int(elapsed / frame_dt))
        self._last_frame_time += frames_to_advance * frame_dt
        self.apply_frame(self.current_frame + frames_to_advance)

    def run_headless(self, steps: int, playback_rate: float, print_every: int) -> None:
        if self.current_motion is None:
            return
        for step_index in range(steps):
            self.apply_frame(step_index)
            if print_every > 0 and (step_index == 0 or (step_index + 1) % print_every == 0 or step_index + 1 == steps):
                base_pos = self.data.qpos[:3].copy() if self.model.nq >= 3 else np.zeros(3, dtype=np.float64)
                print(f"step={step_index + 1:04d} frame={self.current_frame:04d} base_pos={base_pos}")
            if playback_rate > 0.0 and self.current_motion.fps > 0.0:
                time.sleep(1.0 / (self.current_motion.fps * playback_rate))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure MuJoCo NPZ playlist replay with [ and ] navigation.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="Path to MuJoCo MJCF/XML robot model")
    parser.add_argument("--npz-dir", type=Path, required=True, help="Directory of NPZ motions to replay")
    parser.add_argument("--start-index", type=int, default=0, help="Initial motion index")
    parser.add_argument("--playback-rate", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--steps", type=int, default=1, help="Headless mode only: number of frames to apply")
    parser.add_argument("--print-every", type=int, default=50, help="Print status every N frames in headless mode")
    return parser


def run_with_viewer(replay: KinematicReplay, playback_rate: float) -> None:
    if mujoco_viewer is None:
        raise RuntimeError("MuJoCo viewer unavailable. Run with --headless.")

    if hasattr(mujoco_viewer, "glfw"):
        glfw = mujoco_viewer.glfw
        if hasattr(glfw, "KEY_LEFT_BRACKET"):
            PREVIOUS_MOTION_KEYS.add(int(glfw.KEY_LEFT_BRACKET))
        if hasattr(glfw, "KEY_RIGHT_BRACKET"):
            NEXT_MOTION_KEYS.add(int(glfw.KEY_RIGHT_BRACKET))
        if hasattr(glfw, "KEY_B"):
            NEXT_MOTION_KEYS.add(int(glfw.KEY_B))

    def on_key(keycode: int) -> None:
        if keycode in PREVIOUS_MOTION_KEYS:
            replay.previous_motion()
        elif keycode in NEXT_MOTION_KEYS:
            replay.next_motion()

    print("[viewer] ]: next motion, [: previous motion")
    with mujoco_viewer.launch_passive(replay.model, replay.data, key_callback=on_key,show_left_ui=False,show_right_ui=False) as viewer:
        while viewer.is_running():
            replay.maybe_advance(playback_rate)
            viewer.sync()
            time.sleep(0.001)


def main() -> None:
    args = build_argparser().parse_args()
    if args.playback_rate <= 0.0:
        raise ValueError(f"`--playback-rate` must be positive, got {args.playback_rate}")

    mjcf_path = args.mjcf.expanduser().resolve()
    npz_dir = args.npz_dir.expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    motion_library = MotionLibrary.from_directory(npz_dir)
    print(f"[playlist] Loaded {motion_library.motion_count} motions from {npz_dir}")
    for motion_index, motion_name in enumerate(motion_library.motion_names):
        print(f"  [{motion_index:02d}] {motion_name}")

    replay = KinematicReplay(
        mjcf_path=mjcf_path,
        motion_library=motion_library,
        start_index=args.start_index,
    )

    if args.headless:
        replay.run_headless(steps=args.steps, playback_rate=args.playback_rate, print_every=args.print_every)
    else:
        run_with_viewer(replay, playback_rate=args.playback_rate)


if __name__ == "__main__":
    main()