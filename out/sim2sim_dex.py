"""Standalone Mujoco sim2sim runner for Dex EVT ONNX motion policies.

This script intentionally avoids importing the project's Isaac/whole_body_tracking
runtime code. It only depends on standard Python packages plus `mujoco`, `numpy`,
`onnxruntime`, and optionally `pinocchio` (for teleop IK mode).

Modes:
  default  : replay pre-baked ONNX reference trajectory (teacher verification)
  --npz    : teleop simulation — arm joints solved via real-time pinocchio IK
             from NPZ body_pos_w wrist targets; legs use NPZ joint_pos directly
"""


import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort

try:
    import mujoco.viewer as mujoco_viewer
except Exception:
    mujoco_viewer = None

DEFAULT_POLICY = Path(
    "logs/rsl_rl/dex_evt_fix/exported/policy.onnx"
)
DEFAULT_MJCF = Path("source/whole_body_tracking/whole_body_tracking/assets/dex_evt/urdf/dex_evt.xml")

LOOKAHEAD_STEPS = 2
VIEWER_KEY_B = 66
ARM_CLEARANCE_DEFAULT_M = 0.04
ARM_CLEARANCE_DISTMAX_M = 0.25

LEFT_ARM_KEYWORDS = ("shoulder_l", "shoulder_pitch_l", "shoulder_roll_l", "shoulder_yaw_l", "elbow_", "wrist_", "left_", "_l_link")
RIGHT_ARM_KEYWORDS = ("shoulder_r", "shoulder_pitch_r", "shoulder_roll_r", "shoulder_yaw_r", "elbow_", "wrist_", "right_", "_r_link")
ARM_BODY_KEYWORDS = ("shoulder", "elbow", "wrist", "hand", "tcp")


class ArmClearanceViolation(RuntimeError):
    pass


def format_dynamics_extra(sim: "MujocoSim2Sim") -> str:
    if not sim.debug_dynamics or not sim._last_dynamics:
        if sim._last_arm_clearance is None:
            return ""
        return f" arm={float(sim._last_arm_clearance['dist']):+.3f}"

    d = sim._last_dynamics
    extra = (
        f" action={d['max_abs_action']:.2f}"
        f" qerr={d['max_abs_q_err']:.3f}"
        f" qderr={d['max_abs_qd_err']:.2f}"
        f" force={d['max_force_ratio']:.2f}/{d['mean_force_ratio']:.2f}"
        f" fj={d['top_force_joint']}"
        f" qj={d['top_qerr_joint']}"
        f" footF={d['foot_force_l']:.1f},{d['foot_force_r']:.1f}"
    )
    if "arm_clearance" in d:
        extra += f" arm={d['arm_clearance']:+.3f}"
    return extra


def quat_normalize(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat_wxyz)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat_wxyz / norm


def quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat_wxyz)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_multiply(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = quat_normalize(lhs_wxyz)
    rw, rx, ry, rz = quat_normalize(rhs_wxyz)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = yaw / 2.0
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    rotmat = quat_to_rotmat(quat_wxyz)
    return float(np.arctan2(rotmat[1, 0], rotmat[0, 0]))


def rotate_world_z(values: np.ndarray, yaw: float) -> np.ndarray:
    rotated = np.array(values, dtype=np.float64, copy=True)
    c = np.cos(yaw)
    s = np.sin(yaw)
    x = rotated[..., 0].copy()
    y = rotated[..., 1].copy()
    rotated[..., 0] = c * x - s * y
    rotated[..., 1] = s * x + c * y
    return rotated


def quat_rotate(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    rotmat = quat_to_rotmat(quat_wxyz)
    return np.einsum("ij,...j->...i", rotmat, np.asarray(vectors, dtype=np.float64))


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    return np.array(
        [rotmat[0, 0], rotmat[0, 1], rotmat[1, 0], rotmat[1, 1], rotmat[2, 0], rotmat[2, 1]],
        dtype=np.float32,
    )


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def csv_metadata_array(metadata: dict[str, str], key: str, cast) -> list:
    raw = metadata.get(key, "")
    if not raw:
        return []
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def infer_total_steps(onnx_model: onnx.ModelProto) -> int:
    # Read from metadata (written by attach_onnx_metadata for single-clip exports)
    for entry in onnx_model.metadata_props:
        if entry.key == "total_steps":
            return int(entry.value)
    # Dual-head ONNX has no baked motion data — total_steps is per-NPZ
    return 0


@dataclass(frozen=True)
class PolicyMetadata:
    joint_names: list[str]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    joint_effort_limit: np.ndarray
    body_names: list[str]
    anchor_body_name: str
    observation_names: list[str]
    total_steps: int
    obs_dim: int
    action_dim: int

    @classmethod
    def from_onnx(cls, onnx_path: Path) -> "PolicyMetadata":
        model = onnx.load(str(onnx_path))
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        joint_names = csv_metadata_array(metadata, "joint_names", str)
        default_joint_pos = np.asarray(csv_metadata_array(metadata, "default_joint_pos", float), dtype=np.float32)
        action_scale = np.asarray(csv_metadata_array(metadata, "action_scale", float), dtype=np.float32)
        joint_stiffness = np.asarray(csv_metadata_array(metadata, "joint_stiffness", float), dtype=np.float32)
        joint_damping = np.asarray(csv_metadata_array(metadata, "joint_damping", float), dtype=np.float32)
        joint_effort_limit = np.asarray(csv_metadata_array(metadata, "joint_effort_limit", float), dtype=np.float32)
        body_names = csv_metadata_array(metadata, "body_names", str)
        observation_names = csv_metadata_array(metadata, "observation_names", str)
        if not joint_names:
            raise RuntimeError("Missing `joint_names` metadata in ONNX file.")
        if default_joint_pos.shape[0] != len(joint_names):
            raise RuntimeError("`default_joint_pos` length does not match `joint_names`.")
        if action_scale.shape[0] != len(joint_names):
            raise RuntimeError("`action_scale` length does not match `joint_names`.")
        if joint_stiffness.shape[0] != len(joint_names):
            raise RuntimeError("`joint_stiffness` length does not match `joint_names`.")
        if joint_damping.shape[0] != len(joint_names):
            raise RuntimeError("`joint_damping` length does not match `joint_names`.")
        if joint_effort_limit.shape[0] == 0:
            joint_effort_limit = np.maximum(action_scale * joint_stiffness / 0.1875, joint_stiffness * 0.5)
        elif joint_effort_limit.shape[0] != len(joint_names):
            raise RuntimeError("`joint_effort_limit` length does not match `joint_names`.")
        obs_dim = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
        action_dim = int(model.graph.output[0].type.tensor_type.shape.dim[1].dim_value)
        return cls(
            joint_names=joint_names,
            default_joint_pos=default_joint_pos,
            action_scale=action_scale,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            joint_effort_limit=joint_effort_limit,
            body_names=body_names,
            anchor_body_name=metadata.get("anchor_body_name", "pelvis"),
            observation_names=observation_names,
            total_steps=infer_total_steps(model),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )


@dataclass(frozen=True)
class MotionClipInfo:
    name: str
    path: Path


@dataclass
class BlendTransition:
    start_frame: dict
    end_frame: dict
    total_steps: int
    step: int = 0


class OnnxMotionPolicy:
    def __init__(self, policy_path: Path):
        self.policy_path = policy_path
        self.metadata = PolicyMetadata.from_onnx(policy_path)
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.input_obs_name = inputs[0].name
        self.input_step_name = inputs[1].name if len(inputs) > 1 else None
        self.output_names = [output.name for output in self.session.get_outputs()]

    def infer(self, obs: np.ndarray, time_step: int) -> dict[str, np.ndarray]:
        feed = {self.input_obs_name: obs.reshape(1, -1).astype(np.float32)}
        if self.input_step_name is not None:
            feed[self.input_step_name] = np.array([[time_step]], dtype=np.float32)
        outputs = self.session.run(self.output_names, feed)
        return {name: value[0] for name, value in zip(self.output_names, outputs)}


WRIST_BODY_NAMES = ["wrist_roll_l_link", "wrist_roll_r_link"]
FOOT_BODY_NAMES = ["ankle_roll_l_link", "ankle_roll_r_link"]

# Arm joints solved by IK (7-DOF per side: shoulder×3 + elbow×2 + wrist×2)
_ARM_JOINTS_L = [
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
]
_ARM_JOINTS_R = [
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
]
# Dex: 3-DOF waist (yaw, roll, pitch) vs Pro's 1-DOF body_yaw
_LEG_JOINTS = [
    "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

# ---------------------------------------------------------------------------
# NPZ teleop stream
# ---------------------------------------------------------------------------

class NpzTeleopStream:
    """Simulates a real-time teleop input stream from a pre-recorded NPZ file.

    Each call to next_frame() returns:
      - q_ref_full  : (29,) full joint reference — legs from NPZ, arms from IK
      - qd_ref_full : (29,) joint velocity reference — legs from NPZ, arms zeroed
      - body_pos_w  : (N_bodies, 3) reference body positions (for obs construction)
      - body_quat_w : (N_bodies, 4) reference body orientations
      - wrist_pos_l/r: (3,) wrist EE targets used for IK (world frame, pelvis-relative)
    """

    def __init__(
        self,
        npz_path: str,
        policy_joint_names: list[str],
        policy_body_names: list[str],
        low_pass_alpha: float = 0.4,
    ):
        data = np.load(npz_path, allow_pickle=False)
        self.fps = float(np.asarray(data["fps"]).item())
        self.T = int(data["joint_pos"].shape[0])

        # Resolve joint ordering: NPZ → policy order
        npz_joint_names = [str(n) for n in data["joint_names"].tolist()]
        self._joint_order = [npz_joint_names.index(n) for n in policy_joint_names]
        self._joint_pos = data["joint_pos"][:, self._joint_order].astype(np.float32)
        self._joint_vel = data["joint_vel"][:, self._joint_order].astype(np.float32)

        # Body positions/quats (all bodies, for obs)
        self._body_pos_w = data["body_pos_w"].astype(np.float64)   # (T, N, 3)
        self._body_quat_w = data["body_quat_w"].astype(np.float64) # (T, N, 4)
        if "body_lin_vel_w" in data.files:
            self._body_lin_vel_w = data["body_lin_vel_w"].astype(np.float64)
        else:
            self._body_lin_vel_w = np.zeros_like(self._body_pos_w)
        if "body_ang_vel_w" in data.files:
            self._body_ang_vel_w = data["body_ang_vel_w"].astype(np.float64)
        else:
            self._body_ang_vel_w = np.zeros_like(self._body_pos_w)

        # Wrist body indices in NPZ body array
        if "body_names" not in data.files:
            raise RuntimeError(
                f"NPZ file '{npz_path}' is missing 'body_names'. "
                "Please regenerate it with the latest scripts/pkl_to_npz.py."
            )
        npz_body_names = [str(n) for n in data["body_names"].tolist()]
        self._wrist_idx_l = npz_body_names.index("wrist_roll_l_link")
        self._wrist_idx_r = npz_body_names.index("wrist_roll_r_link")

        # Policy body indices (for obs body_pos_w / body_quat_w slicing)
        self._body_order = [npz_body_names.index(n) for n in policy_body_names]

        # Arm joint indices in policy joint list
        self._arm_idx_l = [policy_joint_names.index(n) for n in _ARM_JOINTS_L if n in policy_joint_names]
        self._arm_idx_r = [policy_joint_names.index(n) for n in _ARM_JOINTS_R if n in policy_joint_names]

        self.npz_path = npz_path
        self._alpha = low_pass_alpha
        self._q_arm_l_filt: np.ndarray | None = None
        self._q_arm_r_filt: np.ndarray | None = None
        self._cache_next_step = 0
        self._cursor = 0
        self._frame_cache: dict[int, dict[str, np.ndarray]] = {}
        print(f"[TeleopStream] Loaded {npz_path}  T={self.T}  fps={self.fps}")
        mode = "npz_joint_track"
        print(f"[TeleopStream] Arm mode={mode}  joints L={len(self._arm_idx_l)} R={len(self._arm_idx_r)}")

    @property
    def total_steps(self) -> int:
        return self.T

    def reset(self) -> None:
        self._cache_next_step = 0
        self._cursor = 0
        self._frame_cache.clear()
        self._q_arm_l_filt = None
        self._q_arm_r_filt = None

    def _compute_frame(self, step_index: int) -> dict[str, np.ndarray]:
        t = min(step_index, self.T - 1)

        # Wrist targets: world frame, pelvis-relative
        pelvis_pos = self._body_pos_w[t, 0, :]
        wrist_pos_l = self._body_pos_w[t, self._wrist_idx_l, :] - pelvis_pos
        wrist_pos_r = self._body_pos_w[t, self._wrist_idx_r, :] - pelvis_pos

        q_ref = self._joint_pos[t].copy()
        qd_ref = self._joint_vel[t].copy()

        # Body arrays sliced to policy body order
        body_pos_w = self._body_pos_w[t][self._body_order]
        body_quat_w = self._body_quat_w[t][self._body_order]
        body_lin_vel_w = self._body_lin_vel_w[t][self._body_order]
        body_ang_vel_w = self._body_ang_vel_w[t][self._body_order]

        return {
            "joint_pos": q_ref,
            "joint_vel": qd_ref,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "wrist_pos_l": wrist_pos_l.astype(np.float32),
            "wrist_pos_r": wrist_pos_r.astype(np.float32),
        }

    def reference_at(self, step_index: int) -> dict[str, np.ndarray]:
        t = min(max(step_index, 0), self.T - 1)
        while self._cache_next_step <= t:
            self._frame_cache[self._cache_next_step] = self._compute_frame(self._cache_next_step)
            self._cache_next_step += 1
        return {key: np.array(value, copy=True) for key, value in self._frame_cache[t].items()}

    def next_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame = self.reference_at(self._cursor)
        if self._cursor < self.T - 1:
            self._cursor += 1
        return (
            frame["joint_pos"],
            frame["joint_vel"],
            frame["body_pos_w"],
            frame["body_quat_w"],
            frame["wrist_pos_l"],
            frame["wrist_pos_r"],
        )


class MotionLibrary:
    def __init__(
        self,
        motions: list[MotionClipInfo],
        policy_joint_names: list[str],
        policy_body_names: list[str],
        low_pass_alpha: float = 0.4,
    ):
        if not motions:
            raise RuntimeError("No NPZ motions available.")
        self.motions = motions
        self.policy_joint_names = policy_joint_names
        self.policy_body_names = policy_body_names
        self.low_pass_alpha = low_pass_alpha
        self._stream_cache: dict[int, NpzTeleopStream] = {}

    @classmethod
    def from_directory(
        cls,
        npz_dir: Path,
        policy_joint_names: list[str],
        policy_body_names: list[str],
        low_pass_alpha: float = 0.4,
    ) -> "MotionLibrary":
        motion_paths = sorted(path for path in npz_dir.glob("*.npz") if path.is_file())
        if not motion_paths:
            raise RuntimeError(f"No NPZ motions found in directory: {npz_dir}")
        return cls(
            motions=[MotionClipInfo(name=path.stem, path=path) for path in motion_paths],
            policy_joint_names=policy_joint_names,
            policy_body_names=policy_body_names,
            low_pass_alpha=low_pass_alpha,
        )

    @property
    def motion_names(self) -> list[str]:
        return [motion.name for motion in self.motions]

    @property
    def motion_count(self) -> int:
        return len(self.motions)

    def find_idle_index(self) -> int | None:
        """Return index of the init_pose / idle motion (contains 'init_pose'), or None."""
        for i, motion in enumerate(self.motions):
            if "init_pose" in motion.name:
                return i
        return None

    def get_stream(self, motion_index: int) -> NpzTeleopStream:
        if motion_index not in self._stream_cache:
            motion = self.motions[motion_index]
            self._stream_cache[motion_index] = NpzTeleopStream(
                npz_path=str(motion.path),
                policy_joint_names=self.policy_joint_names,
                policy_body_names=self.policy_body_names,
                low_pass_alpha=self.low_pass_alpha,
            )
        return self._stream_cache[motion_index]


# Dex foot geoms: 7 per foot, matching training URDF collision geometry
FOOT_GEOM_NAMES = [
    "foot_left_front_outer", "foot_left_front_inner",
    "foot_left_mid_outer", "foot_left_mid_inner",
    "foot_left_strip_outer", "foot_left_strip_center", "foot_left_strip_inner",
    "foot_right_front_outer", "foot_right_front_inner",
    "foot_right_mid_outer", "foot_right_mid_inner",
    "foot_right_strip_outer", "foot_right_strip_center", "foot_right_strip_inner",
]
SLIP_CONTACT_FORCE_THRESHOLD = 5.0  # N — below this the foot is considered airborne

# MuJoCo steady-state equilibrium standing pose (symmetrized).
# Measured by running default_pose for 6s and averaging the last 2s.
# Replaces the ONNX default_joint_pos which is from training config and
# does not match MuJoCo's true balance point.
DEX_EVT_STANDING_JOINT_POS = {
    "hip_pitch_l_joint": -0.08609,
    "hip_pitch_r_joint": -0.08609,
    "waist_yaw_joint": 0.00000,
    "hip_roll_l_joint": 0.00948,
    "hip_roll_r_joint": -0.00948,
    "waist_roll_joint": 0.00000,
    "hip_yaw_l_joint": -0.01497,
    "hip_yaw_r_joint": -0.01497,
    "waist_pitch_joint": 0.00000,
    "knee_pitch_l_joint": 0.05264,
    "knee_pitch_r_joint": 0.05264,
    "shoulder_pitch_l_joint": -0.01700,
    "shoulder_pitch_r_joint": -0.01700,
    "ankle_pitch_l_joint": 0.01439,
    "ankle_pitch_r_joint": 0.01439,
    "shoulder_roll_l_joint": 0.08423,
    "shoulder_roll_r_joint": -0.08423,
    "ankle_roll_l_joint": -0.00904,
    "ankle_roll_r_joint": 0.00904,
    "shoulder_yaw_l_joint": -0.00290,
    "shoulder_yaw_r_joint": -0.00290,
    "elbow_pitch_l_joint": -0.09574,
    "elbow_pitch_r_joint": -0.09574,
    "elbow_yaw_l_joint": 0.00700,
    "elbow_yaw_r_joint": 0.00700,
    "wrist_pitch_l_joint": -0.00142,
    "wrist_pitch_r_joint": -0.00142,
    "wrist_roll_l_joint": -0.00003,
    "wrist_roll_r_joint": -0.00003,
}


class MujocoSim2Sim:
    def __init__(
        self,
        mjcf_path: Path,
        policy: OnnxMotionPolicy,
        control_dt: float,
        sim_dt: float,
        zero_xy_ref: bool,
        max_target_step: float | None,
        log_slip: bool = False,
        slip_vel_threshold: float = 0.05,
        teleop_stream: "NpzTeleopStream | None" = None,
        motion_library: "MotionLibrary | None" = None,
        stand_duration_s: float = 1.0,
        zero_action: bool = False,
        debug_dynamics: bool = False,
        effort_scale: float = 1.0,
        action_replay: "np.ndarray | None" = None,
        action_replay_offset: int = 0,
        arm_clearance_threshold: float = ARM_CLEARANCE_DEFAULT_M,
        stop_on_arm_clearance: bool = False,
    ):
        self.mjcf_path = mjcf_path
        self.policy = policy
        self.metadata = policy.metadata
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt
        self.control_dt = control_dt
        self.sim_dt = sim_dt
        self.decimation = max(1, int(round(control_dt / sim_dt)))
        self.zero_xy_ref = zero_xy_ref
        self.max_target_step = max_target_step
        self.log_slip = log_slip
        self.slip_vel_threshold = slip_vel_threshold
        self.teleop_stream = teleop_stream
        self.motion_library = motion_library
        if self.motion_library is not None and self.teleop_stream is not None:
            raise RuntimeError("Use either `teleop_stream` or `motion_library`, not both.")
        self.stand_duration_s = max(float(stand_duration_s), self.control_dt)
        self.zero_action = zero_action
        self.debug_dynamics = debug_dynamics
        self.effort_scale = float(effort_scale)
        self.action_replay = action_replay
        self.action_replay_offset = int(action_replay_offset)
        self.arm_clearance_threshold = float(arm_clearance_threshold)
        self.stop_on_arm_clearance = stop_on_arm_clearance
        self._stand_total_steps = max(1, int(round(self.stand_duration_s / self.control_dt)))
        self._request_lock = threading.Lock()
        self._pending_next_motion_requests = 0
        self._playback_phase = "motion_playback"
        self._active_motion_index: int | None = None
        self._active_motion_name: str | None = None
        self._next_motion_index: int = 0
        self._stand_step_count: int = 0
        self._return_stand_start_ref: dict[str, np.ndarray] | None = None

        self.joint_qpos_addr = self._build_joint_qpos_addr()
        self.joint_qvel_addr = self._build_joint_qvel_addr()
        self.actuator_ids = self._build_actuator_ids()
        self.joint_qpos_indices = np.array(
            [self.joint_qpos_addr[joint_name] for joint_name in self.metadata.joint_names],
            dtype=np.int32,
        )
        self.joint_qvel_indices = np.array(
            [self.joint_qvel_addr[joint_name] for joint_name in self.metadata.joint_names],
            dtype=np.int32,
        )
        self.actuator_index_array = np.array(
            [self.actuator_ids[joint_name] for joint_name in self.metadata.joint_names],
            dtype=np.int32,
        )
        self.sensor_slices = self._build_sensor_slices()
        self._apply_policy_dynamics()
        self._foot_body_ids, self._foot_geom_ids = self._build_foot_ids()
        self._anchor_body_id = self._build_anchor_body_id()
        self._left_arm_geom_ids, self._right_arm_geom_ids = self._build_arm_clearance_geom_ids()

        # Standing joint positions: use MuJoCo-measured equilibrium instead of
        # ONNX config default_joint_pos which doesn't match MuJoCo balance point.
        standing_map = {n: float(v) for n, v in DEX_EVT_STANDING_JOINT_POS.items()}
        self._standing_joint_pos = np.array(
            [standing_map[n] for n in self.metadata.joint_names], dtype=np.float32
        )

        self.zero_obs = np.zeros(self.metadata.obs_dim, dtype=np.float32)
        self._reference_xy_offset = np.zeros(2, dtype=np.float64)
        self._reference_yaw = 0.0
        self._reference_yaw_inv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if self.teleop_stream is not None:
            self.reference_outputs = []
            self._set_reference_frame(self.teleop_stream.reference_at(0))
        elif self.motion_library is not None:
            self.reference_outputs = []
        else:
            self.reference_outputs = self._build_reference_outputs()
        self.default_reference = self._build_default_reference() if self.motion_library is not None else None
        self.last_action = np.zeros(self.metadata.action_dim, dtype=np.float32)
        self.current_step = 0

        # slip tracking
        self._slip_log: list[dict] = []

        # standing action bias — calibrated at startup to cancel systematic drift
        self._standing_action_bias: np.ndarray | None = None
        self._last_dynamics: dict[str, float] = {}
        self._arm_clearance_log: list[dict] = []
        self._last_arm_clearance_warning_step = -10_000
        self._last_arm_clearance: dict | None = None

        self.reset()

    def _build_joint_qpos_addr(self) -> dict[str, int]:
        addr: dict[str, int] = {}
        for joint_name in self.metadata.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint `{joint_name}` not found in Mujoco model.")
            addr[joint_name] = int(self.model.jnt_qposadr[joint_id])
        return addr

    def _build_joint_qvel_addr(self) -> dict[str, int]:
        addr: dict[str, int] = {}
        for joint_name in self.metadata.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint `{joint_name}` not found in Mujoco model.")
            addr[joint_name] = int(self.model.jnt_dofadr[joint_id])
        return addr

    def _build_actuator_ids(self) -> dict[str, int]:
        actuator_ids: dict[str, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name in self.metadata.joint_names:
                actuator_ids[joint_name] = actuator_id
        missing = [joint_name for joint_name in self.metadata.joint_names if joint_name not in actuator_ids]
        if missing:
            raise RuntimeError(f"Missing position actuator for joints: {missing}")
        return actuator_ids

    def _build_sensor_slices(self) -> dict[str, slice]:
        sensor_dims = {
            "orientation": 4,
            "position": 3,
            "angular-velocity": 3,
            "linear-velocity": 3,
        }
        slices: dict[str, slice] = {}
        for sensor_name, expected_dim in sensor_dims.items():
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor_id < 0:
                raise RuntimeError(f"Sensor `{sensor_name}` not found in Mujoco model.")
            start = int(self.model.sensor_adr[sensor_id])
            dim = int(self.model.sensor_dim[sensor_id])
            if dim != expected_dim:
                raise RuntimeError(
                    f"Sensor `{sensor_name}` dim mismatch: expected {expected_dim}, got {dim}."
                )
            slices[sensor_name] = slice(start, start + dim)
        return slices

    def _apply_policy_dynamics(self) -> None:
        for joint_index, joint_name in enumerate(self.metadata.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            dof_id = int(self.model.jnt_dofadr[joint_id])
            actuator_id = self.actuator_ids[joint_name]

            kp = float(self.metadata.joint_stiffness[joint_index])
            kd = float(self.metadata.joint_damping[joint_index])
            effort_limit = float(self.metadata.joint_effort_limit[joint_index]) * self.effort_scale

            self.model.dof_damping[dof_id] = 0.0
            self.model.actuator_gainprm[actuator_id, 0] = kp
            self.model.actuator_biasprm[actuator_id, 1] = -kp
            self.model.actuator_biasprm[actuator_id, 2] = -kd
            self.model.actuator_forcelimited[actuator_id] = 1
            self.model.actuator_forcerange[actuator_id, 0] = -effort_limit
            self.model.actuator_forcerange[actuator_id, 1] = effort_limit

    def _resolve_body_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return body_id
        # Dex MJCF uses "pelvis" directly — no alias needed
        return -1

    def _build_anchor_body_id(self) -> int | None:
        body_id = self._resolve_body_id(self.metadata.anchor_body_name)
        return None if body_id < 0 else body_id

    def _build_foot_ids(self) -> tuple[list[int], set[int]]:
        body_ids = []
        for name in FOOT_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"Foot body `{name}` not found in Mujoco model.")
            body_ids.append(bid)
        geom_ids = set()
        for name in FOOT_GEOM_NAMES:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                geom_ids.add(gid)
        return body_ids, geom_ids

    def _build_arm_clearance_geom_ids(self) -> tuple[list[int], list[int]]:
        left_ids: list[int] = []
        right_ids: list[int] = []
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            full_name = f"{body_name}/{geom_name}"
            if not any(keyword in full_name for keyword in ARM_BODY_KEYWORDS):
                continue
            if "head" in full_name or "waist" in full_name:
                continue
            if any(keyword in full_name for keyword in LEFT_ARM_KEYWORDS) and "_r_" not in full_name and "right" not in full_name:
                left_ids.append(geom_id)
            if any(keyword in full_name for keyword in RIGHT_ARM_KEYWORDS) and "_l_" not in full_name and "left" not in full_name:
                right_ids.append(geom_id)
        if not left_ids or not right_ids:
            print(
                "[arm_clearance] warning: could not find both left/right arm geoms "
                f"(left={len(left_ids)}, right={len(right_ids)}); clearance monitor disabled."
            )
        else:
            print(f"[arm_clearance] monitoring {len(left_ids)} left-arm geom(s) x {len(right_ids)} right-arm geom(s)")
        return left_ids, right_ids

    def _geom_label(self, geom_id: int) -> str:
        geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom[{geom_id}]"
        body_id = int(self.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body[{body_id}]"
        return f"{geom_name}({body_name})"

    def _check_arm_clearance(self, step_index: int) -> None:
        if self.arm_clearance_threshold < 0.0:
            return
        if not self._left_arm_geom_ids or not self._right_arm_geom_ids:
            return

        best_dist = float("inf")
        best_pair: tuple[int, int] | None = None
        best_fromto = np.zeros(6, dtype=np.float64)
        fromto = np.zeros(6, dtype=np.float64)
        for left_gid in self._left_arm_geom_ids:
            for right_gid in self._right_arm_geom_ids:
                dist = float(
                    mujoco.mj_geomDistance(
                        self.model,
                        self.data,
                        left_gid,
                        right_gid,
                        ARM_CLEARANCE_DISTMAX_M,
                        fromto,
                    )
                )
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (left_gid, right_gid)
                    best_fromto[:] = fromto

        if best_pair is None:
            return

        entry = {
            "step": int(step_index),
            "dist": float(best_dist),
            "left_geom": int(best_pair[0]),
            "right_geom": int(best_pair[1]),
            "fromto": best_fromto.copy(),
        }
        self._last_arm_clearance = entry
        self._arm_clearance_log.append(entry)

        if best_dist <= self.arm_clearance_threshold:
            left_label = self._geom_label(best_pair[0])
            right_label = self._geom_label(best_pair[1])
            msg = (
                f"[ARM_CLEARANCE] step={step_index:04d} dist={best_dist:+.4f}m "
                f"threshold={self.arm_clearance_threshold:.4f}m "
                f"{left_label} <-> {right_label}"
            )
            if step_index - self._last_arm_clearance_warning_step >= 10 or best_dist < 0.0:
                print(msg)
                self._last_arm_clearance_warning_step = int(step_index)
            if self.stop_on_arm_clearance:
                raise ArmClearanceViolation(msg)

    def _foot_contact_forces(self) -> tuple[float, float]:
        """Return total normal contact force (N) for left and right foot."""
        force_l = force_r = 0.0
        left_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in FOOT_GEOM_NAMES if "left" in n
        }
        right_geoms = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in FOOT_GEOM_NAMES if "right" in n
        }
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            cf = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, cf)
            normal = abs(float(cf[0]))
            if pair & left_geoms:
                force_l += normal
            if pair & right_geoms:
                force_r += normal
        return force_l, force_r

    def _foot_xy_vel(self) -> tuple[float, float]:
        """Return XY speed (m/s) of each foot body CoM in world frame."""
        vels = []
        for bid in self._foot_body_ids:
            lin_vel = self.data.cvel[bid, 3:6]
            vels.append(float(np.linalg.norm(lin_vel[:2])))
        return vels[0], vels[1]

    def _record_slip(self, step_index: int) -> None:
        force_l, force_r = self._foot_contact_forces()
        vel_l, vel_r = self._foot_xy_vel()
        contact_l = force_l > SLIP_CONTACT_FORCE_THRESHOLD
        contact_r = force_r > SLIP_CONTACT_FORCE_THRESHOLD
        slip_l = vel_l if contact_l else 0.0
        slip_r = vel_r if contact_r else 0.0
        entry = {
            "step": step_index,
            "force_l": force_l, "force_r": force_r,
            "vel_l": vel_l, "vel_r": vel_r,
            "contact_l": contact_l, "contact_r": contact_r,
            "slip_l": slip_l, "slip_r": slip_r,
        }
        self._slip_log.append(entry)
        if self.log_slip:
            slip_flag = ""
            if contact_l and vel_l > self.slip_vel_threshold:
                slip_flag += " [SLIP_L]"
            if contact_r and vel_r > self.slip_vel_threshold:
                slip_flag += " [SLIP_R]"
            if slip_flag or (contact_l or contact_r):
                print(
                    f"  slip step={step_index:04d} "
                    f"L: contact={contact_l} F={force_l:5.1f}N vel={vel_l:.3f}m/s  "
                    f"R: contact={contact_r} F={force_r:5.1f}N vel={vel_r:.3f}m/s"
                    f"{slip_flag}"
                )

    def print_slip_summary(self) -> None:
        if not self._slip_log:
            print("No slip data recorded.")
            return
        contact_steps_l = [e for e in self._slip_log if e["contact_l"]]
        contact_steps_r = [e for e in self._slip_log if e["contact_r"]]
        slip_steps_l = [e for e in contact_steps_l if e["slip_l"] > self.slip_vel_threshold]
        slip_steps_r = [e for e in contact_steps_r if e["slip_r"] > self.slip_vel_threshold]

        def stats(entries, key):
            if not entries:
                return 0.0, 0.0, 0.0
            vals = [e[key] for e in entries]
            return float(np.mean(vals)), float(np.max(vals)), float(np.percentile(vals, 95))

        mean_l, max_l, p95_l = stats(contact_steps_l, "slip_l")
        mean_r, max_r, p95_r = stats(contact_steps_r, "slip_r")

        total = len(self._slip_log)
        print("\n=== Foot Slip Summary ===")
        print(f"  total steps: {total}")
        print(f"  LEFT  — contact: {len(contact_steps_l)} steps  slip>{self.slip_vel_threshold:.2f}m/s: {len(slip_steps_l)} steps  "
              f"mean={mean_l:.3f} max={max_l:.3f} p95={p95_l:.3f} m/s")
        print(f"  RIGHT — contact: {len(contact_steps_r)} steps  slip>{self.slip_vel_threshold:.2f}m/s: {len(slip_steps_r)} steps  "
              f"mean={mean_r:.3f} max={max_r:.3f} p95={p95_r:.3f} m/s")
        slip_rate_l = len(slip_steps_l) / max(len(contact_steps_l), 1) * 100
        slip_rate_r = len(slip_steps_r) / max(len(contact_steps_r), 1) * 100
        print(f"  slip rate: L={slip_rate_l:.1f}%  R={slip_rate_r:.1f}%")

    def print_arm_clearance_summary(self) -> None:
        if self.arm_clearance_threshold < 0.0:
            print("\n=== Arm Clearance Summary ===")
            print("  disabled (--arm-clearance-threshold < 0)")
            return
        if not self._left_arm_geom_ids or not self._right_arm_geom_ids:
            print("\n=== Arm Clearance Summary ===")
            print("  disabled: no left/right arm geoms found")
            return
        if not self._arm_clearance_log:
            print("\n=== Arm Clearance Summary ===")
            print("  no arm clearance data recorded")
            return

        min_entry = min(self._arm_clearance_log, key=lambda entry: entry["dist"])
        below = [entry for entry in self._arm_clearance_log if entry["dist"] <= self.arm_clearance_threshold]
        penetrating = [entry for entry in self._arm_clearance_log if entry["dist"] < 0.0]
        left_label = self._geom_label(int(min_entry["left_geom"]))
        right_label = self._geom_label(int(min_entry["right_geom"]))

        print("\n=== Arm Clearance Summary ===")
        print(f"  threshold: {self.arm_clearance_threshold:.4f} m")
        print(
            f"  min: {float(min_entry['dist']):+.4f} m at step={int(min_entry['step']):04d}  "
            f"{left_label} <-> {right_label}"
        )
        print(
            f"  below threshold: {len(below)}/{len(self._arm_clearance_log)} steps  "
            f"penetrating: {len(penetrating)} steps"
        )

    def _sensor(self, name: str) -> np.ndarray:
        return np.array(self.data.sensordata[self.sensor_slices[name]], dtype=np.float64)

    def _set_reference_frame(self, ref0: dict[str, np.ndarray]) -> None:
        if not self.zero_xy_ref:
            self._reference_xy_offset[:] = 0.0
            self._reference_yaw = 0.0
            self._reference_yaw_inv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return
        self._reference_xy_offset = np.array(ref0["body_pos_w"][0, :2], dtype=np.float64)
        yaw0 = yaw_from_quat(ref0["body_quat_w"][0])
        self._reference_yaw = -yaw0
        self._reference_yaw_inv = quat_from_yaw(-yaw0)

    def _set_reference_frame_from_robot(self, ref0: "dict[str, np.ndarray] | None" = None) -> None:
        """Align the reference coordinate frame to the robot's current anchor pose."""
        if not self.zero_xy_ref:
            self._reference_xy_offset[:] = 0.0
            self._reference_yaw = 0.0
            self._reference_yaw_inv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return
        robot_pos, robot_quat = self._robot_anchor_pose()
        robot_yaw = yaw_from_quat(robot_quat)
        if ref0 is not None:
            ref_xy = np.asarray(ref0["body_pos_w"][0, :2], dtype=np.float64)
            self._reference_xy_offset = ref_xy.copy()
            ref_yaw = yaw_from_quat(ref0["body_quat_w"][0])
            delta_yaw = robot_yaw - ref_yaw
        else:
            self._reference_xy_offset = np.array(robot_pos[:2], dtype=np.float64)
            delta_yaw = robot_yaw
        self._reference_yaw = delta_yaw
        self._reference_yaw_inv = quat_from_yaw(delta_yaw)

    def _normalize_reference_output(self, ref: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized = {
            "joint_pos": np.asarray(ref["joint_pos"], dtype=np.float32).copy(),
            "joint_vel": np.asarray(ref["joint_vel"], dtype=np.float32).copy(),
            "body_pos_w": np.asarray(ref["body_pos_w"], dtype=np.float64).copy(),
            "body_quat_w": np.asarray(ref["body_quat_w"], dtype=np.float64).copy(),
            "body_lin_vel_w": np.asarray(ref["body_lin_vel_w"], dtype=np.float64).copy(),
            "body_ang_vel_w": np.asarray(ref["body_ang_vel_w"], dtype=np.float64).copy(),
        }
        if not self.zero_xy_ref:
            return normalized

        normalized["body_pos_w"][:, :2] -= self._reference_xy_offset
        normalized["body_pos_w"] = rotate_world_z(normalized["body_pos_w"], self._reference_yaw)
        normalized["body_quat_w"] = np.stack(
            [quat_multiply(self._reference_yaw_inv, quat) for quat in normalized["body_quat_w"]],
            axis=0,
        )
        normalized["body_lin_vel_w"] = rotate_world_z(normalized["body_lin_vel_w"], self._reference_yaw)
        normalized["body_ang_vel_w"] = rotate_world_z(normalized["body_ang_vel_w"], self._reference_yaw)
        return normalized

    def _robot_anchor_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._anchor_body_id is not None:
            return (
                np.array(self.data.xpos[self._anchor_body_id], dtype=np.float64),
                quat_normalize(np.array(self.data.xquat[self._anchor_body_id], dtype=np.float64)),
            )
        return self._sensor("position"), quat_normalize(self._sensor("orientation"))

    def _aligned_reference_bodies(self, ref_now: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        robot_anchor_pos_w, robot_anchor_quat_w = self._robot_anchor_pose()
        anchor_pos_w = np.asarray(ref_now["body_pos_w"][0], dtype=np.float64)
        anchor_quat_w = quat_normalize(np.asarray(ref_now["body_quat_w"][0], dtype=np.float64))

        delta_pos_w = robot_anchor_pos_w.copy()
        delta_pos_w[2] = anchor_pos_w[2]
        delta_quat_w = quat_from_yaw(yaw_from_quat(quat_multiply(robot_anchor_quat_w, quat_conjugate(anchor_quat_w))))

        body_pos_w = np.asarray(ref_now["body_pos_w"], dtype=np.float64)
        body_quat_w = np.asarray(ref_now["body_quat_w"], dtype=np.float64)
        aligned_body_pos_w = delta_pos_w + quat_rotate(delta_quat_w, body_pos_w - anchor_pos_w)
        aligned_body_quat_w = np.stack([quat_multiply(delta_quat_w, quat) for quat in body_quat_w], axis=0)
        return aligned_body_pos_w, aligned_body_quat_w

    def _build_reference_outputs(self) -> list[dict[str, np.ndarray]]:
        reference_outputs: list[dict[str, np.ndarray]] = []
        for step_index in range(self.metadata.total_steps):
            ref = self.policy.infer(self.zero_obs, step_index)
            reference_outputs.append(
                {
                    "joint_pos": np.asarray(ref["joint_pos"], dtype=np.float32).copy(),
                    "joint_vel": np.asarray(ref["joint_vel"], dtype=np.float32).copy(),
                    "body_pos_w": np.asarray(ref["body_pos_w"], dtype=np.float64).copy(),
                    "body_quat_w": np.asarray(ref["body_quat_w"], dtype=np.float64).copy(),
                    "body_lin_vel_w": np.asarray(ref["body_lin_vel_w"], dtype=np.float64).copy(),
                    "body_ang_vel_w": np.asarray(ref["body_ang_vel_w"], dtype=np.float64).copy(),
                }
            )

        self._set_reference_frame(reference_outputs[0])
        return [self._normalize_reference_output(ref) for ref in reference_outputs]

    def _build_default_reference(self) -> dict[str, np.ndarray]:
        if self.reference_outputs:
            base_height = float(self.reference_outputs[0]["body_pos_w"][0, 2])
        else:
            base_height = 1.0
        temp_data = mujoco.MjData(self.model)
        temp_data.qpos[:3] = np.array([0.0, 0.0, base_height], dtype=np.float64)
        temp_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        temp_data.qvel[:] = 0.0
        temp_data.qpos[self.joint_qpos_indices] = np.asarray(self.metadata.default_joint_pos, dtype=np.float64)
        temp_data.qvel[self.joint_qvel_indices] = 0.0
        mujoco.mj_forward(self.model, temp_data)

        # Adjust base_height so the lowest foot geom sits exactly on the ground.
        foot_z_min = float("inf")
        for gid in self._foot_geom_ids:
            z = float(temp_data.geom_xpos[gid, 2])
            if z < foot_z_min:
                foot_z_min = z
        if foot_z_min != float("inf") and foot_z_min < 1.0:
            temp_data.qpos[2] -= foot_z_min
            mujoco.mj_forward(self.model, temp_data)

        body_pos_w = []
        body_quat_w = []
        for body_name in self.metadata.body_names:
            body_id = self._resolve_body_id(body_name)
            if body_id < 0:
                raise RuntimeError(f"Body `{body_name}` not found in Mujoco model for default reference.")
            body_pos_w.append(np.array(temp_data.xpos[body_id], dtype=np.float64))
            body_quat_w.append(quat_normalize(np.array(temp_data.xquat[body_id], dtype=np.float64)))

        body_pos_w_arr = np.stack(body_pos_w, axis=0)
        body_quat_w_arr = np.stack(body_quat_w, axis=0)
        return {
            "joint_pos": np.asarray(self.metadata.default_joint_pos, dtype=np.float32).copy(),
            "joint_vel": np.zeros_like(self.metadata.default_joint_pos, dtype=np.float32),
            "body_pos_w": body_pos_w_arr,
            "body_quat_w": body_quat_w_arr,
            "body_lin_vel_w": np.zeros_like(body_pos_w_arr, dtype=np.float64),
            "body_ang_vel_w": np.zeros_like(body_pos_w_arr, dtype=np.float64),
        }

    def _transition_steps(self, duration_s: float) -> int:
        return max(1, int(round(duration_s / self.control_dt)))

    def _build_current_robot_frame(self) -> dict[str, np.ndarray]:
        """Build a reference frame dict from the current MuJoCo simulation state."""
        q_cur = self.data.qpos[self.joint_qpos_indices].astype(np.float32, copy=True)
        qd_cur = self.data.qvel[self.joint_qvel_indices].astype(np.float32, copy=True)
        body_pos_w = []
        body_quat_w = []
        for body_name in self.metadata.body_names:
            body_id = self._resolve_body_id(body_name)
            if body_id < 0:
                raise RuntimeError(f"Body `{body_name}` not found in Mujoco model.")
            body_pos_w.append(np.array(self.data.xpos[body_id], dtype=np.float64))
            body_quat_w.append(quat_normalize(np.array(self.data.xquat[body_id], dtype=np.float64)))
        body_pos_w_arr = np.stack(body_pos_w, axis=0)
        body_quat_w_arr = np.stack(body_quat_w, axis=0)
        return {
            "joint_pos": q_cur,
            "joint_vel": qd_cur,
            "body_pos_w": body_pos_w_arr,
            "body_quat_w": body_quat_w_arr,
            "body_lin_vel_w": np.zeros_like(body_pos_w_arr, dtype=np.float64),
            "body_ang_vel_w": np.zeros_like(body_pos_w_arr, dtype=np.float64),
        }

    @staticmethod
    def _slerp_wxyz(qa: np.ndarray, qb: np.ndarray, alpha: float) -> np.ndarray:
        qa = quat_normalize(qa)
        qb = quat_normalize(qb)
        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb = -qb
            dot = -dot
        dot = min(dot, 1.0)
        if dot > 0.9995:
            return quat_normalize(qa + alpha * (qb - qa))
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        sin_theta_0 = np.sin(theta_0)
        s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        return quat_normalize(s0 * qa + s1 * qb)

    def _interpolate_reference_frames(
        self,
        a: dict[str, np.ndarray],
        b: dict[str, np.ndarray],
        alpha: float,
    ) -> dict[str, np.ndarray]:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        qa = np.asarray(a["body_quat_w"], dtype=np.float64)
        qb = np.asarray(b["body_quat_w"], dtype=np.float64)
        body_quat_w = np.stack(
            [self._slerp_wxyz(qa[i], qb[i], alpha) for i in range(qa.shape[0])], axis=0
        )
        def lerp(key, dtype):
            return (np.asarray(a[key], dtype=dtype) + alpha *
                    (np.asarray(b[key], dtype=dtype) - np.asarray(a[key], dtype=dtype)))
        return {
            "joint_pos": lerp("joint_pos", np.float32),
            "joint_vel": lerp("joint_vel", np.float32),
            "body_pos_w": lerp("body_pos_w", np.float64),
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": lerp("body_lin_vel_w", np.float64),
            "body_ang_vel_w": lerp("body_ang_vel_w", np.float64),
        }

    def _current_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos_indices].astype(np.float32, copy=True)

    def _apply_joint_target(self, q_target: np.ndarray) -> None:
        q_target = np.asarray(q_target, dtype=np.float32)
        if self.max_target_step is not None:
            q_curr = self._current_joint_positions()
            q_target = np.clip(q_target, q_curr - self.max_target_step, q_curr + self.max_target_step)
        self.data.ctrl[self.actuator_index_array] = q_target.astype(np.float64, copy=False)

    def _step_physics(self) -> None:
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self._record_slip(self.current_step)
        self._check_arm_clearance(self.current_step)

    def _run_policy_step(
        self,
        ref_now: dict[str, np.ndarray],
        ref_lookahead: dict[str, np.ndarray],
        step_index: int,
        blend_alpha: float = 1.0,
        obs_ref: "dict[str, np.ndarray] | None" = None,
        obs_ref_lookahead: "dict[str, np.ndarray] | None" = None,
        override_phase: "float | None" = None,
        phase_mode_idx: int = 1,
    ) -> np.ndarray:
        obs = self.build_observation(step_index, ref_now, ref_lookahead,
                                     obs_ref=obs_ref, obs_ref_lookahead=obs_ref_lookahead,
                                     override_phase=override_phase,
                                     phase_mode_idx=phase_mode_idx)
        policy_out = self.policy.infer(obs, step_index)
        raw_action = np.asarray(policy_out["actions"], dtype=np.float32)
        if self.action_replay is not None:
            replay_idx = min(max(int(step_index) + self.action_replay_offset, 0), self.action_replay.shape[0] - 1)
            raw_action = np.asarray(self.action_replay[replay_idx], dtype=np.float32)
        if self.zero_action:
            raw_action = np.zeros_like(raw_action)
        q_ref = np.asarray(ref_now["joint_pos"], dtype=np.float32)
        blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))

        q_target = q_ref + blend_alpha * raw_action * self.metadata.action_scale
        self._apply_joint_target(q_target)
        self._step_physics()
        self.last_action = raw_action * blend_alpha
        if self.debug_dynamics:
            q_cur = self.data.qpos[self.joint_qpos_indices].astype(np.float32, copy=False)
            qd_cur = self.data.qvel[self.joint_qvel_indices].astype(np.float32, copy=False)
            qd_ref = np.asarray(ref_now["joint_vel"], dtype=np.float32)
            forces = np.asarray(self.data.actuator_force[self.actuator_index_array], dtype=np.float64)
            limits = np.asarray(self.metadata.joint_effort_limit, dtype=np.float64) * self.effort_scale
            force_ratio = np.abs(forces) / np.maximum(limits, 1e-6)
            top_force_idx = int(np.argmax(force_ratio))
            q_err = np.abs(q_cur - q_ref)
            top_qerr_idx = int(np.argmax(q_err))
            force_l, force_r = self._foot_contact_forces()
            self._last_dynamics = {
                "max_abs_action": float(np.max(np.abs(raw_action))),
                "max_abs_q_err": float(np.max(q_err)),
                "max_abs_qd_err": float(np.max(np.abs(qd_cur - qd_ref))),
                "max_force_ratio": float(np.max(force_ratio)),
                "mean_force_ratio": float(np.mean(force_ratio)),
                "foot_force_l": float(force_l),
                "foot_force_r": float(force_r),
                "top_force_joint": self.metadata.joint_names[top_force_idx],
                "top_qerr_joint": self.metadata.joint_names[top_qerr_idx],
            }
            if self._last_arm_clearance is not None:
                self._last_dynamics["arm_clearance"] = float(self._last_arm_clearance["dist"])
        return raw_action

    def request_next_motion(self) -> None:
        if self.motion_library is None:
            return
        with self._request_lock:
            self._pending_next_motion_requests += 1

    def _consume_next_motion_request(self) -> bool:
        if self.motion_library is None:
            return False
        with self._request_lock:
            if self._pending_next_motion_requests <= 0:
                return False
            self._pending_next_motion_requests = 0
            return True

    def _start_pre_stand(self) -> None:
        if self.motion_library is None:
            return
        next_index = (
            0 if self._active_motion_index is None
            else (self._active_motion_index + 1) % self.motion_library.motion_count
        )
        self._next_motion_index = next_index
        self._stand_step_count = 0
        self._playback_phase = "pre_stand"
        print(
            f"[playlist] pre-stand ({self.stand_duration_s:.1f}s) before motion "
            f"{next_index + 1}/{self.motion_library.motion_count}: "
            f"{self.motion_library.motion_names[next_index]}"
        )

    def _start_play_motion(self) -> None:
        next_index = self._next_motion_index
        self.teleop_stream = self.motion_library.get_stream(next_index)
        self.teleop_stream.reset()
        ref0_raw = self.teleop_stream.reference_at(0)
        self._set_reference_frame_from_robot(ref0=ref0_raw)
        self._active_motion_index = next_index
        self._active_motion_name = self.motion_library.motion_names[next_index]
        self._playback_phase = "play_motion"
        self.current_step = 0
        self.last_action[:] = 0.0
        print(f"[playlist] playing motion: {self._active_motion_name}")

    def _start_return_stand(self) -> None:
        self._stand_step_count = 0
        self.last_action[:] = 0.0
        self._return_stand_start_ref = self._build_current_robot_frame()
        self._playback_phase = "return_stand"
        print(f"[playlist] return stand ({self.stand_duration_s:.1f}s)")

    def _yaw_aligned_default_ref(self) -> dict[str, np.ndarray]:
        _, robot_quat = self._robot_anchor_pose()
        yaw_quat = quat_from_yaw(yaw_from_quat(robot_quat))
        ref = {k: v.copy() for k, v in self.default_reference.items()}
        ref["body_quat_w"] = np.stack(
            [quat_multiply(yaw_quat, q) for q in self.default_reference["body_quat_w"]],
            axis=0,
        )
        return ref

    def _step_motion_playlist(self) -> bool:
        if self._consume_next_motion_request():
            self._start_pre_stand()

        if self._playback_phase == "default_pose":
            ref = self._yaw_aligned_default_ref()
            self._run_policy_step(ref, ref, 0, phase_mode_idx=0)
            return False

        if self._playback_phase == "pre_stand":
            ref = self._yaw_aligned_default_ref()
            self._run_policy_step(ref, ref, 0, phase_mode_idx=0)
            self._stand_step_count += 1
            if self._stand_step_count >= self._stand_total_steps:
                self._start_play_motion()
            return False

        if self._playback_phase == "play_motion":
            if self.teleop_stream is None:
                self._playback_phase = "default_pose"
                return False
            ref_now = self._reference_output(self.current_step)
            lookahead_step = min(self.current_step + LOOKAHEAD_STEPS, self._total_steps - 1)
            ref_lookahead = self._reference_output(lookahead_step)
            self._run_policy_step(ref_now, ref_lookahead, self.current_step, phase_mode_idx=1)
            if self.current_step >= self._total_steps - 1:
                print(f"[playlist] finished motion: {self._active_motion_name}")
                self._start_return_stand()
            else:
                self.current_step += 1
            return False

        if self._playback_phase == "return_stand":
            target_ref = self._yaw_aligned_default_ref()
            start_ref = self._return_stand_start_ref
            if start_ref is None:
                start_ref = self._build_current_robot_frame()
                self._return_stand_start_ref = start_ref
            alpha = smoothstep((self._stand_step_count + 1) / max(self._stand_total_steps, 1))
            ref = self._interpolate_reference_frames(start_ref, target_ref, alpha)
            self._run_policy_step(ref, ref, 0, phase_mode_idx=0)
            self._stand_step_count += 1
            if self._stand_step_count >= self._stand_total_steps:
                self._playback_phase = "default_pose"
                self._return_stand_start_ref = None
                print("[playlist] return stand complete, back to default_pose")
            return False

        return False

    @property
    def _total_steps(self) -> int:
        if self.teleop_stream is not None:
            return self.teleop_stream.total_steps
        if self.motion_library is not None and self._active_motion_index is not None:
            return self.motion_library.get_stream(self._active_motion_index).total_steps
        return self.metadata.total_steps

    def calibrate_standing_bias(self, num_steps: int = 75) -> None:
        """Run a short standing phase to measure the policy's systematic action bias.

        The averaged raw_action is stored as ``_standing_action_bias`` and
        subtracted during subsequent standing phases (phase_mode_idx==0),
        cancelling drift while preserving balance-correction authority.
        """
        print(f"[calibrate] measuring standing action bias over {num_steps} steps …")
        actions = []
        for _ in range(num_steps):
            ref = self._yaw_aligned_default_ref()
            obs = self.build_observation(0, ref, ref, phase_mode_idx=0)
            policy_out = self.policy.infer(obs, 0)
            raw_action = np.asarray(policy_out["actions"], dtype=np.float32)
            actions.append(raw_action)
            # Apply action through PD to keep physics stable during calibration
            q_ref = np.asarray(ref["joint_pos"], dtype=np.float32)
            q_target = q_ref + raw_action * self.metadata.action_scale
            self._apply_joint_target(q_target)
            self._step_physics()
        self._standing_action_bias = np.mean(np.stack(actions, axis=0), axis=0).astype(np.float32)
        max_bias = float(np.max(np.abs(self._standing_action_bias * self.metadata.action_scale)))
        print(f"[calibrate] done — max residual bias: {max_bias:.4f} rad")

    def _reference_output(self, step_index: int) -> dict[str, np.ndarray]:
        if self.teleop_stream is not None:
            return self._normalize_reference_output(self.teleop_stream.reference_at(step_index))
        return self.reference_outputs[step_index]

    def reset(self) -> None:
        self.current_step = 0
        self.last_action[:] = 0.0
        self._slip_log = []
        self._arm_clearance_log = []
        self._last_arm_clearance = None
        self._last_arm_clearance_warning_step = -10_000
        self._last_dynamics = {}
        if self.teleop_stream is not None:
            self.teleop_stream.reset()
        if self.motion_library is not None:
            with self._request_lock:
                self._pending_next_motion_requests = 0
            self._playback_phase = "default_pose"
            self._stand_step_count = 0
            self._return_stand_start_ref = None
            self._next_motion_index = 0
            self._reference_xy_offset = np.zeros(2, dtype=np.float64)
            self._reference_yaw = 0.0
            self._reference_yaw_inv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mujoco.mj_resetData(self.model, self.data)

        if self.motion_library is not None:
            ref0 = self.default_reference
            if ref0 is None:
                raise RuntimeError("Motion playlist default reference was not initialized.")
        else:
            ref0 = self.reference_outputs[0] if self.teleop_stream is None else self._reference_output(0)
        base_pos = np.array(ref0["body_pos_w"][0], dtype=np.float64)
        base_quat = quat_normalize(np.array(ref0["body_quat_w"][0], dtype=np.float64))

        self.data.qpos[:3] = base_pos
        self.data.qpos[3:7] = base_quat
        self.data.qvel[:3] = np.array(ref0["body_lin_vel_w"][0], dtype=np.float64)
        self.data.qvel[3:6] = np.array(ref0["body_ang_vel_w"][0], dtype=np.float64)

        joint_pos = np.array(ref0["joint_pos"], dtype=np.float64)
        joint_vel = np.array(ref0["joint_vel"], dtype=np.float64)
        self.data.qpos[self.joint_qpos_indices] = joint_pos
        self.data.qvel[self.joint_qvel_indices] = joint_vel
        self.data.ctrl[self.actuator_index_array] = joint_pos

        mujoco.mj_forward(self.model, self.data)

    def rewind_motion(self) -> None:
        if self.motion_library is not None:
            self.reset()
            return
        self.current_step = 0
        self.last_action[:] = 0.0
        self._slip_log = []
        self._arm_clearance_log = []
        self._last_arm_clearance = None
        self._last_arm_clearance_warning_step = -10_000
        self._last_dynamics = {}
        if self.teleop_stream is not None:
            self.teleop_stream.reset()

    def build_observation(
        self,
        step_index: int,
        ref_now: dict[str, np.ndarray],
        ref_lookahead: dict[str, np.ndarray],
        obs_ref: "dict[str, np.ndarray] | None" = None,
        obs_ref_lookahead: "dict[str, np.ndarray] | None" = None,
        override_phase: "float | None" = None,
        phase_mode_idx: int = 1,
    ) -> np.ndarray:
        base_quat_w = quat_normalize(self._sensor("orientation"))
        base_lin_vel_b = self._sensor("linear-velocity")
        base_ang_vel_b = self._sensor("angular-velocity")

        q_cur = self.data.qpos[self.joint_qpos_indices].astype(np.float32, copy=True)
        qd_cur = self.data.qvel[self.joint_qvel_indices].astype(np.float32, copy=True)

        _obs_now = obs_ref if obs_ref is not None else ref_now
        _obs_la  = obs_ref_lookahead if obs_ref_lookahead is not None else ref_lookahead

        q_ref = np.asarray(_obs_now["joint_pos"], dtype=np.float32)
        qd_ref = np.asarray(_obs_now["joint_vel"], dtype=np.float32)
        qd_ref_lookahead = np.asarray(_obs_la["joint_vel"], dtype=np.float32)
        q_ref_lookahead = np.asarray(_obs_la["joint_pos"], dtype=np.float32)

        anchor_quat_ref_w = quat_normalize(np.asarray(_obs_now["body_quat_w"][0], dtype=np.float64))
        anchor_quat_rel = quat_multiply(quat_conjugate(base_quat_w), anchor_quat_ref_w)
        anchor_rot6d = rotmat_to_rot6d(quat_to_rotmat(anchor_quat_rel))

        total_steps = max(1, self._total_steps - 1)
        if override_phase is not None:
            phase_angle = 2.0 * np.pi * float(override_phase)
        else:
            phase = float(np.clip(step_index, 0, total_steps)) / float(total_steps)
            phase_angle = 2.0 * np.pi * phase

        obs = np.concatenate(
            [
                q_ref,
                qd_ref,
                anchor_rot6d,
                base_ang_vel_b.astype(np.float32),
                (q_cur - q_ref).astype(np.float32),
                (qd_cur - qd_ref).astype(np.float32),
                self.last_action.astype(np.float32),
                np.array([np.sin(phase_angle), np.cos(phase_angle)], dtype=np.float32),
                qd_ref_lookahead,
                q_ref_lookahead,
            ],
            axis=0,
        ).astype(np.float32)

        raw_dim = obs.shape[0]
        if self.metadata.obs_dim == raw_dim + 2:
            phase_one_hot = np.zeros(2, dtype=np.float32)
            phase_one_hot[int(np.clip(phase_mode_idx, 0, 1))] = 1.0
            obs = np.concatenate([obs, phase_one_hot], axis=0)

        if obs.shape[0] != self.metadata.obs_dim:
            raise RuntimeError(f"Observation dim mismatch: expected {self.metadata.obs_dim}, got {obs.shape[0]}")
        return obs

    def step(self) -> bool:
        if self.motion_library is not None:
            return self._step_motion_playlist()

        ref_now = self._reference_output(self.current_step)
        lookahead_step = min(self.current_step + LOOKAHEAD_STEPS, self._total_steps - 1)
        ref_lookahead = self._reference_output(lookahead_step)
        self._run_policy_step(ref_now, ref_lookahead, self.current_step)

        motion_finished = self.current_step >= self._total_steps - 1
        if not motion_finished:
            self.current_step += 1
        return motion_finished


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Mujoco sim2sim runner for Dex EVT ONNX policy.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="Path to exported policy.onnx")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="Path to Mujoco MJCF/XML robot model")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Policy control period in seconds")
    parser.add_argument("--sim-dt", type=float, default=0.005, help="Mujoco physics timestep in seconds")
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="Viewer playback rate multiplier: 2.0 = 2x faster, 0.1 = 10x slower",
    )
    parser.add_argument("--steps", type=int, default=None, help="Max control steps to run; default is full motion length")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--loop", action="store_true", help="Loop motion playback")
    parser.add_argument(
        "--max-target-step",
        type=float,
        default=None,
        help="Optional per-control-step joint target clipping in radians; default disables clipping",
    )
    parser.add_argument(
        "--keep-reference-world",
        action="store_true",
        help="Do not shift reference XY to the world origin before playback",
    )
    parser.add_argument("--print-every", type=int, default=50, help="Print status every N control steps")
    parser.add_argument("--log-slip", action="store_true", help="Print per-step foot slip log and summary")
    parser.add_argument("--slip-threshold", type=float, default=0.05, help="Foot XY velocity threshold (m/s) to flag as slip")
    parser.add_argument("--zero-action", action="store_true", help="Debug: ignore policy residual and track reference joints only")
    parser.add_argument("--debug-dynamics", action="store_true", help="Print action, tracking, force, and contact diagnostics")
    parser.add_argument("--effort-scale", type=float, default=1.0, help="Debug: multiply actuator force limits")
    parser.add_argument("--action-replay-npz", type=Path, default=None, help="Debug: replay `action` array from an Isaac rollout NPZ")
    parser.add_argument("--action-replay-offset", type=int, default=0, help="Debug: step offset when replaying actions")
    parser.add_argument(
        "--arm-clearance-threshold",
        type=float,
        default=ARM_CLEARANCE_DEFAULT_M,
        help="Warn when left/right arm-hand geom distance is below this value in meters; set <0 to disable",
    )
    parser.add_argument(
        "--stop-on-arm-clearance",
        action="store_true",
        help="Raise an error and stop when arm clearance falls below --arm-clearance-threshold",
    )
    parser.add_argument("--npz", type=Path, default=None,
        help="NPZ motion file for teleop simulation.")
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=None,
        help="Directory of NPZ motions. Starts at default_joint_pos and switches to the next motion on viewer key B.",
    )
    parser.add_argument(
        "--stand-seconds",
        type=float,
        default=1.0,
        help="Duration of STAND phase before and after each motion clip (default: 1.0s).",
    )
    return parser


def run_headless(sim: MujocoSim2Sim, max_steps: int | None, loop: bool, print_every: int) -> None:
    step_count = 0
    while True:
        try:
            motion_finished = sim.step()
        except ArmClearanceViolation as exc:
            print(f"[STOP] {exc}")
            break
        step_count += 1
        reached_step_limit = max_steps is not None and step_count >= max_steps
        if print_every > 0 and (step_count == 1 or step_count % print_every == 0 or reached_step_limit):
            base_pos = sim.data.qpos[:3].copy()
            extra = format_dynamics_extra(sim)
            print(f"step={step_count:04d} motion_step={sim.current_step:04d} base_pos={base_pos}{extra}")

        if motion_finished:
            if loop:
                sim.rewind_motion()
            elif max_steps is None or step_count <= max_steps:
                break

        if reached_step_limit:
            break

    sim.print_slip_summary()
    sim.print_arm_clearance_summary()


def run_with_viewer(sim: MujocoSim2Sim, max_steps: int | None, loop: bool, print_every: int, playback_rate: float) -> None:
    if mujoco_viewer is None:
        raise RuntimeError(
            "Mujoco viewer is unavailable in this environment. "
            "Install GUI support or run with --headless."
        )
    key_codes = {VIEWER_KEY_B, ord("b"), ord("B")}
    if hasattr(mujoco_viewer, "glfw") and hasattr(mujoco_viewer.glfw, "KEY_B"):
        key_codes.add(int(mujoco_viewer.glfw.KEY_B))

    def on_key(keycode: int) -> None:
        if keycode in key_codes:
            sim.request_next_motion()

    if sim.motion_library is not None:
        print("[viewer] Press B in the MuJoCo window to switch to the next motion.")

    with mujoco_viewer.launch_passive(sim.model, sim.data, key_callback=on_key,show_left_ui=False,show_right_ui=False) as viewer:
        # Pull camera back for a wider view of the full robot
        viewer.cam.distance = 4.0
        viewer.cam.lookat = [0.0, 0.0, 0.9]
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0

        step_count = 0
        while viewer.is_running():
            step_start = time.time()
            try:
                motion_finished = sim.step()
            except ArmClearanceViolation as exc:
                print(f"[STOP] {exc}")
                break
            viewer.sync()
            step_count += 1
            reached_step_limit = max_steps is not None and step_count >= max_steps

            if print_every > 0 and (step_count == 1 or step_count % print_every == 0 or reached_step_limit):
                base_pos = sim.data.qpos[:3].copy()
                extra = format_dynamics_extra(sim)
                print(f"step={step_count:04d} motion_step={sim.current_step:04d} base_pos={base_pos}{extra}")

            elapsed = time.time() - step_start
            sleep_time = sim.control_dt / playback_rate - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

            if motion_finished:
                if loop:
                    sim.rewind_motion()
                elif max_steps is None or step_count <= max_steps:
                    break

            if reached_step_limit:
                break

    sim.print_slip_summary()
    sim.print_arm_clearance_summary()


def main() -> None:
    args = build_argparser().parse_args()
    if args.playback_rate <= 0.0:
        raise ValueError(f"`--playback-rate` must be positive, got {args.playback_rate}")

    policy_path = args.policy.expanduser().resolve()
    mjcf_path = args.mjcf.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

    policy = OnnxMotionPolicy(policy_path)
    policy_raw_obs_dim = policy.metadata.obs_dim - 2 if policy.metadata.obs_dim == 216 else policy.metadata.obs_dim
    policy_mode = "merged_head" if policy.metadata.obs_dim == 216 else "single_head"

    teleop_stream = None
    motion_library = None
    action_replay = None
    if args.action_replay_npz is not None:
        replay_path = args.action_replay_npz.expanduser().resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"Action replay NPZ not found: {replay_path}")
        replay_data = np.load(replay_path, allow_pickle=False)
        if "action" not in replay_data.files:
            raise RuntimeError(f"Action replay NPZ has no `action` array: {replay_path}")
        action_replay = np.asarray(replay_data["action"], dtype=np.float32)
        if action_replay.ndim != 2 or action_replay.shape[1] != policy.metadata.action_dim:
            raise RuntimeError(
                f"Action replay shape mismatch: expected (*, {policy.metadata.action_dim}), got {action_replay.shape}"
            )
        print(f"[replay] actions: {replay_path} shape={action_replay.shape}")
    if args.npz is not None and args.npz_dir is not None:
        raise ValueError("`--npz` and `--npz-dir` are mutually exclusive.")
    if args.npz is not None:
        npz_path = args.npz.expanduser().resolve()
        if not npz_path.is_file():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        teleop_stream = NpzTeleopStream(
            npz_path=str(npz_path),
            policy_joint_names=policy.metadata.joint_names,
            policy_body_names=policy.metadata.body_names,
        )
        print(f"[teleop] NPZ stream: {npz_path}")
    elif args.npz_dir is not None:
        npz_dir = args.npz_dir.expanduser().resolve()
        if not npz_dir.is_dir():
            raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")
        motion_library = MotionLibrary.from_directory(
            npz_dir=npz_dir,
            policy_joint_names=policy.metadata.joint_names,
            policy_body_names=policy.metadata.body_names,
        )
        print(f"[playlist] Loaded {motion_library.motion_count} motions from {npz_dir}")
        for motion_index, motion_name in enumerate(motion_library.motion_names):
            print(f"  [{motion_index:02d}] {motion_name}")
        print("[playlist] Idle target uses policy metadata default_joint_pos.")

    sim = MujocoSim2Sim(
        mjcf_path=mjcf_path,
        policy=policy,
        control_dt=args.control_dt,
        sim_dt=args.sim_dt,
        zero_xy_ref=not args.keep_reference_world,
        max_target_step=args.max_target_step,
        log_slip=args.log_slip,
        slip_vel_threshold=args.slip_threshold,
        teleop_stream=teleop_stream,
        motion_library=motion_library,
        stand_duration_s=args.stand_seconds,
        zero_action=args.zero_action,
        debug_dynamics=args.debug_dynamics,
        effort_scale=args.effort_scale,
        action_replay=action_replay,
        action_replay_offset=args.action_replay_offset,
        arm_clearance_threshold=args.arm_clearance_threshold,
        stop_on_arm_clearance=args.stop_on_arm_clearance,
    )
    max_steps = args.steps
    if max_steps is None and not args.loop and motion_library is None:
        max_steps = teleop_stream.total_steps if teleop_stream is not None else policy.metadata.total_steps

    print(f"policy={policy_path}")
    print(f"policy_mode={policy_mode} raw_obs_dim={policy_raw_obs_dim}")
    print(f"mjcf={mjcf_path}")
    print(f"initial_pose=from_reference_frame_0")
    print(f"obs_dim={policy.metadata.obs_dim} action_dim={policy.metadata.action_dim} total_steps={policy.metadata.total_steps}")
    print(f"control_dt={args.control_dt:.4f} sim_dt={args.sim_dt:.4f} decimation={sim.decimation}")
    print(f"max_steps={'inf' if max_steps is None else max_steps}")
    print(f"playback_rate={args.playback_rate:.3f}x")
    print(f"zero_action={args.zero_action}")
    print(f"debug_dynamics={args.debug_dynamics}")
    print(f"effort_scale={args.effort_scale:.3f}")
    print(f"action_replay={'none' if action_replay is None else action_replay.shape}")
    print(f"action_replay_offset={args.action_replay_offset}")
    print(f"arm_clearance_threshold={args.arm_clearance_threshold:.4f}m")
    print(f"stop_on_arm_clearance={args.stop_on_arm_clearance}")

    if args.headless:
        run_headless(sim, max_steps=max_steps, loop=args.loop, print_every=args.print_every)
    else:
        run_with_viewer(
            sim,
            max_steps=max_steps,
            loop=args.loop,
            print_every=args.print_every,
            playback_rate=args.playback_rate,
        )


if __name__ == "__main__":
    main()
