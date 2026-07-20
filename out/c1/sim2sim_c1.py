"""MuJoCo runner for Walker C1 ONNX policies and NPZ motion playlists."""


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
    "logs/rsl_rl/walker_c1_fix/exported/policy.onnx"
)
DEFAULT_MJCF = Path("source/whole_body_tracking/whole_body_tracking/assets/walker_c1/mjcf/walker_astron.xml")

LOOKAHEAD_STEPS = 2
VIEWER_KEY_B = 66
FOOT_GEOM_NAMES = ("L_ankle_roll_collision", "R_ankle_roll_collision")


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


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    return np.array(
        [rotmat[0, 0], rotmat[0, 1], rotmat[1, 0], rotmat[1, 1], rotmat[2, 0], rotmat[2, 1]],
        dtype=np.float32,
    )


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
            total_steps=infer_total_steps(model),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )


@dataclass(frozen=True)
class MotionClipInfo:
    name: str
    path: Path


class OnnxMotionPolicy:
    def __init__(self, policy_path: Path):
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


class NpzMotionStream:
    """Load NPZ references and expose them in policy joint/body order."""

    def __init__(
        self,
        npz_path: str,
        policy_joint_names: list[str],
        policy_body_names: list[str],
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

        if "body_names" not in data.files:
            raise RuntimeError(
                f"NPZ file '{npz_path}' is missing 'body_names'. "
                "Please regenerate it with the latest scripts/pkl_to_npz.py."
            )
        npz_body_names = [str(n) for n in data["body_names"].tolist()]

        # Policy body indices (for obs body_pos_w / body_quat_w slicing)
        self._body_order = [npz_body_names.index(n) for n in policy_body_names]

        self._cache_next_step = 0
        self._frame_cache: dict[int, dict[str, np.ndarray]] = {}
        print(f"[motion] Loaded {npz_path}  T={self.T}  fps={self.fps}")

    @property
    def total_steps(self) -> int:
        return self.T

    def reset(self) -> None:
        self._cache_next_step = 0
        self._frame_cache.clear()

    def _compute_frame(self, step_index: int) -> dict[str, np.ndarray]:
        t = min(step_index, self.T - 1)

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
        }

    def reference_at(self, step_index: int) -> dict[str, np.ndarray]:
        t = min(max(step_index, 0), self.T - 1)
        while self._cache_next_step <= t:
            self._frame_cache[self._cache_next_step] = self._compute_frame(self._cache_next_step)
            self._cache_next_step += 1
        return {key: np.array(value, copy=True) for key, value in self._frame_cache[t].items()}

class MotionLibrary:
    def __init__(
        self,
        motions: list[MotionClipInfo],
        policy_joint_names: list[str],
        policy_body_names: list[str],
    ):
        if not motions:
            raise RuntimeError("No NPZ motions available.")
        self.motions = motions
        self.policy_joint_names = policy_joint_names
        self.policy_body_names = policy_body_names
        self._stream_cache: dict[int, NpzMotionStream] = {}

    @classmethod
    def from_directory(
        cls,
        npz_dir: Path,
        policy_joint_names: list[str],
        policy_body_names: list[str],
    ) -> "MotionLibrary":
        motion_paths = sorted(path for path in npz_dir.glob("*.npz") if path.is_file())
        if not motion_paths:
            raise RuntimeError(f"No NPZ motions found in directory: {npz_dir}")
        return cls(
            motions=[MotionClipInfo(name=path.stem, path=path) for path in motion_paths],
            policy_joint_names=policy_joint_names,
            policy_body_names=policy_body_names,
        )

    @property
    def motion_names(self) -> list[str]:
        return [motion.name for motion in self.motions]

    @property
    def motion_count(self) -> int:
        return len(self.motions)

    def get_stream(self, motion_index: int) -> NpzMotionStream:
        if motion_index not in self._stream_cache:
            motion = self.motions[motion_index]
            self._stream_cache[motion_index] = NpzMotionStream(
                npz_path=str(motion.path),
                policy_joint_names=self.policy_joint_names,
                policy_body_names=self.policy_body_names,
            )
        return self._stream_cache[motion_index]


class MujocoSim2Sim:
    DEFAULT_POSE = "default_pose"
    PRE_STAND = "pre_stand"
    PLAY_MOTION = "play_motion"

    def __init__(
        self,
        mjcf_path: Path,
        policy: OnnxMotionPolicy,
        control_dt: float,
        sim_dt: float,
        zero_xy_ref: bool,
        max_target_step: float | None,
        motion_stream: "NpzMotionStream | None" = None,
        motion_library: "MotionLibrary | None" = None,
        stand_duration_s: float = 1.0,
        target_ema_alpha: float = 1.0,
    ):
        self.policy = policy
        self.metadata = policy.metadata
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = sim_dt
        self.control_dt = control_dt
        self.decimation = max(1, int(round(control_dt / sim_dt)))
        self.zero_xy_ref = zero_xy_ref
        self.max_target_step = max_target_step
        self.motion_stream = motion_stream
        self.motion_library = motion_library
        if self.motion_library is not None and self.motion_stream is not None:
            raise RuntimeError("Use either `motion_stream` or `motion_library`, not both.")
        self.stand_duration_s = max(float(stand_duration_s), self.control_dt)
        self.target_ema_alpha = float(np.clip(target_ema_alpha, 0.0, 1.0))
        self._filtered_q_target: np.ndarray | None = None
        self._stand_total_steps = max(1, int(round(self.stand_duration_s / self.control_dt)))
        self._request_lock = threading.Lock()
        self._pending_next_motion_requests = 0
        self._playback_phase = self.DEFAULT_POSE
        self._active_motion_index: int | None = None
        self._active_motion_name: str | None = None
        self._next_motion_index: int = 0
        self._stand_step_count: int = 0

        joint_qpos_addr = self._build_joint_qpos_addr()
        joint_qvel_addr = self._build_joint_qvel_addr()
        self.actuator_ids = self._build_actuator_ids()
        self.joint_qpos_indices = np.array(
            [joint_qpos_addr[joint_name] for joint_name in self.metadata.joint_names],
            dtype=np.int32,
        )
        self.joint_qvel_indices = np.array(
            [joint_qvel_addr[joint_name] for joint_name in self.metadata.joint_names],
            dtype=np.int32,
        )
        self.actuator_index_array = np.array(
            [self.actuator_ids[joint_name] for joint_name in self.metadata.joint_names],
            dtype=np.int32,
        )
        self.sensor_slices = self._build_sensor_slices()
        self._apply_policy_dynamics()
        self._foot_geom_ids = self._build_foot_geom_ids()

        # The policy metadata is the single source of truth for its standing pose.
        self._standing_joint_pos = self.metadata.default_joint_pos.copy()

        self.zero_obs = np.zeros(self.metadata.obs_dim, dtype=np.float32)
        self._reference_xy_offset = np.zeros(2, dtype=np.float64)
        self._reference_yaw = 0.0
        self._reference_yaw_inv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if self.motion_stream is not None:
            self.reference_outputs = []
            self._set_reference_frame(self.motion_stream.reference_at(0))
        elif self.motion_library is not None:
            self.reference_outputs = []
        else:
            self.reference_outputs = self._build_reference_outputs()
        self.default_reference = self._build_default_reference() if self.motion_library is not None else None
        self.last_action = np.zeros(self.metadata.action_dim, dtype=np.float32)
        self.current_step = 0

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
            "angular-velocity": 3,
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
        # Walker C1 MJCF uses <motor> actuators (torque ctrl). Convert them at runtime
        # to position-style PD actuators so ctrl is a joint target:
        #   tau = kp * (q_des - q) - kd * qd
        for joint_index, joint_name in enumerate(self.metadata.joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            dof_id = int(self.model.jnt_dofadr[joint_id])
            actuator_id = self.actuator_ids[joint_name]

            kp = float(self.metadata.joint_stiffness[joint_index])
            kd = float(self.metadata.joint_damping[joint_index])
            effort_limit = float(self.metadata.joint_effort_limit[joint_index])

            self.model.dof_damping[dof_id] = 0.0
            self.model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gainprm[actuator_id, 0] = kp
            self.model.actuator_biasprm[actuator_id, 0] = 0.0
            self.model.actuator_biasprm[actuator_id, 1] = -kp
            self.model.actuator_biasprm[actuator_id, 2] = -kd
            # ctrl is a joint position target (rad), not torque
            if int(self.model.jnt_limited[joint_id]) == 1:
                q_lo = float(self.model.jnt_range[joint_id, 0])
                q_hi = float(self.model.jnt_range[joint_id, 1])
            else:
                q_lo, q_hi = -3.14, 3.14
            self.model.actuator_ctrllimited[actuator_id] = 1
            self.model.actuator_ctrlrange[actuator_id, 0] = q_lo
            self.model.actuator_ctrlrange[actuator_id, 1] = q_hi
            self.model.actuator_forcelimited[actuator_id] = 1
            self.model.actuator_forcerange[actuator_id, 0] = -effort_limit
            self.model.actuator_forcerange[actuator_id, 1] = effort_limit

    def _resolve_body_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return body_id
        # C1 MJCF uses "base_link" as root body — no alias needed
        return -1

    def _build_foot_geom_ids(self) -> set[int]:
        """Resolve collision geoms used to place the feet just above the floor."""
        geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_GEOM_NAMES
        }
        geom_ids.discard(-1)
        if not geom_ids:
            raise RuntimeError("No foot collision geoms found in the MuJoCo model.")
        return geom_ids

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

    def _set_reference_frame_from_robot(self, ref0: dict[str, np.ndarray]) -> None:
        """Yaw-align a new motion using only the deployable IMU orientation."""
        if not self.zero_xy_ref:
            self._reference_xy_offset[:] = 0.0
            self._reference_yaw = 0.0
            self._reference_yaw_inv = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return
        robot_quat = quat_normalize(self._sensor("orientation"))
        robot_yaw = yaw_from_quat(robot_quat)
        self._reference_xy_offset = np.asarray(
            ref0["body_pos_w"][0, :2], dtype=np.float64
        ).copy()
        ref_yaw = yaw_from_quat(ref0["body_quat_w"][0])
        delta_yaw = robot_yaw - ref_yaw
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
        # Build the idle body reference by running ONNX default_joint_pos through FK.
        temp_data.qpos[self.joint_qpos_indices] = np.asarray(self._standing_joint_pos, dtype=np.float64)
        temp_data.qvel[self.joint_qvel_indices] = 0.0
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
            "joint_pos": np.asarray(self._standing_joint_pos, dtype=np.float32).copy(),
            "joint_vel": np.zeros_like(self._standing_joint_pos, dtype=np.float32),
            "body_pos_w": body_pos_w_arr,
            "body_quat_w": body_quat_w_arr,
            "body_lin_vel_w": np.zeros_like(body_pos_w_arr, dtype=np.float64),
            "body_ang_vel_w": np.zeros_like(body_pos_w_arr, dtype=np.float64),
        }

    def _place_robot_on_floor(self, clearance: float = 1.0e-4) -> None:
        """Simulation-only: place the initialized robot in contact with the floor.

        This runs once during reset and is not part of policy inference or the
        reference trajectory, so real deployment does not need an equivalent.
        """
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0:
            return

        mujoco.mj_forward(self.model, self.data)
        fromto = np.zeros(6, dtype=np.float64)
        min_dist = min(
            float(
                mujoco.mj_geomDistance(
                    self.model, self.data, floor_id, geom_id, 2.0, fromto
                )
            )
            for geom_id in self._foot_geom_ids
        )

        base_shift = clearance - min_dist
        if abs(base_shift) > 1.0e-9:
            self.data.qpos[2] += base_shift
            mujoco.mj_forward(self.model, self.data)

    def _current_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos_indices].astype(np.float32, copy=True)

    def _apply_joint_target(self, q_target: np.ndarray) -> None:
        q_target = np.asarray(q_target, dtype=np.float32)
        if self.max_target_step is not None:
            q_curr = self._current_joint_positions()
            q_target = np.clip(q_target, q_curr - self.max_target_step, q_curr + self.max_target_step)
        if self.target_ema_alpha < 1.0:
            if self._filtered_q_target is None:
                self._filtered_q_target = q_target.copy()
            else:
                self._filtered_q_target = (
                    self.target_ema_alpha * q_target + (1.0 - self.target_ema_alpha) * self._filtered_q_target
                ).astype(np.float32, copy=False)
            q_target = self._filtered_q_target
        self.data.ctrl[self.actuator_index_array] = q_target.astype(np.float64, copy=False)

    def _step_physics(self) -> None:
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

    def _run_policy_step(
        self,
        ref_now: dict[str, np.ndarray],
        ref_lookahead: dict[str, np.ndarray],
        step_index: int,
        tracking_weight: float = 1.0,
    ) -> np.ndarray:
        obs = self.build_observation(
            step_index, ref_now, ref_lookahead, tracking_weight
        )
        policy_out = self.policy.infer(obs, step_index)
        raw_action = np.asarray(policy_out["actions"], dtype=np.float32)
        q_ref = np.asarray(ref_now["joint_pos"], dtype=np.float32)

        # The actor predicts a residual around the motion reference.
        q_target = q_ref + raw_action * self.metadata.action_scale
        self._apply_joint_target(q_target)
        self._step_physics()
        self.last_action = raw_action
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
        self._playback_phase = self.PRE_STAND
        print(
            f"[playlist] pre-stand ({self.stand_duration_s:.1f}s) before motion "
            f"{next_index + 1}/{self.motion_library.motion_count}: "
            f"{self.motion_library.motion_names[next_index]}"
        )

    def _start_play_motion(self) -> None:
        next_index = self._next_motion_index
        self.motion_stream = self.motion_library.get_stream(next_index)
        self.motion_stream.reset()
        ref0_raw = self.motion_stream.reference_at(0)
        self._set_reference_frame_from_robot(ref0=ref0_raw)
        self._active_motion_index = next_index
        self._active_motion_name = self.motion_library.motion_names[next_index]
        self._playback_phase = self.PLAY_MOTION
        self.current_step = 0
        print(f"[playlist] playing motion: {self._active_motion_name}")

    def _yaw_aligned_default_ref(self) -> dict[str, np.ndarray]:
        robot_quat = quat_normalize(self._sensor("orientation"))
        yaw_quat = quat_from_yaw(yaw_from_quat(robot_quat))
        ref = {k: v.copy() for k, v in self.default_reference.items()}
        ref["body_quat_w"] = np.stack(
            [quat_multiply(yaw_quat, q) for q in self.default_reference["body_quat_w"]],
            axis=0,
        )
        return ref

    def _step_motion_playlist(self) -> bool:
        """Advance DEFAULT_POSE -> PRE_STAND -> PLAY_MOTION state transitions."""
        if self._consume_next_motion_request():
            self._start_pre_stand()

        if self._playback_phase == self.DEFAULT_POSE:
            ref = self._yaw_aligned_default_ref()
            self._run_policy_step(ref, ref, 0, tracking_weight=0.0)
            return False

        if self._playback_phase == self.PRE_STAND:
            ref = self._yaw_aligned_default_ref()
            self._run_policy_step(ref, ref, 0, tracking_weight=0.0)
            self._stand_step_count += 1
            if self._stand_step_count >= self._stand_total_steps:
                self._start_play_motion()
            return False

        if self._playback_phase == self.PLAY_MOTION:
            if self.motion_stream is None:
                self._playback_phase = self.DEFAULT_POSE
                return False
            ref_now = self._reference_output(self.current_step)
            lookahead_step = min(self.current_step + LOOKAHEAD_STEPS, self._total_steps - 1)
            ref_lookahead = self._reference_output(lookahead_step)
            fade_steps = self._stand_total_steps
            # Keep the standing head active during the leading stationary hold,
            # then cross-fade while the reference itself transitions into the
            # source motion. With the default data generator both last one
            # second, matching ``_stand_total_steps``.
            fade_in = float(np.clip(
                (self.current_step - fade_steps + 1) / fade_steps, 0.0, 1.0
            ))
            remaining_steps = self._total_steps - 1 - self.current_step
            fade_out = min(1.0, float(max(remaining_steps, 0)) / fade_steps)
            tracking_weight = min(fade_in, fade_out)
            self._run_policy_step(
                ref_now, ref_lookahead, self.current_step,
                tracking_weight=tracking_weight,
            )
            if self.current_step >= self._total_steps - 1:
                print(f"[playlist] finished motion: {self._active_motion_name}")
                # Every *_with_transition clip already ends at the exact idle
                # pose with zero reference velocity, so no synthetic blend is needed.
                # Do not carry tracking-head history into the standing head.
                self.last_action[:] = 0.0
                self._filtered_q_target = self._current_joint_positions()
                self._playback_phase = self.DEFAULT_POSE
                print("[playlist] back to default_pose")
            else:
                self.current_step += 1
            return False

        return False

    @property
    def _total_steps(self) -> int:
        if self.motion_stream is not None:
            return self.motion_stream.total_steps
        if self.motion_library is not None and self._active_motion_index is not None:
            return self.motion_library.get_stream(self._active_motion_index).total_steps
        return self.metadata.total_steps

    def _reference_output(self, step_index: int) -> dict[str, np.ndarray]:
        if self.motion_stream is not None:
            return self._normalize_reference_output(self.motion_stream.reference_at(step_index))
        return self.reference_outputs[step_index]

    def reset(self) -> None:
        self.current_step = 0
        self.last_action[:] = 0.0
        self._filtered_q_target = None
        if self.motion_stream is not None:
            self.motion_stream.reset()
        if self.motion_library is not None:
            with self._request_lock:
                self._pending_next_motion_requests = 0
            self._playback_phase = self.DEFAULT_POSE
            self._stand_step_count = 0
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
            ref0 = self.reference_outputs[0] if self.motion_stream is None else self._reference_output(0)
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
        self._filtered_q_target = np.asarray(joint_pos, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        self._place_robot_on_floor()

    def rewind_motion(self) -> None:
        if self.motion_library is not None:
            self.reset()
            return
        self.current_step = 0
        self.last_action[:] = 0.0
        self._filtered_q_target = None
        if self.motion_stream is not None:
            self.motion_stream.reset()

    def build_observation(
        self,
        step_index: int,
        ref_now: dict[str, np.ndarray],
        ref_lookahead: dict[str, np.ndarray],
        tracking_weight: float = 1.0,
    ) -> np.ndarray:
        """Build the policy observation in the exact order used during training.

        Merged policies append ``[standing, tracking]`` weights to the common
        214-value observation; single-head policies use only the common part.
        """
        base_quat_w = quat_normalize(self._sensor("orientation"))
        base_ang_vel_b = self._sensor("angular-velocity")

        q_cur = self.data.qpos[self.joint_qpos_indices].astype(np.float32, copy=True)
        qd_cur = self.data.qvel[self.joint_qvel_indices].astype(np.float32, copy=True)

        q_ref = np.asarray(ref_now["joint_pos"], dtype=np.float32)
        qd_ref = np.asarray(ref_now["joint_vel"], dtype=np.float32)
        qd_ref_lookahead = np.asarray(ref_lookahead["joint_vel"], dtype=np.float32)
        q_ref_lookahead = np.asarray(ref_lookahead["joint_pos"], dtype=np.float32)

        anchor_quat_ref_w = quat_normalize(np.asarray(ref_now["body_quat_w"][0], dtype=np.float64))
        anchor_quat_rel = quat_multiply(quat_conjugate(base_quat_w), anchor_quat_ref_w)
        anchor_rot6d = rotmat_to_rot6d(quat_to_rotmat(anchor_quat_rel))

        total_steps = max(1, self._total_steps - 1)
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
            tracking_weight = float(np.clip(tracking_weight, 0.0, 1.0))
            head_weights = np.array(
                [1.0 - tracking_weight, tracking_weight], dtype=np.float32
            )
            obs = np.concatenate([obs, head_weights], axis=0)

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
    parser = argparse.ArgumentParser(description="Standalone Mujoco sim2sim runner for Walker C1 ONNX policy.")
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
        "--target-ema-alpha",
        type=float,
        default=1.0,
        help="Low-pass alpha for commanded joint targets; 1.0 disables target filtering",
    )
    parser.add_argument(
        "--keep-reference-world",
        action="store_true",
        help="Do not shift reference XY to the world origin before playback",
    )
    parser.add_argument("--print-every", type=int, default=50, help="Print status every N control steps")
    parser.add_argument("--npz", type=Path, default=None, help="Track one NPZ motion file")
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
        help="Duration of STAND phase before each motion clip (default: 1.0s).",
    )
    return parser


def run_headless(sim: MujocoSim2Sim, max_steps: int | None, loop: bool, print_every: int) -> None:
    step_count = 0
    while True:
        motion_finished = sim.step()
        step_count += 1
        reached_step_limit = max_steps is not None and step_count >= max_steps
        if print_every > 0 and (step_count == 1 or step_count % print_every == 0 or reached_step_limit):
            base_pos = sim.data.qpos[:3].copy()
            print(f"step={step_count:04d} motion_step={sim.current_step:04d} base_pos={base_pos}")

        if motion_finished:
            if loop:
                sim.rewind_motion()
            elif max_steps is None or step_count <= max_steps:
                break

        if reached_step_limit:
            break

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

    with mujoco_viewer.launch_passive(sim.model, sim.data, key_callback=on_key, show_left_ui=False, show_right_ui=False) as viewer:
        # Pull camera back for a wider view of the full robot
        viewer.cam.distance = 4.0
        viewer.cam.lookat = [0.0, 0.0, 0.9]
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0

        step_count = 0
        while viewer.is_running():
            step_start = time.time()
            motion_finished = sim.step()
            viewer.sync()
            step_count += 1
            reached_step_limit = max_steps is not None and step_count >= max_steps

            if print_every > 0 and (step_count == 1 or step_count % print_every == 0 or reached_step_limit):
                base_pos = sim.data.qpos[:3].copy()
                print(f"step={step_count:04d} motion_step={sim.current_step:04d} base_pos={base_pos}")

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

    motion_stream = None
    motion_library = None
    if args.npz is not None and args.npz_dir is not None:
        raise ValueError("`--npz` and `--npz-dir` are mutually exclusive.")
    if args.npz is not None:
        npz_path = args.npz.expanduser().resolve()
        if not npz_path.is_file():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        motion_stream = NpzMotionStream(
            npz_path=str(npz_path),
            policy_joint_names=policy.metadata.joint_names,
            policy_body_names=policy.metadata.body_names,
        )
        print(f"[motion] NPZ stream: {npz_path}")
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
        print("[playlist] Idle target uses ONNX default_joint_pos metadata.")

    sim = MujocoSim2Sim(
        mjcf_path=mjcf_path,
        policy=policy,
        control_dt=args.control_dt,
        sim_dt=args.sim_dt,
        zero_xy_ref=not args.keep_reference_world,
        max_target_step=args.max_target_step,
        motion_stream=motion_stream,
        motion_library=motion_library,
        stand_duration_s=args.stand_seconds,
        target_ema_alpha=args.target_ema_alpha,
    )
    max_steps = args.steps
    if max_steps is None and not args.loop and motion_library is None:
        max_steps = motion_stream.total_steps if motion_stream is not None else policy.metadata.total_steps

    print(f"policy={policy_path}")
    print(f"policy_mode={policy_mode} raw_obs_dim={policy_raw_obs_dim}")
    print(f"mjcf={mjcf_path}")
    initial_pose_mode = "playlist_default_reference" if motion_library is not None else "from_reference_frame_0"
    print(f"initial_pose={initial_pose_mode}")
    print(f"obs_dim={policy.metadata.obs_dim} action_dim={policy.metadata.action_dim} total_steps={policy.metadata.total_steps}")
    print(f"control_dt={args.control_dt:.4f} sim_dt={args.sim_dt:.4f} decimation={sim.decimation}")
    print(f"max_steps={'inf' if max_steps is None else max_steps}")
    print(f"playback_rate={args.playback_rate:.3f}x")
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
