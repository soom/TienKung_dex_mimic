"""CSV → NPZ converter using MuJoCo FK (no Isaac Sim required).

Single-file pipeline that replaces the three-script workflow:
    csv_to_npz_mujoco.py  +  add_transition.py  +  recompute_npz_body_velocities.py
and the shell orchestrator batch_csv_to_npz_pro.sh.

── Single-file mode (original behaviour) ──────────────────────────────────────
    python scripts/csv_to_npz_mujoco.py \\
        --input_file dataset/csv/foo.csv \\
        --input_fps 30 \\
        --output_name dataset/npz_pro_raw/foo \\
        --robot tienkung2_pro

── Single-file + transition ───────────────────────────────────────────────────
    python scripts/csv_to_npz_mujoco.py \\
        --input_file dataset/csv/foo.csv \\
        --output_name dataset/npz_pro/foo_with_transition \\
        --add_transition

── Batch mode (replaces batch_csv_to_npz_pro.sh) ─────────────────────────────
    python scripts/csv_to_npz_mujoco.py \\
        --batch_dir dataset/csv/ \\
        --raw_output_dir dataset/npz_pro_raw/ \\
        --output_dir dataset/npz_pro/ \\
        --add_transition \\
        --recompute_velocities \\
        [--force]

    # headless / local flags are accepted but ignored (no GUI needed)
    bash scripts/batch_csv_to_npz_pro.sh
"""

import argparse
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Robot definitions
# ---------------------------------------------------------------------------

ROBOT_MJCF = {
    "c1": "source/whole_body_tracking/whole_body_tracking/assets/walker_c1/mjcf/walker_astron.xml",
    "tienkung2_pro": "ros_deploy_pro/src/tienkung2_pro/assets/mjcf/tienkung_new.xml",
    "dex_evt": "source/whole_body_tracking/whole_body_tracking/assets/dex_evt/urdf/evt2.xml",
}

# NPZ body name → MJCF body name (only entries that differ)
# Pro uses "Base_link" as root body in MJCF; dex uses "pelvis" natively.
_NPZ_TO_MJCF_BODY = {"pelvis": "Base_link"}
_NPZ_TO_MJCF_BODY_DEX = {}  # dex body names match MJCF 1:1

# Bodies present in NPZ but absent from MJCF (head links); fall back to pelvis
_MISSING_BODY_FALLBACK = "pelvis"

ROBOT_JOINT_NAMES = {
    "c1": [
        "L_hip_pitch_joint", "L_hip_roll_joint", "L_hip_yaw_joint",
        "L_knee_pitch_joint", "L_ankle_pitch_joint", "L_ankle_roll_joint",
        "R_hip_pitch_joint", "R_hip_roll_joint", "R_hip_yaw_joint",
        "R_knee_pitch_joint", "R_ankle_pitch_joint", "R_ankle_roll_joint",
        "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
        "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
        "L_elbow_pitch_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
        "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
        "R_elbow_pitch_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint",
    ],
    "tienkung2_pro": [
        "body_yaw_joint",
        "hip_roll_l_joint",
        "hip_roll_r_joint",
        "shoulder_pitch_l_joint",
        "shoulder_pitch_r_joint",
        "hip_pitch_l_joint",
        "hip_pitch_r_joint",
        "shoulder_roll_l_joint",
        "shoulder_roll_r_joint",
        "hip_yaw_l_joint",
        "hip_yaw_r_joint",
        "shoulder_yaw_l_joint",
        "shoulder_yaw_r_joint",
        "knee_pitch_l_joint",
        "knee_pitch_r_joint",
        "elbow_pitch_l_joint",
        "elbow_pitch_r_joint",
        "ankle_pitch_l_joint",
        "ankle_pitch_r_joint",
        "elbow_yaw_l_joint",
        "elbow_yaw_r_joint",
        "ankle_roll_l_joint",
        "ankle_roll_r_joint",
        "wrist_pitch_l_joint",
        "wrist_pitch_r_joint",
        "wrist_roll_l_joint",
        "wrist_roll_r_joint",
    ],
    "dex_evt": [
        "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
        "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
        "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
        "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
        "elbow_pitch_l_joint", "elbow_yaw_l_joint",
        "wrist_pitch_l_joint", "wrist_roll_l_joint",
        "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint", "elbow_yaw_r_joint",
        "wrist_pitch_r_joint", "wrist_roll_r_joint",
    ],
}

# NPZ body_names in the order Isaac Sim exports them (must match training code)
ROBOT_BODY_NAMES = {
    "c1": [
        "base_link", "L_hip_pitch_link", "L_hip_roll_link", "L_hip_yaw_link",
        "L_knee_pitch_link", "L_ankle_pitch_link", "L_ankle_roll_link",
        "R_hip_pitch_link", "R_hip_roll_link", "R_hip_yaw_link",
        "R_knee_pitch_link", "R_ankle_pitch_link", "R_ankle_roll_link",
        "waist_yaw_link", "waist_pitch_link", "waist_roll_link",
        "L_shoulder_pitch_link", "L_shoulder_roll_link", "L_shoulder_yaw_link",
        "L_elbow_pitch_link", "L_elbow_yaw_link", "L_wrist_pitch_link", "L_wrist_roll_link",
        "R_shoulder_pitch_link", "R_shoulder_roll_link", "R_shoulder_yaw_link",
        "R_elbow_pitch_link", "R_elbow_yaw_link", "R_wrist_pitch_link", "R_wrist_roll_link",
    ],
    "tienkung2_pro": [
        "pelvis",
        "body_yaw_link",
        "hip_roll_l_link",
        "hip_roll_r_link",
        "head_yaw_link",
        "shoulder_pitch_l_link",
        "shoulder_pitch_r_link",
        "hip_pitch_l_link",
        "hip_pitch_r_link",
        "head_pitch_link",
        "shoulder_roll_l_link",
        "shoulder_roll_r_link",
        "hip_yaw_l_link",
        "hip_yaw_r_link",
        "head_roll_link",
        "shoulder_yaw_l_link",
        "shoulder_yaw_r_link",
        "knee_pitch_l_link",
        "knee_pitch_r_link",
        "elbow_pitch_l_link",
        "elbow_pitch_r_link",
        "ankle_pitch_l_link",
        "ankle_pitch_r_link",
        "elbow_yaw_l_link",
        "elbow_yaw_r_link",
        "ankle_roll_l_link",
        "ankle_roll_r_link",
        "wrist_pitch_l_link",
        "wrist_pitch_r_link",
        "wrist_roll_l_link",
        "wrist_roll_r_link",
    ],
    "dex_evt": [
        "pelvis",
        "hip_pitch_l_link", "hip_roll_l_link", "hip_yaw_l_link",
        "knee_pitch_l_link", "ankle_pitch_l_link", "ankle_roll_l_link",
        "hip_pitch_r_link", "hip_roll_r_link", "hip_yaw_r_link",
        "knee_pitch_r_link", "ankle_pitch_r_link", "ankle_roll_r_link",
        "waist_yaw_link", "waist_roll_link", "waist_pitch_link",
        "shoulder_pitch_l_link", "shoulder_roll_l_link", "shoulder_yaw_l_link",
        "elbow_pitch_l_link", "elbow_yaw_l_link",
        "wrist_pitch_l_link", "wrist_roll_l_link",
        "shoulder_pitch_r_link", "shoulder_roll_r_link", "shoulder_yaw_r_link",
        "elbow_pitch_r_link", "elbow_yaw_r_link",
        "wrist_pitch_r_link", "wrist_roll_r_link",
    ],
}

# ---------------------------------------------------------------------------
# Standing poses (from robot init_state)
# ---------------------------------------------------------------------------
STANDING_JOINT_POS_MAP = {
    "c1": {
        "L_hip_pitch_joint": -0.05, "R_hip_pitch_joint": -0.05,
        "L_knee_pitch_joint": 0.10, "R_knee_pitch_joint": 0.10,
        "L_ankle_pitch_joint": -0.05, "R_ankle_pitch_joint": -0.05,
        "L_shoulder_roll_joint": 0.1, "R_shoulder_roll_joint": -0.1,
    },
    "tienkung2_pro": {
        "hip_pitch_l_joint":    -0.5,
        "hip_pitch_r_joint":    -0.5,
        "knee_pitch_l_joint":    1.0,
        "knee_pitch_r_joint":    1.0,
        "ankle_pitch_l_joint":  -0.5,
        "ankle_pitch_r_joint":  -0.5,
        "shoulder_roll_l_joint":  0.1,
        "shoulder_roll_r_joint": -0.1,
        "elbow_pitch_l_joint":  -0.3,
        "elbow_pitch_r_joint":  -0.3,
    },
    "dex_evt": {
        # Match source/whole_body_tracking/.../robots/dex_evt.py init_state.
        "hip_yaw_l_joint": 0.0,
        "hip_roll_l_joint": 0.0,
        "hip_pitch_l_joint": -0.05,
        "knee_pitch_l_joint": 0.05,
        "ankle_pitch_l_joint": -0.05,
        "ankle_roll_l_joint": 0.0,
        "hip_yaw_r_joint": 0.0,
        "hip_roll_r_joint": 0.0,
        "hip_pitch_r_joint": -0.05,
        "knee_pitch_r_joint": 0.05,
        "ankle_pitch_r_joint": -0.05,
        "ankle_roll_r_joint": 0.0,
        "shoulder_pitch_l_joint": 0.0,
        "shoulder_roll_l_joint": 0.1,
        "shoulder_yaw_l_joint": 0.0,
        "elbow_pitch_l_joint": -0.1,
        "elbow_yaw_l_joint": 0.0,
        "wrist_pitch_l_joint": 0.0,
        "wrist_roll_l_joint": 0.0,
        "shoulder_pitch_r_joint": 0.0,
        "shoulder_roll_r_joint": -0.1,
        "shoulder_yaw_r_joint": 0.0,
        "elbow_pitch_r_joint": -0.1,
        "elbow_yaw_r_joint": 0.0,
        "wrist_pitch_r_joint": 0.0,
        "wrist_roll_r_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
    },
}

DEFAULT_MJCF = ROBOT_MJCF["c1"]


# ---------------------------------------------------------------------------
# CSV loading + interpolation
# ---------------------------------------------------------------------------

def load_csv(
    csv_path: str,
    input_fps: int,
    output_fps: int,
    joint_names: list[str],
    frame_range: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Load CSV, interpolate to output_fps, return raw arrays."""
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    name_to_col = {n: i for i, n in enumerate(header)}
    missing = [n for n in joint_names if n not in name_to_col]
    if missing:
        raise KeyError(f"CSV missing columns: {missing}")

    skiprows = 1
    if frame_range is not None:
        raw = np.loadtxt(
            csv_path, delimiter=",", skiprows=skiprows + frame_range[0] - 1,
            max_rows=frame_range[1] - frame_range[0] + 1, dtype=np.float32,
        )
    else:
        raw = np.loadtxt(csv_path, delimiter=",", skiprows=skiprows, dtype=np.float32)

    raw = np.atleast_2d(raw)
    T_in = raw.shape[0]
    duration = (T_in - 1) / input_fps

    base_pos_in  = raw[:, :3]
    quat_xyzw_in = raw[:, 3:7]
    quat_wxyz_in = quat_xyzw_in[:, [3, 0, 1, 2]]
    joint_pos_in = raw[:, [name_to_col[n] for n in joint_names]]

    print(f"[csv_to_npz] loaded {csv_path}  frames={T_in}  duration={duration:.2f}s")

    times = np.arange(0, duration, 1.0 / output_fps, dtype=np.float64)
    T_out = len(times)
    phase = times / duration
    idx0 = np.floor(phase * (T_in - 1)).astype(int)
    idx1 = np.minimum(idx0 + 1, T_in - 1)
    blend = (phase * (T_in - 1) - idx0).astype(np.float32)

    base_pos  = base_pos_in[idx0] * (1 - blend[:, None]) + base_pos_in[idx1] * blend[:, None]
    joint_pos = joint_pos_in[idx0] * (1 - blend[:, None]) + joint_pos_in[idx1] * blend[:, None]
    base_quat = _slerp_batch(quat_wxyz_in, idx0, idx1, blend)

    print(f"[csv_to_npz] interpolated → frames={T_out}  fps={output_fps}")
    return {
        "base_pos":   base_pos.astype(np.float32),
        "base_quat":  base_quat.astype(np.float32),
        "joint_pos":  joint_pos.astype(np.float32),
    }


def _slerp_batch(
    quats: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    blend: np.ndarray,
) -> np.ndarray:
    q0 = quats[idx0]
    q1 = quats[idx1]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, 0.0, 1.0)

    linear = q0 + blend[:, None] * (q1 - q0)
    linear /= np.linalg.norm(linear, axis=-1, keepdims=True).clip(1e-8)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * blend[:, None]
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0.clip(1e-8)
    s1 = np.sin(theta) / sin_theta_0.clip(1e-8)
    slerp = s0 * q0 + s1 * q1
    slerp /= np.linalg.norm(slerp, axis=-1, keepdims=True).clip(1e-8)

    use_linear = (dot > 0.9995).squeeze(-1)
    return np.where(use_linear[:, None], linear, slerp).astype(np.float32)


# ---------------------------------------------------------------------------
# MuJoCo FK
# ---------------------------------------------------------------------------

def run_fk(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    joint_names: list[str],
    body_names: list[str],
    mjcf_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MuJoCo FK for every frame. Returns body_pos_w (T,B,3) and body_quat_w (T,B,4) wxyz."""
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data  = mujoco.MjData(model)

    joint_qposadr = {}
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint '{jname}' not found in MJCF {mjcf_path}")
        joint_qposadr[jname] = int(model.jnt_qposadr[jid])

    fallback_mjcf = _NPZ_TO_MJCF_BODY.get(_MISSING_BODY_FALLBACK, _MISSING_BODY_FALLBACK)
    fallback_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fallback_mjcf)
    # If the remapped fallback body doesn't exist either (e.g. dex MJCF has
    # no "Base_link"), fall back to the original name directly.
    if fallback_bid < 0:
        fallback_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _MISSING_BODY_FALLBACK)
    body_ids = []
    for bname in body_names:
        mjcf_name = _NPZ_TO_MJCF_BODY.get(bname, bname)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjcf_name)
        # If the remapped name doesn't exist in this MJCF, try the original NPZ name
        if bid < 0:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        body_ids.append(bid if bid >= 0 else fallback_bid)

    T = base_pos.shape[0]
    B = len(body_names)
    out_pos  = np.zeros((T, B, 3), dtype=np.float32)
    out_quat = np.zeros((T, B, 4), dtype=np.float32)

    for t in range(T):
        data.qpos[:3]  = base_pos[t]
        data.qpos[3:7] = base_quat[t]
        for i, jname in enumerate(joint_names):
            data.qpos[joint_qposadr[jname]] = joint_pos[t, i]
        mujoco.mj_forward(model, data)
        for b, bid in enumerate(body_ids):
            out_pos[t, b]  = data.xpos[bid]
            out_quat[t, b] = data.xquat[bid]

    print(f"[csv_to_npz] FK done  frames={T}  bodies={B}")
    return out_pos, out_quat


# ---------------------------------------------------------------------------
# Velocity recomputation (raw NPZ — no segment awareness)
# ---------------------------------------------------------------------------

def recompute_velocities_raw(
    joint_pos: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    fps: float,
    smooth_window: int = 11,
    smooth_poly: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute joint_vel, body_lin_vel_w, body_ang_vel_w via smoothed finite differences."""
    dt = 1.0 / fps
    T = joint_pos.shape[0]

    def smooth(arr):
        if T < smooth_window:
            return arr.astype(np.float32)
        return savgol_filter(arr.astype(np.float32), window_length=smooth_window,
                             polyorder=smooth_poly, axis=0)

    def central_diff(arr):
        out = np.empty_like(arr)
        out[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
        out[0]    = (arr[1]  - arr[0])   / dt
        out[-1]   = (arr[-1] - arr[-2])  / dt
        return out

    def qmul(a, b):
        aw,ax,ay,az = a[...,0],a[...,1],a[...,2],a[...,3]
        bw,bx,by,bz = b[...,0],b[...,1],b[...,2],b[...,3]
        return np.stack([aw*bw-ax*bx-ay*by-az*bz,
                         aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx,
                         aw*bz+ax*by-ay*bx+az*bw], axis=-1)

    def qconj(q):
        c = q.copy(); c[..., 1:] *= -1; return c

    def axis_angle(q):
        sin_half = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
        angle = 2.0 * np.arctan2(sin_half, q[..., :1])
        safe = np.where(sin_half > 1e-8, sin_half, np.ones_like(sin_half))
        return (angle * q[..., 1:] / safe).astype(np.float32)

    def ensure_continuity(q):
        q = q.copy()
        for t in range(1, q.shape[0]):
            dot = np.sum(q[t] * q[t-1], axis=-1, keepdims=True)
            q[t] = np.where(dot < 0, -q[t], q[t])
        return q

    joint_vel = central_diff(smooth(joint_pos))
    body_lin_vel_w = central_diff(smooth(body_pos_w))

    body_quat_w = ensure_continuity(body_quat_w.astype(np.float32))
    if T == 1:
        body_ang_vel_w = np.zeros((1, body_quat_w.shape[1], 3), dtype=np.float32)
    else:
        body_ang_vel_w = np.zeros((T, body_quat_w.shape[1], 3), dtype=np.float32)
        q_rel = qmul(body_quat_w[2:], qconj(body_quat_w[:-2]))
        body_ang_vel_w[1:-1] = axis_angle(q_rel) / (2 * dt)
        body_ang_vel_w[0]    = (axis_angle(qmul(body_quat_w[1:2], qconj(body_quat_w[0:1]))) / dt)[0]
        body_ang_vel_w[-1]   = (axis_angle(qmul(body_quat_w[-1:], qconj(body_quat_w[-2:-1]))) / dt)[0]
        body_ang_vel_w = smooth(body_ang_vel_w)

    return (joint_vel.astype(np.float32),
            body_lin_vel_w.astype(np.float32),
            body_ang_vel_w.astype(np.float32))


def report_velocity_anomalies(joint_vel, body_lin_vel_w, body_ang_vel_w,
                              joint_names, fps, label="motion"):
    """Print actionable diagnostics for implausible or non-finite velocities."""
    print(f"[velocity_check] {label}: joint max={np.nanmax(np.abs(joint_vel)):.2f} rad/s, "
          f"root lin max={np.nanmax(np.linalg.norm(body_lin_vel_w[:, 0], axis=1)):.2f} m/s, "
          f"root ang max={np.nanmax(np.linalg.norm(body_ang_vel_w[:, 0], axis=1)):.2f} rad/s")
    bad = ~np.isfinite(joint_vel).all(axis=1)
    speed = np.abs(joint_vel)
    spikes = np.argwhere(speed > 20.0)
    root_lin = np.linalg.norm(body_lin_vel_w[:, 0], axis=1)
    root_ang = np.linalg.norm(body_ang_vel_w[:, 0], axis=1)
    root_bad = np.where((root_lin > 10.0) | (root_ang > 30.0))[0]
    if bad.any():
        print(f"[velocity_check][异常] {label}: 第 {np.where(bad)[0][:10].tolist()} 帧含 NaN/Inf；原因通常是输入缺失或四元数非法")
    if len(spikes):
        shown = [(int(t), joint_names[int(j)], float(speed[t, j])) for t, j in spikes[:10]]
        print(f"[velocity_check][异常] {label}: 关节速度超过 20 rad/s: {shown}; 可能是原始帧跳变或插值尖峰")
    if len(root_bad):
        print(f"[velocity_check][异常] {label}: 根部速度异常帧 {root_bad[:10].tolist()}; 可能是根轨迹跳变或坐标系不连续")


# ---------------------------------------------------------------------------
# Transition helpers (from add_transition.py)
# ---------------------------------------------------------------------------

def _quintic(t: np.ndarray) -> np.ndarray:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _quat_slerp_batch(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0[None] + t[:, None] * (q1 - q0)[None]
        norms = np.linalg.norm(result, axis=-1, keepdims=True)
        return result / np.where(norms > 0, norms, 1.0)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0[:, None] * q0[None] + s1[:, None] * q1[None]


def _auto_skip_frames(joint_pos: np.ndarray, fps: float, threshold: float = 2.0) -> tuple[int, int]:
    vel = np.abs(np.diff(joint_pos, axis=0)) * fps
    max_vel = vel.max(axis=1)
    N = len(max_vel)
    head_skip = 0
    for i, v in enumerate(max_vel):
        if v < threshold:
            head_skip = i
            break
    else:
        head_skip = N
    tail_skip = 0
    for i in range(N - 1, -1, -1):
        if max_vel[i] < threshold:
            tail_skip = N - 1 - i
            break
    else:
        tail_skip = N
    return head_skip, tail_skip


def _build_standing_pose_fk(
    npz_body_names: list[str],
    joint_names: list[str],
    stand_height: float,
    mjcf_path: str,
    yaw: float = 0.0,
    standing_joint_pos_map: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if standing_joint_pos_map is None:
        standing_joint_pos_map = STANDING_JOINT_POS_MAP.get("tienkung2_pro", {})
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[2] = stand_height
    half_yaw = yaw / 2.0
    data.qpos[3] = float(np.cos(half_yaw))
    data.qpos[6] = float(np.sin(half_yaw))
    for jname, jval in standing_joint_pos_map.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = jval
    mujoco.mj_forward(model, data)

    pelvis_mjcf = _NPZ_TO_MJCF_BODY.get("pelvis", "pelvis")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pelvis_mjcf)
    if pelvis_id < 0:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    fallback_pos  = np.array(data.xpos[pelvis_id],  dtype=np.float32)
    fallback_quat = np.array(data.xquat[pelvis_id], dtype=np.float32)

    stand_bpos  = np.zeros((len(npz_body_names), 3), dtype=np.float32)
    stand_bquat = np.zeros((len(npz_body_names), 4), dtype=np.float32)
    for i, bname in enumerate(npz_body_names):
        mjcf_name = _NPZ_TO_MJCF_BODY.get(bname, bname)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjcf_name)
        if bid < 0:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            stand_bpos[i]  = data.xpos[bid]
            stand_bquat[i] = data.xquat[bid]
        else:
            stand_bpos[i]  = fallback_pos
            stand_bquat[i] = fallback_quat

    stand_jpos = np.zeros(len(joint_names), dtype=np.float32)
    for i, jname in enumerate(joint_names):
        stand_jpos[i] = standing_joint_pos_map.get(jname, 0.0)

    # ── Compute foot sole Z from geom positions (not body origins) ──
    _foot_sole_z = 0.0
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "ankle_roll" not in bname and "foot" not in bname:
            continue
        start = model.body_geomadr[bid]
        end = start + model.body_geomnum[bid]
        for gid in range(start, end):
            gz = data.geom_xpos[gid][2]
            gsize = model.geom_size[gid]
            gtype = model.geom_type[gid]
            # cylinder/capsule: size=(radius, half_length), z-extent = radius
            # box: size=(x, y, z), z-extent = z
            # mesh: use size[2] as approximate margin
            if gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                gz -= gsize[0]  # radius
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                gz -= gsize[2]
            elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                gz -= 0.01  # small margin for mesh
            if gz < _foot_sole_z:
                _foot_sole_z = gz
    return stand_jpos, stand_bpos, stand_bquat, float(_foot_sole_z)


def _recompute_velocities_segmented(
    joint_pos: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    fps: float,
    hold_frames: int = 0,
    trans_frames: int = 0,
    tail_trans_frames: int = 0,
    tail_hold_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment-aware velocity recomputation for transition NPZ."""
    dt = 1.0 / fps
    T = joint_pos.shape[0]

    def _grad_seg(arr, s, e):
        seg = arr[s:e]
        L = e - s
        if L == 1:
            return np.zeros_like(seg)
        out = np.empty_like(seg)
        out[1:-1] = (seg[2:] - seg[:-2]) / (2 * dt)
        out[0]  = (seg[1] - seg[0])  / dt
        out[-1] = (seg[-1] - seg[-2]) / dt
        return out

    def _ang_vel_seg(quat, s, e):
        seg = quat[s:e]
        L, B = seg.shape[0], seg.shape[1]
        if L == 1:
            return np.zeros((L, B, 3), dtype=np.float32)

        def qmul(a, b):
            aw,ax,ay,az = a[...,0],a[...,1],a[...,2],a[...,3]
            bw,bx,by,bz = b[...,0],b[...,1],b[...,2],b[...,3]
            return np.stack([aw*bw-ax*bx-ay*by-az*bz,
                             aw*bx+ax*bw+ay*bz-az*by,
                             aw*by-ax*bz+ay*bw+az*bx,
                             aw*bz+ax*by-ay*bx+az*bw], axis=-1)

        def qconj(q):
            c = q.copy(); c[...,1:] *= -1; return c

        out = np.zeros((L, B, 3), dtype=np.float32)
        dq = qmul(seg[2:], qconj(seg[:-2]))
        out[1:-1] = (2.0 * dq[..., 1:] / (2.0 * dt)).astype(np.float32)
        dq0 = qmul(seg[1:2], qconj(seg[0:1]))
        out[0] = (2.0 * dq0[..., 1:] / dt).astype(np.float32)[0]
        dqe = qmul(seg[-1:], qconj(seg[-2:-1]))
        out[-1] = (2.0 * dqe[..., 1:] / dt).astype(np.float32)[0]
        return out

    orig_start = hold_frames + trans_frames
    orig_end   = T - tail_trans_frames - tail_hold_frames
    segments = []
    if hold_frames > 0:
        segments.append(("hold",       0,           hold_frames))
    if trans_frames > 0:
        segments.append(("trans",      hold_frames, orig_start))
    if orig_end > orig_start:
        segments.append(("orig",       orig_start,  orig_end))
    if tail_trans_frames > 0:
        segments.append(("tail_trans", orig_end,    orig_end + tail_trans_frames))
    if tail_hold_frames > 0:
        segments.append(("tail_hold",  orig_end + tail_trans_frames, T))

    joint_vel      = np.zeros_like(joint_pos)
    body_lin_vel_w = np.zeros_like(body_pos_w)
    body_ang_vel_w = np.zeros((T, body_quat_w.shape[1], 3), dtype=np.float32)

    for seg_type, s, e in segments:
        if seg_type not in ("hold", "tail_hold"):
            joint_vel[s:e]      = _grad_seg(joint_pos,  s, e)
            body_lin_vel_w[s:e] = _grad_seg(body_pos_w, s, e)
            body_ang_vel_w[s:e] = _ang_vel_seg(body_quat_w, s, e)

    return (joint_vel.astype(np.float32),
            body_lin_vel_w.astype(np.float32),
            body_ang_vel_w.astype(np.float32))


def add_transition_to_npz(
    raw_npz: str,
    output_npz: str,
    hold_duration: float,
    trans_duration: float,
    tail_trans_duration: float,
    tail_hold_duration: float,
    stand_height: float,
    head_skip_frames: int,
    tail_skip_frames: int,
    skip_threshold: float,
    mjcf_path: str,
) -> None:
    """Add standing hold + smooth transition to a raw motion NPZ."""
    data = np.load(raw_npz, allow_pickle=True)
    fps = float(np.asarray(data["fps"]).item())
    joint_names = [str(n) for n in data["joint_names"].tolist()]
    npz_body_names = [str(n) for n in data["body_names"].tolist()]

    raw_joint_pos = data["joint_pos"].astype(np.float32)

    auto_head, auto_tail = _auto_skip_frames(raw_joint_pos, fps, threshold=skip_threshold)
    skip_h = auto_head if head_skip_frames < 0 else max(0, head_skip_frames)
    skip_t = auto_tail if tail_skip_frames < 0 else max(0, tail_skip_frames)
    if head_skip_frames < 0:
        print(f"[add_transition] head_skip auto={auto_head} (threshold={skip_threshold} rad/s)")
    if tail_skip_frames < 0:
        print(f"[add_transition] tail_skip auto={auto_tail} (threshold={skip_threshold} rad/s)")

    end_idx = len(raw_joint_pos) - skip_t if skip_t > 0 else len(raw_joint_pos)
    joint_pos_orig = raw_joint_pos[skip_h:end_idx]
    body_pos_orig  = data["body_pos_w"][skip_h:end_idx].astype(np.float32)
    body_quat_orig = data["body_quat_w"][skip_h:end_idx].astype(np.float32)
    if skip_h > 0 or skip_t > 0:
        print(f"[add_transition] cropped: {len(raw_joint_pos)} → {len(joint_pos_orig)} frames"
              f"  (head_skip={skip_h}, tail_skip={skip_t})")

    T, J = joint_pos_orig.shape
    B = body_pos_orig.shape[1]

    hold_frames       = max(1, round(hold_duration       * fps))
    trans_frames      = max(1, round(trans_duration      * fps))
    tail_trans_frames = max(1, round(tail_trans_duration * fps)) if tail_trans_duration > 0 else 0
    tail_hold_frames  = max(1, round(tail_hold_duration  * fps)) if tail_hold_duration  > 0 else 0

    total_frames = T + hold_frames + trans_frames + tail_trans_frames + tail_hold_frames
    print(f"[add_transition] fps={fps:.0f}  original={T} ({T/fps:.2f}s)  output={total_frames} ({total_frames/fps:.2f}s)")

    first_quat = body_quat_orig[0, 0]
    w, x, y, z = first_quat
    first_yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    # Auto-detect robot type from joint names for standing pose
    if "L_hip_pitch_joint" in joint_names:
        stand_map = STANDING_JOINT_POS_MAP["c1"]
    elif "waist_yaw_joint" in joint_names:
        stand_map = STANDING_JOINT_POS_MAP["dex_evt"]
    else:
        stand_map = STANDING_JOINT_POS_MAP["tienkung2_pro"]
    stand_jpos, stand_bpos, stand_bquat, _foot_sole_z = _build_standing_pose_fk(
        npz_body_names, joint_names, stand_height, mjcf_path, yaw=first_yaw,
        standing_joint_pos_map=stand_map)
    xy_offset = body_pos_orig[0, 0, :2] - stand_bpos[0, :2]
    stand_bpos[:, :2] += xy_offset

    # ── Z-align: raise standing pose so foot soles (geoms) sit on ground ──
    if _foot_sole_z < 0:
        stand_bpos[:, 2] += -_foot_sole_z
        print(f"[add_transition] foot-ground Z correction: +{-_foot_sole_z:.4f}m (foot sole was {_foot_sole_z:.4f}m below ground)")

    hold_jpos  = np.tile(stand_jpos[None],  (hold_frames, 1))
    hold_bpos  = np.tile(stand_bpos[None],  (hold_frames, 1, 1))
    hold_bquat = np.tile(stand_bquat[None], (hold_frames, 1, 1))

    t_lin = np.linspace(0.0, 1.0, trans_frames, endpoint=False, dtype=np.float64)
    t_q   = _quintic(t_lin).astype(np.float32)

    trans_jpos = (stand_jpos[None] * (1 - t_q[:, None]) + joint_pos_orig[0][None] * t_q[:, None]).astype(np.float32)
    trans_bpos = (stand_bpos[None] * (1 - t_q[:, None, None]) + body_pos_orig[0][None] * t_q[:, None, None]).astype(np.float32)
    trans_bquat = np.zeros((trans_frames, B, 4), dtype=np.float32)
    for b in range(B):
        trans_bquat[:, b, :] = _quat_slerp_batch(stand_bquat[b], body_quat_orig[0, b], t_q)

    parts_jpos  = [hold_jpos,  trans_jpos,  joint_pos_orig]
    parts_bpos  = [hold_bpos,  trans_bpos,  body_pos_orig]
    parts_bquat = [hold_bquat, trans_bquat, body_quat_orig]

    if tail_trans_frames > 0:
        last_quat = body_quat_orig[-1, 0]
        w, x, y, z = last_quat
        last_yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        _, tail_stand_bpos, tail_stand_bquat, _tail_foot_sole_z = _build_standing_pose_fk(
            npz_body_names, joint_names, stand_height, mjcf_path, yaw=last_yaw,
            standing_joint_pos_map=stand_map)
        xy_offset_tail = body_pos_orig[-1, 0, :2] - tail_stand_bpos[0, :2]
        tail_stand_bpos[:, :2] += xy_offset_tail

        # Z-align tail standing pose: foot soles on ground
        if _tail_foot_sole_z < 0:
            tail_stand_bpos[:, 2] += -_tail_foot_sole_z

        tt_lin = np.linspace(0.0, 1.0, tail_trans_frames, endpoint=False, dtype=np.float64)
        tt_q   = _quintic(tt_lin).astype(np.float32)

        tail_trans_jpos = (joint_pos_orig[-1][None] * (1 - tt_q[:, None]) + stand_jpos[None] * tt_q[:, None]).astype(np.float32)
        tail_trans_bpos = (body_pos_orig[-1][None] * (1 - tt_q[:, None, None]) + tail_stand_bpos[None] * tt_q[:, None, None]).astype(np.float32)
        tail_trans_bquat = np.zeros((tail_trans_frames, B, 4), dtype=np.float32)
        for b in range(B):
            tail_trans_bquat[:, b, :] = _quat_slerp_batch(body_quat_orig[-1, b], tail_stand_bquat[b], tt_q)

        parts_jpos.append(tail_trans_jpos)
        parts_bpos.append(tail_trans_bpos)
        parts_bquat.append(tail_trans_bquat)

        if tail_hold_frames > 0:
            parts_jpos.append(np.tile(stand_jpos[None],        (tail_hold_frames, 1)))
            parts_bpos.append(np.tile(tail_stand_bpos[None],   (tail_hold_frames, 1, 1)))
            parts_bquat.append(np.tile(tail_stand_bquat[None], (tail_hold_frames, 1, 1)))

    full_jpos  = np.concatenate(parts_jpos,  axis=0)
    full_bpos  = np.concatenate(parts_bpos,  axis=0)
    full_bquat = np.concatenate(parts_bquat, axis=0)

    full_jvel, full_blinvel, full_bangvel = _recompute_velocities_segmented(
        full_jpos, full_bpos, full_bquat, fps,
        hold_frames=hold_frames, trans_frames=trans_frames,
        tail_trans_frames=tail_trans_frames, tail_hold_frames=tail_hold_frames,
    )

    np.savez(
        output_npz,
        fps=data["fps"],
        joint_names=data["joint_names"],
        body_names=data["body_names"],
        joint_pos=full_jpos,
        joint_vel=full_jvel,
        body_pos_w=full_bpos,
        body_quat_w=full_bquat,
        body_lin_vel_w=full_blinvel,
        body_ang_vel_w=full_bangvel,
    )
    print(f"[add_transition] saved → {output_npz}")


# ---------------------------------------------------------------------------
# Recompute body velocities for an existing NPZ (from recompute_npz_body_velocities.py)
# ---------------------------------------------------------------------------

def _recompute_body_velocities_file(path: Path, smooth_window: int = 11, smooth_poly: int = 3) -> None:
    with np.load(path, allow_pickle=True) as d:
        payload = {k: d[k] for k in d.files}

    fps = float(np.asarray(payload["fps"]).item())
    body_pos_w  = payload["body_pos_w"].astype(np.float32)
    body_quat_w = payload["body_quat_w"].astype(np.float32)

    dt = 1.0 / fps
    num_frames = body_pos_w.shape[0]

    def _smooth(arr):
        if num_frames < smooth_window:
            return arr.astype(np.float32)
        return savgol_filter(arr.astype(np.float32), window_length=smooth_window,
                             polyorder=smooth_poly, axis=0)

    def _qmul(a, b):
        aw,ax,ay,az = a[...,0],a[...,1],a[...,2],a[...,3]
        bw,bx,by,bz = b[...,0],b[...,1],b[...,2],b[...,3]
        return np.stack([aw*bw-ax*bx-ay*by-az*bz,
                         aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx,
                         aw*bz+ax*by-ay*bx+az*bw], axis=-1)

    def _qconj(q):
        c = q.copy(); c[...,1:] *= -1; return c

    def _axis_angle(q):
        sin_half = np.linalg.norm(q[...,1:], axis=-1, keepdims=True)
        angle = 2.0 * np.arctan2(sin_half, q[...,:1])
        safe = np.where(sin_half > 1e-8, sin_half, np.ones_like(sin_half))
        return (angle * q[...,1:] / safe).astype(np.float32)

    def _ensure_cont(q):
        q = q.copy()
        for t in range(1, q.shape[0]):
            dot = np.sum(q[t] * q[t-1], axis=-1, keepdims=True)
            q[t] = np.where(dot < 0, -q[t], q[t])
        return q

    norm = np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
    body_quat_w = body_quat_w / np.clip(norm, 1e-8, None)
    body_quat_w = _ensure_cont(body_quat_w)

    pos_smooth = _smooth(body_pos_w)
    if num_frames == 1:
        body_lin_vel_w = np.zeros_like(pos_smooth)
    else:
        body_lin_vel_w = np.empty_like(pos_smooth)
        body_lin_vel_w[1:-1] = (pos_smooth[2:] - pos_smooth[:-2]) / (2.0 * dt)
        body_lin_vel_w[0]  = (pos_smooth[1]  - pos_smooth[0])  / dt
        body_lin_vel_w[-1] = (pos_smooth[-1] - pos_smooth[-2]) / dt

    if num_frames == 1:
        body_ang_vel_w = np.zeros((1, body_quat_w.shape[1], 3), dtype=np.float32)
    else:
        body_ang_vel_w = np.zeros((num_frames, body_quat_w.shape[1], 3), dtype=np.float32)
        q_rel_mid = _qmul(body_quat_w[2:], _qconj(body_quat_w[:-2]))
        body_ang_vel_w[1:-1] = _axis_angle(q_rel_mid) / (2.0 * dt)
        body_ang_vel_w[0]  = (_axis_angle(_qmul(body_quat_w[1:2],  _qconj(body_quat_w[0:1])))  / dt)[0]
        body_ang_vel_w[-1] = (_axis_angle(_qmul(body_quat_w[-1:],  _qconj(body_quat_w[-2:-1]))) / dt)[0]
        body_ang_vel_w = _smooth(body_ang_vel_w)

    payload["body_lin_vel_w"] = body_lin_vel_w
    payload["body_ang_vel_w"] = body_ang_vel_w

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + ".", suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    np.savez(tmp_path, **payload)
    tmp_path.replace(path)
    print(f"[recompute_vel] patched → {path}")


# ---------------------------------------------------------------------------
# Core single-file convert + save
# ---------------------------------------------------------------------------

def convert_single(
    input_file: str,
    output_name: str,
    input_fps: int,
    output_fps: int,
    robot: str,
    mjcf_path: str,
    frame_range: tuple[int, int] | None,
) -> None:
    joint_names = ROBOT_JOINT_NAMES[robot]
    body_names  = ROBOT_BODY_NAMES[robot]

    arrays = load_csv(
        csv_path=input_file,
        input_fps=input_fps,
        output_fps=output_fps,
        joint_names=joint_names,
        frame_range=frame_range,
    )
    body_pos_w, body_quat_w = run_fk(
        base_pos=arrays["base_pos"],
        base_quat=arrays["base_quat"],
        joint_pos=arrays["joint_pos"],
        joint_names=joint_names,
        body_names=body_names,
        mjcf_path=mjcf_path,
    )
    joint_vel, body_lin_vel_w, body_ang_vel_w = recompute_velocities_raw(
        joint_pos=arrays["joint_pos"],
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        fps=float(output_fps),
    )
    report_velocity_anomalies(joint_vel, body_lin_vel_w, body_ang_vel_w, joint_names, output_fps, input_file)

    out_path = output_name if output_name.endswith(".npz") else output_name + ".npz"
    np.savez(
        out_path,
        fps=np.array(output_fps),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        joint_pos=arrays["joint_pos"],
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"[csv_to_npz] saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CSV → NPZ pipeline: convert, add transition, recompute velocities."
    )

    # ── Input / output ──────────────────────────────────────────────────────
    parser.add_argument("--input_file",  default=None, help="Input CSV (single-file mode)")
    parser.add_argument("--output_name", default=None, help="Output NPZ path without .npz (single-file mode)")
    parser.add_argument("--batch_dir",   default=None, help="Directory of CSVs (batch mode)")
    parser.add_argument("--raw_output_dir", default=None,
                        help="Where to write raw NPZ files in batch mode (default: <batch_dir>/../npz_pro_raw)")
    parser.add_argument("--output_dir",  default=None,
                        help="Where to write final NPZ files in batch mode (default: <batch_dir>/../npz_pro)")

    # ── Robot / MJCF ────────────────────────────────────────────────────────
    parser.add_argument("--robot",  default="c1", choices=list(ROBOT_MJCF.keys()))
    parser.add_argument("--mjcf",   default=None, help="Override MJCF path")
    parser.add_argument("--input_fps",  type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"))

    # ── Batch control ───────────────────────────────────────────────────────
    parser.add_argument("--force", action="store_true",
                        help="Re-generate all NPZ even if they already exist")

    # ── Transition ──────────────────────────────────────────────────────────
    parser.add_argument("--add_transition", action="store_true",
                        help="Add standing hold + smooth transition after conversion")
    parser.add_argument("--hold_duration",       type=float, default=1.0)
    parser.add_argument("--trans_duration",      type=float, default=1.0)
    parser.add_argument("--tail_trans_duration", type=float, default=2.5)
    parser.add_argument("--tail_hold_duration",  type=float, default=1.0)
    parser.add_argument("--stand_height",        type=float, default=0.87)
    parser.add_argument("--head_skip_frames",    type=int,   default=-1)
    parser.add_argument("--tail_skip_frames",    type=int,   default=-1)
    parser.add_argument("--skip_threshold",      type=float, default=1.5)

    # ── Recompute velocities ─────────────────────────────────────────────────
    parser.add_argument("--recompute_velocities", action="store_true",
                        help="Recompute body velocities for all NPZ in output_dir after batch")

    # ── Compatibility flags (ignored) ────────────────────────────────────────
    parser.add_argument("--local",    action="store_true")
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()

    mjcf_path = args.mjcf or ROBOT_MJCF[args.robot]

    # ── Batch mode ───────────────────────────────────────────────────────────
    if args.batch_dir is not None:
        batch_dir = Path(args.batch_dir)
        root = batch_dir.parent

        raw_dir = Path(args.raw_output_dir) if args.raw_output_dir else root / "npz_pro_raw"
        out_dir = Path(args.output_dir)     if args.output_dir     else root / "npz_pro"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(batch_dir.glob("*.csv"))
        if not csv_files:
            print(f"[batch] No CSV files found in {batch_dir}")
            sys.exit(1)

        print(f"=== Phase 1: CSV → NPZ (+ transition) ===")
        for csv in csv_files:
            name = csv.stem
            raw_npz        = raw_dir / f"{name}.npz"
            transition_npz = out_dir  / f"{name}_with_transition.npz"

            # Convert CSV → raw NPZ
            if not raw_npz.exists() or args.force:
                print(f"[CONVERT] {name}")
                convert_single(
                    input_file=str(csv),
                    output_name=str(raw_npz.with_suffix("")),
                    input_fps=args.input_fps,
                    output_fps=args.output_fps,
                    robot=args.robot,
                    mjcf_path=mjcf_path,
                    frame_range=tuple(args.frame_range) if args.frame_range else None,
                )
            else:
                print(f"[SKIP] {name} raw npz already exists")

            # Add transition
            if not args.add_transition:
                continue
            if transition_npz.exists() and not args.force:
                print(f"[SKIP] {name}_with_transition.npz already exists")
                continue
            print(f"[TRANSITION] {name}")
            add_transition_to_npz(
                raw_npz=str(raw_npz),
                output_npz=str(transition_npz),
                hold_duration=args.hold_duration,
                trans_duration=args.trans_duration,
                tail_trans_duration=args.tail_trans_duration,
                tail_hold_duration=args.tail_hold_duration,
                stand_height=args.stand_height,
                head_skip_frames=args.head_skip_frames,
                tail_skip_frames=args.tail_skip_frames,
                skip_threshold=args.skip_threshold,
                mjcf_path=mjcf_path,
            )

        print(f"\n=== Done ===")
        n_trans = len(list(out_dir.glob("*_with_transition.npz")))
        print(f"Transitioned NPZ files: {n_trans}")

        # Recompute body velocities for all NPZ in output_dir
        if args.recompute_velocities:
            print(f"\n=== Phase 2: Recompute body velocities in {out_dir} ===")
            npz_files = sorted(out_dir.glob("*.npz"))
            for p in npz_files:
                _recompute_body_velocities_file(p)
        return

    # ── Single-file mode ─────────────────────────────────────────────────────
    if args.input_file is None or args.output_name is None:
        parser.error("Provide --input_file and --output_name (single-file mode) or --batch_dir (batch mode).")

    raw_out = args.output_name
    convert_single(
        input_file=args.input_file,
        output_name=raw_out,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        robot=args.robot,
        mjcf_path=mjcf_path,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
    )

    if args.add_transition:
        raw_npz = raw_out if raw_out.endswith(".npz") else raw_out + ".npz"
        base = raw_out[:-4] if raw_out.endswith(".npz") else raw_out
        transition_npz = base + "_with_transition.npz"
        add_transition_to_npz(
            raw_npz=raw_npz,
            output_npz=transition_npz,
            hold_duration=args.hold_duration,
            trans_duration=args.trans_duration,
            tail_trans_duration=args.tail_trans_duration,
            tail_hold_duration=args.tail_hold_duration,
            stand_height=args.stand_height,
            head_skip_frames=args.head_skip_frames,
            tail_skip_frames=args.tail_skip_frames,
            skip_threshold=args.skip_threshold,
            mjcf_path=mjcf_path,
        )


if __name__ == "__main__":
    main()
