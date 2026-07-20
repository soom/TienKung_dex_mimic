"""PKL → NPZ converter (no Isaac Sim required).

Supports two PKL formats:

1. Standard pickle (Pro): fps, root_pos (T,3), root_rot (T,4 xyzw), dof_pos (T,27), local_body_pos (T,28,3)
2. Joblib pickle (Dex):  {motion_name: {dof (T,29), root_trans_offset (T,3), root_rot (T,4 xyzw), fps}}

Pipeline mirrors csv_to_npz_mujoco.py:
  load PKL → interpolate → reorder joints → FK → velocities → [transition] → NPZ

── Single-file mode ───────────────────────────────────────────────────────────
    python scripts/pkl_to_npz.py \\
        --input_file dataset/pkl/foo.pkl \\
        --output_name dataset/npz_pro_raw/foo

── Single-file + transition ───────────────────────────────────────────────────
    python scripts/pkl_to_npz.py \\
        --input_file dataset/pkl/foo.pkl \\
        --output_name dataset/npz_pro/foo_with_transition \\
        --add_transition

── Batch mode ─────────────────────────────────────────────────────────────────
    python scripts/pkl_to_npz.py \\
        --batch_dir dataset/pkl/ \\
        --raw_output_dir dataset/npz_pro_raw/ \\
        --output_dir dataset/npz_pro/ \\
        --add_transition \\
        --recompute_velocities
"""

import argparse
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np

# Import shared pipeline functions from csv_to_npz_mujoco
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
from csv_to_npz_mujoco import (
    ROBOT_BODY_NAMES,
    ROBOT_JOINT_NAMES,
    ROBOT_MJCF,
    add_transition_to_npz,
    recompute_velocities_raw,
    report_velocity_anomalies,
    run_fk,
    _recompute_body_velocities_file,
)

# ---------------------------------------------------------------------------
# PKL joint order (kinematic chain, matches CSV column order)
# Verified by FK comparison against local_body_pos in PKL.
# ---------------------------------------------------------------------------
PKL_JOINT_ORDER = [
    "body_yaw_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",    "wrist_roll_r_joint",
    "hip_roll_l_joint",  "hip_pitch_l_joint",  "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_roll_r_joint",  "hip_pitch_r_joint",  "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
]

# Dex joblib PKL joint order (29 joints, full kinematic chain including fixed joints).
# From xMimic mujoco_edit_pkl.py JOINT_NAMES_TG.
DEX_PKL_JOINT_ORDER = [
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
]

# Walker C1 PKL order (29 DOF, matching the C1 dataset exports).
C1_PKL_JOINT_ORDER = [
    "L_hip_pitch_joint", "L_hip_roll_joint", "L_hip_yaw_joint",
    "L_knee_pitch_joint", "L_ankle_pitch_joint", "L_ankle_roll_joint",
    "R_hip_pitch_joint", "R_hip_roll_joint", "R_hip_yaw_joint",
    "R_knee_pitch_joint", "R_ankle_pitch_joint", "R_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
    "L_elbow_pitch_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
    "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
    "R_elbow_pitch_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint",
]


# ---------------------------------------------------------------------------
# PKL loader + interpolation
# ---------------------------------------------------------------------------

def _slerp_batch(quats: np.ndarray, idx0: np.ndarray, idx1: np.ndarray,
                 blend: np.ndarray) -> np.ndarray:
    q0 = quats[idx0]
    q1 = quats[idx1]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(0.0, 1.0)
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


def _is_joblib_pkl(pkl_path: str) -> bool:
    """Detect whether a PKL file is joblib format by trying joblib.load first."""
    try:
        with open(pkl_path, "rb") as f:
            header = f.read(4)
        # Joblib files start with a specific magic header
        # Standard pickle starts with b'\x80' (protocol >= 2)
        if header[:1] == b"\x80":
            return False
        # Try joblib load
        data = joblib.load(pkl_path)
        # Joblib dex format: top-level is a dict with one key → sub-dict
        key = list(data.keys())[0]
        return isinstance(data[key], dict) and "dof" in data[key]
    except Exception:
        return False


def load_pkl(pkl_path: str, output_fps: int, robot: str) -> dict[str, np.ndarray]:
    """Load PKL (pickle or joblib), interpolate to output_fps, reorder joints."""
    is_joblib = _is_joblib_pkl(pkl_path)

    if is_joblib:
        # ── Joblib format (Dex) ──────────────────────────────────────────
        data = joblib.load(pkl_path)
        key = list(data.keys())[0]
        motion = data[key]

        input_fps = float(motion["fps"])
        dof_raw_in = np.asarray(motion["dof"], dtype=np.float64)  # (T, 29)
        root_pos_in = np.asarray(motion["root_trans_offset"], dtype=np.float64)
        root_rot_in = np.asarray(motion["root_rot"], dtype=np.float64)

        T_in = dof_raw_in.shape[0]
        duration = (T_in - 1) / input_fps

        # Reorder: DEX_PKL_JOINT_ORDER (29) → ROBOT_JOINT_NAMES[robot] (29)
        target_names = ROBOT_JOINT_NAMES[robot]
        source_order = C1_PKL_JOINT_ORDER if robot == "c1" else DEX_PKL_JOINT_ORDER
        pkl_idx = {n: i for i, n in enumerate(source_order)}
        missing = [n for n in target_names if n not in pkl_idx]
        if missing:
            raise KeyError(f"PKL (joblib) missing joints: {missing}")
        reorder = np.array([pkl_idx[n] for n in target_names], dtype=int)
        dof_pos_in = dof_raw_in[:, reorder]

        print(f"[pkl_to_npz] joblib {pkl_path}  frames={T_in}  fps={input_fps:.2f}  duration={duration:.2f}s")
    else:
        # ── Standard pickle format (Pro) ─────────────────────────────────
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        input_fps = float(data["fps"])
        root_pos_in = np.asarray(data["root_pos"], dtype=np.float64)
        root_rot_in = np.asarray(data["root_rot"], dtype=np.float64)
        dof_raw_in = np.asarray(data["dof_pos"], dtype=np.float64)

        T_in = root_pos_in.shape[0]
        duration = (T_in - 1) / input_fps

        target_names = ROBOT_JOINT_NAMES[robot]
        # Auto-detect joint order by DOF count: 29 → Dex, 27 → Pro
        dof_count = dof_raw_in.shape[1]
        if dof_count == 29:
            pkl_order = C1_PKL_JOINT_ORDER if robot == "c1" else DEX_PKL_JOINT_ORDER
        elif dof_count == 27:
            pkl_order = PKL_JOINT_ORDER
        else:
            raise KeyError(f"PKL has {dof_count} DOF — expected 27 or 29")
        pkl_idx = {n: i for i, n in enumerate(pkl_order)}
        missing = [n for n in target_names if n not in pkl_idx]
        if missing:
            raise KeyError(f"PKL missing joints: {missing}")
        reorder = np.array([pkl_idx[n] for n in target_names], dtype=int)
        dof_pos_in = dof_raw_in[:, reorder]

        print(f"[pkl_to_npz] pickle {pkl_path}  frames={T_in}  fps={input_fps:.2f}  dof={dof_count}  duration={duration:.2f}s")

    # Convert root_rot xyzw → wxyz (MuJoCo convention)
    quat_wxyz_in = root_rot_in[:, [3, 0, 1, 2]]

    # Interpolate to output_fps
    times = np.arange(0, duration, 1.0 / output_fps, dtype=np.float64)
    T_out = len(times)
    phase = times / duration
    idx0 = np.floor(phase * (T_in - 1)).astype(int)
    idx1 = np.minimum(idx0 + 1, T_in - 1)
    blend = (phase * (T_in - 1) - idx0).astype(np.float32)

    base_pos = (root_pos_in[idx0] * (1 - blend[:, None]) + root_pos_in[idx1] * blend[:, None]).astype(np.float32)
    joint_pos = (dof_pos_in[idx0] * (1 - blend[:, None]) + dof_pos_in[idx1] * blend[:, None]).astype(np.float32)
    base_quat = _slerp_batch(quat_wxyz_in.astype(np.float32), idx0, idx1, blend)

    print(f"[pkl_to_npz] interpolated → frames={T_out}  fps={output_fps}")
    return {
        "base_pos":  base_pos,
        "base_quat": base_quat,
        "joint_pos": joint_pos,
    }


# ---------------------------------------------------------------------------
# Core single-file convert + save
# ---------------------------------------------------------------------------

def convert_single(
    input_file: str,
    output_name: str,
    output_fps: int,
    robot: str,
    mjcf_path: str,
) -> None:
    joint_names = ROBOT_JOINT_NAMES[robot]
    body_names  = ROBOT_BODY_NAMES[robot]

    arrays = load_pkl(input_file, output_fps, robot)

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
    print(f"[pkl_to_npz] saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PKL → NPZ pipeline: convert, add transition, recompute velocities."
    )

    parser.add_argument("--input_file",  default=None)
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--batch_dir",   default=None)
    parser.add_argument("--raw_output_dir", default=None)
    parser.add_argument("--output_dir",  default=None)

    parser.add_argument("--robot",      default="c1", choices=list(ROBOT_MJCF.keys()), help="Robot type (c1 or dex_evt)")
    parser.add_argument("--mjcf",       default=None)
    parser.add_argument("--output_fps", type=int, default=50)

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--add_transition",      action="store_true")
    parser.add_argument("--hold_duration",        type=float, default=1.0)
    parser.add_argument("--trans_duration",       type=float, default=1.0)
    parser.add_argument("--tail_trans_duration",  type=float, default=2.5)
    parser.add_argument("--tail_hold_duration",   type=float, default=1.0)
    parser.add_argument("--stand_height",         type=float, default=0.87)
    parser.add_argument("--head_skip_frames",     type=int,   default=-1)
    parser.add_argument("--tail_skip_frames",     type=int,   default=-1)
    parser.add_argument("--skip_threshold",       type=float, default=1.5)

    parser.add_argument("--recompute_velocities", action="store_true")

    args = parser.parse_args()
    mjcf_path = args.mjcf or ROBOT_MJCF[args.robot]

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch_dir is not None:
        batch_dir = Path(args.batch_dir)
        pkl_files = sorted(batch_dir.glob("*.pkl"))
        if not pkl_files:
            print(f"[batch] No PKL files found in {batch_dir}")
            sys.exit(1)

        if args.add_transition:
            # Two-phase: raw → transition
            root = batch_dir.parent
            raw_dir = Path(args.raw_output_dir) if args.raw_output_dir else root / "npz_pro_raw"
            out_dir = Path(args.output_dir)     if args.output_dir     else root / "npz_pro"
            raw_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"=== Phase 1: PKL → NPZ (+ transition) ===")
            for pkl in pkl_files:
                name = pkl.stem
                raw_npz        = raw_dir / f"{name}.npz"
                transition_npz = out_dir  / f"{name}_with_transition.npz"

                if not raw_npz.exists() or args.force:
                    print(f"[CONVERT] {name}")
                    convert_single(str(pkl), str(raw_npz.with_suffix("")),
                                   args.output_fps, args.robot, mjcf_path)
                else:
                    print(f"[SKIP] {name} raw npz already exists")

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

            if args.recompute_velocities:
                print(f"\n=== Phase 2: Recompute body velocities in {out_dir} ===")
                for p in sorted(out_dir.glob("*.npz")):
                    _recompute_body_velocities_file(p)
        else:
            # Direct mode: PKL → NPZ to output_dir
            out_dir = Path(args.output_dir) if args.output_dir else batch_dir.parent / "npz_dex"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"=== PKL → NPZ (direct, no transition) ===")
            print(f"  Output: {out_dir}")
            for pkl in pkl_files:
                name = pkl.stem
                out_npz = out_dir / f"{name}.npz"
                if out_npz.exists() and not args.force:
                    print(f"[SKIP] {name}.npz already exists")
                    continue
                print(f"[CONVERT] {name}")
                convert_single(str(pkl), str(out_npz.with_suffix("")),
                               args.output_fps, args.robot, mjcf_path)
            print(f"\n=== Done ===")

            if args.recompute_velocities:
                print(f"\n=== Recompute body velocities in {out_dir} ===")
                for p in sorted(out_dir.glob("*.npz")):
                    _recompute_body_velocities_file(p)
        return

    # ── Single-file mode ──────────────────────────────────────────────────────
    if args.input_file is None or args.output_name is None:
        parser.error("Provide --input_file and --output_name, or --batch_dir.")

    convert_single(args.input_file, args.output_name,
                   args.output_fps, args.robot, mjcf_path)

    if args.add_transition:
        raw_npz = args.output_name if args.output_name.endswith(".npz") else args.output_name + ".npz"
        base = args.output_name[:-4] if args.output_name.endswith(".npz") else args.output_name
        add_transition_to_npz(
            raw_npz=raw_npz,
            output_npz=base + "_with_transition.npz",
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
