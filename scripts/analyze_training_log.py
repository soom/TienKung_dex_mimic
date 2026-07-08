#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = WORKSPACE_ROOT / "logs" / "rsl_rl"
EVENT_GLOB = "events.out.tfevents.*"
DEFAULT_MOTION_FILE = WORKSPACE_ROOT / "motions" / "s3_rom3_stageii_with_transition.npz"


DEFAULT_TAGS = [
    "─ HEALTH ─────────────────────",
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Policy/mean_noise_std",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "─ CURRICULUM ──────────────────",
    "Metrics/motion/curriculum_stage",
    "Metrics/motion/raw_action_rate",
    "Metrics/motion/blacklisted_clip_count",
    "─ TERMINATION ─────────────────",
    "Episode_Termination/time_out",
    "Episode_Termination/anchor_pos",
    "Episode_Termination/anchor_ori",
    "Episode_Termination/anchor_ori_full",
    "Episode_Termination/ee_body_pos",
    "─ TRACKING ──────────────────────",
    "Metrics/motion/error_anchor_pos",
    "Episode_Reward/motion_global_anchor_pos",
    "Episode_Reward/anchor_planar_drift",
    "Episode_Reward/anchor_planar_drift_ungated",
    "Episode_Reward/anchor_static_planar_vel",
    "Metrics/motion/error_anchor_rot",
    "Episode_Reward/motion_global_anchor_ori",
    "Metrics/motion/error_anchor_lin_vel",
    "Metrics/motion/error_anchor_ang_vel",
    "Metrics/motion/error_body_pos",
    "Episode_Reward/motion_body_pos",
    "Metrics/motion/error_body_rot",
    "Episode_Reward/motion_body_ori",
    "Metrics/motion/error_body_lin_vel",
    "Episode_Reward/motion_body_lin_vel",
    "Metrics/motion/error_body_ang_vel",
    "Episode_Reward/motion_body_ang_vel",
    "Episode_Reward/motion_torso_ori",
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_joint_vel",
    "Episode_Reward/motion_joint_pos",
    "Episode_Reward/motion_joint_vel",
    "─ PENALTIES ───────────────────",
    "Episode_Reward/upper_body_lin_vel_penalty",
    "Episode_Reward/upper_body_ang_vel_penalty",
    "Episode_Reward/foot_slip",
    "Episode_Reward/undesired_contacts",
    "Episode_Reward/wrist_contact_penalty",
    "Episode_Reward/action_rate_l2",
]

PERCENTILE_POINTS = [0.0, 0.5, 1.0]

REWARD_TARGETS = {
    # ── Tracking ────────────────────────────────────────────────────────
    "Episode_Reward/motion_global_anchor_pos":   3.0,
    "Episode_Reward/motion_global_anchor_ori":   2.0,
    "Episode_Reward/anchor_planar_drift":        -8.0,
    "Episode_Reward/anchor_planar_drift_ungated": -5.0,
    "Episode_Reward/anchor_static_planar_vel":   -4.0,
    "Episode_Reward/motion_body_pos":            1.0,
    "Episode_Reward/motion_body_ori":            0.5,
    "Episode_Reward/motion_torso_ori":           1.5,
    "Episode_Reward/motion_body_lin_vel":        1.5,
    "Episode_Reward/motion_body_ang_vel":        0.0,
    "Episode_Reward/upper_body_lin_vel_penalty": -0.25,
    "Episode_Reward/upper_body_ang_vel_penalty": -0.3,
    "Episode_Reward/motion_joint_pos":           5.0,
    "Episode_Reward/motion_joint_vel":           2.5,
    "Episode_Reward/action_rate_l2":            -0.75,
    "Episode_Reward/foot_slip":                  -5.0,
    "Episode_Reward/undesired_contacts":         -0.1,
    "Episode_Reward/wrist_contact_penalty":      -0.5,
}

ZERO_WATCH_TAGS = [
    "Episode_Reward/motion_body_ang_vel",
]

# Per-group velocity reward back-calculation config (Simple config uses full-body joint vel only).
JOINT_VEL_GROUPS: list = []

# Short display names for long metric tags
TAG_SHORT_NAMES: dict[str, str] = {
    "Episode_Reward/anchor_planar_drift":        "anc_drift",
    "Episode_Reward/anchor_planar_drift_ungated": "anc_drift_ug",
    "Episode_Reward/anchor_static_planar_vel":   "anc_sv",
    "Episode_Reward/wrist_contact_penalty":      "wrist_con",
    "motion_global_anchor_pos":   "anc_pos",
    "motion_global_anchor_ori":   "anc_ori",
    "error_anchor_pos":           "e_anc_pos",
    "error_anchor_rot":           "e_anc_rot",
    "error_anchor_lin_vel":       "e_anc_lv",
    "error_anchor_ang_vel":       "e_anc_av",
    "error_body_pos":             "e_body_pos",
    "error_body_rot":             "e_body_rot",
    "error_body_lin_vel":         "e_body_lv",
    "error_body_ang_vel":         "e_body_av",
    "error_joint_pos":            "e_jnt_pos",
    "error_joint_vel":            "e_jnt_vel",
    "motion_body_lin_vel":        "body_lv",
    "motion_body_ang_vel":        "body_av",
    "upper_body_lin_vel_penalty": "ub_lv_pen",
    "upper_body_ang_vel_penalty": "ub_av_pen",
    "motion_torso_ori":           "torso_ori",
    "motion_joint_pos":           "jnt_pos",
    "motion_joint_vel":           "jnt_vel",
    "foot_slip":                  "ft_slip",
    "undesired_contacts":         "und_cont",
    "curriculum_stage":           "cur_stage",
    "raw_action_rate":            "act_rate",
    "blacklisted_clip_count":     "blk_clips",
    "full_episode_length":        "full_ep",
    "mean_noise_std":             "noise_std",
    "mean_episode_length":        "ep_len",
    "mean_reward":                "reward",
    "value_function":             "val_loss",
}


@dataclass
class SeriesSummary:
    tag: str
    first_step: int
    last_step: int
    first_value: float
    last_value: float
    best_value: float
    best_step: int
    prev_window_mean: float
    last_window_mean: float

    @property
    def window_delta(self) -> float:
        return self.last_window_mean - self.prev_window_mean


@dataclass
class SegmentBucketSummary:
    index: int
    sample_count: int
    start_ratio: float
    end_ratio: float
    mean_episode_length: float
    mean_time_out: float
    mean_anchor_pos: float
    mean_anchor_ori: float
    mean_ee_body_pos: float
    mean_undesired_contacts: float
    failure_score: float = 0.0


def percentile_indices(length: int) -> list[int]:
    if length <= 0:
        return []
    return [min(length - 1, max(0, round((length - 1) * point))) for point in PERCENTILE_POINTS]


def build_percentile_table(
    accumulator: event_accumulator.EventAccumulator,
    tags: list[str],
    drop_first: int,
    mini: bool = False,
    window: int = 20,
) -> tuple[list[str], list[list[str]]]:
    series_map: dict[str, list[tuple[int, float]]] = {}
    for tag in tags:
        if tag.startswith("─"):
            continue
        try:
            pairs = scalar_values(accumulator, tag, drop_first)
        except KeyError:
            continue  # tag not in this log file
        if pairs:
            series_map[tag] = pairs

    if not series_map:
        return [], []

    reference_tag = max(series_map, key=lambda key: len(series_map[key]))
    reference_pairs = series_map[reference_tag]
    ref_indices = percentile_indices(len(reference_pairs))

    if mini:
        base_headers = ["metric", "1.0", "%"]
    else:
        base_headers = ["metric"]
        for point, ref_index in zip(PERCENTILE_POINTS, ref_indices):
            step = reference_pairs[ref_index][0]
            base_headers.append(f"{step}")
        # base_headers.append("avg")
        base_headers.append("%")

    rows: list[list[str]] = []
    group_break_indices: list[int] = []  # track where to insert separators
    for tag in tags:
        if tag.startswith("─"):  # group header
            label = tag.replace("─", "").strip()
            rows.append([f"── {label} " + "─" * 40])
            continue
        pairs = series_map.get(tag)
        if not pairs:
            continue
        idxs = percentile_indices(len(pairs))
        short = tag.split('/')[-1]
        short = TAG_SHORT_NAMES.get(tag, TAG_SHORT_NAMES.get(short, short))
        row = [short]
        if mini:
            row.append(f"{pairs[idxs[-1]][1]:.3f}")
        else:
            for idx in idxs:
                row.append(f"{pairs[idx][1]:.3f}")
        target = REWARD_TARGETS.get(tag)
        if target is not None and target != 0.0:
            current_value = pairs[-1][1]
            if current_value == 0.0:
                row.append("0%")
            elif target < 0:
                mae = current_value / target
                row.append(f"[{mae:.2f}]")
            else:
                current_pct = current_value / target * 100.0
                row.append(f"{current_pct:.0f}%")
        else:
            row.append("")
        rows.append(row)

    cols = 3
    headers = base_headers * cols
    packed_rows: list[list[str]] = []
    empty_col = [""] * len(base_headers)

    # Build groups split by separators; pack each group into 3-col rows
    groups: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for r in rows:
        if r[0].startswith("──"):
            if cur: groups.append(cur); cur = []
            groups.append([r])
        else:
            cur.append(r)
    if cur: groups.append(cur)

    for g in groups:
        if g[0][0].startswith("──"):
            # Section header — keep label, print_table handles formatting
            packed_rows.append(g[0])
            continue
        data = list(g)
        while len(data) % cols != 0:
            data.append(empty_col)
        for j in range(0, len(data), cols):
            r = data[j]
            for o in range(1, cols):
                r = r + data[j + o]
            packed_rows.append(r)

    return headers, packed_rows


def print_table(headers: list[str], rows: list[list[str]]):
    if not headers or not rows:
        return
    # Only compute widths from multi-column data rows, skip section headers
    data_rows = [r for r in rows if len(r) > 1]
    widths = [len(header) for header in headers]
    for row in data_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        if len(row) == 1:
            return row[0]
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    hdr_sep = "-+-".join("-" * width for width in widths)
    print(format_row(headers))
    print(hdr_sep)
    for row in rows:
        if len(row) == 1 and row[0].startswith("──"):
            # Section header: ──── LABEL ──── centered
            total = sum(widths) + 3 * (len(widths) - 1)
            label = row[0].replace("──", "").strip()
            side = (total - len(label) - 2) // 2
            print("─" * side + " " + label + " " + "─" * (total - side - len(label) - 2))
        else:
            print(format_row(row))


def load_events(event_file: Path) -> event_accumulator.EventAccumulator:
    accumulator = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def latest_event_file(search_root: Path) -> Path:
    candidates = [path for path in search_root.rglob(EVENT_GLOB) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event files found under: {search_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_event_file(target: Path | None) -> Path:
    if target is None:
        return latest_event_file(DEFAULT_LOG_ROOT)

    resolved = target.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        return latest_event_file(resolved)
    raise FileNotFoundError(f"Path does not exist: {target}")


def scalar_values(accumulator: event_accumulator.EventAccumulator, tag: str, drop_first: int) -> list[tuple[int, float]]:
    events = accumulator.Scalars(tag)
    pairs = [(int(item.step), float(item.value)) for item in events]
    if drop_first > 0:
        pairs = pairs[drop_first:]
    return pairs


def split_windows(values: list[float], window: int) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    usable_window = min(window, len(values))
    last_window = values[-usable_window:]
    prev_start = max(0, len(values) - 2 * usable_window)
    prev_window = values[prev_start : len(values) - usable_window]
    if not prev_window:
        prev_window = values[:usable_window]
    return prev_window, last_window


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def resolve_motion_info(motion_file: Path | None) -> tuple[int | None, float | None]:
    if motion_file is None:
        return None, None
    resolved = motion_file.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {motion_file}")
    with np.load(resolved) as motion:
        frame_count = int(motion["joint_pos"].shape[0])
        fps = float(np.asarray(motion["fps"]).item())
    return frame_count, fps


def align_series(reference_steps: list[int], pairs: list[tuple[int, float]]) -> list[float | None]:
    if not reference_steps:
        return []
    if not pairs:
        return [None] * len(reference_steps)

    aligned: list[float | None] = []
    pair_index = 0
    current_value: float | None = None
    for step in reference_steps:
        while pair_index < len(pairs) and pairs[pair_index][0] <= step:
            current_value = pairs[pair_index][1]
            pair_index += 1
        aligned.append(current_value)
    return aligned


def _normalize(value: float, lower: float, upper: float, *, invert: bool = False) -> float:
    if upper - lower <= 1e-9:
        return 0.0
    score = (value - lower) / (upper - lower)
    score = max(0.0, min(1.0, score))
    return 1.0 - score if invert else score


def format_segment_range(
    start_ratio: float,
    end_ratio: float,
    motion_frames: int | None,
    motion_fps: float | None,
) -> str:
    if motion_frames is not None and motion_fps is not None:
        start_frame = int(round(start_ratio * motion_frames))
        end_frame = int(round(end_ratio * motion_frames))
        return f"{start_frame / motion_fps:.1f}-{end_frame / motion_fps:.1f}s"
    return f"{start_ratio * 100.0:.0f}-{end_ratio * 100.0:.0f}%"


def build_segment_summary(
    accumulator: event_accumulator.EventAccumulator,
    drop_first: int,
    segment_count: int,
) -> list[SegmentBucketSummary]:
    sampling_pairs = scalar_values(accumulator, "Metrics/motion/sampling_top1_bin", drop_first)
    if not sampling_pairs:
        return []

    reference_steps = [step for step, _ in sampling_pairs]
    aligned_episode_length = align_series(
        reference_steps, scalar_values(accumulator, "Train/mean_episode_length", drop_first)
    )
    aligned_time_out = align_series(
        reference_steps, scalar_values(accumulator, "Episode_Termination/time_out", drop_first)
    )
    aligned_anchor_pos = align_series(
        reference_steps, scalar_values(accumulator, "Episode_Termination/anchor_pos", drop_first)
    )
    aligned_anchor_ori = align_series(
        reference_steps, scalar_values(accumulator, "Episode_Termination/anchor_ori", drop_first)
    )
    aligned_ee_body_pos = align_series(
        reference_steps, scalar_values(accumulator, "Episode_Termination/ee_body_pos", drop_first)
    )
    aligned_undesired_contacts = align_series(
        reference_steps, scalar_values(accumulator, "Episode_Reward/undesired_contacts", drop_first)
    )

    buckets: list[dict[str, list[float]]] = [
        {
            "episode_length": [],
            "time_out": [],
            "anchor_pos": [],
            "anchor_ori": [],
            "ee_body_pos": [],
            "undesired_contacts": [],
        }
        for _ in range(segment_count)
    ]

    for sample_index, (_, top1_bin) in enumerate(sampling_pairs):
        clipped = min(max(top1_bin, 0.0), 1.0 - 1e-9)
        bucket_index = min(segment_count - 1, int(clipped * segment_count))
        bucket = buckets[bucket_index]

        values = {
            "episode_length": aligned_episode_length[sample_index],
            "time_out": aligned_time_out[sample_index],
            "anchor_pos": aligned_anchor_pos[sample_index],
            "anchor_ori": aligned_anchor_ori[sample_index],
            "ee_body_pos": aligned_ee_body_pos[sample_index],
            "undesired_contacts": aligned_undesired_contacts[sample_index],
        }
        for key, value in values.items():
            if value is not None:
                bucket[key].append(value)

    summaries: list[SegmentBucketSummary] = []
    for bucket_index, bucket in enumerate(buckets):
        if not bucket["episode_length"]:
            continue
        summaries.append(
            SegmentBucketSummary(
                index=bucket_index,
                sample_count=len(bucket["episode_length"]),
                start_ratio=bucket_index / segment_count,
                end_ratio=(bucket_index + 1) / segment_count,
                mean_episode_length=mean(bucket["episode_length"]),
                mean_time_out=mean(bucket["time_out"]),
                mean_anchor_pos=mean(bucket["anchor_pos"]),
                mean_anchor_ori=mean(bucket["anchor_ori"]),
                mean_ee_body_pos=mean(bucket["ee_body_pos"]),
                mean_undesired_contacts=mean(bucket["undesired_contacts"]),
            )
        )

    if not summaries:
        return []

    episode_lengths = [item.mean_episode_length for item in summaries]
    time_outs = [item.mean_time_out for item in summaries]
    anchor_pos_terms = [item.mean_anchor_pos for item in summaries]
    anchor_ori_terms = [item.mean_anchor_ori for item in summaries]
    ee_body_terms = [item.mean_ee_body_pos for item in summaries]
    undesired_contacts = [item.mean_undesired_contacts for item in summaries]

    for item in summaries:
        short_episode = _normalize(item.mean_episode_length, min(episode_lengths), max(episode_lengths), invert=True)
        anchor_pos_fail = _normalize(item.mean_anchor_pos, min(anchor_pos_terms), max(anchor_pos_terms))
        anchor_ori_fail = _normalize(item.mean_anchor_ori, min(anchor_ori_terms), max(anchor_ori_terms))
        ee_body_fail = _normalize(item.mean_ee_body_pos, min(ee_body_terms), max(ee_body_terms))
        undesired_contact_fail = _normalize(
            item.mean_undesired_contacts, min(undesired_contacts), max(undesired_contacts)
        )
        timeout_relief = _normalize(item.mean_time_out, min(time_outs), max(time_outs))
        item.failure_score = (
            short_episode
            + anchor_pos_fail
            + ee_body_fail
            + 0.5 * anchor_ori_fail
            + 0.5 * undesired_contact_fail
            - 0.5 * timeout_relief
        )

    return summaries


def summarize_series(
    accumulator: event_accumulator.EventAccumulator,
    tag: str,
    window: int,
    drop_first: int,
) -> SeriesSummary | None:
    pairs = scalar_values(accumulator, tag, drop_first)
    if not pairs:
        return None

    steps = [step for step, _ in pairs]
    values = [value for _, value in pairs]
    prev_window, last_window = split_windows(values, window)

    if "error_" in tag:
        best_index = min(range(len(values)), key=values.__getitem__)
    else:
        best_index = max(range(len(values)), key=values.__getitem__)

    return SeriesSummary(
        tag=tag,
        first_step=steps[0],
        last_step=steps[-1],
        first_value=values[0],
        last_value=values[-1],
        best_value=values[best_index],
        best_step=steps[best_index],
        prev_window_mean=mean(prev_window),
        last_window_mean=mean(last_window),
    )


def classify_stage(last_episode_length: float, last_time_out: float, last_anchor_pos: float, last_noise_std: float) -> str:
    if last_noise_std > 1.5:
        return "diverging"
    if last_episode_length >= 400 and last_time_out >= 5.0 and last_anchor_pos <= 1.0:
        return "late-stable"
    if last_episode_length >= 120 and last_time_out > 0.0:
        return "mid-improving"
    return "early-unstable"


def plateau_note(summary_map: dict[str, SeriesSummary], plateau_tol: float) -> str:
    monitor_tags = [
        "Train/mean_episode_length",
        "Metrics/motion/error_body_pos",
        "Metrics/motion/error_body_rot",
        "Metrics/motion/error_body_lin_vel",
        "Metrics/motion/error_body_ang_vel",
        "Metrics/motion/error_anchor_pos",
        "Metrics/motion/error_joint_pos",
    ]
    active = [summary_map[tag] for tag in monitor_tags if tag in summary_map]
    if len(active) < 3:
        return "insufficient-data"

    stable_count = 0
    for item in active:
        reference = max(abs(item.prev_window_mean), 1e-6)
        if abs(item.window_delta) / reference < plateau_tol:
            stable_count += 1

    if stable_count >= 4:
        return "possible-plateau"
    return "still-changing"


def print_summary(summary: SeriesSummary):
    print(f"[{summary.tag}]")
    print(f"  first: step={summary.first_step} value={summary.first_value:.4f}")
    print(f"  last : step={summary.last_step} value={summary.last_value:.4f}")
    print(f"  best : step={summary.best_step} value={summary.best_value:.4f}")
    print(
        "  trend: "
        f"prev_window_mean={summary.prev_window_mean:.4f} "
        f"last_window_mean={summary.last_window_mean:.4f} "
        f"delta={summary.window_delta:+.4f}"
    )


def find_zero_reward_terms(summary_map: dict[str, SeriesSummary], tags: list[str], tol: float = 1e-8) -> list[str]:
    zero_terms: list[str] = []
    for tag in tags:
        summary = summary_map.get(tag)
        if summary is None:
            continue
        if (
            abs(summary.first_value) <= tol
            and abs(summary.last_value) <= tol
            and abs(summary.best_value) <= tol
            and abs(summary.prev_window_mean) <= tol
            and abs(summary.last_window_mean) <= tol
        ):
            zero_terms.append(tag.split("/")[-1])
    return zero_terms


def print_joint_vel_breakdown(summary_map: dict[str, SeriesSummary], e_jnt_vel: float, cur_stage: int = 0):
    """Estimate per-group contribution to error_joint_vel from per-group velocity rewards."""
    import math

    rows = []
    sum_sq = 0.0
    for name, tag, kernel, weight_or_schedule, std, n_joints in JOINT_VEL_GROUPS:
        summary = summary_map.get(tag)
        if summary is None:
            continue
        # Resolve weight: scalar or per-stage tuple
        if isinstance(weight_or_schedule, tuple):
            idx = min(cur_stage, len(weight_or_schedule) - 1)
            weight = weight_or_schedule[idx]
        else:
            weight = weight_or_schedule
        val = summary.last_value
        if kernel == "exp":
            # reward = weight * exp(-mse/std²)  →  mse = -std² * ln(reward/weight)
            norm = max(val / max(weight, 1e-9), 1e-9)
            per_joint_err = std * math.sqrt(max(-(math.log(norm)), 0.0))
        else:
            per_joint_err = val / weight if abs(weight) > 1e-9 else 0.0
        contrib = per_joint_err * math.sqrt(n_joints)
        sum_sq += n_joints * per_joint_err ** 2
        rows.append((name, per_joint_err, contrib))

    estimated_total = math.sqrt(sum_sq) if sum_sq > 0 else 0.0
    denom = estimated_total if estimated_total > 1e-6 else 1.0
    rows.sort(key=lambda r: r[2], reverse=True)

    parts = ", ".join(f"{name}={contrib:.1f}" for name, _, contrib in rows)
    print(f"  e_jnt_vel={e_jnt_vel:.2f} est={estimated_total:.2f}  {parts}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RSL-RL TensorBoard event logs.")
    parser.add_argument(
        "event_file",
        nargs="?",
        type=Path,
        help="Path to a TensorBoard event file or a run directory. If omitted, the latest event file under logs/rsl_rl is used.",
    )
    parser.add_argument("--window", type=int, default=20, help="Number of points per trailing trend window.")
    parser.add_argument(
        "--drop-first",
        type=int,
        default=1,
        help="Drop the first N scalar points from each tag. Useful after resume when the first point is noisy.",
    )
    parser.add_argument(
        "--plateau-tol",
        type=float,
        default=0.03,
        help="Relative trailing-window change threshold used by the coarse plateau heuristic.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=DEFAULT_TAGS,
        help="Scalar tags to summarize. Defaults to the main Tienkung2 Pro tracking metrics.",
    )
    parser.add_argument(
        "--segment-summary",
        action="store_true",
        help="Group training health by sampled motion segment using sampling_top1_bin.",
    )
    parser.add_argument("--segments", type=int, default=10, help="Number of motion segments for --segment-summary.")
    parser.add_argument(
        "--motion-file",
        type=Path,
        default=DEFAULT_MOTION_FILE if DEFAULT_MOTION_FILE.is_file() else None,
        help="Optional motion NPZ used to print segment ranges in seconds instead of percentages.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Compact table: only keep the 1.0(latest) column and target percentage/mae column.",
    )
    args = parser.parse_args()

    event_file = resolve_event_file(args.event_file)

    accumulator = load_events(event_file)
    available = set(accumulator.Tags().get("scalars", []))
    data_tags = [t for t in args.tags if not t.startswith("─")]
    selected = [tag for tag in data_tags if tag in available]
    missing = [tag for tag in data_tags if tag not in available]

    print(f"event_file: {event_file}")
    print(f"available_scalar_count: {len(available)}",end="    ")
    print(f"selected_tags: {len(selected)}",end="   ")
    if missing:
        print(f"missing_tags: {len(missing)}")
    else:
        print('')

    headers, rows = build_percentile_table(accumulator, args.tags, args.drop_first, mini=args.mini, window=args.window)
    print_table(headers, rows)

    summary_map: dict[str, SeriesSummary] = {}
    for tag in selected:
        summary = summarize_series(accumulator, tag, args.window, args.drop_first)
        if summary is None:
            continue
        summary_map[tag] = summary
        # print_summary(summary)

    if not summary_map:
        return

    episode_length = summary_map.get("Train/mean_episode_length")
    time_out = summary_map.get("Episode_Termination/time_out")
    anchor_pos = summary_map.get("Episode_Termination/anchor_pos")
    noise_std = summary_map.get("Policy/mean_noise_std")
    if episode_length and time_out and anchor_pos:
        noise_std_val = noise_std.last_value if noise_std else 0.0
        stage = classify_stage(episode_length.last_value, time_out.last_value, anchor_pos.last_value, noise_std_val)
        plateau = plateau_note(summary_map, args.plateau_tol)
        print("run_assessment:")
        print(f"  stage={stage}")
        print(f"  plateau={plateau}")
        print(
            "  note="
            f"episode_length={episode_length.last_value:.2f}, "
            f"time_out={time_out.last_value:.2f}, "
            f"anchor_pos={anchor_pos.last_value:.2f}"
            + (f", noise_std={noise_std_val:.3f}" if noise_std else "")
        )
        zero_terms = find_zero_reward_terms(summary_map, ZERO_WATCH_TAGS)
        if zero_terms:
            print(f"  zero_reward_terms={', '.join(zero_terms)}")

        e_jnt_vel_summary = summary_map.get("Metrics/motion/error_joint_vel")
        cur_stage = summary_map.get("Metrics/motion/curriculum_stage")
        cur_stage_int = int(cur_stage.last_value) if cur_stage is not None else 0
        if e_jnt_vel_summary is not None:
            print_joint_vel_breakdown(summary_map, e_jnt_vel_summary.last_value, cur_stage_int)

        act_rate = summary_map.get("Metrics/motion/raw_action_rate")
        e_body_pos = summary_map.get("Metrics/motion/error_body_pos")
        e_body_av = summary_map.get("Metrics/motion/error_body_ang_vel")
        if cur_stage is not None:
            parts = [f"stage={cur_stage.last_value:.0f}"]
            if e_body_pos:
                parts.append(f"e_body_pos={e_body_pos.last_value:.3f}")
            if e_body_av:
                parts.append(f"e_body_av={e_body_av.last_value:.2f}")
            if e_jnt_vel_summary:
                parts.append(f"e_jnt_vel={e_jnt_vel_summary.last_value:.2f}")
            if act_rate:
                parts.append(f"act_rate={act_rate.last_value:.2f}")
            print(f"  curriculum: {', '.join(parts)}")

    if args.segment_summary:
        motion_frames, motion_fps = resolve_motion_info(args.motion_file)
        segment_summaries = build_segment_summary(accumulator, args.drop_first, args.segments)
        if not segment_summaries:
            print("segment_summary:")
            print("  unavailable=missing Metrics/motion/sampling_top1_bin")
            return

        print("segment_summary:")
        segment_headers = [
            "segment",
            "range",
            "samples",
            "ep_len",
            "time_out",
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
            "contact",
            "fail_score",
        ]
        segment_rows: list[list[str]] = []
        for item in segment_summaries:
            segment_rows.append(
                [
                    f"{item.index}",
                    format_segment_range(item.start_ratio, item.end_ratio, motion_frames, motion_fps),
                    f"{item.sample_count}",
                    f"{item.mean_episode_length:.1f}",
                    f"{item.mean_time_out:.2f}",
                    f"{item.mean_anchor_pos:.2f}",
                    f"{item.mean_anchor_ori:.2f}",
                    f"{item.mean_ee_body_pos:.2f}",
                    f"{item.mean_undesired_contacts:.2f}",
                    f"{item.failure_score:.2f}",
                ]
            )
        print_table(segment_headers, segment_rows)

        hotspot = max(segment_summaries, key=lambda item: item.failure_score)
        print("segment_hotspot:")
        print(
            "  "
            + f"range={format_segment_range(hotspot.start_ratio, hotspot.end_ratio, motion_frames, motion_fps)} "
            + f"samples={hotspot.sample_count} "
            + f"ep_len={hotspot.mean_episode_length:.1f} "
            + f"anchor_pos={hotspot.mean_anchor_pos:.2f} "
            + f"ee_body_pos={hotspot.mean_ee_body_pos:.2f} "
            + f"contact={hotspot.mean_undesired_contacts:.2f}"
        )


if __name__ == "__main__":
    main()
