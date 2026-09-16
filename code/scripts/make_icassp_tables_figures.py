from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


POLICY_ROWS = [
    ("Motion ref. (upper bound)", "Motion-only-mask"),
    ("RAT fusion", "RAT-no-trajectory-mask"),
    ("RAT+traj.", "RAT-mask"),
    ("VideoMAE-only", "VideoMAE-only-mask"),
    ("Random", "Random-mask"),
    ("Shifted fusion", "Shifted-RAT-fusion-mask"),
]

PAIR_ROWS = [
    ("RAT--Random", "RAT-no-trajectory-mask", "Random-mask"),
    ("RAT--Shift", "RAT-no-trajectory-mask", "Shifted-RAT-fusion-mask"),
    ("RAT--VideoMAE", "RAT-no-trajectory-mask", "VideoMAE-only-mask"),
    ("RAT--RAT+traj.", "RAT-no-trajectory-mask", "RAT-mask"),
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_method(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {row["method"]: row for row in rows}


def fmt4(value: str) -> str:
    return f"{float(value):.4f}"


def fmt_ci(lo: str, hi: str) -> str:
    return f"[{float(lo):.4f},{float(hi):.4f}]"


def fmt_p(value: str) -> str:
    p = float(value)
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}".rstrip("0").rstrip(".")


def find_pair(
    rows: List[Dict[str, str]],
    left: str,
    right: str,
    metric: str,
) -> Tuple[Dict[str, str], bool]:
    forward = f"{left} vs {right}"
    reverse = f"{right} vs {left}"
    for row in rows:
        if row["comparison"] == forward and row["metric"] == metric:
            return row, False
    for row in rows:
        if row["comparison"] == reverse and row["metric"] == metric:
            return row, True
    raise KeyError(f"missing pair: {left} vs {right} / {metric}")


def signed_pair_values(
    rows: List[Dict[str, str]],
    left: str,
    right: str,
    metric: str,
) -> Tuple[int, int, int, float, float, float, float]:
    row, inverted = find_pair(rows, left, right, metric)
    n = int(row["n"])
    if inverted:
        wins = int(row["right_wins"])
        losses = int(row["left_wins"])
        diff = -float(row["mean_diff_left_minus_right"])
        ci_lo = -float(row["ci95_high"])
        ci_hi = -float(row["ci95_low"])
    else:
        wins = int(row["left_wins"])
        losses = int(row["right_wins"])
        diff = float(row["mean_diff_left_minus_right"])
        ci_lo = float(row["ci95_low"])
        ci_hi = float(row["ci95_high"])
    p = float(row["exact_sign_test_p"])
    return n, wins, losses, diff, ci_lo, ci_hi, p


def print_table_1(summary: Path, incremental: Path) -> None:
    summary_by_method = by_method(read_csv(summary))
    inc_by_method = by_method(read_csv(incremental))
    print("% Table 1 values generated from saved CSV files")
    for label, method in POLICY_ROWS:
        row = summary_by_method[method]
        inc = inc_by_method[method]
        values = [
            fmt4(row["global_frame_psnr_avg"]),
            fmt4(row["motion_reference_coverage_avg"]),
            fmt4(row["motion_distortion_share_avg"]),
            fmt4(inc["incremental_motion_distortion_share_avg"]),
            fmt4(row["motion_low_psnr_avg"]),
        ]
        print(f"{label} & {' & '.join(values)} \\\\")


def print_table_2(paired: Path, incremental_paired: Path) -> None:
    paired_rows = read_csv(paired)
    inc_rows = read_csv(incremental_paired)
    print("% Table 2 values generated from saved CSV files")
    for label, left, right in PAIR_ROWS:
        for metric_label, source_rows, metric in [
            ("Cover", paired_rows, "motion_reference_coverage"),
            ("RawErr", paired_rows, "motion_distortion_share"),
            ("IncErr", inc_rows, "incremental_motion_distortion_share"),
        ]:
            n, wins, _losses, diff, ci_lo, ci_hi, p = signed_pair_values(source_rows, left, right, metric)
            print(
                f"{label} & {metric_label} & {wins}/{n} & {diff:.4f} & "
                f"[{ci_lo:.4f},{ci_hi:.4f}] & {fmt_p(str(p))} \\\\"
            )


def write_proxy_alignment_figure(paired: Path, incremental_paired: Path, output: Path) -> None:
    paired_rows = read_csv(paired)
    inc_rows = read_csv(incremental_paired)
    controls = [
        ("Random", "Random-mask"),
        ("Shifted", "Shifted-RAT-fusion-mask"),
        ("VideoMAE", "VideoMAE-only-mask"),
        ("RAT+traj.", "RAT-mask"),
    ]
    metrics = [
        ("Cover", paired_rows, "motion_reference_coverage", "#d94b45"),
        ("RawErr", paired_rows, "motion_distortion_share", "#3b75af"),
        ("IncErr", inc_rows, "incremental_motion_distortion_share", "#59a14f"),
    ]
    fig, ax = plt.subplots(figsize=(11.92, 3.67), dpi=160)
    y_base = np.arange(len(controls))
    offsets = [-0.18, 0.0, 0.18]
    for offset, (metric_label, source_rows, metric, color) in zip(offsets, metrics):
        xs, xerr_low, xerr_high = [], [], []
        for _control_label, control_method in controls:
            _n, _wins, _losses, diff, ci_lo, ci_hi, _p = signed_pair_values(
                source_rows,
                "RAT-no-trajectory-mask",
                control_method,
                metric,
            )
            xs.append(diff)
            xerr_low.append(diff - ci_lo)
            xerr_high.append(ci_hi - diff)
        ax.errorbar(
            xs,
            y_base + offset,
            xerr=[xerr_low, xerr_high],
            fmt="o",
            markersize=6,
            linewidth=2,
            capsize=3,
            label=metric_label,
            color=color,
        )
    ax.axvline(0.0, color="0.35", linewidth=1.4)
    ax.set_yticks(y_base)
    ax.set_yticklabels([item[0] for item in controls])
    ax.invert_yaxis()
    ax.set_xlabel("RAT fusion minus control")
    ax.set_title("Paired deltas over videos", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", ncol=3, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, transparent=False)
    plt.close(fig)


def read_video_frames(path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return frames, fps


def split_bounds(length: int, parts: int) -> List[Tuple[int, int]]:
    chunks = np.array_split(np.arange(length), parts)
    bounds: List[Tuple[int, int]] = []
    last_end = 0
    for arr in chunks:
        if len(arr) == 0:
            bounds.append((last_end, last_end))
            continue
        bounds.append((int(arr[0]), int(arr[-1]) + 1))
        last_end = int(arr[-1]) + 1
    return bounds


def topk_pixel_masks(score_maps: np.ndarray, height: int, width: int, top_k: int) -> List[np.ndarray]:
    block_grid = int(score_maps.shape[1])
    y_bounds = split_bounds(height, block_grid)
    x_bounds = split_bounds(width, block_grid)
    out = []
    for score_map in score_maps:
        idx = np.argsort(score_map.reshape(-1))[-top_k:]
        mask = np.zeros((height, width), dtype=bool)
        for flat in idx:
            bi, bj = int(flat // block_grid), int(flat % block_grid)
            y0, y1 = y_bounds[bi]
            x0, x1 = x_bounds[bj]
            mask[y0:y1, x0:x1] = True
        out.append(mask)
    return out


def overlay(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = frame.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = 0.52 * out[mask] + 0.48 * color_arr
    return out.clip(0, 255).astype(np.uint8)


def write_mask_example_figure(run_manifest: Path, output: Path, video_id: str | None) -> None:
    rows = read_csv(run_manifest)
    by_video: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        by_video.setdefault(row["video_id"], {})[row["method"]] = row
    selected_video = video_id or sorted(by_video)[0]
    methods = by_video[selected_video]
    required = {
        "Motion ref. (upper bound)": ("Motion-only-mask", (0, 205, 220)),
        "RAT fusion": ("RAT-no-trajectory-mask", (230, 70, 55)),
        "Random": ("Random-mask", (55, 170, 80)),
        "Shifted": ("Shifted-RAT-fusion-mask", (120, 85, 210)),
    }
    raw_frames, _fps = read_video_frames(Path(next(iter(methods.values()))["input_video"]))
    motion_scores = np.load(methods["Motion-only-mask"]["score_maps"]).astype(np.float32)
    frame_idx = int(np.argmax(np.sum(motion_scores, axis=(1, 2))))
    top_k = int(next(iter(methods.values()))["top_k"])
    h, w = raw_frames[frame_idx].shape[:2]

    panels = [("Raw frame", raw_frames[frame_idx])]
    for label, (method, color) in required.items():
        scores = np.load(methods[method]["score_maps"]).astype(np.float32)
        masks = topk_pixel_masks(scores, h, w, top_k)
        panels.append((label, overlay(raw_frames[frame_idx], masks[frame_idx], color)))

    fig, axes = plt.subplots(1, len(panels), figsize=(9.62, 1.58), dpi=160)
    for ax, (label, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.03)
    fig.savefig(output, transparent=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate ICASSP paper table values and Fig. 1 from saved experiment artifacts.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--example-video", default="")
    parser.add_argument("--print-tables", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    fig_dir = Path(args.figure_dir)
    summary = exp_dir / "mask_isolation_summary.csv"
    paired = exp_dir / "mask_isolation_paired_stats.csv"
    incremental = exp_dir / "incremental_errshare_summary.csv"
    incremental_paired = exp_dir / "incremental_errshare_paired_stats.csv"
    run_manifest = exp_dir / "run_manifest.csv"

    if args.print_tables:
        print_table_1(summary, incremental)
        print()
        print_table_2(paired, incremental_paired)

    write_proxy_alignment_figure(paired, incremental_paired, fig_dir / "proxy_alignment_summary.png")
    write_mask_example_figure(
        run_manifest,
        fig_dir / "mask_policy_example.png",
        args.example_video or "BalanceBeam_v_BalanceBeam_g01_c01",
    )
    print(f"[Done] figures written to {fig_dir}")


if __name__ == "__main__":
    main()
