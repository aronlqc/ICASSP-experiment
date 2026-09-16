from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_mask_isolation_icassp import (  # noqa: E402
    align_to_gt,
    bootstrap_mean_ci,
    evaluate_regions,
    exact_sign_test_pvalue,
    finite_mean,
    finite_std,
    fmt,
    motion_proxy_score_maps,
    read_video_frames,
    stable_seed,
    topk_pixel_masks,
)


METHOD_ORDER = [
    "Motion-only-mask",
    "RAT-no-trajectory-mask",
    "RAT-mask",
    "VideoMAE-only-mask",
    "Random-mask",
    "Shifted-RAT-fusion-mask",
    "Shifted-RAT-mask",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    out: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        if method not in groups:
            continue
        items = groups[method]
        row: Dict[str, object] = {"method": method, "n": len(items)}
        for metric in [
            "incremental_motion_distortion_share",
            "incremental_motion_distortion_lift",
            "incremental_motion_total_psnr",
            "incremental_motion_high_psnr",
            "incremental_motion_low_psnr",
        ]:
            vals = [float(item[metric]) for item in items]
            row[f"{metric}_avg"] = fmt(finite_mean(vals))
            row[f"{metric}_std"] = fmt(finite_std(vals))
        out.append(row)
    return out


def paired_stats(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    available = {str(row["method"]) for row in rows}
    comparisons = [
        ("RAT-no-trajectory-mask", "Random-mask"),
        ("RAT-no-trajectory-mask", "Shifted-RAT-fusion-mask"),
        ("RAT-no-trajectory-mask", "VideoMAE-only-mask"),
        ("RAT-mask", "RAT-no-trajectory-mask"),
        ("RAT-no-trajectory-mask", "Motion-only-mask"),
    ]
    comparisons = [(l, r) for l, r in comparisons if l in available and r in available]
    by_video: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_video[str(row["video_id"])][str(row["method"])] = row

    out: List[Dict[str, object]] = []
    metric = "incremental_motion_distortion_share"
    for left, right in comparisons:
        diffs: List[float] = []
        left_wins = 0
        right_wins = 0
        ties = 0
        for method_rows in by_video.values():
            if left not in method_rows or right not in method_rows:
                continue
            lv = float(method_rows[left][metric])
            rv = float(method_rows[right][metric])
            if not (np.isfinite(lv) and np.isfinite(rv)):
                continue
            diff = lv - rv
            diffs.append(diff)
            if abs(diff) <= 1e-12:
                ties += 1
            elif diff > 0:
                left_wins += 1
            else:
                right_wins += 1
        ci_lo, ci_hi = bootstrap_mean_ci(diffs, seed=stable_seed(20260905, left, right, metric))
        out.append({
            "comparison": f"{left} vs {right}",
            "metric": metric,
            "n": len(diffs),
            "left_wins": left_wins,
            "right_wins": right_wins,
            "ties": ties,
            "mean_diff_left_minus_right": fmt(finite_mean(diffs)),
            "ci95_low": fmt(ci_lo),
            "ci95_high": fmt(ci_hi),
            "exact_sign_test_p": fmt(exact_sign_test_pvalue(diffs)),
        })
    return out


def write_summary(path: Path, aggregate_rows: List[Dict[str, object]], paired_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Incremental Perturbation ErrShare",
        "",
        "This analysis reuses existing releases and computes motion-proxy error share using backend-input-to-output error, ||Z-Y||^2, rather than raw-video-to-output error, ||X-Y||^2.",
        "",
        "## Aggregate",
        "",
        "| Method | n | Incr. ErrShare | Incr. Lift | Incr. total PSNR | Incr. high PSNR | Incr. low PSNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {method} | {n} | {incremental_motion_distortion_share_avg} | {incremental_motion_distortion_lift_avg} | {incremental_motion_total_psnr_avg} | {incremental_motion_high_psnr_avg} | {incremental_motion_low_psnr_avg} |".format(**row)
        )
    lines.extend([
        "",
        "## Paired Statistics",
        "",
        "| Comparison | n | Left Wins | Right Wins | Mean Diff | 95% CI | Sign p |",
        "|---|---:|---:|---:|---:|---|---:|",
    ])
    for row in paired_rows:
        lines.append(
            "| {comparison} | {n} | {left_wins} | {right_wins} | {mean_diff_left_minus_right} | [{ci95_low}, {ci95_high}] | {exact_sign_test_p} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute incremental Z-to-Y ErrShare for existing ICASSP mask-isolation releases.")
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_rows = read_csv(Path(args.run_manifest))
    metrics_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    cache: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]] = {}

    for row in run_rows:
        video_id = row["video_id"]
        method = row["method"]
        try:
            if video_id not in cache:
                raw_frames, _ = read_video_frames(Path(row["input_video"]))
                backend_frames, _ = read_video_frames(Path(row["backend_input_video"]))
                backend_frames, _ = align_to_gt(raw_frames, backend_frames)
                scores = np.load(row["score_maps"]).astype(np.float32)
                block_grid = int(scores.shape[1])
                top_k = int(row["top_k"])
                motion_scores = motion_proxy_score_maps(raw_frames, block_grid=block_grid)
                motion_masks = topk_pixel_masks(
                    motion_scores,
                    raw_frames[0].shape[0],
                    raw_frames[0].shape[1],
                    top_k,
                )
                cache[video_id] = (raw_frames, backend_frames, motion_masks)

            raw_frames, backend_frames, motion_masks = cache[video_id]
            output_frames, _ = read_video_frames(Path(row["output_video"]))
            output_frames, _ = align_to_gt(raw_frames, output_frames)
            incr = evaluate_regions(backend_frames, output_frames, motion_masks)
            metrics_rows.append({
                "video_id": video_id,
                "class": row.get("class", ""),
                "method": method,
                "top_k": row.get("top_k", ""),
                "incremental_motion_distortion_share": fmt(incr["distortion_share"]),
                "incremental_motion_distortion_lift": fmt(incr["distortion_lift"]),
                "incremental_motion_total_psnr": fmt(incr["total_mse_psnr"]),
                "incremental_motion_high_psnr": fmt(incr["high_psnr"]),
                "incremental_motion_low_psnr": fmt(incr["low_psnr"]),
            })
        except Exception as exc:
            failures.append({"video_id": video_id, "method": method, "error": repr(exc)})
            print(f"[IncrementalErrShare] {video_id} {method}: FAIL {exc}")

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "incremental_errshare_metrics.csv", metrics_rows)
    aggregate_rows = aggregate(metrics_rows)
    paired_rows = paired_stats(metrics_rows)
    write_csv(output_dir / "incremental_errshare_summary.csv", aggregate_rows)
    write_csv(output_dir / "incremental_errshare_paired_stats.csv", paired_rows)
    write_csv(output_dir / "incremental_errshare_failures.csv", failures)
    write_summary(output_dir / "incremental_errshare_summary.md", aggregate_rows, paired_rows)
    print(f"[Done] rows={len(metrics_rows)} failures={len(failures)}")
    print(output_dir / "incremental_errshare_summary.md")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
