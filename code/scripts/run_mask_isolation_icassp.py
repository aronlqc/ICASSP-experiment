from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RATDPConfig
from postprocess.dct_perturb import (
    eps_from_masked_block_gaussian_topk,
    perturb_video_masked_blocks_dct_dp_temporal,
)


METHODS = {
    "RAT-mask": "rat_mask",
    "RAT-no-trajectory-mask": "rat_no_trajectory_mask",
    "VideoMAE-only-mask": "videomae_only_mask",
    "Shifted-RAT-fusion-mask": "shifted_rat_fusion_mask",
    "Motion-only-mask": "motion_only_mask",
    "Random-mask": "random_mask",
    "Shifted-RAT-mask": "shifted_rat_mask",
}


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(seed: int, *parts: str) -> int:
    h = hashlib.sha256(str(seed).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], byteorder="little", signed=False) % (2**32 - 1)


def read_video_frames(path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    return frames, fps


def write_video_frames(frames: List[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("frames must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*("XVID" if path.suffix.lower() == ".avi" else "mp4v"))
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(np.ascontiguousarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def align_to_gt(gt_frames: List[np.ndarray], frames: List[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
    if len(gt_frames) != len(frames):
        raise ValueError(f"Frame count mismatch: gt={len(gt_frames)} frames={len(frames)}")
    gh, gw = gt_frames[0].shape[:2]
    out: List[np.ndarray] = []
    resized = False
    for frame in frames:
        if frame.shape[:2] != (gh, gw):
            frame = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_LINEAR)
            resized = True
        out.append(frame)
    return out, resized


def split_bounds(length: int, parts: int) -> List[Tuple[int, int]]:
    chunks = np.array_split(np.arange(length), parts)
    bounds: List[Tuple[int, int]] = []
    last_end = 0
    for arr in chunks:
        if len(arr) == 0:
            bounds.append((last_end, last_end))
            continue
        start = int(arr[0])
        end = int(arr[-1]) + 1
        if start < last_end:
            start = last_end
        bounds.append((start, end))
        last_end = end
    return bounds


def topk_block_mask(score_map: np.ndarray, top_k: int) -> np.ndarray:
    b = int(score_map.shape[0])
    idx = np.argsort(score_map.reshape(-1))[-int(top_k):]
    mask = np.zeros((b, b), dtype=bool)
    for flat in idx:
        mask[int(flat // b), int(flat % b)] = True
    return mask


def topk_pixel_masks(score_maps: np.ndarray, height: int, width: int, top_k: int) -> List[np.ndarray]:
    if score_maps.ndim != 3 or score_maps.shape[1] != score_maps.shape[2]:
        raise ValueError(f"Expected [T,B,B] score maps, got {score_maps.shape}")
    block_grid = int(score_maps.shape[1])
    y_bounds = split_bounds(height, block_grid)
    x_bounds = split_bounds(width, block_grid)
    masks: List[np.ndarray] = []
    for score_map in score_maps:
        block_mask = topk_block_mask(score_map, top_k=top_k)
        pixel_mask = np.zeros((height, width), dtype=bool)
        for bi in range(block_grid):
            for bj in range(block_grid):
                if not block_mask[bi, bj]:
                    continue
                y0, y1 = y_bounds[bi]
                x0, x1 = x_bounds[bj]
                pixel_mask[y0:y1, x0:x1] = True
        masks.append(pixel_mask)
    return masks


def motion_proxy_score_maps(frames: List[np.ndarray], block_grid: int) -> np.ndarray:
    height, width = frames[0].shape[:2]
    y_bounds = split_bounds(height, block_grid)
    x_bounds = split_bounds(width, block_grid)
    grays = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) for frame in frames]
    maps = []
    for t, gray in enumerate(grays):
        if len(grays) == 1:
            diff = np.zeros_like(gray, dtype=np.float32)
        elif t == 0:
            diff = np.abs(grays[1] - gray)
        elif t == len(grays) - 1:
            diff = np.abs(gray - grays[t - 1])
        else:
            diff = 0.5 * (np.abs(gray - grays[t - 1]) + np.abs(grays[t + 1] - gray))

        score = np.zeros((block_grid, block_grid), dtype=np.float32)
        for bi, (y0, y1) in enumerate(y_bounds):
            for bj, (x0, x1) in enumerate(x_bounds):
                block = diff[y0:y1, x0:x1]
                score[bi, bj] = float(np.mean(block)) if block.size else 0.0
        maps.append(score)
    arr = np.stack(maps, axis=0).astype(np.float32)
    lo = float(np.quantile(arr, 0.02))
    hi = float(np.quantile(arr, 0.98))
    if hi > lo + 1e-8:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)


def shifted_score_maps(score_maps: np.ndarray) -> np.ndarray:
    block_grid = int(score_maps.shape[1])
    return np.roll(score_maps, shift=(max(1, block_grid // 4), max(1, block_grid // 3)), axis=(1, 2)).astype(np.float32)


def random_score_maps(shape: Tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def load_extra_score_maps(root: Path, video_id: str, variant: str, expected_shape: Tuple[int, int, int]) -> np.ndarray:
    path = root / video_id / f"score_maps_{variant}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing extra score maps: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.shape != expected_shape:
        raise ValueError(f"extra score shape mismatch for {variant}: got {arr.shape}, expected {expected_shape}")
    return arr


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def finite_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    return float(np.nanstd(arr)) if np.any(np.isfinite(arr)) else float("nan")


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan" if np.isnan(value) else "inf"
    return f"{value:.6f}"


def frame_psnr_mean(gt_frames: List[np.ndarray], pred_frames: List[np.ndarray]) -> float:
    vals = []
    for gt, pred in zip(gt_frames, pred_frames):
        diff = gt.astype(np.float64) - pred.astype(np.float64)
        vals.append(psnr_from_mse(float(np.mean(diff * diff))))
    return finite_mean(vals)


def evaluate_regions(
    gt_frames: List[np.ndarray],
    pred_frames: List[np.ndarray],
    masks: List[np.ndarray],
) -> Dict[str, float]:
    if len(gt_frames) != len(pred_frames) or len(gt_frames) != len(masks):
        raise ValueError("gt/pred/mask length mismatch")
    high_sse = 0.0
    low_sse = 0.0
    total_sse = 0.0
    high_count = 0
    low_count = 0
    total_count = 0
    for gt, pred, mask in zip(gt_frames, pred_frames, masks):
        diff2 = (gt.astype(np.float64) - pred.astype(np.float64)) ** 2
        mask3 = np.repeat(mask[..., None], 3, axis=2)
        high_sse += float(diff2[mask3].sum())
        low_sse += float(diff2[~mask3].sum())
        total_sse += float(diff2.sum())
        high_count += int(mask.sum()) * 3
        low_count += int((~mask).sum()) * 3
        total_count += int(diff2.size)
    high_mse = high_sse / max(high_count, 1)
    low_mse = low_sse / max(low_count, 1)
    total_mse = total_sse / max(total_count, 1)
    area = high_count / max(total_count, 1)
    share = high_sse / total_sse if total_sse > 1e-12 else float("nan")
    return {
        "area": area,
        "total_mse_psnr": psnr_from_mse(total_mse),
        "high_psnr": psnr_from_mse(high_mse),
        "low_psnr": psnr_from_mse(low_mse),
        "distortion_share": share,
        "distortion_lift": share / area if area > 0 else float("nan"),
        "high_low_mse_ratio": high_mse / low_mse if low_mse > 1e-12 else float("inf"),
    }


def mask_alignment(candidate_masks: List[np.ndarray], reference_masks: List[np.ndarray]) -> Dict[str, float]:
    inter = 0
    cand_count = 0
    ref_count = 0
    union = 0
    total = 0
    for cand, ref in zip(candidate_masks, reference_masks):
        cand_count += int(cand.sum())
        ref_count += int(ref.sum())
        inter += int(np.logical_and(cand, ref).sum())
        union += int(np.logical_or(cand, ref).sum())
        total += int(cand.size)
    cand_area = cand_count / max(total, 1)
    ref_area = ref_count / max(total, 1)
    recall = inter / max(ref_count, 1)
    precision = inter / max(cand_count, 1)
    return {
        "candidate_area": cand_area,
        "reference_area": ref_area,
        "reference_coverage": recall,
        "candidate_precision": precision,
        "iou": inter / max(union, 1),
        "coverage_lift": recall / cand_area if cand_area > 0 else float("nan"),
    }


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, str]:
    return {f"{prefix}_{key}": fmt(value) for key, value in metrics.items()}


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = [
        "global_frame_psnr",
        "backend_input_psnr",
        "motion_reference_coverage",
        "motion_coverage_lift",
        "motion_distortion_share",
        "motion_distortion_lift",
        "motion_high_psnr",
        "motion_low_psnr",
        "rat_region_distortion_share",
        "rat_region_distortion_lift",
        "own_region_distortion_share",
        "own_region_distortion_lift",
    ]
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    out: List[Dict[str, object]] = []
    for method in sorted(groups):
        items = groups[method]
        row: Dict[str, object] = {"method": method, "n": len(items)}
        for metric in metrics:
            vals = [float(item[metric]) for item in items]
            row[f"{metric}_avg"] = fmt(finite_mean(vals))
            row[f"{metric}_std"] = fmt(finite_std(vals))
        out.append(row)
    return out


def exact_sign_test_pvalue(diffs: List[float]) -> float:
    signs = [1 if diff > 0 else -1 if diff < 0 else 0 for diff in diffs]
    positives = sum(1 for sign in signs if sign > 0)
    negatives = sum(1 for sign in signs if sign < 0)
    n = positives + negatives
    if n == 0:
        return 1.0
    tail = min(positives, negatives)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / float(2 ** n)
    return min(1.0, 2.0 * prob)


def bootstrap_mean_ci(diffs: List[float], seed: int = 20260901) -> Tuple[float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(10000, len(arr)), replace=True)
    means = np.mean(samples, axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_stats(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    available = {str(row["method"]) for row in rows}
    candidate_comparisons = [
        ("RAT-no-trajectory-mask", "Random-mask"),
        ("RAT-no-trajectory-mask", "Shifted-RAT-fusion-mask"),
        ("RAT-no-trajectory-mask", "Motion-only-mask"),
        ("RAT-no-trajectory-mask", "VideoMAE-only-mask"),
        ("RAT-mask", "RAT-no-trajectory-mask"),
        ("RAT-mask", "Random-mask"),
        ("RAT-mask", "Shifted-RAT-mask"),
        ("RAT-mask", "Motion-only-mask"),
        ("Motion-only-mask", "Random-mask"),
        ("Motion-only-mask", "Shifted-RAT-mask"),
    ]
    comparisons = [
        (left, right)
        for left, right in candidate_comparisons
        if left in available and right in available
    ]
    metrics = [
        ("motion_reference_coverage", "higher"),
        ("motion_distortion_share", "higher"),
    ]

    by_video: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_video[str(row["video_id"])][str(row["method"])] = row

    out: List[Dict[str, object]] = []
    for left, right in comparisons:
        for metric, direction in metrics:
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
                elif (diff > 0 and direction == "higher") or (diff < 0 and direction == "lower"):
                    left_wins += 1
                else:
                    right_wins += 1

            ci_lo, ci_hi = bootstrap_mean_ci(diffs, seed=stable_seed(20260901, left, right, metric))
            out.append({
                "comparison": f"{left} vs {right}",
                "metric": metric,
                "direction": direction,
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


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = frame.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)[None, None, :]
    out[mask] = 0.55 * out[mask] + 0.45 * color_arr
    return out.round().clip(0, 255).astype(np.uint8)


def error_heatmap(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    err = np.mean(np.abs(gt.astype(np.float32) - pred.astype(np.float32)), axis=2)
    hi = float(np.quantile(err, 0.98))
    if hi <= 1e-6:
        hi = 1.0
    norm = np.clip(err / hi, 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def label_panel(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_example_figure(
    figure_path: Path,
    gt_frames: List[np.ndarray],
    pred_by_method: Dict[str, List[np.ndarray]],
    masks_by_method: Dict[str, List[np.ndarray]],
    motion_masks: List[np.ndarray],
    motion_scores: np.ndarray,
) -> None:
    frame_idx = int(np.argmax(np.sum(motion_scores, axis=(1, 2))))
    panels = [
        label_panel(gt_frames[frame_idx], "GT"),
        label_panel(overlay_mask(gt_frames[frame_idx], motion_masks[frame_idx], (0, 220, 255)), "Motion proxy"),
        label_panel(overlay_mask(pred_by_method["RAT-mask"][frame_idx], masks_by_method["RAT-mask"][frame_idx], (255, 40, 40)), "RAT mask"),
    ]
    if "Motion-only-mask" in pred_by_method:
        panels.append(label_panel(
            overlay_mask(
                pred_by_method["Motion-only-mask"][frame_idx],
                masks_by_method["Motion-only-mask"][frame_idx],
                (255, 190, 40),
            ),
            "Motion-only",
        ))
    panels.extend([
        label_panel(overlay_mask(pred_by_method["Random-mask"][frame_idx], masks_by_method["Random-mask"][frame_idx], (40, 200, 80)), "Random mask"),
        label_panel(overlay_mask(pred_by_method["Shifted-RAT-mask"][frame_idx], masks_by_method["Shifted-RAT-mask"][frame_idx], (160, 90, 255)), "Shifted RAT"),
        label_panel(error_heatmap(gt_frames[frame_idx], pred_by_method["RAT-mask"][frame_idx]), "RAT error"),
    ])
    height = min(180, gt_frames[frame_idx].shape[0])
    resized = []
    for panel in panels:
        scale = height / panel.shape[0]
        resized.append(cv2.resize(panel, (int(round(panel.shape[1] * scale)), height), interpolation=cv2.INTER_AREA))
    grid = np.concatenate(resized, axis=1)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(figure_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def write_report(
    path: Path,
    method: str,
    input_video: Path,
    backend_video: Path,
    output_video: Path,
    score_path: Path,
    score_origin: str,
    cfg: RATDPConfig,
    eps: float,
    alpha: float,
    rho: float,
    delta_bound: float,
    stats: Dict[str, object],
) -> None:
    report = {
        "method": method,
        "input_video": str(input_video),
        "input_sha256": sha256_file(input_video),
        "backend_input_video": str(backend_video),
        "backend_input_sha256": sha256_file(backend_video),
        "output_video": str(output_video),
        "output_sha256": sha256_file(output_video),
        "score_map_path": str(score_path),
        "score_map_sha256": sha256_file(score_path),
        "score_origin": score_origin,
        "config": {
            "epsilon": float(eps),
            "delta": float(cfg.dp_delta),
            "block_grid": int(cfg.block_grid),
            "dp_top_k": int(cfg.dp_top_k),
            "dp_C_blk": float(cfg.dp_C_blk),
            "dp_sigma": float(cfg.dp_sigma),
            "seed": int(cfg.seed),
        },
        "final_release_dp": {
            "mechanism": "fixed_mask_topk_y_lowfreq_dct_iid_gaussian_postprocessed",
            "theorem_scope": "fixed_mask_conditional_release_only",
            "epsilon_best": float(eps),
            "alpha_best": float(alpha),
            "rho_best": float(rho),
            "Delta_upper_bound": float(delta_bound),
            "num_compositions": int(stats.get("T", 0)),
            "release_stats": stats,
            "note": "This same-backend experiment changes only the score map used by the fixed Gaussian release.",
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(
    path: Path,
    aggregate_rows: List[Dict[str, object]],
    paired_rows: List[Dict[str, object]],
    failures: List[Dict[str, object]],
) -> None:
    lines = [
        "# Same-Backend Mask Isolation",
        "",
        "Protocol: for each UCF101 clip, all variants use the saved RAT pre-release video as the backend input, the same fixed-mask DCT Gaussian release, the same top-k, the same sigma, and the same noise seed. Only the score map/mask changes.",
        "",
        "Motion proxy: block-level frame-difference saliency from the raw GT video. It is a cheap independent proxy for action/motion salience, not a privacy label.",
        "",
        "## Aggregate",
        "",
        "| Method | n | PSNR | Motion Cover | Motion Lift | Motion Dist Share | Motion Dist Lift | Motion High PSNR | Motion Low PSNR | RAT-Region Share | Own-Region Share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {method} | {n} | {global_frame_psnr_avg} | {motion_reference_coverage_avg} | {motion_coverage_lift_avg} | {motion_distortion_share_avg} | {motion_distortion_lift_avg} | {motion_high_psnr_avg} | {motion_low_psnr_avg} | {rat_region_distortion_share_avg} | {own_region_distortion_share_avg} |".format(**row)
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Random-mask, Shifted-RAT-fusion-mask, and VideoMAE-only-mask are controls for the narrowed allocation claim.",
        "- Motion-only-mask is the constructed motion-salience proxy reference, not a deployable baseline that RAT is expected to beat.",
        "- RAT-no-trajectory-mask is the main paper method (`RAT fusion`); RAT-mask is retained only as the trajectory-smoothed ablation (`RAT+traj.`).",
        "- The trajectory-smoothed variant is statistically indistinguishable from direct cue fusion, so the paper should not claim a trajectory-propagation gain.",
        "- Global PSNR is secondary here because all rows share the same backend and only the allocation changes.",
    ])
    if paired_rows:
        lines.extend([
            "",
            "## Paired Main-Metric Statistics",
            "",
            "| Comparison | Metric | n | Left Wins | Right Wins | Mean Diff | 95% CI | Sign p |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in paired_rows:
            lines.append(
                "| {comparison} | {metric} | {n} | {left_wins} | {right_wins} | {mean_diff_left_minus_right} | [{ci95_low}, {ci95_high}] | {exact_sign_test_p} |".format(**row)
            )
    if failures:
        lines.extend(["", "## Failures", "", "| Video | Error |", "|---|---|"])
        for failure in failures:
            lines.append(f"| {failure.get('video_id', '')} | {failure.get('error', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-backend mask-isolation controls for the ICASSP rescue.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dp-c-blk", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--extra-score-root", default="", help="Optional root containing per-video score_maps_<variant>.npy files.")
    parser.add_argument("--include-no-trajectory", action="store_true")
    parser.add_argument("--include-videomae-only", action="store_true")
    parser.add_argument("--include-shifted-no-trajectory", action="store_true")
    args = parser.parse_args()

    manifest_rows = read_csv(Path(args.manifest))
    by_video: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in manifest_rows:
        by_video[row["video_id"]][row["method"]] = row

    cfg_template = RATDPConfig()
    cfg_template.seed = int(args.seed)
    cfg_template.dp_top_k = int(args.top_k)
    cfg_template.dp_C_blk = float(args.dp_c_blk)

    output_dir = Path(args.output_dir)
    rows: List[Dict[str, object]] = []
    run_manifest: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    videos = sorted(by_video.items())
    if args.limit > 0:
        videos = videos[: int(args.limit)]

    example_saved = False
    for video_id, method_rows in videos:
        if "RAT-DP" not in method_rows:
            failures.append({"video_id": video_id, "error": "missing RAT-DP manifest row"})
            continue
        try:
            rat_row = method_rows["RAT-DP"]
            input_video = Path(rat_row["input_video"])
            rat_dir = Path(rat_row["output_dir"])
            backend_video = rat_dir / "rat_intermediate_merged.mp4"
            score_path = rat_dir / "masks" / "private_risk_scores.npy"
            if not backend_video.exists():
                raise FileNotFoundError(f"missing backend video: {backend_video}")
            if not score_path.exists():
                raise FileNotFoundError(f"missing score map: {score_path}")

            gt_frames, gt_fps = read_video_frames(input_video)
            backend_frames, backend_fps = read_video_frames(backend_video)
            backend_frames, backend_resized = align_to_gt(gt_frames, backend_frames)
            rat_scores = np.load(score_path).astype(np.float32)
            if rat_scores.ndim != 3:
                raise ValueError(f"RAT score maps must be [T,B,B], got {rat_scores.shape}")
            if rat_scores.shape[0] != len(gt_frames):
                raise ValueError(f"score/frame mismatch: scores={rat_scores.shape[0]} gt={len(gt_frames)}")
            block_grid = int(rat_scores.shape[1])
            if rat_scores.shape[1] != rat_scores.shape[2]:
                raise ValueError(f"RAT score maps must be square, got {rat_scores.shape}")

            cfg = RATDPConfig()
            cfg.seed = int(args.seed)
            cfg.dp_top_k = int(args.top_k)
            cfg.dp_C_blk = float(args.dp_c_blk)
            cfg.dp_delta = float(rat_row.get("delta") or cfg.dp_delta)
            cfg.dp_sigma = float(rat_row.get("rat_sigma") or cfg.dp_sigma)
            cfg.block_grid = block_grid

            motion_scores = motion_proxy_score_maps(gt_frames, block_grid=block_grid)
            motion_masks = topk_pixel_masks(motion_scores, gt_frames[0].shape[0], gt_frames[0].shape[1], cfg.dp_top_k)
            rat_masks = topk_pixel_masks(rat_scores, gt_frames[0].shape[0], gt_frames[0].shape[1], cfg.dp_top_k)

            score_variants = {
                "RAT-mask": (rat_scores, "private RAT-DP risk scores from the existing run"),
                "Motion-only-mask": (
                    motion_proxy_score_maps(backend_frames, block_grid=block_grid),
                    "backend frame-difference motion-only scores",
                ),
                "Random-mask": (
                    random_score_maps(rat_scores.shape, stable_seed(args.seed, video_id, "random-mask")),
                    "public seeded random scores independent of video content",
                ),
                "Shifted-RAT-mask": (shifted_score_maps(rat_scores), "RAT scores shifted by a fixed public spatial offset"),
            }
            if args.extra_score_root:
                extra_root = Path(args.extra_score_root)
                no_trajectory_scores = None
                if args.include_no_trajectory or args.include_shifted_no_trajectory:
                    no_trajectory_scores = load_extra_score_maps(extra_root, video_id, "no_trajectory", rat_scores.shape)
                if args.include_no_trajectory:
                    score_variants["RAT-no-trajectory-mask"] = (
                        no_trajectory_scores,
                        "same RAT cue fusion without trajectory propagation",
                    )
                if args.include_shifted_no_trajectory:
                    score_variants["Shifted-RAT-fusion-mask"] = (
                        shifted_score_maps(no_trajectory_scores),
                        "RAT cue-fusion scores shifted by a fixed public spatial offset",
                    )
                if args.include_videomae_only:
                    score_variants["VideoMAE-only-mask"] = (
                        load_extra_score_maps(extra_root, video_id, "videomae_only", rat_scores.shape),
                        "VideoMAE semantic-token salience only",
                    )

            pred_by_method: Dict[str, List[np.ndarray]] = {}
            masks_by_method: Dict[str, List[np.ndarray]] = {}
            backend_psnr = frame_psnr_mean(gt_frames, backend_frames)

            for method, (scores, score_origin) in score_variants.items():
                method_slug = METHODS[method]
                method_dir = output_dir / video_id / f"seed_{args.seed}" / method_slug
                method_dir.mkdir(parents=True, exist_ok=True)
                variant_score_path = method_dir / "score_maps.npy"
                np.save(str(variant_score_path), scores.astype(np.float32))

                released_frames, stats = perturb_video_masked_blocks_dct_dp_temporal(
                    frames_rgb=backend_frames,
                    score_maps_per_frame=[scores[t] for t in range(scores.shape[0])],
                    block_grid=cfg.block_grid,
                    top_k=cfg.dp_top_k,
                    C_blk=cfg.dp_C_blk,
                    sigma=cfg.dp_sigma,
                    temporal_smooth=0.0,
                    seed=cfg.seed,
                    alpha=1.0,
                )
                stats = dict(stats)
                stats["T"] = len(released_frames)
                output_video = method_dir / "release.mp4"
                write_video_frames(released_frames, output_video, fps=backend_fps or gt_fps)
                eps, alpha, rho, delta_bound = eps_from_masked_block_gaussian_topk(
                    T=len(released_frames),
                    k=cfg.dp_top_k,
                    C_blk=cfg.dp_C_blk,
                    sigma=cfg.dp_sigma,
                    delta=cfg.dp_delta,
                    alphas=cfg.rdp_alphas,
                    release_stride=cfg.release_stride,
                )
                report_path = method_dir / "report.json"
                write_report(
                    path=report_path,
                    method=method,
                    input_video=input_video,
                    backend_video=backend_video,
                    output_video=output_video,
                    score_path=variant_score_path,
                    score_origin=score_origin,
                    cfg=cfg,
                    eps=eps,
                    alpha=alpha,
                    rho=rho,
                    delta_bound=delta_bound,
                    stats=stats,
                )

                decoded_frames, _ = read_video_frames(output_video)
                decoded_frames, output_resized = align_to_gt(gt_frames, decoded_frames)
                candidate_masks = topk_pixel_masks(scores, gt_frames[0].shape[0], gt_frames[0].shape[1], cfg.dp_top_k)
                pred_by_method[method] = decoded_frames
                masks_by_method[method] = candidate_masks

                motion_region = evaluate_regions(gt_frames, decoded_frames, motion_masks)
                rat_region = evaluate_regions(gt_frames, decoded_frames, rat_masks)
                own_region = evaluate_regions(gt_frames, decoded_frames, candidate_masks)
                align_motion = mask_alignment(candidate_masks, motion_masks)
                align_rat = mask_alignment(candidate_masks, rat_masks)

                row: Dict[str, object] = {
                    "video_id": video_id,
                    "class": rat_row.get("class", ""),
                    "method": method,
                    "seed": args.seed,
                    "input_video": str(input_video),
                    "backend_input_video": str(backend_video),
                    "backend_input_psnr": fmt(backend_psnr),
                    "backend_resized_to_gt": str(bool(backend_resized)),
                    "score_origin": score_origin,
                    "output_video": str(output_video),
                    "report": str(report_path),
                    "output_resized_to_gt": str(bool(output_resized)),
                    "epsilon": fmt(eps),
                    "delta": fmt(cfg.dp_delta),
                    "sigma": fmt(cfg.dp_sigma),
                    "top_k": cfg.dp_top_k,
                    "block_grid": cfg.block_grid,
                    "global_frame_psnr": fmt(frame_psnr_mean(gt_frames, decoded_frames)),
                }
                row.update(prefix_metrics("motion", motion_region))
                row.update(prefix_metrics("rat_region", rat_region))
                row.update(prefix_metrics("own_region", own_region))
                row.update(prefix_metrics("motion_alignment", align_motion))
                row.update(prefix_metrics("rat_alignment", align_rat))
                # Short aliases used by the aggregate table.
                row["motion_reference_coverage"] = row["motion_alignment_reference_coverage"]
                row["motion_coverage_lift"] = row["motion_alignment_coverage_lift"]
                row["rat_reference_coverage"] = row["rat_alignment_reference_coverage"]
                row["rat_coverage_lift"] = row["rat_alignment_coverage_lift"]
                rows.append(row)
                run_manifest.append({
                    "video_id": video_id,
                    "class": rat_row.get("class", ""),
                    "method": method,
                    "output_dir": str(method_dir),
                    "output_video": str(output_video),
                    "report": str(report_path),
                    "score_maps": str(variant_score_path),
                    "input_video": str(input_video),
                    "backend_input_video": str(backend_video),
                    "seed": args.seed,
                    "sigma": fmt(cfg.dp_sigma),
                    "epsilon": fmt(eps),
                    "delta": fmt(cfg.dp_delta),
                    "top_k": cfg.dp_top_k,
                })

            if not example_saved:
                save_example_figure(
                    output_dir / "figures" / f"mask_isolation_{video_id}.png",
                    gt_frames=gt_frames,
                    pred_by_method=pred_by_method,
                    masks_by_method=masks_by_method,
                    motion_masks=motion_masks,
                    motion_scores=motion_scores,
                )
                example_saved = True
            print(f"[MaskIsolation] {video_id}: PASS")
        except Exception as exc:
            failures.append({"video_id": video_id, "error": repr(exc)})
            print(f"[MaskIsolation] {video_id}: FAIL {exc}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "mask_isolation_metrics.csv", rows)
    aggregate_rows = aggregate(rows)
    write_csv(output_dir / "mask_isolation_summary.csv", aggregate_rows)
    paired_rows = paired_stats(rows)
    write_csv(output_dir / "mask_isolation_paired_stats.csv", paired_rows)
    write_csv(output_dir / "run_manifest.csv", run_manifest)
    write_csv(output_dir / "failures.csv", failures)
    write_summary(output_dir / "mask_isolation_summary.md", aggregate_rows, paired_rows, failures)
    print(f"[Done] rows={len(rows)} failures={len(failures)}")
    print(f"[Done] {output_dir / 'mask_isolation_summary.md'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
