from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RATDPConfig
from postprocess.dct_perturb import eps_from_masked_block_gaussian_topk, perturb_video_masked_blocks_dct_dp_temporal
from scripts.run_24h import sigma_for_target_eps
from scripts.run_mask_isolation_icassp import (
    align_to_gt,
    evaluate_regions,
    frame_psnr_mean,
    mask_alignment,
    motion_proxy_score_maps,
    random_score_maps,
    read_video_frames,
    stable_seed,
    topk_pixel_masks,
    write_video_frames,
)


SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
K_VALUES = [10, 20, 30, 40]


@dataclass(frozen=True)
class Policy:
    name: str
    slug: str
    scores: np.ndarray
    score_origin: str
    offset_row: Optional[int] = None
    offset_col: Optional[int] = None


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan" if np.isnan(value) else "inf"
    return f"{value:.9f}"


def shift_scores(scores: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    return np.roll(scores, shift=offset, axis=(1, 2)).astype(np.float32)


def load_score(root: Path, video_id: str, variant: str) -> np.ndarray:
    path = root / video_id / f"score_maps_{variant}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing score map: {path}")
    return np.load(path).astype(np.float32)


def distortion_share(reference: List[np.ndarray], released: List[np.ndarray], masks: List[np.ndarray]) -> float:
    return float(evaluate_regions(reference, released, masks)["distortion_share"])


def policy_set(
    experiment: str,
    video_id: str,
    seed: int,
    shape: Tuple[int, int, int],
    backend_frames: List[np.ndarray],
    rat_traj: np.ndarray,
    score_root: Path,
    trajectory_root: Optional[Path],
) -> List[Policy]:
    full = load_score(score_root, video_id, "no_trajectory")
    videomae = load_score(score_root, video_id, "videomae_only")
    motion = motion_proxy_score_maps(backend_frames, block_grid=shape[1])
    random_scores = random_score_maps(shape, stable_seed(seed, video_id, "random-mask"))

    if experiment == "E1":
        if trajectory_root is None:
            raise ValueError("E1 requires --trajectory-score-root")
        rat_traj = load_score(trajectory_root / f"seed_{seed}", video_id, "trajectory")
        return [
            Policy("RAT+traj", "rat_traj", rat_traj, f"trajectory fusion with selector seed {seed}"),
            Policy("Motion-reference", "motion_reference", motion, "backend frame-difference calibration reference"),
            Policy("Random", "random", random_scores, f"stable random mask under master seed {seed}"),
            Policy("Shifted-RAT+traj", "shifted_rat_traj_r3_c4", shift_scores(rat_traj, (3, 4)), "RAT+traj shifted by public offset (3,4)", 3, 4),
            Policy("RAT-fusion", "rat_fusion", full, "full cue fusion without trajectory propagation"),
            Policy("Shifted-RAT-fusion", "shifted_rat_fusion_r3_c4", shift_scores(full, (3, 4)), "RAT fusion shifted by public offset (3,4)", 3, 4),
            Policy("VideoMAE-only", "videomae_only", videomae, "VideoMAE-only score map"),
        ]
    if experiment == "E2":
        return [
            Policy("RAT+traj", "rat_traj", rat_traj, "saved trajectory-smoothed RAT score map"),
            Policy("Motion-reference", "motion_reference", motion, "backend frame-difference calibration reference"),
            Policy("Random", "random", random_scores, f"stable random mask under master seed {seed}"),
            Policy("Shifted-RAT+traj", "shifted_rat_traj_r3_c4", shift_scores(rat_traj, (3, 4)), "RAT+traj shifted by public offset (3,4)", 3, 4),
            Policy("RAT-fusion", "rat_fusion", full, "full cue fusion without trajectory propagation"),
            Policy("Shifted-RAT-fusion", "shifted_rat_fusion_r3_c4", shift_scores(full, (3, 4)), "RAT fusion shifted by public offset (3,4)", 3, 4),
            Policy("VideoMAE-only", "videomae_only", videomae, "VideoMAE-only score map"),
        ]
    raise ValueError(experiment)


def configurations(experiment: str) -> Sequence[Tuple[int, int]]:
    if experiment == "E1":
        return [(seed, 20) for seed in SEEDS]
    if experiment == "E2":
        return [(42, k) for k in K_VALUES]
    raise ValueError(experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen E1/E2 allocation evidence protocol.")
    parser.add_argument("--experiment", required=True, choices=["E1", "E2"])
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--trajectory-score-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--dp-c-blk", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest_rows = [row for row in read_csv(Path(args.base_manifest)) if row.get("method") == "RAT-DP"]
    if args.limit > 0:
        manifest_rows = manifest_rows[: args.limit]
    output_dir = Path(args.output_dir)
    score_root = Path(args.score_root)
    trajectory_root = Path(args.trajectory_score_root) if args.trajectory_score_root else None
    metrics_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for base_row in manifest_rows:
        video_id = base_row["video_id"]
        input_video = Path(base_row["input_video"])
        rat_dir = Path(base_row["output_dir"])
        backend_video = rat_dir / "rat_intermediate_merged.mp4"
        rat_score_path = rat_dir / "masks" / "private_risk_scores.npy"
        try:
            raw_frames, raw_fps = read_video_frames(input_video)
            backend_frames, backend_fps = read_video_frames(backend_video)
            backend_frames, _ = align_to_gt(raw_frames, backend_frames)
            rat_traj = np.load(rat_score_path).astype(np.float32)
            if rat_traj.shape[0] != len(raw_frames):
                raise ValueError(f"score/frame mismatch for {video_id}")
            block_grid = rat_traj.shape[1]
            motion_reference_scores = motion_proxy_score_maps(raw_frames, block_grid)

            for seed, top_k in configurations(args.experiment):
                sigma = sigma_for_target_eps(
                    T=len(raw_frames),
                    target_eps=args.epsilon,
                    k=top_k,
                    C_blk=args.dp_c_blk,
                    delta=args.delta,
                    alphas=range(2, 129),
                )
                reference_masks = topk_pixel_masks(
                    motion_reference_scores,
                    raw_frames[0].shape[0],
                    raw_frames[0].shape[1],
                    top_k,
                )
                policies = policy_set(
                    args.experiment,
                    video_id,
                    seed,
                    rat_traj.shape,
                    backend_frames,
                    rat_traj,
                    score_root,
                    trajectory_root,
                )
                for policy in policies:
                    run_dir = output_dir / video_id / f"seed_{seed}" / f"k_{top_k}" / policy.slug
                    run_dir.mkdir(parents=True, exist_ok=True)
                    score_path = run_dir / "score_maps.npy"
                    output_video = run_dir / "release.mp4"
                    report_path = run_dir / "report.json"
                    np.save(score_path, policy.scores.astype(np.float32))
                    if not output_video.exists():
                        released, stats = perturb_video_masked_blocks_dct_dp_temporal(
                            frames_rgb=backend_frames,
                            score_maps_per_frame=[policy.scores[t] for t in range(len(policy.scores))],
                            block_grid=block_grid,
                            top_k=top_k,
                            C_blk=args.dp_c_blk,
                            sigma=sigma,
                            temporal_smooth=0.0,
                            seed=seed,
                            alpha=1.0,
                        )
                        write_video_frames(released, output_video, backend_fps or raw_fps)
                    decoded, _ = read_video_frames(output_video)
                    decoded, _ = align_to_gt(raw_frames, decoded)
                    candidate_masks = topk_pixel_masks(
                        policy.scores,
                        raw_frames[0].shape[0],
                        raw_frames[0].shape[1],
                        top_k,
                    )
                    cover = mask_alignment(candidate_masks, reference_masks)["reference_coverage"]
                    raw_err = distortion_share(raw_frames, decoded, reference_masks)
                    inc_err = distortion_share(backend_frames, decoded, reference_masks)
                    eps, alpha, rho, sensitivity = eps_from_masked_block_gaussian_topk(
                        T=len(raw_frames),
                        k=top_k,
                        C_blk=args.dp_c_blk,
                        sigma=sigma,
                        delta=args.delta,
                        alphas=range(2, 129),
                    )
                    row: Dict[str, object] = {
                        "experiment": args.experiment,
                        "video_id": video_id,
                        "class": base_row.get("class", ""),
                        "policy": policy.name,
                        "policy_slug": policy.slug,
                        "seed": seed,
                        "top_k": top_k,
                        "offset_row": "" if policy.offset_row is None else policy.offset_row,
                        "offset_col": "" if policy.offset_col is None else policy.offset_col,
                        "sigma": fmt(sigma),
                        "epsilon": fmt(eps),
                        "delta": fmt(args.delta),
                        "cover": fmt(cover),
                        "raw_errshare": fmt(raw_err),
                        "inc_errshare": fmt(inc_err),
                        "raw_psnr": fmt(frame_psnr_mean(raw_frames, decoded)),
                        "incremental_psnr": fmt(frame_psnr_mean(backend_frames, decoded)),
                    }
                    metrics_rows.append(row)
                    report = {
                        "experiment": args.experiment,
                        "video_id": video_id,
                        "policy": policy.name,
                        "score_origin": policy.score_origin,
                        "config": {
                            "seed": seed,
                            "top_k": top_k,
                            "block_grid": block_grid,
                            "sigma": sigma,
                            "epsilon": eps,
                            "delta": args.delta,
                            "C_blk": args.dp_c_blk,
                            "sensitivity_upper_bound": sensitivity,
                            "best_alpha": alpha,
                            "rho": rho,
                            "offset": [policy.offset_row, policy.offset_col],
                        },
                        "files": {
                            "input_video": str(input_video),
                            "input_sha256": sha256_file(input_video),
                            "backend_input_video": str(backend_video),
                            "backend_input_sha256": sha256_file(backend_video),
                            "score_maps": str(score_path),
                            "score_maps_sha256": sha256_file(score_path),
                            "output_video": str(output_video),
                            "output_video_sha256": sha256_file(output_video),
                        },
                        "metrics": row,
                    }
                    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    run_rows.append({
                        "experiment": args.experiment,
                        "video_id": video_id,
                        "class": base_row.get("class", ""),
                        "policy": policy.name,
                        "seed": seed,
                        "top_k": top_k,
                        "sigma": fmt(sigma),
                        "input_video": str(input_video),
                        "backend_input_video": str(backend_video),
                        "score_maps": str(score_path),
                        "output_video": str(output_video),
                        "report": str(report_path),
                    })
                write_csv(output_dir / "metrics.csv", metrics_rows)
                write_csv(output_dir / "run_manifest.csv", run_rows)
                print(f"[{args.experiment}] {video_id} seed={seed} k={top_k}: PASS")
        except Exception as exc:
            failures.append({"video_id": video_id, "error": repr(exc)})
            write_csv(output_dir / "failures.csv", failures)
            print(f"[{args.experiment}] {video_id}: FAIL {exc}")

    write_csv(output_dir / "metrics.csv", metrics_rows)
    write_csv(output_dir / "run_manifest.csv", run_rows)
    write_csv(output_dir / "failures.csv", failures)
    print(f"[Done] rows={len(metrics_rows)} failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
