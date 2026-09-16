from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RATDPConfig
from main import _handle_oom_and_retry, _pad_clip_to_tubelet_multiple, merge_frame_score_maps
from models.raft_model import RAFTFlowEstimator
from models.videomae_model import VideoMAEOfficialWrapper
from modules.feature_clip import clip_l2
from modules.risk_assessment import compute_block_risk_map
from modules.trajectory import build_block_trajectories, trajectory_private_weights
from utils.preprocess import preprocess_frames
from utils.sliding_window import split_into_clips
from utils.video_io import read_video_frames


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def interpolate(score_map: np.ndarray, clip_len: int, tubelet_size: int) -> List[np.ndarray]:
    output = []
    for frame_index in range(clip_len):
        position = frame_index / tubelet_size
        low = int(np.floor(position))
        high = min(low + 1, score_map.shape[0] - 1)
        weight = position - low
        output.append(((1.0 - weight) * score_map[low] + weight * score_map[high]).astype(np.float32))
    return output


def init_models(cfg: RATDPConfig) -> Tuple[VideoMAEOfficialWrapper, RAFTFlowEstimator, str]:
    try:
        videomae = VideoMAEOfficialWrapper(
            model_root=cfg.videomae_root,
            device=cfg.device,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            tubelet_size=cfg.tubelet_size,
            weight_path=cfg.videomae_weight_path or None,
        )
        raft = RAFTFlowEstimator(
            raft_root=cfg.raft_root,
            ckpt_path=cfg.raft_ckpt,
            device=cfg.device,
            iters=cfg.raft_iters,
        )
        return videomae, raft, cfg.device
    except RuntimeError as exc:
        if "cuda" not in str(exc).lower() and "out of memory" not in str(exc).lower():
            raise
        print("[MultiSeedScores] CUDA initialization failed; falling back to CPU.")
        cfg.device = "cpu"
        torch.cuda.empty_cache()
        return init_models(cfg)


def compute_video(
    cfg: RATDPConfig,
    videomae: VideoMAEOfficialWrapper,
    raft: RAFTFlowEstimator,
    seeds: List[int],
) -> Dict[int, np.ndarray]:
    raw_frames, _ = read_video_frames(cfg.input_video)
    clips = split_into_clips(raw_frames, clip_len=cfg.clip_len, stride=max(1, cfg.clip_len // 2))
    mean = torch.tensor(cfg.mean, device=cfg.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(cfg.std, device=cfg.device).view(1, 1, 3, 1, 1)
    per_seed: Dict[int, List[Tuple[int, List[np.ndarray]]]] = {seed: [] for seed in seeds}

    for clip_index, (start, clip_frames) in enumerate(clips):
        processed, _ = preprocess_frames(
            _pad_clip_to_tubelet_multiple(clip_frames, cfg.tubelet_size),
            target_size=cfg.image_size,
        )
        flows = raft.compute_video_flows(processed)
        clip = torch.from_numpy(np.stack(processed)).float().div(255.0).permute(0, 3, 1, 2).unsqueeze(0).to(cfg.device)
        tokens = _handle_oom_and_retry(videomae.extract_patch_tokens, (clip - mean) / std, videomae=videomae)[0]
        tokens = clip_l2(tokens, cfg.C_h)
        _, clean, _, _ = compute_block_risk_map(
            patch_features=tokens,
            flows=flows,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            tubelet_size=cfg.tubelet_size,
            block_grid=cfg.block_grid,
            lambda_similarity=cfg.lambda_similarity,
            sigma_r=cfg.sigma_r,
            motion_norm=cfg.motion_norm,
            reference_frame_rgb=processed[0],
        )
        trajectories, confidence = build_block_trajectories(
            flows=flows,
            image_size=cfg.image_size,
            block_grid=cfg.block_grid,
            tubelet_size=cfg.tubelet_size,
            tau_conf=cfg.tau_conf,
        )
        for seed in seeds:
            generator = torch.Generator(device=clean.device)
            generator.manual_seed(seed + clip_index)
            _, _, _, _, smooth = trajectory_private_weights(
                risk_map=clean,
                trajectories=trajectories,
                confidence_map=confidence.to(clean.device),
                lambda_s=cfg.lambda_s,
                sigma_s=cfg.sigma_s,
                beta=cfg.beta_traj,
                eta_smooth=max(0.25, float(cfg.eta_smooth)),
                generator=generator,
            )
            score_map = smooth.detach().cpu().numpy().reshape(smooth.shape[0], cfg.block_grid, cfg.block_grid)
            per_seed[seed].append((start, interpolate(score_map, len(processed), cfg.tubelet_size)))
        print(f"[MultiSeedScores] clip {clip_index + 1}/{len(clips)} start={start}")

    return {
        seed: np.stack(
            merge_frame_score_maps(items, total_len=len(raw_frames), clip_len=cfg.clip_len),
            axis=0,
        ).astype(np.float32)
        for seed, items in per_seed.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAT+traj score maps for multiple selector-noise seeds in one feature pass.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raft-iters", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--videomae-root", required=True)
    parser.add_argument("--videomae-weight-path", required=True)
    parser.add_argument("--raft-root", required=True)
    parser.add_argument("--raft-ckpt", required=True)
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows = [row for row in read_csv(Path(args.manifest)) if row.get("method") == "RAT-DP"]
    if args.limit > 0:
        rows = rows[: args.limit]
    output_root = Path(args.output_root)
    manifest_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    models = None

    for row in rows:
        video_id = row["video_id"]
        try:
            expected = {seed: output_root / f"seed_{seed}" / video_id / "score_maps_trajectory.npy" for seed in seeds}
            if not all(path.exists() for path in expected.values()):
                cfg = RATDPConfig()
                cfg.input_video = row["input_video"]
                cfg.device = args.device
                cfg.raft_iters = args.raft_iters
                cfg.videomae_root = args.videomae_root
                cfg.videomae_weight_path = args.videomae_weight_path
                cfg.raft_root = args.raft_root
                cfg.raft_ckpt = args.raft_ckpt
                if models is None:
                    models = init_models(cfg)
                videomae, raft, actual_device = models
                cfg.device = actual_device
                arrays = compute_video(cfg, videomae, raft, seeds)
                for seed, array in arrays.items():
                    expected[seed].parent.mkdir(parents=True, exist_ok=True)
                    np.save(expected[seed], array)
            for seed, path in expected.items():
                array = np.load(path, mmap_mode="r")
                manifest_rows.append({
                    "video_id": video_id,
                    "class": row.get("class", ""),
                    "variant": "trajectory",
                    "selector_seed": seed,
                    "score_maps": str(path),
                    "score_sha256": sha256_file(path),
                    "frames": array.shape[0],
                    "block_grid": array.shape[1],
                    "input_video": row["input_video"],
                })
            print(f"[MultiSeedScores] {video_id}: PASS")
        except Exception as exc:
            failures.append({"video_id": video_id, "error": repr(exc)})
            print(f"[MultiSeedScores] {video_id}: FAIL {exc}")
        write_csv(output_root / "score_manifest.csv", manifest_rows)
        write_csv(output_root / "score_failures.csv", failures)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
