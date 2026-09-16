from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RATDPConfig
from utils.video_io import read_video_frames
from utils.preprocess import preprocess_frames
from utils.sliding_window import split_into_clips
from modules.feature_clip import clip_l2
from modules.risk_assessment import compute_block_risk_map
from models.videomae_model import VideoMAEOfficialWrapper
from models.raft_model import RAFTFlowEstimator
from main import (
    _handle_oom_and_retry,
    _pad_clip_to_tubelet_multiple,
    merge_frame_score_maps,
)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cfg(args: argparse.Namespace, row: Dict[str, str]) -> RATDPConfig:
    cfg = RATDPConfig()
    cfg.input_video = row["input_video"]
    cfg.seed = int(args.seed)
    cfg.dp_top_k = int(args.top_k)
    cfg.dp_C_blk = float(args.dp_c_blk)
    cfg.dp_delta = float(row.get("delta") or cfg.dp_delta)
    cfg.dp_sigma = float(row.get("rat_sigma") or cfg.dp_sigma)
    cfg.device = str(args.device)
    cfg.raft_iters = int(args.raft_iters)
    if args.videomae_root:
        cfg.videomae_root = args.videomae_root
    if args.videomae_weight_path:
        cfg.videomae_weight_path = args.videomae_weight_path
    if args.raft_root:
        cfg.raft_root = args.raft_root
    if args.raft_ckpt:
        cfg.raft_ckpt = args.raft_ckpt
    return cfg


def init_models(cfg: RATDPConfig) -> Tuple[VideoMAEOfficialWrapper, RAFTFlowEstimator, str]:
    try:
        videomae = VideoMAEOfficialWrapper(
            model_root=cfg.videomae_root,
            device=cfg.device,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            tubelet_size=cfg.tubelet_size,
            weight_path=getattr(cfg, "videomae_weight_path", "") or None,
        )
        raft_estimator = RAFTFlowEstimator(
            raft_root=cfg.raft_root,
            ckpt_path=cfg.raft_ckpt,
            device=cfg.device,
            iters=cfg.raft_iters,
        )
        return videomae, raft_estimator, cfg.device
    except RuntimeError as exc:
        if "cuda" not in str(exc).lower() and "out of memory" not in str(exc).lower():
            raise
        print("[AblationScores] CUDA init failed; retrying on CPU.")
        cfg.device = "cpu"
        torch.cuda.empty_cache()
        videomae = VideoMAEOfficialWrapper(
            model_root=cfg.videomae_root,
            device=cfg.device,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            tubelet_size=cfg.tubelet_size,
            weight_path=getattr(cfg, "videomae_weight_path", "") or None,
        )
        raft_estimator = RAFTFlowEstimator(
            raft_root=cfg.raft_root,
            ckpt_path=cfg.raft_ckpt,
            device=cfg.device,
            iters=cfg.raft_iters,
        )
        return videomae, raft_estimator, cfg.device


def compute_variant_scores(
    cfg: RATDPConfig,
    videomae: VideoMAEOfficialWrapper,
    raft_estimator: RAFTFlowEstimator,
    variants: List[str],
) -> Dict[str, np.ndarray]:
    raw_frames, _fps = read_video_frames(cfg.input_video)
    clips = split_into_clips(raw_frames, clip_len=cfg.clip_len, stride=max(1, cfg.clip_len // 2))
    mean = torch.tensor(cfg.mean, device=cfg.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(cfg.std, device=cfg.device).view(1, 1, 3, 1, 1)

    by_variant: Dict[str, List[Tuple[int, List[np.ndarray]]]] = {name: [] for name in variants}

    for clip_idx, (start_idx, clip_frames) in enumerate(clips):
        print(f"  -> score clip {clip_idx + 1}/{len(clips)} start={start_idx}")
        clip_frames = _pad_clip_to_tubelet_multiple(clip_frames, cfg.tubelet_size)
        processed_frames, _metas = preprocess_frames(clip_frames, target_size=cfg.image_size)
        flows = raft_estimator.compute_video_flows(processed_frames)

        clip_array = np.stack(processed_frames, axis=0)
        clip_rgb = torch.from_numpy(clip_array).float() / 255.0
        clip_rgb = clip_rgb.permute(0, 3, 1, 2).unsqueeze(0).to(cfg.device)
        clip_norm = (clip_rgb - mean) / std

        tokens = _handle_oom_and_retry(
            videomae.extract_patch_tokens,
            clip_norm,
            videomae=videomae,
        )[0]
        tokens = clip_l2(tokens, cfg.C_h)

        _, clean_risk_map, _, _ = compute_block_risk_map(
            patch_features=tokens,
            flows=flows,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            tubelet_size=cfg.tubelet_size,
            block_grid=cfg.block_grid,
            lambda_similarity=cfg.lambda_similarity,
            sigma_r=cfg.sigma_r,
            motion_norm=cfg.motion_norm,
            reference_frame_rgb=processed_frames[0],
        )
        maps: Dict[str, np.ndarray] = {
            "no_trajectory": clean_risk_map.detach().cpu().numpy().astype(np.float32)
        }

        if "videomae_only" in variants:
            _, attn_only_map, _, _ = compute_block_risk_map(
                patch_features=tokens,
                flows=flows,
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                tubelet_size=cfg.tubelet_size,
                block_grid=cfg.block_grid,
                lambda_similarity=cfg.lambda_similarity,
                sigma_r=cfg.sigma_r,
                motion_norm=cfg.motion_norm,
                w_attn=1.0,
                w_temp=0.0,
                w_tex=0.0,
                w_flow=0.0,
                reference_frame_rgb=processed_frames[0],
            )
            maps["videomae_only"] = attn_only_map.detach().cpu().numpy().astype(np.float32)

        clip_T = len(processed_frames)
        for variant in variants:
            score_map_3d = maps[variant]
            per_frame_scores: List[np.ndarray] = []
            for t in range(clip_T):
                u = t / cfg.tubelet_size
                u0 = int(np.floor(u))
                u1 = min(u0 + 1, score_map_3d.shape[0] - 1)
                alpha = u - u0
                score = (1 - alpha) * score_map_3d[u0] + alpha * score_map_3d[u1]
                per_frame_scores.append(score.astype(np.float32))
            by_variant[variant].append((start_idx, per_frame_scores))

    return {
        variant: np.stack(
            merge_frame_score_maps(by_variant[variant], total_len=len(raw_frames), clip_len=cfg.clip_len),
            axis=0,
        ).astype(np.float32)
        for variant in variants
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mask-ablation score maps for ICASSP same-backend evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dp-c-blk", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raft-iters", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variants", default="no_trajectory,videomae_only")
    parser.add_argument("--videomae-root", default=None)
    parser.add_argument("--videomae-weight-path", default=None)
    parser.add_argument("--raft-root", default=None)
    parser.add_argument("--raft-ckpt", default=None)
    args = parser.parse_args()

    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    allowed = {"no_trajectory", "videomae_only"}
    bad = [item for item in requested if item not in allowed]
    if bad:
        raise ValueError(f"Unknown variants: {bad}")

    manifest_rows = [row for row in read_csv(Path(args.manifest)) if row.get("method") == "RAT-DP"]
    if args.limit > 0:
        manifest_rows = manifest_rows[: int(args.limit)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for row in manifest_rows:
        video_id = row["video_id"]
        video_dir = output_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg = make_cfg(args, row)
            set_seed(cfg.seed)
            videomae, raft_estimator, actual_device = init_models(cfg)
            scores_by_variant = compute_variant_scores(cfg, videomae, raft_estimator, requested)
            for variant, scores in scores_by_variant.items():
                score_path = video_dir / f"score_maps_{variant}.npy"
                np.save(str(score_path), scores.astype(np.float32))
                rows.append({
                    "video_id": video_id,
                    "class": row.get("class", ""),
                    "variant": variant,
                    "score_maps": str(score_path),
                    "score_sha256": sha256_file(score_path),
                    "frames": int(scores.shape[0]),
                    "block_grid": int(scores.shape[1]),
                    "seed": int(cfg.seed),
                    "device": actual_device,
                    "input_video": cfg.input_video,
                    "rat_output_dir": row.get("output_dir", ""),
                    "rat_sigma": float(cfg.dp_sigma),
                    "delta": float(cfg.dp_delta),
                    "top_k": int(cfg.dp_top_k),
                })
            report = {
                "video_id": video_id,
                "input_video": cfg.input_video,
                "input_sha256": sha256_file(Path(cfg.input_video)),
                "variants": requested,
                "config": {
                    "seed": int(cfg.seed),
                    "clip_len": int(cfg.clip_len),
                    "tubelet_size": int(cfg.tubelet_size),
                    "image_size": int(cfg.image_size),
                    "patch_size": int(cfg.patch_size),
                    "block_grid": int(cfg.block_grid),
                    "weights_no_trajectory": "0.85 VideoMAE + 0.10 temporal + 0.03 Sobel + 0.02 RAFT, no trajectory propagation",
                    "weights_videomae_only": "1.00 VideoMAE, no temporal/Sobel/RAFT contribution",
                    "device": actual_device,
                    "raft_iters": int(cfg.raft_iters),
                },
            }
            (video_dir / "score_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[AblationScores] {video_id}: PASS")
        except Exception as exc:
            failures.append({"video_id": video_id, "error": repr(exc)})
            print(f"[AblationScores] {video_id}: FAIL {exc}")

    write_csv(output_dir / "score_manifest.csv", rows)
    write_csv(output_dir / "score_failures.csv", failures)
    print(f"[Done] score rows={len(rows)} failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
