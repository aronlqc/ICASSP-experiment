from __future__ import annotations

import argparse
import json
import hashlib
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import RATDPConfig
from utils.video_io import read_video_frames, write_video_frames
from utils.preprocess import preprocess_frames
from utils.resize_back import restore_frame_to_original
from utils.sliding_window import split_into_clips, merge_clips_with_flow_guidance

from modules.feature_clip import clip_l2
from modules.risk_assessment import compute_block_risk_map
from modules.trajectory import build_block_trajectories, trajectory_private_weights

from postprocess.dct_perturb import (
    perturb_clip_in_dct_domain,               # 非DP（可选）
    eps_from_masked_block_gaussian_topk,      # 严格会计（匹配邻接）
    perturb_video_masked_blocks_dct_dp_temporal,
 
)

from models.videomae_model import VideoMAEOfficialWrapper
from models.raft_model import RAFTFlowEstimator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _handle_oom_and_retry(fn, *args, videomae=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" not in msg and "cuda" not in msg:
            raise

        print("[RAT-DP] CUDA OOM detected, falling back to CPU for this model.")
        import gc

        if videomae is not None and hasattr(videomae, "model"):
            try:
                videomae.model.to("cpu")
                videomae.device = torch.device("cpu")
            except Exception:
                pass

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        def _move_cpu(x):
            return x.cpu() if isinstance(x, torch.Tensor) else x

        args_cpu = tuple(_move_cpu(a) for a in args)
        kwargs_cpu = {k: _move_cpu(v) for k, v in kwargs.items()}
        return fn(*args_cpu, **kwargs_cpu)


def _pad_clip_to_tubelet_multiple(frames: List[np.ndarray], tubelet_size: int) -> List[np.ndarray]:
    if tubelet_size <= 0:
        raise ValueError("tubelet_size 必须 > 0")
    if len(frames) == 0:
        return frames
    out = list(frames)
    while len(out) % tubelet_size != 0:
        out.append(out[-1].copy())
    return out


def _normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / (hi - lo + eps)


def save_block_heatmap_overlay(frame_rgb: np.ndarray, block_map_2d, out_path: str, alpha: float = 0.45, draw_grid: bool = True):
    if isinstance(block_map_2d, torch.Tensor):
        block_map_2d = block_map_2d.detach().cpu().numpy()
    block_map_2d = np.asarray(block_map_2d, dtype=np.float32)
    if block_map_2d.ndim == 1:
        B = int(np.sqrt(block_map_2d.shape[0]))
        block_map_2d = block_map_2d.reshape(B, B)

    m = _normalize01(block_map_2d)
    h, w = frame_rgb.shape[:2]
    heat = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = (heat * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    base = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)

    if draw_grid:
        B = block_map_2d.shape[0]
        for i in range(1, B):
            x = int(round(i * w / B))
            y = int(round(i * h / B))
            cv2.line(overlay, (x, 0), (x, h - 1), (255, 255, 255), 1)
            cv2.line(overlay, (0, y), (w - 1, y), (255, 255, 255), 1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, overlay)


def _clip_edge_weight(local_i: int, clip_len: int, power: float = 1.7) -> float:
    if clip_len <= 1:
        return 1.0
    center = (clip_len - 1) / 2.0
    denom = max(center, 1e-6)
    w = 1.0 - abs(local_i - center) / denom
    w = float(np.clip(w, 0.0, 1.0))
    return float(w ** power)


def merge_frame_score_maps(
    clips_scores: List[Tuple[int, List[np.ndarray]]],
    total_len: int,
    clip_len: int,
) -> List[np.ndarray]:
    if total_len <= 0:
        return []
    if len(clips_scores) == 0:
        raise ValueError("clips_scores is empty")

    clips_scores = sorted(clips_scores, key=lambda x: x[0])

    B = clips_scores[0][1][0].shape[0]
    accum = [np.zeros((B, B), dtype=np.float32) for _ in range(total_len)]
    wsum = np.zeros((total_len,), dtype=np.float32)

    for start_idx, score_list in clips_scores:
        for local_i, s2d in enumerate(score_list):
            global_i = start_idx + local_i
            if global_i >= total_len:
                break
            w = _clip_edge_weight(local_i, clip_len=clip_len, power=1.7)
            accum[global_i] += s2d.astype(np.float32) * w
            wsum[global_i] += w

    out = []
    for i in range(total_len):
        if wsum[i] > 0:
            s = accum[i] / max(float(wsum[i]), 1e-6)
        else:
            s = accum[i - 1] if i > 0 else accum[0]
        out.append(np.clip(s, 0.0, 1.0).astype(np.float32))
    return out


def _score_maps_to_array(score_maps, block_grid: int, total_len: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(score_maps, dtype=np.float32)
    if arr.ndim == 2 and arr.shape == (block_grid, block_grid):
        repeat_len = int(total_len) if total_len is not None else 1
        arr = np.repeat(arr[None, :, :], repeat_len, axis=0)
    if arr.ndim == 2 and arr.shape[1] == block_grid * block_grid:
        arr = arr.reshape(arr.shape[0], block_grid, block_grid)
    if arr.ndim != 3 or arr.shape[1:] != (block_grid, block_grid):
        raise ValueError(f"mask/score maps must be [T,{block_grid},{block_grid}], got {arr.shape}")
    return arr.astype(np.float32)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_if_exists(path: str) -> str:
    if not path or not Path(path).exists():
        return ""
    return _sha256_file(path)


def _resolve_release_score_maps(cfg: RATDPConfig, computed_scores: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
    mode = str(getattr(cfg, "mask_mode", "private_risk_experimental"))
    info: Dict = {
        "mask_mode": mode,
        "theorem_scope": (
            "covered_by_fixed_mask_conditional_theorem"
            if mode in {"external_mask", "fixed_precomputed_mask"}
            else "not_covered_by_fixed_mask_conditional_theorem"
        ),
    }

    if mode in {"external_mask", "fixed_precomputed_mask"}:
        mask_path = str(getattr(cfg, "external_mask_path", "") or "")
        if not mask_path:
            raise ValueError("external_mask_path is required for external/fixed mask modes")
        arr = np.load(mask_path)
        arr = _score_maps_to_array(arr, cfg.block_grid, total_len=len(computed_scores))
        if arr.shape[0] == 1 and len(computed_scores) != 1:
            arr = np.repeat(arr, len(computed_scores), axis=0)
        if arr.shape[0] != len(computed_scores):
            raise ValueError(f"external mask length {arr.shape[0]} != video length {len(computed_scores)}")
        info.update({
            "mask_path": mask_path,
            "mask_sha256": _sha256_file(mask_path),
        })
        return [arr[t].astype(np.float32) for t in range(arr.shape[0])], info

    arr = _score_maps_to_array(computed_scores, cfg.block_grid, total_len=len(computed_scores))
    private_mask_dir = Path(cfg.output_dir) / "masks"
    private_mask_dir.mkdir(parents=True, exist_ok=True)
    private_mask_path = private_mask_dir / "private_risk_scores.npy"
    np.save(str(private_mask_path), arr)
    info.update({
        "mask_path": str(private_mask_path),
        "mask_sha256": _sha256_file(str(private_mask_path)),
        "warning": (
            "These masks were inferred from the private input video. The fixed-mask "
            "conditional theorem does not cover the end-to-end risk-estimation pipeline."
        ),
    })
    return computed_scores, info


def main():
    cfg = RATDPConfig()
    parser = argparse.ArgumentParser(description="Run RAT-DP release.")
    parser.add_argument("--input", default=None, help="Input video path.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mask-mode", default=None, choices=[
        "private_risk_experimental",
        "external_mask",
        "fixed_precomputed_mask",
    ])
    parser.add_argument("--external-mask-path", default=None)
    parser.add_argument("--dp-delta", type=float, default=None)
    parser.add_argument("--dp-sigma", type=float, default=None)
    parser.add_argument("--dp-top-k", type=int, default=None)
    parser.add_argument("--dp-c-blk", type=float, default=None)
    parser.add_argument("--videomae-root", default=None)
    parser.add_argument("--videomae-weight-path", default=None)
    parser.add_argument("--raft-root", default=None)
    parser.add_argument("--raft-ckpt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--raft-iters", type=int, default=None)
    parser.add_argument("--disable-flow-merge-smoothing", action="store_true")
    args = parser.parse_args()

    if args.input:
        cfg.input_video = args.input
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.mask_mode:
        cfg.mask_mode = args.mask_mode
    if args.external_mask_path is not None:
        cfg.external_mask_path = args.external_mask_path
    if args.dp_delta is not None:
        cfg.dp_delta = args.dp_delta
    if args.dp_sigma is not None:
        cfg.dp_sigma = args.dp_sigma
    if args.dp_top_k is not None:
        cfg.dp_top_k = args.dp_top_k
    if args.dp_c_blk is not None:
        cfg.dp_C_blk = args.dp_c_blk
    if args.videomae_root is not None:
        cfg.videomae_root = args.videomae_root
    if args.videomae_weight_path is not None:
        cfg.videomae_weight_path = args.videomae_weight_path
    if args.raft_root is not None:
        cfg.raft_root = args.raft_root
    if args.raft_ckpt is not None:
        cfg.raft_ckpt = args.raft_ckpt
    if args.device is not None:
        cfg.device = args.device
    if args.raft_iters is not None:
        cfg.raft_iters = args.raft_iters
    if args.disable_flow_merge_smoothing:
        cfg.enable_flow_merge_smoothing = False

    cfg.ensure_dirs()
    set_seed(cfg.seed)

    print("[1/6] 读取视频 ...")
    raw_frames, fps = read_video_frames(cfg.input_video)
    

    print("[2/6] overlap 滑动窗口切分 ...")
    clips = split_into_clips(
        raw_frames,
        clip_len=cfg.clip_len,
        stride=max(1, cfg.clip_len // 2),
    )

    print("[3/6] 初始化 VideoMAE ...")
    videomae = VideoMAEOfficialWrapper(
        model_root=cfg.videomae_root,
        device=cfg.device,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        tubelet_size=cfg.tubelet_size,
        weight_path=getattr(cfg, "videomae_weight_path", "") or None,
    )

    print("[4/6] 初始化 RAFT ...")
    raft_estimator = RAFTFlowEstimator(
        raft_root=cfg.raft_root,
        ckpt_path=cfg.raft_ckpt,
        device=cfg.device,
        iters=cfg.raft_iters,
    )

    debug_dir = Path(cfg.output_dir) / "debug_maps"
    debug_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "config": {
            "clip_len": cfg.clip_len,
            "tubelet_size": cfg.tubelet_size,
            "image_size": cfg.image_size,
            "patch_size": cfg.patch_size,
            "block_grid": cfg.block_grid,
            "lambda_similarity": cfg.lambda_similarity,
            "sigma_r": cfg.sigma_r,
            "motion_norm": cfg.motion_norm,
            "dp_delta": cfg.dp_delta,
            "dp_top_k": cfg.dp_top_k,
            "dp_C_blk": cfg.dp_C_blk,
            "dp_sigma": cfg.dp_sigma,
            "mask_mode": getattr(cfg, "mask_mode", "private_risk_experimental"),
            "external_mask_path": getattr(cfg, "external_mask_path", ""),
            "input_video": cfg.input_video,
            "input_sha256": _sha256_if_exists(cfg.input_video),
            "videomae_root": cfg.videomae_root,
            "videomae_weight_path": getattr(cfg, "videomae_weight_path", ""),
            "videomae_weight_sha256": _sha256_if_exists(getattr(cfg, "videomae_weight_path", "")),
            "raft_root": cfg.raft_root,
            "raft_ckpt": cfg.raft_ckpt,
            "raft_ckpt_sha256": _sha256_if_exists(cfg.raft_ckpt),
            "device": cfg.device,
            "raft_iters": cfg.raft_iters,
            "enable_flow_merge_smoothing": cfg.enable_flow_merge_smoothing,
            "rdp_alphas": list(cfg.rdp_alphas),
            "note": (
                "Only the final merged video is released. "
                "The formal theorem is fixed-mask/mask-conditional: it applies only when "
                "the mask sequence is externally supplied or fixed before the release. "
                "Data-dependent risk masks remain experimental unless separately privatized/proven."
            ),
        },
        "clips_internal": [],
        "final_release_dp": {},
    }

    mean = torch.tensor(cfg.mean, device=cfg.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(cfg.std, device=cfg.device).view(1, 1, 3, 1, 1)

    print("[5/6] 逐 clip 处理（内部中间结果，不计入最终DP）...")
    all_clip_outputs = []
    all_clip_frame_scores: List[Tuple[int, List[np.ndarray]]] = []
    prev_tail_sigma_map: Optional[np.ndarray] = None

    for idx, (start_idx, clip_frames) in enumerate(clips):
        print(f"  -> clip {idx + 1}/{len(clips)} (start={start_idx})")

        clip_frames = _pad_clip_to_tubelet_multiple(clip_frames, cfg.tubelet_size)
        processed_frames, metas = preprocess_frames(clip_frames, target_size=cfg.image_size)
        flows = raft_estimator.compute_video_flows(processed_frames)

        clip_array = np.stack(processed_frames, axis=0)  # [T,H,W,3]
        clip_rgb = torch.from_numpy(clip_array).float() / 255.0
        clip_rgb = clip_rgb.permute(0, 3, 1, 2).unsqueeze(0).to(cfg.device)  # [1,T,3,H,W]
        clip_norm = (clip_rgb - mean) / std

        tokens = _handle_oom_and_retry(
            videomae.extract_patch_tokens,
            clip_norm,
            videomae=videomae,
        )[0]  # [L,D]

        clipped_tokens = clip_l2(tokens, cfg.C_h)

        noisy_risk_map, clean_risk_map, sim_map, motion_map = compute_block_risk_map(
            patch_features=clipped_tokens,
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
        
        trajectories, confidence_map = build_block_trajectories(
            flows=flows,
            image_size=cfg.image_size,
            block_grid=cfg.block_grid,
            tubelet_size=cfg.tubelet_size,
            tau_conf=cfg.tau_conf,
        )


        gen = torch.Generator(device=noisy_risk_map.device)
        gen.manual_seed(cfg.seed + idx)

        traj_scores, traj_scores_noisy, traj_weights, raw_weight_map, smooth_weight_map = trajectory_private_weights(
            risk_map=clean_risk_map,
            trajectories=trajectories,
            confidence_map=confidence_map.to(noisy_risk_map.device),
            lambda_s=cfg.lambda_s,
            sigma_s=cfg.sigma_s,
            beta=cfg.beta_traj,
            eta_smooth=max(0.25, float(cfg.eta_smooth)),
            generator=gen,
        )
        smooth_weight_map_np = smooth_weight_map.detach().cpu().numpy().reshape(
            smooth_weight_map.shape[0], cfg.block_grid, cfg.block_grid
        ).astype(np.float32)

        clip_T = len(processed_frames)
        per_frame_scores: List[np.ndarray] = []
        for t in range(clip_T):
            u = t / cfg.tubelet_size
            u0 = int(np.floor(u))
            u1 = min(u0 + 1, smooth_weight_map_np.shape[0] - 1)
            alpha = u - u0

            score = (1 - alpha) * smooth_weight_map_np[u0] + alpha * smooth_weight_map_np[u1]
            per_frame_scores.append(score)

        all_clip_frame_scores.append((start_idx, per_frame_scores))

        # 非DP按block DCT扰动（仅用于生成内部中间结果）
        perturbed_processed_frames, sigma_map_np = perturb_clip_in_dct_domain(
            frames_rgb=processed_frames,
            weight_map=smooth_weight_map_np,
            image_size=cfg.image_size,
            block_grid=cfg.block_grid,
            tubelet_size=cfg.tubelet_size,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            gamma=cfg.gamma,
            seed=cfg.seed,
            frame_offset=start_idx,
            prev_sigma_tail=prev_tail_sigma_map,
            spatial_kernel=3,
            temporal_ema=0.85,
            effect_gain=1.0,
            freq_low_cut=0.18,
            freq_power=2.0,
            noise_soft_clip_k=4.0,
        )

        save_block_heatmap_overlay(
            processed_frames[0],
            smooth_weight_map_np[0],
            str(debug_dir / f"clip_{idx:03d}_seg_00_weight.png"),
            alpha=0.50,
            draw_grid=True,
        )
        save_block_heatmap_overlay(
            processed_frames[0],
            sigma_map_np[0],
            str(debug_dir / f"clip_{idx:03d}_seg_00_sigma.png"),
            alpha=0.50,
            draw_grid=True,
        )

        prev_tail_sigma_map = sigma_map_np[-1].copy()

        restored_frames = [restore_frame_to_original(frame, meta) for frame, meta in zip(perturbed_processed_frames, metas)]
        all_clip_outputs.append((start_idx, restored_frames))

        report["clips_internal"].append({
            "clip_index": idx,
            "start_idx": start_idx,
            "note": "Internal intermediate clip output. Not counted in final DP.",
        })

    print("[6/6] merge + FINAL strict DP release (masked top-k blocks in block-DCT space) ...")
    intermediate_merged = merge_clips_with_flow_guidance(
        all_clip_outputs,
        total_len=len(raw_frames),
        clip_len=cfg.clip_len,
        stride=max(1, cfg.clip_len // 2),
        raft_estimator=raft_estimator,
        enable_flow_smoothing=cfg.enable_flow_merge_smoothing,
        temporal_alpha=0.82,
        flow_alpha=0.72,
    )

    merged_scores_per_frame = merge_frame_score_maps(
        clips_scores=all_clip_frame_scores,
        total_len=len(raw_frames),
        clip_len=cfg.clip_len,
    )
    release_scores_per_frame, mask_info = _resolve_release_score_maps(cfg, merged_scores_per_frame)

    write_video_frames(intermediate_merged, cfg.intermediate_video_path, fps=fps)

    released_frames, dp_stats = perturb_video_masked_blocks_dct_dp_temporal(
        frames_rgb=intermediate_merged,
        score_maps_per_frame=release_scores_per_frame,
        block_grid=cfg.block_grid,
        top_k=cfg.dp_top_k,
        C_blk=cfg.dp_C_blk,
        sigma=cfg.dp_sigma,
        seed=cfg.seed,
        temporal_smooth=0.0,
        alpha=1.0,
    )
  
    sigma_release = float(cfg.dp_sigma)
    sigma_account = sigma_release


    print(f"[DP-Accounting] T={len(intermediate_merged)}, iid_sigma={sigma_account}")

    eps_best, alpha_best, rho_best, Delta = eps_from_masked_block_gaussian_topk(
        T=len(intermediate_merged),
        k=cfg.dp_top_k,
        C_blk=cfg.dp_C_blk,
        sigma=sigma_account,
        delta=cfg.dp_delta,
        alphas=cfg.rdp_alphas,
        release_stride=cfg.release_stride,
    )

    report["final_release_dp"] = {
        "mechanism": "fixed_mask_topk_y_lowfreq_dct_iid_gaussian_postprocessed",
        "adjacency": (
            "same_length; conditioned_on_fixed_mask_P; per_frame_only_selected_blocks_may_differ; "
            "release_query_is_clipped_2x2_Y_DCT_vector; unselected_blocks_identical"
        ),
        "theorem_scope": mask_info.get("theorem_scope"),
        "mask_info": mask_info,
        "dp_delta": float(cfg.dp_delta),
        "dp_top_k": int(cfg.dp_top_k),
        "dp_C_blk": float(cfg.dp_C_blk),
        "dp_sigma": float(cfg.dp_sigma),
        "T": int(len(intermediate_merged)),
        "Delta_upper_bound": float(Delta),
        "avg_clip_scale": float(dp_stats.get("avg_clip_scale", 1.0)),
        "epsilon_best": float(eps_best),
        "alpha_best": float(alpha_best),
        "rho_best": float(rho_best),
        "note": (
            "The accounting is valid for the fixed-mask conditional release query. "
            "If mask_mode=private_risk_experimental, the data-dependent mask selection is not covered "
            "by this theorem and must be reported as an experimental risk-localization pipeline."
        ),
        "dp_sigma_release": float(sigma_release),
        "dp_sigma_account": float(sigma_account),
        "dp_sigma_account_kappa": 1.0,
        "note_sigma": (
            "Release uses iid Gaussian noise. Any temporal/spatial smoothing must be applied only "
            "after this DP release as post-processing; no kappa heuristic is used."
        ),
        "release_stats": dp_stats,
     
        "num_compositions": int(len(intermediate_merged)),
    }

    write_video_frames(released_frames, cfg.final_video_path, fps=fps)
    report["intermediate_video"] = cfg.intermediate_video_path
    report["intermediate_video_sha256"] = _sha256_if_exists(cfg.intermediate_video_path)
    report["released_video"] = cfg.final_video_path
    report["released_video_sha256"] = _sha256_if_exists(cfg.final_video_path)

    with open(cfg.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Done] Intermediate (non-DP) merged video: {cfg.intermediate_video_path}")
    print(f"[Done] Released DP video: {cfg.final_video_path}")
    print(f"[Done] Report: {cfg.report_path}")
    print(f"[DP] eps={eps_best:.6f}, delta={cfg.dp_delta}, best_alpha={alpha_best}, Delta={Delta:.3f}")


if __name__ == "__main__":
    main()
