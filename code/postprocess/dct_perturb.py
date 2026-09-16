from typing import List, Tuple, Optional, Dict, Sequence
import math

import cv2
import numpy as np
from postprocess.flow_warp import warp_frame_by_flow


def _split_bounds(length: int, parts: int) -> List[Tuple[int, int]]:
    if parts <= 0:
        raise ValueError("parts 必须 > 0")
    idx_splits = np.array_split(np.arange(length), parts)
    bounds: List[Tuple[int, int]] = []
    last_end = 0
    for arr in idx_splits:
        if len(arr) == 0:
            bounds.append((last_end, last_end))
            continue
        start = int(arr[0])
        end = int(arr[-1]) + 1
        if start < last_end:
            start = last_end
        bounds.append((start, end))
        last_end = end
    if len(bounds) < parts:
        bounds.extend([(last_end, last_end)] * (parts - len(bounds)))
    return bounds


def _robust_normalize_01(
    w: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95
) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    lo = float(np.quantile(w, q_low))
    hi = float(np.quantile(w, q_high))
    if hi - lo < 1e-8:
        return np.clip(w, 0.0, 1.0)
    w2 = (w - lo) / (hi - lo)
    return np.clip(w2, 0.0, 1.0)


def weights_to_sigma_map(
    weight_map: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    gamma: float,
    invert: bool = False,
    robust_q_low: float = 0.05,
    robust_q_high: float = 0.95,
) -> np.ndarray:
    w = np.asarray(weight_map, dtype=np.float32)
    if w.ndim != 3:
        raise ValueError(f"weight_map 期望 [U,B,B]，实际是 {w.shape}")

    w = _robust_normalize_01(w, q_low=robust_q_low, q_high=robust_q_high)
    w = np.clip(w, 0.0, 1.0)

    if invert:
        w = 1.0 - w

    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma 必须 > 0")
    w = np.power(w, gamma)

    sigma_min = float(sigma_min)
    sigma_max = float(sigma_max)
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max 必须 > sigma_min")

    sigma = sigma_min + (sigma_max - sigma_min) * w
    return sigma.astype(np.float32)


def _smooth_sigma_sequence(
    sigma_map: np.ndarray,
    spatial_kernel: int = 3,
    temporal_ema: float = 0.85
) -> np.ndarray:
    if sigma_map.ndim != 3:
        raise ValueError(f"sigma_map 期望 [U,B,B]，实际是 {sigma_map.shape}")

    U, B, _ = sigma_map.shape
    out = np.empty_like(sigma_map, dtype=np.float32)
    for u in range(U):
        cur = sigma_map[u].astype(np.float32)
        if spatial_kernel > 1:
            cur = cv2.blur(cur, (spatial_kernel, spatial_kernel))
        if u == 0:
            out[u] = cur
        else:
            out[u] = temporal_ema * cur + (1.0 - temporal_ema) * out[u - 1]
    return np.clip(out, 1e-6, None).astype(np.float32)


def _frequency_mask(h: int, w: int, low_cut: float = 0.18, power: float = 2.0) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    freq = (yy + xx) * 0.5
    x = np.clip((freq - low_cut) / max(1.0 - low_cut, 1e-6), 0.0, 1.0)
    m = np.exp(-(x ** float(power)))
    return m.astype(np.float32)


def _soft_clip_noise(noise: np.ndarray, clip_k: float) -> np.ndarray:
    if clip_k is None or clip_k <= 0:
        return noise
    return np.tanh(noise / clip_k) * clip_k


def _seed_mix(seed: int, *parts: int) -> int:
    x = np.uint64(seed)
    for p in parts:
        x ^= np.uint64((int(p) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
    return int(x % np.uint64(2**32 - 1))


def _rgb_to_ycbcr01(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]

    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.500000 * b)
    cr = 0.5 + (0.500000 * r - 0.418688 * g - 0.081312 * b)

    out = np.stack([y, cb, cr], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _ycbcr01_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    x = np.asarray(ycbcr, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)

    y = x[..., 0]
    cb = x[..., 1] - 0.5
    cr = x[..., 2] - 0.5

    r = y + 1.402000 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772000 * cb

    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _dct2(x: np.ndarray) -> np.ndarray:
    return cv2.dct(np.ascontiguousarray(x.astype(np.float32)))


def _idct2(x: np.ndarray) -> np.ndarray:
    return cv2.idct(np.ascontiguousarray(x.astype(np.float32)))


def _lowfreq_keep_dims(h: int, w: int, keep_ratio: float) -> Tuple[int, int]:
    keep_ratio = float(keep_ratio)
    if keep_ratio <= 0:
        raise ValueError("keep_ratio 必须 > 0")
    kh = max(1, int(round(h * keep_ratio)))
    kw = max(1, int(round(w * keep_ratio)))
    kh = min(kh, h)
    kw = min(kw, w)
    return kh, kw


def _extract_lowfreq_vec(coeff: np.ndarray, keep_h: int, keep_w: int) -> np.ndarray:
    return coeff[:keep_h, :keep_w].reshape(-1)


def _insert_lowfreq_vec(
    coeff_shape: Tuple[int, int],
    vec: np.ndarray,
    keep_h: int,
    keep_w: int,
) -> np.ndarray:
    out = np.zeros(coeff_shape, dtype=np.float32)
    out[:keep_h, :keep_w] = vec.reshape(keep_h, keep_w)
    return out


def _topk_block_indices(score_map: np.ndarray, block_grid: int, top_k: int) -> List[Tuple[int, int]]:
    score_map = np.asarray(score_map, dtype=np.float32)
    if score_map.shape != (block_grid, block_grid):
        raise ValueError(f"score_map must be {(block_grid, block_grid)}, got {score_map.shape}")
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    top_k = min(top_k, block_grid * block_grid)
    flat_idx = np.argsort(score_map.reshape(-1))[-top_k:]
    return [(int(idx // block_grid), int(idx % block_grid)) for idx in flat_idx]


def extract_masked_block_lowfreq_query(
    frame_rgb: np.ndarray,
    score_map: np.ndarray,
    block_grid: int,
    top_k: int,
    C_blk: float,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[float]]:
    """Return the clipped 2x2 Y-DCT query used by the fixed-mask release."""
    frame = np.asarray(frame_rgb, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame_rgb must be [H,W,3], got {frame.shape}")
    if frame.max() > 1.0:
        frame = frame / 255.0

    H, W = frame.shape[:2]
    y_bounds = _split_bounds(H, block_grid)
    x_bounds = _split_bounds(W, block_grid)
    Y = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]

    coords = _topk_block_indices(score_map, block_grid=block_grid, top_k=top_k)
    vecs: List[np.ndarray] = []
    clip_scales: List[float] = []
    for bi, bj in coords:
        y0, y1 = y_bounds[bi]
        x0, x1 = x_bounds[bj]
        block = Y[y0:y1, x0:x1]
        if block.size == 0:
            vec = np.zeros((4,), dtype=np.float32)
            scale = 1.0
        else:
            dct = cv2.dct(np.ascontiguousarray(block.astype(np.float32)))
            vec = dct[:2, :2].reshape(-1).astype(np.float32)
            norm = np.linalg.norm(vec)
            scale = min(1.0, float(C_blk) / (float(norm) + 1e-6))
            vec = vec * scale
        vecs.append(vec.astype(np.float32))
        clip_scales.append(float(scale))
    return np.stack(vecs, axis=0), coords, clip_scales


def _soft_post_smooth_rgb_block(
    block_rgb01: np.ndarray,
    ksize: int = 3,
    blend: float = 0.18,
) -> np.ndarray:
 
    if ksize is None or int(ksize) <= 1 or blend <= 0:
        return np.clip(block_rgb01, 0.0, 1.0).astype(np.float32)

    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1

    x = np.clip(block_rgb01.astype(np.float32), 0.0, 1.0)
    x_u8 = (x * 255.0).round().clip(0, 255).astype(np.uint8)
    blur_u8 = cv2.GaussianBlur(x_u8, (ksize, ksize), 0)
    blur = blur_u8.astype(np.float32) / 255.0

    out = (1.0 - float(blend)) * x + float(blend) * blur
    return np.clip(out, 0.0, 1.0).astype(np.float32)



def perturb_clip_in_dct_domain(
    frames_rgb: List[np.ndarray],
    weight_map: np.ndarray,
    image_size: int,
    block_grid: int,
    tubelet_size: int,
    sigma_min: float,
    sigma_max: float,
    gamma: float,
    seed: int = 0,
    frame_offset: int = 0,
    prev_sigma_tail: Optional[np.ndarray] = None,
    spatial_kernel: int = 3,
    temporal_ema: float = 0.85,
    effect_gain: float = 1.0,
    freq_low_cut: float = 0.18,
    freq_power: float = 2.0,
    noise_soft_clip_k: float = 4.0,
) -> Tuple[List[np.ndarray], np.ndarray]:

    if len(frames_rgb) == 0:
        return [], np.zeros_like(weight_map, dtype=np.float32)

    sigma_map = weights_to_sigma_map(
        weight_map=weight_map,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        gamma=gamma,
        invert=False,
    )

    effect_gain = float(effect_gain)
    if effect_gain <= 0:
        effect_gain = 1.0
    sigma_map = sigma_map * effect_gain

    sigma_map = _smooth_sigma_sequence(
        sigma_map,
        spatial_kernel=spatial_kernel,
        temporal_ema=temporal_ema,
    )

    if prev_sigma_tail is not None:
        prev_sigma_tail = np.asarray(prev_sigma_tail, dtype=np.float32)
        if prev_sigma_tail.shape == sigma_map[0].shape:
            sigma_map[0] = temporal_ema * sigma_map[0] + (1.0 - temporal_ema) * prev_sigma_tail
            sigma_map[0] = np.clip(sigma_map[0], 1e-6, None)

    h, w = frames_rgb[0].shape[:2]
    if h != image_size or w != image_size:
        raise ValueError(f"输入帧尺寸必须已经是 {image_size}x{image_size}")

    y_bounds = _split_bounds(image_size, block_grid)
    x_bounds = _split_bounds(image_size, block_grid)

    out_frames: List[np.ndarray] = []

    for t, frame in enumerate(frames_rgb):
        seg = min(t // tubelet_size, sigma_map.shape[0] - 1)
        frame_global_idx = frame_offset + t

        frame_f = frame.astype(np.float32) / 255.0
        out = frame_f.copy()

        for bi, (y0, y1) in enumerate(y_bounds):
            for bj, (x0, x1) in enumerate(x_bounds):
                patch = frame_f[y0:y1, x0:x1, :]
                if patch.size == 0:
                    continue

                sigma = float(sigma_map[seg, bi, bj])
                if sigma <= 0:
                    out[y0:y1, x0:x1, :] = patch
                    continue

                freq_mask = _frequency_mask(
                    patch.shape[0],
                    patch.shape[1],
                    low_cut=freq_low_cut,
                    power=freq_power,
                )
                recon = np.empty_like(patch, dtype=np.float32)

                for c in range(3):
                    coeff = _dct2(np.ascontiguousarray(patch[:, :, c].astype(np.float32)))

                    block_seed = _seed_mix(seed, frame_global_idx, seg, bi, bj, c)
                    local_rng = np.random.default_rng(block_seed)

                    noise = local_rng.normal(
                        loc=0.0,
                        scale=sigma,
                        size=coeff.shape
                    ).astype(np.float32)
                    noise = noise * freq_mask
                    noise = _soft_clip_noise(noise, noise_soft_clip_k)

                    coeff_noisy = coeff + noise
                    recon[:, :, c] = _idct2(np.ascontiguousarray(coeff_noisy))

                out[y0:y1, x0:x1, :] = np.clip(recon, 0.0, 1.0)

        out_frames.append((out * 255.0).round().clip(0, 255).astype(np.uint8))

    return out_frames, sigma_map




def eps_from_masked_block_gaussian_topk(
    T: int,
    k: int,
    C_blk: float,
    sigma: float,
    delta: float,
    alphas: Sequence[float],
    release_stride: int = 4,
) -> Tuple[float, float, float, float]:
   
    T = int(T)
    k = int(k)
    C_blk = float(C_blk)
    sigma = float(sigma)
    release_stride = int(release_stride)
    delta = float(min(max(delta, 1e-12), 1.0 - 1e-12))

    if T <= 0:
        raise ValueError("T must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")
    if C_blk <= 0:
        raise ValueError("C_blk must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if release_stride <= 0:
        raise ValueError("release_stride must be > 0")

    print("[ACCOUNT-FUNC] sigma =", sigma, "T =", T, "k =", k, "C_blk =", C_blk, "delta =", delta)
    private_frames = int(T)
    private_frames = max(1, private_frames)

    Delta_call = 2.0 * C_blk * math.sqrt(float(k))

    best_eps = float("inf")
    best_alpha = None
    best_rho = None

    for a in alphas:
        a = float(a)
        if a <= 1.0:
            continue
        rho_one = a * (Delta_call ** 2) / (2.0 * (sigma ** 2))
        rho_total = private_frames * rho_one
        eps = rho_total + math.log(1.0 / delta) / (a - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_alpha = a
            best_rho = rho_total

    return float(best_eps), float(best_alpha), float(best_rho), float(Delta_call)

def perturb_video_masked_blocks_dct_dp_temporal(
    frames_rgb,
    score_maps_per_frame,
    block_grid: int,
    top_k: int,
    C_blk: float,
    sigma: float,
    temporal_smooth: float = 0.0,
    seed: int = 0,
    alpha: float = 1.0,
    neutral_chroma: float = 0.5,
):
    """Release selected blocks through an iid Gaussian mechanism.

    The released content of each selected block is reconstructed only from the
    clipped/noised 2x2 Y-channel DCT vector plus fixed chroma and zero
    high-frequency coefficients. This intentionally avoids leaking raw
    high-frequency or chroma content through the final video.

    ``temporal_smooth`` is now a post-processing knob applied to already
    released frames. It is not used to correlate the Gaussian noise.
    """

    print("[RELEASE] iid sigma =", sigma)
    rng = np.random.default_rng(seed)
    T = len(frames_rgb)
    H, W = frames_rgb[0].shape[:2]
    y_bounds = _split_bounds(H, block_grid)
    x_bounds = _split_bounds(W, block_grid)

    out_frames = []
    clip_scales = []
    selected_blocks = 0

    if abs(float(alpha) - 1.0) > 1e-8:
        print("[RELEASE] alpha mixing with raw blocks is disabled for DP; using alpha=1.0.")

    for t, (frame, score_map) in enumerate(zip(frames_rgb, score_maps_per_frame)):
        frame = frame.astype(np.float32) / 255.0

        Y = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        Cb = -0.168736 * frame[..., 0] - 0.331264 * frame[..., 1] + 0.5 * frame[..., 2] + 0.5
        Cr = 0.5 * frame[..., 0] - 0.418688 * frame[..., 1] - 0.081312 * frame[..., 2] + 0.5

        query_vecs, selected_coords, frame_clip_scales = extract_masked_block_lowfreq_query(
            frame_rgb=frame,
            score_map=score_map,
            block_grid=block_grid,
            top_k=top_k,
            C_blk=C_blk,
        )
        clip_scales.extend(frame_clip_scales)

        for vec, (bi, bj) in zip(query_vecs, selected_coords):
            y0, y1 = y_bounds[bi]
            x0, x1 = x_bounds[bj]
            block = Y[y0:y1, x0:x1]
            if block.size == 0:
                continue

            noise = rng.normal(0.0, float(sigma), size=vec.shape).astype(np.float32)
            vec_noisy = vec + noise

            dct_release = np.zeros_like(block, dtype=np.float32)
            dct_release[:2, :2] = vec_noisy.reshape(2, 2)
            noisy_block = cv2.idct(dct_release)

            Y[y0:y1, x0:x1] = noisy_block
            Cb[y0:y1, x0:x1] = float(neutral_chroma)
            Cr[y0:y1, x0:x1] = float(neutral_chroma)
            selected_blocks += 1

        R = Y + 1.402 * (Cr - 0.5)
        G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
        B = Y + 1.772 * (Cb - 0.5)
        rgb = np.stack([R, G, B], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        out_frames.append((rgb * 255).astype(np.uint8))

    if temporal_smooth is not None and float(temporal_smooth) > 0.0 and len(out_frames) > 1:
        ema = float(np.clip(temporal_smooth, 0.0, 0.99))
        smoothed = [out_frames[0]]
        prev = out_frames[0].astype(np.float32)
        for frame in out_frames[1:]:
            cur = frame.astype(np.float32)
            post = ema * prev + (1.0 - ema) * cur
            post_u8 = post.round().clip(0, 255).astype(np.uint8)
            smoothed.append(post_u8)
            prev = post
        out_frames = smoothed

    stats = {
        "avg_clip_scale": float(np.mean(clip_scales)) if clip_scales else 1.0,
        "selected_blocks": int(selected_blocks),
        "noise_distribution": "iid_gaussian",
        "raw_alpha_mixing_used": False,
        "selected_block_high_frequency": "zeroed_before_inverse_dct",
        "selected_block_chroma": "fixed_neutral_value",
        "postprocess_temporal_smooth": float(temporal_smooth or 0.0),
    }
    return out_frames, stats


import numpy as np

def build_dct_freq_weight(N: int,
                          r_low: float,
                          r_high: float,
                          mode: str = "cosine",
                          dc_zero: bool = True) -> np.ndarray:

    uu, vv = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt(uu * uu + vv * vv)

    w = np.zeros((N, N), dtype=np.float32)

    if r_high <= r_low:
        w[r >= r_low] = 1.0
    else:
        t = (r - r_low) / (r_high - r_low)   # 低于0->低频, 高于1->高频
        t = np.clip(t, 0.0, 1.0)
        if mode == "linear":
            w = t.astype(np.float32)
        elif mode == "cosine":
            w = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
        else:
            raise ValueError(f"unknown mode: {mode}")

    if dc_zero:
        w[0, 0] = 0.0
    return w
