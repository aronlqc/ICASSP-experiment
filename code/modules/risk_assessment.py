
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def _split_bounds(length: int, parts: int):
    if parts <= 0:
        raise ValueError("parts 必须 > 0")
    idx_splits = np.array_split(np.arange(length), parts)
    bounds = []
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


def _smooth_grid_map(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    对 [U, B, B] 或 [B, B] 做空间平滑。
    """
    if kernel_size <= 1:
        return x

    orig_dim = x.dim()
    if orig_dim == 2:
        x = x.unsqueeze(0)
    elif orig_dim != 3:
        raise ValueError(f"_smooth_grid_map 期望 [B,B] 或 [U,B,B]，实际是 {x.shape}")

    pad = kernel_size // 2
    x4 = x.unsqueeze(1)
    x4 = F.pad(x4, (pad, pad, pad, pad), mode="reflect")
    x4 = F.avg_pool2d(x4, kernel_size=kernel_size, stride=1)
    out = x4.squeeze(1)

    if orig_dim == 2:
        out = out.squeeze(0)
    return out


def _minmax_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def _sobel_texture_patch_map(frame_rgb_uint8: np.ndarray, patch_size: int, hp: int, wp: int) -> torch.Tensor:
    """
    计算参考帧的纹理梯度强度，并聚合到 patch 网格 [hp, wp].
    """
    if frame_rgb_uint8.dtype != np.uint8:
        frame_rgb_uint8 = frame_rgb_uint8.astype(np.uint8)

    gray = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    H, W = mag.shape
    ph = patch_size
    pw = patch_size
    Hc = (H // ph) * ph
    Wc = (W // pw) * pw
    mag = mag[:Hc, :Wc]

    mag_t = torch.from_numpy(mag).float()[None, None, :, :]
    pooled = F.avg_pool2d(mag_t, kernel_size=(ph, pw), stride=(ph, pw))
    pooled = pooled[0, 0]

    if pooled.shape[0] != hp or pooled.shape[1] != wp:
        pooled4 = pooled[None, None, :, :]
        pooled4 = F.interpolate(pooled4, size=(hp, wp), mode="bilinear", align_corners=False)
        pooled = pooled4[0, 0]

    return pooled


def compute_block_risk_map(
    patch_features: torch.Tensor,
    flows: List[np.ndarray],
    image_size: int,
    patch_size: int,
    tubelet_size: int,
    block_grid: int,
    lambda_similarity: float = 0.7,
    sigma_r: float = 0.15,
    motion_norm: float = 20.0,
    w_attn: float = 0.85,
    w_temp: float = 0.10,
    w_tex: float = 0.03,
    w_flow: float = 0.02,
    tau_attn: float = 1.0,
    reference_frame_rgb: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    输出:
      noisy_risk_map: [U,B,B]
      clean_risk_map: [U,B,B]
      sim_map:        [U,B,B]
      motion_map:     [U,B,B]
    """
    if patch_features.dim() != 2:
        raise ValueError("patch_features 必须是 [L, D]")

    device = patch_features.device
    L, D = patch_features.shape

    hp = image_size // patch_size
    wp = image_size // patch_size
    tokens_per_tubelet = hp * wp

    if tokens_per_tubelet <= 0:
        raise ValueError("patch_size 或 image_size 设置不合法")
    if L % tokens_per_tubelet != 0:
        raise ValueError(f"token 数 L={L} 不能被每个 tubelet 的 token 数 {tokens_per_tubelet} 整除")

    U = L // tokens_per_tubelet
    tokens = patch_features.view(U, hp, wp, D)

    y_bounds = _split_bounds(hp, block_grid)
    x_bounds = _split_bounds(wp, block_grid)

    R_attn_map = torch.zeros(U, block_grid, block_grid, device=device)
    R_flow_map = torch.zeros_like(R_attn_map)
    R_tex_map = torch.zeros_like(R_attn_map)
    R_temp_map = torch.zeros_like(R_attn_map)

    flow_count = len(flows)

    if reference_frame_rgb is not None:
        tex_patch = _sobel_texture_patch_map(reference_frame_rgb, patch_size=patch_size, hp=hp, wp=wp).to(device)
        tex_patch = _minmax_01(tex_patch)
    else:
        tex_patch = torch.zeros((hp, wp), device=device)

    if U <= 1:
        temp_patch = torch.zeros((U, hp, wp), device=device)
    else:
        diff = torch.norm(tokens[1:] - tokens[:-1], p=2, dim=-1)
        z0 = torch.zeros((1, hp, wp), device=device, dtype=diff.dtype)
        temp_patch = torch.cat([z0, diff], dim=0)
        temp_patch = _minmax_01(temp_patch)

    for u in range(U):
        start_flow = u * tubelet_size
        end_flow = min((u + 1) * tubelet_size - 1, flow_count - 1)
        flow_ids = list(range(start_flow, end_flow + 1)) if flow_count > 0 and start_flow <= end_flow else []

        for bi, (y0, y1) in enumerate(y_bounds):
            for bj, (x0, x1) in enumerate(x_bounds):
                feats = tokens[u, y0:y1, x0:x1, :].reshape(-1, D)
                if feats.numel() == 0:
                    continue

                # 相似性/中心性代理
                f_norm = F.normalize(feats, p=2, dim=-1)
                logits = (f_norm @ f_norm.t()) / max(float(tau_attn), 1e-6)
                P = torch.softmax(logits, dim=-1)
                attn_scalar = P.mean(dim=0).mean().clamp(0.0, 1.0)

                # 运动代理
                if len(flow_ids) == 0:
                    motion_scalar = torch.tensor(0.0, device=device)
                else:
                    vals = []
                    for fid in flow_ids:
                        flow = torch.from_numpy(flows[fid]).to(device=device, dtype=torch.float32)
                        px_y0 = y0 * patch_size
                        px_y1 = y1 * patch_size
                        px_x0 = x0 * patch_size
                        px_x1 = x1 * patch_size
                        flow_block = flow[px_y0:px_y1, px_x0:px_x1, :]
                        if flow_block.numel() == 0:
                            continue
                        mag = torch.linalg.norm(flow_block, dim=-1).mean() / max(float(motion_norm), 1e-6)
                        vals.append(mag)
                    motion_scalar = torch.stack(vals).mean() if len(vals) > 0 else torch.tensor(0.0, device=device)
                    motion_scalar = motion_scalar.clamp(0.0, 1.0)

                tex_scalar = tex_patch[y0:y1, x0:x1].mean() if (y1 > y0 and x1 > x0) else torch.tensor(0.0, device=device)
                tmp_scalar = temp_patch[u, y0:y1, x0:x1].mean() if (y1 > y0 and x1 > x0) else torch.tensor(0.0, device=device)

                R_attn_map[u, bi, bj] = attn_scalar
                R_flow_map[u, bi, bj] = motion_scalar
                R_tex_map[u, bi, bj] = tex_scalar.clamp(0.0, 1.0)
                R_temp_map[u, bi, bj] = tmp_scalar.clamp(0.0, 1.0)

    R_attn_map = _minmax_01(R_attn_map)
    R_flow_map = _minmax_01(R_flow_map)
    R_tex_map = _minmax_01(R_tex_map)
    R_temp_map = _minmax_01(R_temp_map)

    clean_risk_map = (
        float(w_attn) * R_attn_map
        + float(w_temp) * R_temp_map
        + float(w_tex) * R_tex_map
        + float(w_flow) * R_flow_map
    )
    clean_risk_map = _minmax_01(clean_risk_map).clamp(0.0, 1.0)

    clean_risk_map = _smooth_grid_map(clean_risk_map, kernel_size=3).clamp(0.0, 1.0)
    R_attn_map = _smooth_grid_map(R_attn_map, kernel_size=3).clamp(0.0, 1.0)
    R_flow_map = _smooth_grid_map(R_flow_map, kernel_size=3).clamp(0.0, 1.0)

    noisy_risk_map = (clean_risk_map + torch.randn_like(clean_risk_map) * float(sigma_r)).clamp(0.0, 1.0)

    sim_map = R_attn_map
    motion_map = R_flow_map

    return noisy_risk_map, clean_risk_map, sim_map, motion_map
