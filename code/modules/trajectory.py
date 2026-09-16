
from typing import List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F


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


def _block_centers(image_size: int, block_grid: int) -> np.ndarray:
    y_bounds = _split_bounds(image_size, block_grid)
    x_bounds = _split_bounds(image_size, block_grid)
    centers = []
    for y0, y1 in y_bounds:
        for x0, x1 in x_bounds:
            centers.append(((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0))
    return np.asarray(centers, dtype=np.float32)


def _block_region(image_size: int, block_grid: int, block_idx: int) -> Tuple[int, int, int, int]:
    y_bounds = _split_bounds(image_size, block_grid)
    x_bounds = _split_bounds(image_size, block_grid)
    bi = block_idx // block_grid
    bj = block_idx % block_grid
    y0, y1 = y_bounds[bi]
    x0, x1 = x_bounds[bj]
    return y0, y1, x0, x1


def _nearest_block(center_xy: Tuple[float, float], centers: np.ndarray) -> int:
    x, y = center_xy
    d = (centers[:, 0] - x) ** 2 + (centers[:, 1] - y) ** 2
    return int(np.argmin(d))


def _smooth_grid_map(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    对 [U, B^2] 或 [U, B, B] 做空间平滑。
    """
    if kernel_size <= 1:
        return x

    orig_dim = x.dim()
    if orig_dim == 2:
        U, G2 = x.shape
        B = int(math.sqrt(G2))
        if B * B != G2:
            raise ValueError(f"无法将 [U, {G2}] reshape 为正方形网格")
        x = x.view(U, B, B)
    elif orig_dim != 3:
        raise ValueError(f"_smooth_grid_map 期望 [U,B^2] 或 [U,B,B]，实际是 {x.shape}")

    pad = kernel_size // 2
    x4 = x.unsqueeze(1)
    x4 = F.pad(x4, (pad, pad, pad, pad), mode="reflect")
    x4 = F.avg_pool2d(x4, kernel_size=kernel_size, stride=1)
    out = x4.squeeze(1)

    if orig_dim == 2:
        out = out.view(x.shape[0], -1)
    return out


def build_block_trajectories(
    flows: List[np.ndarray],
    image_size: int,
    block_grid: int,
    tubelet_size: int,
    tau_conf: float = 12.0,
) -> Tuple[List[List[int]], torch.Tensor]:
    """
    用 RAFT 光流构建 block 轨迹。

    修正版：
      - 每个 segment 不只看单一边界 flow，而是平均 segment 内的 flows；
      - 降低单帧流噪声对轨迹的扰动。
    """
    num_frames = len(flows) + 1
    U = int(math.ceil(num_frames / float(tubelet_size)))
    G2 = block_grid * block_grid

    centers = _block_centers(image_size, block_grid)
    next_index_map = np.zeros((max(U - 1, 0), G2), dtype=np.int64)
    confidence_map = torch.ones(U, G2, dtype=torch.float32)

    if len(flows) == 0:
        trajectories = [[i] * U for i in range(G2)]
        return trajectories, confidence_map

    for seg in range(1, U):
        start_f = (seg - 1) * tubelet_size
        end_f = min(seg * tubelet_size - 1, len(flows) - 1)
        flow_ids = list(range(start_f, end_f + 1)) if start_f <= end_f else []

        if len(flow_ids) == 0:
            flow = flows[min(len(flows) - 1, max(0, seg * tubelet_size - 1))].astype(np.float32)
        else:
            flow_stack = np.stack([flows[fid].astype(np.float32) for fid in flow_ids], axis=0)
            flow = flow_stack.mean(axis=0)

        for prev_idx in range(G2):
            y0, y1, x0, x1 = _block_region(image_size, block_grid, prev_idx)
            flow_block = flow[y0:y1, x0:x1, :]
            if flow_block.size == 0:
                cur_idx = prev_idx
                conf = 1.0
            else:
                dx = float(flow_block[..., 0].mean())
                dy = float(flow_block[..., 1].mean())
                mag = float(np.linalg.norm(flow_block, axis=-1).mean())
                prev_center = centers[prev_idx]
                cur_center = (prev_center[0] + dx, prev_center[1] + dy)
                cur_idx = _nearest_block(cur_center, centers)
                conf = float(np.clip(np.exp(-mag / max(tau_conf, 1e-6)), 0.0, 1.0))

            next_index_map[seg - 1, prev_idx] = cur_idx
            confidence_map[seg, cur_idx] = max(float(confidence_map[seg, cur_idx].item()), conf)

    trajectories: List[List[int]] = []
    for start_block in range(G2):
        path = [start_block]
        cur = start_block
        for seg in range(U - 1):
            cur = int(next_index_map[seg, cur])
            path.append(cur)
        trajectories.append(path)

    return trajectories, confidence_map


def trajectory_private_weights(
    risk_map: torch.Tensor,
    trajectories: List[List[int]],
    confidence_map: torch.Tensor,
    lambda_s: float,
    sigma_s: float,
    beta: float,
    eta_smooth: float,
    generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    修正版：
      1) traj score 用 mean + top-25% mean，降低极值抖动；
      2) raw_weight_map 和 smooth_weight_map 做空间平滑；
      3) eta_smooth 默认不再为 0。
    """
    if risk_map.dim() != 3:
        raise ValueError("risk_map 必须是 [U, B, B]")

    device = risk_map.device
    U, B1, B2 = risk_map.shape
    if B1 != B2:
        raise ValueError("risk_map 最后两个维度必须相同")

    G2 = B1 * B1
    flat_risk = risk_map.reshape(U, G2)
    confidence_map = confidence_map.to(device=device, dtype=torch.float32)

    traj_scores = []
    for path in trajectories:
        vals = torch.stack([flat_risk[u, int(path[u])] for u in range(U)], dim=0)

        k_high = max(1, int(math.ceil(0.25 * vals.numel())))
        top_vals = torch.topk(vals, k=k_high, largest=True).values
        high_stat = top_vals.mean()

        score = lambda_s * vals.mean() + (1.0 - lambda_s) * high_stat
        traj_scores.append(score.clamp(0.0, 1.0))
    traj_scores = torch.stack(traj_scores, dim=0)

    if generator is not None:
        noise = torch.randn(
            traj_scores.shape,
            device=device,
            generator=generator,
            dtype=traj_scores.dtype,
        ) * sigma_s
    else:
        noise = torch.randn(
            traj_scores.shape,
            device=device,
            dtype=traj_scores.dtype,
        ) * sigma_s

    traj_scores_noisy = (traj_scores + noise).clamp(0.0, 1.0)
    traj_weights = torch.softmax(beta * traj_scores_noisy, dim=0)

    raw_weight_sum = torch.zeros(U, G2, device=device)
    raw_weight_count = torch.zeros(U, G2, device=device)

    for j, path in enumerate(trajectories):
        for u in range(U):
            idx = int(path[u])
            raw_weight_sum[u, idx] += traj_weights[j]
            raw_weight_count[u, idx] += 1.0

    raw_weight_map = torch.where(
        raw_weight_count > 0,
        raw_weight_sum / raw_weight_count.clamp_min(1.0),
        raw_weight_sum,
    ).clamp(0.0, 1.0)

    raw_weight_map = _smooth_grid_map(raw_weight_map, kernel_size=3).clamp(0.0, 1.0)

    smooth_sum = torch.zeros_like(raw_weight_map)
    smooth_count = torch.zeros_like(raw_weight_map)

    for j, path in enumerate(trajectories):
        idx0 = int(path[0])
        bar = raw_weight_map[0, idx0]
        smooth_sum[0, idx0] += bar
        smooth_count[0, idx0] += 1.0

        for u in range(1, U):
            idx = int(path[u])
            c = confidence_map[u, idx]
            w = raw_weight_map[u, idx]
            bar = eta_smooth * c * w + (1.0 - eta_smooth * c) * bar
            smooth_sum[u, idx] += bar
            smooth_count[u, idx] += 1.0

    smooth_weight_map = torch.where(
        smooth_count > 0,
        smooth_sum / smooth_count.clamp_min(1.0),
        raw_weight_map,
    ).clamp(0.0, 1.0)

    smooth_weight_map = _smooth_grid_map(smooth_weight_map, kernel_size=3).clamp(0.0, 1.0)

    return traj_scores, traj_scores_noisy, traj_weights, raw_weight_map, smooth_weight_map
