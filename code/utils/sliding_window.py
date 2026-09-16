
from __future__ import annotations

from typing import List, Tuple, Optional

import cv2
import numpy as np

from postprocess.flow_warp import warp_frame_by_flow


def split_into_clips(frames, clip_len: int = 16, stride: int = 8):
    """
    生成重叠滑窗 clip。
    返回 [(start_idx, clip_frames), ...]
    """
    if clip_len <= 0:
        raise ValueError("clip_len 必须 > 0")
    if stride <= 0:
        raise ValueError("stride 必须 > 0")
    if len(frames) == 0:
        return []

    clips = []
    n = len(frames)

    for start in range(0, n, stride):
        end = start + clip_len
        clip = list(frames[start:end])

        if len(clip) == 0:
            break
        if len(clip) < clip_len:
            clip = clip + [clip[-1].copy()] * (clip_len - len(clip))

        clips.append((start, clip))
        if end >= n:
            break

    return clips


def _clip_edge_weight(local_i: int, clip_len: int, power: float = 1.6) -> float:
    """
    中心权重大，边缘权重小。
    """
    if clip_len <= 1:
        return 1.0
    center = (clip_len - 1) / 2.0
    denom = max(center, 1e-6)
    w = 1.0 - abs(local_i - center) / denom
    w = float(np.clip(w, 0.0, 1.0))
    return float(w ** power)


def _normalize_frame(frame_f32: np.ndarray) -> np.ndarray:
    return np.clip(frame_f32, 0.0, 255.0).astype(np.uint8)


def merge_clips_with_flow_guidance(
    clips_outputs,
    total_len: int,
    clip_len: int = 16,
    stride: int = 8,
    raft_estimator=None,
    enable_flow_smoothing: bool = True,
    temporal_alpha: float = 0.82,
    flow_alpha: float = 0.72,
):
    """
    更稳的融合策略：
    1) 先对所有 clip 的同一 global frame 做加权平均；
    2) 再可选做一次 causal temporal smoothing；
    3) 若提供 RAFT，则使用 flow warp 的前向时序平滑，但只做一次，不再叠加多重平滑。

    这样可以明显降低：
    - overlap 边界闪烁
    - clip 之间噪声 realization 不一致
    - 过度重复 warping 带来的拖影
    """
    if len(clips_outputs) == 0:
        return []

    clips_outputs = sorted(clips_outputs, key=lambda x: x[0])
    frame_shape = clips_outputs[0][1][0].shape
    accum = [np.zeros(frame_shape, dtype=np.float32) for _ in range(total_len)]
    weight_sum = np.zeros((total_len,), dtype=np.float32)

    for start_idx, frames in clips_outputs:
        for local_i, curr_frame in enumerate(frames):
            global_i = start_idx + local_i
            if global_i >= total_len:
                break
            w = _clip_edge_weight(local_i, clip_len=clip_len, power=1.7)
            accum[global_i] += curr_frame.astype(np.float32) * w
            weight_sum[global_i] += w

    merged = []
    for i in range(total_len):
        if weight_sum[i] > 0:
            frame = accum[i] / max(float(weight_sum[i]), 1e-6)
        else:
            frame = accum[i - 1] if i > 0 else accum[0]
        merged.append(_normalize_frame(frame))

    if not enable_flow_smoothing or len(merged) <= 1:
        return merged

    # 只做一次 causal smoothing：当前帧与上一帧 warp 结果融合
    smoothed = [merged[0]]
    for t in range(1, len(merged)):
        cur = merged[t]
        prev = smoothed[-1]

        if raft_estimator is not None:
            try:
                flow = raft_estimator.compute_flow(prev, cur)
                prev_warp = warp_frame_by_flow(prev, flow)
            except Exception:
                prev_warp = prev
        else:
            prev_warp = prev

        fused = cv2.addWeighted(cur, float(flow_alpha), prev_warp, float(1.0 - flow_alpha), 0.0)
        fused = cv2.addWeighted(fused, float(temporal_alpha), prev, float(1.0 - temporal_alpha), 0.0)
        smoothed.append(_normalize_frame(fused.astype(np.float32)))

    return smoothed



def merge_frame_score_maps(clips_scores: list, total_len: int, clip_len: int):
    """
    简单消融版：对每帧的 score map 取平均
    clips_scores: List[Tuple[start_idx, List[np.ndarray]]]
    """
    merged = [None] * total_len
    counts = [0] * total_len
    for start_idx, score_list in clips_scores:
        for i, score in enumerate(score_list):
            idx = start_idx + i
            if idx >= total_len:
                break
            if merged[idx] is None:
                merged[idx] = np.array(score, dtype=np.float32)
            else:
                merged[idx] += score
            counts[idx] += 1
    for i in range(total_len):
        if counts[i] > 0:
            merged[i] /= counts[i]
        else:
            merged[i] = np.zeros_like(score_list[0], dtype=np.float32)
    return merged