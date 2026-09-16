
from typing import List, Tuple, Dict
import cv2
import numpy as np


def resize_with_aspect_ratio_and_padding(
    frame: np.ndarray,
    target_size: int = 224
) -> Tuple[np.ndarray, Dict]:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("无效帧尺寸")

    scale = target_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
    pad_top = max((target_size - new_h) // 2, 0)
    pad_bottom = max(target_size - new_h - pad_top, 0)
    pad_left = max((target_size - new_w) // 2, 0)
    pad_right = max(target_size - new_w - pad_left, 0)

    y_end = min(pad_top + new_h, target_size)
    x_end = min(pad_left + new_w, target_size)
    src_h = y_end - pad_top
    src_w = x_end - pad_left

    canvas[pad_top:y_end, pad_left:x_end] = resized[:src_h, :src_w]

    meta = {
        "orig_h": h,
        "orig_w": w,
        "scale": scale,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "target_size": target_size,
    }
    return canvas, meta


def preprocess_frames(
    frames: List[np.ndarray],
    target_size: int = 224
) -> Tuple[List[np.ndarray], List[Dict]]:
    processed = []
    metas = []
    for frame in frames:
        p, meta = resize_with_aspect_ratio_and_padding(frame, target_size=target_size)
        processed.append(p)
        metas.append(meta)
    return processed, metas
