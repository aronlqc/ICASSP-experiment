
import cv2
from typing import Dict
import numpy as np


def restore_frame_to_original(processed_frame: np.ndarray, meta: Dict) -> np.ndarray:
    """
    将 square canvas 帧恢复回原始尺寸。
    输入/输出均为 RGB, uint8
    """
    target_size = meta.get("target_size", processed_frame.shape[0])
    orig_h = meta.get("orig_h", target_size)
    orig_w = meta.get("orig_w", target_size)

    pad_top = int(meta.get("pad_top", 0))
    pad_left = int(meta.get("pad_left", 0))
    new_h = int(meta.get("new_h", target_size))
    new_w = int(meta.get("new_w", target_size))

    y_end = min(pad_top + new_h, processed_frame.shape[0])
    x_end = min(pad_left + new_w, processed_frame.shape[1])

    cropped = processed_frame[pad_top:y_end, pad_left:x_end]
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return restored
