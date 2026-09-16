
import numpy as np
import cv2


def warp_frame_by_flow(frame_rgb: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    用光流把 frame warp 到下一帧坐标系。
    """
    h, w = frame_rgb.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        frame_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return warped
