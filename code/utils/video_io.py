
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


def read_video_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"视频中没有读到任何帧: {video_path}")

    return frames, fps


def write_video_frames(frames: List[np.ndarray], output_path: str, fps: float = 25.0) -> None:
    if len(frames) == 0:
        raise ValueError("write_video_frames 收到空帧列表")

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_path}")

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()


