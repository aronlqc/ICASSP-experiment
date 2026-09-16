
from pathlib import Path
from typing import List
import sys
import importlib
import argparse

import numpy as np
import torch


def _import_raft_class(raft_root: str):
    """
    兼容 RAFT 官方仓库常见结构：
      - core.raft: RAFT
      - raft: RAFT
    """
    raft_root = str(Path(raft_root))
    if raft_root not in sys.path:
        sys.path.insert(0, raft_root)

    candidates = [
        ("core.raft", "RAFT"),
        ("raft", "RAFT"),
    ]

    last_err = None
    for module_name, class_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, class_name)
        except Exception as e:
            last_err = e

    raise ImportError(f"无法导入 RAFT，最后错误: {last_err}")


def _import_input_padder(raft_root: str):
    raft_root = str(Path(raft_root))
    if raft_root not in sys.path:
        sys.path.insert(0, raft_root)

    candidates = [
        ("utils.utils", "InputPadder"),
        ("core.utils.utils", "InputPadder"),
    ]

    for module_name, class_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, class_name)
        except Exception:
            pass

    return None


class RAFTFlowEstimator:
    def __init__(self, raft_root: str, ckpt_path: str, device: str = "cuda", iters: int = 12):
        self.raft_root = str(raft_root)
        self.ckpt_path = str(ckpt_path)
        self.device = torch.device(device)
        self.iters = int(iters)

        RAFT = _import_raft_class(self.raft_root)
        self.InputPadder = _import_input_padder(self.raft_root)

        args = argparse.Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            model=self.ckpt_path,
        )

        self.model = RAFT(args)
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self):
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"RAFT checkpoint 不存在: {self.ckpt_path}")

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        cleaned = {}
        for k, v in ckpt.items():
            cleaned[k.replace("module.", "")] = v

        try:
            self.model.load_state_dict(cleaned)
        except Exception:
            self.model.load_state_dict(cleaned, strict=False)

    @staticmethod
    def _to_tensor(frame_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        RAFT 官方 forward 内部会做 /255.0，因此这里保留 0~255 float。
        输入 frame_rgb: HWC, uint8
        输出: [1, 3, H, W]
        """
        x = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0)
        return x.to(device)

    @torch.no_grad()
    def compute_flow(self, frame1_rgb: np.ndarray, frame2_rgb: np.ndarray) -> np.ndarray:
        image1 = self._to_tensor(frame1_rgb, self.device)
        image2 = self._to_tensor(frame2_rgb, self.device)

        if self.InputPadder is not None:
            padder = self.InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
        else:
            padder = None
            _, _, h, w = image1.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                image1 = torch.nn.functional.pad(image1, (0, pad_w, 0, pad_h))
                image2 = torch.nn.functional.pad(image2, (0, pad_w, 0, pad_h))

        flow_low, flow_up = self.model(
            image1,
            image2,
            iters=self.iters,
            test_mode=True,
        )

        if padder is not None:
            flow_up = padder.unpad(flow_up)

        flow = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()  # H, W, 2
        h0, w0 = frame1_rgb.shape[:2]
        flow = flow[:h0, :w0, :]
        return flow

    @torch.no_grad()
    def compute_video_flows(self, frames_rgb: List[np.ndarray]) -> List[np.ndarray]:
        flows = []
        if len(frames_rgb) < 2:
            return flows
        for i in range(len(frames_rgb) - 1):
            flows.append(self.compute_flow(frames_rgb[i], frames_rgb[i + 1]))
        return flows
