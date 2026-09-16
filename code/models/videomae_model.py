from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

try:
    from transformers import VideoMAEForPreTraining, VideoMAEConfig
except Exception:
    VideoMAEForPreTraining = None
    VideoMAEConfig = None


class VideoMAEOfficialWrapper:
    """
    Robust VideoMAE wrapper:
      - resolve a valid VideoMAE checkpoint under model_root
      - avoid accidentally loading unrelated .pth files (e.g. RAFT weights)
      - expose patch token extraction
    """

    def __init__(
        self,
        model_root: str,
        device: str,
        image_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        weight_path: Optional[str] = None,
    ):
        self.model_root = str(model_root)
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.weight_path = str(weight_path) if weight_path else ""

        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _find_weight_path(self) -> Path:
        if self.weight_path:
            p = Path(self.weight_path)
            if not p.exists():
                raise FileNotFoundError(f"VideoMAE checkpoint 不存在: {p}")
            return p

        root = Path(self.model_root)
        if not root.exists():
            raise FileNotFoundError(f"VideoMAE root 不存在: {root}")

        candidates = [
            root / "checkpoints" / "videomae_pretrain_base_patch16_224.pth",
            root / "checkpoints" / "videomae_pretrain_base_patch16_224.pth.tar",
            root / "checkpoints" / "videomae_base_patch16_224.pth",
            root / "videomae_pretrain_base_patch16_224.pth",
            root / "videomae_base_patch16_224.pth",
            root / "pretrained" / "videomae_pretrain_base_patch16_224.pth",
            root / "weights" / "videomae_pretrain_base_patch16_224.pth",
        ]
        for p in candidates:
            if p.exists():
                return p

        found = []
        for ext in ("*videomae*.pth", "*videomae*.pt", "*videomae*.bin", "*videomae*.pth.tar"):
            found.extend(root.rglob(ext))
        found = [p for p in found if p.exists()]
        if found:
            return sorted(found)[0]

        raise FileNotFoundError(f"找不到 VideoMAE 权重，已搜索目录: {root}")

    def _load_model(self):
        if VideoMAEForPreTraining is None or VideoMAEConfig is None:
            raise ImportError("未安装 transformers")

        config = VideoMAEConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
        )
        model = VideoMAEForPreTraining(config)

        weight_path = self._find_weight_path()
        print(f"[VideoMAE] Loading checkpoint from: {weight_path}")

        ckpt = torch.load(str(weight_path), map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                ckpt = ckpt["model"]
            elif "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

        cleaned = {k.replace("module.", ""): v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[VideoMAE] missing keys: {len(missing)}")
        print(f"[VideoMAE] unexpected keys: {len(unexpected)}")

        return model

    @torch.no_grad()
    def extract_patch_tokens(self, pixel_values_norm: torch.Tensor) -> torch.Tensor:
        if pixel_values_norm.dim() != 5:
            raise ValueError(f"VideoMAE 输入必须是 [B,T,C,H,W]，实际是 {pixel_values_norm.shape}")

        backbone = getattr(self.model, "videomae", None)
        if backbone is None:
            backbone = self.model

        outputs = backbone(
            pixel_values=pixel_values_norm.to(self.device),
            output_hidden_states=True,
            return_dict=True,
        )

        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None:
            if isinstance(outputs, (tuple, list)):
                tokens = outputs[0]
            else:
                raise RuntimeError("无法从 VideoMAE 输出中解析 token")

        B, T, C, H, W = pixel_values_norm.shape
        expected = (T // self.tubelet_size) * (H // self.patch_size) * (W // self.patch_size)

        if tokens.shape[1] == expected + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != expected:
            raise RuntimeError(
                f"VideoMAE token 数不匹配: got={tokens.shape[1]}, expected={expected} or {expected+1}"
            )
        return tokens
