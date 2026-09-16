
import torch


def clip_l2(features: torch.Tensor, C_h: float) -> torch.Tensor:
    """
    features: [N, d] 或 [B, N, d]
    """
    if C_h <= 0:
        raise ValueError("C_h 必须 > 0")

    orig_shape = features.shape
    x = features.reshape(-1, orig_shape[-1])

    norms = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    scale = torch.minimum(torch.ones_like(norms), x.new_tensor(float(C_h)) / norms)
    x_clip = x * scale
    return x_clip.reshape(orig_shape)
