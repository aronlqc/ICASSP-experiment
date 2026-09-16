from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import torch
except Exception:
    torch = None


def _default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class RATDPConfig:

    videomae_root: str = str(_PROJECT_ROOT / "VideoMAE")
    videomae_weight_path: str = str(_PROJECT_ROOT / "VideoMAE" / "checkpoints" / "videomae_pretrain_base_patch16_224.pth")
    raft_root: str = str(_PROJECT_ROOT / "RAFT")
    raft_ckpt: str = str(_PROJECT_ROOT / "RAFT" / "models" / "raft-things.pth")
    input_video: str = r"E:\数据集\UCF101\data\UCF101\UCF-101\BalanceBeam\v_BalanceBeam_g10_c03.avi"
    output_dir: str = str(_PROJECT_ROOT / "outputs" / "rat_dp")

    release_stride: int = 16
    
  
    clip_len: int = 16
    tubelet_size: int = 2
    image_size: int = 224
    patch_size: int = 16
    block_grid: int = 14


    C_h: float = 10.0
    lambda_similarity: float = 0.95
    sigma_r: float = 0.10
    motion_norm: float = 20.0

   
    lambda_s: float = 0.6
    sigma_s: float = 0.05
    beta_traj: float = 8.0
    eta_smooth: float = 0.8
    tau_conf: float = 12.0


    sigma_min: float = 0.01
    sigma_max: float = 0.07
    gamma: float = 2.0


    dp_delta: float = 1e-4
    dp_top_k: int = 20            
    dp_C_blk: float = 2   
    dp_sigma: float = 240       
    rdp_alphas: tuple = tuple(range(2, 129))


    raft_iters: int = 12


    seed: int = 42
    device: str = field(default_factory=_default_device)

    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)



    dp_account_per_frame: bool = True

    # Deprecated: correlated-noise/kappa accounting is not used by the fixed-mask theorem.
    dp_sigma_account_kappa: float = 1.0

    # external_mask/fixed_precomputed_mask are covered by the conditional theorem.
    # private_risk_experimental keeps the original data-dependent risk pipeline and
    # must not be reported as end-to-end DP without an additional proof.
    mask_mode: str = "private_risk_experimental"
    external_mask_path: str = ""
    enable_flow_merge_smoothing: bool = True



    use_adaptive_risk: bool = True
    use_trajectory: bool = True
    use_dct_sparse_perturbation: bool = True
    use_temporal_consistency_in_release: bool = True


    def ensure_dirs(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def final_video_path(self) -> str:
        return str(Path(self.output_dir) / "rat_dp_release_mask_topk_blocks_dct.mp4")

    @property
    def intermediate_video_path(self) -> str:
        return str(Path(self.output_dir) / "rat_intermediate_merged.mp4")

    @property
    def report_path(self) -> str:
        return str(Path(self.output_dir) / "rat_dp_report.json")
    
    
