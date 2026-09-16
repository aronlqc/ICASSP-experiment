# RAT-DP Release Path Audit

## Audited Path

`raw video -> preprocess -> VideoMAE/RAFT -> risk map -> trajectory weights -> clip merge -> final DCT release -> saved mp4`

## Original Blocking Risks

1. The original final release selected top-k blocks from private score maps and treated `P_t` as public side information.
2. The original final release perturbed only selected Y-channel 2x2 low-frequency coefficients.
3. The original final release kept raw high-frequency DCT coefficients in selected blocks.
4. The original final release kept raw Cb/Cr chroma in selected blocks.
5. The original final release blended the raw Y block back into the released block with `alpha=0.3`.
6. The original noise was temporally smoothed and spatially blurred before release, then accounted with `sigma_release / kappa`.

Those points are enough to block a released-video DP claim.

## Current Code Changes

Changed `postprocess/dct_perturb.py::perturb_video_masked_blocks_dct_dp_temporal`:

- Gaussian noise is iid per frame, selected block, and low-frequency coefficient.
- All four 2x2 Y-channel low-frequency coefficients receive Gaussian noise, including DC.
- Selected-block high-frequency Y coefficients are zeroed before inverse DCT.
- Selected-block Cb/Cr are replaced with fixed neutral chroma.
- Raw alpha mixing is disabled; calls should pass `alpha=1.0`.
- Temporal smoothing is treated only as post-processing after the DP release.
- Block bounds now use split bounds so non-divisible frame sizes do not silently drop border pixels.

Changed `main.py`:

- Added CLI overrides for input, output directory, seed, mask mode, external mask path, delta, sigma, top-k, and C.
- Added `external_mask` and `fixed_precomputed_mask` support through `.npy` mask/score tensors.
- Saves private risk scores under `masks/private_risk_scores.npy` when running the original data-dependent risk mode.
- Reports `theorem_scope`, `mask_info`, and iid sigma accounting.
- Removes `sigma_release / kappa` accounting.

Changed `ablation/common.py`:

- Uses iid sigma for accounting.
- Calls final DCT release with `alpha=1.0`.
- Marks private-risk masks as not covered by the fixed-mask theorem.

## Remaining Theorem Boundary

The current code can support the fixed-mask theorem only when the final release uses `mask_mode=external_mask` or `mask_mode=fixed_precomputed_mask`.

The default `private_risk_experimental` mode still computes masks from the private video. Its final Gaussian layer is cleaner than before, but the end-to-end mask selection remains outside the current theorem.

## Stop Rule

Before large experiments, run a small release and inspect `rat_dp_report.json`.

Proceed only if:

- `noise_distribution` is `iid_gaussian`;
- `raw_alpha_mixing_used` is `false`;
- `selected_block_high_frequency` is `zeroed_before_inverse_dct`;
- `selected_block_chroma` is `fixed_neutral_value`;
- `dp_sigma_account_kappa` is `1.0`;
- paper claims match `theorem_scope`.

If the paper needs an end-to-end theorem with data-dependent masks, this implementation is not enough. The mask selection itself must be privatized or separately proven.
