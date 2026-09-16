# RAT-DP Privacy Specification v1

## Scope

This version only claims fixed-mask / mask-conditional differential privacy for the final release mechanism.

It does not claim end-to-end DP for masks inferred from the private input video by VideoMAE, RAFT, semantic similarity, motion, or trajectory propagation.

## Private Input

The private input is a same-length RGB video sequence `X = (x_1, ..., x_T)`.

The release query for each selected block is the 2x2 low-frequency DCT vector of the Y channel after L2 clipping:

`q_{t,b}(X) = clip_C(DCT_Y(x_t[b])[:2, :2])`.

## Mask Assumption

The theorem applies only when the per-frame mask/score sequence `P = (P_1, ..., P_T)` is fixed before the release, externally supplied, or otherwise identical for adjacent videos.

Implemented modes:

- `external_mask`: load a mask/score tensor from `external_mask_path`.
- `fixed_precomputed_mask`: load a fixed precomputed mask/score tensor from `external_mask_path`.
- `private_risk_experimental`: compute mask/score maps from the private input. This mode is useful for experiments but is not covered by the current fixed-mask theorem.

## Adjacency

For a fixed mask sequence `P`, two videos `X` and `X'` are adjacent if:

- they have the same number of frames;
- unselected blocks are identical;
- selected blocks may differ by replacement;
- the release query is restricted to the clipped 2x2 Y-channel DCT vector of selected blocks.

With `k` selected blocks per frame and `||q_{t,b}||_2 <= C`, replacement adjacency gives:

`Delta <= 2 C sqrt(k)`.

For the current default `C = 2`, `k = 20`:

`Delta <= 17.88854381999832`.

## Mechanism

For each frame, concatenate the selected clipped vectors and release:

`M(X) = q(X) + N(0, sigma^2 I)`.

The implementation uses independent Gaussian noise for every selected block and every coefficient. The final selected block is reconstructed only from:

- the noisy 2x2 Y-channel low-frequency vector;
- zero high-frequency Y coefficients;
- fixed neutral chroma values.

No raw high-frequency coefficients, raw chroma, or raw alpha-mixed block content may enter the selected block output.

Any temporal smoothing or spatial filtering must be applied after the iid Gaussian release as post-processing.

## RDP Accounting

For order `alpha > 1`, one frame has:

`rho_alpha = alpha Delta^2 / (2 sigma^2)`.

For `T` composed frames:

`rho_total = T rho_alpha`.

Conversion to `(epsilon, delta)`:

`epsilon = rho_total + log(1 / delta) / (alpha - 1)`.

The selected epsilon is the minimum over the configured RDP orders.

No `sigma_release / kappa` correlated-noise accounting is used in this version.

## Reporting Rule

If `mask_mode` is `private_risk_experimental`, the report must state:

`not_covered_by_fixed_mask_conditional_theorem`.

The paper may use that mode only as a risk-localization experiment, not as an end-to-end DP theorem.
