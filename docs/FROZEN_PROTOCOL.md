# Frozen paper protocol

Frozen on 2026-09-06. The handoff scope is the current manuscript, the main
20-video same-backend mask-isolation experiment, E1, and E2.

## Scientific question

At a fixed per-clip DCT-Gaussian backend, does changing only the selected mask
change how perturbation aligns with a frame-difference motion proxy?

Primary metrics are motion-proxy coverage (Cover), raw-video error share
(RawErr), and incremental backend error share (IncErr). The video is the paired
statistical unit.

## E1

- 20 clips, seven policies, ten master seeds (0--8 and 42), `k=20`.
- 1,400 completed policy runs and no failures.
- Metrics are averaged across seeds inside each video before paired inference.

## E2

- 20 clips, seven policies, seed 42, `k in {10,20,30,40}`.
- 560 completed policy runs and no failures.
- Sigma is recalibrated per clip and budget to preserve
  `epsilon=8, delta=1e-4`.

The motion-reference policy is a proxy-aligned empirical upper bound. It is not
a formal mathematical upper bound or a human-privacy target. The privacy claim
is limited to selected DCT queries under fixed/external masks and
mask-restricted adjacency.
