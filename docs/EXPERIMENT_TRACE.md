# ICASSP E1/E2 experiment trace

All paths below are relative to the root of `Minimal_package`. Run commands
from that root.

## Frozen evaluated subset

The manuscript uses a fixed, balanced, non-random UCF101 diagnostic subset:
four clips from each of BalanceBeam, Basketball, Biking, Diving, and
WalkingWithDog. Exact source entries, video metadata, and SHA-256 values are in
`data/ucf101_rescue/data_manifest.csv`.

The portable experiment manifest is `experiments_24h_k64/manifest.csv`. It has
one RAT-DP row per video and points to:

- raw clips under `data/ucf101_rescue/UCF-101/`;
- common backend inputs under `data/backend_inputs/`;
- original trajectory-smoothed score maps under each backend directory's
  `masks/private_risk_scores.npy`.

## Main mask-isolation experiment

Results are under `experiments_icassp4_mask_ablation_20260905/`.

- `mask_isolation_metrics.csv`: per-video/per-policy Cover, RawErr, PSNR, and
  region metrics.
- `mask_isolation_summary.csv`: aggregate values used in Table 1 except IncErr.
- `mask_isolation_paired_stats.csv`: paired Cover/RawErr tests and intervals.
- `incremental_errshare_metrics.csv`: per-video/per-policy IncErr.
- `incremental_errshare_summary.csv`: aggregate IncErr used in Table 1.
- `incremental_errshare_paired_stats.csv`: paired IncErr tests and intervals.
- `run_manifest.csv`: portable input/backend/score-map references for all 140
  runs. Generated output-video/report columns are blank because those large
  regenerable artifacts are not part of the minimal handoff.

Deterministic RAT-fusion and VideoMAE-only score maps are under
`experiments_icassp4_ablation_scores_20260905/`.

## E1 and E2

`extended_evidence_suite_20260906/E1_multi_seed/` retains 1,400 metric rows,
the ten-seed run manifest, video-first paired statistics, per-video seed
aggregation, and seed-level deltas. The frozen seeds are 0--8 and 42.

`extended_evidence_suite_20260906/E2_k_sensitivity/` retains 560 metric rows,
the run manifest, aggregate table, and paired statistics for
`k in {10,20,30,40}`. Sigma is recalibrated at each budget while preserving
epsilon 8 and delta 1e-4.

The 200 E1 trajectory selector arrays are retained under
`extended_evidence_suite_20260906/E1_trajectory_scores/`. E1/E2 release videos
and per-run JSON files are omitted because they can be regenerated from the
retained inputs, score maps, seeds, and code.

## Source scripts

- `code/scripts/generate_mask_ablation_scores_icassp.py`: deterministic
  no-trajectory fusion and VideoMAE-only selector maps.
- `code/scripts/generate_multiseed_trajectory_scores.py`: trajectory maps for
  multiple selector-noise seeds.
- `code/scripts/run_mask_isolation_icassp.py`: main same-backend release.
- `code/scripts/analyze_incremental_errshare_icassp.py`: IncErr analysis.
- `code/scripts/run_extended_allocation_suite.py`: E1/E2 runner.
- `code/scripts/analyze_extended_evidence.py`: E1/E2 statistics and plots.
- `code/scripts/make_icassp_tables_figures.py`: manuscript table values and
  Figure 1 assets.

## Randomness and statistical units

- Main release seed: 42.
- E1 master seeds: 0--8 and 42.
- E2 master seed: 42.
- Random masks use a stable hash of seed, video ID, and `random-mask`.
- The same Gaussian RNG stream is reused within a clip across policies, but
  values are not spatially matched because selected-block orders differ.
- E1 averages seeds within each video before paired testing; the statistical
  unit remains the 20 videos, not 200 video-seed rows.
