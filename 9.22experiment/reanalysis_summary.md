# Per-video paired reanalysis (2026-09-22)

The saved per-video logs were re-used; no videos were regenerated.
Differences are RAT fusion (RAT-no-trajectory-mask) minus each control.
CI uses 10,000 paired bootstrap resamples (percentile interval, seed 20260922).
Effect size is paired Hedges g_z: the standardized paired difference with the small-sample correction.

| Metric | Control | n | Mean diff | Median diff | Mean 95% CI | Median 95% CI | Hedges g_z | Sign p |
|---|---|---:|---:|---:|---|---|---:|---:|
| Cover | Random | 20 | 0.1520 | 0.1371 | [0.1113, 0.1948] | [0.0861, 0.1939] | 1.47 | 0.000002 |
| Cover | Shifted fusion | 20 | 0.1718 | 0.1714 | [0.1296, 0.2178] | [0.0873, 0.2197] | 1.58 | 0.000002 |
| Cover | VideoMAE-only | 20 | 0.1853 | 0.1739 | [0.1132, 0.2530] | [0.1384, 0.2695] | 1.08 | 0.002577 |
| Cover | RAT+traj. | 20 | 0.0039 | 0.0042 | [-0.0112, 0.0194] | [-0.0214, 0.0204] | 0.10 | 1.000000 |
| RawErr | Random | 20 | 0.1360 | 0.1338 | [0.0977, 0.1757] | [0.0752, 0.1695] | 1.42 | 0.000002 |
| RawErr | Shifted fusion | 20 | 0.1522 | 0.1491 | [0.1112, 0.1937] | [0.0772, 0.1913] | 1.50 | 0.000002 |
| RawErr | VideoMAE-only | 20 | 0.1678 | 0.1514 | [0.1063, 0.2287] | [0.1270, 0.2369] | 1.11 | 0.000402 |
| RawErr | RAT+traj. | 20 | 0.0034 | 0.0065 | [-0.0106, 0.0183] | [-0.0162, 0.0178] | 0.10 | 0.823803 |
| IncErr | Random | 20 | 0.1486 | 0.1407 | [0.1080, 0.1902] | [0.0811, 0.1839] | 1.48 | 0.000002 |
| IncErr | Shifted fusion | 20 | 0.1649 | 0.1668 | [0.1213, 0.2090] | [0.0856, 0.2075] | 1.55 | 0.000002 |
| IncErr | VideoMAE-only | 20 | 0.1798 | 0.1687 | [0.1134, 0.2440] | [0.1360, 0.2574] | 1.12 | 0.000402 |
| IncErr | RAT+traj. | 20 | 0.0037 | 0.0072 | [-0.0119, 0.0197] | [-0.0179, 0.0191] | 0.10 | 0.823803 |
