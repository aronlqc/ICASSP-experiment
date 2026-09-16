# Same-Backend Mask Isolation

Protocol: for each UCF101 clip, all variants use the saved RAT pre-release video as the backend input, the same fixed-mask DCT Gaussian release, the same top-k, the same sigma, and the same noise seed. Only the score map/mask changes.

Motion proxy: block-level frame-difference saliency from the raw GT video. It is a cheap independent proxy for action/motion salience, not a privacy label.

## Aggregate

| Method | n | PSNR | Motion Cover | Motion Lift | Motion Dist Share | Motion Dist Lift | Motion High PSNR | Motion Low PSNR | RAT-Region Share | Own-Region Share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Motion-only-mask | 20 | 14.811470 | 0.731555 | 7.181457 | 0.672666 | 6.595544 | 6.528197 | 19.269644 | 0.332408 | 0.913784 |
| RAT-mask | 20 | 14.692309 | 0.251123 | 2.472896 | 0.237400 | 2.328862 | 11.267968 | 15.280938 | 0.911989 | 0.911989 |
| RAT-no-trajectory-mask | 20 | 14.708745 | 0.254993 | 2.517506 | 0.240830 | 2.363014 | 11.119715 | 15.311097 | 0.551037 | 0.911466 |
| Random-mask | 20 | 14.633758 | 0.102987 | 1.009239 | 0.104841 | 1.028094 | 14.379074 | 14.495091 | 0.103468 | 0.907855 |
| Shifted-RAT-fusion-mask | 20 | 14.710748 | 0.083197 | 0.819184 | 0.088655 | 0.869442 | 15.380504 | 14.488516 | 0.062392 | 0.906862 |
| Shifted-RAT-mask | 20 | 14.709702 | 0.079792 | 0.785457 | 0.086195 | 0.845029 | 15.597245 | 14.478038 | 0.043427 | 0.906916 |
| VideoMAE-only-mask | 20 | 14.371944 | 0.069704 | 0.690568 | 0.073016 | 0.717413 | 17.819255 | 14.099740 | 0.027344 | 0.911059 |

## Interpretation

- Random-mask, Shifted-RAT-fusion-mask, and VideoMAE-only-mask are controls for the narrowed allocation claim.
- Motion-only-mask is the constructed motion-salience proxy reference, not a deployable baseline that RAT is expected to beat.
- RAT-no-trajectory-mask is the main paper method (`RAT fusion`); RAT-mask is retained only as the trajectory-smoothed ablation (`RAT+traj.`).
- The trajectory-smoothed variant is statistically indistinguishable from direct cue fusion, so the paper should not claim a trajectory-propagation gain.
- Global PSNR is secondary here because all rows share the same backend and only the allocation changes.

## Paired Main-Metric Statistics

| Comparison | Metric | n | Left Wins | Right Wins | Mean Diff | 95% CI | Sign p |
|---|---|---:|---:|---:|---:|---:|---:|
| RAT-no-trajectory-mask vs Random-mask | motion_reference_coverage | 20 | 20 | 0 | 0.152005 | [0.111120, 0.195839] | 0.000002 |
| RAT-no-trajectory-mask vs Random-mask | motion_distortion_share | 20 | 20 | 0 | 0.135989 | [0.097657, 0.176501] | 0.000002 |
| RAT-no-trajectory-mask vs Shifted-RAT-fusion-mask | motion_reference_coverage | 20 | 20 | 0 | 0.171796 | [0.128564, 0.218040] | 0.000002 |
| RAT-no-trajectory-mask vs Shifted-RAT-fusion-mask | motion_distortion_share | 20 | 20 | 0 | 0.152175 | [0.112165, 0.193506] | 0.000002 |
| RAT-no-trajectory-mask vs Motion-only-mask | motion_reference_coverage | 20 | 0 | 20 | -0.476563 | [-0.522875, -0.428003] | 0.000002 |
| RAT-no-trajectory-mask vs Motion-only-mask | motion_distortion_share | 20 | 0 | 20 | -0.431836 | [-0.475460, -0.388656] | 0.000002 |
| RAT-no-trajectory-mask vs VideoMAE-only-mask | motion_reference_coverage | 20 | 17 | 3 | 0.185288 | [0.114453, 0.254344] | 0.002577 |
| RAT-no-trajectory-mask vs VideoMAE-only-mask | motion_distortion_share | 20 | 18 | 2 | 0.167814 | [0.103053, 0.227542] | 0.000402 |
| RAT-mask vs RAT-no-trajectory-mask | motion_reference_coverage | 20 | 10 | 10 | -0.003870 | [-0.020082, 0.011095] | 1.000000 |
| RAT-mask vs RAT-no-trajectory-mask | motion_distortion_share | 20 | 9 | 11 | -0.003429 | [-0.018104, 0.010695] | 0.823803 |
| RAT-mask vs Random-mask | motion_reference_coverage | 20 | 19 | 1 | 0.148135 | [0.104104, 0.192681] | 0.000040 |
| RAT-mask vs Random-mask | motion_distortion_share | 20 | 18 | 2 | 0.132560 | [0.090467, 0.175112] | 0.000402 |
| RAT-mask vs Shifted-RAT-mask | motion_reference_coverage | 20 | 20 | 0 | 0.171331 | [0.126221, 0.217929] | 0.000002 |
| RAT-mask vs Shifted-RAT-mask | motion_distortion_share | 20 | 20 | 0 | 0.151205 | [0.110295, 0.193643] | 0.000002 |
| RAT-mask vs Motion-only-mask | motion_reference_coverage | 20 | 0 | 20 | -0.480433 | [-0.526350, -0.432441] | 0.000002 |
| RAT-mask vs Motion-only-mask | motion_distortion_share | 20 | 0 | 20 | -0.435266 | [-0.478134, -0.392109] | 0.000002 |
| Motion-only-mask vs Random-mask | motion_reference_coverage | 20 | 20 | 0 | 0.628568 | [0.579858, 0.671363] | 0.000002 |
| Motion-only-mask vs Random-mask | motion_distortion_share | 20 | 20 | 0 | 0.567826 | [0.519861, 0.611425] | 0.000002 |
| Motion-only-mask vs Shifted-RAT-mask | motion_reference_coverage | 20 | 20 | 0 | 0.651764 | [0.605427, 0.696028] | 0.000002 |
| Motion-only-mask vs Shifted-RAT-mask | motion_distortion_share | 20 | 20 | 0 | 0.586471 | [0.539583, 0.631604] | 0.000002 |
