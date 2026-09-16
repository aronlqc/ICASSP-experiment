# Incremental Perturbation ErrShare

This analysis reuses existing releases and computes motion-proxy error share using backend-input-to-output error, ||Z-Y||^2, rather than raw-video-to-output error, ||X-Y||^2.

## Aggregate

| Method | n | Incr. ErrShare | Incr. Lift | Incr. total PSNR | Incr. high PSNR | Incr. low PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Motion-only-mask | 20 | 0.722277 | 7.082385 | 15.154328 | 6.707038 | 20.535239 |
| RAT-no-trajectory-mask | 20 | 0.245182 | 2.405837 | 15.014169 | 11.504439 | 15.801338 |
| RAT-mask | 20 | 0.241507 | 2.369266 | 15.000672 | 11.667530 | 15.772142 |
| VideoMAE-only-mask | 20 | 0.065409 | 0.643007 | 14.643526 | 22.468759 | 14.488762 |
| Random-mask | 20 | 0.096624 | 0.947406 | 14.910635 | 15.166569 | 14.885169 |
| Shifted-RAT-fusion-mask | 20 | 0.080320 | 0.787685 | 14.985516 | 16.317487 | 14.884528 |
| Shifted-RAT-mask | 20 | 0.077442 | 0.759152 | 14.983489 | 16.589025 | 14.869133 |

## Paired Statistics

| Comparison | n | Left Wins | Right Wins | Mean Diff | 95% CI | Sign p |
|---|---:|---:|---:|---:|---|---:|
| RAT-no-trajectory-mask vs Random-mask | 20 | 20 | 0 | 0.148558 | [0.109359, 0.190198] | 0.000002 |
| RAT-no-trajectory-mask vs Shifted-RAT-fusion-mask | 20 | 20 | 0 | 0.164862 | [0.122479, 0.208964] | 0.000002 |
| RAT-no-trajectory-mask vs VideoMAE-only-mask | 20 | 18 | 2 | 0.179774 | [0.111933, 0.243443] | 0.000402 |
| RAT-mask vs RAT-no-trajectory-mask | 20 | 9 | 11 | -0.003675 | [-0.019909, 0.011500] | 0.823803 |
| RAT-no-trajectory-mask vs Motion-only-mask | 20 | 0 | 20 | -0.477095 | [-0.523257, -0.427553] | 0.000002 |
