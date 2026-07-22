# CPU reference benchmark baseline

Deterministic CPU baseline for certified array reductions. CUDA is not enabled.

- Schema: `1.0.0`
- Created: `2026-07-17T01:51:55.890440+00:00`
- Backend planned: `cpu`

| Dataset | Stage | backend_actual | Median seconds | p95 seconds | Graph tasks | Peak RSS | Peak VRAM | Transfer bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_analytic | assemble_monthly | cpu | 0.002639 | 0.002836 | 72 | n/a | n/a | 0 |
| B0_analytic | monthly_reduction | cpu | 0.002964 | 0.003015 | 72 | n/a | n/a | 0 |
| B0_analytic | occurrence | cpu | 0.006851 | 0.012577 | 72 | n/a | n/a | 0 |
| B0_analytic | apsec | cpu | 0.000899 | 0.001366 | 72 | n/a | n/a | 0 |
| B0_fragmentation | assemble_monthly | cpu | 0.002622 | 0.002639 | 384 | n/a | n/a | 0 |
| B0_fragmentation | monthly_reduction | cpu | 0.012759 | 0.013213 | 384 | n/a | n/a | 0 |
| B0_fragmentation | occurrence | cpu | 0.007719 | 0.007810 | 384 | n/a | n/a | 0 |
| B0_fragmentation | apsec | cpu | 0.000962 | 0.000973 | 384 | n/a | n/a | 0 |
| B0_scale | assemble_monthly | cpu | 0.002853 | 0.003760 | 288 | n/a | n/a | 0 |
| B0_scale | monthly_reduction | cpu | 0.010660 | 0.010702 | 288 | n/a | n/a | 0 |
| B0_scale | occurrence | cpu | 0.015579 | 0.019336 | 288 | n/a | n/a | 0 |
| B0_scale | apsec | cpu | 0.001213 | 0.001853 | 288 | n/a | n/a | 0 |

Checksums are scientific-output evidence; timing values are host-specific.
