# Numba kernel benchmark

Numeric-parity and warm/cold speedup evidence for Numba candidate kernels against the pure-Python/NumPy reference.

- Schema: `1.0.0`
- Created: `2026-07-21T12:35:55.053347+00:00`
- Numba available: `True`
- All kernels pass parity: `True`

| Kernel | parity_pass | speedup_pass | warm baseline (s) | warm numba (s) | speedup ratio | cold (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| inter_pool_gap_runs | True | True | 0.009654 | 0.000195 | 49.41 | 0.163532 |

