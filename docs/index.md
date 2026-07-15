# HydroFragments documentation

HydroFragments v1.2 measures **river surface-water patch dynamics** from aligned
water/valid time series. Canonical output is tidy Parquet with manifest provenance.

## Start here

| Audience | Document |
|----------|----------|
| New users | [README](../README.md) quickstart |
| Migrating from EcoFragments/iRiverMetrics | [migration_v1_2.md](./migration_v1_2.md) |
| Input contracts | [input_format.md](./input_format.md) |
| Architecture | [architecture.md](./architecture.md) |

## Install

```bash
pip install -e ".[test]"
```

## Quick use

```python
from hydrofragments import HydroConfig, analyze, open_water_cube
```

See the README quickstart for a runnable minimal example.

## Legacy modules (quarantined)

- [module1.md](./module1.md) — `waterdetect_batch` is **not implemented** in this repository.
- [module2.md](./module2.md) — historical legacy metric reference only.

## Spec and audits

- [HydroFragments v1.2 spec](./HydroFragments_v1.2_spec.md)
- [Testing guide](./testing.md)
