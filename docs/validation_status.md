# HydroFragments Validation Status

This document summarizes the empirical validation status and verification evidence for HydroFragments `0.1.0`.

## Validated Claims

1. **Numerical Accuracy:** All metric algorithms (APSEC, Occurrence, Refuge Area, Pool Count, LPI, AWRe, AWMSI, Pool Width, Connectivity) match exact mathematical definitions and goldens across synthetic and real satellite datasets.
2. **Memory Boundedness:** Monthly section analysis computes data one month at a time, preventing Out-Of-Memory (OOM) failures on multi-decade catchment-scale raster cubes.
3. **Parallel Invariance:** Metric calculations produce byte-identical output across single-threaded, multi-threaded, and multi-process worker counts.
4. **Reproducibility:** Run manifests generate cryptographic SHA-256 digests over configuration inputs and output metric tables.

## Catchment Verification

Validated on Landsat/Sentinel WOfS time series (1986–2026) across Australian intermittent river catchments (e.g., Fitzroy River Basin, WA).
