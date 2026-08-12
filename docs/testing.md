# HydroFragments Testing Guide

HydroFragments maintains a comprehensive pytest suite covering numerical correctness, memory boundedness, API contracts, and output schema validation.

## Test Suite Structure

- `tests/analysis/`: Monthly dataset materialization, laziness, and parallel section analysis.
- `tests/api/`: Public `analyze()` and `open_water_cube()` API contracts.
- `tests/metrics/`: Vectorized metric numerical parity and correctness.
- `tests/patches/`: Connected component labeling, thresholding, and morphological properties.
- `tests/spatial/`: Channel centrelines, windowing, and spatial geometry.
- `tests/output/`: Parquet/CSV table formatting and JSON manifest schema verification.
- `tests/release/`: Package version, metadata, pyproject, and branding guards.
- `tests/docs/`: Example execution and docstring code block validation.

## Running Tests

### Fast Test Suite (default CI)

```powershell
python -m pytest -m "not slow" -q
```

### Full Test Suite (including slow integration benchmarks)

```powershell
python -m pytest -q
```

### Release Metadata and Branding Guard

```powershell
python -m pytest tests/release/ -q
```
