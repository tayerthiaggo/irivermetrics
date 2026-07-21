# Performance evidence

HydroFragments keeps CPU as the scientific reference. The committed baseline
in [`benchmarks/results/cpu_baseline.json`](../benchmarks/results/cpu_baseline.json)
contains deterministic B0 analytic and fragmentation cases, output checksums,
per-stage timings, Dask graph/task metadata, hardware and scheduler context,
memory/VRAM fields, planned/actual backend metadata, and host-device transfer
bytes/time. Unavailable CPU-only measurements remain explicit `null`, never
invented. The paired
[`cpu_baseline.md`](../benchmarks/results/cpu_baseline.md) file is the human
summary.

Regenerate on a clean CPU-only environment with:

```powershell
python -c "from hydrofragments.benchmarks.cpu_baseline import write_cpu_baseline; write_cpu_baseline('benchmarks/results', repeats=3, warmup=True)"
```

Timing values are host-specific. Checksums and backend records are the gate
evidence. The runtime default is `compute.accelerator="auto"`: HydroFragments
probes CuPy/CUDA when planning a run and records the actual backend used for
each stage. No CUDA stage is enabled by this baseline; optional acceleration
requires parity and transfer-cost evidence against these CPU outputs. Set
`compute.accelerator="none"` to force a CPU-only run.

See [`acceleration.md`](./acceleration.md) for how the CUDA evidence gate
works end to end (candidate stages, the parity and transfer-cost benchmark
harnesses, the `cuda_baseline.json` schema, and why no CUDA stage is
enabled in this repository today).
