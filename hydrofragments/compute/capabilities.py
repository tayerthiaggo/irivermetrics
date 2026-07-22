"""Optional accelerator detection, stage planning, and truthful backend records."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Iterable


class CapabilityError(RuntimeError):
    """Raised when a requested accelerator cannot be initialized safely."""


DEFAULT_STAGES: tuple[str, ...] = (
    "sentinel_normalization",
    "masks",
    "valid_counts",
    "monthly_reduction",
    "occurrence",
    "skeleton",
    "graph",
    "vector",
)

# Candidate kernels are intentionally separate from enabled kernels. No CUDA
# stage is enabled until parity and transfer-cost evidence is attached to a
# release baseline.
CUDA_CANDIDATE_STAGES: tuple[str, ...] = (
    "sentinel_normalization",
    "masks",
    "valid_counts",
    "monthly_reduction",
    "occurrence",
)

# Twin of CUDA_CANDIDATE_STAGES for Numba JIT prototypes: a kernel name is a
# candidate, not a promise. Only "inter_pool_gap_runs" has a Numba prototype
# today (hydrofragments/metrics/clustering.py's
# _compute_inter_pool_gaps_numba) -- the other hot-loop candidate named in
# the plan (per-crop EDT/width extraction in
# hydrofragments/patches/morphology.py::_measure_component) was profiled and
# rejected: skimage's compiled medial_axis() dominates wall time by roughly
# 3-4 orders of magnitude over the surrounding pure-Python/NumPy
# post-processing (a single vectorized ``(2.0 * dist[axis]).max()``
# reduction), so there is no tractable Numba win available there without
# reimplementing medial_axis itself, which is out of scope. See
# docs/acceleration.md's Numba section for the full writeup.
NUMBA_CANDIDATE_KERNELS: tuple[str, ...] = ("inter_pool_gap_runs",)

FLOATING_TOLERANCES: dict[str, float] = {
    "float32": 1e-5,
    "float64": 1e-12,
}


@dataclass(frozen=True)
class BackendCapabilities:
    """Immutable runtime capability evidence.

    ``enabled_cuda_stages`` is evidence-gated, not a synonym for installed
    CuPy. An environment can have a working GPU while this tuple remains empty
    until parity and transfer-cost benchmarks approve a stage.
    """

    cupy_available: bool = False
    cuda_available: bool = False
    cupy_version: str | None = None
    cuda_runtime_version: int | None = None
    device_count: int = 0
    free_memory_bytes: int | None = None
    total_memory_bytes: int | None = None
    enabled_cuda_stages: tuple[str, ...] = ()
    numba_enabled_kernels: tuple[str, ...] = ()
    reason: str | None = None

    @classmethod
    def cpu_only(cls, reason: str) -> "BackendCapabilities":
        return cls(reason=reason)

    def to_mapping(self) -> dict[str, object]:
        return {
            "cupy_available": self.cupy_available,
            "cupy_version": self.cupy_version,
            "cuda_available": self.cuda_available,
            "cuda_runtime_version": self.cuda_runtime_version,
            "device_count": self.device_count,
            "free_memory_bytes": self.free_memory_bytes,
            "total_memory_bytes": self.total_memory_bytes,
            "enabled_cuda_stages": list(self.enabled_cuda_stages),
            "numba_enabled_kernels": list(self.numba_enabled_kernels),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """Planned and actual backend by stage for one run."""

    requested_accelerator: str
    planned_backend: str
    actual_backend_by_stage: dict[str, str]
    fallback_reason: str | None
    capabilities: BackendCapabilities

    def actual_backend(self, stage: str) -> str:
        try:
            return self.actual_backend_by_stage[stage]
        except KeyError as error:
            raise CapabilityError(f"stage is not in execution plan: {stage}") from error

    def to_mapping(self) -> dict[str, object]:
        return {
            "requested_accelerator": self.requested_accelerator,
            "planned": self.planned_backend,
            "actual_by_stage": dict(self.actual_backend_by_stage),
            "fallback_reason": self.fallback_reason,
            "cuda_enabled_stages": list(self.capabilities.enabled_cuda_stages),
            "capabilities": self.capabilities.to_mapping(),
            "floating_tolerances": dict(FLOATING_TOLERANCES),
        }


# Baseline evidence file produced by running the benchmark harnesses in
# hydrofragments/benchmarks/ (cuda_parity.py + cuda_transfer_cost.py) and
# hand-assembled into this compact per-stage gate summary. Lives at
# benchmarks/results/cuda_baseline.json (repo root) -- the same location as
# the existing committed benchmarks/results/cpu_baseline.json, per this
# repo's established convention (see docs/performance.md). See
# docs/acceleration.md for the full schema and how to (re)generate it. This
# repo ships without this file by default -- CUDA stays fully CPU-fallback
# until a real GPU host records evidence here.
_DEFAULT_BASELINE_PATH: Path = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "cuda_baseline.json"
)

# Twin of _DEFAULT_BASELINE_PATH for the Numba evidence gate. Produced by
# hand-distilling hydrofragments/benchmarks/numba_kernels.py's parity +
# speedup report into this compact per-kernel gate summary, the same
# curatorial step docs/acceleration.md describes for cuda_baseline.json. This
# repo ships without this file by default -- every NUMBA_CANDIDATE_KERNELS
# entry stays disabled (falls back to the pure-Python/NumPy implementation)
# until a benchmark run records evidence here.
_DEFAULT_NUMBA_BASELINE_PATH: Path = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "numba_baseline.json"
)


def gated_stages_from_baseline(
    baseline_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Read the evidence-gate baseline JSON and return graduated stages.

    A stage graduates from :data:`CUDA_CANDIDATE_STAGES` only if the baseline
    records both ``parity_pass: true`` (numeric correctness, see
    ``benchmarks/cuda_parity.py``) and ``net_speedup_pass: true`` (CUDA wall
    time including transfer beats CPU, see ``benchmarks/cuda_transfer_cost.py``)
    for that stage. Missing file, unreadable/malformed JSON, an empty
    mapping, or a stage outside ``CUDA_CANDIDATE_STAGES`` are all treated as
    "no evidence" rather than raising -- this function must never be the
    reason a truthful CPU fallback turns into a crash.
    """

    path = Path(baseline_path) if baseline_path is not None else _DEFAULT_BASELINE_PATH
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ()

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return ()

    if not isinstance(payload, dict):
        return ()
    stages = payload.get("stages")
    if not isinstance(stages, dict):
        return ()

    gated = []
    for stage in CUDA_CANDIDATE_STAGES:
        evidence = stages.get(stage)
        if not isinstance(evidence, dict):
            continue
        if evidence.get("parity_pass") is True and evidence.get("net_speedup_pass") is True:
            gated.append(stage)
    return tuple(gated)


def gated_kernels_from_baseline(
    baseline_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Read the Numba evidence-gate baseline JSON and return graduated kernels.

    Twin of :func:`gated_stages_from_baseline` for Numba JIT kernels instead
    of CUDA stages. A kernel graduates from :data:`NUMBA_CANDIDATE_KERNELS`
    only if the baseline records both ``parity_pass: true`` (numerically
    identical to the existing pure-Python/NumPy implementation, see
    ``benchmarks/numba_kernels.py``) and ``speedup_pass: true`` (the warmed-up
    ``@njit`` call beats the baseline implementation's wall time) for that
    kernel. Missing file, unreadable/malformed JSON, an empty mapping, or a
    kernel name outside ``NUMBA_CANDIDATE_KERNELS`` are all treated as "no
    evidence" rather than raising -- same never-crash contract as
    :func:`gated_stages_from_baseline`.

    Unlike CUDA, Numba has no hardware-availability probe step (a JIT
    compiler works on any CPU, there is no device to detect), so this
    function is the entire gate -- there is no probe-then-gate two-step
    equivalent to ``_resolve_cuda_capabilities_from_probe``.
    """

    path = (
        Path(baseline_path)
        if baseline_path is not None
        else _DEFAULT_NUMBA_BASELINE_PATH
    )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ()

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return ()

    if not isinstance(payload, dict):
        return ()
    kernels = payload.get("kernels")
    if not isinstance(kernels, dict):
        return ()

    gated = []
    for kernel in NUMBA_CANDIDATE_KERNELS:
        evidence = kernels.get(kernel)
        if not isinstance(evidence, dict):
            continue
        if evidence.get("parity_pass") is True and evidence.get("speedup_pass") is True:
            gated.append(kernel)
    return tuple(gated)


def _resolve_cuda_capabilities_from_probe(
    *,
    cupy_available: bool,
    cuda_available: bool,
    cupy_version: str | None,
    cuda_runtime_version: int | None,
    device_count: int,
    free_memory_bytes: int | None,
    total_memory_bytes: int | None,
) -> BackendCapabilities:
    """Build the CUDA-detected ``BackendCapabilities``, gated by baseline evidence.

    Split out from :func:`detect_capabilities` so the baseline-reading step
    is unit-testable without real CUDA hardware: tests call this directly
    with hand-constructed probe results instead of needing a GPU to reach
    this code path through the CuPy smoke test above it.
    """

    enabled_cuda_stages = gated_stages_from_baseline()
    if enabled_cuda_stages:
        reason = f"CUDA detected; baseline evidence enables: {', '.join(enabled_cuda_stages)}"
    else:
        reason = "CUDA detected; no stage has transfer-cost benefit evidence"

    return BackendCapabilities(
        cupy_available=cupy_available,
        cuda_available=cuda_available,
        cupy_version=cupy_version,
        cuda_runtime_version=cuda_runtime_version,
        device_count=device_count,
        free_memory_bytes=free_memory_bytes,
        total_memory_bytes=total_memory_bytes,
        enabled_cuda_stages=enabled_cuda_stages,
        numba_enabled_kernels=gated_kernels_from_baseline(),
        reason=reason,
    )


def detect_capabilities(*, probe_cuda: bool = False) -> BackendCapabilities:
    """Return capability evidence without importing CuPy by default.

    CPU policy construction calls this function with its default. Accelerator
    resolution opts into the import and catches all runtime initialization
    failures so auto mode can report a truthful CPU fallback.

    Numba kernel gating is independent of the CUDA probe: unlike CuPy, Numba
    has no hardware-availability step to gate behind (JIT compilation works
    on any CPU), so ``numba_enabled_kernels`` is resolved from baseline
    evidence on every call, including the ``probe_cuda=False`` default path.
    """

    numba_enabled_kernels = gated_kernels_from_baseline()

    if not probe_cuda:
        return BackendCapabilities(
            reason="CUDA probe disabled by CPU policy",
            numba_enabled_kernels=numba_enabled_kernels,
        )

    try:
        cupy = importlib.import_module("cupy")
    except Exception as error:  # optional dependency and runtime failures
        return BackendCapabilities(
            reason=f"CuPy unavailable: {error}",
            numba_enabled_kernels=numba_enabled_kernels,
        )

    try:
        if not bool(cupy.is_available()):
            return BackendCapabilities(
                cupy_available=True,
                cupy_version=getattr(cupy, "__version__", None),
                numba_enabled_kernels=numba_enabled_kernels,
                reason="CUDA runtime unavailable",
            )
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        if device_count < 1:
            return BackendCapabilities(
                cupy_available=True,
                cupy_version=getattr(cupy, "__version__", None),
                numba_enabled_kernels=numba_enabled_kernels,
                reason="CUDA reports no visible devices",
            )
        with cupy.cuda.Device(0) as device:
            free_bytes, total_bytes = device.mem_info
            smoke = cupy.zeros(1, dtype=cupy.float32)
            smoke += 1
            cupy.cuda.Stream.null.synchronize()
            del smoke
        runtime_version = int(cupy.cuda.runtime.runtimeGetVersion())
    except Exception as error:
        return BackendCapabilities(
            cupy_available=True,
            cupy_version=getattr(cupy, "__version__", None),
            numba_enabled_kernels=numba_enabled_kernels,
            reason=f"CUDA initialization failed: {error}",
        )

    return _resolve_cuda_capabilities_from_probe(
        cupy_available=True,
        cuda_available=True,
        cupy_version=getattr(cupy, "__version__", None),
        cuda_runtime_version=runtime_version,
        device_count=device_count,
        free_memory_bytes=int(free_bytes),
        total_memory_bytes=int(total_bytes),
    )


def resolve_execution_plan(
    *,
    accelerator: str,
    cuda_strict: bool = False,
    stages: Iterable[str] = DEFAULT_STAGES,
    capabilities: BackendCapabilities | None = None,
) -> ExecutionPlan:
    """Resolve CPU/auto/CUDA policy and record actual backend per stage."""

    if accelerator not in {"none", "auto", "cuda"}:
        raise CapabilityError(f"unsupported accelerator: {accelerator}")
    if cuda_strict and accelerator != "cuda":
        raise CapabilityError("cuda_strict requires accelerator='cuda'")
    resolved_capabilities = capabilities
    if resolved_capabilities is None:
        resolved_capabilities = detect_capabilities(probe_cuda=accelerator != "none")

    requested_stages = tuple(stages)
    unknown = set(requested_stages).difference(DEFAULT_STAGES)
    if unknown:
        raise CapabilityError(f"unsupported execution stages: {sorted(unknown)}")

    enabled_for_request = set(resolved_capabilities.enabled_cuda_stages).intersection(
        requested_stages
    )
    if accelerator == "none":
        planned_backend = "cpu"
        fallback_reason = "accelerator=none"
    elif not resolved_capabilities.cuda_available:
        reason = resolved_capabilities.reason or "CUDA unavailable"
        if accelerator == "cuda":
            raise CapabilityError(f"CUDA unavailable: {reason}")
        planned_backend = "cpu"
        fallback_reason = reason
    elif not enabled_for_request:
        reason = resolved_capabilities.reason or "no CUDA stages are enabled"
        if accelerator == "cuda" and cuda_strict:
            raise CapabilityError(f"CUDA unavailable: {reason}")
        planned_backend = "cpu"
        fallback_reason = reason
    else:
        planned_backend = "cuda"
        fallback_reason = None

    actual = {
        stage: "cuda"
        if planned_backend == "cuda" and stage in enabled_for_request
        else "cpu"
        for stage in requested_stages
    }
    return ExecutionPlan(
        requested_accelerator=accelerator,
        planned_backend=planned_backend,
        actual_backend_by_stage=actual,
        fallback_reason=fallback_reason,
        capabilities=resolved_capabilities,
    )


__all__ = [
    "BackendCapabilities",
    "CUDA_CANDIDATE_STAGES",
    "NUMBA_CANDIDATE_KERNELS",
    "CapabilityError",
    "DEFAULT_STAGES",
    "ExecutionPlan",
    "FLOATING_TOLERANCES",
    "detect_capabilities",
    "gated_kernels_from_baseline",
    "gated_stages_from_baseline",
    "resolve_execution_plan",
]
