"""Optional accelerator detection, stage planning, and truthful backend records."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
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


def detect_capabilities(*, probe_cuda: bool = False) -> BackendCapabilities:
    """Return capability evidence without importing CuPy by default.

    CPU policy construction calls this function with its default. Accelerator
    resolution opts into the import and catches all runtime initialization
    failures so auto mode can report a truthful CPU fallback.
    """

    if not probe_cuda:
        return BackendCapabilities.cpu_only("CUDA probe disabled by CPU policy")

    try:
        cupy = importlib.import_module("cupy")
    except Exception as error:  # optional dependency and runtime failures
        return BackendCapabilities.cpu_only(f"CuPy unavailable: {error}")

    try:
        if not bool(cupy.is_available()):
            return BackendCapabilities(
                cupy_available=True,
                cupy_version=getattr(cupy, "__version__", None),
                reason="CUDA runtime unavailable",
            )
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        if device_count < 1:
            return BackendCapabilities(
                cupy_available=True,
                cupy_version=getattr(cupy, "__version__", None),
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
            reason=f"CUDA initialization failed: {error}",
        )

    # Deliberately empty: benchmark evidence has not approved any stage yet.
    return BackendCapabilities(
        cupy_available=True,
        cuda_available=True,
        cupy_version=getattr(cupy, "__version__", None),
        cuda_runtime_version=runtime_version,
        device_count=device_count,
        free_memory_bytes=int(free_bytes),
        total_memory_bytes=int(total_bytes),
        reason="CUDA detected; no stage has transfer-cost benefit evidence",
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
    "CapabilityError",
    "DEFAULT_STAGES",
    "ExecutionPlan",
    "FLOATING_TOLERANCES",
    "detect_capabilities",
    "resolve_execution_plan",
]
