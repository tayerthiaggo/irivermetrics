from __future__ import annotations

import pytest


def test_cuda_request_without_capabilities_still_raises_exactly_as_before() -> None:
    """Backward compatibility: existing callers that don't pass `capabilities`
    keep today's blanket refuse-cuda behavior unchanged."""
    from hydrofragments.compute.policy import ComputePolicy, ComputePolicyError

    with pytest.raises(ComputePolicyError, match="CUDA.*not certified"):
        ComputePolicy(accelerator="cuda")


def test_cuda_request_with_capabilities_but_no_enabled_stages_still_raises() -> None:
    from hydrofragments.compute.capabilities import BackendCapabilities
    from hydrofragments.compute.policy import ComputePolicy, ComputePolicyError

    capabilities = BackendCapabilities.cpu_only("CUDA unavailable")
    with pytest.raises(ComputePolicyError, match="CUDA.*not certified"):
        ComputePolicy(accelerator="cuda", capabilities=capabilities)


def test_cuda_request_with_evidence_gated_stage_does_not_raise() -> None:
    """A capabilities object carrying evidence-approved CUDA stages unlocks
    the accelerator='cuda' lane -- this is the whole point of the relaxation."""
    from hydrofragments.compute.capabilities import BackendCapabilities
    from hydrofragments.compute.policy import ComputePolicy

    capabilities = BackendCapabilities(
        cupy_available=True,
        cuda_available=True,
        enabled_cuda_stages=("valid_counts",),
        reason="CUDA detected; baseline evidence enables: valid_counts",
    )
    policy = ComputePolicy(accelerator="cuda", capabilities=capabilities)

    assert policy.accelerator == "cuda"


def test_default_capabilities_is_safe_no_cuda_stages_enabled() -> None:
    """The default `capabilities=None` must resolve to a safe "no CUDA
    stages enabled" object, not merely skip validation."""
    from hydrofragments.compute.policy import ComputePolicy

    policy = ComputePolicy(accelerator="none")

    assert policy.capabilities.enabled_cuda_stages == ()
    assert policy.capabilities.cuda_available is False
