from __future__ import annotations

import sys

import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr


def test_safe_storage_chunks_are_retained_and_reported() -> None:
    from hydrofragments.compute.chunks import validate_chunk_budget

    source = xr.DataArray(
        da.zeros((4, 8, 8), chunks=(2, 4, 4), dtype=np.float32),
        dims=("time", "y", "x"),
    )
    original_chunks = source.chunks

    diagnostics = validate_chunk_budget(
        source,
        target_chunk_bytes=512,
        live_array_multiplier=2.0,
        stage="input_normalization",
    )

    assert source.chunks == original_chunks
    assert diagnostics.stage == "input_normalization"
    assert diagnostics.max_chunk_shape == (2, 4, 4)
    assert diagnostics.max_chunk_bytes == 128
    assert diagnostics.live_array_multiplier == 2.0
    assert diagnostics.estimated_live_bytes == 256
    assert diagnostics.target_chunk_bytes == 512


def test_chunk_budget_rejects_amplified_live_arrays() -> None:
    from hydrofragments.compute.chunks import ChunkBudgetError, validate_chunk_budget

    source = xr.DataArray(
        da.zeros((4, 8, 8), chunks=(4, 8, 8), dtype=np.uint8),
        dims=("time", "y", "x"),
    )

    with pytest.raises(
        ChunkBudgetError,
        match=r"estimated_live_bytes=768.*target_chunk_bytes=512.*multiplier=3",
    ):
        validate_chunk_budget(
            source,
            target_chunk_bytes=512,
            live_array_multiplier=3.0,
            stage="monthly_composite",
        )


def test_chunk_budget_rejects_unknown_chunk_sizes_without_computing() -> None:
    from hydrofragments.compute.chunks import ChunkBudgetError, validate_chunk_budget

    @delayed
    def unknown_length() -> np.ndarray:
        return np.zeros((3, 2), dtype=np.uint8)

    source = da.from_delayed(
        unknown_length(),
        shape=(np.nan, 2),
        dtype=np.uint8,
    )

    with pytest.raises(ChunkBudgetError, match="unknown chunk sizes"):
        validate_chunk_budget(
            source,
            target_chunk_bytes=512,
            live_array_multiplier=2.0,
            stage="input_normalization",
        )


def test_cpu_policy_needs_no_cuda_packages() -> None:
    from hydrofragments.compute.policy import ComputePolicy

    sys.modules.pop("cupy", None)
    policy = ComputePolicy(
        target_chunk_bytes=1024,
        live_array_multiplier=2.0,
        checkpoint="zarr",
        accelerator="none",
    )

    assert policy.actual_backend == "cpu"
    assert "cupy" not in sys.modules


def test_cuda_request_fails_instead_of_silently_reporting_cpu() -> None:
    from hydrofragments.compute.policy import ComputePolicy, ComputePolicyError

    with pytest.raises(ComputePolicyError, match="CUDA.*not certified"):
        ComputePolicy(accelerator="cuda")


@pytest.mark.parametrize("checkpoint", ["memory", "disk", "cuda"])
def test_policy_rejects_unknown_checkpoint_modes(checkpoint: str) -> None:
    from hydrofragments.compute.policy import ComputePolicy, ComputePolicyError

    with pytest.raises(ComputePolicyError, match="checkpoint"):
        ComputePolicy(checkpoint=checkpoint)
