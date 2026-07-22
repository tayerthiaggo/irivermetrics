from __future__ import annotations

import numpy as np


def test_cpu_certified_reductions_have_exact_integer_and_declared_float_outputs() -> None:
    from hydrofragments.compute.backends.cpu import CPUBackend
    from hydrofragments.compute.capabilities import FLOATING_TOLERANCES

    water = np.array([[True, False, True], [False, True, True]])
    valid_obs = np.array([[True, True, False], [True, True, True]])
    backend = CPUBackend()

    np.testing.assert_array_equal(
        backend.valid_counts(valid_obs), np.array([2, 2, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        backend.wet_counts(water, valid_obs), np.array([1, 1, 1], dtype=np.int64)
    )
    np.testing.assert_allclose(
        backend.mean(water.astype(np.float32), axis=0),
        np.array([0.5, 0.5, 1.0], dtype=np.float32),
        atol=FLOATING_TOLERANCES["float32"],
        rtol=0,
    )


def test_cuda_backend_import_is_optional_and_reports_unavailability() -> None:
    from hydrofragments.compute.backends.cuda import CUDABackend

    try:
        CUDABackend()
    except RuntimeError as error:
        assert "CuPy" in str(error) or "CUDA" in str(error)


def test_cpu_and_cuda_contracts_match_with_numpy_array_api_shim() -> None:
    from hydrofragments.compute.backends.cpu import CPUBackend
    from hydrofragments.compute.backends.cuda import CUDABackend

    water = np.array([[True, False, True], [False, True, True]])
    valid_obs = np.array([[True, True, False], [True, True, True]])
    cpu = CPUBackend()
    cuda = CUDABackend(cupy_module=np)

    np.testing.assert_array_equal(
        cpu.valid_counts(valid_obs), cuda.valid_counts(valid_obs)
    )
    np.testing.assert_array_equal(
        cpu.wet_counts(water, valid_obs), cuda.wet_counts(water, valid_obs)
    )
    np.testing.assert_array_equal(
        cpu.mean(water.astype(np.float32), axis=0),
        cuda.mean(water.astype(np.float32), axis=0),
    )
