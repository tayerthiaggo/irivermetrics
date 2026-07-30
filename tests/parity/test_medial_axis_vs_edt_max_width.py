"""W3.5 parity gate: medial-axis-derived pool width vs. global EDT maximum.

``hydrofragments/patches/morphology.py::_measure_component`` currently computes
``width_pixels`` (the maximum inscribed pool diameter) as::

    axis, dist = medial_axis(mask, return_distance=True)
    width_pixels = float((2.0 * dist[axis]).max())

i.e. it restricts the Euclidean distance transform ``dist`` to the medial-axis
skeleton pixels before taking the max. The plan text for this task (task-3.5)
asserts this is always exactly equal to the *unrestricted* global maximum of
the same distance transform::

    dist = distance_transform_edt(mask)
    width_pixels = float(2.0 * dist.max())

claiming "the medial axis is guaranteed to contain the EDT maximum, so the
max over the axis equals the global max." The brief requires this claim be
PROVEN empirically against random masks and real data before the kernel is
swapped -- not assumed.

**Empirical result: the equality claim is FALSE for `skimage.morphology.medial_axis`.**

``skimage.morphology.medial_axis`` computes its skeleton via topology-preserving
thinning (see its own docstring/source: a pixel is removed if it has more than
one neighbor AND removing it does not change the number of connected
components). This is a discrete approximation of "ridges of the distance
transform," not a hard guarantee that every global-maximum-distance pixel
survives thinning. A pixel deep in the interior of a "blob"-shaped component
(e.g. the exact center of a solid disk) can legitimately be thinned away,
because removing it does not disconnect anything -- even though it is the
unique point attaining the global EDT maximum.

Minimal reproduction (a solid disk, radius 10, the textbook case):

    >>> rr, cc = np.ogrid[-10:11, -10:11]
    >>> disk = (rr**2 + cc**2) <= 100
    >>> axis, dist = medial_axis(disk, return_distance=True)
    >>> edt = distance_transform_edt(disk)
    >>> axis[np.unravel_index(np.argmax(edt), edt.shape)]
    np.False_(False)   # the disk's own center is NOT on skimage's medial axis

And empirically on the real Fitzroy catchment fixture (``tests/wmask_ts.nc``):
9 of 274 real connected water-mask components (~3.3%) across all 63 timesteps
disagree, by up to ~12.3% relative difference in width_pixels, with the
mismatch always in the same direction: the medial-axis-restricted max
UNDER-estimates the true global EDT max, never over-estimates it (verified:
0 violations of ``new >= old`` across 300 random-mask trials plus every real
fixture component -- expected, since the axis is always a strict subset of
the mask's pixels, so restricting the max to it can only lower or match the
value, never raise it).

**Conclusion, and what this changes about the task:**

The medial-axis restriction was never a deliberate scientific choice to
under-measure pool width -- Milestone 10's own plan text describes this
field's intent as "EDT medial-axis widths" (i.e. EDT-derived, using the
medial axis as an implementation detail/convenience for finding the ridge),
not "restricted to whatever skimage's thinning heuristic happens to keep."
The global EDT max is the mathematically correct "maximum inscribed pool
diameter" -- the medial-axis-restricted value has been a latent,
silent *underestimate* for blob-shaped pools in production all along.
Switching to ``distance_transform_edt(mask).max()`` is therefore a
**bug fix bundled with the performance win**, not a value-preserving
refactor. This is explicitly pre-authorized by the plan's own text
(``docs/superpowers/plans/2026-07-27-dea-zones-and-catchment-speed.md``,
Global Constraints section): "A changed checksum means a bug, except for
3.5 where the parity test carries the argument." This test file IS that
argument: it proves (a) the new value is never smaller than the old value,
(b) they agree exactly on non-blob/ridge-dominated shapes (including the
existing frozen T-shaped fixture in ``test_morphology_width_parity.py``,
which is unaffected), and (c) documents precisely which real shapes change
and by how much, rather than silently asserting an equality that does not
hold.

This module runs its content-characterising tests against the OLD
(medial-axis) implementation first (as the RED sanity check that the parity
apparatus and real-fixture loading itself are correct), then continues to
pass unchanged after the kernel swap in ``morphology.py`` (as the GREEN
check), since every assertion here is stated in terms of the two standalone
kernel functions, not the (soon-to-change) internals of
``_measure_component``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis

from hydrofragments.patches.components import extract_component_crops
from hydrofragments.patches.labels import label_components
from hydrofragments.patches.morphology import measure_components

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "wmask_ts.nc"


def _medial_axis_width_pixels(mask: np.ndarray) -> float:
    """The OLD kernel: max distance restricted to the medial-axis skeleton."""
    axis, dist = medial_axis(mask, return_distance=True)
    return float((2.0 * dist[axis]).max())


def _edt_max_width_pixels(mask: np.ndarray) -> float:
    """The NEW kernel: max of the unrestricted Euclidean distance transform."""
    dist = distance_transform_edt(mask)
    return float(2.0 * dist.max())


def _random_mask(
    rng: np.random.Generator, *, shape: tuple[int, int], p_wet: float
) -> np.ndarray:
    return rng.random(shape) < p_wet


def _iter_real_fixture_components() -> list[np.ndarray]:
    """Every real connected component's cropped boolean mask across all
    timesteps of the real Fitzroy-catchment ``tests/wmask_ts.nc`` fixture.

    ``water == 1`` is the wet class (see ``inspect_water_mask_netcdf``'s
    ``value_domain`` handling -- this legacy netCDF fixture uses -1 as a
    sentinel/nodata value, 0 dry, 1 wet). Components with fewer than 4 pixels
    are skipped since a single/double pixel component has a degenerate EDT
    ridge that is covered separately and explicitly below.
    """
    if not FIXTURE_PATH.exists():
        return []
    ds = xr.open_dataset(FIXTURE_PATH)
    try:
        water = (ds["water"].values == 1)
    finally:
        ds.close()

    masks: list[np.ndarray] = []
    for t in range(water.shape[0]):
        frame = water[t]
        if not frame.any():
            continue
        labels = label_components(frame, connectivity=8, min_patch_pixels=1).labels
        for crop in extract_component_crops(labels):
            mask = np.asarray(crop.mask, dtype=bool)
            if np.count_nonzero(mask) >= 4:
                masks.append(mask)
    return masks


REAL_FIXTURE_COMPONENTS = _iter_real_fixture_components()


# ---------------------------------------------------------------------------
# 1. The true universal guarantee: new >= old, never new < old.
# ---------------------------------------------------------------------------
#
# This IS a hard mathematical guarantee (not merely empirical): the medial
# axis skeleton is always a subset of the mask's own pixels, so restricting
# the distance-transform max to that subset can only lower or match the
# unrestricted max over the whole mask, never raise it. The tests below
# confirm this holds with zero violations, which is the correctness property
# the kernel swap actually relies on (monotone widening, never shrinking).


@pytest.mark.parametrize("seed", range(50))
def test_random_mask_edt_max_never_smaller_than_medial_axis_max(seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = tuple(rng.integers(4, 40, size=2).tolist())
    p_wet = float(rng.uniform(0.15, 0.85))
    mask = _random_mask(rng, shape=shape, p_wet=p_wet)

    if not mask.any():
        pytest.skip("all-dry random draw; not a meaningful case for this kernel")

    old = _medial_axis_width_pixels(mask)
    new = _edt_max_width_pixels(mask)

    assert new >= old, (
        f"seed={seed} shape={shape} p_wet={p_wet}: "
        f"global EDT max {new!r} is smaller than medial-axis max {old!r}, "
        "which should be mathematically impossible (axis is a subset of mask pixels)"
    )


def test_real_fixture_edt_max_never_smaller_than_medial_axis_max() -> None:
    """Exhaustive sweep over every real component pulled from every timestep
    of the real Fitzroy fixture: new width is never smaller than old."""
    assert REAL_FIXTURE_COMPONENTS, "real fixture produced no components to check"
    violations = []
    for i, mask in enumerate(REAL_FIXTURE_COMPONENTS):
        old = _medial_axis_width_pixels(mask)
        new = _edt_max_width_pixels(mask)
        if new < old:
            violations.append((i, int(np.count_nonzero(mask)), old, new))
    assert not violations, (
        f"{len(violations)}/{len(REAL_FIXTURE_COMPONENTS)} real fixture components "
        f"had global EDT max smaller than medial-axis max (should be impossible): "
        f"{violations[:10]}"
    )


# ---------------------------------------------------------------------------
# 2. Where they DO agree exactly: non-blob / ridge-dominated shapes.
# ---------------------------------------------------------------------------
#
# For shapes whose widest point sits on a genuine thin arm or skeleton branch
# (not deep inside an interior blob that thinning can safely erode), the two
# kernels agree exactly. This is the "expected common case" and covers the
# existing frozen fixture from the m4 rewrite (``test_morphology_width_parity.py``),
# proving that rewrite's regression values are completely unaffected by this
# kernel swap.


def test_single_pixel_component_medial_axis_max_equals_global_edt_max() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert _medial_axis_width_pixels(mask) == _edt_max_width_pixels(mask)


def test_two_pixel_line_component_medial_axis_max_equals_global_edt_max() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[1, 2] = True
    assert _medial_axis_width_pixels(mask) == _edt_max_width_pixels(mask)


def test_long_thin_bar_medial_axis_max_equals_global_edt_max() -> None:
    """A single-pixel-wide bar is already its own skeleton everywhere -- no
    interior blob pixel exists for thinning to erode away."""
    mask = np.zeros((3, 20), dtype=bool)
    mask[1, :] = True
    assert _medial_axis_width_pixels(mask) == _edt_max_width_pixels(mask)


def test_irregular_branching_mask_medial_axis_max_equals_global_edt_max() -> None:
    """The same nontrivial T-shaped-with-a-notch fixture used by
    ``tests/parity/test_morphology_width_parity.py`` (m4's frozen-value
    regression gate). Multiple arms and a notch, but the widest point sits on
    a genuine arm rather than deep in an interior blob, so the medial-axis
    restriction happens not to lose it -- this is exactly why that frozen
    test's ``width_pixels``/``width_m`` values are unaffected by this task's
    kernel swap.
    """
    mask = np.zeros((14, 18), dtype=bool)
    mask[2:10, 2:6] = True
    mask[6:9, 2:14] = True
    mask[3:6, 9:13] = True
    mask[7, 6:9] = False
    assert _medial_axis_width_pixels(mask) == _edt_max_width_pixels(mask)


def test_real_fixture_majority_of_components_agree_exactly() -> None:
    """The disagreement is a real minority, not the common case: assert most
    (not necessarily all -- proven false above) real components agree
    exactly, so the kernel swap's behaviour change is characterised as rare
    and bounded, not pervasive."""
    assert REAL_FIXTURE_COMPONENTS, "real fixture produced no components to check"
    n_total = len(REAL_FIXTURE_COMPONENTS)
    n_mismatch = 0
    for mask in REAL_FIXTURE_COMPONENTS:
        old = _medial_axis_width_pixels(mask)
        new = _edt_max_width_pixels(mask)
        if old != new:
            n_mismatch += 1
    mismatch_frac = n_mismatch / n_total
    assert mismatch_frac < 0.10, (
        f"{n_mismatch}/{n_total} ({mismatch_frac:.1%}) real components disagreed -- "
        "expected a small minority (documented ~3.3% at test-authoring time), a much "
        "larger fraction would suggest the two kernels are not measuring the same "
        "quantity at all"
    )


# ---------------------------------------------------------------------------
# 3. Documented, deliberate counterexamples: real data where they DIFFER.
# ---------------------------------------------------------------------------
#
# These are not failures to fix -- they are the concrete evidence that the
# plan's equality claim is false, pinned so the exact real-world magnitude of
# the (intentional, bug-fixing) behaviour change is visible and cannot
# silently drift. Values captured directly from tests/wmask_ts.nc components.


def test_solid_disk_is_the_textbook_counterexample() -> None:
    """A solid disk's own center is the unique global EDT maximum, but
    skimage's topology-preserving thinning erodes it away (removing the
    center pixel never disconnects a disk), so the medial-axis-restricted
    max is strictly smaller than the true global max. This is the classic,
    well-known limitation of discrete thinning-based medial axis
    approximations versus the true continuous medial axis of a shape.
    """
    rr, cc = np.ogrid[-10:11, -10:11]
    disk = (rr**2 + cc**2) <= 100

    old = _medial_axis_width_pixels(disk)
    new = _edt_max_width_pixels(disk)

    assert new > old, (
        "expected the disk's center to be eroded from skimage's medial axis "
        f"(old={old!r}, new={new!r}) -- if this now fails, skimage's "
        "medial_axis implementation behaviour has changed upstream"
    )
    assert new == pytest.approx(2 * 10.04987562112089, rel=1e-9)


def _medial_axis_width_pixels_seeded(mask: np.ndarray, *, rng: int) -> float:
    """Same OLD kernel, but with an explicit ``rng`` seed for the tie-break
    order, so pinned counterexample values below are reproducible.

    ``skimage.morphology.medial_axis`` defaults to ``rng=None``, which
    reseeds a fresh ``np.random.default_rng()`` on every call -- see the
    nondeterminism finding in
    ``test_real_fixture_medial_axis_max_can_be_nondeterministic_across_calls``
    below. Pinning specific real-data counterexample values requires a fixed
    seed; the production code path does NOT pass one (a second, independent
    finding from this task), so this helper exists only for reproducible test
    pinning, not to characterise production behaviour.
    """
    axis, dist = medial_axis(mask, return_distance=True, rng=rng)
    return float((2.0 * dist[axis]).max())


def test_real_fixture_medial_axis_max_can_be_nondeterministic_across_calls() -> None:
    """A second, independent finding from this task's investigation: because
    ``_measure_component`` calls ``medial_axis(mask, return_distance=True)``
    with the default ``rng=None``, its tie-breaking order is reseeded fresh
    every call. For a "blob" component whose global EDT maximum is not
    uniquely and robustly retained by thinning, this means the CURRENT
    production ``width_pixels`` value for that exact same real component can
    differ between two calls with the exact same input -- not just between
    the old and new kernels, but between two runs of the OLD kernel alone.

    This is demonstrated directly on a real fixture component (index 105)
    that empirically takes two different values across repeated calls at
    test-authoring time. The EDT-max kernel has no such nondeterminism
    (``distance_transform_edt`` involves no random tie-breaking at all),
    so the kernel swap also fixes this run-to-run instability, not only the
    interior-blob underestimate.
    """
    mask = REAL_FIXTURE_COMPONENTS[105]
    observed = {_medial_axis_width_pixels(mask) for _ in range(20)}
    assert len(observed) > 1, (
        "expected this component's medial-axis max to vary across repeated "
        f"calls (default rng=None), got a single stable value {observed!r} -- "
        "if this now fails, skimage's medial_axis tie-breaking behaviour or "
        "this component's shape may have changed"
    )
    # The EDT-max kernel is unaffected: identical across repeated calls.
    assert len({_edt_max_width_pixels(mask) for _ in range(20)}) == 1


@pytest.mark.parametrize(
    "index,expected_old,expected_new",
    [
        (53, 8.0, 8.246211251235321),
        (93, 8.94427190999916, 10.0),
        (137, 8.0, 8.246211251235321),
        (140, 8.0, 8.246211251235321),
        (179, 8.0, 8.246211251235321),
        (184, 6.324555320336759, 7.211102550927978),
        (233, 6.324555320336759, 7.211102550927978),
    ],
)
def test_real_fixture_documented_counterexamples(
    index: int, expected_old: float, expected_new: float
) -> None:
    """Pin real-data components (by index into ``REAL_FIXTURE_COMPONENTS``,
    deterministic given the fixture file and the
    ``connectivity=8``/``min_patch_pixels=1`` labeling used above) where the
    medial-axis max and global EDT max genuinely disagree at a fixed
    ``rng=0`` tie-break seed, with their exact old/new ``width_pixels``
    values. A fixed seed is used specifically to make these pinned values
    reproducible (see the nondeterminism finding above for why the
    production default ``rng=None`` cannot be pinned this way -- two other
    real components, #105 and #204, also disagree under some tie-break
    orderings but happen to agree under ``rng=0`` specifically, which is
    exactly the nondeterminism the previous test demonstrates). If the real
    fixture file ever changes, this test's indices/expected values would
    need re-deriving -- that is a deliberate tripwire, not a maintenance
    nuisance, since a silent change to the real-data evidence backing this
    task's bug-fix argument should be caught immediately.
    """
    assert len(REAL_FIXTURE_COMPONENTS) > index, (
        "fixture produced fewer components than expected -- has "
        "tests/wmask_ts.nc changed?"
    )
    mask = REAL_FIXTURE_COMPONENTS[index]
    old = _medial_axis_width_pixels_seeded(mask, rng=0)
    new = _edt_max_width_pixels(mask)
    assert old == pytest.approx(expected_old, rel=1e-12)
    assert new == pytest.approx(expected_new, rel=1e-12)
    assert new >= old


# ---------------------------------------------------------------------------
# 4. End-to-end through the production entry point.
# ---------------------------------------------------------------------------


def test_measure_components_width_pixels_never_exceeds_global_edt_max_real_fixture() -> None:
    """Whole-pipeline check: ``measure_components(..., include_width=True)``'s
    ``width_pixels`` output, for real components drawn from the real Fitzroy
    fixture, must never exceed the independently computed global EDT max --
    the true upper bound on any inscribed-diameter measurement for that mask.

    Deliberately does NOT assert exact equality against a second, independent
    ``medial_axis(mask, return_distance=True)`` call for the OLD kernel: the
    production call site uses the default ``rng=None``, and (per the
    nondeterminism finding above) a second call with its own fresh default
    RNG can legitimately land on a different tie-break outcome than the one
    ``measure_components`` happened to get internally -- that would make an
    exact-equality assertion here flaky by construction, not a real bug. The
    ``<=`` bound is the true invariant that holds regardless of tie-break
    order, both before and after the kernel swap (before: medial-axis-restricted
    max is always <= global max; after: they are the same value, still <=).
    """
    assert REAL_FIXTURE_COMPONENTS, "real fixture produced no components to check"
    rng = np.random.default_rng(7)
    sample = rng.choice(
        len(REAL_FIXTURE_COMPONENTS),
        size=min(40, len(REAL_FIXTURE_COMPONENTS)),
        replace=False,
    )
    for i in sample:
        mask = REAL_FIXTURE_COMPONENTS[int(i)]
        labels = label_components(mask, connectivity=8, min_patch_pixels=1).labels
        (crop,) = extract_component_crops(labels)
        (measured,) = measure_components(
            (crop,), pixel_size_m=30.0, include_width=True
        )
        new = _edt_max_width_pixels(mask)
        assert measured.width_pixels <= new, (
            f"component index {i} (area={int(np.count_nonzero(mask))}px): "
            f"measure_components width_pixels {measured.width_pixels!r} exceeds "
            f"the global EDT max upper bound {new!r}"
        )


def test_measure_components_width_pixels_equals_global_edt_max_after_kernel_swap() -> None:
    """The actual post-swap correctness gate: once ``_measure_component`` is
    rewritten to use ``distance_transform_edt`` directly (this task's Step 2),
    ``measure_components(..., include_width=True)``'s ``width_pixels`` output
    must equal the independent global EDT max EXACTLY (not merely ``<=``) for
    every one of the real fixture's documented counterexample components --
    the ones where the OLD medial-axis kernel provably under-measured. This
    test is expected to FAIL against the pre-swap implementation (medial-axis
    restriction can legitimately be strictly smaller for these components,
    per the pinned counterexamples above) and PASS only once the swap is
    done; it is the concrete, falsifiable proof that the swap actually
    reaches production code, not just the standalone kernel helpers in this
    file.
    """
    counterexample_indices = [53, 93, 137, 140, 179, 184, 233]
    for i in counterexample_indices:
        mask = REAL_FIXTURE_COMPONENTS[i]
        labels = label_components(mask, connectivity=8, min_patch_pixels=1).labels
        (crop,) = extract_component_crops(labels)
        (measured,) = measure_components(
            (crop,), pixel_size_m=30.0, include_width=True
        )
        expected = _edt_max_width_pixels(mask)
        assert measured.width_pixels == expected, (
            f"component index {i} (area={int(np.count_nonzero(mask))}px): "
            f"measure_components width_pixels {measured.width_pixels!r} != "
            f"global EDT max {expected!r} -- has the kernel swap been applied?"
        )
