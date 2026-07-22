# Validity / reliability sensitivity report (U2 / Q1)

**Captured:** 2026-07-14
**Input:** `data/wofs_monthly_masks_1986_2026.zarr` — Fitzroy (Kimberley) monthly water mask, 1986–2026
**Provenance:** byte-identical (`.zmetadata` SHA-256 `c69f7e8b0706...36e790`) to maintainer-supplied
`D:\RLH\5.6\data_local\raw\WaterMask-TSFill\cache\wofs_monthly_masks_1986_2026.zarr`
**Method:** full-cube scan, no mutation. Per-month class counts computed directly from `water_mask` values;
raw per-month table archived at `docs/audit/evidence/validity_reliability_per_month.csv` (480 rows,
SHA-256 `716d9494ff04...31eda2`).

## 1. What this real product actually exposes

This is **not** the four-variable canonical contract (`water_mask`/`confidence`/`method_flag`/`observed`)
described in `upstream_validity_contract.md` from `WaterMask-TSFill/watermask_tsfill/contracts.py`. The
delivered artifact is a **single-variable** Zarr:

| Array | Shape | Dtype | Role |
|---|---|---|---|
| `water_mask` | `(480, 539, 1117)` | `int8` | Only data variable |
| `time` | `(480,)` | `int64` | Monthly, 1987-01 → 2026-12 (`days since 1987-01-01`) |
| `x`, `y` | — | `float64` | EPSG:3577, 30 m grid |

`water_mask` value domain (confirmed by full-cube scan, no other values present):

| Value | Count (of 289,990,240 cells) | Share | Meaning |
|---|---|---|---|
| `-2` | 47,469,060 | 16.4% | Outside AOI / static nodata mask |
| `-1` | 29,764,676 | 10.3% | **Unobserved** (in-AOI, no resolvable observation that month) |
| `0` | 208,072,191 | 71.8% | Dry (observed) |
| `1` | 3,684,313 | 1.3% | Wet (observed) |

Per-pixel `confidence` and `method_flag` bands **do not exist** in this delivery. Provenance is exposed only
at **whole-month granularity** via two global attributes on `water_mask`: `inserted_months` (15 months,
fully gap-filled with no source imagery) and `source_months` (465 months, nominally real).

**Consequence for the P-provenance policy in `upstream_validity_contract.md`:** it is not implementable
against this delivered artifact as-is — there is no `method_flag` to filter on. P-provenance would require
either a future upstream delivery that includes the per-pixel bands, or accepting month-level-only
provenance (see finding 3 below, which shows month-level provenance is itself incomplete).

## 2. Denominator sensitivity: P-native vs P-resolved

Two candidate denominators were computed per month directly from this cube:

- **P-native** wet-fraction = `count(==1) / (count(==1) + count(==0))` — i.e. observed pixels only.
- **P-resolved** wet-fraction = `count(==1) / (count(==1) + count(==0) + count(==-1))` — i.e. all in-AOI
  pixels, treating unobserved pixels as if they were part of a fixed, always-valid denominator (they are
  implicitly counted as "not wet").

| Statistic | Value |
|---|---|
| `observed_frac_of_aoi` — min | 0.0000 (18 of 480 months have **zero** observed pixels: the 15 `inserted_months` plus 3 nominal "source" months — see §3) |
| `observed_frac_of_aoi` — 5th pct | 0.0725 |
| `observed_frac_of_aoi` — median | 0.9785 |
| `observed_frac_of_aoi` — mean | 0.8824 |
| \|P-native − P-resolved\| wet-fraction — median | 0.0002 |
| \|P-native − P-resolved\| wet-fraction — mean | 0.0027 |
| \|P-native − P-resolved\| wet-fraction — **max** | **0.2708** (27.1 percentage points, 2011-03) |
| correlation(\|diff\|, `observed_frac_of_aoi`) | **−0.311** — divergence grows as coverage drops, as expected |

**Finding:** for the ~93% of months with `observed_frac_of_aoi ≥ 0.90`, the two denominators are
practically interchangeable (median divergence 0.02 pp). For the remaining ~7% of months, divergence is
large enough to change occurrence/RA interpretation materially (up to 27 pp on wet-fraction in the worst
case). A fixed denominator choice without a reliability flag would silently mix these two regimes.

## 3. Month-level provenance attribute is incomplete

The `inserted_months` attribute (15 months) undercounts genuinely unreliable months. Full-cube scan finds
**28 of the 465 "source" (non-inserted) months (6.0%)** have `observed_frac_of_aoi < 0.50`, including
**3 months labeled "source" that have zero observed pixels** (`1991-01`, `1992-12`, `1996-03` at
`observed_frac_of_aoi = 0.0000/0.0000/0.0000`), indistinguishable in the attribute from a fully-reliable
month.

Histogram of `observed_frac_of_aoi` for the 465 nominal "source" months:

| Range | Months |
|---|---|
| [0.00, 0.10) | 10 |
| [0.10, 0.30) | 11 |
| [0.30, 0.50) | 7 |
| [0.50, 0.70) | 9 |
| [0.70, 0.90) | 30 |
| [0.90, 0.95) | 12 |
| [0.95, 0.99) | 386 |

**Finding:** the whole-cube `inserted_months`/`source_months` attribute lists are **not sufficient** as a
reliability signal on their own. Any HydroFragments reliability diagnostic must be computed empirically
from the per-pixel mask each period (`observed_frac_of_aoi` as derived above), not read from upstream
metadata attributes.

## 4. Seasonal missingness is not at random (MNAR) — the load-bearing finding

Section 2 above treats reliability as a single scalar per *month* (a cross-sectional view). But
occurrence/RA/recurrence/hydroperiod are **temporal aggregates per pixel/AOI across the full record**, and
for those metrics the *distribution of missingness across calendar months* matters more than its overall
magnitude. Mean `observed_frac_of_aoi` and mean P-native wet-fraction, grouped by calendar month across all
40 years:

| Month | Mean observed coverage | Mean wet-fraction (P-native, among observed) |
|---|---|---|
| Jan | 78.8% | 2.07% |
| Feb | 78.4% | **8.80%** |
| Mar | 82.3% | 5.65% |
| Apr | 93.1% | 1.58% |
| May | 96.7% | 0.88% |
| Jun | 86.3% | 0.69% |
| Jul | 90.0% | 0.61% |
| Aug | 90.2% | 0.51% |
| Sep | 91.1% | 0.39% |
| Oct | 95.3% | 0.30% |
| Nov | 92.2% | 0.23% |
| Dec | 84.5% | 0.55% |

**Coverage is lowest exactly when wetness is highest** (Jan–Mar monsoon peak: 78–82% coverage vs 90–97% in
the dry season) — cloud cover obscures observation *because* it is raining, and it is raining *because* it
is monsoon season, which is when the catchment is wettest. This is missing-not-at-random (MNAR) with
respect to the very quantity being measured, not merely missing-at-random noise.

**Consequence:** a naive pooled P-native ratio (sum wet / sum observed across all 480 months) implicitly
weights each calendar month by how often it was actually observed. Since dry-season months are observed
more often, the pooled estimate over-weights dry conditions and under-weights wet conditions relative to
their true calendar-month frequency. Neither P-native nor P-resolved as a pure per-timestep denominator
choice fixes this — P-resolved makes it worse (unobserved wet-season pixels are actively counted as dry);
naive-pooled P-native merely avoids injecting a wrong signal, but still under-samples the wet season.

**Quantified on the real Fitzroy cube (whole-AOI aggregate, all 480 months):**

| Estimator | Wet-fraction |
|---|---|
| Naive pooled P-native (sum wet / sum observed, all months) | 1.7399% |
| Season-stratified P-native (mean of 12 per-calendar-month ratios, equal-weighted) | **1.8548%** |
| Absolute difference | +0.115 pp |
| Relative difference | **+6.6%** |

A 6.6% relative underestimate from the naive pooled approach, at whole-AOI aggregate level, purely from
calendar-month sampling imbalance. This is expected to be larger for specific pixels/features where wetness
is more concentrated in the monsoon peak (e.g. floodplain/ephemeral cells vs perennial pools), and is a
**minimum bound**, not a worst case.

## 5. Recommended policy (evidence-backed, pending maintainer sign-off)

1. **Denominator = P-native equivalent** for any per-period cross-sectional value: valid/observed =
   `{0, 1}` (dry, wet); `-1` (unobserved) and `-2` (outside AOI) excluded from both numerator and
   denominator.
2. **Any temporal aggregate (occurrence, RA, recurrence, hydroperiod, dry-down anchors) MUST use a
   season-stratified estimator**, not a simple pooled ratio: compute the P-native ratio separately per
   calendar month across the record, then combine the 12 calendar-month estimates with equal weight. A
   naive pooled ratio is **not acceptable** for these metrics given the confirmed MNAR seasonal pattern.
3. **Mandatory reliability diagnostics**, computed per output period/pixel × AOI, not sourced from upstream
   attributes:
   - `observed_frac_of_aoi` — per period, as in §2.
   - `observed_frac_by_calendar_month` — 12 values, to expose the seasonal coverage imbalance directly.
   - `naive_vs_stratified_delta` — difference between the pooled and season-stratified estimate, so users
     can see the magnitude of the correction being applied.
4. **Low-confidence flag**: periods with `observed_frac_of_aoi < 0.70` are marked `low_confidence=true` in
   tidy output and excluded from headline summary statistics by default (opt-in to include). The 0.70
   threshold sits in the histogram gap between the long reliable tail (≥0.95, 386 months) and the unreliable
   cluster (<0.90, 69 months); 0.70 keeps the 30-month [0.70,0.90) band as usable-with-caveat rather than
   excluded. This flag is orthogonal to the seasonal-stratification fix — it catches whole-period dropouts,
   not calendar-month sampling imbalance.
5. **P-provenance is deferred**, not abandoned: this delivered product cannot support it (no
   `method_flag`/`confidence` bands). If/when WaterMask-TSFill delivers the full four-variable contract for
   a validation catchment, re-run this same sensitivity script against it before adopting P-provenance.
6. Document explicitly in manager-facing docs that (a) `-1` months/pixels are absent from P-native metrics,
   (b) temporal occurrence-style numbers are season-stratified, not simple time-averages, and why — surface
   `observed_frac_of_aoi` and `observed_frac_by_calendar_month` alongside every occurrence/RA number.

**Scope impact:** item 2 is a real algorithm-design requirement, not a denominator toggle — it affects how
occurrence/RA/recurrence/hydroperiod are computed in Milestones 3/5/9/10, not just how validity is
filtered. This should be reflected in the implementation plan for those milestones.

## 6. Reproducibility

Analysis performed with `zarr`/`numpy` full-cube scan (no `xarray`, no mutation of source data). Whole-cube
class totals and per-month table are in `docs/audit/evidence/validity_reliability_per_month.csv`. This
report and the CSV should be promoted into a `tests/contracts/` reliability-diagnostic test once the policy
above is signed off, so `observed_frac_of_aoi` computation is regression-tested against this real fixture.
