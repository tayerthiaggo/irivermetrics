# HydroFragments Metric Specification

This document provides the formal mathematical definitions and algorithm specifications for metrics computed by HydroFragments `0.1.0`.

## 1. Metric Families

HydroFragments metrics are grouped into seven canonical families:

1. **Extent:** Surface water area, APSEC (Area Percentage of Section Covered), valid observation statistics.
2. **Persistence:** Water occurrence frequency ($P$) and refuge area ($RA$).
3. **Fragmentation:** Number of pools ($N_p$), Largest Patch Index ($LPI$), patch area distribution.
4. **Morphology:** Area-Weighted Relative Edge ($AWRe$), Area-Weighted Mean Shape Index ($AWMSI$), pool width statistics.
5. **Dynamics:** Wetting and drying transition rates and temporal persistence classes.
6. **Channel:** Channel-aligned wet reach profiles and pool spacing along drainage centrelines.
7. **Connectivity:** Structural river connectivity indices ($RC$, $TCF$, $DCI$).

## 2. Mathematical Definitions

### 2.1 APSEC (Area Percentage of Section Covered)

$$\text{APSEC}(t) = \frac{\sum_{i \in \text{AOI}} W(i, t) \cdot a_{\text{pixel}}}{A_{\text{section}}} \times 100$$

where $W(i, t) \in \{0, 1\}$ indicates surface water presence, $a_{\text{pixel}}$ is pixel area (m$^2$), and $A_{\text{section}}$ is total section area (m$^2$).

### 2.2 Occurrence Frequency

$$P(i) = \frac{\sum_{t \in V_i} W(i, t)}{|V_i|} \times 100$$

where $V_i$ is the set of valid observations at pixel $i$.

### 2.3 Refuge Area

$$\text{RA} = \sum_{i \in \text{AOI}} \mathbb{I}\left(P(i) \ge P_{\text{threshold}}\right) \cdot a_{\text{pixel}}$$

where $P_{\text{threshold}}$ defaults to $80\%$.

### 2.4 Largest Patch Index (LPI)

$$\text{LPI}(t) = \frac{\max_{k} A_k(t)}{A_{\text{section}}} \times 100$$

where $A_k(t)$ is the area of water patch $k$ at time step $t$.

### 2.5 Area-Weighted Mean Shape Index (AWMSI)

$$\text{AWMSI}(t) = \sum_{k} \left[ \left( \frac{p_k(t)}{4 \sqrt{A_k(t)}} \right) \left( \frac{A_k(t)}{\sum_j A_j(t)} \right) \right]$$

where $p_k(t)$ is the perimeter of patch $k$.

### 2.6 Area-Weighted Relative Edge (AWRe)

$$\text{AWRe}(t) = \frac{\sum_k p_k(t)}{\sqrt{A_{\text{section}}}}$$

## 3. Input Contract and Validity Rules

- Inputs must be aligned DataArrays with dimensions `("time", "y", "x")`.
- `valid_obs` masks restrict occurrence denominators to valid observations.
- Missing values (-1, -2, NaN) are excluded from valid observation totals.
