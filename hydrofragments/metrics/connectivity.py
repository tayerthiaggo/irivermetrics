"""Realised connectivity (RC) and temporal connectivity frequency (TCF).

Decision Gate contract for this module (U8, approved 2026-07-17; spec section
6.13 and 6.11):

- **Node source is fixed and locked: `external_network`.** Nodes are
  drainage reach `HydroID` values from the approved U4/Q6 drainage dataset
  (`data/fitzroy_kimberley_drainage.gpkg`), never derived from monthly
  water-mask patch labels. There is no transient monthly patch identity in
  this module -- a node exists for the whole series or not at all.
- **Reaches never wet across the series are pre-filtered** out of the node
  set before RC/TCF, so `|V|`, `|E_max|`, and `choose(|V|,2)` denominators
  are not diluted by structurally-dry reaches that could never activate.
- **Edge rule is fixed and locked: configurable dry-gap threshold, default
  0.** Edge `(i,j)` is active at month `t` iff the water mask shows a direct
  wet connection between adjacent reach nodes, or the dry gap between them
  is `<= connectivity_gap_threshold` pixels; default `0` is equivalent to
  direct-wet-touch-only (spec section 6.13).
- **RC is a snapshot metric.** DCI positioning: `RC_pair_t` with
  reach-length-weighted nodes is structurally close to a monthly DCI
  snapshot (spec section 6.11a); DCI itself stays citation-only unless
  riverconn/Conefor parity (V6) passes (Q4, approved).
- This module is an optional profile (`metric_profiles: ["connectivity"]`)
  and must not affect core (non-connectivity) results.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FixedGraph:
    """A fixed connectivity graph: same nodes/edges for every month.

    ``node_source`` is always ``"external_network"`` in v1.2 (U8). ``nodes``
    and ``edges`` are stable-ordered tuples so repeated construction from the
    same inputs is reproducible.
    """

    node_source: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


def build_fixed_graph(
    topology: Sequence[Mapping[str, object]],
    *,
    wet_any_month: Mapping[str, bool],
) -> FixedGraph:
    """Build the fixed RC/TCF graph from drainage reach topology.

    ``topology`` is a sequence of reach records with ``HydroID``,
    ``From_Node``, ``To_Node`` (already validated upstream by
    :func:`hydrofragments.spatial.context.validate_drainage_topology`).
    ``wet_any_month`` reports, per reach ``HydroID``, whether that reach's
    geometry intersected the water mask in at least one month of the series;
    reaches missing or `False` here are dropped from the node set entirely
    -- they can never activate, so keeping them would only dilute RC/TCF
    denominators with structurally-dry reaches.
    """
    kept_ids = {
        str(record["HydroID"])
        for record in topology
        if wet_any_month.get(str(record["HydroID"]), False)
    }
    nodes = tuple(
        str(record["HydroID"])
        for record in topology
        if str(record["HydroID"]) in kept_ids
    )

    from_to_node: dict[str, tuple[object, object]] = {
        str(record["HydroID"]): (record["From_Node"], record["To_Node"])
        for record in topology
    }

    edges: list[tuple[str, str]] = []
    for i, node_a in enumerate(nodes):
        _, to_node_a = from_to_node[node_a]
        for node_b in nodes[i + 1:]:
            from_node_b, _ = from_to_node[node_b]
            if to_node_a == from_node_b:
                edges.append((node_a, node_b))

    return FixedGraph(
        node_source="external_network",
        nodes=nodes,
        edges=tuple(edges),
    )


@dataclass(frozen=True)
class RealisedConnectivityResult:
    """One month's RC snapshot on the fixed graph (spec section 6.13).

    ``rc_edge_pct`` is the edge-fraction form; ``rc_pair_pct`` is the
    reachable-pair form (structurally close to a monthly DCI snapshot with
    reach-length node weights, spec section 6.11a -- state that relationship
    in docs, not code, per the audit's positioning requirement). Both are
    ``NaN`` (not 0) when the fixed graph has no edges/pairs to evaluate,
    since 0/0 is undefined, not "no connectivity."
    """

    rc_edge_pct: float
    rc_pair_pct: float
    n_edges_active: int
    n_edges_total: int
    node_source: str
    edge_rule: str


def _find(parent: dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent: dict[str, str], a: str, b: str) -> None:
    root_a, root_b = _find(parent, a), _find(parent, b)
    if root_a != root_b:
        parent[root_a] = root_b


def compute_realised_connectivity(
    graph: FixedGraph,
    *,
    wet_gap_by_edge: Mapping[tuple[str, str], "int | float | None"],
    gap_threshold: "int | float" = 0,
) -> RealisedConnectivityResult:
    """Compute one month's RC snapshot: edge-fraction and reachable-pair forms.

    ``wet_gap_by_edge`` maps each edge in ``graph.edges`` to the dry-gap
    distance observed that month (``0`` for direct wet touch), or ``None``
    if the edge shows no wet connection at all. An edge is active iff its
    gap value is not ``None`` and is ``<= gap_threshold`` (spec section
    6.13; default ``gap_threshold=0`` is direct-wet-touch-only, U8).
    """
    active_edges = [
        edge
        for edge in graph.edges
        if (gap := wet_gap_by_edge.get(edge)) is not None
        and gap <= gap_threshold
    ]
    n_edges_total = len(graph.edges)
    n_edges_active = len(active_edges)
    rc_edge_pct = (
        float("nan")
        if n_edges_total == 0
        else 100.0 * n_edges_active / n_edges_total
    )

    parent = {node: node for node in graph.nodes}
    for node_a, node_b in active_edges:
        _union(parent, node_a, node_b)

    total_pairs = comb(len(graph.nodes), 2)
    if total_pairs == 0:
        rc_pair_pct = float("nan")
    else:
        roots = [_find(parent, node) for node in graph.nodes]
        reachable_pairs = sum(
            1
            for i in range(len(roots))
            for j in range(i + 1, len(roots))
            if roots[i] == roots[j]
        )
        rc_pair_pct = 100.0 * reachable_pairs / total_pairs

    return RealisedConnectivityResult(
        rc_edge_pct=rc_edge_pct,
        rc_pair_pct=rc_pair_pct,
        n_edges_active=n_edges_active,
        n_edges_total=n_edges_total,
        node_source=graph.node_source,
        edge_rule=f"gap_threshold={gap_threshold}",
    )


def compute_length_weighted_rc_pair(
    graph: FixedGraph,
    *,
    wet_gap_by_edge: Mapping[tuple[str, str], "int | float | None"],
    gap_threshold: "int | float" = 0,
    length_by_node: Mapping[str, float],
) -> float:
    """Reach-length-weighted RC_pair -- the DCI form (Cote et al. 2009, spec 6.17).

    ``DCI_t = 100 * sum_{i,j}(len_i * len_j * c_ij,t) / (sum(len_i))^2`` where
    ``c_ij,t = 1`` if fragments i,j are connected under the active-edge
    subgraph (and ``c_ii = 1`` always -- a fragment is connected to itself),
    else 0. Because the sum runs over *all* ordered index pairs including the
    diagonal, this reduces to ``100 * sum_k(L_k)^2 / (sum_k L_k)^2`` where
    ``L_k`` is the total length of connected component ``k`` -- exactly the
    Cote et al. 2009 DCI: a single fully-connected network scores 100, and a
    network split into fragments of length 10 and 30 (of 40 total) scores
    ``100 * (10^2 + 30^2) / 40^2 = 62.5``.

    Positioned as citation-only validation support (Q4) -- this function
    existing does not make DCI a shipped runtime metric.
    """
    active_edges = [
        edge
        for edge in graph.edges
        if (gap := wet_gap_by_edge.get(edge)) is not None
        and gap <= gap_threshold
    ]
    parent = {node: node for node in graph.nodes}
    for node_a, node_b in active_edges:
        _union(parent, node_a, node_b)

    total_length = sum(length_by_node[node] for node in graph.nodes)
    if total_length == 0:
        return float("nan")

    component_length: dict[str, float] = {}
    for node in graph.nodes:
        root = _find(parent, node)
        component_length[root] = component_length.get(root, 0.0) + length_by_node[node]

    numerator = sum(length ** 2 for length in component_length.values())
    return 100.0 * numerator / (total_length ** 2)


@dataclass(frozen=True)
class TcfResult:
    """One fixed node's temporal connectivity frequency (spec section 6.11).

    ``tcf_pct = 100 * active_months / valid_months``. A node that is never
    validly observed across the whole series (``valid_months == 0``) reports
    ``NaN`` -- 0/0 is undefined, not "never connected." A chronically
    isolated node (valid every month, active none) legitimately reports
    ``0.0``; that is the exact case TCF exists to surface, not an error
    state.
    """

    node_id: str
    active_months: int
    valid_months: int
    tcf_pct: float


def compute_tcf(
    graph: FixedGraph,
    *,
    monthly_active: Sequence[Mapping[str, bool]],
    monthly_valid: Sequence[Mapping[str, bool]],
) -> tuple[TcfResult, ...]:
    """Compute TCF for every node in the fixed graph over a monthly series.

    ``monthly_active`` and ``monthly_valid`` are one mapping per month,
    keyed by node ``HydroID``; a month/node pair absent from a mapping is
    treated as not valid (never counted as active or valid). Node identity
    always comes from ``graph.nodes`` -- ids appearing only in the monthly
    inputs (e.g. a transient patch label) never produce a result, since TCF
    node identity is fixed by the drainage graph, not by monthly detections.
    """
    results = []
    for node_id in graph.nodes:
        valid_months = sum(
            1 for month in monthly_valid if month.get(node_id, False)
        )
        active_months = sum(
            1
            for active_month, valid_month in zip(monthly_active, monthly_valid)
            if valid_month.get(node_id, False) and active_month.get(node_id, False)
        )
        tcf_pct = (
            float("nan")
            if valid_months == 0
            else 100.0 * active_months / valid_months
        )
        results.append(
            TcfResult(
                node_id=node_id,
                active_months=active_months,
                valid_months=valid_months,
                tcf_pct=tcf_pct,
            )
        )
    return tuple(results)


__all__ = [
    "FixedGraph",
    "RealisedConnectivityResult",
    "TcfResult",
    "build_fixed_graph",
    "compute_length_weighted_rc_pair",
    "compute_realised_connectivity",
    "compute_tcf",
]
