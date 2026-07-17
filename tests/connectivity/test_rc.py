"""Milestone 11 -- realised connectivity (RC), spec section 6.13.

Load-bearing contract (U8, approved 2026-07-17):

- ``RC_t = 100 * sum(e_ij,t) / |E_max|`` -- fraction of possible edges active.
- ``RC_pair_t = 100 * sum(I(component_i(t)=component_j(t))) / choose(|V|,2)``
  -- fraction of reachable node pairs, via connected components on the
  currently-active edge subgraph.
- Edge rule is fixed: configurable dry-gap threshold, default 0 (direct wet
  touch only when gap_threshold=0).
- RC is a snapshot metric on the fixed graph -- no monthly patch identity.
"""
from __future__ import annotations

import math

import pytest

from hydrofragments.metrics.connectivity import FixedGraph, compute_realised_connectivity


def _linear_graph() -> FixedGraph:
    # A - B - C - D chain, 3 edges, 4 nodes -> choose(4,2) = 6 pairs.
    return FixedGraph(
        node_source="external_network",
        nodes=("A", "B", "C", "D"),
        edges=(("A", "B"), ("B", "C"), ("C", "D")),
    )


def test_rc_edge_fraction_all_edges_active():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): 0, ("B", "C"): 0, ("C", "D"): 0},
        gap_threshold=0,
    )

    assert result.rc_edge_pct == pytest.approx(100.0)


def test_rc_edge_fraction_half_edges_active():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): 0, ("B", "C"): None, ("C", "D"): 0},
        gap_threshold=0,
    )

    assert result.n_edges_active == 2
    assert result.n_edges_total == 3
    assert result.rc_edge_pct == pytest.approx(200.0 / 3.0)


def test_rc_edge_fraction_no_edges_active():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): None, ("B", "C"): None, ("C", "D"): None},
        gap_threshold=0,
    )

    assert result.rc_edge_pct == pytest.approx(0.0)


def test_rc_pair_fully_connected_chain():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): 0, ("B", "C"): 0, ("C", "D"): 0},
        gap_threshold=0,
    )

    # All 4 nodes in one component -> all choose(4,2)=6 pairs reachable.
    assert result.rc_pair_pct == pytest.approx(100.0)


def test_rc_pair_broken_middle_edge_splits_reachability():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): 0, ("B", "C"): None, ("C", "D"): 0},
        gap_threshold=0,
    )

    # Components {A,B} and {C,D}: reachable pairs = AB, CD = 2 of 6.
    assert result.rc_pair_pct == pytest.approx(200.0 / 6.0)


def test_rc_pair_isolated_nodes_have_zero_reachability():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): None, ("B", "C"): None, ("C", "D"): None},
        gap_threshold=0,
    )

    assert result.rc_pair_pct == pytest.approx(0.0)


def test_dry_gap_within_threshold_activates_edge():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): 2, ("B", "C"): 0, ("C", "D"): 0},
        gap_threshold=2,
    )

    assert result.n_edges_active == 3
    assert result.rc_edge_pct == pytest.approx(100.0)


def test_dry_gap_beyond_threshold_does_not_activate_edge():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): 3, ("B", "C"): 0, ("C", "D"): 0},
        gap_threshold=2,
    )

    assert result.n_edges_active == 2


def test_default_gap_threshold_is_zero_direct_touch_only():
    graph = _linear_graph()
    # No config passed: default gap_threshold=0 means a 1-pixel gap does not
    # activate the edge.
    result = compute_realised_connectivity(
        graph,
        wet_gap_by_edge={("A", "B"): 1, ("B", "C"): 0, ("C", "D"): 0},
    )

    assert result.n_edges_active == 2


def test_single_node_graph_has_no_pairs_and_reports_nan_rc_pair():
    graph = FixedGraph(node_source="external_network", nodes=("A",), edges=())
    result = compute_realised_connectivity(
        graph, wet_gap_by_edge={}, gap_threshold=0
    )

    assert math.isnan(result.rc_pair_pct)


def test_empty_graph_has_no_edges_and_reports_nan_rc_edge():
    graph = FixedGraph(node_source="external_network", nodes=(), edges=())
    result = compute_realised_connectivity(
        graph, wet_gap_by_edge={}, gap_threshold=0
    )

    assert math.isnan(result.rc_edge_pct)
    assert math.isnan(result.rc_pair_pct)


def test_result_records_node_source_and_edge_rule():
    graph = _linear_graph()
    result = compute_realised_connectivity(
        graph, wet_gap_by_edge={("A", "B"): 0, ("B", "C"): 0, ("C", "D"): 0},
        gap_threshold=0,
    )

    assert result.node_source == "external_network"
    assert result.edge_rule == "gap_threshold=0"
