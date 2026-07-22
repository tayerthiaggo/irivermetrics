"""Milestone 11 — fixed connectivity graph construction (spec section 6.13).

Load-bearing contract (U8, approved 2026-07-17):

- Node source is `external_network`: nodes are drainage reach `HydroID`
  values from the approved U4/Q6 drainage dataset, never monthly patch
  labels.
- Reaches that are never wet across the whole series are pre-filtered out
  of the node set before RC/TCF, so denominators are not diluted by
  structurally-dry reaches.
- The node set and edge set (`E_max`) are identical for every month in the
  series -- only which edges are *active* changes month to month. This test
  file only covers graph construction, not monthly activation.
"""
from __future__ import annotations

import pytest

from hydrofragments.metrics.connectivity import FixedGraph, build_fixed_graph


def _linear_topology() -> list[dict]:
    # Three-reach headwater -> outlet chain: A -> B -> C.
    return [
        {"HydroID": "A", "From_Node": 1, "To_Node": 2, "NextDownID": "B"},
        {"HydroID": "B", "From_Node": 2, "To_Node": 3, "NextDownID": "C"},
        {"HydroID": "C", "From_Node": 3, "To_Node": 4, "NextDownID": "-1"},
    ]


def test_node_source_is_external_network():
    graph = build_fixed_graph(
        _linear_topology(), wet_any_month={"A": True, "B": True, "C": True}
    )

    assert graph.node_source == "external_network"


def test_never_wet_reaches_are_filtered_from_node_set():
    graph = build_fixed_graph(
        _linear_topology(), wet_any_month={"A": True, "B": False, "C": True}
    )

    assert graph.nodes == ("A", "C")


def test_edges_only_connect_reaches_present_in_filtered_node_set():
    graph = build_fixed_graph(
        _linear_topology(), wet_any_month={"A": True, "B": False, "C": True}
    )

    # B is filtered out, so the A-B and B-C topology edges cannot appear.
    assert graph.edges == ()


def test_edges_follow_shared_node_topology_adjacency():
    graph = build_fixed_graph(
        _linear_topology(), wet_any_month={"A": True, "B": True, "C": True}
    )

    assert graph.edges == (("A", "B"), ("B", "C"))


def test_node_and_edge_set_identical_across_repeated_calls():
    topology = _linear_topology()
    wet_any_month = {"A": True, "B": True, "C": True}

    first = build_fixed_graph(topology, wet_any_month=wet_any_month)
    second = build_fixed_graph(topology, wet_any_month=wet_any_month)

    assert first.nodes == second.nodes
    assert first.edges == second.edges


def test_all_reaches_never_wet_yields_empty_graph():
    graph = build_fixed_graph(
        _linear_topology(), wet_any_month={"A": False, "B": False, "C": False}
    )

    assert graph.nodes == ()
    assert graph.edges == ()


def test_fixed_graph_is_frozen_dataclass_instance():
    graph = build_fixed_graph(
        _linear_topology(), wet_any_month={"A": True, "B": True, "C": True}
    )

    assert isinstance(graph, FixedGraph)
    with pytest.raises(AttributeError):
        graph.nodes = ()  # type: ignore[misc]
