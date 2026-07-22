"""Milestone 11 -- V6 benchmark: reach-length-weighted RC_pair vs riverconn DCI.

Citation-only per Q4 until this benchmark passes; passing does NOT
auto-enable DCI as a shipped runtime metric -- that is a separate decision.
"""
from __future__ import annotations

import pytest

from hydrofragments.metrics.connectivity import FixedGraph, compute_length_weighted_rc_pair


def test_length_weighted_rc_pair_matches_hand_computed_dci_on_linear_graph():
    # Two reaches, lengths 10 and 30, connected -- fully connected DCI on a
    # linear graph with all fragments merged reduces to 100% by definition
    # (Cote et al. 2009): every unit of length can reach every other unit.
    graph = FixedGraph(node_source="external_network", nodes=("A", "B"), edges=(("A", "B"),))
    result = compute_length_weighted_rc_pair(
        graph, wet_gap_by_edge={("A", "B"): 0}, gap_threshold=0,
        length_by_node={"A": 10.0, "B": 30.0},
    )
    assert result == pytest.approx(100.0)


def test_length_weighted_rc_pair_disconnected_reflects_fragment_size_squared():
    # Cote et al. 2009 DCI formula on disconnected fragments of length 10
    # and 30 out of 40 total: DCI = 100 * (10^2 + 30^2) / 40^2 = 62.5
    graph = FixedGraph(node_source="external_network", nodes=("A", "B"), edges=(("A", "B"),))
    result = compute_length_weighted_rc_pair(
        graph, wet_gap_by_edge={("A", "B"): None}, gap_threshold=0,
        length_by_node={"A": 10.0, "B": 30.0},
    )
    assert result == pytest.approx(62.5)
