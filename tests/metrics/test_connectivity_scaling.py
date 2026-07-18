"""Test stable output for connectivity graph build and reachable-pair counting.

This module pins the numeric output of build_fixed_graph and
compute_realised_connectivity before and after O(V) refactoring (M3).
"""
from math import comb
from collections import Counter
from hydrofragments.metrics.connectivity import (
    build_fixed_graph, compute_realised_connectivity,
)


def _topology():
    # chain: A->n1->n2->n3 ; B branches at n2
    return [
        {"HydroID": "A", "From_Node": 0, "To_Node": 1},
        {"HydroID": "B", "From_Node": 1, "To_Node": 2},
        {"HydroID": "C", "From_Node": 2, "To_Node": 3},
        {"HydroID": "D", "From_Node": 1, "To_Node": 9},  # sibling of B
    ]


def test_graph_edges_stable():
    g = build_fixed_graph(_topology(), wet_any_month={k: True for k in "ABCD"})
    # A.To=1 == B.From=1 and D.From=1 -> edges (A,B),(A,D); B.To=2 == C.From=2 -> (B,C)
    assert g.edges == (("A", "B"), ("A", "D"), ("B", "C"))


def test_rc_pair_value_stable():
    g = build_fixed_graph(_topology(), wet_any_month={k: True for k in "ABCD"})
    edges = {e: 0 for e in g.edges}  # all active, gap 0
    res = compute_realised_connectivity(g, wet_gap_by_edge=edges, gap_threshold=0)
    # all 4 connected -> comb(4,2)/comb(4,2) = 100
    assert res.rc_pair_pct == 100.0
