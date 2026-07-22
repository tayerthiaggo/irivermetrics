"""Milestone 11 -- Temporal Connectivity Frequency (TCF), spec section 6.11.

Load-bearing contract (U8, approved 2026-07-17):

- ``TCF = active_months / valid_months`` per fixed node -- not per calendar
  span, so months with no valid observation for a node do not penalise it.
- **Chronically isolated** node: valid every month, active never ->
  ``tcf_pct == 0.0`` (not skipped -- this is the exact case TCF exists to
  surface, spec section 6.11 "distinguishes reliably-linked from
  chronically-isolated refuges").
- **Always-connected** node: valid every month, active every month ->
  ``tcf_pct == 100.0``.
- A node with zero valid months across the whole series (never observed)
  reports ``tcf_pct = NaN`` -- 0/0 is undefined, not "never connected."
- Node identity is fixed (`HydroID`), never a transient monthly patch label.
"""
from __future__ import annotations

import math

import pytest

from hydrofragments.metrics.connectivity import FixedGraph, compute_tcf


def _graph() -> FixedGraph:
    return FixedGraph(
        node_source="external_network",
        nodes=("A", "B", "C"),
        edges=(("A", "B"), ("B", "C")),
    )


def test_chronically_isolated_node_has_zero_tcf():
    graph = _graph()
    # Node A: valid all 3 months, never active.
    monthly_active = [{"A": False}, {"A": False}, {"A": False}]
    monthly_valid = [{"A": True}, {"A": True}, {"A": True}]

    results = compute_tcf(graph, monthly_active=monthly_active, monthly_valid=monthly_valid)
    by_node = {result.node_id: result for result in results}

    assert by_node["A"].tcf_pct == pytest.approx(0.0)
    assert by_node["A"].active_months == 0
    assert by_node["A"].valid_months == 3


def test_always_connected_node_has_full_tcf():
    graph = _graph()
    monthly_active = [{"B": True}, {"B": True}, {"B": True}]
    monthly_valid = [{"B": True}, {"B": True}, {"B": True}]

    results = compute_tcf(graph, monthly_active=monthly_active, monthly_valid=monthly_valid)
    by_node = {result.node_id: result for result in results}

    assert by_node["B"].tcf_pct == pytest.approx(100.0)


def test_tcf_denominator_is_valid_months_not_calendar_months():
    graph = _graph()
    # 4 calendar months, but C only valid in 2 of them; active in both.
    monthly_active = [{"C": True}, {"C": False}, {"C": True}, {"C": False}]
    monthly_valid = [{"C": True}, {"C": False}, {"C": True}, {"C": False}]

    results = compute_tcf(graph, monthly_active=monthly_active, monthly_valid=monthly_valid)
    by_node = {result.node_id: result for result in results}

    assert by_node["C"].valid_months == 2
    assert by_node["C"].active_months == 2
    assert by_node["C"].tcf_pct == pytest.approx(100.0)


def test_node_never_valid_reports_nan_not_zero():
    graph = _graph()
    monthly_active = [{"A": False}, {"A": False}]
    monthly_valid = [{"A": False}, {"A": False}]

    results = compute_tcf(graph, monthly_active=monthly_active, monthly_valid=monthly_valid)
    by_node = {result.node_id: result for result in results}

    assert by_node["A"].valid_months == 0
    assert math.isnan(by_node["A"].tcf_pct)


def test_partial_connectivity_produces_fractional_tcf():
    graph = _graph()
    monthly_active = [{"A": True}, {"A": False}, {"A": True}, {"A": False}]
    monthly_valid = [{"A": True}, {"A": True}, {"A": True}, {"A": True}]

    results = compute_tcf(graph, monthly_active=monthly_active, monthly_valid=monthly_valid)
    by_node = {result.node_id: result for result in results}

    assert by_node["A"].active_months == 2
    assert by_node["A"].valid_months == 4
    assert by_node["A"].tcf_pct == pytest.approx(50.0)


def test_missing_month_entry_treated_as_not_valid():
    graph = _graph()
    # Month 2 has no key at all for A -- must not count as valid or active.
    monthly_active = [{"A": True}, {}]
    monthly_valid = [{"A": True}, {}]

    results = compute_tcf(graph, monthly_active=monthly_active, monthly_valid=monthly_valid)
    by_node = {result.node_id: result for result in results}

    assert by_node["A"].valid_months == 1
    assert by_node["A"].active_months == 1
    assert by_node["A"].tcf_pct == pytest.approx(100.0)


def test_results_cover_every_node_in_fixed_graph():
    graph = _graph()
    monthly_active = [{"A": True, "B": True, "C": False}]
    monthly_valid = [{"A": True, "B": True, "C": True}]

    results = compute_tcf(graph, monthly_active=monthly_active, monthly_valid=monthly_valid)
    node_ids = {result.node_id for result in results}

    assert node_ids == {"A", "B", "C"}


def test_no_transient_monthly_patch_identity_node_ids_match_fixed_graph():
    # Node identity comes from the fixed graph, not from any monthly input
    # -- passing extra keys not present in graph.nodes must not create
    # results for them.
    graph = _graph()
    monthly_active = [{"A": True, "PATCH_9": True}]
    monthly_valid = [{"A": True, "PATCH_9": True}]

    results = compute_tcf(graph, monthly_active=monthly_active, monthly_valid=monthly_valid)
    node_ids = {result.node_id for result in results}

    assert node_ids == {"A", "B", "C"}
    assert "PATCH_9" not in node_ids
