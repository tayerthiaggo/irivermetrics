"""Test MESH unit label matches the actual computed value."""

from hydrofragments.metrics import registry


def test_mesh_unit_label_matches_value():
    """MESH value is computed in m², so registry label must be 'm2', not 'km2'."""
    spec = registry.METRIC_REGISTRY["mesh"]
    assert spec.unit == "m2"
