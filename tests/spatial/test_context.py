import pytest
from hydrofragments.spatial.context import create_spatial_context

def test_create_spatial_context():
    ctx = create_spatial_context(aoi_id="test_aoi", area_m2=1000.0)
    assert ctx.aoi_id == "test_aoi"
    assert ctx.area_m2 == 1000.0
