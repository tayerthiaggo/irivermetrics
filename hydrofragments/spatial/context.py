from dataclasses import dataclass
from typing import Optional

@dataclass
class SpatialContext:
    aoi_id: str
    area_m2: float
    drainage_id: Optional[str] = None
    l_ref_m: Optional[float] = None

def create_spatial_context(aoi_id: str, area_m2: float) -> SpatialContext:
    return SpatialContext(aoi_id=aoi_id, area_m2=area_m2)
