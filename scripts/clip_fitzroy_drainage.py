"""Clip AHGFNetworkStream to Fitzroy Kimberley AOI from Geofabric zip."""
from pathlib import Path

from osgeo import gdal, ogr, osr

gdal.UseExceptions()

GDB = r"/vsizip/D:/RLH/5.6/data_local/raw/SH_Network_GDB_V3_3.zip/SH_Network_GDB/SH_Network.gdb"
AOI_PATH = Path(r"D:/RLH/5.6/repos/HydroFragments/data/fitzroy_kimberley_aoi.geojson")
OUT_GPKG = Path(r"D:/RLH/5.6/repos/HydroFragments/data/fitzroy_kimberley_drainage.gpkg")
OUT_GEOJSON = Path(r"D:/RLH/5.6/repos/HydroFragments/data/fitzroy_kimberley_drainage.geojson")


def _srs_epsg(code: int) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(code)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _extract_lines(geom: ogr.Geometry) -> ogr.Geometry | None:
    """Return MultiLineString from intersection geometry; drop points/polygons."""
    if geom is None or geom.IsEmpty():
        return None
    gtype = geom.GetGeometryType()
    line_types = {
        ogr.wkbLineString,
        ogr.wkbLineString25D,
        ogr.wkbMultiLineString,
        ogr.wkbMultiLineString25D,
    }
    if gtype in (ogr.wkbLineString, ogr.wkbLineString25D):
        ml = ogr.Geometry(ogr.wkbMultiLineString)
        ml.AddGeometry(geom)
        return ml
    if gtype in (ogr.wkbMultiLineString, ogr.wkbMultiLineString25D):
        return geom
    if gtype in (ogr.wkbGeometryCollection, ogr.wkbGeometryCollection25D):
        ml = ogr.Geometry(ogr.wkbMultiLineString)
        for j in range(geom.GetGeometryCount()):
            part = geom.GetGeometryRef(j)
            pt = part.GetGeometryType()
            if pt in (ogr.wkbLineString, ogr.wkbLineString25D):
                ml.AddGeometry(part)
            elif pt in (ogr.wkbMultiLineString, ogr.wkbMultiLineString25D):
                for k in range(part.GetGeometryCount()):
                    ml.AddGeometry(part.GetGeometryRef(k))
        return ml if ml.GetGeometryCount() else None
    if gtype not in line_types:
        return None
    return geom


def main() -> None:
    aoi_ds = ogr.Open(str(AOI_PATH))
    aoi_lyr = aoi_ds.GetLayer(0)
    aoi_feat = aoi_lyr.GetNextFeature()
    aoi_geom = aoi_feat.GetGeometryRef().Clone()
    # CRS84 / WGS84 ≈ GDA94 lon/lat for this AOI clip; treat AOI as EPSG:4326
    aoi_srs = _srs_epsg(4326)
    aoi_geom.AssignSpatialReference(aoi_srs)

    src = ogr.Open(GDB)
    streams = src.GetLayerByName("AHGFNetworkStream")
    src_srs = streams.GetSpatialRef()
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    print("stream CRS:", src_srs.GetAuthorityCode(None))

    # EPSG:4326 and GDA94 (4283) are locally equivalent here — use AOI as-is in lon/lat
    aoi_in_src = aoi_geom.Clone()
    aoi_in_src.AssignSpatialReference(src_srs)
    minx, maxx, miny, maxy = aoi_in_src.GetEnvelope()
    print(f"AOI envelope: {minx:.6f},{miny:.6f} -> {maxx:.6f},{maxy:.6f}")

    streams.SetSpatialFilterRect(minx, miny, maxx, maxy)
    print("bbox candidates:", streams.GetFeatureCount())

    dst_srs = _srs_epsg(3577)
    src_to_dst = osr.CoordinateTransformation(src_srs, dst_srs)

    driver = ogr.GetDriverByName("GPKG")
    if OUT_GPKG.exists():
        driver.DeleteDataSource(str(OUT_GPKG))
    out_ds = driver.CreateDataSource(str(OUT_GPKG))
    out_lyr = out_ds.CreateLayer(
        "AHGFNetworkStream",
        srs=dst_srs,
        geom_type=ogr.wkbMultiLineString,
        options=["GEOMETRY_NAME=geom"],
    )

    src_defn = streams.GetLayerDefn()
    field_names: list[str] = []
    for i in range(src_defn.GetFieldCount()):
        fdefn = src_defn.GetFieldDefn(i)
        out_lyr.CreateField(fdefn)
        field_names.append(fdefn.GetName())

    streams.ResetReading()
    n_in = n_out = 0
    total_len_m = 0.0
    while True:
        feat = streams.GetNextFeature()
        if feat is None:
            break
        n_in += 1
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        clipped = _extract_lines(geom.Intersection(aoi_in_src))
        if clipped is None:
            continue
        clipped.Transform(src_to_dst)
        out_feat = ogr.Feature(out_lyr.GetLayerDefn())
        out_feat.SetGeometry(clipped)
        for name in field_names:
            idx = feat.GetFieldIndex(name)
            if idx >= 0 and feat.IsFieldSet(idx):
                out_feat.SetField(name, feat.GetField(idx))
        out_lyr.CreateFeature(out_feat)
        n_out += 1
        total_len_m += clipped.Length()

    out_ds = None
    src = None
    print(f"clipped features: {n_out} / bbox {n_in}")
    print(f"total length m (EPSG:3577): {total_len_m:.1f}")

    if OUT_GEOJSON.exists():
        OUT_GEOJSON.unlink()
    gdal.VectorTranslate(
        str(OUT_GEOJSON),
        str(OUT_GPKG),
        format="GeoJSON",
        layers=["AHGFNetworkStream"],
        dstSRS="EPSG:4326",
        reproject=True,
        layerCreationOptions=["RFC7946=YES", "WRITE_BBOX=YES"],
    )
    print("wrote", OUT_GPKG)
    print("wrote", OUT_GEOJSON)


if __name__ == "__main__":
    main()
