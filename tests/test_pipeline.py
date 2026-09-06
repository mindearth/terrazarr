import json
import os
import time

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.windows import from_bounds

from geozarr_pyramid import geozarr, utils
from geozarr_pyramid.store import get_zarr_store, set_spatial_info
from synth import make_input


def _run(inp, out, **kw):
    ds = xr.open_dataset(get_zarr_store(inp), engine="zarr", chunks={"y": kw["spatial_chunk"], "x": kw["spatial_chunk"]}, consolidated=False)
    ds = set_spatial_info(ds)
    params = dict(groups=["/"], output_path=out, min_dimension=kw.get("tile_width", 64), tile_width=64, max_retries=1)
    params.update(kw)
    return geozarr.create_geozarr_dataset(xr.DataTree(ds), **params)


def _open_level(out, level):
    return xr.open_dataset(get_zarr_store(out), group=str(level), engine="zarr", consolidated=False, decode_coords="all")


def test_precondition_spatial_chunk_multiple_of_tile(tmp_path):
    with pytest.raises(ValueError, match="multiple of tile_width"):
        geozarr.create_geozarr_dataset(xr.DataTree(), ["/"], str(tmp_path / "o.zarr"), spatial_chunk=300, tile_width=256)


def test_encoding_shards_bounded_and_clipped():
    ds = xr.Dataset({"v": (("t", "y", "x"), np.zeros((12, 100, 3000), dtype="uint8"), {"grid_mapping": "spatial_ref"})})
    ds = ds.assign_coords(spatial_ref=xr.DataArray(0))
    enc = geozarr._create_geozarr_encoding(ds, None, tile_width=256, spatial_chunk=1024, enable_sharding=True)
    assert enc["v"]["chunks"] == (1, 100, 256)
    assert enc["v"]["shards"] == (1, 100, 1024), "one slice per shard, shard clipped to a chunk multiple"
    assert enc["spatial_ref"] == {"compressors": None}


def test_pipeline_2d_levels_and_georeferencing(tmp_path, dask_client):
    # 1000 is not a power of two: exercises trimming + constant pixel size
    inp = make_input(tmp_path / "in.zarr", shape=(1000, 1000), input_chunk=250)
    out = str(tmp_path / "out.zarr")
    _run(inp, out, spatial_chunk=256, tile_width=64, enable_sharding=True, method="mean", nodata_value=0)

    root = zarr.open_group(get_zarr_store(out), mode="r")
    layout = root.attrs["multiscales"]["layout"]
    assert [e["asset"] for e in layout] == ["0", "1", "2", "3"]
    for e in layout:
        lv = int(e["asset"])
        px = e["spatial:transform"][0]
        assert px == pytest.approx(10.0 * 2**lv), "level pixel size is exactly 2**L native pixels"
        ds = _open_level(out, lv)
        assert ds.rio.transform().a == pytest.approx(px)
        assert ds.rio.transform().e == pytest.approx(-px)
        assert ds.sizes["y"] == 1000 // 2**lv and ds.sizes["x"] == 1000 // 2**lv
        from_bounds(*ds.rio.bounds(recalc=True), transform=ds.rio.transform())  # rasterio-consistent
        assert ds.rio.crs is not None and ds.rio.crs.to_epsg() == 3857, "grid_mapping must survive the write"
        assert ds["data"].attrs.get("grid_mapping") == "spatial_ref" or ds["data"].encoding.get("grid_mapping") == "spatial_ref"
        # level origin is the native origin (top-left anchored)
        assert ds.rio.transform().c == pytest.approx(1_000_000.0)
        assert ds.rio.transform().f == pytest.approx(5_000_000.0)

    # level 1 values: rounded mean of 2x2 blocks with nodata handling
    l0 = _open_level(out, 0)["data"].values
    l1 = _open_level(out, 1)["data"].values
    b = l0.reshape(500, 2, 500, 2).astype("float64")
    valid = b != 0
    cnt = valid.sum(axis=(1, 3))
    ref = np.where(cnt >= 1.2, np.rint(np.where(valid, b, 0).sum(axis=(1, 3)) / np.maximum(cnt, 1)), 0).astype("uint8")
    np.testing.assert_array_equal(l1, ref)


def test_pipeline_3d_sharded(tmp_path, dask_client):
    inp = make_input(tmp_path / "in3.zarr", shape=(3, 512, 512), input_chunk=256)
    out = str(tmp_path / "out3.zarr")
    _run(inp, out, spatial_chunk=256, tile_width=64, enable_sharding=True, method="max", nodata_value=0)
    arr = zarr.open_array(get_zarr_store(out), path="0/data", mode="r")
    assert arr.shards == (1, 256, 256) and arr.chunks == (1, 64, 64)
    l1 = _open_level(out, 1)["data"]
    assert l1.dims == ("t", "y", "x") and l1.shape == (3, 256, 256)
    l0 = _open_level(out, 0)["data"].values
    np.testing.assert_array_equal(l1.values, l0.reshape(3, 256, 2, 256, 2).max(axis=(2, 4)))


def test_resume_skips_written_bands_and_levels(tmp_path, dask_client):
    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=256, nan_corner=True)
    out = str(tmp_path / "out.zarr")
    _run(inp, out, spatial_chunk=256, tile_width=64, enable_sharding=True, method="mean")
    chunk_files = sorted(
        os.path.join(r, f) for r, _, fs in os.walk(os.path.join(out, "0", "data")) for f in fs if r.endswith("/c") or "/c/" in r
    )
    assert chunk_files
    before = {f: os.stat(f).st_mtime_ns for f in chunk_files}
    time.sleep(0.05)
    _run(inp, out, spatial_chunk=256, tile_width=64, enable_sharding=True, method="mean")
    after = {f: os.stat(f).st_mtime_ns for f in chunk_files}
    assert before == after, "a complete level 0 (NaN in the top-left pixel) must not be rewritten"


def test_validate_existing_band_accepts_nan_and_rejects_dtype_mismatch(tmp_path, dask_client):
    inp = make_input(tmp_path / "in.zarr", shape=(256, 256), input_chunk=128, nan_corner=True)
    ds = set_spatial_info(xr.open_dataset(get_zarr_store(inp), engine="zarr", chunks={}, consolidated=False))
    ds["data"].attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
    assert utils.validate_existing_band_data(ds, "data", ds)
    other = ds.copy()
    other["data"] = other["data"].astype("uint8")
    assert not utils.validate_existing_band_data(ds, "data", other)


def _reference_levels(l0, nlevels, nodata, dtype):
    levels = [l0]
    for _ in range(1, nlevels):
        prev = levels[-1]
        levels.append(utils.reduce_block(prev, 2, 2, "mean", nodata_value=nodata, out_dtype=dtype))
    return levels


@pytest.mark.parametrize("fuse", [True, False])
def test_fused_level1_store_levels_and_resume(tmp_path, dask_client, monkeypatch, fuse):
    """Level 1 fused with the level-0 write (or not), level 2 from the store one shard per
    task (16 shards), level 3 from memory (4 shards); all equal to the iterated kernel."""
    monkeypatch.setattr(geozarr, "FUSE_LEVEL_1", fuse)
    inp = make_input(tmp_path / "in.zarr", shape=(1000, 1000), input_chunk=250)
    out = str(tmp_path / "out.zarr")
    _run(inp, out, spatial_chunk=64, tile_width=64, enable_sharding=True, method="mean", nodata_value=0)
    root = zarr.open_group(get_zarr_store(out), mode="r")
    nlevels = len(root.attrs["multiscales"]["layout"])
    assert nlevels == 4
    l0 = _open_level(out, 0)["data"].values
    for lv, ref in enumerate(_reference_levels(l0, nlevels, 0, "uint8")):
        got = _open_level(out, lv)
        np.testing.assert_array_equal(got["data"].values, ref)
        assert got.rio.crs is not None
        assert zarr.open_array(get_zarr_store(out), path=f"{lv}/data", mode="r").shards == (min(64, ref.shape[0]), min(64, ref.shape[1]))

    # resume: nothing is rewritten
    files = sorted(os.path.join(r, f) for lv in range(nlevels) for r, _, fs in os.walk(os.path.join(out, str(lv), "data", "c")) for f in fs)
    assert len(files) > 20
    before = {f: os.stat(f).st_mtime_ns for f in files}
    time.sleep(0.05)
    _run(inp, out, spatial_chunk=64, tile_width=64, enable_sharding=True, method="mean", nodata_value=0)
    assert {f: os.stat(f).st_mtime_ns for f in files} == before


def test_pipeline_float_nan_nodata_and_compressor_none(tmp_path, dask_client):
    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=256, dtype="float32", nodata=None)
    out = str(tmp_path / "out.zarr")
    from geozarr_pyramid.geozarr import make_compressor
    _run(inp, out, spatial_chunk=64, tile_width=64, enable_sharding=True, method="mean", compressor=make_compressor("none"))
    arr = zarr.open_array(get_zarr_store(out), path="1/data", mode="r")
    assert arr.compressors == ()
    l0 = _open_level(out, 0)["data"].values
    l1 = _open_level(out, 1)["data"].values
    np.testing.assert_allclose(l1, utils.reduce_block(l0, 2, 2, "mean", out_dtype="float32"), rtol=1e-6)
    assert np.isnan(l1[-1, -1]), "NaN block stays NaN"
