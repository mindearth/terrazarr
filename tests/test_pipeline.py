import os
import time

import numpy as np
import pytest
import xarray as xr
import zarr
from rasterio.windows import from_bounds
from synth import make_input

from terrazarr import geozarr, utils
from terrazarr.store import get_zarr_store, set_spatial_info


def _run(inp, out, **kw):
    ds = xr.open_dataset(get_zarr_store(inp), engine="zarr", chunks={"y": kw["shard_size"], "x": kw["shard_size"]}, consolidated=False)
    ds = set_spatial_info(ds)
    params = dict(groups=["/"], output_path=out, min_dimension=kw.get("chunk_size", 64), chunk_size=64, max_retries=1)
    params.update(kw)
    return geozarr.create_geozarr_dataset(xr.DataTree(ds), **params)


def _open_level(out, level):
    return xr.open_dataset(get_zarr_store(out), group=str(level), engine="zarr", consolidated=False, decode_coords="all")


def test_precondition_shard_size_multiple_of_chunk_size(tmp_path):
    with pytest.raises(ValueError, match="multiple of chunk_size"):
        geozarr.create_geozarr_dataset(xr.DataTree(), ["/"], str(tmp_path / "o.zarr"), shard_size=300, chunk_size=256)


def test_encoding_shards_bounded_and_clipped():
    ds = xr.Dataset({"v": (("t", "y", "x"), np.zeros((12, 100, 3000), dtype="uint8"), {"grid_mapping": "spatial_ref"})})
    ds = ds.assign_coords(spatial_ref=xr.DataArray(0))
    enc = geozarr._create_geozarr_encoding(ds, None, chunk_size=256, shard_size=1024, enable_sharding=True)
    assert enc["v"]["chunks"] == (1, 100, 256)
    assert enc["v"]["shards"] == (1, 100, 1024), "one slice per shard, shard clipped to a chunk multiple"
    assert enc["spatial_ref"] == {"compressors": None}


def test_pipeline_2d_levels_and_georeferencing(tmp_path, dask_client):
    # 1000 is not a power of two: exercises trimming + constant pixel size
    inp = make_input(tmp_path / "in.zarr", shape=(1000, 1000), input_chunk=250)
    out = str(tmp_path / "out.zarr")
    _run(inp, out, shard_size=256, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0)

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
    _run(inp, out, shard_size=256, chunk_size=64, enable_sharding=True, method="max", nodata_value=0)
    arr = zarr.open_array(get_zarr_store(out), path="0/data", mode="r")
    assert arr.shards == (1, 256, 256) and arr.chunks == (1, 64, 64)
    l1 = _open_level(out, 1)["data"]
    assert l1.dims == ("t", "y", "x") and l1.shape == (3, 256, 256)
    l0 = _open_level(out, 0)["data"].values
    np.testing.assert_array_equal(l1.values, l0.reshape(3, 256, 2, 256, 2).max(axis=(2, 4)))


def test_resume_skips_written_bands_and_levels(tmp_path, dask_client):
    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=256, nan_corner=True)
    out = str(tmp_path / "out.zarr")
    _run(inp, out, shard_size=256, chunk_size=64, enable_sharding=True, method="mean")
    chunk_files = sorted(
        os.path.join(r, f) for r, _, fs in os.walk(os.path.join(out, "0", "data")) for f in fs if r.endswith("/c") or "/c/" in r
    )
    assert chunk_files
    before = {f: os.stat(f).st_mtime_ns for f in chunk_files}
    time.sleep(0.05)
    _run(inp, out, shard_size=256, chunk_size=64, enable_sharding=True, method="mean")
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
    _run(inp, out, shard_size=64, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0)
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
    _run(inp, out, shard_size=64, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0)
    assert {f: os.stat(f).st_mtime_ns for f in files} == before


def test_pipeline_float_nan_nodata_and_compressor_none(tmp_path, dask_client):
    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=256, dtype="float32", nodata=None)
    out = str(tmp_path / "out.zarr")
    from terrazarr.geozarr import make_compressor
    _run(inp, out, shard_size=64, chunk_size=64, enable_sharding=True, method="mean", compressor=make_compressor("none"))
    arr = zarr.open_array(get_zarr_store(out), path="1/data", mode="r")
    assert arr.compressors == ()
    l0 = _open_level(out, 0)["data"].values
    l1 = _open_level(out, 1)["data"].values
    np.testing.assert_allclose(l1, utils.reduce_block(l0, 2, 2, "mean", out_dtype="float32"), rtol=1e-6)
    assert np.isnan(l1[-1, -1]), "NaN block stays NaN"


def test_window_failure_is_retried_then_resumed(tmp_path, dask_client, monkeypatch, capsys):
    """A write-and-reduce task that fails once is retried within its window; one that fails for
    good fails the level cleanly, and the next run resumes from the windows already done
    without rewriting them. Judged by values: a missing shard reads as the fill value."""
    monkeypatch.setattr(geozarr, "FUSE_LEVEL_1", True)
    orig = geozarr._write_and_reduce
    flag = tmp_path / "failed_once"   # dask pickles the task function, so the state lives in a file

    def flaky(block, arr, *args, block_info=None, **kwargs):
        # dask passes block_info only to functions that name it; block_info is relative to
        # the window, the pipeline's offset kwarg makes it global
        off = kwargs.get("offset", (0, 0))
        origin = tuple(a + o for (a, _), o in zip(block_info[0]["array-location"], off)) if block_info else None
        if origin == (64, 64) and not flag.exists():
            flag.touch()
            raise RuntimeError("injected shard failure")
        if origin == (192, 192):
            raise RuntimeError("permanent shard failure")
        return orig(block, arr, *args, block_info=block_info, **kwargs)

    monkeypatch.setattr(geozarr, "_write_and_reduce", flaky)
    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=128)
    src = xr.open_dataset(get_zarr_store(inp), engine="zarr", consolidated=False)["data"].values
    out = str(tmp_path / "out.zarr")
    with pytest.raises(RuntimeError, match="Failed to write all bands"):
        _run(inp, out, shard_size=64, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0,
             max_retries=2, window_shards=2)
    log = capsys.readouterr().out
    assert "injected shard failure" in log and "retrying" in log, "the once-failing window was not retried"
    assert "permanent shard failure" in log
    store = get_zarr_store(out)
    done, complete, size = geozarr._read_window_marker(store, "0")
    assert not complete and size == 2 and 0 < len(done) < 16, (done, complete)
    # shards of the windows marked done (every window but (1, 1), i.e. blocks 2-3 × 2-3)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(os.path.join(out, "0", "data", "c")) for f in fs
                   if f.isdigit() and not (int(os.path.basename(r)) >= 2 and int(f) >= 2))
    assert files
    before = {f: os.stat(f).st_mtime_ns for f in files}
    time.sleep(0.05)

    monkeypatch.setattr(geozarr, "_write_and_reduce", orig)
    _run(inp, out, shard_size=64, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0,
         max_retries=1, window_shards=2)
    assert "resuming from" in capsys.readouterr().out
    assert geozarr._read_window_marker(store, "0")[1], "completion marker missing after the resume"
    assert {f: os.stat(f).st_mtime_ns for f in files} == before, "windows already done were rewritten"
    nlevels = len(zarr.open_group(store, mode="r").attrs["multiscales"]["layout"])
    l0 = _open_level(out, 0)["data"].values
    np.testing.assert_array_equal(l0, src)
    for lv, ref in enumerate(_reference_levels(l0, nlevels, 0, "uint8")):
        np.testing.assert_array_equal(_open_level(out, lv)["data"].values, ref)


@pytest.mark.parametrize("window_shards", [2, 4, 64])
def test_windowed_write_equals_reference(tmp_path, dask_client, window_shards):
    """Level 0 and the fused level 1 written in windows of 2, 4 and (one window) 64 blocks give
    the same pyramid; a 1000² raster at chunk 64 is 16 × 16 blocks, so windows are partial at
    the edge and level-1 regions end on the trimmed level-1 shape."""
    inp = make_input(tmp_path / "in.zarr", shape=(1000, 1000), input_chunk=250)
    out = str(tmp_path / "out.zarr")
    _run(inp, out, shard_size=64, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0,
         window_shards=window_shards)
    store = get_zarr_store(out)
    nlevels = len(zarr.open_group(store, mode="r").attrs["multiscales"]["layout"])
    l0 = _open_level(out, 0)["data"].values
    np.testing.assert_array_equal(l0, xr.open_dataset(get_zarr_store(inp), engine="zarr", consolidated=False)["data"].values)
    for lv, ref in enumerate(_reference_levels(l0, nlevels, 0, "uint8")):
        np.testing.assert_array_equal(_open_level(out, lv)["data"].values, ref)
    done, complete, size = geozarr._read_window_marker(store, "0")
    assert complete and size == window_shards and len(done) == (16 // window_shards + (16 % window_shards > 0)) ** 2


def test_all_fill_blocks_skip_the_write(tmp_path, dask_client):
    """A level-0 block equal to the array fill value is not written (zarr would store nothing
    for it, but only after walking every inner chunk); a block of nodata that is not the fill
    value is stored as before."""
    # uint8, fill 0 == nodata 0: an all-zero 128² block leaves no shard
    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=128)
    zarr.open_array(get_zarr_store(inp), path="data", mode="r+")[:128, :128] = 0
    out = str(tmp_path / "out.zarr")
    _run(inp, out, shard_size=128, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0)
    arr = zarr.open_array(get_zarr_store(out), path="0/data", mode="r")
    src = xr.open_dataset(get_zarr_store(inp), engine="zarr", consolidated=False)["data"].values
    assert not src[:128, :128].any() and arr.fill_value == 0
    assert not os.path.exists(os.path.join(out, "0", "data", "c", "0", "0")), "the all-fill shard was written"
    assert os.path.exists(os.path.join(out, "0", "data", "c", "0", "1"))
    np.testing.assert_array_equal(arr[:], src)
    # float32 with NaN fill: zeros are data and every block is stored
    inp2 = make_input(tmp_path / "in2.zarr", shape=(512, 512), input_chunk=128, dtype="float32", nodata=None)
    zarr.open_array(get_zarr_store(inp2), path="data", mode="r+")[:128, :128] = 0.0
    src2 = xr.open_dataset(get_zarr_store(inp2), engine="zarr", consolidated=False)["data"].values
    out2 = str(tmp_path / "out2.zarr")
    _run(inp2, out2, shard_size=128, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0)
    arr2 = zarr.open_array(get_zarr_store(out2), path="0/data", mode="r")
    assert np.isnan(arr2.fill_value)
    assert os.path.exists(os.path.join(out2, "0", "data", "c", "0", "0")), "zeros are data when the fill value is NaN"
    np.testing.assert_array_equal(arr2[:], src2)


def test_band_last_source_is_one_task_per_block(tmp_path, dask_client):
    """A (y, x, band) source whose block holds every band keeps the bands together in the dask
    block (one read, one transpose, one task writing all band shards) and gives the same
    pyramid as the per-band reference."""
    H = W = 512
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, (H, W, 3), dtype="uint8")
    data[:, :200, :] = 0
    ds = xr.Dataset({"data": (("y", "x", "band"), data)},
                    coords={"y": 5e6 - np.arange(H) * 10.0, "x": 1e6 + np.arange(W) * 10.0, "band": np.arange(3)})
    ds = ds.rio.write_crs("EPSG:3857")
    inp = str(tmp_path / "bl.zarr")
    ds.to_zarr(inp, mode="w", zarr_format=3, consolidated=False, encoding={"data": {"chunks": (128, 128, 3)}})
    enc = {"chunks": (1, 64, 64), "shards": (1, 128, 128)}
    src = xr.open_dataset(get_zarr_store(inp), engine="zarr", chunks={"y": 128, "x": 128}, consolidated=False)["data"].transpose("band", "y", "x")
    assert geozarr._dask_chunks_for(enc, src.dims, 128, src.data.chunks) == {"band": 3, "y": 128, "x": 128}
    out = str(tmp_path / "out.zarr")
    _run(inp, out, shard_size=128, chunk_size=64, enable_sharding=True, method="mean", nodata_value=0)
    arr = zarr.open_array(get_zarr_store(out), path="0/data", mode="r")
    assert arr.shape == (3, H, W) and arr.shards == (1, 128, 128)
    ref0 = np.moveaxis(data, -1, 0)
    np.testing.assert_array_equal(arr[:], ref0)
    l1 = _open_level(out, 1)["data"].values
    np.testing.assert_array_equal(l1, utils.reduce_block(ref0, 2, 2, "mean", nodata_value=0, out_dtype="uint8"))
