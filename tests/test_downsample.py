import dask.array as da
import numpy as np
import pytest

from geozarr_pyramid import utils


def _ref_mean(a, nodata):
    h, w = a.shape[0] // 2, a.shape[1] // 2
    b = a[: 2 * h, : 2 * w].reshape(h, 2, w, 2).astype("float64")
    valid = ~np.isnan(b)
    if nodata is not None:
        valid &= b != nodata
    cnt = valid.sum(axis=(1, 3))
    tot = np.where(valid, b, 0).sum(axis=(1, 3))
    out = tot / np.maximum(cnt, 1)
    return np.where(cnt >= 0.3 * 4, out, nodata if nodata is not None else np.nan)


def test_mean_matches_reference_with_nodata():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 66), dtype="uint8")
    a[::3, ::5] = 0
    out = utils.downsample_2d_array(da.from_array(a, chunks=(32, 33)), 32, 33, nodata_value=0, method="mean").compute()
    np.testing.assert_allclose(out, _ref_mean(a, 0))


@pytest.mark.parametrize("method", ["min", "max"])
def test_min_max_stay_in_source_dtype(method):
    a = np.arange(1, 65, dtype="uint8").reshape(8, 8)
    a[0, 0] = 0
    out = utils.downsample_2d_array(da.from_array(a, chunks=(4, 4)), 4, 4, nodata_value=0, method=method)
    assert out.dtype == np.uint8, "no float64 promotion for integer min/max"
    ref = a.reshape(4, 2, 4, 2)
    ref = np.where(ref == 0, 255 if method == "min" else 0, ref)
    ref = ref.min(axis=(1, 3)) if method == "min" else ref.max(axis=(1, 3))
    np.testing.assert_array_equal(out.compute(), ref)


def test_median_uses_float32_for_small_integers():
    a = np.arange(64, dtype="uint8").reshape(8, 8)
    out = utils.downsample_2d_array(da.from_array(a, chunks=(8, 8)), 4, 4, method="median")
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.compute(), np.median(a.reshape(4, 2, 4, 2), axis=(1, 3)))


def test_float_nan_is_nodata_and_valid_fraction_rule():
    a = np.ones((4, 4), dtype="float32")
    a[0, :2] = np.nan            # block (0,0): 2 of 4 valid -> keep (50% >= 30%)
    a[2:, :2] = np.nan           # block (1,0): 0 of 4 valid -> nodata
    out = utils.downsample_2d_array(da.from_array(a, chunks=(4, 4)), 2, 2, method="mean").compute()
    assert out[0, 0] == 1.0 and np.isnan(out[1, 0]) and out[0, 1] == 1.0


def test_no_rechunk_for_even_chunks():
    a = da.zeros((4096, 4096), dtype="uint8", chunks=(1024, 1024))
    out = utils.downsample_2d_array(a, 2048, 2048, method="mean", nodata_value=0)
    assert out.chunks == ((512,) * 4, (512,) * 4)
    assert not any("rechunk" in name for name in out.dask.layers)
