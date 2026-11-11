import pickle

import numpy as np
import pytest
import xarray as xr
import zarr
from synth import make_input

from terrazarr import geozarr
from terrazarr.store import (
    IcechunkBranchStore,
    get_zarr_store,
    icechunk_read_store,
    open_icechunk_repo,
    set_spatial_info,
)

icechunk = pytest.importorskip("icechunk")

KW = dict(shard_size=64, chunk_size=64, min_dimension=64, sharding=True, method="mean", nodata=0, max_retries=1, window_shards=2)


def _run(inp, out, **kw):
    ds = xr.open_dataset(get_zarr_store(inp), engine="zarr", chunks={}, consolidated=False)
    return geozarr.to_geozarr(set_spatial_info(ds), out, **{**KW, **kw})


def _levels(store, n):
    return [xr.open_dataset(store, group=str(lv), engine="zarr", consolidated=False, decode_coords="all") for lv in range(n)]


def _assert_same_pyramid(ref_path, ic_store):
    ref_attrs = zarr.open_group(get_zarr_store(ref_path), mode="r").attrs.asdict()
    ic_attrs = zarr.open_group(ic_store, mode="r", use_consolidated=False).attrs.asdict()
    assert ic_attrs["multiscales"] == ref_attrs["multiscales"]
    n = len(ref_attrs["multiscales"]["layout"])
    for lv, (ref, got) in enumerate(zip(_levels(get_zarr_store(ref_path), n), _levels(ic_store, n))):
        np.testing.assert_array_equal(got["data"].values, ref["data"].values, err_msg=f"level {lv}")
        assert got.rio.crs == ref.rio.crs and got.rio.transform() == ref.rio.transform()
        ref_arr = zarr.open_array(get_zarr_store(ref_path), path=f"{lv}/data", mode="r")
        got_arr = zarr.open_array(ic_store, path=f"{lv}/data", mode="r")
        assert (got_arr.chunks, got_arr.shards) == (ref_arr.chunks, ref_arr.shards)


@pytest.mark.parametrize("fuse", [True, False])
@pytest.mark.parametrize("shape", [(512, 512), (2, 256, 256)])
def test_icechunk_output_equals_zarr_output(tmp_path, dask_client, monkeypatch, fuse, shape):
    monkeypatch.setattr(geozarr, "FUSE_LEVEL_1", fuse)
    inp = make_input(tmp_path / "in.zarr", shape=shape, input_chunk=128)
    _run(inp, str(tmp_path / "ref.zarr"))
    repo = open_icechunk_repo(str(tmp_path / "repo"), create=True)
    _run(inp, repo)
    _assert_same_pyramid(str(tmp_path / "ref.zarr"), icechunk_read_store(repo, "main"))
    messages = [s.message for s in repo.ancestry(branch="main")][:-1]  # the last is the repo's first
    windows = (shape[-1] // 64 // KW["window_shards"]) ** 2
    assert sum("level 0 window" in m for m in messages) == windows, "one commit per level-0 window"
    assert all(m.startswith("terrazarr: ") for m in messages)


@pytest.mark.parametrize("mixed", [False, True])
def test_icechunk_output_from_numpy_data(tmp_path, dask_client, mixed):
    """Eager variables are also written by the workers, so through a fork as well; with
    ``mixed`` one variable is dask-backed and the other numpy-backed."""
    inp = make_input(tmp_path / "in.zarr", shape=(256, 256), input_chunk=128)
    ds = set_spatial_info(xr.open_dataset(get_zarr_store(inp), engine="zarr", chunks={}, consolidated=False))
    if mixed:
        ds["extra"] = (ds["data"] // 2).load()
    else:
        ds = ds.load()
    geozarr.to_geozarr(ds, str(tmp_path / "ref.zarr"), **KW)
    repo = open_icechunk_repo(str(tmp_path / "repo"), create=True)
    geozarr.to_geozarr(ds, repo, **KW)
    _assert_same_pyramid(str(tmp_path / "ref.zarr"), icechunk_read_store(repo, "main"))
    if mixed:
        ref, got = (xr.open_dataset(s, group="1", engine="zarr", consolidated=False)["extra"].values
                    for s in (get_zarr_store(str(tmp_path / "ref.zarr")), icechunk_read_store(repo, "main")))
        np.testing.assert_array_equal(got, ref)


def test_icechunk_output_with_process_workers(tmp_path):
    """Workers in other processes: every store reaching them is pickled, every write is a fork."""
    from dask.distributed import Client

    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=128)
    repo = open_icechunk_repo(str(tmp_path / "repo"), create=True)
    with Client(processes=True, n_workers=2, threads_per_worker=1, dashboard_address=":0", silence_logs=50):
        _run(inp, str(tmp_path / "ref.zarr"))
        _run(inp, repo)
    _assert_same_pyramid(str(tmp_path / "ref.zarr"), icechunk_read_store(repo, "main"))


def test_icechunk_window_failure_resumes_from_the_last_commit(tmp_path, dask_client, monkeypatch, capsys):
    """A window that fails is not committed; the windows committed before it (row-major, the
    failing one is the sixth of 4x4) stay, with their marker, and a resumed run writes only the rest."""
    monkeypatch.setattr(geozarr, "FUSE_LEVEL_1", True)
    orig = geozarr._write_and_reduce

    def failing(block, arr, *args, block_info=None, **kwargs):
        off = kwargs.get("offset", (0, 0))
        origin = tuple(a + o for (a, _), o in zip(block_info[0]["array-location"], off)) if block_info else None
        if origin == (192, 192):
            raise RuntimeError("permanent shard failure")
        return orig(block, arr, *args, block_info=block_info, **kwargs)

    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=128)
    _run(inp, str(tmp_path / "ref.zarr"))
    repo = open_icechunk_repo(str(tmp_path / "repo"), create=True)
    monkeypatch.setattr(geozarr, "_write_and_reduce", failing)
    with pytest.raises(RuntimeError, match="Failed to write all bands"):
        _run(inp, repo, max_retries=2)
    assert "permanent shard failure" in capsys.readouterr().out
    done, complete, _ = geozarr._read_window_marker(icechunk_read_store(repo, "main"), "0")
    assert not complete and done == {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)}, "the windows before the failing one"

    window_commits = sum("level 0 window" in s.message for s in repo.ancestry(branch="main"))
    monkeypatch.setattr(geozarr, "_write_and_reduce", orig)
    _run(inp, repo, resume=True)
    assert "resuming from 5 windows" in capsys.readouterr().out
    assert sum("level 0 window" in s.message for s in repo.ancestry(branch="main")) == window_commits + 11
    _assert_same_pyramid(str(tmp_path / "ref.zarr"), icechunk_read_store(repo, "main"))


def test_icechunk_input_and_output_from_the_cli(tmp_path, monkeypatch):
    from terrazarr import cli

    inp = make_input(tmp_path / "in.zarr", shape=(512, 512), input_chunk=128)
    src_repo = open_icechunk_repo(str(tmp_path / "src"), create=True)
    session = src_repo.writable_session("main")
    xr.open_dataset(get_zarr_store(inp), engine="zarr", consolidated=False).load().to_zarr(
        session.store, mode="w", consolidated=False, zarr_format=3
    )
    session.commit("source raster")

    monkeypatch.setattr(cli, "get_dask_client", lambda **_: type("C", (), {"dashboard_link": "", "close": lambda self: None})())
    out = str(tmp_path / "out")
    cli.main([
        "--input", str(tmp_path / "src"), "--input-icechunk", "--input-ref", "main",
        "--output", out, "--output-icechunk", "--output-branch", "pyramid",
        "--chunk-size", "64", "--shard-size", "128", "--sharding", "--nodata", "0", "--window-shards", "2",
    ])
    repo = open_icechunk_repo(out)
    assert repo.list_branches() == {"main", "pyramid"}
    assert len(list(repo.ancestry(branch="main"))) == 1, "main is left as it was"
    store = icechunk_read_store(repo, "pyramid")
    src = xr.open_dataset(get_zarr_store(inp), engine="zarr", consolidated=False)["data"].values
    np.testing.assert_array_equal(xr.open_dataset(store, group="0", engine="zarr", consolidated=False)["data"].values, src)
    assert zarr.open_array(store, path="1/data", mode="r").shards == (128, 128)


def test_branch_store_pickles_as_the_last_commit(tmp_path):
    repo = open_icechunk_repo(str(tmp_path / "repo"), create=True)
    store = IcechunkBranchStore(repo, "work")
    zarr.create_array(store, name="a", shape=(4,), dtype="int8", fill_value=0)[:] = 1
    first = store.commit("a")
    remote = pickle.loads(pickle.dumps(store))
    assert remote.read_only
    assert remote.session.snapshot_id == first
    np.testing.assert_array_equal(zarr.open_array(remote, path="a", mode="r")[:], 1)
    zarr.open_array(store, path="a", mode="r+")[:] = 2  # not committed: workers still read 1
    remote = pickle.loads(pickle.dumps(store))
    np.testing.assert_array_equal(zarr.open_array(remote, path="a", mode="r")[:], 1)
    np.testing.assert_array_equal(zarr.open_array(store, path="a", mode="r")[:], 2)
    with pytest.raises(Exception, match="read.only"):
        zarr.open_array(remote, path="a", mode="r+")[:] = 3
    assert "work" in repo.list_branches()


@pytest.mark.parametrize("flags", [["--input-ref", "dev"], ["--output-branch", "dev"]])
def test_cli_icechunk_options_need_their_flag(flags):
    from terrazarr import cli

    with pytest.raises(SystemExit):
        cli.main(["--input", "in.zarr", "--output", "out.zarr", *flags])


def test_output_repo_is_written_with_split_manifests(tmp_path):
    from terrazarr.store import get_output_store

    repo = open_icechunk_repo(str(tmp_path / "repo"), create=True)
    store = get_output_store(repo, "main", window_shards=8)
    splitting = store._branch.repo.config.manifest.splitting
    assert splitting is not None, "a commit would rewrite the whole array's manifest"
    assert "16" in repr(splitting), "split every 2 * window_shards"


def test_an_unknown_ref_is_reported(tmp_path):
    repo = open_icechunk_repo(str(tmp_path / "repo"), create=True)
    with pytest.raises(ValueError, match="not a branch, a tag or a snapshot id"):
        icechunk_read_store(repo, "no-such-branch")


def test_an_icechunk_session_or_store_is_rejected(tmp_path):
    repo = open_icechunk_repo(str(tmp_path / "repo"), create=True)
    ds = xr.Dataset()
    with pytest.raises(TypeError, match="Repository"):
        geozarr.to_geozarr(ds, repo.writable_session("main"))
    with pytest.raises(TypeError, match="Repository"):
        geozarr.to_geozarr(ds, "unused", store=repo.writable_session("main").store)
