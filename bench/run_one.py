"""
Run one implementation on an input store and print JSON metrics: ``optimized`` is this package,
``baseline`` the upstream eopf-geozarr as published (run it with ``.venv-baseline/bin/python``).

The input is a zarr store (local or s3://) or a GeoTIFF (striped or COG, local or s3://,
read through rasterio; store counters then cover the output only).

Runs in-process (threaded dask workers) so store traffic and peak RSS of the
whole pipeline are captured by one process. Launch one process per run.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def count_objects(path: str) -> int:
    """Files under a local path, or objects under an s3:// prefix."""
    if path.startswith("s3://"):
        import obstore

        from terrazarr.store import get_obstore

        return sum(1 for _ in obstore.list(get_obstore(path)).collect())
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
    return n


def remove_output(path: str) -> None:
    if path.startswith("s3://"):
        import obstore

        from terrazarr.store import get_obstore

        st = get_obstore(path)
        keys = [o["path"] for o in obstore.list(st).collect()]
        if keys:
            obstore.delete(st, keys)
        return
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def gdal_env() -> None:
    """GDAL settings for reading GeoTIFF inputs, local or on MinIO through /vsis3/.

    Set before the dask client starts so worker processes inherit them.
    """
    ep = os.environ.get("AWS_ENDPOINT_URL")
    if ep:
        os.environ.setdefault("AWS_S3_ENDPOINT", ep.split("://", 1)[-1])
        os.environ.setdefault("AWS_HTTPS", "YES" if ep.startswith("https") else "NO")
        os.environ.setdefault("AWS_VIRTUAL_HOSTING", "FALSE")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
    os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", str(16 * 1024 * 1024))
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", str(512 * 1024 * 1024))


def open_geotiff(path: str, shard_size: int):
    """A GeoTIFF (striped or COG, local or s3://) as the same dataset the zarr input gives:
    one variable ``data`` on (y, x) in ``shard_size`` dask blocks, CRS on ``spatial_ref``.

    ``lock=False`` gives every dask thread its own GDAL handle, so tile decodes run in
    parallel; GDAL's block cache (``GDAL_CACHEMAX``, MB) is what makes a striped file
    bearable, since every block of a block-row needs the same full-width strips.
    """
    import rioxarray

    if path.startswith("s3://"):
        path = "/vsis3/" + path[len("s3://"):]
    da_ = rioxarray.open_rasterio(
        path, chunks={"band": 1, "y": shard_size, "x": shard_size}, lock=False, masked=False,
    )
    da_ = da_.squeeze("band", drop=True)
    ds = da_.to_dataset(name="data")
    ds["data"].attrs.pop("_FillValue", None)
    ds["data"].encoding.pop("_FillValue", None)
    return ds


def provenance(impl: str) -> dict:
    """Git commit, package versions and machine, so a results file says what produced it."""
    import platform
    import subprocess
    from importlib.metadata import PackageNotFoundError, version

    def v(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    try:
        commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    return {
        "commit": commit,
        "versions": {k: v(k) for k in ("terrazarr" if impl == "optimized" else "eopf-geozarr", "zarr", "dask", "xarray", "numpy", "obstore")},
        "python": platform.python_version(),
        "machine": {"cpus": os.cpu_count(), "platform": platform.platform()},
    }


def instrument_stores(counters: Counter) -> None:
    """Count chunk/metadata traffic on every zarr store class either implementation can use."""
    import zarr

    def is_chunk(key: str) -> bool:
        return "/c/" in key or key.startswith("c/")

    classes = [zarr.storage.LocalStore, zarr.storage.ObjectStore]
    try:
        classes.append(zarr.storage.FsspecStore)
    except AttributeError:
        pass
    for cls in classes:
        _get, _set = cls.get, cls.set

        async def get(self, key, prototype, byte_range=None, _get=_get):
            counters["get_chunk" if is_chunk(key) else "get_meta"] += 1
            return await _get(self, key, prototype, byte_range)

        async def set_(self, key, value, _set=_set):
            counters["set_chunk" if is_chunk(key) else "set_meta"] += 1
            return await _set(self, key, value)

        cls.get, cls.set = get, set_
        if hasattr(cls, "delete"):
            _delete = cls.delete

            async def delete(self, key, _delete=_delete):
                counters["delete"] += 1
                return await _delete(self, key)

            cls.delete = delete


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["baseline", "optimized"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--method", default="mean")
    p.add_argument("--nodata", type=float, default=None)
    p.add_argument("--sharding", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--compressor", default="zstd", help="optimized only: blosc codec name or none")
    p.add_argument("--clevel", type=int, default=3, help="optimized only: blosc level")
    p.add_argument("--workers", type=int, default=1, help="dask worker processes (>1: threads are split across them; store counters then cover the main process only)")
    args = p.parse_args()

    import rioxarray  # noqa: F401
    import xarray as xr
    import zarr
    from dask.distributed import Client, get_task_stream

    if args.impl == "baseline":
        # the upstream this project was forked from, as published (.venv-baseline, eopf-geozarr 0.7.1);
        # it has no --method/--nodata (overviews are plain means) and wants the data under a child group
        from eopf_geozarr.conversion.geozarr import create_geozarr_dataset as eopf_create

        if args.method != "mean" or args.nodata is not None:
            print(f"baseline: eopf-geozarr ignores --method {args.method} and --nodata {args.nodata}", file=sys.stderr)

        def create_geozarr_dataset(dt, groups, output_path, shard_size, min_dimension, chunk_size, max_retries,
                                   enable_sharding, method, nodata_value, **_):
            dt = xr.DataTree.from_dict({"/measurements": dt.to_dataset()})
            return eopf_create(dt, ["/measurements"], output_path, spatial_chunk=shard_size, min_dimension=min_dimension,
                               tile_width=chunk_size, max_retries=max_retries, enable_sharding=enable_sharding)

        def get_zarr_store(path, _profile=None):
            return zarr.storage.LocalStore(path) if not path.startswith("s3://") else path

        def set_spatial_info(ds):
            ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
            return ds if ds.rio.crs else ds.rio.write_crs("EPSG:4326")
    else:
        from terrazarr.geozarr import create_geozarr_dataset, make_compressor
        from terrazarr.store import get_zarr_store, set_spatial_info

    counters: Counter = Counter()
    instrument_stores(counters)
    gdal_env()

    if args.workers > 1:
        client = Client(processes=True, n_workers=args.workers, threads_per_worker=max(1, args.threads // args.workers), dashboard_address=":0", silence_logs=50)
    else:
        client = Client(processes=False, n_workers=1, threads_per_worker=args.threads, dashboard_address=":0", silence_logs=50)

    if args.quiet:
        sys.stdout = open(os.devnull, "w")
    remove_output(args.output)

    if args.input.lower().endswith((".tif", ".tiff")):
        ds = open_geotiff(args.input, args.shard_size)
    else:
        ds = xr.open_dataset(
            get_zarr_store(args.input), engine="zarr",
            chunks={"y": args.shard_size, "x": args.shard_size}, consolidated=False,
        )
    ds = set_spatial_info(ds)
    dt = xr.DataTree(ds)

    t0 = time.perf_counter()
    err = None
    task_stream = get_task_stream(client=client)
    task_stream.__enter__()
    extra = {}
    if args.impl == "optimized":
        extra["compressor"] = make_compressor(args.compressor, args.clevel)
    try:
        create_geozarr_dataset(
            dt, groups=["/"], output_path=args.output,
            shard_size=args.shard_size, min_dimension=args.chunk_size, chunk_size=args.chunk_size,
            max_retries=1, enable_sharding=args.sharding, method=args.method, nodata_value=args.nodata,
            **extra,
        )
    except Exception as e:  # report, don't hide
        err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    task_stream.__exit__(None, None, None)
    tasks_executed = len(task_stream.data)
    client.close()

    sys.stdout = sys.__stdout__
    print(json.dumps({
        "provenance": provenance(args.impl),
        "impl": args.impl,
        "ok": err is None,
        "error": err,
        "wall_s": round(wall, 2),
        "tasks_executed": tasks_executed,
        "store_get_chunk": counters["get_chunk"],
        "store_set_chunk": counters["set_chunk"],
        "store_get_meta": counters["get_meta"],
        "store_set_meta": counters["set_meta"],
        "store_delete": counters["delete"],
        "objects": count_objects(args.output),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }))


if __name__ == "__main__":
    main()
