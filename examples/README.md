# Examples

| script | needs | what |
|---|---|---|
| `01_synthetic.py` | nothing | a 2048² raster written on the fly, pyramid with 2 workers, level 1 checked against the reference; under a minute |
| `02_geotiff_to_geozarr.py` | a GeoTIFF with a CRS, e.g. the public 32k window COG ([218 MB](https://me-public-assets.s3.eu-central-1.amazonaws.com/terrazarr/inputs/WSF3Dv3_Italy_win32k_cog.tif), CC BY 4.0) | GeoTIFF on disk to a sharded GeoZarr pyramid |
| `03_s3_to_s3.py` | S3 credentials and an endpoint of your own | a large sparse zarr on S3 to a pyramid on S3 with the settings measured for a 2 M × 2 M orthophoto, resumable |

Run them from the repo root with the package installed (`uv pip install -e .`).
