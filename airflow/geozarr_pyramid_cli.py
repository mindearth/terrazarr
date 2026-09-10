from __future__ import annotations

from datetime import datetime, timedelta
from shlex import quote

from airflow.decorators import dag, task
from airflow.models import Param
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule

# Set this once to the repository ECR name used by this project.
ECR_REPOSITORY = "me-geozarr"
ECR_IMAGE = f"813732138169.dkr.ecr.eu-central-1.amazonaws.com/{ECR_REPOSITORY}"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DOCKER_ENV = {
    "AWS_ACCESS_KEY_ID": "{{ var.value.MINIO_ACCESS_KEY_ID }}",
    "AWS_SECRET_ACCESS_KEY": "{{ var.value.MINIO_SECRET_ACCESS_KEY }}",
    "AWS_ENDPOINT_URL": "{{ var.value.MINIO_ENDPOINT }}",
    "AWS_REGION": "{{ var.value.AWS_REGION | default('eu-central-1') }}",
}


def q(value: object) -> str:
    return quote(str(value))


@dag(
    dag_id="geozarr-pyramid-cli",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    params={
        "tag": Param(type="string", description="Docker image tag to run (required)."),
        "input": Param(
            type="string", description="Input Zarr path (local or s3://...)."
        ),
        "output": Param(
            type="string", description="Output GeoZarr path (local or s3://...)."
        ),
        "chunk_size": Param(
            4096, type="integer", description="Spatial chunk/shard size on y/x."
        ),
        "tile_width": Param(256, type="integer", description="Zarr tile width on y/x."),
        "method": Param(
            "min",
            type="string",
            enum=["mean", "min", "max", "median", "nearest"],
            description="Resampling method.",
        ),
        "sharding": Param(
            False, type="boolean", description="Enable Zarr sharding when true."
        ),
        "nodata": Param(
            None,
            type=["null", "number"],
            description="Nodata value; null keeps source semantics.",
        ),
        "compressor": Param(
            "zstd",
            type="string",
            enum=["zstd", "lz4", "lz4hc", "blosclz", "zlib", "none"],
            description="Blosc codec for data variables.",
        ),
        "clevel": Param(3, type="integer", description="Compression level."),
        "workers": Param(8, type="integer", description="Dask worker process count."),
        "threads_per_worker": Param(
            1, type="integer", description="Threads per Dask worker."
        ),
        "memory_limit": Param(
            "auto", type="string", description="Memory limit per worker, e.g. 12GB."
        ),
    },
    tags=["geozarr"],
)
def pipeline():
    @task
    def command() -> str:
        from airflow.operators.python import get_current_context

        params = get_current_context()["params"]

        cmd = [
            "python /app/src/geozarr_pyramid/cli.py",
            f"--input {q(params['input'])}",
            f"--output {q(params['output'])}",
            f"--chunk-size {int(params['chunk_size'])}",
            f"--tile-width {int(params['tile_width'])}",
            f"--method {q(params['method'])}",
            f"--compressor {q(params['compressor'])}",
            f"--clevel {int(params['clevel'])}",
            f"--workers {int(params['workers'])}",
            f"--threads-per-worker {int(params['threads_per_worker'])}",
            f"--memory-limit {q(params['memory_limit'])}",
        ]

        if params["sharding"]:
            cmd.append("--sharding")
        if params["nodata"] is not None:
            cmd.append(f"--nodata {q(params['nodata'])}")

        return " ".join(cmd)

    DockerOperator(
        task_id="run",
        image=f"{ECR_IMAGE}:{{{{ params.tag }}}}",
        command=command(),
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        docker_conn_id="docker-ecr",
        network_mode="bridge",
        environment=DOCKER_ENV,
        mount_tmp_dir=False,
        trigger_rule=TriggerRule.NONE_FAILED,
    )


pipeline()
