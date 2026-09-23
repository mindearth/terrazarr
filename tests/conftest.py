import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))


@pytest.fixture(scope="session")
def dask_client():
    from dask.distributed import Client

    client = Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=":0", silence_logs=50)
    yield client
    client.close()
