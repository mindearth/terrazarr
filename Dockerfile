FROM 813732138169.dkr.ecr.eu-central-1.amazonaws.com/me-geospatial:py312-meutils-1.3.0-meresources-1.2.7.5

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .
