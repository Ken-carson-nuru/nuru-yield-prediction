FROM ghcr.io/mlflow/mlflow:v2.14.1

USER root
RUN pip install --no-cache-dir psycopg2-binary

# (optional but recommended)
RUN adduser --disabled-password mlflowuser
USER mlflowuser
