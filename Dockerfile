# Use official Airflow image
FROM apache/airflow:2.9.2-python3.10

USER root
# Install extra packages if needed
RUN apt-get update && apt-get install -y gcc g++ libpq-dev

USER airflow

# Copy Python requirements
COPY requirements.txt /requirements.txt
RUN pip install --default-timeout=200 --retries=20 --no-cache-dir -r /requirements.txt


# Copy your application code into Airflow environment
COPY src/ /opt/airflow/dags/repo/src/
COPY config/ /opt/airflow/dags/repo/config/
COPY airflow/dags/ /opt/airflow/dags/

# Include both repo root and src so imports like `from src...` and `from config...` work
ENV PYTHONPATH=/opt/airflow/dags/repo:/opt/airflow/dags/repo/src
