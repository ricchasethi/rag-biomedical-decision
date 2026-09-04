FROM apache/airflow:3.3.1

# psycopg2 and some ML wheels need a compiler + libpq headers at install time.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Never pip install as root in the Airflow image.
USER airflow

COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
