FROM apache/airflow:2.8.1


USER root

RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

USER airflow

COPY Requirements.txt /tmp/Requirements.txt
RUN pip install -r /tmp/Requirements.txt

