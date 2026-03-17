FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libc6-dev \
        patch \
        procps \
        tar \
        wget \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/iflow
RUN mkdir -p /opt/cache /work /root/.iflow

ENV PYTHONUNBUFFERED=1
CMD ["sleep", "infinity"]
