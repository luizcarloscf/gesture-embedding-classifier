FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/project/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists

WORKDIR /opt/project
RUN python3 -m venv .venv && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade wheel

WORKDIR /opt/project/code
COPY . .
RUN python3 -m pip install .