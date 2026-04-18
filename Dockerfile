# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT

# Multi-stage build: base image with system dependencies + app layer
FROM condaforge/miniforge3:latest AS base

LABEL maintainer="Abhinav Mishra <mishraabhinav@gmail.com>"
LABEL org.opencontainers.image.title="CNA-ResistDynamics"
LABEL org.opencontainers.image.description="ODE/PDE modelling of chemotherapy resistance from liquidCNA"
LABEL org.opencontainers.image.source="https://github.com/bibymaths/cna-resistdynamics"
LABEL org.opencontainers.image.licenses="MIT"

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Create conda environment ---
WORKDIR /opt/tumorfits
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean --all -y

# Activate the environment for all subsequent RUN/CMD/ENTRYPOINT
SHELL ["conda", "run", "-n", "cna-resist-dynamics", "/bin/bash", "-c"]

# --- Install the package ---
COPY pyproject.toml .
COPY tumorfits/ tumorfits/
RUN pip install -e . --no-deps

# --- Runtime stage ---
FROM base AS runtime

# Mount point for user data — do NOT COPY data into the image
VOLUME ["/data", "/results", "/config"]

WORKDIR /workspace

# Default: show CLI help
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cna-resist-dynamics", "tumorfits"]
CMD ["--help"]
