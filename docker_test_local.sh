#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# Build the Docker image locally and run the CLI help command to verify it works.
set -euo pipefail

IMAGE="tumorfits-local-test"

echo "Building ${IMAGE} ..."
docker build -t "${IMAGE}" .

echo "Running CLI smoke test ..."
docker run --rm "${IMAGE}" --help

echo "Running extract-data help ..."
docker run --rm "${IMAGE}" extract-data --help

echo "All Docker smoke tests passed."
