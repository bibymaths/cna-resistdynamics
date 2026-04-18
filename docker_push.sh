#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# Build and push the Docker image to GHCR.
# Usage: ./docker_push.sh [tag]
set -euo pipefail

IMAGE="ghcr.io/bibymaths/cna-resistdynamics"
TAG="${1:-latest}"

echo "Building ${IMAGE}:${TAG} ..."
docker build -t "${IMAGE}:${TAG}" .

echo "Pushing ${IMAGE}:${TAG} ..."
docker push "${IMAGE}:${TAG}"
echo "Done."
