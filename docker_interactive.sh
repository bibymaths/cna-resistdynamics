#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# Open an interactive shell in the development container.
set -euo pipefail
docker compose -f docker-compose.dev.yml run --rm tumorfits-dev bash
