#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# Start the tumorfits service using docker compose.
set -euo pipefail
docker compose up "$@"
