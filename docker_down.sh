#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# Stop and remove the tumorfits compose stack.
set -euo pipefail
docker compose down "$@"
