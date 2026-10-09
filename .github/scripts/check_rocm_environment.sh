#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "::error::Required ROCm preflight tool python3 was not found."
  exit 1
fi

exec python3 -u build_tools/ci/rocm_environment.py "$@"
