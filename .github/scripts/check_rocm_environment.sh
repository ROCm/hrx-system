#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

if ! command -v rocminfo >/dev/null 2>&1; then
  echo "::error::ROCm environment on this runner is broken; rocminfo was not found."
  exit 1
fi

echo "rocminfo path: $(command -v rocminfo)"
if ! rocminfo; then
  echo "::error::ROCm environment on this runner is broken; rocminfo failed before GPU tests."
  exit 1
fi
