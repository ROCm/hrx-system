#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Regenerates the checked-in VMVX executable used by the remote HAL integration
# tests. Normal runtime builds only embed the binary and do not require the IREE
# compiler. An iree-compile build matching the runtime ABI must be on PATH when
# this fixture is updated.

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
SOURCE_FILE="${ROOT_DIR}/runtime/src/iree/hal/cts/testdata/command_buffer_dispatch_test.mlir"
OUTPUT_FILE="${ROOT_DIR}/runtime/src/iree/hal/remote/testdata/command_buffer_dispatch_test.bin"
IREE_COMPILE="${IREE_COMPILE:-iree-compile}"

"${IREE_COMPILE}" \
  --compile-mode=hal-executable \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  -o "${OUTPUT_FILE}" \
  "${SOURCE_FILE}"
