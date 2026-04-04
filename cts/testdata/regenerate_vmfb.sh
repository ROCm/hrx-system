#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}/../../../../"
IREE_COMPILE="${PYRE_IREE_COMPILER_CLI:-${WORKSPACE_ROOT}/build/iree-full/tools/iree-compile}"

"${IREE_COMPILE}" "${SCRIPT_DIR}/add_i64.mlir" -o "${SCRIPT_DIR}/add_i64.vmfb"
