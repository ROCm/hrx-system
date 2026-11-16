#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

# The fixture forks loom-test and then execs loom-benchmark. Instrument the
# correctness process without detaching its parent or changing either result.
export GFX120X_QUEUE_DEBUGGER="${BASH_SOURCE[0]%.sh}.py"
exec() {
  if [[ "${1:-}" == */loom-test ]]; then
    builtin exec /usr/bin/gdb -nx -batch -x "$GFX120X_QUEUE_DEBUGGER" \
      --args "$@" --execution-progress
  fi
  builtin exec "$@"
}
export -f exec

builtin exec /bin/bash "$@"
