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

if ! command -v timeout >/dev/null 2>&1; then
  echo "::error::Required process supervisor timeout was not found."
  exit 1
fi

readonly probe_timeout_seconds=30
readonly probe_kill_after_seconds=5
readonly runner_name="${RUNNER_NAME:-unknown}"

echo "rocminfo path: $(command -v rocminfo)"
printf 'Checking ROCm hardware on runner %s (timeout: %ss).\n' \
  "${runner_name}" "${probe_timeout_seconds}"

probe_status=0
timeout --kill-after="${probe_kill_after_seconds}s" \
  "${probe_timeout_seconds}s" rocminfo || probe_status=$?

case "${probe_status}" in
  0)
    ;;
  124)
    printf '::error::ROCm hardware probe exceeded %ss on runner %s; '\
'rocminfo or the HSA driver is wedged.\n' \
      "${probe_timeout_seconds}" "${runner_name}"
    exit 124
    ;;
  137)
    printf '::error::ROCm hardware probe on runner %s required SIGKILL '\
'after exceeding %ss.\n' "${runner_name}" "${probe_timeout_seconds}"
    exit 137
    ;;
  *)
    printf '::error::ROCm hardware probe failed with exit code %s on '\
'runner %s before GPU tests.\n' "${probe_status}" "${runner_name}"
    exit "${probe_status}"
    ;;
esac
