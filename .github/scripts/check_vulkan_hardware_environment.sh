#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

for required_tool in python3 timeout vulkaninfo; do
  if ! command -v "${required_tool}" >/dev/null 2>&1; then
    printf 'Required Vulkan preflight tool %s was not found.\n' "${required_tool}" >&2
    exit 1
  fi
done

readonly probe_timeout_seconds=30
readonly probe_kill_after_seconds=5
readonly runner_name="${RUNNER_NAME:-unknown}"

printf 'Checking Vulkan hardware on runner %s (timeout: %ss).\n' \
  "${runner_name}" "${probe_timeout_seconds}"

probe_status=0
timeout --kill-after="${probe_kill_after_seconds}s" \
  "${probe_timeout_seconds}s" python3 -u build_tools/ci/vulkan_environment.py "$@" \
  || probe_status=$?

case "${probe_status}" in
  0)
    ;;
  124)
    printf 'Vulkan hardware probe exceeded %ss on runner %s; the last phase '\
'above identifies whether device access or the native Vulkan query wedged.\n' \
      "${probe_timeout_seconds}" "${runner_name}" >&2
    exit 124
    ;;
  137)
    printf 'Vulkan hardware probe on runner %s required SIGKILL after '\
'exceeding %ss.\n' "${runner_name}" "${probe_timeout_seconds}" >&2
    exit 137
    ;;
  *)
    exit "${probe_status}"
    ;;
esac
