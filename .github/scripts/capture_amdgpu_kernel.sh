#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
export TZ=UTC

printf 'GPU preflight started at %s UTC; captured at %s UTC\n' "$1" "$(date -u '+%F %T')"
uname -a
printf 'boot_id='
cat /proc/sys/kernel/random/boot_id
for attribute in version srcversion taint; do
  if [[ -r /sys/module/amdgpu/$attribute ]]; then
    printf 'amdgpu.%s=' "$attribute"
    cat "/sys/module/amdgpu/$attribute"
  fi
done
timeout --kill-after=5s 15s dmesg --since "$1" --time-format=iso --level=err,warn 2>&1 | tail -n 200
