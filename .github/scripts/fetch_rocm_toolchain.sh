#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

: "${HRX_OUTPUT_DIR:?}"
: "${HRX_PYTHON:?}"
: "${HRX_ROCM_ROOT:?}"

python3 -m venv "${HRX_OUTPUT_DIR}/python"
"${HRX_PYTHON}" -m pip install --upgrade pip boto3 zstandard
"${HRX_PYTHON}" build_tools/ci_core_linux.py fetch-rocm

append_path() {
  local path="$1"
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "${path}" >>"${GITHUB_PATH}"
  else
    echo "PATH += ${path}"
  fi
}

export_env() {
  local name="$1"
  local value="$2"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "${name}=${value}" >>"${GITHUB_ENV}"
  else
    export "${name}=${value}"
    echo "${name}=${value}"
  fi
}

append_path "${HRX_ROCM_ROOT}/lib/llvm/bin"
append_path "${HRX_ROCM_ROOT}/bin"
export_env "CC" "${HRX_ROCM_ROOT}/lib/llvm/bin/clang"
export_env "CXX" "${HRX_ROCM_ROOT}/lib/llvm/bin/clang++"
export_env "CMAKE_PREFIX_PATH" \
  "${HRX_ROCM_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export_env "LD_LIBRARY_PATH" \
  "${HRX_ROCM_ROOT}/lib:${HRX_ROCM_ROOT}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
