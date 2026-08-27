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

unversioned_rocm_root="${HRX_ROCM_ROOT}"

python3 -m venv "${HRX_OUTPUT_DIR}/python"
"${HRX_PYTHON}" -m pip install --upgrade pip boto3 zstandard
"${HRX_PYTHON}" build_tools/ci/ci_core_linux.py fetch-rocm

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

rebase_rocm_env_path() {
  local name="$1"
  local value="${!name:-}"
  if [[ "${value}" == "${unversioned_rocm_root}" || \
        "${value}" == "${unversioned_rocm_root}/"* ]]; then
    export_env "${name}" "${HRX_ROCM_ROOT}${value#"${unversioned_rocm_root}"}"
  fi
}

# Bazel's local C++ toolchain repository tracks the compiler path but not the
# contents of a compiler installed outside its repository graph. Give each
# immutable TheRock artifact set its own root so a nightly transition cannot
# reuse a toolchain or compile action created for a different compiler.
rocm_artifact_identity="$(
  "${HRX_PYTHON}" -c \
    'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["artifact_identity"])' \
    "${unversioned_rocm_root}/.hrx-rocm-artifacts.json"
)"
versioned_rocm_root="${unversioned_rocm_root}-${rocm_artifact_identity}"
if [[ -e "${versioned_rocm_root}" ]]; then
  echo "Versioned ROCm root already exists: ${versioned_rocm_root}" >&2
  exit 1
fi
mv -- "${unversioned_rocm_root}" "${versioned_rocm_root}"
HRX_ROCM_ROOT="${versioned_rocm_root}"
export_env "HRX_ROCM_ROOT" "${HRX_ROCM_ROOT}"

# Preserve explicit system toolchains while relocating paths selected from the
# fetched ROCm root.
for name in AR CC CXX IREE_CLANG_TIDY_LLVM_ROOT IREE_ROCM_PATH; do
  rebase_rocm_env_path "${name}"
done

# Keep the ROCm LLVM directory off PATH. Workflows select their host compiler
# explicitly; AMDGPU device compilation receives this root through explicit
# build configuration instead of ambient PATH order.
append_path "${HRX_ROCM_ROOT}/bin"
export_env "CMAKE_PREFIX_PATH" \
  "${HRX_ROCM_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export_env "IREE_HAL_AMDGPU_LIBHSA_PATH" \
  "${HRX_ROCM_ROOT}/lib/libhsa-runtime64.so.1"
export_env "LD_LIBRARY_PATH" \
  "${HRX_ROCM_ROOT}/lib:${HRX_ROCM_ROOT}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
