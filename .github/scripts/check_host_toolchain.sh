#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

expected_origin="${1:---expect-system}"
case "${expected_origin}" in
  --expect-system | --expect-artifact) ;;
  *)
    echo "usage: $0 [--expect-system|--expect-artifact]" >&2
    exit 2
    ;;
esac
if [[ "$#" -gt 1 ]]; then
  echo "usage: $0 [--expect-system|--expect-artifact]" >&2
  exit 2
fi

resolved_artifact_llvm_bin=""
if [[ -n "${HRX_ROCM_ROOT:-}" ]]; then
  resolved_artifact_llvm_bin="$(realpath "${HRX_ROCM_ROOT}/lib/llvm/bin")"
elif [[ "${expected_origin}" == "--expect-artifact" ]]; then
  echo "::error::HRX_ROCM_ROOT is required when expecting a fetched host toolchain."
  exit 1
fi

check_host_tool() {
  local environment_variable="$1"
  local default_tool="$2"
  local tool="${!environment_variable:-${default_tool}}"
  local tool_path
  if ! tool_path="$(command -v "${tool}")"; then
    echo "::error::Host ${environment_variable} tool '${tool}' was not found."
    exit 1
  fi
  local resolved_tool_path
  resolved_tool_path="$(realpath "${tool_path}")"
  local is_artifact_tool=0
  if [[ -n "${resolved_artifact_llvm_bin}" && \
        ("${tool_path}" == "${HRX_ROCM_ROOT}"/* || \
         "${resolved_tool_path}" == "${resolved_artifact_llvm_bin}"/*) ]]; then
    is_artifact_tool=1
  fi
  if [[ "${expected_origin}" == "--expect-system" && "${is_artifact_tool}" -ne 0 ]]; then
    echo "::error::Host ${environment_variable} unexpectedly resolves into the fetched toolchain: ${resolved_tool_path}"
    exit 1
  fi
  if [[ "${expected_origin}" == "--expect-artifact" && "${is_artifact_tool}" -eq 0 ]]; then
    echo "::error::Host ${environment_variable} does not resolve into the expected fetched toolchain: ${resolved_tool_path}"
    exit 1
  fi
  echo "Host ${environment_variable}: ${resolved_tool_path}"
  "${tool}" --version
}

check_host_tool CC cc
check_host_tool CXX c++
check_host_tool AR ar
