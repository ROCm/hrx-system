#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs the public JIT task-pool example and stages its source and checked output
# for MkDocs.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "$#" -gt 1 ]]; then
  printf 'usage: %s [output-directory]\n' "$0" >&2
  exit 64
fi

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  workspace="${TEST_WORKSPACE:-_main}"
  repo_root="${RUNFILES_DIR}/${workspace}"
  jit_task_pool="${repo_root}/loom/binding/c/example/jit_task_pool"
else
  repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
  "${repo_root}/build_tools/bin/iree-bazel-build" \
    --config=asan \
    //loom/binding/c/example:jit_task_pool
  jit_task_pool="${repo_root}/bazel-bin/loom/binding/c/example/jit_task_pool"
fi

source_file="${repo_root}/loom/binding/c/example/jit_task_pool.c"
for required_path in "${jit_task_pool}" "${source_file}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'required JIT task-pool input not found: %s\n' "${required_path}" >&2
    exit 127
  fi
done

output_dir="${1:-${TEST_UNDECLARED_OUTPUTS_DIR:-${repo_root}/build/loom-docs/examples/integration/jit-task-pool}}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

summary_output="${output_dir}/summary.txt"
"${jit_task_pool}" >"${summary_output}"
expected_summary='compiled=8'
actual_summary="$(<"${summary_output}")"
if [[ "${actual_summary}" != "${expected_summary}" ]]; then
  printf 'unexpected JIT task-pool summary:\n%s\n' "${actual_summary}" >&2
  exit 1
fi

cp -- "${source_file}" "${output_dir}/jit_task_pool.c"

printf 'Generated documentation snippets:\n'
printf '  %s\n' \
  "${output_dir}/jit_task_pool.c" \
  "${output_dir}/summary.txt"
