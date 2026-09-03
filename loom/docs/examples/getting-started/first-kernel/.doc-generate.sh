#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs the hardware-independent portion of the public quickstart, verifies its
# products, and stages the exact source and Bazel declaration included by the
# page.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "$#" -gt 1 ]]; then
  printf 'usage: %s [output-directory]\n' "$0" >&2
  exit 64
fi

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  workspace="${TEST_WORKSPACE:-_main}"
  repo_root="${RUNFILES_DIR}/${workspace}"
  export LOOM_FORMAT="${repo_root}/loom/src/loom/tools/loom-format/loom-format"
  export LOOM_COMPILE="${repo_root}/loom/src/loom/tools/loom-compile/loom-compile"
  export IREE_BENCHMARK_LOOM="${repo_root}/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom"
else
  repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
  "${repo_root}/build_tools/bin/iree-bazel-build" \
    --config=asan \
    //loom/src/loom/tools/loom-format:loom-format \
    //loom/src/loom/tools/loom-compile:loom-compile \
    //loom/src/loom/tools/iree-benchmark-loom:iree-benchmark-loom
  export LOOM_FORMAT="${repo_root}/bazel-bin/loom/src/loom/tools/loom-format/loom-format"
  export LOOM_COMPILE="${repo_root}/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile"
  export IREE_BENCHMARK_LOOM="${repo_root}/bazel-bin/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom"
fi

source_file="${script_dir}/saxpy.loom"
build_file="${script_dir}/BUILD.bazel"
output_dir="${1:-${TEST_UNDECLARED_OUTPUTS_DIR:-${repo_root}/build/loom-docs/examples/getting-started/first-kernel}}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/loom-doc-first-kernel.XXXXXX")"
cleanup() {
  rm -r -- "${temporary_root}"
}
trap cleanup EXIT

artifact_root="${temporary_root}/artifacts"
"${script_dir}/run.sh" gfx11-generic "${artifact_root}" --compile-only

grep -Fq '"benchmark":"saxpy_f32_64k"' \
  "${artifact_root}/saxpy-plan.json"
grep -Fq '"element_count":65536' "${artifact_root}/saxpy-plan.json"
test -s "${artifact_root}/saxpy.hsaco"

cp -- "${source_file}" "${output_dir}/saxpy.loom"
cp -- "${build_file}" "${output_dir}/BUILD.bazel"

printf 'Generated documentation snippets:\n'
printf '  %s\n' \
  "${output_dir}/saxpy.loom" \
  "${output_dir}/BUILD.bazel"
