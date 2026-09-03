#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=loom/docs/examples/example.shlib
source "${script_dir}/../../example.shlib"

if [[ "$#" -gt 3 ]]; then
  printf 'usage: %s [target] [output-directory] [--compile-only]\n' "$0" >&2
  exit 64
fi

target="${1:-gfx11-generic}"
output_dir="${2:-build/first-kernel/${target}}"
mode="${3:-}"
if [[ -n "${mode}" && "${mode}" != "--compile-only" ]]; then
  printf 'unsupported mode: %s\n' "${mode}" >&2
  printf 'supported mode: --compile-only\n' >&2
  exit 64
fi

loom_example_configure_target "${target}"

if [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

loom_format="${LOOM_FORMAT:-loom-format}"
loom_compile="${LOOM_COMPILE:-loom-compile}"
iree_benchmark_loom="${IREE_BENCHMARK_LOOM:-iree-benchmark-loom}"

loom_example_require_tool "${loom_format}"
loom_example_require_tool "${loom_compile}"
loom_example_require_tool "${iree_benchmark_loom}"

if [[ -z "${mode}" ]]; then
  iree_test_loom="${IREE_TEST_LOOM:-iree-test-loom}"
  loom_example_require_tool "${iree_test_loom}"
fi

cd -- "${script_dir}"

loom_example_section "Check the authored source"
loom_example_run_tool loom-format "${loom_format}" --check saxpy.loom

loom_example_section "Plan the named benchmark workload"
loom_example_run_tool iree-benchmark-loom "${iree_benchmark_loom}" \
  saxpy.loom \
  --benchmark=@saxpy_f32_64k \
  --dry-run \
  --output="${output_dir}/saxpy-plan.json"

loom_example_section "Compile the deployment kernel for ${target}"
loom_example_run_tool loom-compile "${loom_compile}" \
  saxpy.loom \
  --backend="${LOOM_EXAMPLE_BACKEND}" \
  --target="${LOOM_EXAMPLE_TARGET}" \
  --root=@saxpy_f32 \
  --output="${output_dir}/saxpy.hsaco"

if [[ -z "${mode}" ]]; then
  loom_example_section "Execute every correctness sample on AMDGPU"
  printf '  $ iree-test-loom saxpy.loom --device=amdgpu > %q\n' \
    "${output_dir}/saxpy-test.json"
  "${iree_test_loom}" saxpy.loom --device=amdgpu \
    >"${output_dir}/saxpy-test.json"

  loom_example_section "Benchmark the proven 64K workload"
  loom_example_run_tool iree-benchmark-loom "${iree_benchmark_loom}" \
    saxpy.loom \
    --device=amdgpu \
    --benchmark=@saxpy_f32_64k \
    --measure=dispatch_complete \
    --batch-size=64 \
    --output="${output_dir}/saxpy-benchmark.json"
fi

loom_example_section "Artifacts"
printf '  %s\n' \
  "${output_dir}/saxpy-plan.json" \
  "${output_dir}/saxpy.hsaco"
if [[ -z "${mode}" ]]; then
  printf '  %s\n' \
    "${output_dir}/saxpy-test.json" \
    "${output_dir}/saxpy-benchmark.json"
fi
