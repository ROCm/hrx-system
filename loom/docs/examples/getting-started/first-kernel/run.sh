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
loom_import_cxx="${LOOM_IMPORT_CXX:-loom-import-cxx}"
loom_compile="${LOOM_COMPILE:-loom-compile}"
loom_benchmark="${LOOM_BENCHMARK:-loom-benchmark}"

loom_example_require_tool "${loom_format}"
loom_example_require_tool "${loom_import_cxx}"
loom_example_require_tool "${loom_compile}"
loom_example_require_tool "${loom_benchmark}"

if [[ -z "${mode}" ]]; then
  loom_test="${LOOM_TEST:-loom-test}"
  loom_example_require_tool "${loom_test}"
fi

cd -- "${script_dir}"

loom_example_section "Check the authored source"
loom_example_run_tool loom-format "${loom_format}" --check saxpy.loom

loom_example_section "Import the same kernel from C++"
loom_example_run_tool loom-import-cxx "${loom_import_cxx}" \
  saxpy.cxx --output="${output_dir}/saxpy-cxx.loom"
loom_example_run_tool loom-format "${loom_format}" \
  --check "${output_dir}/saxpy-cxx.loom"

for source_file in saxpy.loom "${output_dir}/saxpy-cxx.loom"; do
  source_name="${source_file##*/}"
  source_name="${source_name%.loom}"

  loom_example_section "Plan the named benchmark workload in ${source_name}"
  loom_example_run_tool loom-benchmark "${loom_benchmark}" \
    "${source_file}" \
    --benchmark=@saxpy_f32_64k \
    --dry-run \
    --output="${output_dir}/${source_name}-plan.json"

  loom_example_section "Compile ${source_name} for ${target}"
  loom_example_run_tool loom-compile "${loom_compile}" \
    "${source_file}" \
    --format="${LOOM_EXAMPLE_FORMAT}" \
    --target="amdgpu:${LOOM_EXAMPLE_TARGET}" \
    --root=@saxpy_f32 \
    --output="${output_dir}/${source_name}.hsaco"

  if [[ -z "${mode}" ]]; then
    loom_example_section "Execute every correctness sample in ${source_name}"
    printf '  $ loom-test %q --device=amdgpu > %q\n' \
      "${source_file}" "${output_dir}/${source_name}-test.json"
    "${loom_test}" "${source_file}" --device=amdgpu \
      >"${output_dir}/${source_name}-test.json"

    loom_example_section "Benchmark the proven 64K workload in ${source_name}"
    loom_example_run_tool loom-benchmark "${loom_benchmark}" \
      "${source_file}" \
      --device=amdgpu \
      --benchmark=@saxpy_f32_64k \
      --measure=dispatch_complete \
      --batch-size=64 \
      --output="${output_dir}/${source_name}-benchmark.json"
  fi
done

loom_example_section "Artifacts"
printf '  %s\n' "${output_dir}/saxpy-cxx.loom"
for source_name in saxpy saxpy-cxx; do
  printf '  %s\n' \
    "${output_dir}/${source_name}-plan.json" \
    "${output_dir}/${source_name}.hsaco"
  if [[ -z "${mode}" ]]; then
    printf '  %s\n' \
      "${output_dir}/${source_name}-test.json" \
      "${output_dir}/${source_name}-benchmark.json"
  fi
done
