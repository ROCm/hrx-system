#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs the hardware-independent portion of the public quickstart, verifies its
# products, and stages both source forms and the Bazel declaration included by
# the page.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=loom/docs/examples/example.shlib
source "${script_dir}/../../example.shlib"
repo_root="${LOOM_EXAMPLE_REPO_ROOT}"

if [[ "$#" -gt 1 ]]; then
  printf 'usage: %s [output-directory]\n' "$0" >&2
  exit 64
fi

loom_example_configure_inputs \
  --config=loom-importer-cxx \
  tool:loom-format \
  tool:loom-import-cxx \
  tool:loom-compile \
  tool:loom-benchmark

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
grep -Fq '"benchmark":"saxpy_f32_64k"' \
  "${artifact_root}/saxpy-cxx-plan.json"
grep -Fq '"case":"saxpy_f32_65536"' "${artifact_root}/saxpy-cxx-plan.json"
test -s "${artifact_root}/saxpy-cxx.hsaco"

cp -- "${source_file}" "${output_dir}/saxpy.loom"
cp -- "${script_dir}/saxpy.cxx" "${output_dir}/saxpy.cxx"
cp -- "${build_file}" "${output_dir}/BUILD.bazel"

printf 'Generated documentation snippets:\n'
printf '  %s\n' \
  "${output_dir}/saxpy.loom" \
  "${output_dir}/saxpy.cxx" \
  "${output_dir}/BUILD.bazel"
