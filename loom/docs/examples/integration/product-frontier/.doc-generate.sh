#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs the public product-frontier example and stages its source and checked
# output for MkDocs.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "$#" -gt 1 ]]; then
  printf 'usage: %s [output-directory]\n' "$0" >&2
  exit 64
fi

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  workspace="${TEST_WORKSPACE:-_main}"
  repo_root="${RUNFILES_DIR}/${workspace}"
  product_frontier="${repo_root}/loom/binding/c/example/product_frontier"
else
  repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
  "${repo_root}/build_tools/bin/iree-bazel-build" \
    --config=asan \
    //loom/binding/c/example:product_frontier
  product_frontier="${repo_root}/bazel-bin/loom/binding/c/example/product_frontier"
fi

source_file="${repo_root}/loom/binding/c/example/product_frontier.c"
for required_path in "${product_frontier}" "${source_file}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'required product-frontier input not found: %s\n' "${required_path}" >&2
    exit 127
  fi
done

output_dir="${1:-${TEST_UNDECLARED_OUTPUTS_DIR:-${repo_root}/build/loom-docs/examples/integration/product-frontier}}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/loom-doc-product-frontier.XXXXXX")"
cleanup() {
  rm -r -- "${temporary_root}"
}
trap cleanup EXIT

summary_output="${temporary_root}/summary.txt"
"${product_frontier}" >"${summary_output}"
expected_summary='command=run programs=1 requirements=2 kernel_requests=1'
actual_summary="$(<"${summary_output}")"
if [[ "${actual_summary}" != "${expected_summary}" ]]; then
  printf 'unexpected product-frontier summary:\n%s\n' "${actual_summary}" >&2
  exit 1
fi

cp -- "${source_file}" "${output_dir}/product_frontier.c"
cp -- "${summary_output}" "${output_dir}/summary.txt"

printf 'Generated documentation snippets:\n'
printf '  %s\n' \
  "${output_dir}/product_frontier.c" \
  "${output_dir}/summary.txt"
