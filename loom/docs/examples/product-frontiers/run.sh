#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=loom/docs/examples/example.shlib
source "${script_dir}/../example.shlib"

if [[ "$#" -gt 1 ]]; then
  printf 'usage: %s [output-directory]\n' "$0" >&2
  exit 64
fi

output_dir="${1:-build/product-frontiers}"
if [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

loom_format="${LOOM_FORMAT:-loom-format}"
loom_link="${LOOM_LINK:-loom-link}"
loom_compile="${LOOM_COMPILE:-loom-compile}"

loom_example_require_tool "${loom_format}"
loom_example_require_tool "${loom_link}"
loom_example_require_tool "${loom_compile}"

cd -- "${script_dir}"

loom_example_section "Check each independently owned source module"
loom_example_run_tool loom-format "${loom_format}" --check \
  requester.loom stage.loom providers.loom

loom_example_section "Package one provider as bytecode"
loom_example_run_tool loom-link "${loom_link}" \
  providers.loom \
  --mode=merge \
  --to=bc \
  --output="${output_dir}/providers.loombc"

loom_example_section "Seal the requester over mixed text and bytecode libraries"
loom_example_run_tool loom-link "${loom_link}" \
  requester.loom \
  --library=stage.loom \
  --library="${output_dir}/providers.loombc" \
  --mode=link \
  --root=@heterogeneous \
  --allow-unresolved \
  --to=bc \
  --output="${output_dir}/heterogeneous.loombc"

loom_example_section "Emit the parent command product and child requests"
loom_example_run_tool loom-compile "${loom_compile}" \
  "${output_dir}/heterogeneous.loombc" \
  --format=loom-command \
  --root=@heterogeneous \
  --output="${output_dir}/commands.json" \
  --emit-command-artifacts="${output_dir}/commands" \
  --emit-kernel-requests="${output_dir}/kernel-requests"

request_paths=("${output_dir}"/kernel-requests/*.loombc)
if [[ "${#request_paths[@]}" -ne 3 ]]; then
  printf 'expected three kernel requests, found %u\n' \
    "${#request_paths[@]}" >&2
  exit 1
fi

for request_path in "${request_paths[@]}"; do
  request_name="$(basename -- "${request_path}" .loombc)"
  loom_example_run_tool loom-format "${loom_format}" \
    "${request_path}" \
    --to=text \
    --output="${output_dir}/${request_name}.loom"
  loom_example_run_tool loom-format "${loom_format}" --check \
    "${output_dir}/${request_name}.loom"
done

grep -Fq '"symbol":"heterogeneous"' "${output_dir}/commands.json"
grep -Fq '"symbol":"external"' "${output_dir}/commands.json"
grep -Fq 'kernel.def retain @local' "${output_dir}/kernel-0.loom"
grep -Fq 'template.call @small' "${output_dir}/kernel-1.loom"
grep -Fq 'template.call @large' "${output_dir}/kernel-2.loom"

loom_example_section "Artifacts"
printf '  %s\n' \
  "${output_dir}/heterogeneous.loombc" \
  "${output_dir}/commands.json" \
  "${output_dir}/commands/program-0.loomcmd" \
  "${request_paths[@]}"
