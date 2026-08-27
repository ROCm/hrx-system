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

output_dir="${1:-build/module-composition}"
if [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

loom_format="${LOOM_FORMAT:-loom-format}"
loom_link="${LOOM_LINK:-loom-link}"

loom_example_require_tool "${loom_format}"
loom_example_require_tool "${loom_link}"

cd -- "${script_dir}"

loom_example_section "Check each independently verifiable module"
loom_example_run_tool loom-format "${loom_format}" --check \
  root.loom layer.loom kernels.loom

loom_example_section "Link the available layer and retain its kernel demand"
loom_example_run_tool loom-link "${loom_link}" \
  root.loom \
  --library=layer.loom \
  --mode=link \
  --root=@entry \
  --allow-unresolved \
  --to=bc \
  --output="${output_dir}/partial.loombc"
loom_example_run_tool loom-format "${loom_format}" \
  "${output_dir}/partial.loombc" \
  --to=text \
  --output="${output_dir}/partial.loom"

if grep -Fq 'func.decl @project_layer' "${output_dir}/partial.loom"; then
  printf 'satisfied layer declaration survived the partial link\n' >&2
  exit 1
fi
grep -Fq 'func.decl pure @scale' "${output_dir}/partial.loom"
grep -Fq 'func.def @project_layer' "${output_dir}/partial.loom"

loom_example_section "Reload the partial artifact and complete the program"
loom_example_run_tool loom-link "${loom_link}" \
  "${output_dir}/partial.loombc" \
  --library=kernels.loom \
  --mode=link \
  --root=@entry \
  --to=text \
  --output="${output_dir}/linked.loom"
loom_example_run_tool loom-format "${loom_format}" --check \
  "${output_dir}/linked.loom"

if grep -Eq '^func\.decl' "${output_dir}/linked.loom"; then
  printf 'completed link retained a declaration\n' >&2
  exit 1
fi
grep -Fq 'func.def public retain @entry' "${output_dir}/linked.loom"
grep -Fq 'func.def @project_layer' "${output_dir}/linked.loom"
grep -Fq 'func.def pure @scale' "${output_dir}/linked.loom"

loom_example_section "Artifacts"
printf '  %s\n' \
  "${output_dir}/partial.loombc" \
  "${output_dir}/partial.loom" \
  "${output_dir}/linked.loom"
