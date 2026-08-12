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

if [[ "$#" -gt 2 ]]; then
  printf 'usage: %s [target] [output-directory]\n' "$0" >&2
  exit 64
fi

target="${1:-gfx11-generic}"
output_dir="${2:-build/elementwise-transform/${target}}"
loom_example_configure_target "${target}"

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

loom_example_section "Check the authored modules"
loom_example_run_tool loom-format "${loom_format}" --check \
  motif.loom kernel.loom model.loom

loom_example_section "Inspect the private motif providers"
loom_example_run_tool loom-link "${loom_link}" motif.loom --list-symbols

loom_example_section "Link and specialize the command root"
loom_example_run_tool loom-link "${loom_link}" \
  model.loom \
  --library=kernel.loom \
  --library=motif.loom \
  --mode=link \
  --root=@elementwise_transform \
  --to=text \
  --output="${output_dir}/elementwise-transform.loom"

loom_example_section "Compile the selected kernel for ${target}"
loom_example_run_tool loom-compile "${loom_compile}" \
  "${output_dir}/elementwise-transform.loom" \
  --backend="${LOOM_EXAMPLE_BACKEND}" \
  --target="${LOOM_EXAMPLE_TARGET}" \
  --root=@elementwise_transform_f32 \
  --output="${output_dir}/elementwise-transform.vmfb" \
  --emit-target-artifact="${output_dir}/elementwise-transform.hsaco" \
  --dump-ir-after=low-select-operand-forms \
  --dump-ir-output="${output_dir}/elementwise-transform-gfx11.loom"

loom_example_section "Artifacts"
printf '  %s\n' \
  "${output_dir}/elementwise-transform.loom" \
  "${output_dir}/elementwise-transform-gfx11.loom" \
  "${output_dir}/elementwise-transform.vmfb" \
  "${output_dir}/elementwise-transform.hsaco"
