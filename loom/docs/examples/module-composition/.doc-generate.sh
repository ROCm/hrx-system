#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Executes the reader-facing partial-link workflow and stages only its verified
# textual artifacts for inclusion by MkDocs.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=loom/docs/examples/example.shlib
source "${script_dir}/../example.shlib"
repo_root="${LOOM_EXAMPLE_REPO_ROOT}"

if [[ "$#" -gt 1 ]]; then
  printf 'usage: %s [output-directory]\n' "$0" >&2
  exit 64
fi

loom_example_configure_inputs tool:loom-format tool:loom-link

output_dir="${1:-${TEST_UNDECLARED_OUTPUTS_DIR:-${repo_root}/build/loom-docs/examples/module-composition}}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

"${script_dir}/run.sh" "${output_dir}"

"${LOOM_FORMAT}" --check \
  "${output_dir}/partial.loom" \
  "${output_dir}/linked.loom"

printf 'Generated documentation snippets:\n'
printf '  %s\n' \
  "${output_dir}/partial.loom" \
  "${output_dir}/linked.loom"
