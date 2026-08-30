#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Executes the heterogeneous product workflow and stages its canonical
# request modules and manifest for the documentation build.

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
  export LOOM_LINK="${repo_root}/loom/src/loom/tools/loom-link/loom-link"
  export LOOM_COMPILE="${repo_root}/loom/src/loom/tools/loom-compile/loom-compile"
else
  repo_root="$(cd -- "${script_dir}/../../../.." && pwd -P)"
  "${repo_root}/build_tools/bin/iree-bazel-build" \
    --config=asan \
    //loom/src/loom/tools/loom-format:loom-format \
    //loom/src/loom/tools/loom-link:loom-link \
    //loom/src/loom/tools/loom-compile:loom-compile
  export LOOM_FORMAT="${repo_root}/bazel-bin/loom/src/loom/tools/loom-format/loom-format"
  export LOOM_LINK="${repo_root}/bazel-bin/loom/src/loom/tools/loom-link/loom-link"
  export LOOM_COMPILE="${repo_root}/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile"
fi

output_dir="${1:-${TEST_UNDECLARED_OUTPUTS_DIR:-${repo_root}/build/loom-docs/examples/product-frontiers}}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

"${script_dir}/run.sh" "${output_dir}"

printf 'Generated documentation products:\n'
printf '  %s\n' \
  "${output_dir}/commands.json" \
  "${output_dir}/kernel-0.loom" \
  "${output_dir}/kernel-1.loom" \
  "${output_dir}/kernel-2.loom"
