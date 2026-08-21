#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs both public in-memory composition modes, proves that their linked
# programs agree, and stages the compiled source and output for MkDocs.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "$#" -gt 1 ]]; then
  printf 'usage: %s [output-directory]\n' "$0" >&2
  exit 64
fi

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  workspace="${TEST_WORKSPACE:-_main}"
  repo_root="${RUNFILES_DIR}/${workspace}"
  link_modules="${repo_root}/loom/binding/c/example/link_modules"
  loom_format="${repo_root}/loom/src/loom/tools/loom-format/loom-format"
else
  repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
  "${repo_root}/build_tools/bin/iree-bazel-build" \
    --config=asan \
    //loom/binding/c/example:link_modules \
    //loom/src/loom/tools/loom-format:loom-format
  link_modules="${repo_root}/bazel-bin/loom/binding/c/example/link_modules"
  loom_format="${repo_root}/bazel-bin/loom/src/loom/tools/loom-format/loom-format"
fi

source_file="${repo_root}/loom/binding/c/example/link_modules.c"
for required_path in "${link_modules}" "${loom_format}" "${source_file}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'required composition input not found: %s\n' "${required_path}" >&2
    exit 127
  fi
done

output_dir="${1:-${TEST_UNDECLARED_OUTPUTS_DIR:-${repo_root}/build/loom-docs/examples/integration/module-composition}}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/loom-doc-module-composition.XXXXXX")"
cleanup() {
  rm -rf -- "${temporary_root}"
}
trap cleanup EXIT

declaration_output="${temporary_root}/declaration-only.loom"
provider_import_output="${temporary_root}/provider-import.loom"
"${link_modules}" >"${declaration_output}"
"${link_modules}" --provider-import >"${provider_import_output}"

"${loom_format}" --check \
  "${declaration_output}" \
  "${provider_import_output}"
if ! cmp -s "${declaration_output}" "${provider_import_output}"; then
  printf 'declaration-only and provider-import links produced different programs\n' >&2
  exit 1
fi

cp -- "${provider_import_output}" "${output_dir}/linked.loom"
cp -- "${source_file}" "${output_dir}/link_modules.c"

printf 'Generated documentation snippets:\n'
printf '  %s\n' \
  "${output_dir}/linked.loom" \
  "${output_dir}/link_modules.c"
