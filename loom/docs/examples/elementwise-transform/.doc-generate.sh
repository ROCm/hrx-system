#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds the reader-facing example through run.sh, verifies its products, and
# stages only generated Loom snippets for the documentation build.

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
  loom_check="${repo_root}/loom/src/loom/tools/loom-check/loom-check"
else
  repo_root="$(cd -- "${script_dir}/../../../.." && pwd -P)"
  "${repo_root}/build_tools/bin/iree-bazel-build" \
    --config=asan \
    //loom/src/loom/tools/loom-format:loom-format \
    //loom/src/loom/tools/loom-link:loom-link \
    //loom/src/loom/tools/loom-compile:loom-compile \
    //loom/src/loom/tools/loom-check:loom-check
  export LOOM_FORMAT="${repo_root}/bazel-bin/loom/src/loom/tools/loom-format/loom-format"
  export LOOM_LINK="${repo_root}/bazel-bin/loom/src/loom/tools/loom-link/loom-link"
  export LOOM_COMPILE="${repo_root}/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile"
  loom_check="${repo_root}/bazel-bin/loom/src/loom/tools/loom-check/loom-check"
fi

if [[ ! -x "${loom_check}" ]]; then
  printf 'required Loom tool not found: %s\n' "${loom_check}" >&2
  exit 127
fi

output_dir="${1:-${TEST_UNDECLARED_OUTPUTS_DIR:-${repo_root}/build/loom-docs/examples/elementwise-transform}}"
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/loom-doc-elementwise-transform.XXXXXX")"
cleanup() {
  rm -rf -- "${temporary_root}"
}
trap cleanup EXIT

artifact_root="${temporary_root}/artifacts"
"${script_dir}/run.sh" gfx11-generic "${artifact_root}"

linked_program="${artifact_root}/elementwise-transform.loom"
grep -Fq '@wave32_elementwise_transform' "${linked_program}"
grep -Fq 'amdgpu.target<gfx11-generic>' "${linked_program}"
if grep -Fq '@portable_elementwise_transform' "${linked_program}"; then
  printf 'portable provider survived the target-aware product link\n' >&2
  exit 1
fi

kernel_dump="${artifact_root}/elementwise-transform-gfx11.loom"
kernel_output="${output_dir}/elementwise-transform-gfx11.loom"
kernel_unformatted="${temporary_root}/elementwise-transform-gfx11.loom"
awk '!found && /^low\.kernel\.def / { found = 1 } found { print }' \
  "${kernel_dump}" >"${kernel_unformatted}"
"${LOOM_FORMAT}" "${kernel_unformatted}" \
  --to=text --output="${kernel_output}"
"${LOOM_FORMAT}" --check "${kernel_output}"

# The public workflow above emits the deployment artifacts. Generate the
# readable target-neutral Low view separately for the walkthrough.
command_fixture="${temporary_root}/elementwise-transform-program.loom-test"
{
  printf '// RUN: emit command-program @elementwise_transform\n\n'
  sed -n '1,$p' "${artifact_root}/elementwise-transform.loom"
  printf '\n// ----\n'
} >"${command_fixture}"

update_log="${temporary_root}/command-program-update.log"
set +e
"${loom_check}" --update "${command_fixture}" >"${update_log}" 2>&1
update_status="$?"
set -e
if [[ "${update_status}" -ne 0 && "${update_status}" -ne 1 ]]; then
  sed -n '1,$p' "${update_log}" >&2
  exit "${update_status}"
fi
if ! "${loom_check}" "${command_fixture}"; then
  sed -n '1,$p' "${update_log}" >&2
  exit 1
fi

command_output="${output_dir}/elementwise-transform-program.loom"
awk 'found { print } $0 == "// ----" { found = 1 }' \
  "${command_fixture}" >"${command_output}"

grep -Fq '@elementwise_transform_f32' "${kernel_output}"
grep -Fq 'amdgpu.global_load_b32_saddr' "${kernel_output}"
grep -Fq 'amdgpu.v_add_f32' "${kernel_output}"
grep -Fq 'amdgpu.global_store_b32_saddr' "${kernel_output}"
grep -Fq '"format":"loom-command-set"' \
  "${artifact_root}/elementwise-transform.commands.json"
grep -Fq '"symbol":"elementwise_transform"' \
  "${artifact_root}/elementwise-transform.commands.json"
test -s "${artifact_root}/elementwise-transform.commands/program-0.loomcmd"
grep -Fq '@elementwise_transform() asm' "${command_output}"
grep -Fq 'cmd.dispatch.' "${command_output}"

printf 'Generated documentation snippets:\n'
printf '  %s\n' "${kernel_output}" "${command_output}"
