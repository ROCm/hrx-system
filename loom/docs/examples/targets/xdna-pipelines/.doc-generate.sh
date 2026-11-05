#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${RUNFILES_DIR:-}" ]]; then
  repo_root="${RUNFILES_DIR}/${TEST_WORKSPACE:-_main}"
  export LOOM_FORMAT="${repo_root}/loom/src/loom/tools/loom-format/loom-format"
  export LOOM_IMPORT_CXX="${repo_root}/loom/src/loom/tools/loom-import-cxx/loom-import-cxx"
  export LOOM_COMPILE="${repo_root}/loom/src/loom/tools/loom-compile/loom-compile"
elif [[ -n "${LOOM_DOCS_TOOL_DIR:-}" ]]; then
  repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
  export LOOM_FORMAT="${LOOM_DOCS_TOOL_DIR}/loom-format"
  export LOOM_IMPORT_CXX="${LOOM_DOCS_TOOL_DIR}/loom-import-cxx"
  export LOOM_COMPILE="${LOOM_DOCS_TOOL_DIR}/loom-compile"
else
  repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
  "${repo_root}/build_tools/bin/iree-bazel-build" --config=asan \
    --config=loom-importer-cxx \
    //loom/src/loom/tools/loom-format \
    //loom/src/loom/tools/loom-import-cxx \
    //loom/src/loom/tools/loom-compile
  export LOOM_FORMAT="${repo_root}/bazel-bin/loom/src/loom/tools/loom-format/loom-format"
  export LOOM_IMPORT_CXX="${repo_root}/bazel-bin/loom/src/loom/tools/loom-import-cxx/loom-import-cxx"
  export LOOM_COMPILE="${repo_root}/bazel-bin/loom/src/loom/tools/loom-compile/loom-compile"
fi

output_dir="${1:-${TEST_UNDECLARED_OUTPUTS_DIR:-${repo_root}/build/loom-docs/examples/targets/xdna-pipelines}}"
mkdir -p -- "${output_dir}"
source_file="${repo_root}/loom/docs/examples/targets/xdna-pipelines/gated-projection.cxx"
temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/loom-doc-xdna-pipeline.XXXXXX")"
cleanup() {
  rm -r -- "${temporary_root}"
}
trap cleanup EXIT

for profile in amd.xdna.strix.17f0_10 amd.xdna.strix_halo.17f0_11; do
  "${repo_root}/loom/docs/examples/targets/xdna-pipelines/run.sh" \
    "${source_file}" "${temporary_root}/${profile}" "${profile}"
done

# Publish source and complete compiler-produced IR, without executable caches.
cp -- "${source_file}" "${output_dir}/gated-projection.cxx"
cp -- "${temporary_root}/amd.xdna.strix_halo.17f0_11/imported.loom" \
  "${output_dir}/gated-projection.loom"
sed -n '/^pipeline.def/,$p' \
  "${temporary_root}/amd.xdna.strix_halo.17f0_11/imported.loom" \
  >"${output_dir}/pipeline.loom"
for boundary in outline-pipeline-strands aie2p-lower-pipeline; do
  cp -- "${temporary_root}/amd.xdna.strix_halo.17f0_11/${boundary}.loom" \
    "${output_dir}/${boundary}.loom"
done
