#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

product_root="${TEST_SRCDIR}/${TEST_WORKSPACE}/loom/build_tools/bazel/test/testdata/command_binary"
loom_format="${TEST_SRCDIR}/${TEST_WORKSPACE}/loom/src/loom/tools/loom-format/loom-format"

assert_elf() {
  local artifact_path="$1"
  test -s "${artifact_path}"
  local artifact_magic
  artifact_magic="$(od -An -tx1 -N4 "${artifact_path}" | tr -d '[:space:]')"
  test "${artifact_magic}" = "7f454c46"
}

assert_product() {
  local prefix="$1"
  local expected_program_count="$2"
  local manifest_path="${product_root}/${prefix}.commands.json"
  local artifact_directory="${product_root}/${prefix}.commands"
  local kernel_path="${product_root}/${prefix}.kernels.hsaco"

  test -s "${manifest_path}"
  grep -q '"format":"loom-command-set"' "${manifest_path}"
  test -d "${artifact_directory}"
  local actual_program_count
  actual_program_count="$(find -L "${artifact_directory}" -maxdepth 1 -type f -name '*.loomcmd' | wc -l)"
  test "${actual_program_count}" -eq "${expected_program_count}"
  assert_elf "${kernel_path}"
}

assert_product "deps_subject" 2
assert_product "explicit_subject" 1
assert_product "srcs_subject" 1
assert_product "mixed_subject" 2

grep -q '"symbol":"scale_once"' "${product_root}/deps_subject.commands.json"
grep -q '"symbol":"scale_twice"' "${product_root}/deps_subject.commands.json"
grep -q '"symbol":"scale_once"' "${product_root}/explicit_subject.commands.json"
if grep -q '"symbol":"scale_twice"' "${product_root}/explicit_subject.commands.json"; then
  echo "explicit command root retained an unselected sibling" >&2
  exit 1
fi
grep -q '"symbol":"standalone"' "${product_root}/srcs_subject.commands.json"

grep -q '"backend":"command"' "${product_root}/deps_subject.commands.compile.json"
grep -q '"backend":"amdgpu-hal"' "${product_root}/deps_subject.kernels.compile.json"

linked_text="${TEST_TMPDIR}/deps_subject.loom"
"${loom_format}" "${product_root}/deps_subject.linked.loombc" \
  --to=text --output="${linked_text}"
grep -q '@scale_once' "${linked_text}"
grep -q '@scale_twice' "${linked_text}"
grep -q '@scale' "${linked_text}"
if grep -q '@unused' "${linked_text}"; then
  echo "unused transitive kernel reached the command product" >&2
  exit 1
fi
