#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

product_root="${TEST_SRCDIR}/${TEST_WORKSPACE}/loom/docs/examples/elementwise-transform"
iree_dump_module="${TEST_SRCDIR}/${TEST_WORKSPACE}/$1"

assert_elf() {
  local artifact_path="$1"
  test -s "${artifact_path}"
  local artifact_magic
  artifact_magic="$(od -An -tx1 -N4 "${artifact_path}" | tr -d '[:space:]')"
  test "${artifact_magic}" = "7f454c46"
}

assert_elf "${product_root}/elementwise_kernel.hsaco"
assert_elf "${product_root}/elementwise_command.kernels.hsaco"

command_manifest="${product_root}/elementwise_command.commands.json"
command_artifacts="${product_root}/elementwise_command.commands"
grep -q '"format":"loom-command-set"' "${command_manifest}"
grep -q '"symbol":"elementwise_transform"' "${command_manifest}"
test "$(find -L "${command_artifacts}" -maxdepth 1 -type f -name '*.loomcmd' | wc -l)" -eq 1

vmfb="${product_root}/elementwise_vm.vmfb"
test -s "${vmfb}"
"${iree_dump_module}" "${vmfb}" | grep -q "@double_i32"
