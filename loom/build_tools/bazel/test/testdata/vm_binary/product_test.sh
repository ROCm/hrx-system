#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

product_root="${TEST_SRCDIR}/${TEST_WORKSPACE}/loom/build_tools/bazel/test/testdata/vm_binary"
loom_format="${TEST_SRCDIR}/${TEST_WORKSPACE}/$1"
iree_dump_module="${TEST_SRCDIR}/${TEST_WORKSPACE}/$2"

for prefix in deps_subject explicit_subject mixed_subject srcs_subject; do
  test -s "${product_root}/${prefix}.vmfb"
done

"${iree_dump_module}" "${product_root}/deps_subject.vmfb" \
  | grep -q "@double_once"
"${iree_dump_module}" "${product_root}/deps_subject.vmfb" \
  | grep -q "@double_twice"
"${iree_dump_module}" "${product_root}/explicit_subject.vmfb" \
  | grep -q "@double_once"
"${iree_dump_module}" "${product_root}/srcs_subject.vmfb" \
  | grep -q "@standalone"

if "${iree_dump_module}" "${product_root}/explicit_subject.vmfb" \
  | grep -q "@double_twice"; then
  echo "explicit VM root retained an unselected sibling" >&2
  exit 1
fi

grep -q '"artifact_kind":"target-artifact"' \
  "${product_root}/deps_subject.compile.json"

linked_text="${TEST_TMPDIR}/deps_subject.loom"
"${loom_format}" "${product_root}/deps_subject.linked.loombc" \
  --to=text --output="${linked_text}"
grep -q '@double_once' "${linked_text}"
grep -q '@double_twice' "${linked_text}"
grep -q '@double_template' "${linked_text}"
if grep -q '@unused' "${linked_text}"; then
  echo "unused transitive function reached the VM product" >&2
  exit 1
fi
