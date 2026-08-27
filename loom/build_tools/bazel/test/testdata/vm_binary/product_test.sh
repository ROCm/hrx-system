#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

product_root="${TEST_SRCDIR}/${TEST_WORKSPACE}/loom/build_tools/bazel/test/testdata/vm_binary"
loom_format="${TEST_SRCDIR}/${TEST_WORKSPACE}/$1"
iree_run_module="${TEST_SRCDIR}/${TEST_WORKSPACE}/$2"

for prefix in deps_subject explicit_subject mixed_subject srcs_subject; do
  test -s "${product_root}/${prefix}.vmfb"
done

"${iree_run_module}" \
  --module="${product_root}/deps_subject.vmfb" \
  --function=double_once \
  --input=21 \
  | grep -q "i32=42"
"${iree_run_module}" \
  --module="${product_root}/deps_subject.vmfb" \
  --function=double_twice \
  --input=10 \
  | grep -q "i32=40"
"${iree_run_module}" \
  --module="${product_root}/explicit_subject.vmfb" \
  --function=double_once \
  --input=21 \
  | grep -q "i32=42"
"${iree_run_module}" \
  --module="${product_root}/srcs_subject.vmfb" \
  --function=standalone \
  --input=21 \
  | grep -q "i32=42"

if "${iree_run_module}" \
  --module="${product_root}/explicit_subject.vmfb" \
  --function=double_twice \
  --input=10 >/dev/null 2>&1; then
  echo "explicit VM root retained an unselected sibling" >&2
  exit 1
fi

grep -q '"artifact_kind":"vm-archive"' \
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
