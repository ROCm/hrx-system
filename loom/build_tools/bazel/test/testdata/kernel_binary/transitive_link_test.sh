#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

artifact_path="$1"
report_path="$2"
linked_module_path="$3"
loom_format="$4"

test -s "${artifact_path}"
test -s "${report_path}"

artifact_magic="$(od -An -tx1 -N4 "${artifact_path}" | tr -d '[:space:]')"
test "${artifact_magic}" = "7f454c46"
grep -q '"schema_version"' "${report_path}"

linked_text="${TEST_TMPDIR}/linked.loom"
"${loom_format}" "${linked_module_path}" --to=text --output="${linked_text}"
grep -q '@scale' "${linked_text}"
if grep -q '@unused' "${linked_text}"; then
  echo "unused transitive kernel reached the linked product" >&2
  exit 1
fi
