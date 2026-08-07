# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C tables and exact classifier for the scalar type vocabulary."""

from __future__ import annotations

from collections import defaultdict

from loom.gen.support.c import c_string_arg
from loom.gen.support.generated_file import line_comment_header
from loom.scalar_type import SCALAR_TYPE_SPELLINGS, ScalarTypeKind

_COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

type ScalarTypeCase = tuple[ScalarTypeKind, str]


def _scalar_type_cases() -> tuple[ScalarTypeCase, ...]:
    """Returns validated scalar kinds and spellings in ordinal order."""
    cases = tuple((ScalarTypeKind(ordinal), spelling) for ordinal, spelling in enumerate(SCALAR_TYPE_SPELLINGS))
    for kind, spelling in cases:
        if not spelling or not spelling.isascii() or not spelling.isalnum():
            raise ValueError(f"scalar type {kind.name} spelling must be non-empty ASCII alphanumeric text: {spelling!r}")
    return cases


def _c_constant(kind: ScalarTypeKind) -> str:
    return f"LOOM_SCALAR_TYPE_{kind.name}"


def generate_scalar_type_table_inc() -> str:
    """Generates scalar type ordinal checks, names, and exact classification."""
    cases = _scalar_type_cases()
    lines = [_COPYRIGHT.rstrip()]
    lines.extend(
        line_comment_header(
            "//",
            generator="loom.gen.ops.c_tables",
            regenerate="python3 loom/py/loom/gen/run.py c_tables --in-place",
        )
    )
    lines.extend(["// clang-format off", ""])

    for kind, _ in cases:
        lines.append(f'static_assert({_c_constant(kind)} == {kind.value}, "{_c_constant(kind)} ordinal does not match Python");')
    lines.append(f'static_assert(LOOM_SCALAR_TYPE_COUNT_ == {len(cases)}, "scalar type count does not match Python");')
    lines.append("")

    lines.append("static const char* const loom_scalar_type_names[LOOM_SCALAR_TYPE_COUNT_] = {")
    for kind, spelling in cases:
        lines.append(f"    [{_c_constant(kind)}] = {c_string_arg(spelling)},")
    lines.extend(["};", ""])

    lines.append("static loom_scalar_type_t loom_scalar_type_classify_name(iree_string_view_t name) {")
    by_length: dict[int, list[ScalarTypeCase]] = defaultdict(list)
    for case in cases:
        by_length[len(case[1])].append(case)
    lines.append("  switch (name.size) {")
    for spelling_length, length_cases in sorted(by_length.items()):
        lines.append(f"    case {spelling_length}:")
        for kind, spelling in length_cases:
            lines.append(f"      if (iree_string_view_equal(name, IREE_SV({c_string_arg(spelling)}))) {{")
            lines.append(f"        return {_c_constant(kind)};")
            lines.append("      }")
        lines.append("      break;")
    lines.extend(
        [
            "  }",
            "  return LOOM_SCALAR_TYPE_COUNT_;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)
