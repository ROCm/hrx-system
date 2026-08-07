# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C tables and exact classifier for built-in location tags."""

from __future__ import annotations

from loom.gen.support.c import c_string_arg, c_string_classifier_lines
from loom.gen.support.generated_file import line_comment_header
from loom.location_tag import LOCATION_TAG_USER_BASE, BuiltinLocationTag

_COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""


def _c_constant(tag: BuiltinLocationTag) -> str:
    return f"LOOM_LOCATION_TAG_{tag.name}"


def generate_location_tag_table_inc() -> str:
    """Generates built-in location tag checks, names, and classification."""
    tags = tuple(BuiltinLocationTag)
    lines = [_COPYRIGHT.rstrip()]
    lines.extend(
        line_comment_header(
            "//",
            generator="loom.gen.ops.c_tables",
            regenerate="python3 loom/py/loom/gen/run.py c_tables --in-place",
        )
    )
    lines.extend(["// clang-format off", ""])

    lines.extend((f'static_assert({_c_constant(tag)} == {tag.value}, "{_c_constant(tag)} value does not match Python");') for tag in tags)
    lines.append(f'static_assert(LOOM_LOCATION_TAG_USER_BASE == {LOCATION_TAG_USER_BASE}, "location tag user base does not match Python");')
    lines.append("")

    lines.append("static iree_string_view_t loom_location_tag_builtin_name(loom_location_tag_t tag) {")
    lines.append("  switch (tag) {")
    for tag in tags:
        lines.append(f"    case {_c_constant(tag)}:")
        lines.append(f"      return IREE_SV({c_string_arg(tag.spelling)});")
    lines.extend(
        [
            "    default:",
            "      return iree_string_view_empty();",
            "  }",
            "}",
            "",
        ]
    )

    lines.append("static loom_location_tag_t loom_location_tag_classify_name(iree_string_view_t name) {")
    lines.extend(
        c_string_classifier_lines(
            tuple((tag.spelling, _c_constant(tag)) for tag in tags),
            input_name="name",
            unmatched_result="LOOM_LOCATION_TAG_INVALID",
            indent="  ",
        )
    )
    lines.extend(["}", ""])
    return "\n".join(lines)
