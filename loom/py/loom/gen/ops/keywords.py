# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Assembly keyword declarations -> C keyword tables."""

from __future__ import annotations

from loom.gen.assembly.tokens import KEYWORD_MAP

COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""


def generate_keyword_enum_inc() -> str:
    """Generate keyword_enum.inc, the enum body for loom_keyword_id_e."""
    lines = [COPYRIGHT, "// clang-format off", ""]
    for ordinal, (text, c_name) in enumerate(KEYWORD_MAP.items()):
        display = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  {c_name} = {ordinal},  // "{display}"')
    lines.append("")
    return "\n".join(lines)


def generate_keyword_table_inc() -> str:
    """Generate keyword_table.inc, the bstring initializer table."""
    lines = [COPYRIGHT, "// clang-format off", ""]
    for text, c_name in KEYWORD_MAP.items():
        text_length = len(text)
        c_text = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    [{c_name}] = (const uint8_t*)"\\x{text_length:02x}" "{c_text}",')
    lines.append("")
    return "\n".join(lines)
