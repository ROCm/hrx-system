# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C naming helpers for generated op tables."""

from __future__ import annotations

from typing import Any

from loom.dsl import Op
from loom.gen.support.generated_file import line_comment_header

COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

GENERATED_HEADER = COPYRIGHT + "\n" + "\n".join(line_comment_header("//", generator="loom.gen.ops.c_tables")) + "\n// clang-format off"


def c_prefix(op: Op) -> str:
    """Returns the C function/variable prefix for an op.

    test.addi -> loom_test_addi
    """
    return "loom_" + op.name.replace(".", "_")


def c_enum_name(op: Op) -> str:
    """Returns the C enum constant name for an op kind.

    test.addi -> LOOM_OP_TEST_ADDI
    """
    return "LOOM_OP_" + op.name.replace(".", "_").upper()


def c_dialect_enum(dialect_name: str) -> str:
    """Returns the C dialect ID enum name.

    test -> LOOM_DIALECT_TEST
    """
    return "LOOM_DIALECT_" + dialect_name.upper()


def c_dialect_path(dialect: Any) -> str:
    """Returns the generated C source path under loom/src/loom."""
    return dialect.c_path or f"ops/{dialect.name}"


def c_dialect_include_path(dialect: Any) -> str:
    """Returns the generated C include path rooted at loom/src."""
    return f"loom/{c_dialect_path(dialect)}"


def guard_name(dialect_name: str) -> str:
    """Returns the generated ops.h include guard name."""
    return f"LOOM_OPS_{dialect_name.upper()}_OPS_H_"
