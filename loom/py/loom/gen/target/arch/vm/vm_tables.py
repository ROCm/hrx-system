# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates the complete Loom-facing Core VM table family."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from loom.gen.ops.c_names import COPYRIGHT, c_enum_name
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.low.c_spelling import descriptor_ref_constant_name
from loom.gen.target.low.low_descriptors import write_descriptor_set_to_paths
from loom.target.arch.vm.projection import (
    VM_CORE_DESCRIPTOR_SET,
    VM_SOURCE_LOWERINGS,
)


def _scalar_type_name(scalar_type) -> str:
    return f"LOOM_SCALAR_TYPE_{scalar_type.name}"


def generate_lowering_rows() -> str:
    """Returns the X-macro source-op signature projection."""

    maximum_operand_count = max((len(row.operand_types) for row in VM_SOURCE_LOWERINGS), default=0)
    maximum_result_count = max((len(row.result_types) for row in VM_SOURCE_LOWERINGS), default=0)
    lines = [
        COPYRIGHT.rstrip(),
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.vm.vm_tables"),
        "",
        "LOOM_VM_SOURCE_LOWERING_LIMITS(",
        f"    {maximum_operand_count}, {maximum_result_count})",
    ]
    for row in VM_SOURCE_LOWERINGS:
        operand_types = [_scalar_type_name(scalar_type) for scalar_type in row.operand_types]
        result_types = [_scalar_type_name(scalar_type) for scalar_type in row.result_types]
        operand_types.extend(["LOOM_SCALAR_TYPE_INDEX"] * (maximum_operand_count - len(operand_types)))
        result_types.extend(["LOOM_SCALAR_TYPE_INDEX"] * (maximum_result_count - len(result_types)))
        descriptor_name = descriptor_ref_constant_name(VM_CORE_DESCRIPTOR_SET, row.descriptor_key)
        arguments = [
            c_enum_name(row.source_op),
            descriptor_name,
            str(len(row.operand_types)),
            str(len(row.result_types)),
            *operand_types,
            *result_types,
        ]
        lines.append("LOOM_VM_SOURCE_LOWERING_ROW(")
        lines.append("    " + ", ".join(arguments) + ")")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Loom-facing Core VM C table family.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--lowering-rows", required=True, type=Path)
    args = parser.parse_args(argv)

    write_descriptor_set_to_paths(
        VM_CORE_DESCRIPTOR_SET,
        header_path=args.header,
        source_path=args.source,
    )
    write_text_file(args.lowering_rows, generate_lowering_rows())
    return 0


if __name__ == "__main__":
    sys.exit(main())
