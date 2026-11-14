# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native cooperative arithmetic selection rows, shared with packet typing."""

from __future__ import annotations

import argparse
from pathlib import Path

from loom.gen.support.c import CIdentifierCase, c_identifier
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.target.arch.spirv.cooperative_matrix import (
    COOPERATIVE_MATRIX_ARITHMETIC,
    COOPERATIVE_MATRIX_TYPES,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.extended_math import (
    COOPERATIVE_MATRIX_MATH_INSTRUCTIONS,
    COOPERATIVE_MATRIX_MATH_OPERATIONS,
)
from loom.target.low_descriptors import descriptor_set_relative_name


def generate_tables() -> str:
    descriptor_set = SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
    descriptors = {descriptor.key: descriptor for descriptor in descriptor_set.descriptors}
    sources = tuple(operation.source for operation in COOPERATIVE_MATRIX_ARITHMETIC) + tuple(operation.source_op_key for operation in COOPERATIVE_MATRIX_MATH_OPERATIONS)
    math_keys = {(row.operation.source_op_key, row.value_type): row.descriptor_key for row in COOPERATIVE_MATRIX_MATH_INSTRUCTIONS}
    required_flags = {operation.source_op_key: operation.required_fastmath_flags for operation in COOPERATIVE_MATRIX_MATH_OPERATIONS}
    lines = [
        *line_comment_header("//", generator="loom.gen.target.arch.spirv.spirv_cooperative_arithmetic"),
        "",
        "static const uint8_t kCooperativeArithmeticSlots[] = {",
    ]
    for index, source in enumerate(sources):
        lines.append(f"  [LOOM_OP_VECTOR_{source.upper()} - LOOM_OP_VECTOR_ADDF] = {index + 1},")
    lines.extend(("};", "", "static const uint8_t kCooperativeArithmeticRequiredFlags[] = {"))
    for source in sources:
        flags = " | ".join(f"LOOM_VECTOR_FASTMATHFLAGS_{flag.upper()}" for flag in required_flags.get(source, ())) or "0"
        lines.append(f"  {flags},")
    lines.extend(("};", "", "static const loom_spirv_cooperative_arithmetic_row_t kCooperativeArithmeticRows[] = {"))
    for matrix_type in COOPERATIVE_MATRIX_TYPES:
        keys = {operation.source: matrix_type.arithmetic_key(operation) for operation in COOPERATIVE_MATRIX_ARITHMETIC if operation.supports(matrix_type)}
        keys.update({source: key for (source, value_type), key in math_keys.items() if value_type == matrix_type})
        if not keys:
            continue
        lines.extend(
            (
                "  {",
                "    .type = {",
                "      .value_class = LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX,",
                f"      .scalar_type = {matrix_type.scalar.scalar_enum},",
                "      .cooperative_matrix = {",
                f"        .rows = {matrix_type.rows}, .columns = {matrix_type.columns},",
                f"        .scope = LOOM_SPIRV_SCOPE_SUBGROUP, .use = {matrix_type.use_c_expression},",
                "      },",
                "    },",
                "    .descriptors = {",
            )
        )
        for index, source in enumerate(sources):
            key = keys.get(source)
            if key is None:
                continue
            if key not in descriptors:
                raise ValueError(f"native arithmetic has no descriptor: {key}")
            if len(descriptors[key].feature_mask_words) != 1:
                raise ValueError(f"native arithmetic needs one feature word: {key}")
            suffix = c_identifier(descriptor_set_relative_name(descriptor_set, key), case=CIdentifierCase.UPPER)
            lines.append(f"      [{index}] = SPIRV_LOGICAL_CORE_DESCRIPTOR_REF_{suffix} + 1,")
        lines.extend(("    },", "  },"))
    lines.extend(
        (
            "};",
            f"static_assert(IREE_ARRAYSIZE(kCooperativeArithmeticRows[0].descriptors) == {len(sources)},",
            '              "cooperative arithmetic slots must match their generated rows");',
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True, type=Path)
    args = parser.parse_args()
    write_text_file(args.tables, generate_tables())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
