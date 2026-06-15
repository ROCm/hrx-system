# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: SPIR-V cooperative property facts -> compact C tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import CIdentifierCase, c_identifier  # noqa: E402
from loom.gen.support.c import c_string_literal as _c_string_literal  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.spirv.cooperative_matrix import (  # noqa: E402
    COOPERATIVE_MATRIX_CASES,
    CooperativeMatrixCase,
)
from loom.target.arch.spirv.cooperative_vector import (  # noqa: E402
    COOPERATIVE_VECTOR_CASES,
    CooperativeVectorCase,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET  # noqa: E402
from loom.target.low_descriptors import descriptor_set_relative_name  # noqa: E402


def _c_identifier(value: str) -> str:
    return c_identifier(value, case=CIdentifierCase.LOWER, empty="empty")


def _descriptor_ref_constant_name(descriptor_key: str) -> str:
    descriptor_name = descriptor_set_relative_name(
        SPIRV_LOGICAL_CORE_DESCRIPTOR_SET,
        descriptor_key,
    )
    descriptor_suffix = _c_identifier(descriptor_name).upper()
    return f"{SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.c_enum_prefix}_DESCRIPTOR_REF_{descriptor_suffix}"


def _mul_add_descriptor_key(case: CooperativeMatrixCase) -> str:
    return case.descriptor_key(
        "op_cooperative_matrix_mul_add_khr",
        include_operand_mode=True,
    )


class _ShapeKeyCase(Protocol):
    @property
    def shape_key(self) -> int: ...


def _matrix_property_row(case: CooperativeMatrixCase) -> list[str]:
    return [
        "    {",
        f'        .name = IREE_SVL("{_c_string_literal(case.property_name)}"),',
        f"        .required_feature_bits = {case.feature_bits_c_expression},",
        f"        .m_size = {case.m_size},",
        f"        .n_size = {case.n_size},",
        f"        .k_size = {case.k_size},",
        f"        .lhs_type = {case.lhs_scalar.scalar_enum},",
        f"        .rhs_type = {case.rhs_scalar.scalar_enum},",
        f"        .accumulator_type = {case.accumulator_scalar.scalar_enum},",
        f"        .result_type = {case.result_scalar.scalar_enum},",
        "        .scope = LOOM_SPIRV_SCOPE_SUBGROUP,",
        "        .layout_flags = MATRIX_LAYOUT_ANY,",
        f"        .storage_class_flags = {case.property_storage_flags},",
        f"        .operand_flags = {case.property_operand_flags},",
        (f"        .mul_add_descriptor_ref = {_descriptor_ref_constant_name(_mul_add_descriptor_key(case))},"),
        "    },",
    ]


def _vector_property_row(case: CooperativeVectorCase) -> list[str]:
    return [
        "    {",
        f'        .name = IREE_SVL("{_c_string_literal(case.property_name)}"),',
        f"        .required_feature_bits = {case.feature_bits_c_expression},",
        f"        .m_size = {case.m_size},",
        f"        .k_size = {case.k_size},",
        f"        .input_type = {case.input_type},",
        f"        .input_interpretation = {case.input_interpretation},",
        f"        .matrix_interpretation = {case.matrix_interpretation},",
        f"        .bias_interpretation = {case.bias_interpretation},",
        f"        .result_type = {case.result_type},",
        f"        .matrix_layout_flags = {case.matrix_layout_flags},",
        f"        .storage_class_flags = {case.storage_class_flags},",
        f"        .flags = {case.flags},",
        "    },",
    ]


def _shape_spans(
    cases: Sequence[_ShapeKeyCase],
) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    previous_shape_key: int | None = None
    for index, case in enumerate(cases):
        if case.shape_key != previous_shape_key:
            spans.append((case.shape_key, index, 1))
            previous_shape_key = case.shape_key
        else:
            shape_key, start, count = spans[-1]
            spans[-1] = (shape_key, start, count + 1)
    return spans


def _shape_span_row(shape_key: int, start: int, count: int, hex_digits: int) -> str:
    return f"    {{.shape_key = UINT64_C(0x{shape_key:0{hex_digits}x}), .start = {start}, .count = {count}}},"


def _validate_matrix_cases(cases: Sequence[CooperativeMatrixCase]) -> None:
    descriptor_keys = {descriptor.key for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors}
    names: set[str] = set()
    previous_shape_key = 0
    for index, case in enumerate(cases):
        if case.property_name in names:
            raise ValueError(f"duplicate SPIR-V cooperative matrix property {case.property_name!r}")
        names.add(case.property_name)
        if index != 0 and case.shape_key < previous_shape_key:
            raise ValueError("SPIR-V cooperative matrix property rows must be sorted")
        descriptor_key = _mul_add_descriptor_key(case)
        if descriptor_key not in descriptor_keys:
            raise ValueError(f"SPIR-V cooperative matrix property references missing descriptor: {descriptor_key}")
        previous_shape_key = case.shape_key


def _validate_vector_cases(cases: Sequence[CooperativeVectorCase]) -> None:
    names: set[str] = set()
    previous_shape_key = 0
    for index, case in enumerate(cases):
        if case.property_name in names:
            raise ValueError(f"duplicate SPIR-V cooperative vector property {case.property_name!r}")
        names.add(case.property_name)
        if index != 0 and case.shape_key < previous_shape_key:
            raise ValueError("SPIR-V cooperative vector property rows must be sorted")
        if "LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING" in case.flags and "cooperative_vector_training_nv" not in case.feature_atoms:
            raise ValueError(f"training cooperative vector property {case.property_name!r} is missing the training feature")
        previous_shape_key = case.shape_key


def generate_tables() -> str:
    _validate_matrix_cases(COOPERATIVE_MATRIX_CASES)
    _validate_vector_cases(COOPERATIVE_VECTOR_CASES)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.spirv.spirv_cooperative_properties"),
        "",
        ("static const loom_spirv_cooperative_matrix_property_t kCooperativeMatrixProperties[] = {"),
    ]
    for matrix_case in COOPERATIVE_MATRIX_CASES:
        lines.extend(_matrix_property_row(matrix_case))
    lines.extend(
        [
            "};",
            "",
            ("static const loom_spirv_cooperative_property_span_t kCooperativeMatrixShapeSpans[] = {"),
        ]
    )
    for shape_key, start, count in _shape_spans(COOPERATIVE_MATRIX_CASES):
        lines.append(_shape_span_row(shape_key, start, count, 12))
    lines.extend(
        [
            "};",
            "",
            ("static const loom_spirv_cooperative_vector_property_t kCooperativeVectorProperties[] = {"),
        ]
    )
    for vector_case in COOPERATIVE_VECTOR_CASES:
        lines.extend(_vector_property_row(vector_case))
    lines.extend(
        [
            "};",
            "",
            ("static const loom_spirv_cooperative_property_span_t kCooperativeVectorShapeSpans[] = {"),
        ]
    )
    for shape_key, start, count in _shape_spans(COOPERATIVE_VECTOR_CASES):
        lines.append(_shape_span_row(shape_key, start, count, 8))
    lines.extend(
        [
            "};",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args(argv)
    args.tables.write_text(generate_tables(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
