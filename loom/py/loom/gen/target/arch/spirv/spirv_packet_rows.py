# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: SPIR-V target-low descriptors -> compact packet emission rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import CIdentifierCase, c_identifier  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.spirv.builtins import (  # noqa: E402
    BUILTIN_DIMENSIONS,
    BUILTIN_INDEX_QUERIES,
)
from loom.target.arch.spirv.cooperative_matrix import (  # noqa: E402
    COOPERATIVE_MATRIX_CASES,
    CooperativeMatrixCase,
)
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET  # noqa: E402
from loom.target.arch.spirv.scalar_alu import (  # noqa: E402
    FLOAT_BINARY_OPERATIONS,
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    OFFSET64_ALU_TYPE,
    OFFSET64_COMPARE_PREDICATES,
    SCALAR_ALU_TYPES,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    SIGNED_INTEGER_SCALAR_ALU_TYPES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    ScalarAluType,
    ScalarBinaryOperation,
)
from loom.target.arch.spirv.scalar_conversion import (  # noqa: E402
    INTEGER_VALUE_VIEW_CONVERSIONS,
    LOW_SCALAR_CONVERSIONS,
    IntegerValueViewConversion,
    ScalarConversion,
)
from loom.target.arch.spirv.scalar_memory import (  # noqa: E402
    STORAGE_BUFFER_SCALARS,
    StorageBufferScalar,
)
from loom.target.low_descriptors import descriptor_set_relative_name  # noqa: E402


def _c_identifier(value: str) -> str:
    return c_identifier(value, case=CIdentifierCase.LOWER, empty="empty")


def _descriptor_ref_constant_name(descriptor_key: str) -> str:
    descriptor_name = descriptor_set_relative_name(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, descriptor_key)
    return f"{SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.c_enum_prefix}_DESCRIPTOR_REF_{_c_identifier(descriptor_name).upper()}"


def _value_type(
    value_class: str,
    scalar_enum: str = "LOOM_SPIRV_SCALAR_TYPE_UNKNOWN",
) -> str:
    return f"{{.value_class = {value_class}, .scalar_type = {scalar_enum}}}"


def _cooperative_matrix_value(
    *,
    scalar_enum: str,
    rows: int,
    columns: int,
    matrix_use: str,
) -> str:
    return (
        "{.value_class = LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX, "
        f".scalar_type = {scalar_enum}, "
        f".rows = {rows}, "
        f".columns = {columns}, "
        ".scope = LOOM_SPIRV_SCOPE_SUBGROUP, "
        f".cooperative_matrix_use = {matrix_use}}}"
    )


def _scalar_value(scalar: StorageBufferScalar) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", scalar.scalar_enum)


def _alu_scalar_value(scalar: ScalarAluType) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", scalar.scalar_enum)


def _storage_buffer_address_value() -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS")


def _physical_storage_buffer_pointer_value(scalar: StorageBufferScalar) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER", scalar.scalar_enum)


def _workgroup_pointer_value(scalar: StorageBufferScalar) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP", scalar.scalar_enum)


def _workgroup_array_pointer_value(scalar: StorageBufferScalar) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_PTR_WORKGROUP_ARRAY", scalar.scalar_enum)


def _offset64_value() -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_OFFSET64", "LOOM_SPIRV_SCALAR_TYPE_U64")


def _bool_value() -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_BOOL")


def _unknown_value() -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_UNKNOWN")


def _row(
    descriptor_key: str,
    *,
    opcode: str,
    form: str,
    result_type: str | None = None,
    operand_types: tuple[str, ...] = (),
    result_count: int = 0,
    immediate_index: int | None = None,
    literal_word_count: int = 0,
    memory_alignment: int = 0,
    builtin: str | None = None,
    component_index: int | None = None,
    execution_scope: str | None = None,
    memory_scope: str | None = None,
    memory_semantics: str | None = None,
    cooperative_matrix_layout: str | None = None,
    cooperative_matrix_stride: int = 0,
    cooperative_matrix_operands: str | None = None,
) -> str:
    lines = [
        f"    [{_descriptor_ref_constant_name(descriptor_key)}] =",
        "        {",
        f"            .opcode = {opcode},",
        f"            .form = {form},",
        f"            .result_type = {result_type or _unknown_value()},",
    ]
    if operand_types:
        lines.append("            .operand_types =")
        lines.append("                {")
        lines.extend(f"                    {operand_type}," for operand_type in operand_types)
        lines.append("                },")
    lines.extend(
        [
            f"            .result_count = {result_count},",
            f"            .operand_count = {len(operand_types)},",
            "            .immediate_index = " + ("LOOM_SPIRV_PACKET_IMMEDIATE_NONE" if immediate_index is None else str(immediate_index)) + ",",
        ]
    )
    if literal_word_count:
        lines.append(f"            .literal_word_count = {literal_word_count},")
    if memory_alignment:
        lines.append(f"            .memory_alignment = {memory_alignment},")
    if builtin is not None:
        lines.append(f"            .builtin = {builtin},")
    if component_index is not None:
        lines.append(f"            .component_index = {component_index},")
    if execution_scope is not None:
        lines.append(f"            .execution_scope = {execution_scope},")
    if memory_scope is not None:
        lines.append(f"            .memory_scope = {memory_scope},")
    if memory_semantics is not None:
        lines.append(f"            .memory_semantics = {memory_semantics},")
    if cooperative_matrix_layout is not None:
        lines.append(f"            .cooperative_matrix_layout = {cooperative_matrix_layout},")
    if cooperative_matrix_stride:
        lines.append(f"            .cooperative_matrix_stride = {cooperative_matrix_stride},")
    if cooperative_matrix_operands is not None:
        lines.append(f"            .cooperative_matrix_operands = {cooperative_matrix_operands},")
    lines.extend(
        [
            "        },",
        ]
    )
    return "\n".join(lines)


def _control_barrier_rows() -> list[str]:
    memory_semantics = "LOOM_SPIRV_MEMORY_SEMANTICS_ACQUIRE_RELEASE_MASK | LOOM_SPIRV_MEMORY_SEMANTICS_WORKGROUP_MEMORY_MASK"
    return [
        _row(
            f"spirv.op_control_barrier.{scope}.workgroup.acq_rel",
            opcode="LOOM_SPIRV_OP_CONTROL_BARRIER",
            form="LOOM_SPIRV_PACKET_FORM_CONTROL_BARRIER",
            execution_scope=scope_enum,
            memory_scope="LOOM_SPIRV_SCOPE_WORKGROUP",
            memory_semantics=memory_semantics,
        )
        for scope, scope_enum in (
            ("subgroup", "LOOM_SPIRV_SCOPE_SUBGROUP"),
            ("workgroup", "LOOM_SPIRV_SCOPE_WORKGROUP"),
        )
    ]


def _storage_buffer_rows() -> list[str]:
    rows: list[str] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        rows.append(
            _row(
                f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset",
                opcode="LOOM_SPIRV_OP_PTR_ACCESS_CHAIN",
                form="LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN",
                result_type=_physical_storage_buffer_pointer_value(scalar),
                operand_types=(
                    _storage_buffer_address_value(),
                    _offset64_value(),
                ),
                result_count=1,
            )
        )
        rows.append(
            _row(
                f"spirv.op_load.storage_buffer.{scalar.suffix}",
                opcode="LOOM_SPIRV_OP_LOAD",
                form="LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED",
                result_type=_scalar_value(scalar),
                operand_types=(_physical_storage_buffer_pointer_value(scalar),),
                result_count=1,
                memory_alignment=scalar.byte_width,
            )
        )
        rows.append(
            _row(
                f"spirv.op_store.storage_buffer.{scalar.suffix}",
                opcode="LOOM_SPIRV_OP_STORE",
                form="LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED",
                operand_types=(
                    _physical_storage_buffer_pointer_value(scalar),
                    _scalar_value(scalar),
                ),
                memory_alignment=scalar.byte_width,
            )
        )
    return rows


def _workgroup_rows() -> list[str]:
    rows: list[str] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        rows.append(
            _row(
                f"spirv.op_access_chain.workgroup.{scalar.suffix}.element_index",
                opcode="LOOM_SPIRV_OP_ACCESS_CHAIN",
                form="LOOM_SPIRV_PACKET_FORM_ACCESS_CHAIN",
                result_type=_workgroup_pointer_value(scalar),
                operand_types=(
                    _workgroup_array_pointer_value(scalar),
                    _value_type(
                        "LOOM_SPIRV_VALUE_CLASS_SCALAR",
                        "LOOM_SPIRV_SCALAR_TYPE_S32",
                    ),
                ),
                result_count=1,
            )
        )
        rows.append(
            _row(
                f"spirv.op_load.workgroup.{scalar.suffix}",
                opcode="LOOM_SPIRV_OP_LOAD",
                form="LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED",
                result_type=_scalar_value(scalar),
                operand_types=(_workgroup_pointer_value(scalar),),
                result_count=1,
                memory_alignment=scalar.byte_width,
            )
        )
        rows.append(
            _row(
                f"spirv.op_store.workgroup.{scalar.suffix}",
                opcode="LOOM_SPIRV_OP_STORE",
                form="LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED",
                operand_types=(
                    _workgroup_pointer_value(scalar),
                    _scalar_value(scalar),
                ),
                memory_alignment=scalar.byte_width,
            )
        )
    return rows


def _row_byte_stride(columns: int, scalar: StorageBufferScalar) -> int:
    return columns * scalar.byte_width


def _cooperative_matrix_rows_for_case(case: CooperativeMatrixCase) -> list[str]:
    lhs_scalar = case.lhs_scalar
    rhs_scalar = case.rhs_scalar
    accumulator_scalar = case.accumulator_scalar
    result_scalar = case.result_scalar
    lhs_value = _cooperative_matrix_value(
        scalar_enum=lhs_scalar.scalar_enum,
        rows=case.lhs_rows,
        columns=case.lhs_columns,
        matrix_use="LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR",
    )
    rhs_value = _cooperative_matrix_value(
        scalar_enum=rhs_scalar.scalar_enum,
        rows=case.rhs_rows,
        columns=case.rhs_columns,
        matrix_use="LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_BKHR",
    )
    accumulator_value = _cooperative_matrix_value(
        scalar_enum=accumulator_scalar.scalar_enum,
        rows=case.accumulator_rows,
        columns=case.accumulator_columns,
        matrix_use="LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_ACCUMULATOR_KHR",
    )
    result_value = _cooperative_matrix_value(
        scalar_enum=result_scalar.scalar_enum,
        rows=case.accumulator_rows,
        columns=case.accumulator_columns,
        matrix_use="LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_ACCUMULATOR_KHR",
    )
    row_major_layout = "LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR"
    lhs_stride = _row_byte_stride(case.lhs_columns, lhs_scalar)
    rhs_stride = _row_byte_stride(case.rhs_columns, rhs_scalar)
    accumulator_stride = _row_byte_stride(case.accumulator_columns, accumulator_scalar)
    result_stride = _row_byte_stride(case.accumulator_columns, result_scalar)
    return [
        _row(
            case.descriptor_key(
                "op_cooperative_matrix_load_khr",
                role="lhs",
                layout="row_major",
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LOAD_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD",
            result_type=lhs_value,
            operand_types=(_physical_storage_buffer_pointer_value(lhs_scalar),),
            result_count=1,
            memory_alignment=16,
            cooperative_matrix_layout=row_major_layout,
            cooperative_matrix_stride=lhs_stride,
        ),
        _row(
            case.descriptor_key(
                "op_cooperative_matrix_load_khr",
                role="rhs",
                layout="row_major",
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LOAD_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD",
            result_type=rhs_value,
            operand_types=(_physical_storage_buffer_pointer_value(rhs_scalar),),
            result_count=1,
            memory_alignment=16,
            cooperative_matrix_layout=row_major_layout,
            cooperative_matrix_stride=rhs_stride,
        ),
        _row(
            case.descriptor_key(
                "op_cooperative_matrix_load_khr",
                role="init",
                layout="row_major",
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LOAD_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD",
            result_type=accumulator_value,
            operand_types=(_physical_storage_buffer_pointer_value(accumulator_scalar),),
            result_count=1,
            memory_alignment=16,
            cooperative_matrix_layout=row_major_layout,
            cooperative_matrix_stride=accumulator_stride,
        ),
        _row(
            case.descriptor_key(
                "op_cooperative_matrix_mul_add_khr",
                role=None,
                include_operand_mode=True,
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_MUL_ADD_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_MUL_ADD",
            result_type=result_value,
            operand_types=(lhs_value, rhs_value, accumulator_value),
            result_count=1,
            cooperative_matrix_operands=case.packet_operand_mask,
        ),
        _row(
            case.descriptor_key(
                "op_cooperative_matrix_store_khr",
                role="result",
                layout="row_major",
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_STORE_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_STORE",
            operand_types=(
                _physical_storage_buffer_pointer_value(result_scalar),
                result_value,
            ),
            memory_alignment=16,
            cooperative_matrix_layout=row_major_layout,
            cooperative_matrix_stride=result_stride,
        ),
    ]


def _cooperative_matrix_rows() -> list[str]:
    rows: list[str] = []
    for case in COOPERATIVE_MATRIX_CASES:
        rows.extend(_cooperative_matrix_rows_for_case(case))
    return rows


def _scalar_binary_row(scalar: ScalarAluType, operation: ScalarBinaryOperation) -> str:
    scalar_value = _alu_scalar_value(scalar)
    return _row(
        f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}",
        opcode=operation.opcode,
        form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
        result_type=scalar_value,
        operand_types=(scalar_value, scalar_value),
        result_count=1,
    )


def _scalar_binary_rows() -> list[str]:
    rows = [_scalar_binary_row(scalar, operation) for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES for operation in SIGNED_INTEGER_BINARY_OPERATIONS]
    rows.extend(_scalar_binary_row(scalar_pair.unsigned, operation) for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS)
    rows.extend(_scalar_binary_row(scalar, operation) for scalar in FLOAT_SCALAR_ALU_TYPES for operation in FLOAT_BINARY_OPERATIONS)
    return rows


def _conversion_row(row: ScalarConversion) -> str:
    return _row(
        row.key,
        opcode=row.opcode,
        form="LOOM_SPIRV_PACKET_FORM_UNARY_CONVERT",
        result_type=_alu_scalar_value(row.result_type),
        operand_types=(_alu_scalar_value(row.source_type),),
        result_count=1,
    )


def _integer_value_view_row(row: IntegerValueViewConversion) -> str:
    return _row(
        row.key,
        opcode="LOOM_SPIRV_OP_BITCAST",
        form="LOOM_SPIRV_PACKET_FORM_UNARY_CONVERT",
        result_type=_alu_scalar_value(row.result_type),
        operand_types=(_alu_scalar_value(row.source_type),),
        result_count=1,
    )


def _conversion_rows() -> list[str]:
    rows = [_conversion_row(row) for row in LOW_SCALAR_CONVERSIONS]
    rows.extend(_integer_value_view_row(row) for row in INTEGER_VALUE_VIEW_CONVERSIONS)
    rows.append(
        _row(
            "spirv.op_uconvert.i32.offset64",
            opcode="LOOM_SPIRV_OP_U_CONVERT",
            form="LOOM_SPIRV_PACKET_FORM_UNARY_CONVERT",
            result_type=_offset64_value(),
            operand_types=(
                _value_type(
                    "LOOM_SPIRV_VALUE_CLASS_SCALAR",
                    "LOOM_SPIRV_SCALAR_TYPE_S32",
                ),
            ),
            result_count=1,
        )
    )
    return rows


def _builtin_index_rows() -> list[str]:
    return [
        _row(
            f"spirv.op_load_builtin.{query.descriptor_suffix}.{dimension.source_keyword}",
            opcode="LOOM_SPIRV_OP_LOAD",
            form="LOOM_SPIRV_PACKET_FORM_LOAD_BUILTIN",
            result_type=_value_type(
                "LOOM_SPIRV_VALUE_CLASS_SCALAR",
                "LOOM_SPIRV_SCALAR_TYPE_S32",
            ),
            result_count=1,
            builtin=query.builtin_enum,
            component_index=dimension.component_index,
        )
        for query in BUILTIN_INDEX_QUERIES
        for dimension in BUILTIN_DIMENSIONS
    ]


def _coordinate_binary_rows() -> list[str]:
    i32_value = _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", "LOOM_SPIRV_SCALAR_TYPE_S32")
    offset64_value = _offset64_value()
    rows = [
        _row(
            "spirv.op_shift_left_logical.i32",
            opcode="LOOM_SPIRV_OP_SHIFT_LEFT_LOGICAL",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=i32_value,
            operand_types=(i32_value, i32_value),
            result_count=1,
        ),
        _row(
            "spirv.op_iadd.offset64",
            opcode="LOOM_SPIRV_OP_I_ADD",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=offset64_value,
            operand_types=(offset64_value, offset64_value),
            result_count=1,
        ),
        _row(
            "spirv.op_isub.offset64",
            opcode="LOOM_SPIRV_OP_I_SUB",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=offset64_value,
            operand_types=(offset64_value, offset64_value),
            result_count=1,
        ),
    ]
    return rows


def _mul_add_rows() -> list[str]:
    i32_value = _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", "LOOM_SPIRV_SCALAR_TYPE_S32")
    return [
        _row(
            "spirv.op_imul_add.i32",
            opcode="LOOM_SPIRV_OP_I_MUL",
            form="LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD",
            result_type=i32_value,
            operand_types=(i32_value, i32_value, i32_value),
            result_count=1,
        ),
    ]


def _integer_compare_rows() -> list[str]:
    offset64_value = _offset64_value()
    bool_value = _bool_value()
    rows = [
        _row(
            f"spirv.op_{predicate.descriptor_suffix}.{scalar.suffix}",
            opcode=predicate.opcode,
            form="LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE",
            result_type=bool_value,
            operand_types=(_alu_scalar_value(scalar), _alu_scalar_value(scalar)),
            result_count=1,
        )
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for predicate in SIGNED_INTEGER_COMPARE_PREDICATES
    ]
    rows.extend(
        [
            _row(
                f"spirv.op_{predicate.descriptor_suffix}.{scalar_pair.unsigned.suffix}",
                opcode=predicate.opcode,
                form="LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE",
                result_type=bool_value,
                operand_types=(
                    _alu_scalar_value(scalar_pair.unsigned),
                    _alu_scalar_value(scalar_pair.unsigned),
                ),
                result_count=1,
            )
            for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
            for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES
        ]
    )
    rows.extend(
        [
            _row(
                f"spirv.op_{predicate.descriptor_suffix}.offset64",
                opcode=predicate.opcode,
                form="LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE",
                result_type=bool_value,
                operand_types=(offset64_value, offset64_value),
                result_count=1,
            )
            for predicate in OFFSET64_COMPARE_PREDICATES
        ]
    )
    return rows


def _select_rows() -> list[str]:
    offset64_value = _offset64_value()
    bool_value = _bool_value()
    rows = [
        _row(
            f"spirv.op_select.{scalar.suffix}",
            opcode="LOOM_SPIRV_OP_SELECT",
            form="LOOM_SPIRV_PACKET_FORM_SELECT",
            result_type=_alu_scalar_value(scalar),
            operand_types=(
                bool_value,
                _alu_scalar_value(scalar),
                _alu_scalar_value(scalar),
            ),
            result_count=1,
        )
        for scalar in SCALAR_ALU_TYPES
    ]
    rows.append(
        _row(
            "spirv.op_select.offset64",
            opcode="LOOM_SPIRV_OP_SELECT",
            form="LOOM_SPIRV_PACKET_FORM_SELECT",
            result_type=offset64_value,
            operand_types=(bool_value, offset64_value, offset64_value),
            result_count=1,
        )
    )
    return rows


def _validate_rows() -> None:
    descriptor_keys = {descriptor.key for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors}
    row_keys = {
        "spirv.op_constant.i32",
        "spirv.op_constant.offset64",
        "spirv.op_imul_add.i32",
        "spirv.op_uconvert.i32.offset64",
        "spirv.op_shift_left_logical.i32",
        "spirv.op_iadd.offset64",
        "spirv.op_isub.offset64",
        "spirv.op_select.offset64",
    }
    for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES:
        for operation in SIGNED_INTEGER_BINARY_OPERATIONS:
            row_keys.add(f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}")
        for predicate in SIGNED_INTEGER_COMPARE_PREDICATES:
            row_keys.add(f"spirv.op_{predicate.descriptor_suffix}.{scalar.suffix}")
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS:
            row_keys.add(f"spirv.op_{operation.descriptor_suffix}.{scalar_pair.unsigned.suffix}")
        for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES:
            row_keys.add(f"spirv.op_{predicate.descriptor_suffix}.{scalar_pair.unsigned.suffix}")
    for scalar in FLOAT_SCALAR_ALU_TYPES:
        for operation in FLOAT_BINARY_OPERATIONS:
            row_keys.add(f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}")
    for conversion in LOW_SCALAR_CONVERSIONS:
        row_keys.add(conversion.key)
    for view_conversion in INTEGER_VALUE_VIEW_CONVERSIONS:
        row_keys.add(view_conversion.key)
    for query in BUILTIN_INDEX_QUERIES:
        for dimension in BUILTIN_DIMENSIONS:
            row_keys.add(f"spirv.op_load_builtin.{query.descriptor_suffix}.{dimension.source_keyword}")
    for scalar in SCALAR_ALU_TYPES:
        row_keys.add(f"spirv.op_select.{scalar.suffix}")
    for predicate in OFFSET64_COMPARE_PREDICATES:
        row_keys.add(f"spirv.op_{predicate.descriptor_suffix}.{OFFSET64_ALU_TYPE.suffix}")
    for storage_scalar in STORAGE_BUFFER_SCALARS:
        row_keys.add(f"spirv.op_ptr_access_chain.storage_buffer.{storage_scalar.suffix}.byte_offset")
        row_keys.add(f"spirv.op_load.storage_buffer.{storage_scalar.suffix}")
        row_keys.add(f"spirv.op_store.storage_buffer.{storage_scalar.suffix}")
        row_keys.add(f"spirv.op_access_chain.workgroup.{storage_scalar.suffix}.element_index")
        row_keys.add(f"spirv.op_load.workgroup.{storage_scalar.suffix}")
        row_keys.add(f"spirv.op_store.workgroup.{storage_scalar.suffix}")
    row_keys.add("spirv.op_control_barrier.subgroup.workgroup.acq_rel")
    row_keys.add("spirv.op_control_barrier.workgroup.workgroup.acq_rel")
    for case in COOPERATIVE_MATRIX_CASES:
        for role in ("lhs", "rhs", "init"):
            row_keys.add(
                case.descriptor_key(
                    "op_cooperative_matrix_load_khr",
                    role=role,
                    layout="row_major",
                )
            )
        row_keys.add(
            case.descriptor_key(
                "op_cooperative_matrix_mul_add_khr",
                role=None,
                include_operand_mode=True,
            )
        )
        row_keys.add(
            case.descriptor_key(
                "op_cooperative_matrix_store_khr",
                role="result",
                layout="row_major",
            )
        )
    missing = sorted(row_keys - descriptor_keys)
    if missing:
        raise ValueError("SPIR-V packet rows reference missing descriptors: " + ", ".join(missing))
    suffixes = [scalar.suffix for scalar in STORAGE_BUFFER_SCALARS]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError("SPIR-V storage-buffer scalar suffixes must be unique")


def generate_tables() -> str:
    _validate_rows()
    i32_value = _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", "LOOM_SPIRV_SCALAR_TYPE_S32")
    rows = [
        _row(
            "spirv.op_constant.i32",
            opcode="LOOM_SPIRV_OP_CONSTANT",
            form="LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT",
            result_type=i32_value,
            result_count=1,
            immediate_index=0,
            literal_word_count=1,
        ),
        _row(
            "spirv.op_constant.offset64",
            opcode="LOOM_SPIRV_OP_CONSTANT",
            form="LOOM_SPIRV_PACKET_FORM_INTEGER_CONSTANT",
            result_type=_offset64_value(),
            result_count=1,
            immediate_index=0,
            literal_word_count=2,
        ),
        *_scalar_binary_rows(),
        *_conversion_rows(),
        *_builtin_index_rows(),
        *_coordinate_binary_rows(),
        *_mul_add_rows(),
        *_integer_compare_rows(),
        *_select_rows(),
        *_storage_buffer_rows(),
        *_workgroup_rows(),
        *_control_barrier_rows(),
        *_cooperative_matrix_rows(),
    ]
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.spirv.spirv_packet_rows"),
        "",
        "static const loom_spirv_packet_row_t kSpirvLogicalCorePacketRows[] = {",
        *rows,
        "};",
        "",
    ]
    return "\n".join(lines)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_arguments(argv)
    tables = generate_tables()
    if args.check:
        return 0
    if args.tables is None:
        sys.stdout.write(tables)
    else:
        args.tables.write_text(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
