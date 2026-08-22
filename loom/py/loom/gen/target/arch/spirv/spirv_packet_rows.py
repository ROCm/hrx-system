# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: SPIR-V target-low descriptors -> compact packet emission rows."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import CIdentifierCase, c_identifier  # noqa: E402
from loom.gen.support.files import write_text_file  # noqa: E402
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
from loom.target.arch.spirv.ordinary_vector import (  # noqa: E402
    ORDINARY_VECTOR_INSTRUCTIONS,
    OrdinaryVectorComponentType,
    OrdinaryVectorInstructionType,
    OrdinaryVectorType,
)
from loom.target.arch.spirv.ordinary_vector_bit_layout import (  # noqa: E402
    ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer import (  # noqa: E402
    ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
)
from loom.target.arch.spirv.ordinary_vector_integer_conversion import (  # noqa: E402
    ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
)
from loom.target.arch.spirv.scalar_alu import (  # noqa: E402
    BOOLEAN_BINARY_OPERATIONS,
    BOOLEAN_CONSTANTS,
    FLOAT_BINARY_OPERATIONS,
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_BITWISE_BINARY_OPERATIONS,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    OFFSET64_COMPARE_PREDICATES,
    SCALAR_ALU_TYPES,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    SIGNED_INTEGER_SCALAR_ALU_TYPES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    BooleanConstant,
    IntegerAluTypePair,
    ScalarAluType,
    ScalarBinaryOperation,
)
from loom.target.arch.spirv.scalar_constant import (  # noqa: E402
    BFLOAT16_CONSTANT_TYPE,
    FLOAT_CONSTANT_TYPES,
    FloatConstantType,
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

_PACKET_MAX_OPERAND_COUNT = 4
_PACKET_OPERAND_TYPE_CAPACITY = 3


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
        ".cooperative_matrix = {"
        f".rows = {rows}, "
        f".columns = {columns}, "
        ".scope = LOOM_SPIRV_SCOPE_SUBGROUP, "
        f".use = {matrix_use}}}}}"
    )


def _scalar_value(scalar: StorageBufferScalar) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", scalar.scalar_enum)


def _alu_scalar_value(scalar: ScalarAluType) -> str:
    return _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", scalar.scalar_enum)


def _ordinary_vector_value(vector_type: OrdinaryVectorType) -> str:
    component_type = vector_type.component_type
    return f"{{.value_class = {component_type.vector_value_class}, .scalar_type = {component_type.scalar_enum}, .vector = {{.lane_count = {vector_type.lane_count}}}}}"


def _ordinary_vector_component_value(
    component_type: OrdinaryVectorComponentType,
) -> str:
    return _value_type(
        component_type.scalar_value_class,
        component_type.scalar_enum,
    )


def _ordinary_vector_instruction_value(
    value_type: OrdinaryVectorInstructionType,
) -> str:
    if isinstance(value_type, OrdinaryVectorType):
        return _ordinary_vector_value(value_type)
    return _ordinary_vector_component_value(value_type)


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


@dataclass(frozen=True, slots=True)
class _PacketRow:
    descriptor_key: str
    opcode: str
    form: str
    result_type: str | None = None
    operand_types: tuple[str, ...] = ()
    result_count: int = 0
    immediate_index: int | None = None
    literal_word_count: int = 0
    memory_alignment: int = 0
    builtin: str | None = None
    component_index: int | None = None
    execution_scope: str | None = None
    memory_scope: str | None = None
    memory_semantics: str | None = None
    cooperative_matrix_layout: str | None = None
    cooperative_matrix_byte_stride: int = 0
    cooperative_matrix_operands: str | None = None

    def encoded_operand_types(self) -> tuple[str, ...]:
        if len(self.operand_types) <= _PACKET_OPERAND_TYPE_CAPACITY:
            return self.operand_types
        if len(self.operand_types) <= _PACKET_MAX_OPERAND_COUNT and len(set(self.operand_types)) == 1:
            return self.operand_types[:1]
        raise ValueError(f"{self.descriptor_key}: packet operand types do not fit")

    def render(self) -> str:
        encoded_operand_types = self.encoded_operand_types()
        lines = [
            f"    [{_descriptor_ref_constant_name(self.descriptor_key)}] =",
            "        {",
            f"            .opcode = {self.opcode},",
            f"            .form = {self.form},",
            f"            .result_type = {self.result_type or _unknown_value()},",
        ]
        if encoded_operand_types:
            lines.append("            .operand_types =")
            lines.append("                {")
            lines.extend(f"                    {operand_type}," for operand_type in encoded_operand_types)
            lines.append("                },")
        lines.extend(
            [
                f"            .result_count = {self.result_count},",
                f"            .operand_count = {len(self.operand_types)},",
                "            .immediate_index = " + ("LOOM_SPIRV_PACKET_IMMEDIATE_NONE" if self.immediate_index is None else str(self.immediate_index)) + ",",
            ]
        )
        if self.literal_word_count:
            lines.append(f"            .literal_word_count = {self.literal_word_count},")
        if self.memory_alignment:
            lines.append(f"            .memory_alignment = {self.memory_alignment},")
        if self.builtin is not None:
            lines.append(f"            .builtin = {self.builtin},")
        if self.component_index is not None:
            lines.append(f"            .component_index = {self.component_index},")
        if self.execution_scope is not None:
            lines.append(f"            .execution_scope = {self.execution_scope},")
        if self.memory_scope is not None:
            lines.append(f"            .memory_scope = {self.memory_scope},")
        if self.memory_semantics is not None:
            lines.append(f"            .memory_semantics = {self.memory_semantics},")
        if self.cooperative_matrix_layout is not None:
            lines.append(f"            .cooperative_matrix_layout = {self.cooperative_matrix_layout},")
        if self.cooperative_matrix_byte_stride:
            lines.append(f"            .cooperative_matrix_byte_stride = {self.cooperative_matrix_byte_stride},")
        if self.cooperative_matrix_operands is not None:
            lines.append(f"            .cooperative_matrix_operands = {self.cooperative_matrix_operands},")
        lines.append("        },")
        return "\n".join(lines)


def _control_barrier_rows() -> list[_PacketRow]:
    memory_semantics = "LOOM_SPIRV_MEMORY_SEMANTICS_ACQUIRE_RELEASE_MASK | LOOM_SPIRV_MEMORY_SEMANTICS_WORKGROUP_MEMORY_MASK"
    return [
        _PacketRow(
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


def _storage_buffer_rows() -> list[_PacketRow]:
    rows: list[_PacketRow] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        rows.append(
            _PacketRow(
                f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset",
                opcode="LOOM_SPIRV_OP_CONVERT_U_TO_PTR",
                form="LOOM_SPIRV_PACKET_FORM_PHYSICAL_STORAGE_BUFFER_BYTE_OFFSET",
                result_type=_physical_storage_buffer_pointer_value(scalar),
                operand_types=(
                    _storage_buffer_address_value(),
                    _offset64_value(),
                ),
                result_count=1,
            )
        )
        rows.append(
            _PacketRow(
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
            _PacketRow(
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


def _workgroup_rows() -> list[_PacketRow]:
    rows: list[_PacketRow] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        rows.append(
            _PacketRow(
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
            _PacketRow(
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
            _PacketRow(
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


def _cooperative_matrix_rows_for_case(case: CooperativeMatrixCase) -> list[_PacketRow]:
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
    byte_pointer = case.byte_pointer_scalar
    lhs_byte_stride = _row_byte_stride(case.lhs_columns, lhs_scalar)
    rhs_byte_stride = _row_byte_stride(case.rhs_columns, rhs_scalar)
    accumulator_byte_stride = _row_byte_stride(case.accumulator_columns, accumulator_scalar)
    result_byte_stride = _row_byte_stride(case.accumulator_columns, result_scalar)
    return [
        _PacketRow(
            case.descriptor_key(
                "op_cooperative_matrix_load_khr",
                role="lhs",
                layout="row_major",
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LOAD_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD",
            result_type=lhs_value,
            operand_types=(_physical_storage_buffer_pointer_value(byte_pointer),),
            result_count=1,
            memory_alignment=16,
            cooperative_matrix_layout=row_major_layout,
            cooperative_matrix_byte_stride=lhs_byte_stride,
        ),
        _PacketRow(
            case.descriptor_key(
                "op_cooperative_matrix_load_khr",
                role="rhs",
                layout="row_major",
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LOAD_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD",
            result_type=rhs_value,
            operand_types=(_physical_storage_buffer_pointer_value(byte_pointer),),
            result_count=1,
            memory_alignment=16,
            cooperative_matrix_layout=row_major_layout,
            cooperative_matrix_byte_stride=rhs_byte_stride,
        ),
        _PacketRow(
            case.descriptor_key(
                "op_cooperative_matrix_load_khr",
                role="init",
                layout="row_major",
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LOAD_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD",
            result_type=accumulator_value,
            operand_types=(_physical_storage_buffer_pointer_value(byte_pointer),),
            result_count=1,
            memory_alignment=16,
            cooperative_matrix_layout=row_major_layout,
            cooperative_matrix_byte_stride=accumulator_byte_stride,
        ),
        _PacketRow(
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
        _PacketRow(
            case.descriptor_key(
                "op_cooperative_matrix_store_khr",
                role="result",
                layout="row_major",
            ),
            opcode="LOOM_SPIRV_OP_COOPERATIVE_MATRIX_STORE_KHR",
            form="LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_STORE",
            operand_types=(
                _physical_storage_buffer_pointer_value(byte_pointer),
                result_value,
            ),
            memory_alignment=16,
            cooperative_matrix_layout=row_major_layout,
            cooperative_matrix_byte_stride=result_byte_stride,
        ),
    ]


def _cooperative_matrix_rows() -> list[_PacketRow]:
    rows: list[_PacketRow] = []
    for case in COOPERATIVE_MATRIX_CASES:
        rows.extend(_cooperative_matrix_rows_for_case(case))
    return rows


def _scalar_binary_row(scalar: ScalarAluType, operation: ScalarBinaryOperation) -> _PacketRow:
    scalar_value = _alu_scalar_value(scalar)
    return _PacketRow(
        f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}",
        opcode=operation.opcode,
        form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
        result_type=scalar_value,
        operand_types=(scalar_value, scalar_value),
        result_count=1,
    )


def _integer_constant_row(scalar_pair: IntegerAluTypePair) -> _PacketRow:
    scalar = scalar_pair.signed
    return _PacketRow(
        f"spirv.op_constant.{scalar.suffix}",
        opcode="LOOM_SPIRV_OP_CONSTANT",
        form="LOOM_SPIRV_PACKET_FORM_SCALAR_CONSTANT",
        result_type=_alu_scalar_value(scalar),
        result_count=1,
        immediate_index=0,
        literal_word_count=scalar_pair.literal_word_count,
    )


def _float_constant_row(scalar: FloatConstantType) -> _PacketRow:
    return _PacketRow(
        f"spirv.op_constant.{scalar.suffix}",
        opcode="LOOM_SPIRV_OP_CONSTANT",
        form="LOOM_SPIRV_PACKET_FORM_SCALAR_CONSTANT",
        result_type=_value_type(
            "LOOM_SPIRV_VALUE_CLASS_SCALAR",
            scalar.scalar_enum,
        ),
        result_count=1,
        immediate_index=0,
        literal_word_count=scalar.literal_word_count,
    )


def _boolean_constant_row(row: BooleanConstant) -> _PacketRow:
    return _PacketRow(
        f"spirv.op_constant_{row.descriptor_suffix}.bool",
        opcode=row.opcode,
        form="LOOM_SPIRV_PACKET_FORM_BOOLEAN_CONSTANT",
        result_type=_bool_value(),
        result_count=1,
    )


def _boolean_binary_row(operation: ScalarBinaryOperation) -> _PacketRow:
    bool_value = _bool_value()
    return _PacketRow(
        f"spirv.op_{operation.descriptor_suffix}.bool",
        opcode=operation.opcode,
        form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
        result_type=bool_value,
        operand_types=(bool_value, bool_value),
        result_count=1,
    )


def _scalar_binary_rows() -> list[_PacketRow]:
    rows = [_scalar_binary_row(scalar, operation) for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES for operation in SIGNED_INTEGER_BINARY_OPERATIONS]
    rows.extend(_scalar_binary_row(scalar, operation) for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES for operation in INTEGER_BITWISE_BINARY_OPERATIONS)
    rows.extend(_scalar_binary_row(scalar_pair.unsigned, operation) for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS)
    rows.extend(_scalar_binary_row(scalar, operation) for scalar in FLOAT_SCALAR_ALU_TYPES for operation in FLOAT_BINARY_OPERATIONS)
    rows.extend(_boolean_binary_row(operation) for operation in BOOLEAN_BINARY_OPERATIONS)
    return rows


def _conversion_row(row: ScalarConversion) -> _PacketRow:
    return _PacketRow(
        row.key,
        opcode=row.opcode,
        form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
        result_type=_alu_scalar_value(row.result_type),
        operand_types=(_alu_scalar_value(row.source_type),),
        result_count=1,
    )


def _integer_value_view_row(row: IntegerValueViewConversion) -> _PacketRow:
    return _PacketRow(
        row.key,
        opcode="LOOM_SPIRV_OP_BITCAST",
        form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
        result_type=_alu_scalar_value(row.result_type),
        operand_types=(_alu_scalar_value(row.source_type),),
        result_count=1,
    )


def _conversion_rows() -> list[_PacketRow]:
    rows = [_conversion_row(row) for row in LOW_SCALAR_CONVERSIONS]
    rows.extend(_integer_value_view_row(row) for row in INTEGER_VALUE_VIEW_CONVERSIONS)
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS:
        scalar = scalar_pair.signed if scalar_pair.bit_width == 64 else scalar_pair.unsigned
        suffix = scalar.suffix
        opcode = "LOOM_SPIRV_OP_BITCAST" if scalar_pair.bit_width == 64 else "LOOM_SPIRV_OP_U_CONVERT"
        descriptor_opcode = "bitcast" if scalar_pair.bit_width == 64 else "uconvert"
        scalar_value = _alu_scalar_value(scalar)
        rows.append(
            _PacketRow(
                f"spirv.op_{descriptor_opcode}.{suffix}.offset64",
                opcode=opcode,
                form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
                result_type=_offset64_value(),
                operand_types=(scalar_value,),
                result_count=1,
            )
        )
        rows.append(
            _PacketRow(
                f"spirv.op_{descriptor_opcode}.offset64.{suffix}",
                opcode=opcode,
                form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
                result_type=scalar_value,
                operand_types=(_offset64_value(),),
                result_count=1,
            )
        )
    return rows


def _ordinary_vector_rows() -> list[_PacketRow]:
    return [
        _PacketRow(
            row.key,
            opcode=row.opcode,
            form=row.packet_form,
            result_type=_ordinary_vector_instruction_value(row.result_type),
            operand_types=tuple(_ordinary_vector_instruction_value(operand_type) for operand_type in row.operand_types),
            result_count=1,
            immediate_index=(0 if row.component_index_maximum is not None else None),
        )
        for row in (
            *ORDINARY_VECTOR_INSTRUCTIONS,
            *ORDINARY_VECTOR_INTEGER_INSTRUCTIONS,
            *ORDINARY_VECTOR_INTEGER_CONVERSION_INSTRUCTIONS,
            *ORDINARY_VECTOR_BIT_LAYOUT_INSTRUCTIONS,
        )
    ]


def _builtin_index_rows() -> list[_PacketRow]:
    return [
        _PacketRow(
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


def _coordinate_binary_rows() -> list[_PacketRow]:
    offset64_value = _offset64_value()
    rows = [
        _PacketRow(
            "spirv.op_iadd.offset64",
            opcode="LOOM_SPIRV_OP_I_ADD",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=offset64_value,
            operand_types=(offset64_value, offset64_value),
            result_count=1,
        ),
        _PacketRow(
            "spirv.op_isub.offset64",
            opcode="LOOM_SPIRV_OP_I_SUB",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=offset64_value,
            operand_types=(offset64_value, offset64_value),
            result_count=1,
        ),
        _PacketRow(
            "spirv.op_imul.offset64",
            opcode="LOOM_SPIRV_OP_I_MUL",
            form="LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE",
            result_type=offset64_value,
            operand_types=(offset64_value, offset64_value),
            result_count=1,
        ),
    ]
    return rows


def _coordinate_unary_rows() -> list[_PacketRow]:
    i32_value = _value_type(
        "LOOM_SPIRV_VALUE_CLASS_SCALAR",
        "LOOM_SPIRV_SCALAR_TYPE_S32",
    )
    return [
        _PacketRow(
            "spirv.op_copy_object.i32",
            opcode="LOOM_SPIRV_OP_COPY_OBJECT",
            form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
            result_type=i32_value,
            operand_types=(i32_value,),
            result_count=1,
        ),
        _PacketRow(
            "spirv.op_bit_count.i32",
            opcode="LOOM_SPIRV_OP_BIT_COUNT",
            form="LOOM_SPIRV_PACKET_FORM_UNARY_TYPED",
            result_type=i32_value,
            operand_types=(i32_value,),
            result_count=1,
        ),
    ]


def _mul_add_rows() -> list[_PacketRow]:
    i32_value = _value_type("LOOM_SPIRV_VALUE_CLASS_SCALAR", "LOOM_SPIRV_SCALAR_TYPE_S32")
    return [
        _PacketRow(
            "spirv.op_imul_add.i32",
            opcode="LOOM_SPIRV_OP_I_MUL",
            form="LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD",
            result_type=i32_value,
            operand_types=(i32_value, i32_value, i32_value),
            result_count=1,
        ),
    ]


def _integer_compare_rows() -> list[_PacketRow]:
    offset64_value = _offset64_value()
    bool_value = _bool_value()
    rows = [
        _PacketRow(
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
            _PacketRow(
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
            _PacketRow(
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


def _select_rows() -> list[_PacketRow]:
    offset64_value = _offset64_value()
    bool_value = _bool_value()
    rows = [
        _PacketRow(
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
    bf16_value = _value_type(
        "LOOM_SPIRV_VALUE_CLASS_SCALAR",
        BFLOAT16_CONSTANT_TYPE.scalar_enum,
    )
    rows.append(
        _PacketRow(
            "spirv.op_select.bf16",
            opcode="LOOM_SPIRV_OP_SELECT",
            form="LOOM_SPIRV_PACKET_FORM_SELECT",
            result_type=bf16_value,
            operand_types=(bool_value, bf16_value, bf16_value),
            result_count=1,
        )
    )
    rows.append(
        _PacketRow(
            "spirv.op_select.bool",
            opcode="LOOM_SPIRV_OP_SELECT",
            form="LOOM_SPIRV_PACKET_FORM_SELECT",
            result_type=bool_value,
            operand_types=(bool_value, bool_value, bool_value),
            result_count=1,
        )
    )
    rows.append(
        _PacketRow(
            "spirv.op_select.offset64",
            opcode="LOOM_SPIRV_OP_SELECT",
            form="LOOM_SPIRV_PACKET_FORM_SELECT",
            result_type=offset64_value,
            operand_types=(bool_value, offset64_value, offset64_value),
            result_count=1,
        )
    )
    return rows


_PACKETLESS_DESCRIPTOR_KEYS = frozenset(
    {
        # Deliberate low-verifier fixture for the missing-packet diagnostic.
        "spirv.op_variable.function.ptr",
    }
)


def _packet_rows() -> tuple[_PacketRow, ...]:
    return (
        *(_boolean_constant_row(row) for row in BOOLEAN_CONSTANTS),
        *(_integer_constant_row(scalar_pair) for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS),
        *(_float_constant_row(scalar) for scalar in FLOAT_CONSTANT_TYPES),
        _PacketRow(
            "spirv.op_constant.offset64",
            opcode="LOOM_SPIRV_OP_CONSTANT",
            form="LOOM_SPIRV_PACKET_FORM_SCALAR_CONSTANT",
            result_type=_offset64_value(),
            result_count=1,
            immediate_index=0,
            literal_word_count=2,
        ),
        *_scalar_binary_rows(),
        *_conversion_rows(),
        *_ordinary_vector_rows(),
        *_builtin_index_rows(),
        *_coordinate_binary_rows(),
        *_coordinate_unary_rows(),
        *_mul_add_rows(),
        *_integer_compare_rows(),
        *_select_rows(),
        *_storage_buffer_rows(),
        *_workgroup_rows(),
        *_control_barrier_rows(),
        *_cooperative_matrix_rows(),
    )


def _validate_rows(rows: tuple[_PacketRow, ...]) -> None:
    descriptor_keys = {descriptor.key for descriptor in SPIRV_LOGICAL_CORE_DESCRIPTOR_SET.descriptors}
    row_keys = tuple(row.descriptor_key for row in rows)
    row_key_counts = Counter(row_keys)
    duplicate_row_keys = sorted(key for key, count in row_key_counts.items() if count > 1)
    if duplicate_row_keys:
        raise ValueError("SPIR-V packet descriptor keys must be unique: " + ", ".join(duplicate_row_keys))

    unknown_packetless_keys = sorted(_PACKETLESS_DESCRIPTOR_KEYS - descriptor_keys)
    if unknown_packetless_keys:
        raise ValueError("SPIR-V packetless exceptions reference missing descriptors: " + ", ".join(unknown_packetless_keys))

    emitted_row_keys = set(row_keys)
    unknown_row_keys = sorted(emitted_row_keys - descriptor_keys)
    if unknown_row_keys:
        raise ValueError("SPIR-V packet rows reference missing descriptors: " + ", ".join(unknown_row_keys))

    emitted_packetless_keys = sorted(emitted_row_keys & _PACKETLESS_DESCRIPTOR_KEYS)
    if emitted_packetless_keys:
        raise ValueError("SPIR-V packetless descriptors must not have packet rows: " + ", ".join(emitted_packetless_keys))

    missing_row_keys = sorted(descriptor_keys - _PACKETLESS_DESCRIPTOR_KEYS - emitted_row_keys)
    if missing_row_keys:
        raise ValueError("SPIR-V descriptors are missing packet rows: " + ", ".join(missing_row_keys))

    over_capacity_rows = sorted(row.descriptor_key for row in rows if len(row.operand_types) > _PACKET_MAX_OPERAND_COUNT)
    if over_capacity_rows:
        raise ValueError(f"SPIR-V packet rows exceed the {_PACKET_MAX_OPERAND_COUNT}-operand maximum: " + ", ".join(over_capacity_rows))

    unencodable_type_rows = sorted(row.descriptor_key for row in rows if len(row.operand_types) > _PACKET_OPERAND_TYPE_CAPACITY and len(set(row.operand_types)) != 1)
    if unencodable_type_rows:
        raise ValueError("SPIR-V packet rows exceed the operand-type capacity without a repeated type: " + ", ".join(unencodable_type_rows))

    suffixes = [scalar.suffix for scalar in STORAGE_BUFFER_SCALARS]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError("SPIR-V storage-buffer scalar suffixes must be unique")


def generate_tables() -> str:
    rows = _packet_rows()
    _validate_rows(rows)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.spirv.spirv_packet_rows"),
        "",
        f'static_assert(LOOM_SPIRV_PACKET_MAX_OPERAND_COUNT == {_PACKET_MAX_OPERAND_COUNT}, "generated packet operand maximum");',
        f'static_assert(LOOM_SPIRV_PACKET_OPERAND_TYPE_CAPACITY == {_PACKET_OPERAND_TYPE_CAPACITY}, "generated packet operand-type capacity");',
        "",
        "static const loom_spirv_packet_row_t kSpirvLogicalCorePacketRows[] = {",
        *(row.render() for row in rows),
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
        write_text_file(args.tables, tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
