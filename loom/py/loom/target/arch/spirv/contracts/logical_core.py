# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V logical source-to-low contract fragment."""

from __future__ import annotations

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.kernel import ALL_KERNEL_OPS
from loom.dialect.kernel import defs as kernel
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.scf import defs as scf
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.arch.spirv.barrier import (
    CONTROL_BARRIER_CASES,
    ControlBarrierCase,
)
from loom.target.arch.spirv.builtins import (
    BUILTIN_DIMENSIONS,
    BUILTIN_INDEX_QUERIES,
    BUILTIN_SCALAR_INDEX_QUERIES,
    BuiltinDimension,
    BuiltinQuery,
)
from loom.target.arch.spirv.contracts.atomic import SPIRV_ATOMIC_CONTRACT_CASES
from loom.target.arch.spirv.contracts.bfloat import (
    bfloat_carrier_from_i16_rule,
    bfloat_carrier_memory_rules,
    bfloat_carrier_to_i16_rule,
    bfloat_narrow_rules,
    bfloat_widen_rules,
)
from loom.target.arch.spirv.contracts.cooperative_matrix import (
    COOPERATIVE_INITIALIZER_MATERIALIZER,
    COOPERATIVE_MATRIX_INITIALIZER_RULES,
    COOPERATIVE_MATRIX_MEMORY_RULES,
)
from loom.target.arch.spirv.contracts.descriptor_rule import (
    descriptor_feature_guards as _feature_guards,
)
from loom.target.arch.spirv.contracts.descriptor_rule import (
    emit_descriptor_op as _descriptor_emit,
)
from loom.target.arch.spirv.contracts.descriptor_rule import (
    logical_core_descriptor as _descriptor,
)
from loom.target.arch.spirv.contracts.extended_math import (
    SPIRV_EXTENDED_MATH_CONTRACT_CASES,
)
from loom.target.arch.spirv.contracts.extrema import extrema_rules
from loom.target.arch.spirv.contracts.float8 import float8_narrow_rules
from loom.target.arch.spirv.contracts.float_width import (
    float_width_narrow_rules,
    float_width_widen_rules,
)
from loom.target.arch.spirv.contracts.index import (
    SPIRV_INDEX_CONVERSION_RULES,
    SPIRV_INDEX_NUMERIC_RULES,
)
from loom.target.arch.spirv.contracts.integer_bits import (
    bit_count_rules,
    rotation_rules,
)
from loom.target.arch.spirv.contracts.memory import (
    STORAGE_BUFFER_MEMORY_SPACES as _STORAGE_BUFFER_MEMORY_SPACES,
)
from loom.target.arch.spirv.contracts.memory import (
    WORKGROUP_MEMORY_SPACES as _WORKGROUP_MEMORY_SPACES,
)
from loom.target.arch.spirv.contracts.memory import (
    source_memory_address_feature_guards as _source_memory_address_feature_guards,
)
from loom.target.arch.spirv.contracts.memory import (
    storage_buffer_address_materializer as _storage_buffer_address_materializer,
)
from loom.target.arch.spirv.contracts.memory import (
    storage_buffer_source_memory as _storage_buffer_source_memory,
)
from loom.target.arch.spirv.contracts.memory import (
    storage_element_format_guards as _storage_element_format_guards,
)
from loom.target.arch.spirv.contracts.memory import (
    workgroup_address_materializer as _workgroup_address_materializer,
)
from loom.target.arch.spirv.contracts.memory import (
    workgroup_carrier_guard as _workgroup_carrier_guard,
)
from loom.target.arch.spirv.contracts.memory import (
    workgroup_source_memory as _workgroup_source_memory,
)
from loom.target.arch.spirv.contracts.ordinary_vector import (
    SPIRV_ORDINARY_VECTOR_CONTRACT_CASES,
)
from loom.target.arch.spirv.contracts.ordinary_vector_alu import (
    SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES,
)
from loom.target.arch.spirv.contracts.predicate import integer_to_boolean_rule
from loom.target.arch.spirv.contracts.reference_fallback import (
    SPIRV_REFERENCE_FALLBACK_CONTRACT_CASES,
)
from loom.target.arch.spirv.contracts.rounding import rounding_rules
from loom.target.arch.spirv.contracts.subgroup import SPIRV_SUBGROUP_CONTRACT_CASES
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.ordinary_vector import NATIVE_ORDINARY_VECTOR_LANE_COUNTS
from loom.target.arch.spirv.ordinary_vector_conversion import (
    NATIVE_VECTOR_SCALAR_CONVERSIONS,
)
from loom.target.arch.spirv.scalar_alu import (
    BFLOAT16_SCALAR_TYPE,
    BOOLEAN_BINARY_OPERATIONS,
    BOOLEAN_CONSTANTS,
    FLOAT_BINARY_OPERATIONS,
    FLOAT_COMPARE_PREDICATES,
    FLOAT_COMPARE_TYPES,
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_BITWISE_BINARY_OPERATIONS,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    SCALAR_ALU_TYPES,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    SIGNED_INTEGER_SCALAR_ALU_TYPES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    BooleanConstant,
    ComparePredicate,
    IntegerAluTypePair,
    ScalarAluType,
    ScalarBinaryOperation,
)
from loom.target.arch.spirv.scalar_constant import (
    FLOAT_CONSTANT_TYPES,
    FloatConstantType,
)
from loom.target.arch.spirv.scalar_conversion import (
    DIRECT_SCALAR_CONVERSIONS,
    UNSIGNED_SCALAR_CONVERSIONS,
    ScalarConversion,
)
from loom.target.arch.spirv.scalar_memory import (
    RAW_STORAGE_BUFFER_BYTE,
    SOURCE_STORAGE_BUFFER_SCALARS,
    STORAGE_BUFFER_SCALARS,
    StorageBufferScalar,
    storage_buffer_scalar_by_suffix,
)
from loom.target.contracts import (
    AttrProject,
    Buffer,
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorMatrixRule,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ResultTypeBinding,
    Scalar,
    SourceMemoryAddressCoordinateType,
    SourceMemoryOperation,
    SourceValueKind,
    TypePattern,
    ValueAliasRule,
    ValueElideRule,
    ValueProject,
    ValueRef,
    Vector,
    View,
)

_I1 = Scalar("i1")
_I8 = Scalar("i8")
_I32 = Scalar("i32")
_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_SIGNED_INTEGER_ALU_TYPE_BY_BIT_WIDTH = {
    scalar_pair.bit_width: scalar_pair.signed
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
}
_FLOAT_ALU_TYPE_BY_SOURCE_TYPE = {
    scalar.source_type: scalar
    for scalar in (*FLOAT_SCALAR_ALU_TYPES, BFLOAT16_SCALAR_TYPE)
}


def _typed_guards(
    fields: tuple[str, ...],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _integer_constant_rule(scalar_pair: IntegerAluTypePair) -> DescriptorRule:
    scalar = scalar_pair.signed
    descriptor = _descriptor(f"spirv.op_constant.{scalar.suffix}")
    immediate_name = f"{scalar_pair.source_type}_value"
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", Scalar(scalar_pair.source_type)),
            Guard.i64_range(
                "value",
                scalar_pair.signed_minimum,
                scalar_pair.signed_maximum,
            ),
            *_feature_guards(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={immediate_name: AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _float_constant_rule(scalar: FloatConstantType) -> DescriptorRule:
    descriptor = _descriptor(f"spirv.op_constant.{scalar.suffix}")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", Scalar(scalar.source_type)),
            Guard.value_exact_float("result"),
            *_feature_guards(descriptor),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={
                    f"{scalar.source_type}_bits": ValueProject.float_bits("result")
                },
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _boolean_constant_rule(row: BooleanConstant) -> DescriptorRule:
    descriptor = _descriptor(f"spirv.op_constant_{row.descriptor_suffix}.bool")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", _I1),
            Guard.value_i64_range("result", row.value, row.value),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _float_payload_constant_rule(
    source_op: Op, result_pattern: TypePattern, carrier_suffix: str
) -> DescriptorRule:
    # Raw FP8 payloads span 0..255, beyond the signed-i8 constant immediate.
    # An i32 literal followed by integer narrowing preserves every payload bit.
    constant = _descriptor("spirv.op_constant.i32")
    result = ValueRef.result("result")
    payload = result if carrier_suffix == "i32" else ValueRef.temporary("payload")
    emits = [
        EmitDescriptorOp(
            descriptor=constant,
            results={"dst": payload},
            result_types={"dst": _I32},
            immediates={"i32_value": ValueProject.float_bits("result")},
            form=DescriptorEmitForm.CONST,
        )
    ]
    if carrier_suffix != "i32":
        emits.append(
            _descriptor_emit(
                descriptor=_descriptor(f"spirv.op_s_convert.i32.{carrier_suffix}"),
                operands={"input": payload},
                results={"dst": result},
            )
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", result_pattern),
            Guard.value_exact_float("result"),
            *_feature_guards(*(step.descriptor for step in emits)),
        ),
        emit=tuple(emits),
    )


def _scalar_constant_rules() -> tuple[DescriptorRule, ...]:
    return (
        tuple(_boolean_constant_rule(row) for row in BOOLEAN_CONSTANTS)
        + tuple(
            _integer_constant_rule(scalar_pair)
            for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        )
        + tuple(_float_constant_rule(scalar) for scalar in FLOAT_CONSTANT_TYPES)
        + (
            _float_payload_constant_rule(
                scalar_conversion.scalar_constant,
                Scalar(("f8E4M3", "f8E5M2")),
                "i8",
            ),
            _float_payload_constant_rule(
                scalar_conversion.scalar_constant, Scalar("bf16"), "i32"
            ),
        )
    )


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _numeric_type_pattern(
    scalar: ScalarAluType, lane_count: int | None = None
) -> TypePattern:
    if lane_count is None:
        return Scalar(scalar.source_type)
    return Vector(scalar.source_type, lanes=lane_count)


def _value_type_suffix(scalar: ScalarAluType, lane_count: int | None) -> str:
    if lane_count is None or lane_count == 1:
        return scalar.suffix
    return f"v{lane_count}{scalar.suffix}"


def _conversion_rule(
    row: ScalarConversion, lane_count: int | None = None
) -> DescriptorRule:
    source_suffix = _value_type_suffix(row.source_type, lane_count)
    result_suffix = _value_type_suffix(row.result_type, lane_count)
    descriptor = _descriptor(
        f"spirv.op_{row.descriptor_suffix}.{source_suffix}.{result_suffix}"
    )
    emits: list[EmitDescriptorOp] = []
    input_ref = ValueRef.operand("input")
    result_ref = ValueRef.result("result")
    if row.source_op_key in ("extui", "uitofp"):
        source_pair = _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE[row.source_type.source_type]
        input_ref = ValueRef.temporary("unsigned_input")
        emits.append(
            _integer_view_emit(
                source_type=source_pair.signed,
                result_type=source_pair.unsigned,
                input_ref=ValueRef.operand("input"),
                output_ref=input_ref,
                lane_count=lane_count,
            )
        )
    if row.source_op_key in ("extui", "fptoui"):
        result_ref = ValueRef.temporary("unsigned_result")
    result_types = None
    if result_ref.kind == SourceValueKind.TEMPORARY:
        result_types = {"dst": DescriptorResultType()}
    emits.append(
        _descriptor_emit(
            descriptor=descriptor,
            operands={"input": input_ref},
            results={"dst": result_ref},
            result_types=result_types,
        )
    )
    if row.source_op_key in ("extui", "fptoui"):
        result_pair = _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE[row.result_type.source_type]
        emits.append(
            _integer_view_emit(
                source_type=result_pair.unsigned,
                result_type=result_pair.signed,
                input_ref=result_ref,
                output_ref=ValueRef.result("result"),
                lane_count=lane_count,
            )
        )
    return DescriptorRule(
        source_op=(
            _CONVERSION_SOURCE_OPS[row.source_op_key]
            if lane_count is None
            else getattr(vector, f"vector_{row.source_op_key}")
        ),
        descriptor=descriptor,
        guards=(
            Guard.value_type(
                "input", _numeric_type_pattern(row.source_type, lane_count)
            ),
            Guard.value_type(
                "result", _numeric_type_pattern(row.result_type, lane_count)
            ),
            *_feature_guards(descriptor),
        ),
        emit=tuple(emits),
    )


def _conversion_alias_rule(
    source_op: Op,
    input_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=source_op,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
        ),
    )


def _builtin_index_rule(
    query: BuiltinQuery,
    dimension: BuiltinDimension,
) -> DescriptorRule:
    descriptor = _descriptor(
        f"spirv.op_load_builtin.{query.descriptor_suffix}.{dimension.source_keyword}"
    )
    return DescriptorRule(
        source_op=getattr(kernel, f"kernel_{query.source_op_key}"),
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("dimension", dimension.source_keyword),
            Guard.value_type("result", _INDEX),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _builtin_index_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _builtin_index_rule(query, dimension)
        for query in BUILTIN_INDEX_QUERIES
        for dimension in BUILTIN_DIMENSIONS
    )


def _builtin_scalar_index_rule(query: BuiltinQuery) -> DescriptorRule:
    descriptor = _descriptor(f"spirv.op_load_builtin.{query.descriptor_suffix}")
    return DescriptorRule(
        source_op=getattr(kernel, f"kernel_{query.source_op_key}"),
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", _INDEX),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _builtin_scalar_index_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _builtin_scalar_index_rule(query) for query in BUILTIN_SCALAR_INDEX_QUERIES
    )


def _bitcast_key(
    source_type: ScalarAluType,
    result_type: ScalarAluType,
    lane_count: int | None = None,
) -> str:
    source_suffix = _value_type_suffix(source_type, lane_count)
    result_suffix = _value_type_suffix(result_type, lane_count)
    return f"spirv.op_bitcast.{source_suffix}.{result_suffix}"


def _integer_view_emit(
    *,
    source_type: ScalarAluType,
    result_type: ScalarAluType,
    input_ref: ValueRef,
    output_ref: ValueRef,
    lane_count: int | None = None,
) -> EmitDescriptorOp:
    result_type_binding: ResultTypeBinding = _numeric_type_pattern(
        result_type, lane_count
    )
    if output_ref.kind == SourceValueKind.TEMPORARY:
        result_type_binding = DescriptorResultType()
    return _descriptor_emit(
        descriptor=_descriptor(_bitcast_key(source_type, result_type, lane_count)),
        operands={"input": input_ref},
        results={"dst": output_ref},
        result_types={"dst": result_type_binding},
    )


def _unsigned_binary_rule(
    scalar_pair: IntegerAluTypePair,
    operation: ScalarBinaryOperation,
) -> DescriptorRule:
    descriptor = _descriptor(
        f"spirv.op_{operation.descriptor_suffix}.{scalar_pair.unsigned.suffix}"
    )
    type_pattern = Scalar(scalar_pair.source_type)
    return DescriptorRule(
        source_op=_INTEGER_BINARY_SOURCE_OPS[operation.source_op_key],
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _integer_view_emit(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
                input_ref=ValueRef.operand("lhs"),
                output_ref=ValueRef.temporary("unsigned_lhs"),
            ),
            _integer_view_emit(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
                input_ref=ValueRef.operand("rhs"),
                output_ref=ValueRef.temporary("unsigned_rhs"),
            ),
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.temporary("unsigned_lhs"),
                    "rhs": ValueRef.temporary("unsigned_rhs"),
                },
                results={"dst": ValueRef.temporary("unsigned_result")},
                result_types={"dst": DescriptorResultType()},
            ),
            _integer_view_emit(
                source_type=scalar_pair.unsigned,
                result_type=scalar_pair.signed,
                input_ref=ValueRef.temporary("unsigned_result"),
                output_ref=ValueRef.result("result"),
            ),
        ),
    )


def _compare_rule(
    source_op: Op,
    predicate: ComparePredicate,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), type_pattern),
            Guard.value_type("result", _I1),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _unsigned_compare_rule(
    source_op: Op,
    predicate: ComparePredicate,
    scalar_pair: IntegerAluTypePair,
) -> DescriptorRule:
    descriptor = _descriptor(
        f"spirv.op_{predicate.descriptor_suffix}.{scalar_pair.unsigned.suffix}"
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), Scalar(scalar_pair.source_type)),
            Guard.value_type("result", _I1),
            *_feature_guards(descriptor),
        ),
        emit=(
            _integer_view_emit(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
                input_ref=ValueRef.operand("lhs"),
                output_ref=ValueRef.temporary("unsigned_lhs"),
            ),
            _integer_view_emit(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
                input_ref=ValueRef.operand("rhs"),
                output_ref=ValueRef.temporary("unsigned_rhs"),
            ),
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.temporary("unsigned_lhs"),
                    "rhs": ValueRef.temporary("unsigned_rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _select_rule(
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "condition": ValueRef.operand("condition"),
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _storage_source_types(scalar: StorageBufferScalar) -> tuple[str, ...]:
    # FP8 values use the signed-byte carrier for both registers and memory.
    if scalar.source_type == "i8":
        return ("i8", "f8E4M3", "f8E5M2")
    return (scalar.source_type,)


def _buffer_view_rule(
    scalar: StorageBufferScalar,
    *,
    require_storage_element_format: bool = False,
) -> ValueElideRule:
    view_type = View(_storage_source_types(scalar))
    return ValueElideRule(
        source_op=buffer.buffer_view,
        values=(ValueRef.result("result"),),
        guards=(
            Guard.value_type("result", view_type),
            *(
                _storage_element_format_guards("result", scalar)
                if require_storage_element_format
                else ()
            ),
        ),
    )


def _raw_storage_buffer_byte_load_rule() -> DescriptorRule:
    scalar = RAW_STORAGE_BUFFER_BYTE
    address_descriptor = _descriptor(
        f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset"
    )
    load_descriptor = _descriptor(f"spirv.op_load.storage_buffer.{scalar.suffix}")
    extend_descriptor = _descriptor(f"spirv.op_uconvert.{scalar.suffix}.u32")
    i32_pair = _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE["i32"]
    return DescriptorRule(
        source_op=buffer.buffer_load_i8_u,
        descriptor=load_descriptor,
        guards=(
            Guard.value_type("byte_offset", _OFFSET),
            Guard.value_type("result", _I32),
            *_feature_guards(load_descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=address_descriptor,
                operands={
                    "base": ValueRef.operand("source"),
                    "byte_offset": ValueRef.operand("byte_offset"),
                },
                results={"ptr": ValueRef.temporary("byte_address")},
                result_types={"ptr": DescriptorResultType()},
            ),
            _descriptor_emit(
                descriptor=load_descriptor,
                operands={"ptr": ValueRef.temporary("byte_address")},
                results={"dst": ValueRef.temporary("byte")},
                result_types={"dst": DescriptorResultType()},
            ),
            _descriptor_emit(
                descriptor=extend_descriptor,
                operands={"input": ValueRef.temporary("byte")},
                results={"dst": ValueRef.temporary("unsigned_result")},
                result_types={"dst": DescriptorResultType()},
            ),
            _integer_view_emit(
                source_type=i32_pair.unsigned,
                result_type=i32_pair.signed,
                input_ref=ValueRef.temporary("unsigned_result"),
                output_ref=ValueRef.result("result"),
            ),
        ),
    )


def _raw_storage_buffer_byte_store_rule() -> DescriptorRule:
    scalar = RAW_STORAGE_BUFFER_BYTE
    address_descriptor = _descriptor(
        f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset"
    )
    narrow_descriptor = _descriptor(f"spirv.op_uconvert.u32.{scalar.suffix}")
    store_descriptor = _descriptor(f"spirv.op_store.storage_buffer.{scalar.suffix}")
    i32_pair = _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE["i32"]
    return DescriptorRule(
        source_op=buffer.buffer_store_i8,
        descriptor=store_descriptor,
        guards=(
            Guard.value_type("value", _I32),
            Guard.value_type("byte_offset", _OFFSET),
            *_feature_guards(store_descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=address_descriptor,
                operands={
                    "base": ValueRef.operand("target"),
                    "byte_offset": ValueRef.operand("byte_offset"),
                },
                results={"ptr": ValueRef.temporary("byte_address")},
                result_types={"ptr": DescriptorResultType()},
            ),
            _integer_view_emit(
                source_type=i32_pair.signed,
                result_type=i32_pair.unsigned,
                input_ref=ValueRef.operand("value"),
                output_ref=ValueRef.temporary("unsigned_value"),
            ),
            _descriptor_emit(
                descriptor=narrow_descriptor,
                operands={"input": ValueRef.temporary("unsigned_value")},
                results={"dst": ValueRef.temporary("byte")},
                result_types={"dst": DescriptorResultType()},
            ),
            _descriptor_emit(
                descriptor=store_descriptor,
                operands={
                    "ptr": ValueRef.temporary("byte_address"),
                    "value": ValueRef.temporary("byte"),
                },
            ),
        ),
    )


def _storage_subview_rule(scalar: StorageBufferScalar) -> ValueElideRule:
    return ValueElideRule(
        source_op=view.view_subview,
        values=(ValueRef.result("result"),),
        guards=(
            Guard.value_type("result", View(_storage_source_types(scalar))),
            Guard.value_memory_space("result", _STORAGE_BUFFER_MEMORY_SPACES),
        ),
    )


def _workgroup_subview_rule(scalar: StorageBufferScalar) -> ValueAliasRule:
    view_type = View(_storage_source_types(scalar))
    return ValueAliasRule(
        source_op=view.view_subview,
        source=ValueRef.operand("source"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("source", view_type),
            Guard.value_type("result", view_type),
            Guard.value_memory_space("result", _WORKGROUP_MEMORY_SPACES),
        ),
    )


def _float_integer_carrier(
    scalar: StorageBufferScalar,
) -> tuple[ScalarAluType, ScalarAluType, StorageBufferScalar]:
    float_type = _FLOAT_ALU_TYPE_BY_SOURCE_TYPE.get(scalar.source_type)
    integer_type = _SIGNED_INTEGER_ALU_TYPE_BY_BIT_WIDTH.get(scalar.byte_width * 8)
    if float_type is None or integer_type is None:
        raise ValueError(
            f"missing same-width integer carrier for '{scalar.source_type}'"
        )
    integer_storage_scalar = storage_buffer_scalar_by_suffix(integer_type.suffix)
    if integer_storage_scalar is None:
        raise ValueError(f"missing SPIR-V storage scalar '{integer_type.suffix}'")
    return float_type, integer_type, integer_storage_scalar


def _view_load_rule(scalar: StorageBufferScalar) -> DescriptorRule:
    scalar_type = Scalar(_storage_source_types(scalar))
    view_type = View(_storage_source_types(scalar))
    descriptor = _descriptor(f"spirv.op_load.storage_buffer.{scalar.suffix}")
    address_materializer = _storage_buffer_address_materializer(scalar)
    return DescriptorRule(
        source_op=view.view_load,
        descriptor=descriptor,
        guards=(
            Guard.value_type("view", view_type),
            Guard.value_type("result", scalar_type),
            *_source_memory_address_feature_guards(address_materializer),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={"ptr": ValueRef.source_memory_address()},
                results={"dst": ValueRef.result("result")},
                source_memory=_storage_buffer_source_memory(
                    SourceMemoryOperation.LOAD,
                    scalar,
                ),
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _view_store_rule(scalar: StorageBufferScalar) -> DescriptorRule:
    scalar_type = Scalar(_storage_source_types(scalar))
    view_type = View(_storage_source_types(scalar))
    descriptor = _descriptor(f"spirv.op_store.storage_buffer.{scalar.suffix}")
    address_materializer = _storage_buffer_address_materializer(scalar)
    return DescriptorRule(
        source_op=view.view_store,
        descriptor=descriptor,
        guards=(
            Guard.value_type("view", view_type),
            Guard.value_type("value", scalar_type),
            *_source_memory_address_feature_guards(address_materializer),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "ptr": ValueRef.source_memory_address(),
                    "value": ValueRef.operand("value"),
                },
                source_memory=_storage_buffer_source_memory(
                    SourceMemoryOperation.STORE,
                    scalar,
                ),
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _view_load_workgroup_integer_carrier_rule(
    scalar: StorageBufferScalar,
    coordinate_type: SourceMemoryAddressCoordinateType,
) -> DescriptorRule:
    float_type, integer_type, integer_scalar = _float_integer_carrier(scalar)
    descriptor = _descriptor(f"spirv.op_load.workgroup.{integer_scalar.suffix}")
    bitcast_descriptor = _descriptor(_bitcast_key(integer_type, float_type))
    address_materializer = _workgroup_address_materializer(
        integer_scalar, coordinate_type
    )
    return DescriptorRule(
        source_op=view.view_load,
        descriptor=descriptor,
        guards=(
            Guard.value_type("view", View(scalar.source_type)),
            Guard.value_type("result", Scalar(scalar.source_type)),
            _workgroup_carrier_guard(integer_scalar.suffix),
            *_source_memory_address_feature_guards(address_materializer),
            *_feature_guards(descriptor),
            *_feature_guards(bitcast_descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={"ptr": ValueRef.source_memory_address()},
                results={"dst": ValueRef.temporary("integer_value")},
                result_types={"dst": DescriptorResultType()},
                source_memory=_workgroup_source_memory(
                    SourceMemoryOperation.LOAD,
                    scalar,
                ),
                source_memory_address_materializer=address_materializer,
            ),
            _integer_view_emit(
                source_type=integer_type,
                result_type=float_type,
                input_ref=ValueRef.temporary("integer_value"),
                output_ref=ValueRef.result("result"),
            ),
        ),
    )


def _view_store_workgroup_integer_carrier_rule(
    scalar: StorageBufferScalar,
    coordinate_type: SourceMemoryAddressCoordinateType,
) -> DescriptorRule:
    float_type, integer_type, integer_scalar = _float_integer_carrier(scalar)
    descriptor = _descriptor(f"spirv.op_store.workgroup.{integer_scalar.suffix}")
    bitcast_descriptor = _descriptor(_bitcast_key(float_type, integer_type))
    address_materializer = _workgroup_address_materializer(
        integer_scalar, coordinate_type
    )
    return DescriptorRule(
        source_op=view.view_store,
        descriptor=descriptor,
        guards=(
            Guard.value_type("view", View(scalar.source_type)),
            Guard.value_type("value", Scalar(scalar.source_type)),
            _workgroup_carrier_guard(integer_scalar.suffix),
            *_source_memory_address_feature_guards(address_materializer),
            *_feature_guards(descriptor),
            *_feature_guards(bitcast_descriptor),
        ),
        emit=(
            _integer_view_emit(
                source_type=float_type,
                result_type=integer_type,
                input_ref=ValueRef.operand("value"),
                output_ref=ValueRef.temporary("integer_value"),
            ),
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "ptr": ValueRef.source_memory_address(),
                    "value": ValueRef.temporary("integer_value"),
                },
                source_memory=_workgroup_source_memory(
                    SourceMemoryOperation.STORE,
                    scalar,
                ),
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _view_load_workgroup_rule(
    scalar: StorageBufferScalar,
    coordinate_type: SourceMemoryAddressCoordinateType,
) -> DescriptorRule:
    scalar_type = Scalar(_storage_source_types(scalar))
    view_type = View(_storage_source_types(scalar))
    descriptor = _descriptor(f"spirv.op_load.workgroup.{scalar.suffix}")
    address_materializer = _workgroup_address_materializer(scalar, coordinate_type)
    return DescriptorRule(
        source_op=view.view_load,
        descriptor=descriptor,
        guards=(
            Guard.value_type("view", view_type),
            Guard.value_type("result", scalar_type),
            _workgroup_carrier_guard(scalar.suffix),
            *_source_memory_address_feature_guards(address_materializer),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={"ptr": ValueRef.source_memory_address()},
                results={"dst": ValueRef.result("result")},
                source_memory=_workgroup_source_memory(
                    SourceMemoryOperation.LOAD,
                    scalar,
                ),
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _view_store_workgroup_rule(
    scalar: StorageBufferScalar,
    coordinate_type: SourceMemoryAddressCoordinateType,
) -> DescriptorRule:
    scalar_type = Scalar(_storage_source_types(scalar))
    view_type = View(_storage_source_types(scalar))
    descriptor = _descriptor(f"spirv.op_store.workgroup.{scalar.suffix}")
    address_materializer = _workgroup_address_materializer(scalar, coordinate_type)
    return DescriptorRule(
        source_op=view.view_store,
        descriptor=descriptor,
        guards=(
            Guard.value_type("view", view_type),
            Guard.value_type("value", scalar_type),
            _workgroup_carrier_guard(scalar.suffix),
            *_source_memory_address_feature_guards(address_materializer),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "ptr": ValueRef.source_memory_address(),
                    "value": ValueRef.operand("value"),
                },
                source_memory=_workgroup_source_memory(
                    SourceMemoryOperation.STORE,
                    scalar,
                ),
                source_memory_address_materializer=address_materializer,
            ),
        ),
    )


def _storage_buffer_rules() -> tuple[ContractCase, ...]:
    rules: list[ContractCase] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        if scalar.source_rule_enabled or scalar.numeric_format_c_expression is None:
            continue
        rules.append(
            _buffer_view_rule(
                scalar,
                require_storage_element_format=True,
            )
        )
    for scalar in SOURCE_STORAGE_BUFFER_SCALARS:
        rules.append(_buffer_view_rule(scalar))
        rules.append(_storage_subview_rule(scalar))
        rules.append(_workgroup_subview_rule(scalar))
        rules.append(_view_load_rule(scalar))
        rules.append(_view_store_rule(scalar))
        for coordinate_type in (
            SourceMemoryAddressCoordinateType.INDEX,
            SourceMemoryAddressCoordinateType.OFFSET,
        ):
            rules.append(_view_load_workgroup_rule(scalar, coordinate_type))
            rules.append(_view_store_workgroup_rule(scalar, coordinate_type))
            if scalar.source_type in _FLOAT_ALU_TYPE_BY_SOURCE_TYPE:
                rules.append(
                    _view_load_workgroup_integer_carrier_rule(scalar, coordinate_type)
                )
                rules.append(
                    _view_store_workgroup_integer_carrier_rule(scalar, coordinate_type)
                )
    return tuple(rules)


def _control_barrier_rule(case: ControlBarrierCase) -> DescriptorRule:
    descriptor = _descriptor(case.descriptor_key)
    return DescriptorRule(
        source_op=kernel.kernel_barrier,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("memory_space", case.memory_space.source_keyword),
            Guard.enum_attr_equals("ordering", case.ordering.source_keyword),
            Guard.enum_attr_equals("scope", case.scope.source_keyword),
        ),
        emit=(_descriptor_emit(descriptor=descriptor),),
    )


def _control_barrier_rules() -> tuple[DescriptorRule, ...]:
    return tuple(_control_barrier_rule(case) for case in CONTROL_BARRIER_CASES)


_INTEGER_BINARY_SOURCE_OPS = {
    "addi": scalar_arithmetic.scalar_addi,
    "subi": scalar_arithmetic.scalar_subi,
    "muli": scalar_arithmetic.scalar_muli,
    "divui": scalar_arithmetic.scalar_divui,
    "divsi": scalar_arithmetic.scalar_divsi,
    "remui": scalar_arithmetic.scalar_remui,
    "remsi": scalar_arithmetic.scalar_remsi,
}

_BITWISE_BINARY_SOURCE_OPS = {
    "andi": scalar_bitwise.scalar_andi,
    "ori": scalar_bitwise.scalar_ori,
    "xori": scalar_bitwise.scalar_xori,
    "shli": scalar_bitwise.scalar_shli,
    "shrsi": scalar_bitwise.scalar_shrsi,
    "shrui": scalar_bitwise.scalar_shrui,
}

_FLOAT_BINARY_SOURCE_OPS = {
    "addf": scalar_arithmetic.scalar_addf,
    "subf": scalar_arithmetic.scalar_subf,
    "mulf": scalar_arithmetic.scalar_mulf,
    "divf": scalar_arithmetic.scalar_divf,
    "remf": scalar_arithmetic.scalar_remf,
}

_CONVERSION_SOURCE_OPS = {
    "sitofp": scalar_conversion.scalar_sitofp,
    "uitofp": scalar_conversion.scalar_uitofp,
    "fptosi": scalar_conversion.scalar_fptosi,
    "fptoui": scalar_conversion.scalar_fptoui,
    "extf": scalar_conversion.scalar_extf,
    "fptrunc": scalar_conversion.scalar_fptrunc,
    "extsi": scalar_conversion.scalar_extsi,
    "extui": scalar_conversion.scalar_extui,
    "trunci": scalar_conversion.scalar_trunci,
    "bitcast": scalar_conversion.scalar_bitcast,
}

_INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE = {
    scalar_pair.source_type: scalar_pair
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
}


def _scalar_binary_rule(
    scalar: ScalarAluType,
    operation: ScalarBinaryOperation,
    source_ops: dict[str, Op],
) -> DescriptorRule:
    return _binary_rule(
        source_ops[operation.source_op_key],
        _numeric_type_pattern(scalar),
        f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}",
    )


def _scalar_binary_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _scalar_binary_rule(scalar, operation, _INTEGER_BINARY_SOURCE_OPS)
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for operation in SIGNED_INTEGER_BINARY_OPERATIONS
    ]
    rules.extend(
        _scalar_binary_rule(scalar, operation, _BITWISE_BINARY_SOURCE_OPS)
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for operation in INTEGER_BITWISE_BINARY_OPERATIONS
    )
    rules.extend(
        _unsigned_binary_rule(scalar_pair, operation)
        for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS
    )
    rules.extend(
        _scalar_binary_rule(scalar, operation, _FLOAT_BINARY_SOURCE_OPS)
        for scalar in FLOAT_SCALAR_ALU_TYPES
        for operation in FLOAT_BINARY_OPERATIONS
    )
    rules.extend(
        _binary_rule(
            _BITWISE_BINARY_SOURCE_OPS[operation.source_op_key],
            _I1,
            f"spirv.op_{operation.descriptor_suffix}.bool",
        )
        for operation in BOOLEAN_BINARY_OPERATIONS
    )
    return tuple(rules)


def _vector_float_binary_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _binary_rule(
            getattr(vector, f"vector_{operation.source_op_key}"),
            Vector(scalar.source_type, lanes=lane_count),
            f"spirv.op_{operation.descriptor_suffix}."
            + (scalar.suffix if lane_count == 1 else f"v{lane_count}{scalar.suffix}"),
        )
        for scalar in FLOAT_SCALAR_ALU_TYPES
        for lane_count in (1, 2, 3, 4)
        for operation in FLOAT_BINARY_OPERATIONS
    )


def _vector_float_scale_rules() -> tuple[DescriptorRule, ...]:
    rules = []
    for scalar in FLOAT_SCALAR_ALU_TYPES:
        for lane_count in NATIVE_ORDINARY_VECTOR_LANE_COUNTS:
            descriptor = _descriptor(
                f"spirv.op_vector_times_scalar.v{lane_count}{scalar.suffix}"
            )
            vector_type = Vector(scalar.source_type, lanes=lane_count)
            for broadcast, lanes in (("rhs", "lhs"), ("lhs", "rhs")):
                rules.append(
                    DescriptorRule(
                        source_op=vector.vector_mulf,
                        descriptor=descriptor,
                        guards=(
                            *_typed_guards(("lhs", "rhs", "result"), vector_type),
                            *_feature_guards(descriptor),
                            Guard.uniform_element_origin_type(
                                broadcast, Scalar(scalar.source_type)
                            ),
                        ),
                        emit=(
                            _descriptor_emit(
                                descriptor=descriptor,
                                operands={
                                    "vector": ValueRef.operand(lanes),
                                    "scalar": ValueRef.uniform_element_origin_operand(
                                        broadcast
                                    ),
                                },
                                results={"dst": ValueRef.result("result")},
                            ),
                        ),
                    )
                )
    return tuple(rules)


def _conversion_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _conversion_rule(row)
        for row in DIRECT_SCALAR_CONVERSIONS
        if not (
            row.source_op_key == "fptrunc" and row.result_type.source_type == "bf16"
        )
    ]
    rules.extend(float_width_narrow_rules())
    rules.extend(bfloat_narrow_rules())
    rules.extend(bfloat_widen_rules())
    rules.append(bfloat_carrier_from_i16_rule())
    rules.append(bfloat_carrier_to_i16_rule())
    rules.extend(float8_narrow_rules())
    rules.extend(_conversion_rule(row) for row in UNSIGNED_SCALAR_CONVERSIONS)
    rules.extend(
        _conversion_rule(row, lane_count)
        for row in NATIVE_VECTOR_SCALAR_CONVERSIONS
        for lane_count in (1, *NATIVE_ORDINARY_VECTOR_LANE_COUNTS)
    )
    rules.extend(float_width_widen_rules())
    rules.extend(
        integer_to_boolean_rule(scalar_conversion.scalar_trunci, scalar.source_type)
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
    )
    return tuple(rules)


def _compare_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _compare_rule(
            scalar_comparison.scalar_cmpi,
            predicate,
            _numeric_type_pattern(scalar),
            f"spirv.op_{predicate.descriptor_suffix}.{scalar.suffix}",
        )
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for predicate in SIGNED_INTEGER_COMPARE_PREDICATES
    ]
    rules.extend(
        _unsigned_compare_rule(
            scalar_comparison.scalar_cmpi,
            predicate,
            scalar_pair,
        )
        for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES
    )
    rules.extend(
        _compare_rule(
            scalar_comparison.scalar_cmpf,
            predicate,
            _numeric_type_pattern(scalar),
            f"spirv.op_{predicate.descriptor_suffix}.{scalar.suffix}",
        )
        for scalar in FLOAT_COMPARE_TYPES
        for predicate in FLOAT_COMPARE_PREDICATES
    )
    return tuple(rules)


def _select_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _select_rule(_numeric_type_pattern(scalar), f"spirv.op_select.{scalar.suffix}")
        for scalar in SCALAR_ALU_TYPES
    ]
    rules.append(_select_rule(Scalar("bf16"), "spirv.op_select.bf16"))
    rules.append(_select_rule(Scalar(("f8E4M3", "f8E5M2")), "spirv.op_select.i8"))
    rules.append(_select_rule(Scalar("bf16"), "spirv.op_select.i32"))
    rules.append(_select_rule(_I1, "spirv.op_select.bool"))
    rules.append(_select_rule(Buffer(), "spirv.op_select.storage_buffer"))
    return tuple(rules)


SPIRV_LOGICAL_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "kernel": ALL_KERNEL_OPS,
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="spirv.logical.core",
    descriptor_set=SPIRV_LOGICAL_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/spirv/contracts/logical_core.h",
    materializers=(COOPERATIVE_INITIALIZER_MATERIALIZER,),
    cases=(
        *_scalar_constant_rules(),
        *SPIRV_INDEX_CONVERSION_RULES,
        *SPIRV_INDEX_NUMERIC_RULES,
        *_builtin_index_rules(),
        *_builtin_scalar_index_rules(),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F8E4M3, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I8, _F8E4M3),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F8E5M2, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I8, _F8E5M2),
        *_conversion_rules(),
        *_scalar_binary_rules(),
        *_vector_float_scale_rules(),
        *_vector_float_binary_rules(),
        *SPIRV_EXTENDED_MATH_CONTRACT_CASES,
        *extrema_rules(),
        *rounding_rules(),
        *bit_count_rules(),
        *rotation_rules(),
        *SPIRV_ORDINARY_VECTOR_CONTRACT_CASES,
        # Singleton vectors use the scalar physical carrier. Native BF16 rows
        # precede these software alternatives just as for scalar constants.
        _float_payload_constant_rule(
            vector.vector_constant, Vector(("f8E4M3", "f8E5M2"), lanes=1), "i8"
        ),
        _float_payload_constant_rule(
            vector.vector_constant, Vector("bf16", lanes=1), "i32"
        ),
        _select_rule(Vector("bf16", lanes=1), "spirv.op_select.i32"),
        *SPIRV_ORDINARY_VECTOR_ALU_CONTRACT_CASES,
        *_compare_rules(),
        *_select_rules(),
        _raw_storage_buffer_byte_load_rule(),
        _raw_storage_buffer_byte_store_rule(),
        *_storage_buffer_rules(),
        *bfloat_carrier_memory_rules(),
        *SPIRV_ATOMIC_CONTRACT_CASES,
        *SPIRV_SUBGROUP_CONTRACT_CASES,
        *_control_barrier_rules(),
        *COOPERATIVE_MATRIX_MEMORY_RULES,
        *COOPERATIVE_MATRIX_INITIALIZER_RULES,
        DescriptorMatrixRule(
            source_op=vector.vector_mma,
            source="vector_mma",
        ),
        *SPIRV_REFERENCE_FALLBACK_CONTRACT_CASES,
    ),
)
