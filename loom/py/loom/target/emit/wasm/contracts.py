# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wasm source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Iterable

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.error.wasm import (
    ERR_WASM_001,
    ERR_WASM_002,
    ERR_WASM_003,
    ERR_WASM_004,
    ERR_WASM_005,
    ERR_WASM_006,
)
from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryRootKind,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    Vector,
    descriptor_by_key,
    i64_param,
    string_param,
    target_diagnostic,
    value_type_param,
)
from loom.target.contracts.templates import (
    ReductionDescriptorCase,
    reduction_descriptor_rules,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_F32 = Scalar("f32")
_I1 = Scalar("i1")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_V4I1 = Vector("i1", lanes=4)
_V4I32 = Vector("i32", lanes=4)
_V4F32 = Vector("f32", lanes=4)

_I64_ATTR_DIAGNOSTIC = GuardDiagnostic(
    ref=target_diagnostic(
        ERR_WASM_002,
        string_param("field_name", "value"),
    ),
)
_I32_BIT_PATTERN_MIN = -(2**31)
_I32_BIT_PATTERN_MAX = (2**32) - 1

_I32_CONSTANT_RANGE_DIAGNOSTIC = GuardDiagnostic(
    ref=target_diagnostic(
        ERR_WASM_003,
        string_param("field_name", "value"),
        i64_param("minimum", _I32_BIT_PATTERN_MIN),
        i64_param("maximum", _I32_BIT_PATTERN_MAX),
    ),
)
_ZERO_BUFFER_VIEW_OFFSET_DIAGNOSTIC = GuardDiagnostic(
    ref=target_diagnostic(
        ERR_WASM_004,
        string_param("field_name", "byte_offset"),
    ),
)
_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    ref=target_diagnostic(ERR_WASM_005),
)
_WASM32_ADDRESS_DIAGNOSTIC = GuardDiagnostic(
    ref=target_diagnostic(
        ERR_WASM_006,
        i64_param("bit_count", 32),
    ),
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, key)


def _type_text(type_pattern: TypePattern) -> str:
    if type_pattern == _I32:
        return "i32 scalar"
    if type_pattern == _F32:
        return "f32 scalar"
    if type_pattern == _I1:
        return "i1 scalar"
    if type_pattern in (_INDEX, _OFFSET):
        return "index or offset scalar"
    if type_pattern == _V4I1:
        return "vector<4xi1>"
    if type_pattern == _V4I32:
        return "vector<4xi32>"
    if type_pattern == _V4F32:
        return "vector<4xf32>"
    raise ValueError(f"unknown Wasm type pattern: {type_pattern!r}")


def _type_diagnostic(field: str, type_pattern: TypePattern) -> GuardDiagnostic:
    return GuardDiagnostic(
        ref=target_diagnostic(
            ERR_WASM_001,
            string_param("field_name", field),
            value_type_param("actual_type", field),
            string_param("required_type", _type_text(type_pattern)),
        )
    )


def _value_type(field: str, type_pattern: TypePattern) -> Guard:
    return Guard.value_type(
        field,
        type_pattern,
        diagnostic=_type_diagnostic(field, type_pattern),
    )


def _typed_guards(
    fields: Iterable[str],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(_value_type(field, type_pattern) for field in fields)


def _const_i32_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("wasm.i32.const")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
            _value_type("result", result_type),
            Guard.i64_range(
                "value",
                _I32_BIT_PATTERN_MIN,
                _I32_BIT_PATTERN_MAX,
                diagnostic=_I32_CONSTANT_RANGE_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i32_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
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
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _splat_rule() -> DescriptorRule:
    descriptor = _descriptor("wasm.i32x4.splat")
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            _value_type("scalar", _I32),
            _value_type("result", _V4I32),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _select_rule(value_type: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("wasm.v128.bitselect")
    return DescriptorRule(
        source_op=vector.vector_select,
        descriptor=descriptor,
        guards=(
            _value_type("condition", _V4I1),
            _value_type("true_value", value_type),
            _value_type("false_value", value_type),
            _value_type("result", value_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                    "condition": ValueRef.operand("condition"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _compare_rule(
    source_op: Op,
    predicate: str,
    operand_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            _value_type("lhs", operand_type),
            _value_type("rhs", operand_type),
            _value_type("result", _V4I1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _scalar_compare_rule(
    source_op: Op,
    predicate: str,
    operand_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            _value_type("lhs", operand_type),
            _value_type("rhs", operand_type),
            _value_type("result", _I1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _extract_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            _value_type("source", source_type),
            _value_type("result", result_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 0, 3),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "lane": AttrProject.i64_array_element(
                        "static_indices",
                        element=0,
                    )
                },
            ),
        ),
    )


def _insert_rule(
    value_type: TypePattern,
    dest_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            _value_type("value", value_type),
            _value_type("dest", dest_type),
            _value_type("result", dest_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 0, 3),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "dest": ValueRef.operand("dest"),
                    "value": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "lane": AttrProject.i64_array_element(
                        "static_indices",
                        element=0,
                    )
                },
            ),
        ),
    )


_SHUFFLE_BYTE_NAMES = tuple(f"lane{i}" for i in range(16))


def _shuffle_rule(type_pattern: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("wasm.i8x16.shuffle")
    return DescriptorRule(
        source_op=vector.vector_shuffle,
        descriptor=descriptor,
        guards=(
            _value_type("source", type_pattern),
            _value_type("result", type_pattern),
            Guard.i64_array_count("source_lanes", 4),
            Guard.i64_array_elements_range("source_lanes", 0, 3),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("source"),
                    "rhs": ValueRef.operand("source"),
                },
                results={"dst": ValueRef.result("result")},
                immediates=(
                    AttrProject.expand_lane_i64_array_to_byte_lanes(
                        source_attr="source_lanes",
                        source_lane_count=4,
                        bytes_per_lane=4,
                        target_names=_SHUFFLE_BYTE_NAMES,
                    ),
                ),
            ),
        ),
    )


def _buffer_view_rule() -> ValueAliasRule:
    return ValueAliasRule(
        source_op=buffer.buffer_view,
        source=ValueRef.operand("buffer"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_i64_range(
                "byte_offset",
                0,
                0,
                diagnostic=_ZERO_BUFFER_VIEW_OFFSET_DIAGNOSTIC,
            ),
        ),
    )


def _source_memory_constraint(
    operation: SourceMemoryOperation,
    *,
    dynamic: bool,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.BLOCK_ARGUMENT,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=4,
        vector_lane_count=4,
        vector_lane_byte_stride=4,
        static_byte_offset=0,
        dynamic_term_count=1 if dynamic else 0,
        dynamic_index_source=(
            SourceMemoryDynamicIndexSource.VALUE
            if dynamic
            else SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride=4 if dynamic else 0,
        dynamic_offset_unsigned_bit_count=32 if dynamic else 0,
        dynamic_offset_diagnostic=(_WASM32_ADDRESS_DIAGNOSTIC if dynamic else None),
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )


def _dynamic_address_emits() -> tuple[EmitDescriptorOp, ...]:
    stride = ValueRef.temporary("stride")
    offset = ValueRef.temporary("offset")
    address = ValueRef.temporary("address")
    return (
        EmitDescriptorOp(
            descriptor=_descriptor("wasm.i32.const"),
            results={"dst": stride},
            result_types={"dst": _I32},
            immediates={"i32_value": 4},
            form=DescriptorEmitForm.CONST,
        ),
        EmitDescriptorOp(
            descriptor=_descriptor("wasm.i32.mul"),
            operands={
                "lhs": ValueRef.operand("indices"),
                "rhs": stride,
            },
            results={"dst": offset},
            result_types={"dst": _I32},
        ),
        EmitDescriptorOp(
            descriptor=_descriptor("wasm.i32.add"),
            operands={
                "lhs": ValueRef.operand("view"),
                "rhs": offset,
            },
            results={"dst": address},
            result_types={"dst": _I32},
        ),
    )


def _memory_address_ref(dynamic: bool) -> ValueRef:
    if dynamic:
        return ValueRef.temporary("address")
    return ValueRef.operand("view")


def _vector_load_rule(
    result_type: TypePattern,
    *,
    dynamic: bool,
) -> DescriptorRule:
    descriptor = _descriptor("wasm.v128.load")
    memory_emit = EmitDescriptorOp(
        descriptor=descriptor,
        operands={"address": _memory_address_ref(dynamic)},
        results={"dst": ValueRef.result("result")},
        source_memory=_source_memory_constraint(
            SourceMemoryOperation.LOAD,
            dynamic=dynamic,
        ),
        form=DescriptorEmitForm.OP,
    )
    emits = (*_dynamic_address_emits(), memory_emit) if dynamic else (memory_emit,)
    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            _value_type("result", result_type),
        ),
        emit=emits,
    )


def _vector_store_rule(
    value_type: TypePattern,
    *,
    dynamic: bool,
) -> DescriptorRule:
    descriptor = _descriptor("wasm.v128.store")
    memory_emit = EmitDescriptorOp(
        descriptor=descriptor,
        operands={
            "address": _memory_address_ref(dynamic),
            "value": ValueRef.operand("value"),
        },
        source_memory=_source_memory_constraint(
            SourceMemoryOperation.STORE,
            dynamic=dynamic,
        ),
        form=DescriptorEmitForm.OP,
    )
    emits = (*_dynamic_address_emits(), memory_emit) if dynamic else (memory_emit,)
    return DescriptorRule(
        source_op=vector.vector_store,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            _value_type("value", value_type),
        ),
        emit=emits,
    )


WASM_CORE_SIMD128_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
}

WASM_CORE_SIMD128_CONTRACT_FRAGMENT = ContractFragment(
    name="wasm.core.simd128",
    descriptor_set=WASM_CORE_SIMD128_DESCRIPTOR_SET,
    public_header="loom/target/emit/wasm/contracts/core_simd128.h",
    cases=(
        _buffer_view_rule(),
        _binary_rule(scalar_arithmetic.scalar_addi, _I32, "wasm.i32.add"),
        _binary_rule(scalar_arithmetic.scalar_subi, _I32, "wasm.i32.sub"),
        _binary_rule(scalar_arithmetic.scalar_addf, _F32, "wasm.f32.add"),
        _scalar_compare_rule(
            scalar_comparison.scalar_cmpi, "ult", _I32, "wasm.i32.lt_u"
        ),
        _const_i32_rule(scalar_conversion.scalar_constant, _I32),
        _splat_rule(),
        _select_rule(_V4I32),
        _select_rule(_V4F32),
        _compare_rule(vector.vector_cmpi, "eq", _V4I32, "wasm.i32x4.eq"),
        _compare_rule(vector.vector_cmpi, "ne", _V4I32, "wasm.i32x4.ne"),
        _compare_rule(vector.vector_cmpi, "slt", _V4I32, "wasm.i32x4.lt_s"),
        _compare_rule(vector.vector_cmpi, "sle", _V4I32, "wasm.i32x4.le_s"),
        _compare_rule(vector.vector_cmpi, "sgt", _V4I32, "wasm.i32x4.gt_s"),
        _compare_rule(vector.vector_cmpi, "sge", _V4I32, "wasm.i32x4.ge_s"),
        _compare_rule(vector.vector_cmpi, "ult", _V4I32, "wasm.i32x4.lt_u"),
        _compare_rule(vector.vector_cmpi, "ule", _V4I32, "wasm.i32x4.le_u"),
        _compare_rule(vector.vector_cmpi, "ugt", _V4I32, "wasm.i32x4.gt_u"),
        _compare_rule(vector.vector_cmpi, "uge", _V4I32, "wasm.i32x4.ge_u"),
        _compare_rule(vector.vector_cmpf, "oeq", _V4F32, "wasm.f32x4.eq"),
        _compare_rule(vector.vector_cmpf, "ogt", _V4F32, "wasm.f32x4.gt"),
        _compare_rule(vector.vector_cmpf, "oge", _V4F32, "wasm.f32x4.ge"),
        _compare_rule(vector.vector_cmpf, "olt", _V4F32, "wasm.f32x4.lt"),
        _compare_rule(vector.vector_cmpf, "ole", _V4F32, "wasm.f32x4.le"),
        _binary_rule(vector.vector_addf, _V4F32, "wasm.f32x4.add"),
        _binary_rule(vector.vector_mulf, _V4F32, "wasm.f32x4.mul"),
        _binary_rule(vector.vector_addi, _V4I32, "wasm.i32x4.add"),
        _binary_rule(vector.vector_subi, _V4I32, "wasm.i32x4.sub"),
        _binary_rule(vector.vector_muli, _V4I32, "wasm.i32x4.mul"),
        _const_i32_rule(index.index_constant, _INDEX),
        _const_i32_rule(index.index_constant, _OFFSET),
        _scalar_compare_rule(index.index_cmp, "ult", _INDEX, "wasm.i32.lt_u"),
        _binary_rule(index.index_add, _INDEX, "wasm.i32.add"),
        _binary_rule(index.index_add, _OFFSET, "wasm.i32.add"),
        _binary_rule(index.index_sub, _INDEX, "wasm.i32.sub"),
        _binary_rule(index.index_sub, _OFFSET, "wasm.i32.sub"),
        _binary_rule(index.index_mul, _INDEX, "wasm.i32.mul"),
        _extract_rule(_V4I32, _I32, "wasm.i32x4.extract_lane"),
        _extract_rule(_V4F32, _F32, "wasm.f32x4.extract_lane"),
        _insert_rule(_I32, _V4I32, "wasm.i32x4.replace_lane"),
        _insert_rule(_F32, _V4F32, "wasm.f32x4.replace_lane"),
        _shuffle_rule(_V4I32),
        _shuffle_rule(_V4F32),
        _shuffle_rule(_V4I1),
        _vector_load_rule(_V4I32, dynamic=False),
        _vector_load_rule(_V4F32, dynamic=False),
        _vector_load_rule(_V4I32, dynamic=True),
        _vector_load_rule(_V4F32, dynamic=True),
        _vector_store_rule(_V4I32, dynamic=False),
        _vector_store_rule(_V4F32, dynamic=False),
        _vector_store_rule(_V4I32, dynamic=True),
        _vector_store_rule(_V4F32, dynamic=True),
        *reduction_descriptor_rules(
            vector.vector_reduce,
            (
                ReductionDescriptorCase(
                    kind="addi",
                    input_type=_V4I32,
                    accumulator_type=_I32,
                    extract_descriptor=_descriptor("wasm.i32x4.extract_lane"),
                    combine_descriptor=_descriptor("wasm.i32.add"),
                ),
                ReductionDescriptorCase(
                    kind="addf",
                    input_type=_V4F32,
                    accumulator_type=_F32,
                    extract_descriptor=_descriptor("wasm.f32x4.extract_lane"),
                    combine_descriptor=_descriptor("wasm.f32.add"),
                ),
            ),
            lane_count=4,
        ),
    ),
)
