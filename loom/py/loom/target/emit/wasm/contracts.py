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
from loom.error.wasm import (
    ERR_WASM_001,
    ERR_WASM_002,
    ERR_WASM_003,
    ERR_WASM_005,
    ERR_WASM_006,
)
from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    Buffer,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryIntegerConversion,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    TypePattern,
    ValueAliasRule,
    ValueProject,
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

_I1 = Scalar("i1")
_I8 = Scalar("i8")
_I16 = Scalar("i16")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_BYTE_STORAGE = Scalar(("i8", "f8E4M3", "f8E5M2"))
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_V4I1 = Vector("i1", lanes=4)
_V4I32 = Vector("i32", lanes=4)
_V4F32 = Vector("f32", lanes=4)
_V2I64 = Vector("i64", lanes=2)
_V2F64 = Vector("f64", lanes=2)

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
    if type_pattern.kind == "buffer":
        return "buffer"
    if type_pattern == _I1:
        return "i1 scalar"
    if type_pattern == _I8:
        return "i8 scalar"
    if type_pattern == _BYTE_STORAGE:
        return "i8, f8E4M3, or f8E5M2 scalar"
    if type_pattern == _I16:
        return "i16 scalar"
    if type_pattern == _I32:
        return "i32 scalar"
    if type_pattern == _I64:
        return "i64 scalar"
    if type_pattern == _F8E4M3:
        return "f8E4M3 scalar"
    if type_pattern == _F8E5M2:
        return "f8E5M2 scalar"
    if type_pattern == _F16:
        return "f16 scalar"
    if type_pattern == _BF16:
        return "bf16 scalar"
    if type_pattern == _F32:
        return "f32 scalar"
    if type_pattern == _F64:
        return "f64 scalar"
    if type_pattern in (_INDEX, _OFFSET):
        return "index or offset scalar"
    if type_pattern == _V4I1:
        return "vector<4xi1>"
    if type_pattern == _V4I32:
        return "vector<4xi32>"
    if type_pattern == _V4F32:
        return "vector<4xf32>"
    if type_pattern == _V2I64:
        return "vector<2xi64>"
    if type_pattern == _V2F64:
        return "vector<2xf64>"
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


def _const_i1_rule() -> DescriptorRule:
    descriptor = _descriptor("wasm.i32.const")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            _value_type("result", _I1),
            Guard.value_exact_i64("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i32_value": ValueProject.exact_i64("result")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_i64_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("wasm.i64.const")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
            _value_type("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i64_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_float_rule(result_type: TypePattern, descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            _value_type("result", result_type),
            Guard.value_exact_float("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"bits": ValueProject.float_bits("result")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(*_typed_guards(("lhs", "rhs", "result"), type_pattern), *guards),
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


def _index_scale_rule() -> DescriptorRule:
    descriptor = _descriptor("wasm.i32.mul")
    return DescriptorRule(
        source_op=index.index_scale,
        descriptor=descriptor,
        guards=(
            _value_type("index", _INDEX),
            _value_type("stride", _OFFSET),
            _value_type("result", _OFFSET),
            Guard.value_unsigned_bit_count(
                "result",
                32,
                diagnostic=_WASM32_ADDRESS_DIAGNOSTIC,
            ),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("index"),
                    "rhs": ValueRef.operand("stride"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _conversion_rule(
    source_op: Op,
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            _value_type("input", source_type),
            _value_type("result", result_type),
            *guards,
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _conversion_alias_rule(
    source_op: Op,
    source_type: TypePattern,
    result_type: TypePattern,
    *,
    guards: tuple[Guard, ...] = (),
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=source_op,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            _value_type("input", source_type),
            _value_type("result", result_type),
            *guards,
        ),
    )


def _masked_extui_rule(
    source_type: TypePattern,
    mask: int,
) -> DescriptorRule:
    const_descriptor = _descriptor("wasm.i32.const")
    and_descriptor = _descriptor("wasm.i32.and")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_extui,
        descriptor=and_descriptor,
        guards=(
            _value_type("input", source_type),
            _value_type("result", _I32),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=const_descriptor,
                results={"dst": ValueRef.temporary("mask")},
                result_types={"dst": _I32},
                immediates={"i32_value": mask},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=and_descriptor,
                operands={
                    "lhs": ValueRef.operand("input"),
                    "rhs": ValueRef.temporary("mask"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _splat_rule(
    value_type: TypePattern, result_type: TypePattern, descriptor_key: str
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            _value_type("scalar", value_type),
            _value_type("result", result_type),
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


def _whole_value_select_rule(
    value_type: TypePattern, descriptor_key: str
) -> DescriptorRule:
    # A scalar condition chooses the entire value, including a SIMD register.
    # Per-lane predicates use vector.select and v128.bitselect instead.
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            _value_type("condition", _I1),
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
    extracted = (
        ValueRef.temporary("predicate_bits")
        if result_type == _I1
        else ValueRef.result("result")
    )
    emits = [
        EmitDescriptorOp(
            descriptor=descriptor,
            operands={"source": ValueRef.operand("source")},
            results={"dst": extracted},
            result_types={"dst": _I32} if result_type == _I1 else None,
            immediates={
                "lane": AttrProject.i64_array_element("static_indices", element=0)
            },
        ),
    ]
    if result_type == _I1:
        # SIMD comparisons use all-one lanes; scalar i1 values are zero or one.
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=_descriptor("wasm.i32.const"),
                    results={"dst": ValueRef.temporary("mask")},
                    result_types={"dst": _I32},
                    immediates={"i32_value": 1},
                    form=DescriptorEmitForm.CONST,
                ),
                EmitDescriptorOp(
                    descriptor=_descriptor("wasm.i32.and"),
                    operands={"lhs": extracted, "rhs": ValueRef.temporary("mask")},
                    results={"dst": ValueRef.result("result")},
                ),
            )
        )
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            _value_type("source", source_type),
            _value_type("result", result_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", 0, 0, source_type.lanes - 1
            ),
        ),
        emit=tuple(emits),
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
            Guard.i64_array_element_range("static_indices", 0, 0, dest_type.lanes - 1),
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


def _view_alias_rules() -> Iterable[ValueAliasRule]:
    # The retained memory plan owns every byte origin; view values carry only
    # the underlying resource identity.
    for source_op, operand in (
        (buffer.buffer_view, "buffer"),
        (view.view_subview, "source"),
        (view.view_refine, "source"),
    ):
        yield ValueAliasRule(
            source_op=source_op,
            source=ValueRef.operand(operand),
            result=ValueRef.result("result"),
        )


def _buffer_byte_address_emits(
    buffer_field: str,
) -> tuple[ValueRef, tuple[EmitDescriptorOp, ...]]:
    address = ValueRef.temporary("address")
    return address, (
        EmitDescriptorOp(
            descriptor=_descriptor("wasm.i32.add"),
            operands={
                "lhs": ValueRef.operand(buffer_field),
                "rhs": ValueRef.operand("byte_offset"),
            },
            results={"dst": address},
            result_types={"dst": _I32},
            form=DescriptorEmitForm.OP,
        ),
    )


def _buffer_load_i8_u_rule() -> DescriptorRule:
    descriptor = _descriptor("wasm.i32.load8_u")
    address, address_emits = _buffer_byte_address_emits("source")
    return DescriptorRule(
        source_op=buffer.buffer_load_i8_u,
        descriptor=descriptor,
        guards=(
            _value_type("byte_offset", _OFFSET),
            _value_type("result", _I32),
        ),
        emit=(
            *address_emits,
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"address": address},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _buffer_store_i8_rule() -> DescriptorRule:
    descriptor = _descriptor("wasm.i32.store8")
    address, address_emits = _buffer_byte_address_emits("target")
    return DescriptorRule(
        source_op=buffer.buffer_store_i8,
        descriptor=descriptor,
        guards=(
            _value_type("value", _I32),
            _value_type("byte_offset", _OFFSET),
        ),
        emit=(
            *address_emits,
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "address": address,
                    "value": ValueRef.operand("value"),
                },
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _memory_rule(
    source_op: Op,
    operation: SourceMemoryOperation,
    value_type: TypePattern,
    descriptor_key: str,
    *,
    element_byte_count: int,
    lane_count: int,
    dynamic: bool,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    source_memory = SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.BLOCK_ARGUMENT,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=element_byte_count,
        vector_lane_count=lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=0,
        static_byte_offset_maximum=(1 << 32) - 1,
        dynamic_term_count=None if dynamic else 0,
        dynamic_term_count_minimum=1 if dynamic else 0,
        dynamic_view_base_term_count=None,
        allow_dynamic_stride_values=dynamic,
        dynamic_offset_unsigned_bit_count=32,
        dynamic_offset_diagnostic=_WASM32_ADDRESS_DIAGNOSTIC,
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )
    address = ValueRef.operand("view")
    emits = []
    if dynamic:
        address = ValueRef.temporary("address")
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("wasm.i32.add"),
                operands={
                    "lhs": ValueRef.operand("view"),
                    "rhs": ValueRef.source_memory_dynamic_byte_offset(),
                },
                results={"dst": address},
                result_types={"dst": _I32},
                source_memory=source_memory,
                source_memory_byte_offset_materializer=SourceMemoryByteOffsetMaterializer(
                    constant=_descriptor("wasm.i32.const"),
                    add=_descriptor("wasm.i32.add"),
                    multiply=_descriptor("wasm.i32.mul"),
                    shift_left=None,
                    constant_immediate="i32_value",
                    integer_conversions=(
                        SourceMemoryIntegerConversion(
                            "i64", _descriptor("wasm.i32.wrap_i64")
                        ),
                    ),
                ),
            )
        )
    is_load = operation is SourceMemoryOperation.LOAD
    emits.append(
        EmitDescriptorOp(
            descriptor=descriptor,
            operands={"address": address}
            if is_load
            else {
                "address": address,
                "value": ValueRef.operand("value"),
            },
            results={"dst": ValueRef.result("result")} if is_load else {},
            immediates={"offset": SourceMemoryProject.static_byte_offset()},
            source_memory=source_memory,
            form=DescriptorEmitForm.OP,
        )
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(_value_type("result" if is_load else "value", value_type),),
        emit=tuple(emits),
    )


WASM_CORE_SIMD128_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

WASM_CORE_SIMD128_CONTRACT_FRAGMENT = ContractFragment(
    name="wasm.core.simd128",
    descriptor_set=WASM_CORE_SIMD128_DESCRIPTOR_SET,
    public_header="loom/target/emit/wasm/contracts/core_simd128.h",
    cases=(
        *_view_alias_rules(),
        _buffer_load_i8_u_rule(),
        _buffer_store_i8_rule(),
        *(
            _binary_rule(source_op, value_type, f"wasm.{type_name}.{operation}")
            for value_type, type_name in ((_I32, "i32"), (_I64, "i64"))
            for source_op, operation in (
                (scalar_arithmetic.scalar_addi, "add"),
                (scalar_arithmetic.scalar_subi, "sub"),
                (scalar_arithmetic.scalar_muli, "mul"),
                (scalar_arithmetic.scalar_remui, "rem_u"),
                (scalar_bitwise.scalar_andi, "and"),
                (scalar_bitwise.scalar_ori, "or"),
                (scalar_bitwise.scalar_xori, "xor"),
                (scalar_bitwise.scalar_shli, "shl"),
                (scalar_bitwise.scalar_shrsi, "shr_s"),
                (scalar_bitwise.scalar_shrui, "shr_u"),
            )
        ),
        *(
            _binary_rule(source_op, _I1, f"wasm.i32.{operation}")
            for source_op, operation in (
                (scalar_bitwise.scalar_andi, "and"),
                (scalar_bitwise.scalar_ori, "or"),
                (scalar_bitwise.scalar_xori, "xor"),
            )
        ),
        _binary_rule(scalar_arithmetic.scalar_addf, _F32, "wasm.f32.add"),
        *(
            _scalar_compare_rule(
                scalar_comparison.scalar_cmpi,
                predicate,
                value_type,
                f"wasm.{type_name}.{operation}",
            )
            for value_type, type_name in ((_I32, "i32"), (_I64, "i64"))
            for predicate, operation in (
                ("eq", "eq"),
                ("ne", "ne"),
                ("slt", "lt_s"),
                ("sle", "le_s"),
                ("sgt", "gt_s"),
                ("sge", "ge_s"),
                ("ult", "lt_u"),
                ("ule", "le_u"),
                ("ugt", "gt_u"),
                ("uge", "ge_u"),
            )
        ),
        _conversion_rule(
            scalar_conversion.scalar_bitcast,
            _F32,
            _I32,
            "wasm.i32.reinterpret_f32",
        ),
        _conversion_rule(
            scalar_conversion.scalar_bitcast,
            _F64,
            _I64,
            "wasm.i64.reinterpret_f64",
        ),
        _conversion_rule(
            scalar_conversion.scalar_trunci,
            _I64,
            _I32,
            "wasm.i32.wrap_i64",
        ),
        _conversion_rule(
            scalar_conversion.scalar_bitcast, _I32, _F32, "wasm.f32.reinterpret_i32"
        ),
        _conversion_rule(
            scalar_conversion.scalar_bitcast, _I64, _F64, "wasm.f64.reinterpret_i64"
        ),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F8E4M3, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F8E5M2, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F16, _I16),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _BF16, _I16),
        _conversion_alias_rule(scalar_conversion.scalar_extui, _I1, _I32),
        _masked_extui_rule(_I8, 0xFF),
        _masked_extui_rule(_I16, 0xFFFF),
        _const_i32_rule(scalar_conversion.scalar_constant, _I32),
        _const_i1_rule(),
        _const_i64_rule(scalar_conversion.scalar_constant, _I64),
        _const_float_rule(_F32, "wasm.f32.const"),
        _const_float_rule(_F64, "wasm.f64.const"),
        *(
            _whole_value_select_rule(value_type, f"wasm.{type_name}.select")
            for value_type, type_name in (
                (_I1, "i32"),
                (_I32, "i32"),
                (_I64, "i64"),
                (_F32, "f32"),
                (_F64, "f64"),
                (_INDEX, "i32"),
                (_OFFSET, "i32"),
                (_V4I1, "v128"),
                (_V4I32, "v128"),
                (_V4F32, "v128"),
                (_V2I64, "v128"),
                (_V2F64, "v128"),
                (Buffer(), "i32"),
            )
        ),
        _splat_rule(_I32, _V4I32, "wasm.i32x4.splat"),
        _splat_rule(_I64, _V2I64, "wasm.i64x2.splat"),
        _splat_rule(_F32, _V4F32, "wasm.f32x4.splat"),
        _splat_rule(_F64, _V2F64, "wasm.f64x2.splat"),
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
        *(
            _scalar_compare_rule(
                index.index_cmp, predicate, value_type, f"wasm.i32.{operation}"
            )
            for value_type in (_INDEX, _OFFSET)
            for predicate, operation in (
                ("eq", "eq"),
                ("ne", "ne"),
                ("slt", "lt_s"),
                ("sle", "le_s"),
                ("sgt", "gt_s"),
                ("sge", "ge_s"),
                ("ult", "lt_u"),
                ("ule", "le_u"),
                ("ugt", "gt_u"),
                ("uge", "ge_u"),
            )
        ),
        *(
            _conversion_alias_rule(index.index_cast, source_type, result_type)
            for source_type, result_type in (
                (_INDEX, _I32),
                (_I32, _INDEX),
                (_OFFSET, _I32),
                (_I32, _OFFSET),
            )
        ),
        *(
            _conversion_alias_rule(
                index.index_cast,
                source_type,
                result_type,
                guards=(Guard.value_i64_range("input", 0, (1 << 31) - 1),),
            )
            for source_type, result_type in ((_INDEX, _OFFSET), (_OFFSET, _INDEX))
        ),
        *(
            _conversion_rule(
                index.index_cast,
                source_type,
                _I64,
                f"wasm.i64.extend_i32_{signedness}",
            )
            for source_type, signedness in ((_INDEX, "s"), (_OFFSET, "u"))
        ),
        *(
            _conversion_rule(
                index.index_cast,
                _I64,
                result_type,
                "wasm.i32.wrap_i64",
                guards=(Guard.value_i64_range("input", minimum, maximum),),
            )
            for result_type, minimum, maximum in (
                (_INDEX, -(1 << 31), (1 << 31) - 1),
                (_OFFSET, 0, (1 << 32) - 1),
            )
        ),
        _binary_rule(index.index_add, _INDEX, "wasm.i32.add"),
        _binary_rule(index.index_add, _OFFSET, "wasm.i32.add"),
        _binary_rule(index.index_sub, _INDEX, "wasm.i32.sub"),
        _binary_rule(index.index_sub, _OFFSET, "wasm.i32.sub"),
        _binary_rule(index.index_mul, _INDEX, "wasm.i32.mul"),
        _index_scale_rule(),
        *(
            _binary_rule(source_op, _INDEX, f"wasm.i32.{operation}")
            for source_op, operation in (
                (index.index_andi, "and"),
                (index.index_ori, "or"),
                (index.index_xori, "xor"),
                (index.index_shli, "shl"),
                (index.index_shrsi, "shr_s"),
                (index.index_shrui, "shr_u"),
            )
        ),
        _binary_rule(
            index.index_rem,
            _INDEX,
            "wasm.i32.rem_u",
            guards=tuple(
                Guard.value_i64_range(field, 0, (1 << 32) - 1)
                for field in ("lhs", "rhs")
            ),
        ),
        _extract_rule(_V4I1, _I1, "wasm.i32x4.extract_lane"),
        _extract_rule(_V4I32, _I32, "wasm.i32x4.extract_lane"),
        _extract_rule(_V4F32, _F32, "wasm.f32x4.extract_lane"),
        _extract_rule(_V2I64, _I64, "wasm.i64x2.extract_lane"),
        _extract_rule(_V2F64, _F64, "wasm.f64x2.extract_lane"),
        _insert_rule(_I32, _V4I32, "wasm.i32x4.replace_lane"),
        _insert_rule(_F32, _V4F32, "wasm.f32x4.replace_lane"),
        _insert_rule(_I64, _V2I64, "wasm.i64x2.replace_lane"),
        _insert_rule(_F64, _V2F64, "wasm.f64x2.replace_lane"),
        _shuffle_rule(_V4I32),
        _shuffle_rule(_V4F32),
        _shuffle_rule(_V4I1),
        *(
            _memory_rule(
                load_op if operation is SourceMemoryOperation.LOAD else store_op,
                operation,
                value_type,
                "wasm."
                + (
                    load_descriptor
                    if operation is SourceMemoryOperation.LOAD
                    else store_descriptor
                ),
                element_byte_count=element_byte_count,
                lane_count=lane_count,
                dynamic=dynamic,
            )
            for (
                value_type,
                load_descriptor,
                store_descriptor,
                element_byte_count,
                lane_count,
                load_op,
                store_op,
            ) in (
                (_I32, "i32.load", "i32.store", 4, 1, view.view_load, view.view_store),
                (_I64, "i64.load", "i64.store", 8, 1, view.view_load, view.view_store),
                (_F32, "f32.load", "f32.store", 4, 1, view.view_load, view.view_store),
                (_F64, "f64.load", "f64.store", 8, 1, view.view_load, view.view_store),
                (
                    _BYTE_STORAGE,
                    "i32.load8_u",
                    "i32.store8",
                    1,
                    1,
                    view.view_load,
                    view.view_store,
                ),
                (
                    _I16,
                    "i32.load16_u",
                    "i32.store16",
                    2,
                    1,
                    view.view_load,
                    view.view_store,
                ),
                (
                    _V4I32,
                    "v128.load",
                    "v128.store",
                    4,
                    4,
                    vector.vector_load,
                    vector.vector_store,
                ),
                (
                    _V4F32,
                    "v128.load",
                    "v128.store",
                    4,
                    4,
                    vector.vector_load,
                    vector.vector_store,
                ),
            )
            for operation in (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE)
            for dynamic in (True, False)
        ),
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
