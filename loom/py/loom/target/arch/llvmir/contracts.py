# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVMIR generic source-to-low contract fragment."""

from __future__ import annotations

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.kernel import ALL_KERNEL_OPS
from loom.dialect.kernel import defs as kernel
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.scf import defs as scf
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.arch.llvmir.descriptors import LLVMIR_GENERIC_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    ResultTypeBinding,
    Scalar,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    SourceOpProject,
    TypePattern,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I8 = Scalar("i8")
_I16 = Scalar("i16")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_DYNAMIC_INDEX = -(2**63)

_VECTOR_LANE_COUNTS = (2, 3, 4, 8, 16)
_STRUCTURAL_VECTOR_LANE_COUNTS = (*_VECTOR_LANE_COUNTS, 32)
_VECTOR_SELECT_TYPES = ("i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64")
_STRUCTURAL_VECTOR_TYPES = (
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "bf16",
    "f32",
    "f64",
)
_MEMORY_VALUE_TYPES = (
    ("i8", 1),
    ("i16", 2),
    ("i32", 4),
    ("i64", 8),
    ("f16", 2),
    ("bf16", 2),
    ("f32", 4),
    ("f64", 8),
)
_ATOMIC_INTEGER_VALUE_TYPES = (
    ("i8", 1),
    ("i16", 2),
    ("i32", 4),
    ("i64", 8),
)
_ATOMIC_FLOAT_VALUE_TYPES = (
    ("f32", 4),
    ("f64", 8),
)
_ATOMIC_INTEGER_KINDS = (
    ("addi", "add"),
    ("subi", "sub"),
    ("andi", "and"),
    ("ori", "or"),
    ("xori", "xor"),
    ("minsi", "min"),
    ("maxsi", "max"),
    ("minui", "umin"),
    ("maxui", "umax"),
)
_ATOMIC_INTEGER_RMW_KINDS = (
    ("xchgi", "xchg"),
    *_ATOMIC_INTEGER_KINDS,
)
_ATOMIC_FLOAT_KINDS = (
    ("addf", "fadd"),
    ("minimumf", "fminimum"),
    ("maximumf", "fmaximum"),
    ("minnumf", "fmin"),
    ("maxnumf", "fmax"),
)
_ATOMIC_FLOAT_RMW_KINDS = (
    ("xchgf", "xchg"),
    *_ATOMIC_FLOAT_KINDS,
)

_CACHE_POLICY_BUILD_FLAG_SCOPE = 1 << 0
_CACHE_POLICY_BUILD_FLAG_TEMPORAL = 1 << 1
_CACHE_POLICY_BUILD_FLAGS_PRESENT = (
    _CACHE_POLICY_BUILD_FLAG_SCOPE | _CACHE_POLICY_BUILD_FLAG_TEMPORAL
)

_KERNEL_DIMENSIONS = ("x", "y", "z")

_INTEGER_PREDICATES = (
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
)

_FLOAT_PREDICATES = (
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
)

_SCALAR_CAST_SPECS = (
    (scalar_conversion.scalar_trunci, "trunc", "i32", "i8"),
    (scalar_conversion.scalar_trunci, "trunc", "i32", "i16"),
    (scalar_conversion.scalar_trunci, "trunc", "i64", "i32"),
    (scalar_conversion.scalar_extsi, "sext", "i8", "i32"),
    (scalar_conversion.scalar_extsi, "sext", "i16", "i32"),
    (scalar_conversion.scalar_extsi, "sext", "i32", "i64"),
    (scalar_conversion.scalar_extui, "zext", "i8", "i32"),
    (scalar_conversion.scalar_extui, "zext", "i16", "i32"),
    (scalar_conversion.scalar_extui, "zext", "i32", "i64"),
    (scalar_conversion.scalar_sitofp, "sitofp", "i8", "f32"),
    (scalar_conversion.scalar_sitofp, "sitofp", "i32", "f32"),
    (scalar_conversion.scalar_sitofp, "sitofp", "i64", "f64"),
    (scalar_conversion.scalar_uitofp, "uitofp", "i8", "f32"),
    (scalar_conversion.scalar_uitofp, "uitofp", "i32", "f32"),
    (scalar_conversion.scalar_uitofp, "uitofp", "i64", "f64"),
    (scalar_conversion.scalar_fptosi, "fptosi", "f32", "i32"),
    (scalar_conversion.scalar_fptosi, "fptosi", "f64", "i64"),
    (scalar_conversion.scalar_fptoui, "fptoui", "f32", "i32"),
    (scalar_conversion.scalar_fptoui, "fptoui", "f64", "i64"),
    (scalar_conversion.scalar_fptrunc, "fptrunc", "f32", "f16"),
    (scalar_conversion.scalar_fptrunc, "fptrunc", "f32", "bf16"),
    (scalar_conversion.scalar_fptrunc, "fptrunc", "f64", "f32"),
    (scalar_conversion.scalar_extf, "fpext", "f16", "f32"),
    (scalar_conversion.scalar_extf, "fpext", "bf16", "f32"),
    (scalar_conversion.scalar_extf, "fpext", "f32", "f64"),
    (scalar_conversion.scalar_bitcast, "bitcast", "i16", "f16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "i16", "bf16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "f16", "i16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "bf16", "i16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "f16", "bf16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "bf16", "f16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "i32", "f32"),
    (scalar_conversion.scalar_bitcast, "bitcast", "f32", "i32"),
    (scalar_conversion.scalar_bitcast, "bitcast", "i64", "f64"),
    (scalar_conversion.scalar_bitcast, "bitcast", "f64", "i64"),
)

_VECTOR_CAST_SPECS = (
    (vector.vector_extsi, "sext", "i32", "i64"),
    (vector.vector_extui, "zext", "i32", "i64"),
    (vector.vector_trunci, "trunc", "i64", "i32"),
    (vector.vector_sitofp, "sitofp", "i32", "f32"),
    (vector.vector_uitofp, "uitofp", "i32", "f32"),
    (vector.vector_fptosi, "fptosi", "f32", "i32"),
    (vector.vector_fptoui, "fptoui", "f32", "i32"),
    (vector.vector_fptrunc, "fptrunc", "f32", "f16"),
    (vector.vector_fptrunc, "fptrunc", "f32", "bf16"),
    (vector.vector_extf, "fpext", "f16", "f32"),
    (vector.vector_extf, "fpext", "bf16", "f32"),
    (vector.vector_bitcast, "bitcast", "i32", "f32"),
    (vector.vector_bitcast, "bitcast", "f32", "i32"),
    (vector.vector_bitcast, "bitcast", "i64", "f64"),
    (vector.vector_bitcast, "bitcast", "f64", "i64"),
)
_BITCAST_RESHAPE_SPECS = (
    ("i32", 1, "i8", 4),
    ("i32", 2, "i8", 8),
    ("f16", 2, "i32", 1),
    ("bf16", 2, "i32", 1),
)

_I8_MIN = -(2**7)
_I8_MAX = (2**7) - 1
_I1_MIN = 0
_I1_MAX = 1
_I16_MIN = -(2**15)
_I16_MAX = (2**15) - 1
_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="llvmir-generic",
    constraint_key="llvmir.generic.source_memory",
)
_I64_ATTR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="value",
    constraint_key="llvmir.constant.i64_attr",
)
_F64_ATTR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="value",
    constraint_key="llvmir.constant.f64_attr",
)
_PREDICATE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="predicate",
    constraint_key="llvmir.compare.predicate",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(LLVMIR_GENERIC_CORE_DESCRIPTOR_SET, key)


def _source_memory_byte_offset_materializer() -> SourceMemoryByteOffsetMaterializer:
    return SourceMemoryByteOffsetMaterializer(
        const_i64=_descriptor("llvmir.const.i64"),
        add_i64=_descriptor("llvmir.add.i64"),
        mul_i64=_descriptor("llvmir.mul.i64"),
        shl_i64=_descriptor("llvmir.shl.i64"),
    )


def _vector_type(element: str, lane_count: int) -> TypePattern:
    return Vector(element, lanes=lane_count)


def _vector_shape_type(element: str, *dims: int) -> TypePattern:
    return Vector(element, dims=dims)


def _vector_suffix(element: str, lane_count: int) -> str:
    return f"v{lane_count}{element}"


def _typed_guards(
    fields: tuple[str, ...], type_pattern: TypePattern
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _op_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    result_types: dict[str, ResultTypeBinding] | None = None,
    immediates: dict[str, AttrProject | SourceMemoryProject | ValueProject | int]
    | None = None,
    source_memory: SourceMemoryConstraint | None = None,
    source_memory_byte_offset_materializer: (
        SourceMemoryByteOffsetMaterializer | None
    ) = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
        source_memory=source_memory,
        source_memory_byte_offset_materializer=source_memory_byte_offset_materializer,
    )


def _const_i32_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    return _const_integer_rule(
        source_op,
        result_type,
        "llvmir.const.i32",
        minimum=_I32_MIN,
        maximum=_I32_MAX,
    )


def _const_integer_rule(
    source_op: Op,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    guards = [
        Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
        Guard.value_type("result", result_type),
    ]
    if minimum is not None and maximum is not None:
        guards.append(Guard.i64_range("value", minimum, maximum))
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=tuple(guards),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_i8_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    return _const_integer_rule(
        source_op,
        result_type,
        "llvmir.const.i8",
        minimum=_I8_MIN,
        maximum=_I8_MAX,
    )


def _const_i1_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    return _const_integer_rule(
        source_op,
        result_type,
        "llvmir.const.i1",
        minimum=_I1_MIN,
        maximum=_I1_MAX,
    )


def _const_i16_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    return _const_integer_rule(
        source_op,
        result_type,
        "llvmir.const.i16",
        minimum=_I16_MIN,
        maximum=_I16_MAX,
    )


def _const_i64_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    return _const_integer_rule(source_op, result_type, "llvmir.const.i64")


def _const_float_bits_project(element: str) -> ValueProject:
    if element == "f16":
        return ValueProject.f64_as_f16_bits("result")
    if element == "bf16":
        return ValueProject.f64_as_bf16_bits("result")
    if element == "f32":
        return ValueProject.f64_as_f32_bits("result")
    if element == "f64":
        return ValueProject.f64_as_f64_bits("result")
    raise ValueError(f"unsupported LLVMIR float constant type '{element}'")


def _const_float_rule(
    source_op: Op, result_type: TypePattern, descriptor_key: str
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    bits_project = _const_float_bits_project(result_type.element)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64", diagnostic=_F64_ATTR_DIAGNOSTIC),
            Guard.value_type("result", result_type),
            Guard.value_exact_f64("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"bits": bits_project},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _vector_const_i_rule(
    element: str,
    lane_count: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> DescriptorRule:
    descriptor = _descriptor(f"llvmir.const.{_vector_suffix(element, lane_count)}")
    guards = [
        Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
        Guard.value_type("result", _vector_type(element, lane_count)),
    ]
    if minimum is not None and maximum is not None:
        guards.append(Guard.i64_range("value", minimum, maximum))
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=descriptor,
        guards=tuple(guards),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _vector_const_float_rule(element: str, lane_count: int) -> DescriptorRule:
    result_type = _vector_type(element, lane_count)
    descriptor = _descriptor(f"llvmir.const.{_vector_suffix(element, lane_count)}")
    bits_project = _const_float_bits_project(element)
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64", diagnostic=_F64_ATTR_DIAGNOSTIC),
            Guard.value_type("result", result_type),
            Guard.value_exact_f64("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"bits": bits_project},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    source_instance_flags: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    immediates = {}
    if source_instance_flags:
        immediates["fast_math_flags"] = SourceOpProject.instance_flags()
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
                immediates=immediates,
            ),
        ),
    )


def _unary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("input", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _ternary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("a", "b", "c", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand("a"),
                    "b": ValueRef.operand("b"),
                    "c": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _compare_rule(
    source_op: Op,
    operand_type: TypePattern,
    result_type: TypePattern,
    predicate: str,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals(
                "predicate",
                predicate,
                diagnostic=_PREDICATE_DIAGNOSTIC,
            ),
            *_typed_guards(("lhs", "rhs"), operand_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _cast_rule(
    source_op: Op,
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _low_vector_suffix(element: str, lane_count: int) -> str:
    return element if lane_count == 1 else _vector_suffix(element, lane_count)


def _index_cast_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    return _cast_rule(index.index_cast, source_type, result_type, descriptor_key)


def _index_cast_alias_rule(
    source_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=index.index_cast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
    )


def _select_rule(type_pattern: TypePattern, descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    vector_select = type_pattern.kind == "vector"
    condition_type = Vector("i1", lanes=type_pattern.lanes) if vector_select else _I1
    return DescriptorRule(
        source_op=vector.vector_select if vector_select else scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", condition_type),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
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


def _source_memory_constraint(
    operation: SourceMemoryOperation,
    *,
    dynamic_term_count: int,
    element_byte_count: int,
    vector_lane_count: int = 1,
    allow_dynamic_stride_values: bool = False,
    cache_policy: bool = False,
) -> SourceMemoryConstraint:
    if dynamic_term_count == 0:
        dynamic_index_source = SourceMemoryDynamicIndexSource.NONE
        dynamic_byte_stride: int | None = 0
    else:
        dynamic_index_source = SourceMemoryDynamicIndexSource.VALUE
        dynamic_byte_stride = None
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.ANY,
        memory_spaces=(
            "unknown",
            "generic",
            "global",
            "workgroup",
            "private",
            "constant",
            "descriptor",
        ),
        element_byte_count=element_byte_count,
        vector_lane_count=vector_lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=-(2**63),
        static_byte_offset_maximum=(2**63) - 1,
        dynamic_term_count=dynamic_term_count,
        dynamic_index_source=dynamic_index_source,
        dynamic_byte_stride=dynamic_byte_stride,
        allow_dynamic_stride_values=allow_dynamic_stride_values,
        cache_policy_build_flags=(
            _CACHE_POLICY_BUILD_FLAGS_PRESENT if cache_policy else 0
        ),
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )


def _memory_immediates(
    *,
    dynamic_term_count: int,
    materialize_byte_offset: bool,
    cache_policy: bool = False,
) -> dict[str, SourceMemoryProject | AttrProject | int]:
    immediates: dict[str, SourceMemoryProject | AttrProject | int] = {
        "byte_offset": SourceMemoryProject.static_byte_offset()
    }
    if dynamic_term_count != 0:
        immediates["byte_stride"] = (
            1 if materialize_byte_offset else SourceMemoryProject.dynamic_byte_stride()
        )
    if cache_policy:
        immediates["cache_scope"] = AttrProject.enum_ordinal("cache_scope")
        immediates["cache_temporal"] = AttrProject.enum_ordinal("cache_temporal")
    return immediates


def _cache_policy_guards(cache_policy: bool) -> tuple[Guard, ...]:
    if not cache_policy:
        return ()
    return (
        Guard.attr_kind("cache_scope", "enum"),
        Guard.attr_kind("cache_temporal", "enum"),
    )


def _atomic_immediates(
    *,
    dynamic_term_count: int,
    materialize_byte_offset: bool,
    cache_policy: bool = False,
) -> dict[str, SourceMemoryProject | AttrProject | int]:
    immediates = _memory_immediates(
        dynamic_term_count=dynamic_term_count,
        materialize_byte_offset=materialize_byte_offset,
        cache_policy=cache_policy,
    )
    immediates["ordering"] = AttrProject.enum_ordinal("ordering")
    immediates["scope"] = AttrProject.enum_ordinal("scope")
    return immediates


def _atomic_cmpxchg_immediates(
    *,
    dynamic_term_count: int,
    materialize_byte_offset: bool,
    cache_policy: bool = False,
) -> dict[str, SourceMemoryProject | AttrProject | int]:
    immediates = _memory_immediates(
        dynamic_term_count=dynamic_term_count,
        materialize_byte_offset=materialize_byte_offset,
        cache_policy=cache_policy,
    )
    immediates["success_ordering"] = AttrProject.enum_ordinal("success_ordering")
    immediates["failure_ordering"] = AttrProject.enum_ordinal("failure_ordering")
    immediates["scope"] = AttrProject.enum_ordinal("scope")
    return immediates


def _view_load_rule(
    source_op: Op,
    view_field: str,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    vector_lane_count: int = 1,
    low_result_type: TypePattern | None = None,
    materialize_byte_offset: bool | None = None,
    allow_dynamic_stride_values: bool = False,
    cache_policy: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    if materialize_byte_offset is None:
        materialize_byte_offset = dynamic_term_count > 1 or (
            dynamic_term_count != 0 and source_dynamic_count != dynamic_term_count
        )
    if allow_dynamic_stride_values and not materialize_byte_offset:
        raise ValueError("dynamic stride values require byte-offset materialization")
    operands = {"ptr": ValueRef.operand(view_field)}
    if dynamic_term_count != 0:
        if materialize_byte_offset:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif source_dynamic_count:
            operands["index"] = ValueRef.operand("indices")
        else:
            operands["index"] = ValueRef.source_memory_dynamic_term()
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", source_dynamic_count),
            Guard.value_type("result", result_type),
            *_cache_policy_guards(cache_policy),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                result_types=(
                    {"dst": low_result_type} if low_result_type is not None else None
                ),
                immediates=_memory_immediates(
                    dynamic_term_count=dynamic_term_count,
                    materialize_byte_offset=materialize_byte_offset,
                    cache_policy=cache_policy,
                ),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.LOAD,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                    allow_dynamic_stride_values=allow_dynamic_stride_values,
                    cache_policy=cache_policy,
                ),
                source_memory_byte_offset_materializer=(
                    _source_memory_byte_offset_materializer()
                    if materialize_byte_offset
                    else None
                ),
            ),
        ),
    )


def _view_store_rule(
    source_op: Op,
    view_field: str,
    value_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    vector_lane_count: int = 1,
    materialize_byte_offset: bool | None = None,
    allow_dynamic_stride_values: bool = False,
    cache_policy: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    if materialize_byte_offset is None:
        materialize_byte_offset = dynamic_term_count > 1 or (
            dynamic_term_count != 0 and source_dynamic_count != dynamic_term_count
        )
    if allow_dynamic_stride_values and not materialize_byte_offset:
        raise ValueError("dynamic stride values require byte-offset materialization")
    operands = {
        "value": ValueRef.operand("value"),
        "ptr": ValueRef.operand(view_field),
    }
    if dynamic_term_count != 0:
        if materialize_byte_offset:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif source_dynamic_count:
            operands["index"] = ValueRef.operand("indices")
        else:
            operands["index"] = ValueRef.source_memory_dynamic_term()
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", source_dynamic_count),
            Guard.value_type("value", value_type),
            *_cache_policy_guards(cache_policy),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                immediates=_memory_immediates(
                    dynamic_term_count=dynamic_term_count,
                    materialize_byte_offset=materialize_byte_offset,
                    cache_policy=cache_policy,
                ),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.STORE,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                    allow_dynamic_stride_values=allow_dynamic_stride_values,
                    cache_policy=cache_policy,
                ),
                source_memory_byte_offset_materializer=(
                    _source_memory_byte_offset_materializer()
                    if materialize_byte_offset
                    else None
                ),
            ),
        ),
    )


def _view_load_rule_variants(
    source_op: Op,
    view_field: str,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    vector_lane_count: int = 1,
    low_result_type: TypePattern | None = None,
) -> tuple[DescriptorRule, ...]:
    materializes_byte_offset = dynamic_term_count > 1 or (
        dynamic_term_count != 0 and source_dynamic_count != dynamic_term_count
    )
    rules = [
        _view_load_rule(
            source_op,
            view_field,
            result_type,
            descriptor_key,
            source_dynamic_count=source_dynamic_count,
            dynamic_term_count=dynamic_term_count,
            element_byte_count=element_byte_count,
            vector_lane_count=vector_lane_count,
            low_result_type=low_result_type,
            allow_dynamic_stride_values=materializes_byte_offset,
        )
    ]
    rules.append(
        _view_load_rule(
            source_op,
            view_field,
            result_type,
            descriptor_key,
            source_dynamic_count=source_dynamic_count,
            dynamic_term_count=dynamic_term_count,
            element_byte_count=element_byte_count,
            vector_lane_count=vector_lane_count,
            low_result_type=low_result_type,
            allow_dynamic_stride_values=materializes_byte_offset,
            cache_policy=True,
        )
    )
    if dynamic_term_count == 1 and not materializes_byte_offset:
        rules.append(
            _view_load_rule(
                source_op,
                view_field,
                result_type,
                descriptor_key,
                source_dynamic_count=source_dynamic_count,
                dynamic_term_count=dynamic_term_count,
                element_byte_count=element_byte_count,
                vector_lane_count=vector_lane_count,
                low_result_type=low_result_type,
                materialize_byte_offset=True,
                allow_dynamic_stride_values=True,
            )
        )
        rules.append(
            _view_load_rule(
                source_op,
                view_field,
                result_type,
                descriptor_key,
                source_dynamic_count=source_dynamic_count,
                dynamic_term_count=dynamic_term_count,
                element_byte_count=element_byte_count,
                vector_lane_count=vector_lane_count,
                low_result_type=low_result_type,
                materialize_byte_offset=True,
                allow_dynamic_stride_values=True,
                cache_policy=True,
            )
        )
    return tuple(rules)


def _view_store_rule_variants(
    source_op: Op,
    view_field: str,
    value_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    vector_lane_count: int = 1,
) -> tuple[DescriptorRule, ...]:
    materializes_byte_offset = dynamic_term_count > 1 or (
        dynamic_term_count != 0 and source_dynamic_count != dynamic_term_count
    )
    rules = [
        _view_store_rule(
            source_op,
            view_field,
            value_type,
            descriptor_key,
            source_dynamic_count=source_dynamic_count,
            dynamic_term_count=dynamic_term_count,
            element_byte_count=element_byte_count,
            vector_lane_count=vector_lane_count,
            allow_dynamic_stride_values=materializes_byte_offset,
        )
    ]
    rules.append(
        _view_store_rule(
            source_op,
            view_field,
            value_type,
            descriptor_key,
            source_dynamic_count=source_dynamic_count,
            dynamic_term_count=dynamic_term_count,
            element_byte_count=element_byte_count,
            vector_lane_count=vector_lane_count,
            allow_dynamic_stride_values=materializes_byte_offset,
            cache_policy=True,
        )
    )
    if dynamic_term_count == 1 and not materializes_byte_offset:
        rules.append(
            _view_store_rule(
                source_op,
                view_field,
                value_type,
                descriptor_key,
                source_dynamic_count=source_dynamic_count,
                dynamic_term_count=dynamic_term_count,
                element_byte_count=element_byte_count,
                vector_lane_count=vector_lane_count,
                materialize_byte_offset=True,
                allow_dynamic_stride_values=True,
            )
        )
        rules.append(
            _view_store_rule(
                source_op,
                view_field,
                value_type,
                descriptor_key,
                source_dynamic_count=source_dynamic_count,
                dynamic_term_count=dynamic_term_count,
                element_byte_count=element_byte_count,
                vector_lane_count=vector_lane_count,
                materialize_byte_offset=True,
                allow_dynamic_stride_values=True,
                cache_policy=True,
            )
        )
    return tuple(rules)


def _view_atomic_rule(
    source_op: Op,
    value_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    operation: SourceMemoryOperation,
    atomic_kind: str,
    materialize_byte_offset: bool | None = None,
    allow_dynamic_stride_values: bool = False,
    cache_policy: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    if materialize_byte_offset is None:
        materialize_byte_offset = dynamic_term_count > 1 or (
            dynamic_term_count != 0 and source_dynamic_count != dynamic_term_count
        )
    if allow_dynamic_stride_values and not materialize_byte_offset:
        raise ValueError("dynamic stride values require byte-offset materialization")
    operands = {
        "value": ValueRef.operand("value"),
        "ptr": ValueRef.operand("view"),
    }
    if dynamic_term_count != 0:
        if materialize_byte_offset:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif source_dynamic_count:
            operands["index"] = ValueRef.operand("indices")
        else:
            operands["index"] = ValueRef.source_memory_dynamic_term()
    results = {}
    if operation is SourceMemoryOperation.ATOMIC_RMW:
        results["old"] = ValueRef.result("result")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("kind", atomic_kind),
            Guard.operand_segment_count("indices", source_dynamic_count),
            Guard.value_type("value", value_type),
            Guard.attr_kind("ordering", "enum"),
            Guard.attr_kind("scope", "enum"),
            *_cache_policy_guards(cache_policy),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                results=results,
                immediates=_atomic_immediates(
                    dynamic_term_count=dynamic_term_count,
                    materialize_byte_offset=materialize_byte_offset,
                    cache_policy=cache_policy,
                ),
                source_memory=_source_memory_constraint(
                    operation,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                    allow_dynamic_stride_values=allow_dynamic_stride_values,
                    cache_policy=cache_policy,
                ),
                source_memory_byte_offset_materializer=(
                    _source_memory_byte_offset_materializer()
                    if materialize_byte_offset
                    else None
                ),
            ),
        ),
    )


def _view_atomic_rule_variants(
    source_op: Op,
    value_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    operation: SourceMemoryOperation,
    atomic_kind: str,
) -> tuple[DescriptorRule, ...]:
    materializes_byte_offset = dynamic_term_count > 1 or (
        dynamic_term_count != 0 and source_dynamic_count != dynamic_term_count
    )
    rules = [
        _view_atomic_rule(
            source_op,
            value_type,
            descriptor_key,
            source_dynamic_count=source_dynamic_count,
            dynamic_term_count=dynamic_term_count,
            element_byte_count=element_byte_count,
            operation=operation,
            atomic_kind=atomic_kind,
            allow_dynamic_stride_values=materializes_byte_offset,
        )
    ]
    rules.append(
        _view_atomic_rule(
            source_op,
            value_type,
            descriptor_key,
            source_dynamic_count=source_dynamic_count,
            dynamic_term_count=dynamic_term_count,
            element_byte_count=element_byte_count,
            operation=operation,
            atomic_kind=atomic_kind,
            allow_dynamic_stride_values=materializes_byte_offset,
            cache_policy=True,
        )
    )
    if dynamic_term_count == 1 and not materializes_byte_offset:
        rules.append(
            _view_atomic_rule(
                source_op,
                value_type,
                descriptor_key,
                source_dynamic_count=source_dynamic_count,
                dynamic_term_count=dynamic_term_count,
                element_byte_count=element_byte_count,
                operation=operation,
                atomic_kind=atomic_kind,
                materialize_byte_offset=True,
                allow_dynamic_stride_values=True,
            )
        )
        rules.append(
            _view_atomic_rule(
                source_op,
                value_type,
                descriptor_key,
                source_dynamic_count=source_dynamic_count,
                dynamic_term_count=dynamic_term_count,
                element_byte_count=element_byte_count,
                operation=operation,
                atomic_kind=atomic_kind,
                materialize_byte_offset=True,
                allow_dynamic_stride_values=True,
                cache_policy=True,
            )
        )
    return tuple(rules)


def _view_atomic_cmpxchg_rule(
    value_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    materialize_byte_offset: bool | None = None,
    allow_dynamic_stride_values: bool = False,
    cache_policy: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    if materialize_byte_offset is None:
        materialize_byte_offset = dynamic_term_count > 1 or (
            dynamic_term_count != 0 and source_dynamic_count != dynamic_term_count
        )
    if allow_dynamic_stride_values and not materialize_byte_offset:
        raise ValueError("dynamic stride values require byte-offset materialization")
    operands = {
        "expected": ValueRef.operand("expected"),
        "replacement": ValueRef.operand("replacement"),
        "ptr": ValueRef.operand("view"),
    }
    if dynamic_term_count != 0:
        if materialize_byte_offset:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif source_dynamic_count:
            operands["index"] = ValueRef.operand("indices")
        else:
            operands["index"] = ValueRef.source_memory_dynamic_term()
    return DescriptorRule(
        source_op=view.view_atomic_cmpxchg,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", source_dynamic_count),
            Guard.value_type("expected", value_type),
            Guard.value_type("replacement", value_type),
            Guard.value_type("old", value_type),
            Guard.attr_kind("success_ordering", "enum"),
            Guard.attr_kind("failure_ordering", "enum"),
            Guard.attr_kind("scope", "enum"),
            *_cache_policy_guards(cache_policy),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                results={"old": ValueRef.result("old")},
                immediates=_atomic_cmpxchg_immediates(
                    dynamic_term_count=dynamic_term_count,
                    materialize_byte_offset=materialize_byte_offset,
                    cache_policy=cache_policy,
                ),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.ATOMIC_CMPXCHG,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                    allow_dynamic_stride_values=allow_dynamic_stride_values,
                    cache_policy=cache_policy,
                ),
                source_memory_byte_offset_materializer=(
                    _source_memory_byte_offset_materializer()
                    if materialize_byte_offset
                    else None
                ),
            ),
        ),
    )


def _view_atomic_cmpxchg_rule_variants(
    value_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
) -> tuple[DescriptorRule, ...]:
    materializes_byte_offset = dynamic_term_count > 1 or (
        dynamic_term_count != 0 and source_dynamic_count != dynamic_term_count
    )
    rules = [
        _view_atomic_cmpxchg_rule(
            value_type,
            descriptor_key,
            source_dynamic_count=source_dynamic_count,
            dynamic_term_count=dynamic_term_count,
            element_byte_count=element_byte_count,
            allow_dynamic_stride_values=materializes_byte_offset,
        )
    ]
    rules.append(
        _view_atomic_cmpxchg_rule(
            value_type,
            descriptor_key,
            source_dynamic_count=source_dynamic_count,
            dynamic_term_count=dynamic_term_count,
            element_byte_count=element_byte_count,
            allow_dynamic_stride_values=materializes_byte_offset,
            cache_policy=True,
        )
    )
    if dynamic_term_count == 1 and not materializes_byte_offset:
        rules.append(
            _view_atomic_cmpxchg_rule(
                value_type,
                descriptor_key,
                source_dynamic_count=source_dynamic_count,
                dynamic_term_count=dynamic_term_count,
                element_byte_count=element_byte_count,
                materialize_byte_offset=True,
                allow_dynamic_stride_values=True,
            )
        )
        rules.append(
            _view_atomic_cmpxchg_rule(
                value_type,
                descriptor_key,
                source_dynamic_count=source_dynamic_count,
                dynamic_term_count=dynamic_term_count,
                element_byte_count=element_byte_count,
                materialize_byte_offset=True,
                allow_dynamic_stride_values=True,
                cache_policy=True,
            )
        )
    return tuple(rules)


def _memory_descriptor_key(operation: str, suffix: str, *, dynamic: bool) -> str:
    return (
        f"llvmir.{operation}.indexed.{suffix}"
        if dynamic
        else f"llvmir.{operation}.{suffix}"
    )


def _atomic_descriptor_key(
    form: str,
    operation: str,
    suffix: str,
    *,
    dynamic: bool,
) -> str:
    if form == "cmpxchg":
        return (
            f"llvmir.atomic.cmpxchg.indexed.{suffix}"
            if dynamic
            else f"llvmir.atomic.cmpxchg.{suffix}"
        )
    return (
        f"llvmir.atomic.{form}.indexed.{operation}.{suffix}"
        if dynamic
        else f"llvmir.atomic.{form}.{operation}.{suffix}"
    )


def _buffer_view_rule() -> ValueAliasRule:
    return ValueAliasRule(
        source_op=buffer.buffer_view,
        source=ValueRef.operand("buffer"),
        result=ValueRef.result("result"),
    )


def _buffer_alloca_rule(memory_space: str) -> DescriptorRule:
    descriptor = _descriptor(f"llvmir.alloca.{memory_space}.i8")
    return DescriptorRule(
        source_op=buffer.buffer_alloca,
        descriptor=descriptor,
        guards=(
            Guard.value_type("byte_length", _OFFSET),
            Guard.enum_attr_equals("memory_space", memory_space),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"byte_length": ValueRef.operand("byte_length")},
                results={"ptr": ValueRef.result("result")},
                immediates={
                    "base_alignment": AttrProject.direct("base_alignment"),
                },
            ),
        ),
    )


def _scalar_arithmetic_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in ((_I32, "i32"), (_I64, "i64")):
        for source_op, stem in (
            (scalar_arithmetic.scalar_addi, "add"),
            (scalar_arithmetic.scalar_subi, "sub"),
            (scalar_arithmetic.scalar_muli, "mul"),
            (scalar_arithmetic.scalar_divui, "udiv"),
            (scalar_arithmetic.scalar_divsi, "sdiv"),
            (scalar_arithmetic.scalar_remui, "urem"),
            (scalar_arithmetic.scalar_remsi, "srem"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
        for source_op, predicate in (
            (scalar_arithmetic.scalar_minsi, "slt"),
            (scalar_arithmetic.scalar_maxsi, "sgt"),
            (scalar_arithmetic.scalar_minui, "ult"),
            (scalar_arithmetic.scalar_maxui, "ugt"),
        ):
            rules.append(
                _scalar_minmax_rule(source_op, type_pattern, suffix, predicate)
            )
    for type_pattern, suffix in ((_F32, "f32"), (_F64, "f64")):
        for source_op, stem in (
            (scalar_arithmetic.scalar_negf, "neg"),
            (scalar_arithmetic.scalar_absf, "abs"),
        ):
            rules.append(
                _unary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
        for source_op, stem in (
            (scalar_arithmetic.scalar_addf, "add"),
            (scalar_arithmetic.scalar_subf, "sub"),
            (scalar_arithmetic.scalar_mulf, "mul"),
            (scalar_arithmetic.scalar_divf, "div"),
        ):
            rules.append(
                _binary_rule(
                    source_op,
                    type_pattern,
                    f"llvmir.{stem}.{suffix}",
                    source_instance_flags=True,
                )
            )
        for source_op, stem in (
            (scalar_arithmetic.scalar_minnumf, "minnum"),
            (scalar_arithmetic.scalar_maxnumf, "maxnum"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
        rules.append(
            _ternary_rule(
                scalar_math.scalar_fmaf,
                type_pattern,
                f"llvmir.fma.{suffix}",
            )
        )
    return tuple(rules)


def _scalar_minmax_rule(
    source_op: Op, type_pattern: TypePattern, suffix: str, predicate: str
) -> DescriptorRule:
    compare_descriptor = _descriptor(f"llvmir.cmp.{predicate}.{suffix}")
    select_descriptor = _descriptor(f"llvmir.select.{suffix}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=compare_descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("take_lhs")},
                result_types={"dst": _I1},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("take_lhs"),
                    "true_value": ValueRef.operand("lhs"),
                    "false_value": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _scalar_bitwise_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in (
        (_I1, "i1"),
        (_I8, "i8"),
        (_I16, "i16"),
        (_I32, "i32"),
        (_I64, "i64"),
    ):
        for source_op, stem in (
            (scalar_bitwise.scalar_andi, "and"),
            (scalar_bitwise.scalar_ori, "or"),
            (scalar_bitwise.scalar_xori, "xor"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
    for type_pattern, suffix in (
        (_I8, "i8"),
        (_I16, "i16"),
        (_I32, "i32"),
        (_I64, "i64"),
    ):
        for source_op, stem in (
            (scalar_bitwise.scalar_shli, "shl"),
            (scalar_bitwise.scalar_shrui, "lshr"),
            (scalar_bitwise.scalar_shrsi, "ashr"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
    return tuple(rules)


def _index_madd_rule() -> DescriptorRule:
    multiply = _descriptor("llvmir.mul.i64")
    add = _descriptor("llvmir.add.i64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=add,
        guards=_typed_guards(("a", "b", "c", "result"), _INDEX),
        emit=(
            _op_emit(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _INDEX},
            ),
            _op_emit(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_bitwise_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _binary_rule(source_op, _INDEX, f"llvmir.{stem}.i64")
        for source_op, stem in (
            (index.index_andi, "and"),
            (index.index_ori, "or"),
            (index.index_xori, "xor"),
            (index.index_shli, "shl"),
            (index.index_shrui, "lshr"),
            (index.index_shrsi, "ashr"),
        )
    )


def _index_minmax_rule(source_op: Op, predicate: str) -> DescriptorRule:
    compare_descriptor = _descriptor(f"llvmir.cmp.{predicate}.i64")
    select_descriptor = _descriptor("llvmir.select.i64")
    return DescriptorRule(
        source_op=source_op,
        descriptor=compare_descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), _INDEX),
        emit=(
            _op_emit(
                descriptor=compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("take_lhs")},
                result_types={"dst": _I1},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("take_lhs"),
                    "true_value": ValueRef.operand("lhs"),
                    "false_value": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_minmax_rules() -> tuple[DescriptorRule, ...]:
    return (
        _index_minmax_rule(index.index_min, "slt"),
        _index_minmax_rule(index.index_max, "sgt"),
    )


def _vector_minmax_rule(
    source_op: Op, lane_count: int, predicate: str
) -> DescriptorRule:
    value_type = _vector_type("i32", lane_count)
    mask_type = _vector_type("i1", lane_count)
    value_suffix = _vector_suffix("i32", lane_count)
    compare_descriptor = _descriptor(f"llvmir.cmp.{predicate}.{value_suffix}")
    select_descriptor = _descriptor(f"llvmir.select.{value_suffix}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=compare_descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), value_type),
        emit=(
            _op_emit(
                descriptor=compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("take_lhs")},
                result_types={"dst": mask_type},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("take_lhs"),
                    "true_value": ValueRef.operand("lhs"),
                    "false_value": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _vector_arithmetic_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element, source_ops in (
        (
            "i32",
            (
                (vector.vector_addi, "add"),
                (vector.vector_subi, "sub"),
                (vector.vector_muli, "mul"),
            ),
        ),
        (
            "f32",
            (
                (vector.vector_addf, "add"),
                (vector.vector_subf, "sub"),
                (vector.vector_mulf, "mul"),
                (vector.vector_divf, "div"),
            ),
        ),
    ):
        for lane_count in _VECTOR_LANE_COUNTS:
            type_pattern = _vector_type(element, lane_count)
            suffix = _vector_suffix(element, lane_count)
            for source_op, stem in source_ops:
                rules.append(
                    _binary_rule(
                        source_op,
                        type_pattern,
                        f"llvmir.{stem}.{suffix}",
                        source_instance_flags=element == "f32",
                    )
                )
        vector_type = _vector_type(element, 1)
        for source_op, stem in source_ops:
            rules.append(
                _binary_rule(
                    source_op,
                    vector_type,
                    f"llvmir.{stem}.{element}",
                    source_instance_flags=element == "f32",
                )
            )
    for lane_count in _VECTOR_LANE_COUNTS:
        for source_op, predicate in (
            (vector.vector_minsi, "slt"),
            (vector.vector_maxsi, "sgt"),
            (vector.vector_minui, "ult"),
            (vector.vector_maxui, "ugt"),
        ):
            rules.append(_vector_minmax_rule(source_op, lane_count, predicate))
    for source_op, stem in (
        (vector.vector_minnumf, "minnum"),
        (vector.vector_maxnumf, "maxnum"),
    ):
        for lane_count in _VECTOR_LANE_COUNTS:
            type_pattern = _vector_type("f32", lane_count)
            suffix = _vector_suffix("f32", lane_count)
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
        rules.append(
            _binary_rule(source_op, _vector_type("f32", 1), f"llvmir.{stem}.f32")
        )
    for lane_count in _VECTOR_LANE_COUNTS:
        type_pattern = _vector_type("f32", lane_count)
        suffix = _vector_suffix("f32", lane_count)
        for source_op, stem in (
            (vector.vector_negf, "neg"),
            (vector.vector_absf, "abs"),
        ):
            rules.append(
                _unary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
    for source_op, stem in (
        (vector.vector_negf, "neg"),
        (vector.vector_absf, "abs"),
    ):
        rules.append(
            _unary_rule(source_op, _vector_type("f32", 1), f"llvmir.{stem}.f32")
        )
    for lane_count in _VECTOR_LANE_COUNTS:
        type_pattern = _vector_type("f32", lane_count)
        suffix = _vector_suffix("f32", lane_count)
        rules.append(
            _ternary_rule(vector.vector_fmaf, type_pattern, f"llvmir.fma.{suffix}")
        )
    rules.append(
        _ternary_rule(
            vector.vector_fmaf,
            _vector_type("f32", 1),
            "llvmir.fma.f32",
        )
    )
    return tuple(rules)


def _vector_bitwise_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element, source_ops in (
        (
            "i1",
            (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
            ),
        ),
        (
            "i8",
            (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
                (vector.vector_shli, "shl"),
                (vector.vector_shrui, "lshr"),
                (vector.vector_shrsi, "ashr"),
            ),
        ),
        (
            "i16",
            (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
                (vector.vector_shli, "shl"),
                (vector.vector_shrui, "lshr"),
                (vector.vector_shrsi, "ashr"),
            ),
        ),
        (
            "i32",
            (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
                (vector.vector_shli, "shl"),
                (vector.vector_shrui, "lshr"),
                (vector.vector_shrsi, "ashr"),
            ),
        ),
    ):
        for lane_count in _VECTOR_LANE_COUNTS:
            type_pattern = _vector_type(element, lane_count)
            suffix = _vector_suffix(element, lane_count)
            for source_op, stem in source_ops:
                rules.append(
                    _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
                )
        vector_type = _vector_type(element, 1)
        for source_op, stem in source_ops:
            rules.append(
                _binary_rule(source_op, vector_type, f"llvmir.{stem}.{element}")
            )
    return tuple(rules)


def _compare_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix, result_type, source_op, predicates in (
        (_INDEX, "i64", _I1, index.index_cmp, _INTEGER_PREDICATES),
        (_OFFSET, "i64", _I1, index.index_cmp, _INTEGER_PREDICATES),
        (_I32, "i32", _I1, scalar_comparison.scalar_cmpi, _INTEGER_PREDICATES),
        (_I64, "i64", _I1, scalar_comparison.scalar_cmpi, _INTEGER_PREDICATES),
        (_F32, "f32", _I1, scalar_comparison.scalar_cmpf, _FLOAT_PREDICATES),
        (_F64, "f64", _I1, scalar_comparison.scalar_cmpf, _FLOAT_PREDICATES),
    ):
        rules.extend(
            (
                _compare_rule(
                    source_op,
                    type_pattern,
                    result_type,
                    predicate,
                    f"llvmir.cmp.{predicate}.{suffix}",
                )
            )
            for predicate in predicates
        )
    for lane_count in _VECTOR_LANE_COUNTS:
        mask_type = _vector_type("i1", lane_count)
        for element, source_op, predicates in (
            ("i32", vector.vector_cmpi, _INTEGER_PREDICATES),
            ("f32", vector.vector_cmpf, _FLOAT_PREDICATES),
        ):
            value_type = _vector_type(element, lane_count)
            suffix = _vector_suffix(element, lane_count)
            rules.extend(
                (
                    _compare_rule(
                        source_op,
                        value_type,
                        mask_type,
                        predicate,
                        f"llvmir.cmp.{predicate}.{suffix}",
                    )
                )
                for predicate in predicates
            )
    for element, source_op, predicates in (
        ("i32", vector.vector_cmpi, _INTEGER_PREDICATES),
        ("f32", vector.vector_cmpf, _FLOAT_PREDICATES),
    ):
        value_type = _vector_type(element, 1)
        mask_type = _vector_type("i1", 1)
        rules.extend(
            (
                _compare_rule(
                    source_op,
                    value_type,
                    mask_type,
                    predicate,
                    f"llvmir.cmp.{predicate}.{element}",
                )
            )
            for predicate in predicates
        )
    return tuple(rules)


def _cast_rules() -> tuple[DescriptorRule, ...]:
    return (
        tuple(
            _cast_rule(
                source_op,
                Scalar(source_type),
                Scalar(result_type),
                f"llvmir.{stem}.{source_type}.{result_type}",
            )
            for source_op, stem, source_type, result_type in _SCALAR_CAST_SPECS
        )
        + tuple(
            _cast_rule(
                source_op,
                _vector_type(source_type, lane_count),
                _vector_type(result_type, lane_count),
                "llvmir."
                f"{stem}."
                f"{_vector_suffix(source_type, lane_count)}."
                f"{_vector_suffix(result_type, lane_count)}",
            )
            for source_op, stem, source_type, result_type in _VECTOR_CAST_SPECS
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _cast_rule(
                source_op,
                _vector_type(source_type, 1),
                _vector_type(result_type, 1),
                f"llvmir.{stem}.{source_type}.{result_type}",
            )
            for source_op, stem, source_type, result_type in _VECTOR_CAST_SPECS
        )
        + tuple(
            _cast_rule(
                vector.vector_bitcast,
                _vector_type(source_type, source_lane_count),
                _vector_type(result_type, result_lane_count),
                "llvmir."
                f"bitcast."
                f"{_low_vector_suffix(source_type, source_lane_count)}."
                f"{_low_vector_suffix(result_type, result_lane_count)}",
            )
            for (
                source_type,
                source_lane_count,
                result_type,
                result_lane_count,
            ) in _BITCAST_RESHAPE_SPECS
        )
    )


def _bitcast_alias_rule(
    source_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_bitcast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
    )


def _index_cast_rules() -> tuple[DescriptorRule | ValueAliasRule, ...]:
    return (
        _index_cast_rule(_I32, _INDEX, "llvmir.sext.i32.i64"),
        _index_cast_rule(_I32, _OFFSET, "llvmir.zext.i32.i64"),
        _index_cast_rule(_INDEX, _I32, "llvmir.trunc.i64.i32"),
        _index_cast_rule(_OFFSET, _I32, "llvmir.trunc.i64.i32"),
        _index_cast_alias_rule(_I64, _INDEX),
        _index_cast_alias_rule(_I64, _OFFSET),
        _index_cast_alias_rule(_INDEX, _I64),
        _index_cast_alias_rule(_OFFSET, _I64),
        _index_cast_alias_rule(_INDEX, _OFFSET),
        _index_cast_alias_rule(_OFFSET, _INDEX),
    )


def _select_rules() -> tuple[DescriptorRule, ...]:
    return (
        tuple(
            _select_rule(type_pattern, f"llvmir.select.{suffix}")
            for type_pattern, suffix in (
                (_INDEX, "i64"),
                (_OFFSET, "i64"),
                (_I1, "i1"),
                (_I32, "i32"),
                (_I64, "i64"),
                (_F32, "f32"),
                (_F64, "f64"),
            )
        )
        + tuple(
            _select_rule(
                _vector_type(element, lane_count),
                f"llvmir.select.{_vector_suffix(element, lane_count)}",
            )
            for element in _VECTOR_SELECT_TYPES
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _select_rule(_vector_type(element, 1), f"llvmir.select.{element}")
            for element in _VECTOR_SELECT_TYPES
            if element in ("i32", "i64", "f32", "f64")
        )
    )


def _clampf_number_rule(
    source_op: Op,
    type_pattern: TypePattern,
    maxnum_descriptor_key: str,
    minnum_descriptor_key: str,
) -> DescriptorRule:
    maxnum_descriptor = _descriptor(maxnum_descriptor_key)
    minnum_descriptor = _descriptor(minnum_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=maxnum_descriptor,
        guards=(
            Guard.enum_attr_equals("mode", "number"),
            *_typed_guards(("value", "lower", "upper", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                descriptor=maxnum_descriptor,
                operands={
                    "lhs": ValueRef.operand("value"),
                    "rhs": ValueRef.operand("lower"),
                },
                results={"dst": ValueRef.temporary("lower_clamped")},
                result_types={"dst": type_pattern},
            ),
            _op_emit(
                descriptor=minnum_descriptor,
                operands={
                    "lhs": ValueRef.temporary("lower_clamped"),
                    "rhs": ValueRef.operand("upper"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _clampf_ordered_rule(
    source_op: Op,
    type_pattern: TypePattern,
    mask_pattern: TypePattern,
    lower_compare_descriptor_key: str,
    upper_compare_descriptor_key: str,
    select_descriptor_key: str,
) -> DescriptorRule:
    lower_compare_descriptor = _descriptor(lower_compare_descriptor_key)
    upper_compare_descriptor = _descriptor(upper_compare_descriptor_key)
    select_descriptor = _descriptor(select_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=lower_compare_descriptor,
        guards=(
            Guard.enum_attr_equals("mode", "ordered"),
            *_typed_guards(("value", "lower", "upper", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                descriptor=lower_compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("value"),
                    "rhs": ValueRef.operand("lower"),
                },
                results={"dst": ValueRef.temporary("below_lower")},
                result_types={"dst": mask_pattern},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("below_lower"),
                    "true_value": ValueRef.operand("lower"),
                    "false_value": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.temporary("lower_clamped")},
                result_types={"dst": type_pattern},
            ),
            _op_emit(
                descriptor=upper_compare_descriptor,
                operands={
                    "lhs": ValueRef.temporary("lower_clamped"),
                    "rhs": ValueRef.operand("upper"),
                },
                results={"dst": ValueRef.temporary("above_upper")},
                result_types={"dst": mask_pattern},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("above_upper"),
                    "true_value": ValueRef.operand("upper"),
                    "false_value": ValueRef.temporary("lower_clamped"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _scalar_clampf_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in ((_F32, "f32"), (_F64, "f64")):
        rules.append(
            _clampf_number_rule(
                scalar_arithmetic.scalar_clampf,
                type_pattern,
                f"llvmir.maxnum.{suffix}",
                f"llvmir.minnum.{suffix}",
            )
        )
        rules.append(
            _clampf_ordered_rule(
                scalar_arithmetic.scalar_clampf,
                type_pattern,
                _I1,
                f"llvmir.cmp.olt.{suffix}",
                f"llvmir.cmp.ogt.{suffix}",
                f"llvmir.select.{suffix}",
            )
        )
    return tuple(rules)


def _vector_clampf_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for lane_count in _VECTOR_LANE_COUNTS:
        type_pattern = _vector_type("f32", lane_count)
        mask_pattern = _vector_type("i1", lane_count)
        suffix = _vector_suffix("f32", lane_count)
        rules.append(
            _clampf_number_rule(
                vector.vector_clampf,
                type_pattern,
                f"llvmir.maxnum.{suffix}",
                f"llvmir.minnum.{suffix}",
            )
        )
        rules.append(
            _clampf_ordered_rule(
                vector.vector_clampf,
                type_pattern,
                mask_pattern,
                f"llvmir.cmp.olt.{suffix}",
                f"llvmir.cmp.ogt.{suffix}",
                f"llvmir.select.{suffix}",
            )
        )
    type_pattern = _vector_type("f32", 1)
    mask_pattern = _vector_type("i1", 1)
    rules.append(
        _clampf_number_rule(
            vector.vector_clampf,
            type_pattern,
            "llvmir.maxnum.f32",
            "llvmir.minnum.f32",
        )
    )
    rules.append(
        _clampf_ordered_rule(
            vector.vector_clampf,
            type_pattern,
            mask_pattern,
            "llvmir.cmp.olt.f32",
            "llvmir.cmp.ogt.f32",
            "llvmir.select.f32",
        )
    )
    return tuple(rules)


def _clampf_rules() -> tuple[DescriptorRule, ...]:
    return (*_scalar_clampf_rules(), *_vector_clampf_rules())


def _vector_constant_rules() -> tuple[DescriptorRule, ...]:
    return (
        tuple(
            _vector_const_i_rule(element, lane_count, minimum=minimum, maximum=maximum)
            for element, minimum, maximum in (
                ("i8", _I8_MIN, _I8_MAX),
                ("i16", _I16_MIN, _I16_MAX),
                ("i32", _I32_MIN, _I32_MAX),
            )
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _vector_const_i_rule("i64", lane_count)
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _vector_const_float_rule(element, lane_count)
            for element in ("f16", "bf16", "f32", "f64")
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + (
            _const_i8_rule(vector.vector_constant, _vector_type("i8", 1)),
            _const_i16_rule(vector.vector_constant, _vector_type("i16", 1)),
            _const_i32_rule(vector.vector_constant, _vector_type("i32", 1)),
            _const_i64_rule(vector.vector_constant, _vector_type("i64", 1)),
            _const_float_rule(
                vector.vector_constant,
                _vector_type("f16", 1),
                "llvmir.const.f16",
            ),
            _const_float_rule(
                vector.vector_constant,
                _vector_type("bf16", 1),
                "llvmir.const.bf16",
            ),
            _const_float_rule(
                vector.vector_constant,
                _vector_type("f32", 1),
                "llvmir.const.f32",
            ),
            _const_float_rule(
                vector.vector_constant,
                _vector_type("f64", 1),
                "llvmir.const.f64",
            ),
        )
    )


def _one_lane_splat_rule(element: str) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_splat,
        source=ValueRef.operand("scalar"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("scalar", Scalar(element)),
            Guard.value_type("result", _vector_type(element, 1)),
        ),
    )


def _splat_rule(element: str, lane_count: int) -> DescriptorRule:
    scalar_type = Scalar(element)
    result_type = _vector_type(element, lane_count)
    descriptor = _descriptor(f"llvmir.splat.{_vector_suffix(element, lane_count)}")
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", scalar_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _one_lane_from_elements_rule(element: str) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_from_elements,
        source=ValueRef.operand("elements", element=0),
        result=ValueRef.result("result"),
        guards=(
            Guard.operand_segment_count("elements", 1),
            Guard.value_type("elements", Scalar(element)),
            Guard.value_type("result", _vector_type(element, 1)),
        ),
    )


def _from_elements_rule(element: str, lane_count: int) -> DescriptorRule:
    scalar_type = Scalar(element)
    result_type = _vector_type(element, lane_count)
    descriptor = _descriptor(
        f"llvmir.from_elements.{_vector_suffix(element, lane_count)}"
    )
    return DescriptorRule(
        source_op=vector.vector_from_elements,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("elements", lane_count),
            Guard.value_type("elements", scalar_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    f"lane{lane_index}": ValueRef.operand(
                        "elements", element=lane_index
                    )
                    for lane_index in range(lane_count)
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _one_lane_extract_rule(element: str) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_extract,
        source=ValueRef.operand("source"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("source", _vector_type(element, 1)),
            Guard.value_type("result", Scalar(element)),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=0
            ),
        ),
    )


def _extract_rule(element: str, lane_count: int) -> DescriptorRule:
    source_type = _vector_type(element, lane_count)
    result_type = Scalar(element)
    descriptor = _descriptor(f"llvmir.extract.{_vector_suffix(element, lane_count)}")
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=lane_count - 1
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "lane": AttrProject.i64_array_element("static_indices", element=0)
                },
            ),
        ),
    )


def _one_lane_insert_rule(element: str) -> ValueAliasRule:
    vector_type = _vector_type(element, 1)
    return ValueAliasRule(
        source_op=vector.vector_insert,
        source=ValueRef.operand("value"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("value", Scalar(element)),
            Guard.value_type("dest", vector_type),
            Guard.value_type("result", vector_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=0
            ),
        ),
    )


def _insert_rule(element: str, lane_count: int) -> DescriptorRule:
    value_type = Scalar(element)
    dest_type = _vector_type(element, lane_count)
    descriptor = _descriptor(f"llvmir.insert.{_vector_suffix(element, lane_count)}")
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", value_type),
            Guard.value_type("dest", dest_type),
            Guard.value_type("result", dest_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=lane_count - 1
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "dest": ValueRef.operand("dest"),
                    "value": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "lane": AttrProject.i64_array_element("static_indices", element=0)
                },
            ),
        ),
    )


def _dynamic_insert_rule(element: str, lane_count: int) -> DescriptorRule:
    value_type = Scalar(element)
    dest_type = _vector_type(element, lane_count)
    descriptor = _descriptor(
        f"llvmir.insert.dynamic.{_vector_suffix(element, lane_count)}"
    )
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", value_type),
            Guard.value_type("dest", dest_type),
            Guard.value_type("indices", _INDEX),
            Guard.value_type("result", dest_type),
            Guard.operand_segment_count("indices", 1),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices",
                element=0,
                minimum=_DYNAMIC_INDEX,
                maximum=_DYNAMIC_INDEX,
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "dest": ValueRef.operand("dest"),
                    "value": ValueRef.operand("value"),
                    "index": ValueRef.operand("indices"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _shuffle_rule(element: str, lane_count: int) -> DescriptorRule:
    type_pattern = _vector_type(element, lane_count)
    descriptor = _descriptor(f"llvmir.shuffle.{_vector_suffix(element, lane_count)}")
    return DescriptorRule(
        source_op=vector.vector_shuffle,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", type_pattern),
            Guard.value_type("result", type_pattern),
            Guard.i64_array_count("source_lanes", lane_count),
            Guard.i64_array_elements_range(
                "source_lanes", minimum=0, maximum=lane_count - 1
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    f"lane{lane_index}": AttrProject.i64_array_element(
                        "source_lanes", element=lane_index
                    )
                    for lane_index in range(lane_count)
                },
            ),
        ),
    )


def _concat_rule(
    element: str,
    *,
    input_lane_count: int,
    input_count: int,
) -> DescriptorRule:
    input_type = _vector_type(element, input_lane_count)
    result_type = _vector_type(element, input_lane_count * input_count)
    descriptor = _descriptor(
        "llvmir.concat."
        f"{_vector_suffix(element, input_lane_count)}."
        f"{_vector_suffix(element, input_lane_count * input_count)}"
    )
    return DescriptorRule(
        source_op=vector.vector_concat,
        descriptor=descriptor,
        guards=(
            Guard.i64_range("axis", 0, 0),
            Guard.operand_segment_count("inputs", input_count),
            Guard.value_type("inputs", input_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    f"input{input_ordinal}": ValueRef.operand(
                        "inputs", element=input_ordinal
                    )
                    for input_ordinal in range(input_count)
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _transpose_4x4_rule(element: str) -> DescriptorRule:
    source_type = _vector_shape_type(element, 4, 4)
    flat_type = _vector_type(element, 16)
    descriptor = _descriptor(f"llvmir.shuffle.{_vector_suffix(element, 16)}")
    lanes = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
    return DescriptorRule(
        source_op=vector.vector_transpose,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", source_type),
            Guard.i64_array_count("permutation", 2),
            Guard.i64_array_element_range(
                "permutation", element=0, minimum=1, maximum=1
            ),
            Guard.i64_array_element_range(
                "permutation", element=1, minimum=0, maximum=0
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                result_types={"dst": flat_type},
                immediates={
                    f"lane{lane_index}": lane for lane_index, lane in enumerate(lanes)
                },
            ),
        ),
    )


def _slice_rule(
    element: str,
    *,
    source_lane_count: int,
    result_lane_count: int,
) -> DescriptorRule:
    source_type = _vector_type(element, source_lane_count)
    result_type = _vector_type(element, result_lane_count)
    descriptor = _descriptor(
        "llvmir.slice."
        f"{_vector_suffix(element, source_lane_count)}."
        f"{_vector_suffix(element, result_lane_count)}"
    )
    return DescriptorRule(
        source_op=vector.vector_slice,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.operand_segment_count("offsets", 0),
            Guard.i64_array_count("static_offsets", 1),
            Guard.i64_array_element_range(
                "static_offsets",
                element=0,
                minimum=0,
                maximum=source_lane_count - result_lane_count,
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "offset": AttrProject.i64_array_element("static_offsets", element=0)
                },
            ),
        ),
    )


def _structural_vector_rules() -> tuple[DescriptorRule | ValueAliasRule, ...]:
    rules: list[DescriptorRule | ValueAliasRule] = []
    for element in _STRUCTURAL_VECTOR_TYPES:
        for lane_count in _STRUCTURAL_VECTOR_LANE_COUNTS:
            rules.append(_splat_rule(element, lane_count))
            rules.append(_from_elements_rule(element, lane_count))
            rules.append(_extract_rule(element, lane_count))
            rules.append(_insert_rule(element, lane_count))
            rules.append(_dynamic_insert_rule(element, lane_count))
            rules.append(_shuffle_rule(element, lane_count))
        rules.append(_concat_rule(element, input_lane_count=4, input_count=4))
        rules.append(_transpose_4x4_rule(element))
        rules.append(_slice_rule(element, source_lane_count=4, result_lane_count=2))
        rules.append(_slice_rule(element, source_lane_count=16, result_lane_count=4))
        rules.append(
            _bitcast_alias_rule(
                _vector_type(element, 16), _vector_shape_type(element, 4, 4)
            )
        )
        rules.append(
            _bitcast_alias_rule(
                _vector_shape_type(element, 4, 4), _vector_type(element, 16)
            )
        )
        rules.append(_one_lane_splat_rule(element))
        rules.append(_one_lane_from_elements_rule(element))
        rules.append(_one_lane_extract_rule(element))
        rules.append(_one_lane_insert_rule(element))
    return tuple(rules)


def _memory_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element, element_byte_count in _MEMORY_VALUE_TYPES:
        for source_dynamic_count, dynamic_term_count in (
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
        ):
            rules.extend(
                _view_load_rule_variants(
                    view.view_load,
                    "view",
                    Scalar(element),
                    _memory_descriptor_key(
                        "load", element, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
            rules.extend(
                _view_store_rule_variants(
                    view.view_store,
                    "view",
                    Scalar(element),
                    _memory_descriptor_key(
                        "store", element, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
            for lane_count in _VECTOR_LANE_COUNTS:
                vector_type = _vector_type(element, lane_count)
                vector_suffix = _vector_suffix(element, lane_count)
                rules.extend(
                    _view_load_rule_variants(
                        vector.vector_load,
                        "view",
                        vector_type,
                        _memory_descriptor_key(
                            "load", vector_suffix, dynamic=dynamic_term_count != 0
                        ),
                        source_dynamic_count=source_dynamic_count,
                        dynamic_term_count=dynamic_term_count,
                        element_byte_count=element_byte_count,
                        vector_lane_count=lane_count,
                    )
                )
                rules.extend(
                    _view_store_rule_variants(
                        vector.vector_store,
                        "view",
                        vector_type,
                        _memory_descriptor_key(
                            "store", vector_suffix, dynamic=dynamic_term_count != 0
                        ),
                        source_dynamic_count=source_dynamic_count,
                        dynamic_term_count=dynamic_term_count,
                        element_byte_count=element_byte_count,
                        vector_lane_count=lane_count,
                    )
                )
            rank2_tile_type = _vector_shape_type(element, 4, 4)
            rank2_tile_suffix = _vector_suffix(element, 16)
            rules.extend(
                _view_load_rule_variants(
                    vector.vector_load,
                    "view",
                    rank2_tile_type,
                    _memory_descriptor_key(
                        "load", rank2_tile_suffix, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                    vector_lane_count=16,
                    low_result_type=_vector_type(element, 16),
                )
            )
            rules.extend(
                _view_store_rule_variants(
                    vector.vector_store,
                    "view",
                    rank2_tile_type,
                    _memory_descriptor_key(
                        "store", rank2_tile_suffix, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                    vector_lane_count=16,
                )
            )
            vector_type = _vector_type(element, 1)
            rules.extend(
                _view_load_rule_variants(
                    vector.vector_load,
                    "view",
                    vector_type,
                    _memory_descriptor_key(
                        "load", element, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
            rules.extend(
                _view_store_rule_variants(
                    vector.vector_store,
                    "view",
                    vector_type,
                    _memory_descriptor_key(
                        "store", element, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
    return tuple(rules)


def _atomic_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    dynamic_cases = (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    )
    for element, element_byte_count in _ATOMIC_INTEGER_VALUE_TYPES:
        value_type = Scalar(element)
        for atomic_kind, llvmir_operation in _ATOMIC_INTEGER_KINDS:
            for source_dynamic_count, dynamic_term_count in dynamic_cases:
                rules.extend(
                    _view_atomic_rule_variants(
                        view.view_atomic_reduce,
                        value_type,
                        _atomic_descriptor_key(
                            "reduce",
                            llvmir_operation,
                            element,
                            dynamic=dynamic_term_count != 0,
                        ),
                        source_dynamic_count=source_dynamic_count,
                        dynamic_term_count=dynamic_term_count,
                        element_byte_count=element_byte_count,
                        operation=SourceMemoryOperation.ATOMIC_REDUCE,
                        atomic_kind=atomic_kind,
                    )
                )
        for atomic_kind, llvmir_operation in _ATOMIC_INTEGER_RMW_KINDS:
            for source_dynamic_count, dynamic_term_count in dynamic_cases:
                rules.extend(
                    _view_atomic_rule_variants(
                        view.view_atomic_rmw,
                        value_type,
                        _atomic_descriptor_key(
                            "rmw",
                            llvmir_operation,
                            element,
                            dynamic=dynamic_term_count != 0,
                        ),
                        source_dynamic_count=source_dynamic_count,
                        dynamic_term_count=dynamic_term_count,
                        element_byte_count=element_byte_count,
                        operation=SourceMemoryOperation.ATOMIC_RMW,
                        atomic_kind=atomic_kind,
                    )
                )
        for source_dynamic_count, dynamic_term_count in dynamic_cases:
            rules.extend(
                _view_atomic_cmpxchg_rule_variants(
                    value_type,
                    _atomic_descriptor_key(
                        "cmpxchg",
                        "cmpxchg",
                        element,
                        dynamic=dynamic_term_count != 0,
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
    for element, element_byte_count in _ATOMIC_FLOAT_VALUE_TYPES:
        value_type = Scalar(element)
        for atomic_kind, llvmir_operation in _ATOMIC_FLOAT_KINDS:
            for source_dynamic_count, dynamic_term_count in dynamic_cases:
                rules.extend(
                    _view_atomic_rule_variants(
                        view.view_atomic_reduce,
                        value_type,
                        _atomic_descriptor_key(
                            "reduce",
                            llvmir_operation,
                            element,
                            dynamic=dynamic_term_count != 0,
                        ),
                        source_dynamic_count=source_dynamic_count,
                        dynamic_term_count=dynamic_term_count,
                        element_byte_count=element_byte_count,
                        operation=SourceMemoryOperation.ATOMIC_REDUCE,
                        atomic_kind=atomic_kind,
                    )
                )
        for atomic_kind, llvmir_operation in _ATOMIC_FLOAT_RMW_KINDS:
            for source_dynamic_count, dynamic_term_count in dynamic_cases:
                rules.extend(
                    _view_atomic_rule_variants(
                        view.view_atomic_rmw,
                        value_type,
                        _atomic_descriptor_key(
                            "rmw",
                            llvmir_operation,
                            element,
                            dynamic=dynamic_term_count != 0,
                        ),
                        source_dynamic_count=source_dynamic_count,
                        dynamic_term_count=dynamic_term_count,
                        element_byte_count=element_byte_count,
                        operation=SourceMemoryOperation.ATOMIC_RMW,
                        atomic_kind=atomic_kind,
                    )
                )
    return tuple(rules)


def _kernel_query_rule(source_op: Op, query: str, dimension: str) -> DescriptorRule:
    descriptor = _descriptor(f"llvmir.kernel.{query}.{dimension}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("dimension", dimension),
            Guard.value_type("result", _INDEX),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _kernel_scalar_query_rule(source_op: Op, query: str) -> DescriptorRule:
    descriptor = _descriptor(f"llvmir.kernel.{query}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(Guard.value_type("result", _INDEX),),
        emit=(
            _op_emit(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _kernel_query_rules() -> tuple[DescriptorRule, ...]:
    source_ops = (
        (kernel.kernel_workitem_id, "workitem_id"),
        (kernel.kernel_workgroup_id, "workgroup_id"),
        (kernel.kernel_workgroup_size, "workgroup_size"),
        (kernel.kernel_workitem_dispatch_id, "workitem_dispatch_id"),
    )
    dimensioned = tuple(
        _kernel_query_rule(source_op, query, dimension)
        for source_op, query in source_ops
        for dimension in _KERNEL_DIMENSIONS
    )
    scalar = tuple(
        _kernel_scalar_query_rule(source_op, query)
        for source_op, query in (
            (kernel.kernel_subgroup_size, "subgroup_size"),
            (kernel.kernel_subgroup_count, "subgroup_count"),
        )
    )
    return dimensioned + scalar


LLVMIR_GENERIC_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "kernel": ALL_KERNEL_OPS,
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

LLVMIR_GENERIC_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="llvmir.generic.core",
    descriptor_set=LLVMIR_GENERIC_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/llvmir/contracts/generic_core.h",
    cases=(
        _const_i1_rule(scalar_conversion.scalar_constant, _I1),
        _const_i8_rule(scalar_conversion.scalar_constant, _I8),
        _const_i16_rule(scalar_conversion.scalar_constant, _I16),
        _const_i32_rule(scalar_conversion.scalar_constant, _I32),
        _const_i64_rule(index.index_constant, _INDEX),
        _const_i64_rule(scalar_conversion.scalar_constant, _I64),
        _const_i64_rule(index.index_constant, _OFFSET),
        _const_float_rule(
            scalar_conversion.scalar_constant,
            _F16,
            "llvmir.const.f16",
        ),
        _const_float_rule(
            scalar_conversion.scalar_constant,
            _BF16,
            "llvmir.const.bf16",
        ),
        _const_float_rule(
            scalar_conversion.scalar_constant,
            _F32,
            "llvmir.const.f32",
        ),
        _const_float_rule(
            scalar_conversion.scalar_constant,
            _F64,
            "llvmir.const.f64",
        ),
        ValueAliasRule(
            source_op=index.index_assume,
            source=ValueRef.operand("values"),
            result=ValueRef.result("results"),
            guards=(Guard.operand_segment_count("values", 1),),
        ),
        _buffer_alloca_rule("private"),
        _buffer_alloca_rule("workgroup"),
        _buffer_view_rule(),
        *_scalar_arithmetic_rules(),
        *_scalar_bitwise_rules(),
        _binary_rule(index.index_add, _INDEX, "llvmir.add.i64"),
        _binary_rule(index.index_sub, _INDEX, "llvmir.sub.i64"),
        _binary_rule(index.index_mul, _INDEX, "llvmir.mul.i64"),
        _binary_rule(index.index_div, _INDEX, "llvmir.udiv.i64"),
        _binary_rule(index.index_rem, _INDEX, "llvmir.urem.i64"),
        _index_madd_rule(),
        *_index_minmax_rules(),
        *_index_bitwise_rules(),
        _binary_rule(index.index_add, _OFFSET, "llvmir.add.i64"),
        _binary_rule(index.index_sub, _OFFSET, "llvmir.sub.i64"),
        *_vector_arithmetic_rules(),
        *_vector_bitwise_rules(),
        *_clampf_rules(),
        *_compare_rules(),
        *_cast_rules(),
        *_index_cast_rules(),
        *_select_rules(),
        *_kernel_query_rules(),
        *_vector_constant_rules(),
        *_structural_vector_rules(),
        *_memory_rules(),
        *_atomic_rules(),
    ),
)
