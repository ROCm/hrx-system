# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wasm source-to-low contract fragment."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
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
    UnsupportedRule,
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
from loom.target.emit.wasm.float_narrowing import float_narrowing_rules
from loom.target.emit.wasm.predicate import predicate_rules
from loom.target.emit.wasm.vector_integer_arithmetic import (
    integer_arithmetic_rules,
)
from loom.target.emit.wasm.vector_shifts import (
    uniform_shift_rules,
    varying_shift_rules,
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
_WORD_STORAGE = Scalar(("i16", "f16", "bf16"))
_V16_BYTE_STORAGE = Vector(_BYTE_STORAGE.elements, lanes=16)
_V8_WORD_STORAGE = Vector(_WORD_STORAGE.elements, lanes=8)
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_V16I8 = Vector("i8", lanes=16)
_V4I32 = Vector("i32", lanes=4)
_V4F32 = Vector("f32", lanes=4)
_V2I64 = Vector("i64", lanes=2)
_V2F64 = Vector("f64", lanes=2)

# A numeric SIMD value is one bit-preserving v128 carrier. SIMD predicates use
# a separately selected all-zero/all-one lane representation and are not
# numeric payloads.
_NUMERIC_V128_TYPES = (
    _V16_BYTE_STORAGE,
    _V8_WORD_STORAGE,
    Vector(("i32", "f32"), lanes=4),
    Vector(("i64", "f64"), lanes=2),
)
_V128_LANE_TYPES = (
    (_BYTE_STORAGE, _V16_BYTE_STORAGE, "i8x16", "_u"),
    (_WORD_STORAGE, _V8_WORD_STORAGE, "i16x8", "_u"),
    (_I32, _V4I32, "i32x4", ""),
    (_F32, _V4F32, "f32x4", ""),
    (_I64, _V2I64, "i64x2", ""),
    (_F64, _V2F64, "f64x2", ""),
)

# Transient integer values smaller than one physical SIMD register retain their
# source lane count while occupying the low lanes of a v128 carrier. Callable
# arguments and results remain restricted to the exact shapes above.
_PARTIAL_I8 = Vector("i8", minimum_lanes=1, maximum_lanes=15)
_PARTIAL_I16 = Vector("i16", minimum_lanes=1, maximum_lanes=7)
_PARTIAL_I32 = Vector("i32", minimum_lanes=1, maximum_lanes=3)
_PARTIAL_I64 = Vector("i64", minimum_lanes=1, maximum_lanes=1)
_PARTIAL_INTEGER_LANE_TYPES = (
    (_I8, _PARTIAL_I8, "i8x16", "_u", 16),
    (_I16, _PARTIAL_I16, "i16x8", "_u", 8),
    (_I32, _PARTIAL_I32, "i32x4", "", 4),
    (_I64, _PARTIAL_I64, "i64x2", "", 2),
)

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
    if type_pattern.kind == "vector":
        if type_pattern.lanes is not None:
            shape = str(type_pattern.lanes)
        elif (
            type_pattern.minimum_lanes is not None
            and type_pattern.maximum_lanes is not None
        ):
            shape = f"[{type_pattern.minimum_lanes}..{type_pattern.maximum_lanes}]"
        else:
            shape = "static"
        return " or ".join(
            f"vector<{shape}x{element}>" for element in type_pattern.elements
        )
    if type_pattern == _I1:
        return "i1 scalar"
    if type_pattern == _I8:
        return "i8 scalar"
    if type_pattern == _BYTE_STORAGE:
        return "i8, f8E4M3, or f8E5M2 scalar"
    if type_pattern == _I16:
        return "i16 scalar"
    if type_pattern == _WORD_STORAGE:
        return "i16, f16, or bf16 scalar"
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


def _const_v128_rule(
    result_type: TypePattern, value: ValueProject, guard: Guard
) -> DescriptorRule:
    descriptor = _descriptor("wasm.v128.const")
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=descriptor,
        guards=(_value_type("result", result_type), guard),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"lo64": value, "hi64": value},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_vector_splat_rule(
    result_type: TypePattern,
    element_type: str,
    shape_name: str,
    immediate: str,
    value: ValueProject,
    guard: Guard,
) -> DescriptorRule:
    # A scalar splat represents repeated lanes without packing them into
    # v128.const's two immediate words and fixed 16-byte payload.
    constant = _descriptor(f"wasm.{element_type}.const")
    splat = _descriptor(f"wasm.{shape_name}.splat")
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=splat,
        guards=(_value_type("result", result_type), guard),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("element")},
                result_types={"dst": Scalar(element_type)},
                immediates={immediate: value},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=splat,
                operands={"value": ValueRef.temporary("element")},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
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
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _unary_rule(
    source_op: Op, type_pattern: TypePattern, descriptor_key: str
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("input", "result"), type_pattern),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _integer_sign_rules(
    value_type: TypePattern, type_name: str, sign_shift: int
) -> Iterable[DescriptorRule]:
    subtract = _descriptor(f"wasm.{type_name}.sub")
    input_value = ValueRef.operand("input")
    for source_op, constant, intermediate_emits, lhs, rhs in (
        (
            scalar_arithmetic.scalar_negi,
            0,
            (),
            ValueRef.temporary("constant"),
            input_value,
        ),
        (
            scalar_arithmetic.scalar_absi,
            sign_shift,
            (
                EmitDescriptorOp(
                    descriptor=_descriptor(f"wasm.{type_name}.shr_s"),
                    operands={
                        "lhs": input_value,
                        "rhs": ValueRef.temporary("constant"),
                    },
                    results={"dst": ValueRef.temporary("sign")},
                    result_types={"dst": value_type},
                ),
                EmitDescriptorOp(
                    descriptor=_descriptor(f"wasm.{type_name}.xor"),
                    operands={"lhs": input_value, "rhs": ValueRef.temporary("sign")},
                    results={"dst": ValueRef.temporary("magnitude")},
                    result_types={"dst": value_type},
                ),
            ),
            ValueRef.temporary("magnitude"),
            ValueRef.temporary("sign"),
        ),
    ):
        yield DescriptorRule(
            source_op=source_op,
            descriptor=subtract,
            guards=_typed_guards(("input", "result"), value_type),
            emit=(
                EmitDescriptorOp(
                    descriptor=_descriptor(f"wasm.{type_name}.const"),
                    results={"dst": ValueRef.temporary("constant")},
                    result_types={"dst": value_type},
                    immediates={f"{type_name}_value": constant},
                    form=DescriptorEmitForm.CONST,
                ),
                *intermediate_emits,
                EmitDescriptorOp(
                    descriptor=subtract,
                    operands={"lhs": lhs, "rhs": rhs},
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        )


def _index_madd_rule() -> DescriptorRule:
    multiply = _descriptor("wasm.i32.mul")
    add = _descriptor("wasm.i32.add")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=add,
        guards=_typed_guards(("a", "b", "c", "result"), _INDEX),
        emit=(
            EmitDescriptorOp(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _INDEX},
            ),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_extrema_rule(
    source_op: Op,
    compare_descriptor_key: str,
) -> DescriptorRule:
    compare = _descriptor(compare_descriptor_key)
    select = _descriptor("wasm.i32.select")
    return DescriptorRule(
        source_op=source_op,
        descriptor=select,
        guards=_typed_guards(("lhs", "rhs", "result"), _INDEX),
        emit=(
            EmitDescriptorOp(
                descriptor=compare,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("condition")},
                result_types={"dst": _I1},
            ),
            EmitDescriptorOp(
                descriptor=select,
                operands={
                    "true_value": ValueRef.operand("lhs"),
                    "false_value": ValueRef.operand("rhs"),
                    "condition": ValueRef.temporary("condition"),
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


def _narrow_integer_to_float_rule(
    source_type: TypePattern,
    source_bit_count: int,
    result_type: TypePattern,
    result_type_name: str,
    signedness: str,
) -> DescriptorRule:
    convert = _descriptor(f"wasm.{result_type_name}.convert_i32_{signedness}")
    constant = _descriptor("wasm.i32.const")
    if signedness == "s":
        shift = _descriptor("wasm.i32.shl")
        normalize = _descriptor("wasm.i32.shr_s")
        shift_amount = 32 - source_bit_count
        normalization = (
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("shift_amount")},
                result_types={"dst": _I32},
                immediates={"i32_value": shift_amount},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "lhs": ValueRef.operand("input"),
                    "rhs": ValueRef.temporary("shift_amount"),
                },
                results={"dst": ValueRef.temporary("shifted")},
                result_types={"dst": _I32},
            ),
            EmitDescriptorOp(
                descriptor=normalize,
                operands={
                    "lhs": ValueRef.temporary("shifted"),
                    "rhs": ValueRef.temporary("shift_amount"),
                },
                results={"dst": ValueRef.temporary("normalized")},
                result_types={"dst": _I32},
            ),
        )
    else:
        normalize = _descriptor("wasm.i32.and")
        mask = (1 << source_bit_count) - 1
        normalization = (
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("mask")},
                result_types={"dst": _I32},
                immediates={"i32_value": mask},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=normalize,
                operands={
                    "lhs": ValueRef.operand("input"),
                    "rhs": ValueRef.temporary("mask"),
                },
                results={"dst": ValueRef.temporary("normalized")},
                result_types={"dst": _I32},
            ),
        )
    return DescriptorRule(
        source_op=(
            scalar_conversion.scalar_sitofp
            if signedness == "s"
            else scalar_conversion.scalar_uitofp
        ),
        descriptor=convert,
        guards=(
            _value_type("input", source_type),
            _value_type("result", result_type),
        ),
        emit=(
            *normalization,
            EmitDescriptorOp(
                descriptor=convert,
                operands={"input": ValueRef.temporary("normalized")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _bf16_to_f32_rule() -> DescriptorRule:
    constant = _descriptor("wasm.i32.const")
    shift = _descriptor("wasm.i32.shl")
    reinterpret = _descriptor("wasm.f32.reinterpret_i32")
    # The low 16 carrier bits become the high half of the FP32 encoding.
    return DescriptorRule(
        source_op=scalar_conversion.scalar_extf,
        descriptor=reinterpret,
        guards=(
            _value_type("input", _BF16),
            _value_type("result", _F32),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("shift")},
                result_types={"dst": _I32},
                immediates={"i32_value": 16},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "lhs": ValueRef.operand("input"),
                    "rhs": ValueRef.temporary("shift"),
                },
                results={"dst": ValueRef.temporary("bits")},
                result_types={"dst": _I32},
            ),
            EmitDescriptorOp(
                descriptor=reinterpret,
                operands={"input": ValueRef.temporary("bits")},
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
                form=DescriptorEmitForm.OP,
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


def _scalar_compare_rule(
    source_op: Op,
    predicate: str,
    operand_type: TypePattern,
    descriptor_key: str,
    *,
    guards: tuple[Guard, ...] = (),
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
            *guards,
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
                "static_indices",
                0,
                0,
                _maximum_vector_lane_count(source_type) - 1,
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
            Guard.i64_array_element_range(
                "static_indices",
                0,
                0,
                _maximum_vector_lane_count(dest_type) - 1,
            ),
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
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _dynamic_insert_rule(
    value_type: TypePattern,
    dest_type: TypePattern,
    splat_descriptor_key: str,
    *,
    physical_lane_count: int | None = None,
) -> DescriptorRule:
    # Replicate each lane ordinal across its physical bytes. Equality with the
    # broadcast index selects every bit of that lane without scalar expansion.
    if physical_lane_count is None:
        physical_lane_count = _maximum_vector_lane_count(dest_type)
    bytes_per_lane = 16 // physical_lane_count
    ordinals = bytes(byte // bytes_per_lane for byte in range(16))
    descriptor = _descriptor("wasm.v128.bitselect")
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            _value_type("value", value_type),
            _value_type("dest", dest_type),
            _value_type("result", dest_type),
            Guard.operand_segment_count("indices", 1),
            Guard.i64_array_count("static_indices", 1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=_descriptor("wasm.v128.const"),
                results={"dst": ValueRef.temporary("lane_ordinals")},
                result_types={"dst": _V16I8},
                immediates={
                    "lo64": int.from_bytes(ordinals[:8], "little"),
                    "hi64": int.from_bytes(ordinals[8:], "little"),
                },
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=_descriptor("wasm.i8x16.splat"),
                operands={"value": ValueRef.operand("indices")},
                results={"dst": ValueRef.temporary("lane_index")},
                result_types={"dst": _V16I8},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=_descriptor("wasm.i8x16.eq"),
                operands={
                    "lhs": ValueRef.temporary("lane_ordinals"),
                    "rhs": ValueRef.temporary("lane_index"),
                },
                results={"dst": ValueRef.temporary("lane_mask")},
                result_types={"dst": _V16I8},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=_descriptor(splat_descriptor_key),
                operands={"value": ValueRef.operand("value")},
                results={"dst": ValueRef.temporary("replacement")},
                result_types={"dst": _V16I8},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "true_value": ValueRef.temporary("replacement"),
                    "false_value": ValueRef.operand("dest"),
                    "condition": ValueRef.temporary("lane_mask"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _maximum_vector_lane_count(type_pattern: TypePattern) -> int:
    if type_pattern.lanes is not None:
        return type_pattern.lanes
    if isinstance(type_pattern.maximum_lanes, int):
        return type_pattern.maximum_lanes
    raise ValueError(f"vector type has no fixed maximum lane count: {type_pattern!r}")


_SHUFFLE_BYTE_NAMES = tuple(f"lane{i}" for i in range(16))


def _shuffle_rule(type_pattern: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("wasm.i8x16.shuffle")
    return DescriptorRule(
        source_op=vector.vector_shuffle,
        descriptor=descriptor,
        guards=(
            _value_type("source", type_pattern),
            _value_type("result", type_pattern),
            Guard.i64_array_count("source_lanes", type_pattern.lanes),
            Guard.i64_array_elements_range("source_lanes", 0, type_pattern.lanes - 1),
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
                        source_lane_count=type_pattern.lanes,
                        bytes_per_lane=16 // type_pattern.lanes,
                        target_names=_SHUFFLE_BYTE_NAMES,
                    ),
                ),
            ),
        ),
    )


def _byte_offset_materializer() -> SourceMemoryByteOffsetMaterializer:
    return SourceMemoryByteOffsetMaterializer(
        constant=_descriptor("wasm.i32.const"),
        add=_descriptor("wasm.i32.add"),
        multiply=_descriptor("wasm.i32.mul"),
        shift_left=None,
        constant_immediate="i32_value",
        integer_conversions=(
            SourceMemoryIntegerConversion("i64", _descriptor("wasm.i32.wrap_i64")),
        ),
    )


def _view_carrier_rules() -> Iterable[DescriptorRule]:
    # Views crossing control or callable boundaries carry complete addresses;
    # direct memory accesses continue to consume their planned storage roots.
    descriptor = _descriptor("wasm.i32.add")
    for source_op in (buffer.buffer_view, view.view_subview):
        for element_byte_count in (1, 2, 4, 8):
            for dynamic in (False, True):
                source_memory = SourceMemoryConstraint(
                    operation=SourceMemoryOperation.VIEW_CARRIER,
                    root_kind=SourceMemoryRootKind.ANY,
                    memory_spaces=("unknown", "generic", "global"),
                    element_byte_count=element_byte_count,
                    vector_lane_count=1,
                    vector_lane_byte_stride=element_byte_count,
                    static_byte_offset_minimum=0,
                    static_byte_offset_maximum=(1 << 32) - 1,
                    dynamic_term_count=None if dynamic else 0,
                    dynamic_term_count_minimum=1 if dynamic else 0,
                    dynamic_view_base_term_count=None,
                    allow_dynamic_stride_values=dynamic,
                    byte_offset_unsigned_bit_count=32,
                    # The complete root-relative byte offset must fit Wasm's
                    # u32 address space, but an affine dynamic contribution
                    # may be negative before its static bias is added.
                    dynamic_offset_unsigned_bit_count=0,
                    byte_offset_diagnostic=_WASM32_ADDRESS_DIAGNOSTIC,
                    diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
                )
                static_offset = ValueRef.temporary("static_byte_offset")
                byte_offset = static_offset
                emits = [
                    EmitDescriptorOp(
                        descriptor=_descriptor("wasm.i32.const"),
                        results={"dst": static_offset},
                        result_types={"dst": _I32},
                        immediates={
                            "i32_value": SourceMemoryProject.static_byte_offset()
                        },
                        source_memory=source_memory,
                        form=DescriptorEmitForm.CONST,
                    )
                ]
                if dynamic:
                    byte_offset = ValueRef.temporary("byte_offset")
                    emits.append(
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={
                                "lhs": static_offset,
                                "rhs": ValueRef.source_memory_dynamic_byte_offset(),
                            },
                            results={"dst": byte_offset},
                            result_types={"dst": _I32},
                            source_memory=source_memory,
                            source_memory_byte_offset_materializer=(
                                _byte_offset_materializer()
                            ),
                        )
                    )
                emits.append(
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={
                            "lhs": ValueRef.source_memory_root(),
                            "rhs": byte_offset,
                        },
                        results={"dst": ValueRef.result("result")},
                        source_memory=source_memory,
                        form=DescriptorEmitForm.OP,
                    )
                )
                yield DescriptorRule(
                    source_op=source_op,
                    descriptor=descriptor,
                    emit=tuple(emits),
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


class _MemoryAddressForm(Enum):
    DYNAMIC_IMMEDIATE = "dynamic_immediate"
    DYNAMIC_REGISTER = "dynamic_register"
    STATIC = "static"


def _memory_rule(
    source_op: Op,
    operation: SourceMemoryOperation,
    value_type: TypePattern,
    descriptor_key: str,
    *,
    element_byte_count: int,
    lane_count: int,
    address_form: _MemoryAddressForm,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    dynamic = address_form is not _MemoryAddressForm.STATIC
    register_bias = address_form is _MemoryAddressForm.DYNAMIC_REGISTER
    source_memory = SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.ANY,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=element_byte_count,
        vector_lane_count=lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=1 if register_bias else 0,
        static_byte_offset_maximum=(1 << 32) - 1,
        dynamic_term_count=None if dynamic else 0,
        dynamic_term_count_minimum=1 if dynamic else 0,
        dynamic_view_base_term_count=None,
        allow_dynamic_stride_values=dynamic,
        byte_offset_unsigned_bit_count=32,
        dynamic_offset_unsigned_bit_count=(
            32 if address_form is _MemoryAddressForm.DYNAMIC_IMMEDIATE else 0
        ),
        byte_offset_diagnostic=_WASM32_ADDRESS_DIAGNOSTIC,
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )
    address = ValueRef.source_memory_root()
    emits = []
    if dynamic:
        materializer = _byte_offset_materializer()
        byte_offset = ValueRef.source_memory_dynamic_byte_offset()
        if register_bias:
            # Wasm adds its memory immediate without i32 wrap. Keep the bias
            # in modular arithmetic when the dynamic part can be negative.
            bias = ValueRef.temporary("bias")
            biased_offset = ValueRef.temporary("biased_offset")
            emits.extend(
                (
                    EmitDescriptorOp(
                        descriptor=_descriptor("wasm.i32.const"),
                        results={"dst": bias},
                        result_types={"dst": _I32},
                        immediates={
                            "i32_value": SourceMemoryProject.static_byte_offset()
                        },
                        source_memory=source_memory,
                        form=DescriptorEmitForm.CONST,
                    ),
                    EmitDescriptorOp(
                        descriptor=_descriptor("wasm.i32.add"),
                        operands={"lhs": byte_offset, "rhs": bias},
                        results={"dst": biased_offset},
                        result_types={"dst": _I32},
                        source_memory=source_memory,
                        source_memory_byte_offset_materializer=materializer,
                    ),
                )
            )
            byte_offset = biased_offset
        address = ValueRef.temporary("address")
        emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("wasm.i32.add"),
                operands={
                    "lhs": ValueRef.source_memory_root(),
                    "rhs": byte_offset,
                },
                results={"dst": address},
                result_types={"dst": _I32},
                source_memory=source_memory,
                source_memory_byte_offset_materializer=(
                    None if register_bias else materializer
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
            immediates={
                "offset": 0
                if register_bias
                else SourceMemoryProject.static_byte_offset()
            },
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
        *uniform_shift_rules(),
        *_view_carrier_rules(),
        ValueAliasRule(
            source_op=view.view_refine,
            source=ValueRef.operand("source"),
            result=ValueRef.result("result"),
        ),
        _buffer_load_i8_u_rule(),
        _buffer_store_i8_rule(),
        *_integer_sign_rules(_I32, "i32", 31),
        *_integer_sign_rules(_I64, "i64", 63),
        *(
            _binary_rule(source_op, value_type, f"wasm.{type_name}.{operation}")
            for value_type, type_name in ((_I32, "i32"), (_I64, "i64"))
            for source_op, operation in (
                (scalar_arithmetic.scalar_addi, "add"),
                (scalar_arithmetic.scalar_subi, "sub"),
                (scalar_arithmetic.scalar_muli, "mul"),
                (scalar_arithmetic.scalar_divsi, "div_s"),
                (scalar_arithmetic.scalar_divui, "div_u"),
                (scalar_arithmetic.scalar_remsi, "rem_s"),
                (scalar_arithmetic.scalar_remui, "rem_u"),
                (scalar_bitwise.scalar_andi, "and"),
                (scalar_bitwise.scalar_ori, "or"),
                (scalar_bitwise.scalar_xori, "xor"),
                (scalar_bitwise.scalar_shli, "shl"),
                (scalar_bitwise.scalar_shrsi, "shr_s"),
                (scalar_bitwise.scalar_shrui, "shr_u"),
                (scalar_bitwise.scalar_rotli, "rotl"),
                (scalar_bitwise.scalar_rotri, "rotr"),
            )
        ),
        *(
            _unary_rule(source_op, value_type, f"wasm.{type_name}.{operation}")
            for value_type, type_name in ((_I32, "i32"), (_I64, "i64"))
            for source_op, operation in (
                (scalar_bitwise.scalar_ctlzi, "clz"),
                (scalar_bitwise.scalar_cttzi, "ctz"),
                (scalar_bitwise.scalar_ctpopi, "popcnt"),
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
        *(
            _binary_rule(source_op, _F32, f"wasm.f32.{operation}")
            for source_op, operation in (
                (scalar_arithmetic.scalar_addf, "add"),
                (scalar_arithmetic.scalar_subf, "sub"),
                (scalar_arithmetic.scalar_mulf, "mul"),
                (scalar_arithmetic.scalar_divf, "div"),
                (scalar_arithmetic.scalar_minimumf, "min"),
                (scalar_arithmetic.scalar_maximumf, "max"),
                (scalar_arithmetic.scalar_copysignf, "copysign"),
            )
        ),
        *(
            _unary_rule(source_op, _F32, f"wasm.f32.{operation}")
            for source_op, operation in (
                (scalar_arithmetic.scalar_absf, "abs"),
                (scalar_arithmetic.scalar_negf, "neg"),
                (scalar_math.scalar_ceilf, "ceil"),
                (scalar_math.scalar_floorf, "floor"),
                (scalar_math.scalar_truncf, "trunc"),
                (scalar_math.scalar_roundevenf, "nearest"),
                (scalar_math.scalar_sqrtf, "sqrt"),
            )
        ),
        *(
            _scalar_compare_rule(
                scalar_comparison.scalar_cmpf,
                predicate,
                _F32,
                f"wasm.f32.{operation}",
            )
            for predicate, operation in (
                ("oeq", "eq"),
                ("ogt", "gt"),
                ("oge", "ge"),
                ("olt", "lt"),
                ("ole", "le"),
                ("une", "ne"),
            )
        ),
        *(
            _scalar_compare_rule(
                scalar_comparison.scalar_cmpf,
                predicate,
                _F32,
                f"wasm.f32.{operation}",
                guards=(Guard.instance_flags_has_all("fastmath", "nnan"),),
            )
            for predicate, operation in (
                ("one", "ne"),
                ("ueq", "eq"),
                ("ugt", "gt"),
                ("uge", "ge"),
                ("ult", "lt"),
                ("ule", "le"),
            )
        ),
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
        *(
            _conversion_rule(
                source_op,
                source_type,
                result_type,
                f"wasm.{result_type_name}.convert_{source_type_name}_{signedness}",
            )
            for source_op, signedness in (
                (scalar_conversion.scalar_sitofp, "s"),
                (scalar_conversion.scalar_uitofp, "u"),
            )
            for source_type, source_type_name in ((_I32, "i32"), (_I64, "i64"))
            for result_type, result_type_name in ((_F32, "f32"), (_F64, "f64"))
        ),
        *(
            _narrow_integer_to_float_rule(
                source_type,
                source_bit_count,
                result_type,
                result_type_name,
                signedness,
            )
            for source_type, source_bit_count in ((_I1, 1), (_I8, 8), (_I16, 16))
            for result_type, result_type_name in ((_F32, "f32"), (_F64, "f64"))
            for signedness in ("s", "u")
        ),
        _conversion_rule(
            scalar_conversion.scalar_extf,
            _F32,
            _F64,
            "wasm.f64.promote_f32",
        ),
        _conversion_rule(
            scalar_conversion.scalar_fptrunc,
            _F64,
            _F32,
            "wasm.f32.demote_f64",
        ),
        _bf16_to_f32_rule(),
        *float_narrowing_rules(_descriptor, _value_type),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F8E4M3, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I8, _F8E4M3),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F8E5M2, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I8, _F8E5M2),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _F16, _I16),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I16, _F16),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _BF16, _I16),
        _conversion_alias_rule(scalar_conversion.scalar_bitcast, _I16, _BF16),
        *(
            _conversion_alias_rule(vector.vector_bitcast, source_type, result_type)
            for source_type in _NUMERIC_V128_TYPES
            for result_type in _NUMERIC_V128_TYPES
        ),
        *(
            _conversion_alias_rule(vector.vector_bitcast, source_type, result_type)
            for _, source_type, _, _, _ in _PARTIAL_INTEGER_LANE_TYPES
            for _, result_type, _, _, _ in _PARTIAL_INTEGER_LANE_TYPES
        ),
        # Wasm has native aliases for the bitcast carrier shapes above. Every
        # other cast, and every unmatched bitcast shape, deliberately enters
        # the shared lane-wise reference legalization family.
        *(UnsupportedRule(source_op=source_op) for source_op in vector.VECTOR_CAST_OPS),
        _conversion_alias_rule(scalar_conversion.scalar_trunci, _I32, _I8),
        _conversion_alias_rule(scalar_conversion.scalar_trunci, _I32, _I16),
        _conversion_alias_rule(scalar_conversion.scalar_extui, _I1, _I32),
        _masked_extui_rule(_I8, 0xFF),
        _masked_extui_rule(_I16, 0xFFFF),
        _conversion_rule(
            scalar_conversion.scalar_extsi,
            _I32,
            _I64,
            "wasm.i64.extend_i32_s",
        ),
        _conversion_rule(
            scalar_conversion.scalar_extui,
            _I32,
            _I64,
            "wasm.i64.extend_i32_u",
        ),
        _const_i1_rule(),
        _const_i32_rule(scalar_conversion.scalar_constant, _I8),
        _const_i32_rule(scalar_conversion.scalar_constant, _I16),
        _const_i32_rule(scalar_conversion.scalar_constant, _I32),
        _const_i64_rule(scalar_conversion.scalar_constant, _I64),
        _const_float_rule(_F32, "wasm.f32.const"),
        _const_float_rule(_F64, "wasm.f64.const"),
        *predicate_rules(_descriptor, _value_type),
        _const_v128_rule(
            _V2I64, ValueProject.exact_i64("result"), Guard.value_exact_i64("result")
        ),
        _const_v128_rule(
            _PARTIAL_I64,
            ValueProject.exact_i64("result"),
            Guard.value_exact_i64("result"),
        ),
        _const_v128_rule(
            _V2F64, ValueProject.float_bits("result"), Guard.value_exact_float("result")
        ),
        *(
            _const_vector_splat_rule(
                result_type,
                "i32",
                shape_name,
                "i32_value",
                ValueProject.exact_i64("result"),
                Guard.value_exact_i64("result"),
            )
            for result_type, shape_name in (
                (Vector("i8", lanes=16), "i8x16"),
                (Vector("i16", lanes=8), "i16x8"),
                (_V4I32, "i32x4"),
            )
        ),
        *(
            _const_vector_splat_rule(
                vector_type,
                "i32",
                shape_name,
                "i32_value",
                ValueProject.exact_i64("result"),
                Guard.value_exact_i64("result"),
            )
            for _, vector_type, shape_name, _, _ in _PARTIAL_INTEGER_LANE_TYPES[:-1]
        ),
        *(
            _const_vector_splat_rule(
                result_type,
                element_type,
                shape_name,
                immediate,
                ValueProject.float_bits("result"),
                Guard.value_exact_float("result"),
            )
            for result_type, element_type, shape_name, immediate in (
                (Vector(("f8E4M3", "f8E5M2"), lanes=16), "i32", "i8x16", "i32_value"),
                (Vector(("f16", "bf16"), lanes=8), "i32", "i16x8", "i32_value"),
                (_V4F32, "f32", "f32x4", "bits"),
            )
        ),
        *(
            _whole_value_select_rule(value_type, f"wasm.{type_name}.select")
            for value_type, type_name in (
                (_I1, "i32"),
                (_BYTE_STORAGE, "i32"),
                (_WORD_STORAGE, "i32"),
                (_I32, "i32"),
                (_I64, "i64"),
                (_F32, "f32"),
                (_F64, "f64"),
                (_INDEX, "i32"),
                (_OFFSET, "i32"),
                *((value_type, "v128") for value_type in _NUMERIC_V128_TYPES),
                (Buffer(), "i32"),
            )
        ),
        *(
            _splat_rule(scalar_type, vector_type, f"wasm.{shape_name}.splat")
            for scalar_type, vector_type, shape_name, _ in _V128_LANE_TYPES
        ),
        *(
            _splat_rule(scalar_type, vector_type, f"wasm.{shape_name}.splat")
            for (
                scalar_type,
                vector_type,
                shape_name,
                _,
                _,
            ) in _PARTIAL_INTEGER_LANE_TYPES
        ),
        *varying_shift_rules(_descriptor, _value_type),
        *(
            _binary_rule(source_op, value_type, f"wasm.v128.{operation}")
            for source_op, operation in (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
            )
            for value_type in (
                Vector("i8", lanes=16),
                Vector("i16", lanes=8),
                _V4I32,
                _V2I64,
            )
        ),
        *(
            _binary_rule(source_op, vector_type, f"wasm.v128.{operation}")
            for source_op, operation in (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
            )
            for _, vector_type, _, _, _ in _PARTIAL_INTEGER_LANE_TYPES
        ),
        *(
            _binary_rule(source_op, _V4F32, f"wasm.f32x4.{operation}")
            for source_op, operation in (
                (vector.vector_addf, "add"),
                (vector.vector_subf, "sub"),
                (vector.vector_mulf, "mul"),
                (vector.vector_divf, "div"),
                (vector.vector_minimumf, "min"),
                (vector.vector_maximumf, "max"),
            )
        ),
        *(
            _unary_rule(source_op, _V4F32, f"wasm.f32x4.{operation}")
            for source_op, operation in (
                (vector.vector_absf, "abs"),
                (vector.vector_negf, "neg"),
                (vector.vector_ceilf, "ceil"),
                (vector.vector_floorf, "floor"),
                (vector.vector_truncf, "trunc"),
                (vector.vector_roundevenf, "nearest"),
                (vector.vector_sqrtf, "sqrt"),
            )
        ),
        *integer_arithmetic_rules(_descriptor, _value_type),
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
        _index_extrema_rule(index.index_min, "wasm.i32.lt_s"),
        _index_extrema_rule(index.index_max, "wasm.i32.gt_s"),
        _index_madd_rule(),
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
                (index.index_rotli, "rotl"),
                (index.index_rotri, "rotr"),
            )
        ),
        *(
            _unary_rule(source_op, _INDEX, f"wasm.i32.{operation}")
            for source_op, operation in (
                (index.index_ctlzi, "clz"),
                (index.index_cttzi, "ctz"),
                (index.index_ctpopi, "popcnt"),
            )
        ),
        *(
            _binary_rule(
                source_op,
                _INDEX,
                descriptor_key,
                guards=tuple(
                    Guard.value_i64_range(field, 0, (1 << 32) - 1)
                    for field in ("lhs", "rhs")
                ),
            )
            for source_op, descriptor_key in (
                (index.index_div, "wasm.i32.div_u"),
                (index.index_rem, "wasm.i32.rem_u"),
            )
        ),
        *(
            _extract_rule(
                vector_type, scalar_type, f"wasm.{shape_name}.extract_lane{suffix}"
            )
            for scalar_type, vector_type, shape_name, suffix in _V128_LANE_TYPES
        ),
        *(
            _extract_rule(
                vector_type, scalar_type, f"wasm.{shape_name}.extract_lane{suffix}"
            )
            for (
                scalar_type,
                vector_type,
                shape_name,
                suffix,
                _,
            ) in _PARTIAL_INTEGER_LANE_TYPES
        ),
        *(
            _insert_rule(scalar_type, vector_type, f"wasm.{shape_name}.replace_lane")
            for scalar_type, vector_type, shape_name, _ in _V128_LANE_TYPES
        ),
        *(
            _insert_rule(scalar_type, vector_type, f"wasm.{shape_name}.replace_lane")
            for (
                scalar_type,
                vector_type,
                shape_name,
                _,
                _,
            ) in _PARTIAL_INTEGER_LANE_TYPES
        ),
        *(
            _dynamic_insert_rule(scalar_type, vector_type, f"wasm.{shape_name}.splat")
            for scalar_type, vector_type, shape_name, _ in _V128_LANE_TYPES
        ),
        *(
            _dynamic_insert_rule(
                scalar_type,
                vector_type,
                f"wasm.{shape_name}.splat",
                physical_lane_count=physical_lane_count,
            )
            for (
                scalar_type,
                vector_type,
                shape_name,
                _,
                physical_lane_count,
            ) in _PARTIAL_INTEGER_LANE_TYPES
        ),
        *(_shuffle_rule(value_type) for value_type in _NUMERIC_V128_TYPES),
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
                address_form=address_form,
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
                    _WORD_STORAGE,
                    "i32.load16_u",
                    "i32.store16",
                    2,
                    1,
                    view.view_load,
                    view.view_store,
                ),
                (
                    _V16_BYTE_STORAGE,
                    "v128.load",
                    "v128.store",
                    1,
                    16,
                    vector.vector_load,
                    vector.vector_store,
                ),
                (
                    _V8_WORD_STORAGE,
                    "v128.load",
                    "v128.store",
                    2,
                    8,
                    vector.vector_load,
                    vector.vector_store,
                ),
                (
                    Vector(("i32", "f32"), lanes=4),
                    "v128.load",
                    "v128.store",
                    4,
                    4,
                    vector.vector_load,
                    vector.vector_store,
                ),
                (
                    Vector(("i64", "f64"), lanes=2),
                    "v128.load",
                    "v128.store",
                    8,
                    2,
                    vector.vector_load,
                    vector.vector_store,
                ),
            )
            for operation in (SourceMemoryOperation.LOAD, SourceMemoryOperation.STORE)
            for address_form in _MemoryAddressForm
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
