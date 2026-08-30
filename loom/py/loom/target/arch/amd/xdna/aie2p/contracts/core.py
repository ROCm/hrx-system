# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P core source-to-Low contract fragment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

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
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ResultTypeBinding,
    Scalar,
    SourceMemoryAddressLayout,
    SourceMemoryConstraint,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
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
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_I8_VECTOR = Vector("i8", minimum_lanes=1, maximum_lanes=64)
_I16_VECTOR = Vector("i16", minimum_lanes=1, maximum_lanes=32)
_I32_VECTOR = Vector("i32", minimum_lanes=1, maximum_lanes=16)
_I8X64 = Vector("i8", lanes=64)
_I16X32 = Vector("i16", lanes=32)
_I32X16 = Vector("i32", lanes=16)
_I1_VECTOR = Vector("i1", minimum_lanes=1, maximum_lanes=64)
_INTEGER_VECTOR_TYPES = (_I8_VECTOR, _I16_VECTOR, _I32_VECTOR)

_I8_MIN = -(2**7)
_I8_MAX = (2**7) - 1
_I16_MIN = -(2**15)
_I16_MAX = (2**15) - 1
_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1
_SHORT_MIN = -1024
_SHORT_MAX = 1023


def _plain_integer_multiply_control(
    *,
    lhs_signed: bool,
    rhs_signed: bool,
    lhs_mode: int,
    rhs_mode: int,
    variant: int,
) -> int:
    """Encodes the AIE2P control word for a non-accumulating integer multiply."""

    if lhs_mode < 0 or lhs_mode > 0b11:
        raise ValueError("AIE2P multiply lhs mode must fit two bits")
    if rhs_mode < 0 or rhs_mode > 0b11:
        raise ValueError("AIE2P multiply rhs mode must fit two bits")
    if variant < 0 or variant > 0b111:
        raise ValueError("AIE2P multiply variant must fit three bits")
    return (
        int(lhs_signed) << 9
        | int(rhs_signed) << 8
        | lhs_mode << 1
        | rhs_mode << 3
        | variant << 5
    )


_I16_ELEMENTWISE_MULTIPLY_CONTROL = _plain_integer_multiply_control(
    lhs_signed=True,
    rhs_signed=True,
    lhs_mode=1,
    rhs_mode=3,
    variant=2,
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _typed_guards(
    fields: Iterable[str], type_pattern: TypePattern
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _op_emit(
    descriptor: Descriptor,
    *,
    operands: Mapping[str, ValueRef] | None = None,
    results: Mapping[str, ValueRef] | None = None,
    result_types: Mapping[str, ResultTypeBinding] | None = None,
    copy_operands: Sequence[str] = (),
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        copy_operands=copy_operands,
        form=DescriptorEmitForm.OP,
    )


def _const_emit(
    descriptor: Descriptor,
    result: ValueRef,
    value: AttrProject | int,
    *,
    result_type: ResultTypeBinding | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        results={"dst": result},
        result_types=None if result_type is None else {"dst": result_type},
        immediates={"i": value},
        form=DescriptorEmitForm.CONST,
    )


def _constant_rule(
    result_type: TypePattern,
    descriptor_key: str,
    minimum: int,
    maximum: int,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", minimum, maximum),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _logical_constant_rule() -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.constant.i32.short")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", _I1),
            Guard.value_exact_i64("result"),
            Guard.value_i64_range("result", 0, 1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i": ValueProject.exact_i64("result")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _address_constant_rule(
    result_type: TypePattern,
    descriptor_key: str,
    minimum: int,
    maximum: int,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=index.index_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", minimum, maximum),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i": AttrProject.direct("value")},
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
            _op_emit(
                descriptor,
                operands={
                    "s0": ValueRef.operand("lhs"),
                    "s1": ValueRef.operand("rhs"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _full_vector_memory_constraint(
    operation: SourceMemoryOperation,
    *,
    element_byte_count: int,
    vector_lane_count: int,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.BLOCK_ARGUMENT,
        address_layout=SourceMemoryAddressLayout.COMPACT_ROW_MAJOR,
        memory_spaces=("unknown", "generic", "workgroup"),
        element_byte_count=element_byte_count,
        vector_lane_count=vector_lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=-512,
        static_byte_offset_maximum=448,
        minimum_alignment=64,
        dynamic_term_count=0,
    )


def _full_vector_load_rule(
    result_type: TypePattern,
    *,
    element_byte_count: int,
    vector_lane_count: int,
) -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.load.a.i8x64.indexed.immediate")
    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 0),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"ptr": ValueRef.operand("view")},
                results={"dst": ValueRef.result("result")},
                immediates={"imm": SourceMemoryProject.static_byte_offset()},
                source_memory=_full_vector_memory_constraint(
                    SourceMemoryOperation.LOAD,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                ),
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _full_vector_store_rule(
    value_type: TypePattern,
    *,
    element_byte_count: int,
    vector_lane_count: int,
) -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.store.i8x64.indexed.immediate")
    return DescriptorRule(
        source_op=vector.vector_store,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 0),
            Guard.value_type("value", value_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "src": ValueRef.operand("value"),
                    "ptr": ValueRef.operand("view"),
                },
                immediates={"imm": SourceMemoryProject.static_byte_offset()},
                source_memory=_full_vector_memory_constraint(
                    SourceMemoryOperation.STORE,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                ),
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _conversion_rule(
    source_op: Op,
    input_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", input_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={"s0": ValueRef.operand("input")},
                results={"d0": ValueRef.result("result")},
            ),
        ),
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


def _vector_binary_rule(
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
            _op_emit(
                descriptor,
                operands={
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                },
                results={"d": ValueRef.result("result")},
            ),
        ),
    )


def _vector_multiply_i16_rule() -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shift_constant = _descriptor("amd.xdna.aie2p.constant.i32.shift")
    multiply = _descriptor("amd.xdna.aie2p.multiply.i16x32.configured")
    set_rounding = _descriptor("amd.xdna.aie2p.state.rounding.immediate")
    set_srs_mode = _descriptor("amd.xdna.aie2p.state.srs-mode.immediate")
    set_saturation = _descriptor("amd.xdna.aie2p.state.saturation.immediate")
    narrow = _descriptor("amd.xdna.aie2p.narrow.trunc.signed.i16x32")
    return DescriptorRule(
        source_op=vector.vector_muli,
        descriptor=narrow,
        guards=_typed_guards(("lhs", "rhs", "result"), _I16_VECTOR),
        emit=(
            _const_emit(
                config_constant,
                ValueRef.temporary("multiply_control"),
                _I16_ELEMENTWISE_MULTIPLY_CONTROL,
                result_type=DescriptorResultType(),
            ),
            _const_emit(
                shift_constant,
                ValueRef.temporary("narrow_shift"),
                0,
                result_type=DescriptorResultType(),
            ),
            _op_emit(
                multiply,
                operands={
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                    "acc": ValueRef.temporary("multiply_control"),
                },
                results={"dst": ValueRef.temporary("wide_product")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitDescriptorOp(
                descriptor=set_rounding,
                immediates={"i": 0},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=set_srs_mode,
                immediates={"i": 1},
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=set_saturation,
                immediates={"i": 0},
                form=DescriptorEmitForm.OP,
            ),
            _op_emit(
                narrow,
                operands={
                    "src": ValueRef.temporary("wide_product"),
                    "su": ValueRef.temporary("narrow_shift"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _vector_splat_rule(
    scalar_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", scalar_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={"src": ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _vector_predicate_splat_rule() -> DescriptorRule:
    broadcast = _descriptor("amd.xdna.aie2p.splat.i8x64")
    subtract = _descriptor("amd.xdna.aie2p.sub.i8x64")
    compare = _descriptor("amd.xdna.aie2p.cmp.lt.unsigned.i8x64")
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=compare,
        guards=(
            Guard.value_type("scalar", _I1),
            Guard.value_type("result", _I1_VECTOR),
        ),
        emit=(
            _op_emit(
                broadcast,
                operands={"src": ValueRef.operand("scalar")},
                results={"dst": ValueRef.temporary("broadcast_condition")},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                subtract,
                operands={
                    "s1": ValueRef.temporary("broadcast_condition"),
                    "s2": ValueRef.temporary("broadcast_condition"),
                },
                results={"d": ValueRef.temporary("zero")},
                result_types={"d": DescriptorResultType()},
            ),
            _op_emit(
                compare,
                operands={
                    "s1": ValueRef.temporary("zero"),
                    "s2": ValueRef.temporary("broadcast_condition"),
                },
                results={"cmp": ValueRef.result("result")},
            ),
        ),
    )


def _vector_select_rule(
    value_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", _I1_VECTOR),
            *_typed_guards(("true_value", "false_value", "result"), value_type),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={
                    # AIE2P VSEL chooses s1 for a zero mask bit and s2 for a
                    # one bit. Loom vector.select uses one for true.
                    "s1": ValueRef.operand("false_value"),
                    "s2": ValueRef.operand("true_value"),
                    "sel": ValueRef.operand("condition"),
                },
                results={"d": ValueRef.result("result")},
            ),
        ),
    )


def _predicate_complete_emit(
    source: ValueRef,
    result: ValueRef,
) -> EmitDescriptorOp:
    descriptor = _descriptor("amd.xdna.aie2p.predicate.complete.zero.high32")
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={"storage": source},
        results={"dst": result},
        immediates={"i": 0},
        form=DescriptorEmitForm.OP,
    )


def _predicate_binary_emits(
    operation: str,
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    *,
    temporary_prefix: str,
) -> tuple[EmitDescriptorOp, ...]:
    low = _descriptor(f"amd.xdna.aie2p.predicate.{operation}.low32")
    high = _descriptor(f"amd.xdna.aie2p.predicate.{operation}.high32")
    low_result = ValueRef.temporary(f"{temporary_prefix}_low32")
    return (
        _op_emit(
            low,
            operands={"s0": lhs, "s1": rhs},
            results={"d0": low_result},
            result_types={"d0": DescriptorResultType()},
        ),
        _op_emit(
            high,
            operands={
                "s0": lhs,
                "s1": rhs,
                "storage": low_result,
            },
            results={"d0": result},
        ),
    )


def _vector_predicate_binary_rule(
    source_op: Op,
    operation: str,
) -> DescriptorRule:
    descriptor = _descriptor(f"amd.xdna.aie2p.predicate.{operation}.high32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), _I1_VECTOR),
        emit=_predicate_binary_emits(
            operation,
            ValueRef.operand("lhs"),
            ValueRef.operand("rhs"),
            ValueRef.result("result"),
            temporary_prefix="predicate",
        ),
    )


def _vector_compare_rule(
    predicate: str,
    operand_type: TypePattern,
    width: int,
) -> DescriptorRule:
    relation: str
    signedness: str
    swap_operands = predicate in ("sle", "sgt", "ule", "ugt")
    if predicate in ("slt", "sgt"):
        relation, signedness = "lt", "signed"
    elif predicate in ("sle", "sge"):
        relation, signedness = "ge", "signed"
    elif predicate in ("ult", "ugt"):
        relation, signedness = "lt", "unsigned"
    elif predicate in ("ule", "uge"):
        relation, signedness = "ge", "unsigned"
    elif predicate in ("eq", "ne"):
        relation, signedness = "eq", "unsigned"
    else:
        raise ValueError(f"unsupported AIE2P vector comparison predicate {predicate}")

    lane_count = 512 // width
    suffix = ".el.low32" if width != 8 else ""
    lhs = ValueRef.operand("rhs" if swap_operands else "lhs")
    rhs = ValueRef.operand("lhs" if swap_operands else "rhs")
    result = ValueRef.result("result")
    emits: tuple[EmitDescriptorOp, ...]
    if predicate == "eq":
        subtract = _descriptor(f"amd.xdna.aie2p.sub.i{width}x{lane_count}")
        compare = _descriptor(f"amd.xdna.aie2p.cmp.eqz.i{width}x{lane_count}{suffix}")
        difference = ValueRef.temporary("comparison_difference")
        comparison = result if width == 8 else ValueRef.temporary("comparison_low32")
        emits = (
            _op_emit(
                subtract,
                operands={"s1": lhs, "s2": rhs},
                results={"d": difference},
                result_types={"d": DescriptorResultType()},
            ),
            _op_emit(
                compare,
                operands={"s2": difference},
                results={"cmp": comparison},
                result_types=({"cmp": DescriptorResultType()} if width != 8 else None),
            ),
            *((_predicate_complete_emit(comparison, result),) if width != 8 else ()),
        )
        descriptor = (
            compare
            if width == 8
            else _descriptor("amd.xdna.aie2p.predicate.complete.zero.high32")
        )
    elif predicate == "ne":
        compare = _descriptor(
            f"amd.xdna.aie2p.cmp.lt.unsigned.i{width}x{lane_count}{suffix}"
        )
        forward = ValueRef.temporary("comparison_forward")
        reverse = ValueRef.temporary("comparison_reverse")
        low = _descriptor("amd.xdna.aie2p.predicate.or.low32")
        emits = (
            _op_emit(
                compare,
                operands={"s1": ValueRef.operand("lhs"), "s2": ValueRef.operand("rhs")},
                results={"cmp": forward},
                result_types={"cmp": DescriptorResultType()},
            ),
            _op_emit(
                compare,
                operands={"s1": ValueRef.operand("rhs"), "s2": ValueRef.operand("lhs")},
                results={"cmp": reverse},
                result_types={"cmp": DescriptorResultType()},
            ),
        )
        if width == 8:
            emits = (
                *emits,
                *_predicate_binary_emits(
                    "or",
                    forward,
                    reverse,
                    result,
                    temporary_prefix="comparison",
                ),
            )
            descriptor = _descriptor("amd.xdna.aie2p.predicate.or.high32")
        else:
            comparison = ValueRef.temporary("comparison_low32")
            emits = (
                *emits,
                _op_emit(
                    low,
                    operands={"s0": forward, "s1": reverse},
                    results={"d0": comparison},
                    result_types={"d0": DescriptorResultType()},
                ),
                _predicate_complete_emit(comparison, result),
            )
            descriptor = _descriptor("amd.xdna.aie2p.predicate.complete.zero.high32")
    else:
        compare = _descriptor(
            f"amd.xdna.aie2p.cmp.{relation}.{signedness}.i{width}x{lane_count}{suffix}"
        )
        comparison = result if width == 8 else ValueRef.temporary("comparison_low32")
        emits = (
            _op_emit(
                compare,
                operands={"s1": lhs, "s2": rhs},
                results={"cmp": comparison},
                result_types=({"cmp": DescriptorResultType()} if width != 8 else None),
            ),
            *((_predicate_complete_emit(comparison, result),) if width != 8 else ()),
        )
        descriptor = (
            compare
            if width == 8
            else _descriptor("amd.xdna.aie2p.predicate.complete.zero.high32")
        )

    return DescriptorRule(
        source_op=vector.vector_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            *_typed_guards(("lhs", "rhs"), operand_type),
            Guard.value_type("result", _I1_VECTOR),
        ),
        emit=emits,
    )


def _whole_integer_vector_select_rule(
    result_type: TypePattern,
) -> DescriptorRule:
    subtract_one = _descriptor("amd.xdna.aie2p.select.mask.i32")
    select = _descriptor("amd.xdna.aie2p.select.i32x16")
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=select,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=subtract_one,
                operands={"s0": ValueRef.operand("condition")},
                results={"d0": ValueRef.temporary("hardware_selector")},
                result_types={"d0": DescriptorResultType()},
                immediates={"imm": -1},
                form=DescriptorEmitForm.OP,
            ),
            _op_emit(
                select,
                operands={
                    "s1": ValueRef.operand("true_value"),
                    "s2": ValueRef.operand("false_value"),
                    "sel": ValueRef.temporary("hardware_selector"),
                },
                results={"d": ValueRef.result("result")},
            ),
        ),
    )


def _vector_constant_rule(
    result_type: TypePattern,
    constant_descriptor_key: str,
    broadcast_descriptor_key: str,
    minimum: int,
    maximum: int,
) -> DescriptorRule:
    constant = _descriptor(constant_descriptor_key)
    broadcast = _descriptor(broadcast_descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=broadcast,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", minimum, maximum),
        ),
        emit=(
            _const_emit(
                constant,
                ValueRef.temporary("scalar"),
                AttrProject.direct("value"),
                result_type=DescriptorResultType(),
            ),
            _op_emit(
                broadcast,
                operands={"src": ValueRef.temporary("scalar")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _vector_broadcast_alias_rules() -> tuple[ValueAliasRule, ...]:
    return tuple(
        ValueAliasRule(
            source_op=vector.vector_broadcast,
            source=ValueRef.operand("source"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type("source", type_pattern),
                Guard.value_type("result", type_pattern),
                Guard.value_static_element_count_eq("source", "result"),
            ),
        )
        for type_pattern in _INTEGER_VECTOR_TYPES
    )


def _vector_broadcast_rule(
    element_type: str,
    maximum_lanes: int,
    descriptor_key: str,
) -> DescriptorRule:
    source_type = Vector(element_type, lanes=1)
    result_type = Vector(element_type, minimum_lanes=2, maximum_lanes=maximum_lanes)
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_broadcast,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"s1": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={"idx": 0},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _vector_extract_static_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    maximum_index: int,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 0, maximum_index),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"s1": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "idx": AttrProject.i64_array_element("static_indices", element=0)
                },
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _vector_extract_dynamic_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.value_type("indices", _INDEX),
            Guard.operand_segment_count("indices", 1),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, -(2**63), -(2**63)),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={
                    "s1": ValueRef.operand("source"),
                    "idx": ValueRef.operand("indices"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _vector_insert_zero_rule(
    value_type: TypePattern,
    vector_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", value_type),
            Guard.value_type("dest", vector_type),
            Guard.value_type("result", vector_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 0, 0),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={
                    "s1": ValueRef.operand("dest"),
                    "src": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _vector_insert_static_rule(
    value_type: TypePattern,
    vector_type: TypePattern,
    maximum_index: int,
    descriptor_key: str,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", value_type),
            Guard.value_type("dest", vector_type),
            Guard.value_type("result", vector_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 1, maximum_index),
        ),
        emit=(
            _const_emit(
                constant,
                ValueRef.temporary("index"),
                AttrProject.i64_array_element("static_indices", element=0),
                result_type=DescriptorResultType(),
            ),
            _op_emit(
                descriptor,
                operands={
                    "s1": ValueRef.operand("dest"),
                    "idx": ValueRef.temporary("index"),
                    "src": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.result("result")},
                copy_operands=("idx",),
            ),
        ),
    )


def _vector_insert_dynamic_rule(
    value_type: TypePattern,
    vector_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", value_type),
            Guard.value_type("dest", vector_type),
            Guard.value_type("result", vector_type),
            Guard.value_type("indices", _INDEX),
            Guard.operand_segment_count("indices", 1),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, -(2**63), -(2**63)),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={
                    "s1": ValueRef.operand("dest"),
                    "idx": ValueRef.operand("indices"),
                    "src": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.result("result")},
                copy_operands=("idx",),
            ),
        ),
    )


def _vector_xor_rule(
    type_pattern: TypePattern,
    subtract_descriptor_key: str,
) -> DescriptorRule:
    bitwise_or = _descriptor("amd.xdna.aie2p.or.bits512")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.bits512")
    subtract = _descriptor(subtract_descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_xori,
        descriptor=subtract,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                bitwise_or,
                operands={
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                },
                results={"d": ValueRef.temporary("union")},
                result_types={"d": ValueRef.operand("lhs")},
            ),
            _op_emit(
                bitwise_and,
                operands={
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                },
                results={"d": ValueRef.temporary("intersection")},
                result_types={"d": ValueRef.operand("lhs")},
            ),
            _op_emit(
                subtract,
                operands={
                    "s1": ValueRef.temporary("union"),
                    "s2": ValueRef.temporary("intersection"),
                },
                results={"d": ValueRef.result("result")},
            ),
        ),
    )


def _vector_bitcast_alias_rules() -> tuple[ValueAliasRule, ...]:
    return tuple(
        ValueAliasRule(
            source_op=vector.vector_bitcast,
            source=ValueRef.operand("input"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type("input", source_type),
                Guard.value_type("result", result_type),
            ),
        )
        for source_type in _INTEGER_VECTOR_TYPES
        for result_type in _INTEGER_VECTOR_TYPES
    )


def _right_shift_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    zero = _descriptor("amd.xdna.aie2p.constant.i32.short")
    subtract = _descriptor("amd.xdna.aie2p.sub.i32")
    shift = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=shift,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _const_emit(
                zero,
                ValueRef.temporary("zero"),
                0,
                result_type=_I32,
            ),
            _op_emit(
                subtract,
                operands={
                    "s0": ValueRef.temporary("zero"),
                    "s1": ValueRef.operand("rhs"),
                },
                results={"d0": ValueRef.temporary("negative_shift")},
                result_types={"d0": _I32},
            ),
            _op_emit(
                shift,
                operands={
                    "s0": ValueRef.operand("lhs"),
                    "s1": ValueRef.temporary("negative_shift"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _narrow_left_shift_rule(
    type_pattern: TypePattern,
    count_extend_descriptor_key: str,
) -> DescriptorRule:
    count_extend = _descriptor(count_extend_descriptor_key)
    logical_shift = _descriptor("amd.xdna.aie2p.lshl.i32")
    return DescriptorRule(
        source_op=scalar_bitwise.scalar_shli,
        descriptor=logical_shift,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                count_extend,
                operands={"s0": ValueRef.operand("rhs")},
                results={"d0": ValueRef.temporary("shift_count")},
                result_types={"d0": _I32},
            ),
            _op_emit(
                logical_shift,
                operands={
                    "s0": ValueRef.operand("lhs"),
                    "s1": ValueRef.temporary("shift_count"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _narrow_right_shift_rule(
    source_op: Op,
    type_pattern: TypePattern,
    value_extend_descriptor_key: str,
    count_extend_descriptor_key: str,
    shift_descriptor_key: str,
) -> DescriptorRule:
    value_extend = _descriptor(value_extend_descriptor_key)
    count_extend = _descriptor(count_extend_descriptor_key)
    zero = _descriptor("amd.xdna.aie2p.constant.i32.short")
    subtract = _descriptor("amd.xdna.aie2p.sub.i32")
    shift = _descriptor(shift_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=shift,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                value_extend,
                operands={"s0": ValueRef.operand("lhs")},
                results={"d0": ValueRef.temporary("extended_lhs")},
                result_types={"d0": _I32},
            ),
            _op_emit(
                count_extend,
                operands={"s0": ValueRef.operand("rhs")},
                results={"d0": ValueRef.temporary("shift_count")},
                result_types={"d0": _I32},
            ),
            _const_emit(
                zero,
                ValueRef.temporary("zero"),
                0,
                result_type=_I32,
            ),
            _op_emit(
                subtract,
                operands={
                    "s0": ValueRef.temporary("zero"),
                    "s1": ValueRef.temporary("shift_count"),
                },
                results={"d0": ValueRef.temporary("negative_shift")},
                result_types={"d0": _I32},
            ),
            _op_emit(
                shift,
                operands={
                    "s0": ValueRef.temporary("extended_lhs"),
                    "s1": ValueRef.temporary("negative_shift"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _compare_rule(
    predicate: str,
    descriptor_key: str,
    *,
    swap_operands: bool = False,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    operands = {
        "s0": ValueRef.operand("rhs" if swap_operands else "lhs"),
        "s1": ValueRef.operand("lhs" if swap_operands else "rhs"),
    }
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            *_typed_guards(("lhs", "rhs"), _I32),
            Guard.value_type("result", _I1),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands=operands,
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _zero_compare_rule(
    predicate: str,
    descriptor_key: str,
    *,
    zero_field: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    nonzero_field = "rhs" if zero_field == "lhs" else "lhs"
    return DescriptorRule(
        source_op=scalar_comparison.scalar_cmpi,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            *_typed_guards(("lhs", "rhs"), _I32),
            Guard.value_type("result", _I1),
            Guard.value_exact_i64(zero_field),
            Guard.value_i64_range(zero_field, 0, 0),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={"s0": ValueRef.operand(nonzero_field)},
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _bitfield_rule(source_op: Op, descriptor_key: str) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    logical_shift = _descriptor("amd.xdna.aie2p.lshl.i32")
    final_shift = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=final_shift,
        guards=(
            *_typed_guards(("source", "result"), _I32),
            Guard.attr_kind("offset", "i64"),
            Guard.attr_kind("width", "i64"),
            Guard.i64_range("offset", 0, 31),
            Guard.i64_range("width", 1, 32),
        ),
        emit=(
            _const_emit(
                constant,
                ValueRef.temporary("left_shift"),
                AttrProject.i64_literal_minus_attrs(
                    "offset",
                    other_source_attr="width",
                    literal=32,
                ),
                result_type=_I32,
            ),
            _op_emit(
                logical_shift,
                operands={
                    "s0": ValueRef.operand("source"),
                    "s1": ValueRef.temporary("left_shift"),
                },
                results={"d0": ValueRef.temporary("aligned")},
                result_types={"d0": _I32},
            ),
            _const_emit(
                constant,
                ValueRef.temporary("right_shift"),
                AttrProject.i64_attr_minus_literal("width", literal=32),
                result_type=_I32,
            ),
            _op_emit(
                final_shift,
                operands={
                    "s0": ValueRef.temporary("aligned"),
                    "s1": ValueRef.temporary("right_shift"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def aie2p_core_cases() -> Sequence[ContractCase]:
    # Specialized cases precede general cases because the compact runtime table
    # is queried in authored order.
    return (
        ValueAliasRule(
            source_op=buffer.buffer_view,
            source=ValueRef.operand("buffer"),
            result=ValueRef.result("result"),
        ),
        *(
            rule
            for element_byte_count, vector_lane_count, vector_type in (
                (1, 64, _I8X64),
                (2, 32, _I16X32),
                (4, 16, _I32X16),
            )
            for rule in (
                _full_vector_load_rule(
                    vector_type,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                ),
                _full_vector_store_rule(
                    vector_type,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                ),
            )
        ),
        *(
            _address_constant_rule(
                result_type,
                descriptor_key,
                minimum,
                maximum,
            )
            for result_type, minimum, maximum in (
                (_INDEX, _SHORT_MIN, _SHORT_MAX),
                (_OFFSET, 0, _SHORT_MAX),
            )
            for descriptor_key in ("amd.xdna.aie2p.constant.i32.short",)
        ),
        *(
            _address_constant_rule(
                result_type,
                "amd.xdna.aie2p.constant.i32",
                minimum,
                maximum,
            )
            for result_type, minimum, maximum in (
                (_INDEX, _I32_MIN, _I32_MAX),
                (_OFFSET, 0, _I32_MAX),
            )
        ),
        _logical_constant_rule(),
        _constant_rule(
            _I8,
            "amd.xdna.aie2p.constant.i32.short",
            _I8_MIN,
            _I8_MAX,
        ),
        _constant_rule(
            _I16,
            "amd.xdna.aie2p.constant.i32.short",
            _SHORT_MIN,
            _SHORT_MAX,
        ),
        _constant_rule(
            _I16,
            "amd.xdna.aie2p.constant.i32",
            _I16_MIN,
            _I16_MAX,
        ),
        _constant_rule(
            _I32,
            "amd.xdna.aie2p.constant.i32.short",
            _SHORT_MIN,
            _SHORT_MAX,
        ),
        _constant_rule(
            _I32,
            "amd.xdna.aie2p.constant.i32",
            _I32_MIN,
            _I32_MAX,
        ),
        *(
            _conversion_rule(source_op, input_type, _I32, descriptor_key)
            for source_op, input_type, descriptor_key in (
                (
                    scalar_conversion.scalar_extsi,
                    _I8,
                    "amd.xdna.aie2p.extend.signed.i8",
                ),
                (
                    scalar_conversion.scalar_extsi,
                    _I16,
                    "amd.xdna.aie2p.extend.signed.i16",
                ),
                (
                    scalar_conversion.scalar_extui,
                    _I8,
                    "amd.xdna.aie2p.extend.unsigned.i8",
                ),
                (
                    scalar_conversion.scalar_extui,
                    _I16,
                    "amd.xdna.aie2p.extend.unsigned.i16",
                ),
            )
        ),
        *(
            _conversion_alias_rule(
                scalar_conversion.scalar_trunci,
                _I32,
                result_type,
            )
            for result_type in (_I8, _I16)
        ),
        _vector_constant_rule(
            _I8_VECTOR,
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i8x64",
            _I8_MIN,
            _I8_MAX,
        ),
        _vector_constant_rule(
            _I16_VECTOR,
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i16x32",
            _SHORT_MIN,
            _SHORT_MAX,
        ),
        _vector_constant_rule(
            _I16_VECTOR,
            "amd.xdna.aie2p.constant.i32",
            "amd.xdna.aie2p.splat.i16x32",
            _I16_MIN,
            _I16_MAX,
        ),
        _vector_constant_rule(
            _I32_VECTOR,
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i32x16",
            _SHORT_MIN,
            _SHORT_MAX,
        ),
        _vector_constant_rule(
            _I32_VECTOR,
            "amd.xdna.aie2p.constant.i32",
            "amd.xdna.aie2p.splat.i32x16",
            _I32_MIN,
            _I32_MAX,
        ),
        *_vector_broadcast_alias_rules(),
        *(
            _vector_broadcast_rule(element_type, maximum_lanes, descriptor_key)
            for element_type, maximum_lanes, descriptor_key in (
                (
                    "i8",
                    64,
                    "amd.xdna.aie2p.broadcast.i8x64.from-vector",
                ),
                (
                    "i16",
                    32,
                    "amd.xdna.aie2p.broadcast.i16x32.from-vector",
                ),
                (
                    "i32",
                    16,
                    "amd.xdna.aie2p.broadcast.i32x16.from-vector",
                ),
            )
        ),
        *(
            rule
            for (
                scalar_type,
                vector_type,
                maximum_index,
                immediate_key,
                register_key,
            ) in (
                (
                    _I8,
                    _I8_VECTOR,
                    63,
                    "amd.xdna.aie2p.extract.i8.immediate",
                    "amd.xdna.aie2p.extract.i8.register",
                ),
                (
                    _I16,
                    _I16_VECTOR,
                    31,
                    "amd.xdna.aie2p.extract.i16.immediate",
                    "amd.xdna.aie2p.extract.i16.register",
                ),
                (
                    _I32,
                    _I32_VECTOR,
                    15,
                    "amd.xdna.aie2p.extract.i32.immediate",
                    "amd.xdna.aie2p.extract.i32.register",
                ),
            )
            for rule in (
                _vector_extract_static_rule(
                    vector_type,
                    scalar_type,
                    maximum_index,
                    immediate_key,
                ),
                _vector_extract_dynamic_rule(
                    vector_type,
                    scalar_type,
                    register_key,
                ),
            )
        ),
        *(
            rule
            for scalar_type, vector_type, maximum_index, zero_key, register_key in (
                (
                    _I8,
                    _I8_VECTOR,
                    63,
                    "amd.xdna.aie2p.insert.i8.zero",
                    "amd.xdna.aie2p.insert.i8.register",
                ),
                (
                    _I16,
                    _I16_VECTOR,
                    31,
                    "amd.xdna.aie2p.insert.i16.zero",
                    "amd.xdna.aie2p.insert.i16.register",
                ),
                (
                    _I32,
                    _I32_VECTOR,
                    15,
                    "amd.xdna.aie2p.insert.i32.zero",
                    "amd.xdna.aie2p.insert.i32.register",
                ),
            )
            for rule in (
                _vector_insert_zero_rule(scalar_type, vector_type, zero_key),
                _vector_insert_static_rule(
                    scalar_type,
                    vector_type,
                    maximum_index,
                    register_key,
                ),
                _vector_insert_dynamic_rule(
                    scalar_type,
                    vector_type,
                    register_key,
                ),
            )
        ),
        _binary_rule(
            scalar_arithmetic.scalar_addi,
            _I32,
            "amd.xdna.aie2p.add.i32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_subi,
            _I32,
            "amd.xdna.aie2p.sub.i32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_muli,
            _I32,
            "amd.xdna.aie2p.mul.i32",
        ),
        *(
            _binary_rule(source_op, type_pattern, descriptor_key)
            for type_pattern in (_I8, _I16)
            for source_op, descriptor_key in (
                (scalar_arithmetic.scalar_addi, "amd.xdna.aie2p.add.i32"),
                (scalar_arithmetic.scalar_subi, "amd.xdna.aie2p.sub.i32"),
                (scalar_arithmetic.scalar_muli, "amd.xdna.aie2p.mul.i32"),
            )
        ),
        *(
            _vector_binary_rule(source_op, type_pattern, descriptor_key)
            for source_op, type_pattern, descriptor_key in (
                (
                    vector.vector_addi,
                    _I8_VECTOR,
                    "amd.xdna.aie2p.add.i8x64",
                ),
                (
                    vector.vector_subi,
                    _I8_VECTOR,
                    "amd.xdna.aie2p.sub.i8x64",
                ),
                (
                    vector.vector_addi,
                    _I16_VECTOR,
                    "amd.xdna.aie2p.add.i16x32",
                ),
                (
                    vector.vector_subi,
                    _I16_VECTOR,
                    "amd.xdna.aie2p.sub.i16x32",
                ),
                (
                    vector.vector_addi,
                    _I32_VECTOR,
                    "amd.xdna.aie2p.add.i32x16",
                ),
                (
                    vector.vector_subi,
                    _I32_VECTOR,
                    "amd.xdna.aie2p.sub.i32x16",
                ),
            )
        ),
        *(
            _vector_binary_rule(source_op, vector_type, descriptor_key)
            for width, vector_type in (
                (8, _I8_VECTOR),
                (16, _I16_VECTOR),
                (32, _I32_VECTOR),
            )
            for source_op, operation, signedness in (
                (vector.vector_minsi, "min", "signed"),
                (vector.vector_maxsi, "max", "signed"),
                (vector.vector_minui, "min", "unsigned"),
                (vector.vector_maxui, "max", "unsigned"),
            )
            for descriptor_key in (
                f"amd.xdna.aie2p.{operation}.{signedness}.i{width}x{512 // width}",
            )
        ),
        _vector_multiply_i16_rule(),
        *(
            _vector_binary_rule(source_op, type_pattern, descriptor_key)
            for source_op, type_pattern, descriptor_key in (
                (
                    vector.vector_andi,
                    _I8_VECTOR,
                    "amd.xdna.aie2p.and.bits512",
                ),
                (
                    vector.vector_andi,
                    _I16_VECTOR,
                    "amd.xdna.aie2p.and.bits512",
                ),
                (
                    vector.vector_andi,
                    _I32_VECTOR,
                    "amd.xdna.aie2p.and.bits512",
                ),
                (
                    vector.vector_ori,
                    _I8_VECTOR,
                    "amd.xdna.aie2p.or.bits512",
                ),
                (
                    vector.vector_ori,
                    _I16_VECTOR,
                    "amd.xdna.aie2p.or.bits512",
                ),
                (
                    vector.vector_ori,
                    _I32_VECTOR,
                    "amd.xdna.aie2p.or.bits512",
                ),
            )
        ),
        *(
            _vector_xor_rule(type_pattern, subtract_descriptor_key)
            for type_pattern, subtract_descriptor_key in (
                (_I8_VECTOR, "amd.xdna.aie2p.sub.i8x64"),
                (_I16_VECTOR, "amd.xdna.aie2p.sub.i16x32"),
                (_I32_VECTOR, "amd.xdna.aie2p.sub.i32x16"),
            )
        ),
        *(
            _vector_splat_rule(scalar_type, result_type, descriptor_key)
            for scalar_type, result_type, descriptor_key in (
                (_I8, _I8_VECTOR, "amd.xdna.aie2p.splat.i8x64"),
                (_I16, _I16_VECTOR, "amd.xdna.aie2p.splat.i16x32"),
                (_I32, _I32_VECTOR, "amd.xdna.aie2p.splat.i32x16"),
            )
        ),
        _vector_predicate_splat_rule(),
        *(
            _vector_select_rule(value_type, descriptor_key)
            for value_type, descriptor_key in (
                (_I8_VECTOR, "amd.xdna.aie2p.select.i8x64"),
                (_I16_VECTOR, "amd.xdna.aie2p.select.i16x32.mask64"),
                (_I32_VECTOR, "amd.xdna.aie2p.select.i32x16.mask64"),
            )
        ),
        *(
            _vector_predicate_binary_rule(source_op, operation)
            for source_op, operation in (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
            )
        ),
        *(
            _vector_compare_rule(predicate, operand_type, width)
            for width, operand_type in (
                (8, _I8_VECTOR),
                (16, _I16_VECTOR),
                (32, _I32_VECTOR),
            )
            for predicate in (
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
        ),
        *(
            _whole_integer_vector_select_rule(result_type)
            for result_type in _INTEGER_VECTOR_TYPES
        ),
        *_vector_bitcast_alias_rules(),
        *(
            _binary_rule(source_op, type_pattern, descriptor_key)
            for source_op, type_pattern, descriptor_key in (
                (
                    scalar_bitwise.scalar_andi,
                    _I32,
                    "amd.xdna.aie2p.and.i32",
                ),
                (
                    scalar_bitwise.scalar_ori,
                    _I32,
                    "amd.xdna.aie2p.or.i32",
                ),
                (
                    scalar_bitwise.scalar_xori,
                    _I32,
                    "amd.xdna.aie2p.xor.i32",
                ),
                (
                    scalar_bitwise.scalar_andi,
                    _I1,
                    "amd.xdna.aie2p.and.i32",
                ),
                (
                    scalar_bitwise.scalar_ori,
                    _I1,
                    "amd.xdna.aie2p.or.i32",
                ),
                (
                    scalar_bitwise.scalar_xori,
                    _I1,
                    "amd.xdna.aie2p.xor.i32",
                ),
            )
        ),
        *(
            _binary_rule(source_op, type_pattern, descriptor_key)
            for type_pattern in (_I8, _I16)
            for source_op, descriptor_key in (
                (scalar_bitwise.scalar_andi, "amd.xdna.aie2p.and.i32"),
                (scalar_bitwise.scalar_ori, "amd.xdna.aie2p.or.i32"),
                (scalar_bitwise.scalar_xori, "amd.xdna.aie2p.xor.i32"),
            )
        ),
        _binary_rule(
            scalar_bitwise.scalar_shli,
            _I32,
            "amd.xdna.aie2p.lshl.i32",
        ),
        _right_shift_rule(
            scalar_bitwise.scalar_shrsi,
            _I32,
            "amd.xdna.aie2p.ashl.i32",
        ),
        _right_shift_rule(
            scalar_bitwise.scalar_shrui,
            _I32,
            "amd.xdna.aie2p.lshl.i32",
        ),
        _narrow_left_shift_rule(
            _I8,
            "amd.xdna.aie2p.extend.unsigned.i8",
        ),
        _narrow_right_shift_rule(
            scalar_bitwise.scalar_shrsi,
            _I8,
            "amd.xdna.aie2p.extend.signed.i8",
            "amd.xdna.aie2p.extend.unsigned.i8",
            "amd.xdna.aie2p.ashl.i32",
        ),
        _narrow_right_shift_rule(
            scalar_bitwise.scalar_shrui,
            _I8,
            "amd.xdna.aie2p.extend.unsigned.i8",
            "amd.xdna.aie2p.extend.unsigned.i8",
            "amd.xdna.aie2p.lshl.i32",
        ),
        _narrow_left_shift_rule(
            _I16,
            "amd.xdna.aie2p.extend.unsigned.i16",
        ),
        _narrow_right_shift_rule(
            scalar_bitwise.scalar_shrsi,
            _I16,
            "amd.xdna.aie2p.extend.signed.i16",
            "amd.xdna.aie2p.extend.unsigned.i16",
            "amd.xdna.aie2p.ashl.i32",
        ),
        _narrow_right_shift_rule(
            scalar_bitwise.scalar_shrui,
            _I16,
            "amd.xdna.aie2p.extend.unsigned.i16",
            "amd.xdna.aie2p.extend.unsigned.i16",
            "amd.xdna.aie2p.lshl.i32",
        ),
        _bitfield_rule(
            scalar_bitwise.scalar_bitfield_extractu,
            "amd.xdna.aie2p.lshl.i32",
        ),
        _bitfield_rule(
            scalar_bitwise.scalar_bitfield_extracts,
            "amd.xdna.aie2p.ashl.i32",
        ),
        _zero_compare_rule(
            "eq",
            "amd.xdna.aie2p.cmp.eqz.i32",
            zero_field="rhs",
        ),
        _zero_compare_rule(
            "eq",
            "amd.xdna.aie2p.cmp.eqz.i32",
            zero_field="lhs",
        ),
        _zero_compare_rule(
            "ne",
            "amd.xdna.aie2p.cmp.nez.i32",
            zero_field="rhs",
        ),
        _zero_compare_rule(
            "ne",
            "amd.xdna.aie2p.cmp.nez.i32",
            zero_field="lhs",
        ),
        *(
            _compare_rule(predicate, descriptor_key, swap_operands=swap_operands)
            for predicate, descriptor_key, swap_operands in (
                ("eq", "amd.xdna.aie2p.cmp.eq.i32", False),
                ("ne", "amd.xdna.aie2p.cmp.ne.i32", False),
                ("slt", "amd.xdna.aie2p.cmp.slt.i32", False),
                ("sle", "amd.xdna.aie2p.cmp.sge.i32", True),
                ("sgt", "amd.xdna.aie2p.cmp.slt.i32", True),
                ("sge", "amd.xdna.aie2p.cmp.sge.i32", False),
                ("ult", "amd.xdna.aie2p.cmp.ult.i32", False),
                ("ule", "amd.xdna.aie2p.cmp.uge.i32", True),
                ("ugt", "amd.xdna.aie2p.cmp.ult.i32", True),
                ("uge", "amd.xdna.aie2p.cmp.uge.i32", False),
            )
        ),
    )


AIE2P_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
}

AIE2P_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="amd.xdna.aie2p.core",
    descriptor_set=AIE2P_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/amd/xdna/aie2p/contracts/core.h",
    cases=aie2p_core_cases(),
)
