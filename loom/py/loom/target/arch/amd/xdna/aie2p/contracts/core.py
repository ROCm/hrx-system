# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P core source-to-Low rule builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from loom.dialect.index import defs as index
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import defs as scf
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    vector_data_path_control,
)
from loom.target.arch.amd.xdna.aie2p.contracts.i64 import (
    AIE2P_PAIR_SCALAR_TYPES,
    AIE2P_PAIR_VECTOR_TYPES,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    AttrProject,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterSlice,
    Guard,
    RecipeRule,
    ResultTypeBinding,
    Scalar,
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
_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_I8_VECTOR = Vector("i8", minimum_static_elements=1, maximum_static_elements=64)
_I8X16_VECTOR = Vector("i8", lanes=16)
_I8X32_VECTOR = Vector("i8", lanes=32)
_I8X64_VECTOR = Vector("i8", lanes=64)
_F8E4M3_VECTOR = Vector("f8E4M3", minimum_static_elements=1, maximum_static_elements=64)
_F8E5M2_VECTOR = Vector("f8E5M2", minimum_static_elements=1, maximum_static_elements=64)
_I16_VECTOR = Vector("i16", minimum_static_elements=1, maximum_static_elements=32)
_F16_VECTOR = Vector("f16", minimum_static_elements=1, maximum_static_elements=32)
_BF16_VECTOR = Vector("bf16", minimum_static_elements=1, maximum_static_elements=32)
_BF16X8_VECTOR = Vector("bf16", lanes=8)
_I32_VECTOR = Vector("i32", minimum_static_elements=1, maximum_static_elements=16)
_F32_VECTOR = Vector("f32", minimum_static_elements=1, maximum_static_elements=16)
_I32_MATRIX_ACCUMULATOR = Vector("i32", lanes=64)
_I1_VECTOR = Vector("i1", minimum_static_elements=1, maximum_static_elements=64)
_I1X2X64_VECTOR = Vector("i1", dims=(2, 64))
_INTEGER_VECTOR_TYPES = (_I8_VECTOR, _I16_VECTOR, _I32_VECTOR)
_BITCAST_VECTOR_TYPES = (
    _I8_VECTOR,
    _F8E4M3_VECTOR,
    _F8E5M2_VECTOR,
    _I16_VECTOR,
    _F16_VECTOR,
    _BF16_VECTOR,
    _I32_VECTOR,
    _F32_VECTOR,
    *AIE2P_PAIR_VECTOR_TYPES,
)

_I8_MIN = -(2**7)
_I8_MAX = (2**7) - 1
_I16_MIN = -(2**15)
_I16_MAX = (2**15) - 1
_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1
_SHORT_MIN = -1024
_SHORT_MAX = 1023


_I16_ELEMENTWISE_MULTIPLY_CONTROL = vector_data_path_control(
    sign_x=True,
    sign_y=True,
    accumulator_mode=1,
    multiplication_mode=3,
    compute_mode=2,
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
    value: AttrProject | ValueProject | int,
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


def _float_constant_rule(
    result_type: TypePattern, bits: ValueProject
) -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.constant.i32")
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", result_type),
            Guard.value_exact_float("result"),
        ),
        emit=(
            _const_emit(
                descriptor,
                ValueRef.result("result"),
                bits,
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
                descriptor,
                operands={"s0": ValueRef.operand("input")},
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _madd_rule(source_op: Op, type_pattern: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.madd.i32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("a", "b", "c", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor,
                operands={
                    "a0": ValueRef.operand("c"),
                    "s0": ValueRef.operand("a"),
                    "s1": ValueRef.operand("b"),
                },
                results={"d0": ValueRef.result("result")},
                copy_operands=("a0",),
            ),
        ),
    )


def _index_scale_rule() -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.mul.i32")
    return DescriptorRule(
        source_op=index.index_scale,
        descriptor=descriptor,
        guards=(
            Guard.value_type("index", _INDEX),
            Guard.value_type("stride", _OFFSET),
            Guard.value_type("result", _OFFSET),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={
                    "s0": ValueRef.operand("index"),
                    "s1": ValueRef.operand("stride"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _scalar_select_rule(type_pattern: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.select.nonzero.i32")
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                descriptor,
                operands={
                    "s0": ValueRef.operand("true_value"),
                    "s1": ValueRef.operand("false_value"),
                    "s2": ValueRef.operand("condition"),
                },
                results={"d0": ValueRef.result("result")},
                copy_operands=("s2",),
            ),
        ),
    )


def _integer_minmax_rule(
    source_op: Op,
    type_pattern: TypePattern,
    compare_descriptor_key: str,
    *,
    swap_compare_operands: bool,
) -> DescriptorRule:
    compare = _descriptor(compare_descriptor_key)
    select = _descriptor("amd.xdna.aie2p.select.nonzero.i32")
    lhs = ValueRef.operand("rhs" if swap_compare_operands else "lhs")
    rhs = ValueRef.operand("lhs" if swap_compare_operands else "rhs")
    return DescriptorRule(
        source_op=source_op,
        descriptor=select,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                compare,
                operands={"s0": lhs, "s1": rhs},
                results={"d0": ValueRef.temporary("condition")},
                result_types={"d0": DescriptorResultType()},
            ),
            _op_emit(
                select,
                operands={
                    "s0": ValueRef.operand("lhs"),
                    "s1": ValueRef.operand("rhs"),
                    "s2": ValueRef.temporary("condition"),
                },
                results={"d0": ValueRef.result("result")},
            ),
        ),
    )


def _unsigned_division_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    return_quotient: bool,
) -> DescriptorRule:
    move_to_state = _descriptor("amd.xdna.aie2p.move.to.division-state")
    move_from_state = _descriptor("amd.xdna.aie2p.move.from.division-state")
    zero = _descriptor("amd.xdna.aie2p.constant.i32.short")
    divide_step = _descriptor("amd.xdna.aie2p.divide.step.unsigned.i32")

    emits = [
        _op_emit(
            move_to_state,
            operands={"src": ValueRef.operand("lhs")},
            results={"dst": ValueRef.temporary("division_state_0")},
            result_types={"dst": DescriptorResultType()},
        ),
        _const_emit(
            zero,
            ValueRef.temporary("division_remainder_0"),
            0,
            result_type=type_pattern,
        ),
    ]
    for step in range(1, 33):
        is_final_remainder = step == 32 and not return_quotient
        remainder = (
            ValueRef.result("result")
            if is_final_remainder
            else ValueRef.temporary(f"division_remainder_{step}")
        )
        emits.append(
            _op_emit(
                divide_step,
                operands={
                    "sd": ValueRef.temporary(f"division_state_{step - 1}"),
                    "s0": ValueRef.temporary(f"division_remainder_{step - 1}"),
                    "s1": ValueRef.operand("rhs"),
                },
                results={
                    "d0": remainder,
                    "sd_out": ValueRef.temporary(f"division_state_{step}"),
                },
                result_types={
                    "d0": DescriptorResultType(),
                    "sd_out": DescriptorResultType(),
                },
            )
        )
    if return_quotient:
        emits.append(
            _op_emit(
                move_from_state,
                operands={"src": ValueRef.temporary("division_state_32")},
                results={"dst": ValueRef.result("result")},
            )
        )

    return DescriptorRule(
        source_op=source_op,
        descriptor=divide_step,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
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


def _vector_bitunpack_i4_rule(source_op: Op, descriptor_key: str) -> DescriptorRule:
    set_unpack_size = _descriptor("amd.xdna.aie2p.state.unpack-size.immediate")
    unpack = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=unpack,
        guards=(
            Guard.value_type("source", _I8X32_VECTOR),
            Guard.value_type("result", _I8X64_VECTOR),
            Guard.attr_kind("width", "i64"),
            Guard.i64_range("width", 4, 4),
        ),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("source"),
                result=ValueRef.temporary("packed_source"),
                unit_count=1,
            ),
            EmitDescriptorOp(
                descriptor=set_unpack_size,
                immediates={"i": 0},
                form=DescriptorEmitForm.OP,
            ),
            _op_emit(
                unpack,
                operands={"src": ValueRef.temporary("packed_source")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _vector_bitunpack_i1_alias_rule() -> ValueAliasRule:
    """Keeps a packed 128-bit predicate stream in its loaded X carrier."""

    return ValueAliasRule(
        source_op=vector.vector_bitunpacku,
        source=ValueRef.operand("source"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("source", _I8X16_VECTOR),
            Guard.value_type("result", _I1X2X64_VECTOR),
            Guard.attr_kind("width", "i64"),
            Guard.i64_range("width", 1, 1),
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


def _whole_vector_select_rule(
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


def _float_vector_constant_rule(
    result_type: TypePattern,
    broadcast_descriptor_key: str,
    bits: ValueProject,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32")
    broadcast = _descriptor(broadcast_descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=broadcast,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", result_type),
            Guard.value_exact_float("result"),
        ),
        emit=(
            _const_emit(
                constant,
                ValueRef.temporary("scalar"),
                bits,
                result_type=DescriptorResultType(),
            ),
            _op_emit(
                broadcast,
                operands={"src": ValueRef.temporary("scalar")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _matrix_accumulator_zero_rule() -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.accumulator.clear.i32x64")
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", _I32_MATRIX_ACCUMULATOR),
            Guard.i64_range("value", 0, 0),
        ),
        emit=(
            _op_emit(
                descriptor,
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _matrix_fragment_store_rule() -> RecipeRule:
    return RecipeRule(
        source_op=vector.vector_fragment_store,
        guards=(
            Guard.enum_attr_equals("role", "result"),
            Guard.value_type("value", _I32_MATRIX_ACCUMULATOR),
            Guard.value_type("view", TypePattern.view("i32", dims=(8, 8))),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 2),
            Guard.i64_array_elements_range("static_indices", 0, 0),
            Guard.value_i64_range("rows", 8, 8),
            Guard.value_i64_range("columns", 8, 8),
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


def _vector_predicate_extract_rule(*, dynamic_index: bool) -> DescriptorRule:
    """Materializes predicate bits as byte lanes before scalar extraction."""

    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    splat = _descriptor("amd.xdna.aie2p.splat.i8x64")
    subtract = _descriptor("amd.xdna.aie2p.sub.i8x64")
    select = _descriptor("amd.xdna.aie2p.select.i8x64")
    extract = _descriptor(
        "amd.xdna.aie2p.extract.i8.register"
        if dynamic_index
        else "amd.xdna.aie2p.extract.i8.immediate"
    )
    guards = [
        Guard.value_type("source", _I1_VECTOR),
        Guard.value_type("result", _I1),
    ]
    if dynamic_index:
        guards.extend(
            (
                Guard.value_type("indices", _INDEX),
                Guard.operand_segment_count("indices", 1),
                Guard.i64_array_count("static_indices", 1),
                Guard.i64_array_element_range("static_indices", 0, -(2**63), -(2**63)),
            )
        )
        extract_emit = _op_emit(
            extract,
            operands={
                "s1": ValueRef.temporary("boolean_bytes"),
                "idx": ValueRef.operand("indices"),
            },
            results={"dst": ValueRef.result("result")},
        )
    else:
        guards.extend(
            (
                Guard.operand_segment_count("indices", 0),
                Guard.i64_array_count("static_indices", 1),
                Guard.i64_array_element_range("static_indices", 0, 0, 63),
            )
        )
        extract_emit = EmitDescriptorOp(
            descriptor=extract,
            operands={"s1": ValueRef.temporary("boolean_bytes")},
            results={"dst": ValueRef.result("result")},
            immediates={
                "idx": AttrProject.i64_array_element("static_indices", element=0)
            },
            form=DescriptorEmitForm.OP,
        )
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=extract,
        guards=tuple(guards),
        emit=(
            _const_emit(
                constant,
                ValueRef.temporary("one"),
                1,
                result_type=_I8,
            ),
            _op_emit(
                splat,
                operands={"src": ValueRef.temporary("one")},
                results={"dst": ValueRef.temporary("ones")},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                subtract,
                operands={
                    "s1": ValueRef.temporary("ones"),
                    "s2": ValueRef.temporary("ones"),
                },
                results={"d": ValueRef.temporary("zeros")},
                result_types={"d": DescriptorResultType()},
            ),
            _op_emit(
                select,
                operands={
                    "s1": ValueRef.temporary("zeros"),
                    "s2": ValueRef.temporary("ones"),
                    "sel": ValueRef.operand("source"),
                },
                results={"d": ValueRef.temporary("boolean_bytes")},
                result_types={"d": DescriptorResultType()},
            ),
            extract_emit,
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
        for source_type in _BITCAST_VECTOR_TYPES
        for result_type in _BITCAST_VECTOR_TYPES
    )


def _scalar_bitcast_alias_rules() -> tuple[ValueAliasRule, ...]:
    same_width_types = (
        (_I8, _F8E4M3, _F8E5M2),
        (_I16, _F16, _BF16),
        (_I32, _F32),
        AIE2P_PAIR_SCALAR_TYPES,
    )
    return tuple(
        _conversion_alias_rule(
            scalar_conversion.scalar_bitcast,
            source_type,
            result_type,
        )
        for type_group in same_width_types
        for source_type in type_group
        for result_type in type_group
        if source_type != result_type
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
                result_type=type_pattern,
            ),
            _op_emit(
                subtract,
                operands={
                    "s0": ValueRef.temporary("zero"),
                    "s1": ValueRef.operand("rhs"),
                },
                results={"d0": ValueRef.temporary("negative_shift")},
                result_types={"d0": type_pattern},
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


def _rotate_rule(
    source_op: Op,
    type_pattern: TypePattern,
    *,
    rotate_left: bool,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    subtract = _descriptor("amd.xdna.aie2p.sub.i32")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.i32")
    logical_shift = _descriptor("amd.xdna.aie2p.lshl.i32")
    bitwise_or = _descriptor("amd.xdna.aie2p.or.i32")
    emits = [
        _const_emit(
            constant,
            ValueRef.temporary("zero"),
            0,
            result_type=type_pattern,
        ),
        _const_emit(
            constant,
            ValueRef.temporary("shift_mask"),
            31,
            result_type=type_pattern,
        ),
        _op_emit(
            subtract,
            operands={
                "s0": ValueRef.temporary("zero"),
                "s1": ValueRef.operand("rhs"),
            },
            results={"d0": ValueRef.temporary("negative_amount")},
            result_types={"d0": type_pattern},
        ),
        _op_emit(
            bitwise_and,
            operands={
                "s0": ValueRef.operand("rhs"),
                "s1": ValueRef.temporary("shift_mask"),
            },
            results={"d0": ValueRef.temporary("masked_amount")},
            result_types={"d0": type_pattern},
        ),
        _op_emit(
            bitwise_and,
            operands={
                "s0": ValueRef.temporary("negative_amount"),
                "s1": ValueRef.temporary("shift_mask"),
            },
            results={"d0": ValueRef.temporary("wrap_amount")},
            result_types={"d0": type_pattern},
        ),
    ]
    if rotate_left:
        emits.extend(
            (
                _op_emit(
                    logical_shift,
                    operands={
                        "s0": ValueRef.operand("lhs"),
                        "s1": ValueRef.temporary("masked_amount"),
                    },
                    results={"d0": ValueRef.temporary("left_piece")},
                    result_types={"d0": type_pattern},
                ),
                _op_emit(
                    subtract,
                    operands={
                        "s0": ValueRef.temporary("zero"),
                        "s1": ValueRef.temporary("wrap_amount"),
                    },
                    results={"d0": ValueRef.temporary("negative_wrap_amount")},
                    result_types={"d0": type_pattern},
                ),
                _op_emit(
                    logical_shift,
                    operands={
                        "s0": ValueRef.operand("lhs"),
                        "s1": ValueRef.temporary("negative_wrap_amount"),
                    },
                    results={"d0": ValueRef.temporary("right_piece")},
                    result_types={"d0": type_pattern},
                ),
            )
        )
    else:
        emits.extend(
            (
                _op_emit(
                    subtract,
                    operands={
                        "s0": ValueRef.temporary("zero"),
                        "s1": ValueRef.temporary("masked_amount"),
                    },
                    results={"d0": ValueRef.temporary("negative_masked_amount")},
                    result_types={"d0": type_pattern},
                ),
                _op_emit(
                    logical_shift,
                    operands={
                        "s0": ValueRef.operand("lhs"),
                        "s1": ValueRef.temporary("negative_masked_amount"),
                    },
                    results={"d0": ValueRef.temporary("right_piece")},
                    result_types={"d0": type_pattern},
                ),
                _op_emit(
                    logical_shift,
                    operands={
                        "s0": ValueRef.operand("lhs"),
                        "s1": ValueRef.temporary("wrap_amount"),
                    },
                    results={"d0": ValueRef.temporary("left_piece")},
                    result_types={"d0": type_pattern},
                ),
            )
        )
    emits.append(
        _op_emit(
            bitwise_or,
            operands={
                "s0": ValueRef.temporary("left_piece"),
                "s1": ValueRef.temporary("right_piece"),
            },
            results={"d0": ValueRef.result("result")},
        )
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=bitwise_or,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=tuple(emits),
    )


def _count_trailing_zeros_rule(
    source_op: Op,
    type_pattern: TypePattern,
) -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    add_immediate = _descriptor("amd.xdna.aie2p.add.i32.immediate")
    bitwise_xor = _descriptor("amd.xdna.aie2p.xor.i32")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.i32")
    popcount = _descriptor("amd.xdna.aie2p.popcount.i32")
    select_zero = _descriptor("amd.xdna.aie2p.select.zero.i32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=select_zero,
        guards=_typed_guards(("input", "result"), type_pattern),
        emit=(
            _const_emit(
                constant,
                ValueRef.temporary("all_ones"),
                -1,
                result_type=type_pattern,
            ),
            _op_emit(
                bitwise_xor,
                operands={
                    "s0": ValueRef.operand("input"),
                    "s1": ValueRef.temporary("all_ones"),
                },
                results={"d0": ValueRef.temporary("inverted")},
                result_types={"d0": type_pattern},
            ),
            EmitDescriptorOp(
                descriptor=add_immediate,
                operands={"s0": ValueRef.operand("input")},
                results={"d0": ValueRef.temporary("minus_one")},
                result_types={"d0": type_pattern},
                immediates={"imm": -1},
                form=DescriptorEmitForm.OP,
            ),
            _op_emit(
                bitwise_and,
                operands={
                    "s0": ValueRef.temporary("inverted"),
                    "s1": ValueRef.temporary("minus_one"),
                },
                results={"d0": ValueRef.temporary("trailing_mask")},
                result_types={"d0": type_pattern},
            ),
            _op_emit(
                popcount,
                operands={"s0": ValueRef.temporary("trailing_mask")},
                results={"d0": ValueRef.temporary("nonzero_count")},
                result_types={"d0": type_pattern},
            ),
            _const_emit(
                constant,
                ValueRef.temporary("bit_width"),
                32,
                result_type=type_pattern,
            ),
            _op_emit(
                select_zero,
                operands={
                    "s0": ValueRef.temporary("bit_width"),
                    "s1": ValueRef.temporary("nonzero_count"),
                    "s2": ValueRef.operand("input"),
                },
                results={"d0": ValueRef.result("result")},
                copy_operands=("s2",),
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
    source_op: Op,
    type_pattern: TypePattern,
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
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            *_typed_guards(("lhs", "rhs"), type_pattern),
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
