# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P core source-to-Low rule builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum

from loom.dialect.index import defs as index
from loom.dialect.scalar import bitwise as scalar_bitwise
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
from loom.target.arch.amd.xdna.aie2p.contracts.integer_division import (
    append_runtime_unsigned_division_steps,
)
from loom.target.arch.amd.xdna.aie2p.contracts.selection import (
    PredicateUpdate as _PredicateUpdate,
)
from loom.target.arch.amd.xdna.aie2p.contracts.selection import (
    predicate_binary_emits as _predicate_binary_emits,
)
from loom.target.arch.amd.xdna.aie2p.contracts.selection import (
    predicate_select_emits as _predicate_select_emits,
)
from loom.target.arch.amd.xdna.aie2p.contracts.selection import (
    predicate_splat_emits as _vector_predicate_splat_emits,
)
from loom.target.arch.amd.xdna.aie2p.contracts.selection import (
    whole_x_select_emits as _whole_x_select_emits,
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
    EmitRegisterConcat,
    Guard,
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
_I8X128_VECTOR = Vector("i8", lanes=128)
_I8X16_VECTOR = Vector("i8", lanes=16)
_F8E4M3_VECTOR = Vector("f8E4M3", minimum_static_elements=1, maximum_static_elements=64)
_F8E5M2_VECTOR = Vector("f8E5M2", minimum_static_elements=1, maximum_static_elements=64)
_I16_VECTOR = Vector("i16", minimum_static_elements=1, maximum_static_elements=32)
_F16_VECTOR = Vector("f16", minimum_static_elements=1, maximum_static_elements=32)
_BF16_VECTOR = Vector("bf16", minimum_static_elements=1, maximum_static_elements=32)
_I32_VECTOR = Vector("i32", minimum_static_elements=1, maximum_static_elements=16)
_F32_VECTOR = Vector("f32", minimum_static_elements=1, maximum_static_elements=16)
_INDEX_VECTOR = Vector("index", minimum_static_elements=1, maximum_static_elements=16)
_OFFSET_VECTOR = Vector("offset", minimum_static_elements=1, maximum_static_elements=16)
_ADDRESS_VECTOR_TYPES = (_INDEX_VECTOR, _OFFSET_VECTOR)
_F32X32_VECTOR = Vector("f32", lanes=32)
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
_WHOLE_VECTOR_SELECT_TYPES = (*_BITCAST_VECTOR_TYPES, *_ADDRESS_VECTOR_TYPES)

_I8_MIN = -(2**7)
_I8_MAX = (2**7) - 1
_I16_MIN = -(2**15)
_I16_MAX = (2**15) - 1
_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1
_U32_MAX = (2**32) - 1
_SHORT_MIN = -1024
_SHORT_MAX = 1023


class _VectorConstantCarrier(Enum):
    NATIVE = "native"
    WIDE = "wide"
    ACCUMULATOR = "accumulator"


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
    # Unsigned offsets retain their numeric value in source facts while the
    # full-word descriptor encodes the same bits as a signed i32 immediate.
    immediate = (
        ValueProject.exact_i64_i32_word("result", word_index=0)
        if maximum > _I32_MAX
        else AttrProject.direct("value")
    )
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
                immediates={"i": immediate},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
    *,
    result_field: str = "d0",
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
                results={result_field: ValueRef.result("result")},
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
            Guard.value_unsigned_bit_count("result", 32),
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


def _address_select_rule(type_pattern: TypePattern) -> DescriptorRule:
    address_to_scalar = _descriptor("amd.xdna.aie2p.move.local-address-to-scalar")
    scalar_to_address = _descriptor("amd.xdna.aie2p.move.scalar-to-local-address")
    select = _descriptor("amd.xdna.aie2p.select.nonzero.i32")
    true_address = ValueRef.temporary("true_address")
    false_address = ValueRef.temporary("false_address")
    selected_address = ValueRef.temporary("selected_address")
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=select,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                address_to_scalar,
                operands={"src": ValueRef.operand("true_value")},
                results={"dst": true_address},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                address_to_scalar,
                operands={"src": ValueRef.operand("false_value")},
                results={"dst": false_address},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                select,
                operands={
                    "s0": true_address,
                    "s1": false_address,
                    "s2": ValueRef.operand("condition"),
                },
                results={"d0": selected_address},
                result_types={"d0": DescriptorResultType()},
                copy_operands=("s2",),
            ),
            _op_emit(
                scalar_to_address,
                operands={"src": selected_address},
                results={"dst": ValueRef.result("result")},
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
    final_state, _ = append_runtime_unsigned_division_steps(
        emits,
        initial_state=ValueRef.temporary("division_state_0"),
        initial_remainder=ValueRef.temporary("division_remainder_0"),
        divisor=ValueRef.operand("rhs"),
        temporary_prefix="division_",
        final_remainder=None if return_quotient else ValueRef.result("result"),
    )
    if return_quotient:
        emits.append(
            _op_emit(
                move_from_state,
                operands={"src": final_state},
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
    *,
    form: DescriptorEmitForm = DescriptorEmitForm.OP,
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
                    "s1": ValueRef.operand("lhs"),
                    "s2": ValueRef.operand("rhs"),
                },
                results={"d": ValueRef.result("result")},
                form=form,
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


def _vector_wide_f32_splat_rule() -> DescriptorRule:
    """Replicates one F32 value across both native accumulator units."""

    broadcast = _descriptor("amd.xdna.aie2p.splat.i32x16")
    move_to_accumulator = _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512")
    packet = ValueRef.temporary("packet")
    accumulator_unit = ValueRef.temporary("accumulator_unit")
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=broadcast,
        guards=(
            Guard.value_type("scalar", _F32),
            Guard.value_type("result", _F32X32_VECTOR),
        ),
        emit=(
            _op_emit(
                broadcast,
                operands={"src": ValueRef.operand("scalar")},
                results={"dst": packet},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                move_to_accumulator,
                operands={"src": packet},
                results={"dst": accumulator_unit},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterConcat(
                sources=(accumulator_unit, accumulator_unit),
                result=ValueRef.result("result"),
            ),
        ),
        report_key="f32x32_native_splat",
    )


def _vector_predicate_splat_rule() -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=_descriptor("amd.xdna.aie2p.cmp.lt.unsigned.i8x64"),
        guards=(
            Guard.value_type("scalar", _I1),
            Guard.value_type("result", _I1_VECTOR),
        ),
        emit=_vector_predicate_splat_emits(
            ValueRef.operand("scalar"), ValueRef.result("result")
        ),
    )


def _vector_predicate_constant_rule() -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_splati,
        descriptor=_descriptor("amd.xdna.aie2p.cmp.lt.unsigned.i8x64"),
        guards=(
            Guard.value_type("result", _I1_VECTOR),
            Guard.value_exact_i64("result"),
            Guard.value_i64_range("result", 0, 1),
        ),
        emit=(
            _const_emit(
                _descriptor("amd.xdna.aie2p.constant.i32.short"),
                ValueRef.temporary("condition"),
                ValueProject.exact_i64("result"),
                result_type=DescriptorResultType(),
            ),
            *_vector_predicate_splat_emits(
                ValueRef.temporary("condition"), ValueRef.result("result")
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


def _vector_predicate_binary_rules(
    source_op: Op,
    operation: str,
) -> tuple[DescriptorRule, ...]:
    type_guards = _typed_guards(("lhs", "rhs", "result"), _I1_VECTOR)
    return tuple(
        DescriptorRule(
            source_op=source_op,
            descriptor=_descriptor(
                f"amd.xdna.aie2p.predicate.{operation}.high32"
                f"{'.rhs_tied' if update is not _PredicateUpdate.NONE else ''}"
            ),
            guards=(
                *type_guards,
                *((Guard.value_no_uses_after(field),) if field is not None else ()),
            ),
            emit=_predicate_binary_emits(
                operation,
                ValueRef.operand("lhs"),
                ValueRef.operand("rhs"),
                ValueRef.result("result"),
                temporary_prefix="predicate",
                update=update,
            ),
        )
        for update, field in (
            (_PredicateUpdate.RHS, "rhs"),
            (_PredicateUpdate.LHS, "lhs"),
            (_PredicateUpdate.NONE, None),
        )
    )


def _vector_predicate_select_rules() -> tuple[DescriptorRule, ...]:
    type_guards = _typed_guards(
        ("condition", "true_value", "false_value", "result"), _I1_VECTOR
    )
    return tuple(
        DescriptorRule(
            source_op=vector.vector_select,
            descriptor=_descriptor("amd.xdna.aie2p.predicate.xor.high32.rhs_tied"),
            guards=(
                *type_guards,
                *(
                    (Guard.value_no_uses_after("true_value"),)
                    if first_update is _PredicateUpdate.RHS
                    else ()
                ),
            ),
            emit=_predicate_select_emits(
                ValueRef.operand("condition"),
                ValueRef.operand("true_value"),
                ValueRef.operand("false_value"),
                ValueRef.result("result"),
                first_update=first_update,
            ),
        )
        for first_update in (_PredicateUpdate.RHS, _PredicateUpdate.NONE)
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
                    update=_PredicateUpdate.RHS,
                ),
            )
            descriptor = _descriptor("amd.xdna.aie2p.predicate.or.high32.rhs_tied")
        else:
            low = _descriptor("amd.xdna.aie2p.predicate.or.low32")
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
    select = _descriptor("amd.xdna.aie2p.select.i32x16")
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=select,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), result_type),
        ),
        emit=_whole_x_select_emits(
            ValueRef.operand("condition"),
            ValueRef.operand("true_value"),
            ValueRef.operand("false_value"),
            ValueRef.result("result"),
        ),
    )


def _whole_predicate_select_rules() -> tuple[DescriptorRule, ...]:
    """Selects one packed predicate packet with a scalar condition."""

    condition_mask = ValueRef.temporary("condition_mask")
    descriptor = _descriptor("amd.xdna.aie2p.predicate.xor.high32.rhs_tied")
    return tuple(
        DescriptorRule(
            source_op=scf.scf_select,
            descriptor=descriptor,
            guards=(
                Guard.value_type("condition", _I1),
                *_typed_guards(("true_value", "false_value", "result"), _I1_VECTOR),
                *(
                    (Guard.value_no_uses_after("true_value"),)
                    if first_update is _PredicateUpdate.RHS
                    else ()
                ),
            ),
            emit=(
                *_vector_predicate_splat_emits(
                    ValueRef.operand("condition"),
                    condition_mask,
                    result_type=ValueRef.result("result"),
                ),
                *_predicate_select_emits(
                    condition_mask,
                    ValueRef.operand("true_value"),
                    ValueRef.operand("false_value"),
                    ValueRef.result("result"),
                    first_update=first_update,
                ),
            ),
        )
        for first_update in (_PredicateUpdate.RHS, _PredicateUpdate.NONE)
    )


def _vector_constant_emits(
    constant_descriptor_key: str,
    broadcast_descriptor_key: str,
    value: AttrProject | ValueProject,
    carrier: _VectorConstantCarrier,
    unit_count: int,
) -> tuple[EmitDescriptorOp | EmitRegisterConcat, ...]:
    broadcast = _descriptor(broadcast_descriptor_key)
    scalar = ValueRef.temporary("scalar")
    packet = (
        ValueRef.result("result")
        if carrier is _VectorConstantCarrier.NATIVE
        else ValueRef.temporary("packet")
    )
    emits: list[EmitDescriptorOp | EmitRegisterConcat] = [
        _const_emit(
            _descriptor(constant_descriptor_key),
            scalar,
            value,
            result_type=DescriptorResultType(),
        ),
        _op_emit(
            broadcast,
            operands={"src": scalar},
            results={"dst": packet},
            result_types=(
                None
                if carrier is _VectorConstantCarrier.NATIVE
                else {"dst": DescriptorResultType()}
            ),
        ),
    ]
    if carrier is _VectorConstantCarrier.NATIVE:
        return tuple(emits)
    carrier_unit = packet
    if carrier is _VectorConstantCarrier.ACCUMULATOR:
        carrier_unit = ValueRef.temporary("accumulator_unit")
        emits.append(
            _op_emit(
                _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512"),
                operands={"src": packet},
                results={"dst": carrier_unit},
                result_types={"dst": DescriptorResultType()},
            )
        )
    emits.append(
        EmitRegisterConcat(
            sources=(carrier_unit,) * unit_count,
            result=ValueRef.result("result"),
        )
    )
    return tuple(emits)


def _vector_constant_rule(
    result_type: TypePattern,
    constant_descriptor_key: str,
    broadcast_descriptor_key: str,
    minimum: int,
    maximum: int,
    *,
    carrier: _VectorConstantCarrier = _VectorConstantCarrier.NATIVE,
    unit_count: int = 1,
) -> DescriptorRule:
    broadcast = _descriptor(broadcast_descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_splati,
        descriptor=broadcast,
        guards=(
            Guard.value_type("result", result_type),
            Guard.i64_range("value", minimum, maximum),
        ),
        emit=_vector_constant_emits(
            constant_descriptor_key,
            broadcast_descriptor_key,
            AttrProject.direct("value"),
            carrier,
            unit_count,
        ),
    )


def _address_vector_constant_rule(
    result_type: TypePattern,
    constant_descriptor_key: str,
    minimum: int,
    maximum: int,
    *,
    carrier: _VectorConstantCarrier = _VectorConstantCarrier.NATIVE,
    unit_count: int = 1,
) -> DescriptorRule:
    """Materializes target-width address lanes without changing their domain."""

    immediate = (
        ValueProject.exact_i64_i32_word("result", word_index=0)
        if maximum > _I32_MAX
        else AttrProject.direct("value")
    )
    broadcast_descriptor_key = "amd.xdna.aie2p.splat.i32x16"
    return DescriptorRule(
        source_op=vector.vector_splati,
        descriptor=_descriptor(broadcast_descriptor_key),
        guards=(
            Guard.value_type("result", result_type),
            Guard.i64_range("value", minimum, maximum),
        ),
        emit=_vector_constant_emits(
            constant_descriptor_key,
            broadcast_descriptor_key,
            immediate,
            carrier,
            unit_count,
        ),
    )


def _float_vector_constant_rule(
    result_type: TypePattern,
    broadcast_descriptor_key: str,
    bits: ValueProject,
    *,
    carrier: _VectorConstantCarrier = _VectorConstantCarrier.NATIVE,
    unit_count: int = 1,
    extra_guards: Sequence[Guard] = (),
) -> DescriptorRule:
    broadcast = _descriptor(broadcast_descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_splatf,
        descriptor=broadcast,
        guards=(
            Guard.value_type("result", result_type),
            Guard.value_exact_float("result"),
            *extra_guards,
        ),
        emit=_vector_constant_emits(
            "amd.xdna.aie2p.constant.i32",
            broadcast_descriptor_key,
            bits,
            carrier,
            unit_count,
        ),
    )


def _matrix_accumulator_zero_rule() -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.accumulator.clear.i32x64")
    return DescriptorRule(
        source_op=vector.vector_splati,
        descriptor=descriptor,
        guards=(
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


def predicate_boolean_bytes_emits(
    source: ValueRef,
    result: ValueRef,
    *,
    temporary_prefix: str,
) -> tuple[tuple[EmitDescriptorOp, ...], ValueRef]:
    """Materializes one packed predicate packet as canonical 0/1 bytes."""

    one = ValueRef.temporary(f"{temporary_prefix}_one")
    ones = ValueRef.temporary(f"{temporary_prefix}_ones")
    zeros = ValueRef.temporary(f"{temporary_prefix}_zeros")
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    splat = _descriptor("amd.xdna.aie2p.splat.i8x64")
    subtract = _descriptor("amd.xdna.aie2p.sub.i8x64")
    select = _descriptor("amd.xdna.aie2p.select.i8x64")
    return (
        (
            _const_emit(constant, one, 1, result_type=_I8),
            _op_emit(
                splat,
                operands={"src": one},
                results={"dst": ones},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                subtract,
                operands={"s1": ones, "s2": ones},
                results={"d": zeros},
                result_types={"d": DescriptorResultType()},
            ),
            _op_emit(
                select,
                operands={"s1": zeros, "s2": ones, "sel": source},
                results={"d": result},
                result_types={"d": DescriptorResultType()},
            ),
        ),
        zeros,
    )


def _vector_predicate_extract_rule(*, dynamic_index: bool) -> DescriptorRule:
    """Materializes predicate bits as byte lanes before scalar extraction."""

    extract = _descriptor(
        "amd.xdna.aie2p.extract.i8.register"
        if dynamic_index
        else "amd.xdna.aie2p.extract.i8.immediate"
    )
    boolean_bytes = ValueRef.temporary("boolean_bytes")
    materialize_emits, _ = predicate_boolean_bytes_emits(
        ValueRef.operand("source"),
        boolean_bytes,
        temporary_prefix="predicate_extract",
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
                "s1": boolean_bytes,
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
            operands={"s1": boolean_bytes},
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
        emit=(*materialize_emits, extract_emit),
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


def _vector_predicate_insert_rule(
    *,
    dynamic_index: bool,
    zero_index: bool = False,
) -> DescriptorRule:
    """Updates one packed predicate bit through canonical byte lanes."""

    if dynamic_index and zero_index:
        raise ValueError("a predicate insertion index cannot be dynamic and zero")
    insert = _descriptor(
        "amd.xdna.aie2p.insert.i8.zero"
        if zero_index
        else "amd.xdna.aie2p.insert.i8.register"
    )
    compare = _descriptor("amd.xdna.aie2p.cmp.lt.unsigned.i8x64")
    boolean_bytes = ValueRef.temporary("predicate_insert_boolean_bytes")
    updated_bytes = ValueRef.temporary("predicate_insert_updated_bytes")
    materialize_emits, zeros = predicate_boolean_bytes_emits(
        ValueRef.operand("dest"),
        boolean_bytes,
        temporary_prefix="predicate_insert",
    )
    index_emits: tuple[EmitDescriptorOp, ...] = ()
    insert_operands = {
        "s1": boolean_bytes,
        "src": ValueRef.operand("value"),
    }
    copy_operands: tuple[str, ...] = ()
    if dynamic_index:
        index_guards = (
            Guard.value_type("indices", _INDEX),
            Guard.operand_segment_count("indices", 1),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, -(2**63), -(2**63)),
        )
        insert_operands["idx"] = ValueRef.operand("indices")
        copy_operands = ("idx",)
    elif zero_index:
        index_guards = (
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 0, 0),
        )
    else:
        index = ValueRef.temporary("predicate_insert_index")
        index_guards = (
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range("static_indices", 0, 1, 63),
        )
        index_emits = (
            _const_emit(
                _descriptor("amd.xdna.aie2p.constant.i32.short"),
                index,
                AttrProject.i64_array_element("static_indices", element=0),
                result_type=DescriptorResultType(),
            ),
        )
        insert_operands["idx"] = index
        copy_operands = ("idx",)
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=insert,
        guards=(
            Guard.value_type("value", _I1),
            Guard.value_type("dest", _I1_VECTOR),
            Guard.value_type("result", _I1_VECTOR),
            *index_guards,
        ),
        emit=(
            *materialize_emits,
            *index_emits,
            _op_emit(
                insert,
                operands=insert_operands,
                results={"dst": updated_bytes},
                result_types={"dst": DescriptorResultType()},
                copy_operands=copy_operands,
            ),
            _op_emit(
                compare,
                operands={"s1": zeros, "s2": updated_bytes},
                results={"cmp": ValueRef.result("result")},
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


def _right_shift_rules(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> tuple[DescriptorRule, ...]:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    subtract = _descriptor("amd.xdna.aie2p.sub.i32")
    shift = _descriptor(descriptor_key)
    constant_count = DescriptorRule(
        source_op=source_op,
        descriptor=shift,
        priority=1,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            Guard.value_exact_i64("rhs"),
            Guard.value_i64_range("rhs", 0, 31),
        ),
        emit=(
            _const_emit(
                constant,
                ValueRef.temporary("negative_shift"),
                ValueProject.exact_i64_negate("rhs"),
                result_type=type_pattern,
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
    dynamic_count = DescriptorRule(
        source_op=source_op,
        descriptor=shift,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _const_emit(
                constant,
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
    return constant_count, dynamic_count


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
    add_immediate = _descriptor("amd.xdna.aie2p.add.no-carry.i32.immediate")
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
                results={"dst": ValueRef.temporary("minus_one")},
                result_types={"dst": type_pattern},
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
    source_op: Op,
    type_pattern: TypePattern,
    predicate: str,
    descriptor_key: str,
    *,
    zero_field: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    nonzero_field = "rhs" if zero_field == "lhs" else "lhs"
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate),
            *_typed_guards(("lhs", "rhs"), type_pattern),
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
    # Shared legalization keeps low-aligned fields as masks or full-width
    # identities. The native two-shift sequence serves offset fields.
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
            Guard.i64_range("offset", 1, 31),
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
