# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P native floating-point selection rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amd.xdna.aie2p.contracts.conversion import (
    emit_f16_to_f32,
    emit_f32_to_f16,
)
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    F32_ACCUMULATOR_ADD_CONTROL,
    vector_data_path_control,
)
from loom.target.arch.amd.xdna.aie2p.contracts.f32 import emit_f32_multiply
from loom.target.arch.amd.xdna.aie2p.contracts.f32_accumulator import (
    emit_f32_scalar_accumulator_binary,
    emit_f32_vector_accumulator_binary,
)
from loom.target.arch.amd.xdna.aie2p.contracts.scalar_program import ScalarProgram
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    ResultTypeBinding,
    Scalar,
    SourceNode,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_F32 = Scalar("f32")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_BF16_VECTOR = Vector("bf16", minimum_static_elements=1, maximum_static_elements=32)
_BF16X8_VECTOR = Vector("bf16", lanes=8)
_BF16X16_VECTOR = Vector("bf16", lanes=16)
_BF16_DOT2_VECTOR = Vector(
    "bf16", minimum_static_elements=2, maximum_static_elements=32
)
_BF16X32_VECTOR = Vector("bf16", lanes=32)
_BF16X64_VECTOR = Vector("bf16", lanes=64)
_F32_VECTOR = Vector("f32", minimum_static_elements=1, maximum_static_elements=16)
_F32X16_VECTOR = Vector("f32", lanes=16)
_F32X32_VECTOR = Vector("f32", lanes=32)
_I16X32_VECTOR = Vector("i16", lanes=32)
_F32X64_ACCUMULATOR = Vector("f32", lanes=64)

_BF16_ELEMENTWISE_MULTIPLY_CONTROL = vector_data_path_control(
    sign_x=False,
    sign_y=False,
    accumulator_mode=2,
    multiplication_mode=3,
    compute_mode=1,
)
_U16_ELEMENTWISE_MULTIPLY_CONTROL = vector_data_path_control(
    sign_x=False,
    sign_y=False,
    accumulator_mode=1,
    multiplication_mode=3,
    compute_mode=2,
)

# The attention scale 1.4453125 is exactly BF16 0x3FB9. Native BF16 VMUL
# computes its normal and overflowing products exactly in F32 but flushes F32
# subnormal products. For BF16 magnitudes below 89, multiplying the encoded
# magnitude by 185 << 8 constructs the exact F32 subnormal payload in packed
# integer lanes. Shuffle control 18 interleaves those low/high halfwords into
# F32 lanes before the final repair select.
_BF16_ORIGIN_SCALE = 1.4453125
_BF16_ORIGIN_SCALE_BITS = 0x3FB9
_BF16_ORIGIN_SCALE_REPAIR_FACTOR = 185 << 8
_BF16_ORIGIN_SCALE_REPAIR_LIMIT = 89
_BF16_ORIGIN_SCALE_REPAIR_SHUFFLE = 18
# AIE2P T16_32x2_lo/hi select the even and odd BF16 lanes from a 512-bit
# source. Two ordered VMACs over those streams implement vector.dot2f's two
# sequential fused accumulations without weakening its exact source contract.
_BF16_DOT2_DEINTERLEAVE_CONTROLS = (2, 3)

_BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS = (52, 53)
_BF16_OUTER_PRODUCT_MULTIPLY_CONTROL = _BF16_ELEMENTWISE_MULTIPLY_CONTROL


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
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        form=DescriptorEmitForm.OP,
    )


def _constant_emit(
    descriptor: Descriptor,
    result: ValueRef,
    value: int,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        results={"dst": result},
        result_types={"dst": DescriptorResultType()},
        immediates={"i": value},
        form=DescriptorEmitForm.CONST,
    )


def _scalar_multiply_f16_rule() -> DescriptorRule:
    """Multiplies binary16 exactly through the binary32 software datapath."""

    # A binary16 product has at most 22 significant bits and remains within
    # binary32's exponent range, so widening both operands and multiplying in
    # binary32 computes the exact finite product before the final narrowing.
    lhs_program = ScalarProgram("lhs_")
    lhs = emit_f16_to_f32(
        lhs_program,
        ValueRef.operand("lhs"),
        "wide",
    )
    rhs_program = ScalarProgram("rhs_")
    rhs = emit_f16_to_f32(
        rhs_program,
        ValueRef.operand("rhs"),
        "wide",
    )
    multiply_emits = emit_f32_multiply(
        lhs=lhs,
        rhs=rhs,
        result_name="result",
        temporary_prefix="multiply_",
    )
    product = ValueRef.temporary("multiply_result")
    narrow_program = ScalarProgram("narrow_")
    emit_f32_to_f16(narrow_program, product, None)
    return DescriptorRule(
        source_op=scalar_arithmetic.scalar_mulf,
        descriptor=narrow_program.emits[-1].descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), _F16),
        emit=(
            *lhs_program.emits,
            *rhs_program.emits,
            *multiply_emits,
            *narrow_program.emits,
        ),
        report_key="exact_binary16",
    )


def _vector_multiply_bf16_origin_scale_rule() -> DescriptorRule:
    """Multiplies BF16-origin F32 lanes by the qualified attention scale."""

    scalar_constant = _descriptor("amd.xdna.aie2p.constant.i32")
    short_constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shift_constant = _descriptor("amd.xdna.aie2p.constant.i32.shift")
    broadcast = _descriptor("amd.xdna.aie2p.splat.i16x32")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.bits512")
    bitwise_or = _descriptor("amd.xdna.aie2p.or.bits512")
    float_multiply = _descriptor("amd.xdna.aie2p.multiply.bf16x32.configured")
    integer_multiply = _descriptor("amd.xdna.aie2p.multiply.i16x32.configured")
    narrow = _descriptor("amd.xdna.aie2p.narrow.trunc.signed.i16x32")
    add = _descriptor("amd.xdna.aie2p.add.i16x32")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    compare = _descriptor("amd.xdna.aie2p.cmp.lt.unsigned.i16x32.el.low32")
    select = _descriptor("amd.xdna.aie2p.select.i32x16.mask64")
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )
    set_rounding = _descriptor("amd.xdna.aie2p.state.rounding.immediate")
    set_srs_mode = _descriptor("amd.xdna.aie2p.state.srs-mode.immediate")
    set_saturation = _descriptor("amd.xdna.aie2p.state.saturation.immediate")

    emits: list[ContractEmit] = []

    def temporary(name: str) -> ValueRef:
        return ValueRef.temporary(name)

    def constant(
        name: str,
        value: int,
        descriptor: Descriptor = scalar_constant,
    ) -> ValueRef:
        result = temporary(name)
        emits.append(_constant_emit(descriptor, result, value))
        return result

    def operation(
        name: str,
        descriptor: Descriptor,
        result_field: str,
        **operands: ValueRef,
    ) -> ValueRef:
        result = temporary(name)
        emits.append(
            _op_emit(
                descriptor,
                operands=operands,
                results={result_field: result},
                result_types={result_field: DescriptorResultType()},
            )
        )
        return result

    bf16_input = ValueRef.exact_lane_origin_operand("lhs")

    scale_scalar = constant("scale_scalar", _BF16_ORIGIN_SCALE_BITS)
    scale = operation("scale", broadcast, "dst", src=scale_scalar)
    float_control = constant(
        "float_control",
        _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
        config_constant,
    )
    raw_products = operation(
        "raw_products",
        float_multiply,
        "dst",
        s1=bf16_input,
        s2=scale,
        acc=float_control,
    )
    raw_product_unit = temporary("raw_product_unit")
    emits.append(
        EmitRegisterSlice(
            source=raw_products,
            result=raw_product_unit,
            unit_count=1,
        )
    )
    direct_product = operation(
        "direct_product",
        move_from_accumulator,
        "dst",
        src=raw_product_unit,
    )

    absolute_mask_scalar = constant("absolute_mask_scalar", 0x7FFF)
    absolute_mask = operation(
        "absolute_mask", broadcast, "dst", src=absolute_mask_scalar
    )
    magnitudes = operation(
        "magnitudes",
        bitwise_and,
        "d",
        s1=bf16_input,
        s2=absolute_mask,
    )
    repair_factor_scalar = constant(
        "repair_factor_scalar", _BF16_ORIGIN_SCALE_REPAIR_FACTOR
    )
    repair_factor = operation(
        "repair_factor", broadcast, "dst", src=repair_factor_scalar
    )
    integer_control = constant(
        "integer_control",
        _U16_ELEMENTWISE_MULTIPLY_CONTROL,
        config_constant,
    )
    wide_repair = operation(
        "wide_repair",
        integer_multiply,
        "dst",
        s1=magnitudes,
        s2=repair_factor,
        acc=integer_control,
    )
    low_shift = constant("low_shift", 0, shift_constant)
    high_shift = constant("high_shift", 15, shift_constant)
    emits.extend(
        (
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
        )
    )
    low_base = operation("low_base", narrow, "dst", src=wide_repair, su=low_shift)
    high = operation("high", narrow, "dst", src=wide_repair, su=high_shift)
    low = operation("low", add, "d", s1=low_base, s2=low_base)
    sign_mask_scalar = constant("sign_mask_scalar", 0x8000)
    sign_mask = operation("sign_mask", broadcast, "dst", src=sign_mask_scalar)
    sign = operation("sign", bitwise_and, "d", s1=bf16_input, s2=sign_mask)
    signed_high = operation("signed_high", bitwise_or, "d", s1=high, s2=sign)
    repair_shuffle = constant(
        "repair_shuffle",
        _BF16_ORIGIN_SCALE_REPAIR_SHUFFLE,
        config_constant,
    )
    repaired_product = operation(
        "repaired_product",
        shuffle,
        "dst",
        s1=low,
        s2=signed_high,
        mod=repair_shuffle,
    )
    repair_limit_scalar = constant(
        "repair_limit_scalar",
        _BF16_ORIGIN_SCALE_REPAIR_LIMIT,
        short_constant,
    )
    repair_limit = operation("repair_limit", broadcast, "dst", src=repair_limit_scalar)
    repair_lanes = operation(
        "repair_lanes",
        compare,
        "cmp",
        s1=magnitudes,
        s2=repair_limit,
    )
    emits.append(
        _op_emit(
            select,
            operands={
                "s1": direct_product,
                "s2": repaired_product,
                "sel": repair_lanes,
            },
            results={"d": ValueRef.result("result")},
        )
    )

    return DescriptorRule(
        source_op=vector.vector_mulf,
        descriptor=float_multiply,
        guards=(
            Guard.value_type("lhs", _F32X16_VECTOR),
            Guard.exact_lane_origin_type("lhs", _BF16X16_VECTOR),
            Guard.value_type("rhs", _F32X16_VECTOR),
            Guard.value_type("result", _F32X16_VECTOR),
            Guard.value_float_equals("rhs", _BF16_ORIGIN_SCALE),
            Guard.instance_flags_has_all("fastmath", "nnan"),
        ),
        emit=tuple(emits),
        report_key="exact_bf16_origin_scale_1_4453125",
    )


def _bf16_vector_scalar_product_emits(
    lane_count: Literal[16, 32],
    *,
    lhs: ValueRef,
    rhs_scalar: ValueRef,
    result: ValueRef,
    result_type: ResultTypeBinding | None = None,
) -> tuple[ContractEmit, ...]:
    """Emits an exact BF16-origin product rounded back to BF16."""

    scalar_constant = _descriptor("amd.xdna.aie2p.constant.i32")
    short_constant = _descriptor("amd.xdna.aie2p.constant.i32.short")
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shift_constant = _descriptor("amd.xdna.aie2p.constant.i32.shift")
    broadcast = _descriptor("amd.xdna.aie2p.splat.i16x32")
    bitwise_and = _descriptor("amd.xdna.aie2p.and.bits512")
    bitwise_or = _descriptor("amd.xdna.aie2p.or.bits512")
    add = _descriptor("amd.xdna.aie2p.add.i16x32")
    minimum = _descriptor("amd.xdna.aie2p.min.unsigned.i16x32")
    maximum = _descriptor("amd.xdna.aie2p.max.unsigned.i16x32")
    float_multiply = _descriptor("amd.xdna.aie2p.multiply.bf16x32.configured")
    float_narrow = _descriptor(
        f"amd.xdna.aie2p.convert.f32x{lane_count}.to.bf16x{lane_count}"
    )
    integer_multiply = _descriptor("amd.xdna.aie2p.multiply.i16x32.configured")
    integer_narrow = _descriptor("amd.xdna.aie2p.narrow.trunc.signed.i16x32")
    floor_bf16 = _descriptor("amd.xdna.aie2p.convert.floor.bf16x16.to.i32x16")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    compare_zero = _descriptor("amd.xdna.aie2p.cmp.eqz.i16x32.el.low32")
    complete_predicate = _descriptor("amd.xdna.aie2p.predicate.complete.zero.high32")
    select = _descriptor("amd.xdna.aie2p.select.i16x32.mask64")
    set_rounding = _descriptor("amd.xdna.aie2p.state.rounding.immediate")
    set_srs_mode = _descriptor("amd.xdna.aie2p.state.srs-mode.immediate")

    emits: list[ContractEmit] = []

    def temporary(name: str) -> ValueRef:
        return ValueRef.temporary(name)

    def constant(
        name: str,
        value: int,
        descriptor: Descriptor = scalar_constant,
    ) -> ValueRef:
        result = temporary(name)
        emits.append(_constant_emit(descriptor, result, value))
        return result

    def operation(
        name: str,
        descriptor: Descriptor,
        result_field: str,
        **operands: ValueRef,
    ) -> ValueRef:
        result = temporary(name)
        emits.append(
            _op_emit(
                descriptor,
                operands=operands,
                results={result_field: result},
                result_types={result_field: DescriptorResultType()},
            )
        )
        return result

    rhs = operation("rhs", broadcast, "dst", src=rhs_scalar)

    float_control = constant(
        "float_control",
        _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
        config_constant,
    )
    raw_products = operation(
        "raw_products",
        float_multiply,
        "dst",
        s1=lhs,
        s2=rhs,
        acc=float_control,
    )
    raw_product_units = temporary("raw_product_units")
    emits.append(
        EmitRegisterSlice(
            source=raw_products,
            result=raw_product_units,
            unit_count=lane_count // 16,
        )
    )
    emits.append(
        EmitDescriptorOp(
            descriptor=set_rounding,
            immediates={"i": 12},
            form=DescriptorEmitForm.OP,
        )
    )
    if lane_count == 16:
        direct_low = operation("direct_low", float_narrow, "dst", src=raw_product_units)
        padding = temporary("padding")
        emits.append(
            EmitRegisterSlice(
                source=rhs,
                result=padding,
                unit_offset=1,
                unit_count=1,
            )
        )
        direct = temporary("direct")
        emits.append(
            EmitRegisterConcat(
                sources=(direct_low, padding),
                result=direct,
                result_type=_BF16X32_VECTOR,
            )
        )
    else:
        direct = operation("direct", float_narrow, "dst", src=raw_product_units)

    exponent_mask_scalar = constant("exponent_mask_scalar", 0x7F80)
    exponent_mask = operation(
        "exponent_mask", broadcast, "dst", src=exponent_mask_scalar
    )
    lhs_exponent = operation("lhs_exponent", bitwise_and, "d", s1=lhs, s2=exponent_mask)
    rhs_exponent = operation("rhs_exponent", bitwise_and, "d", s1=rhs, s2=exponent_mask)
    hidden_bit_scalar = constant("hidden_bit_scalar", 0x0080, short_constant)
    hidden_bit = operation("hidden_bit", broadcast, "dst", src=hidden_bit_scalar)
    lhs_significand_high = operation(
        "lhs_significand_high", minimum, "d", s1=lhs_exponent, s2=hidden_bit
    )
    rhs_significand_high = operation(
        "rhs_significand_high", minimum, "d", s1=rhs_exponent, s2=hidden_bit
    )
    fraction_mask_scalar = constant("fraction_mask_scalar", 0x007F, short_constant)
    fraction_mask = operation(
        "fraction_mask", broadcast, "dst", src=fraction_mask_scalar
    )
    lhs_fraction = operation("lhs_fraction", bitwise_and, "d", s1=lhs, s2=fraction_mask)
    rhs_fraction = operation("rhs_fraction", bitwise_and, "d", s1=rhs, s2=fraction_mask)
    lhs_significand = operation(
        "lhs_significand",
        bitwise_or,
        "d",
        s1=lhs_fraction,
        s2=lhs_significand_high,
    )
    rhs_significand = operation(
        "rhs_significand",
        bitwise_or,
        "d",
        s1=rhs_fraction,
        s2=rhs_significand_high,
    )

    lhs_effective_exponent = operation(
        "lhs_effective_exponent", maximum, "d", s1=lhs_exponent, s2=hidden_bit
    )
    rhs_effective_exponent = operation(
        "rhs_effective_exponent", maximum, "d", s1=rhs_exponent, s2=hidden_bit
    )
    exponent_sum = operation(
        "exponent_sum",
        add,
        "d",
        s1=lhs_effective_exponent,
        s2=rhs_effective_exponent,
    )
    repair_exponent_limit_scalar = constant("repair_exponent_limit_scalar", 134 << 7)
    repair_exponent_limit = operation(
        "repair_exponent_limit",
        broadcast,
        "dst",
        src=repair_exponent_limit_scalar,
    )
    bounded_exponent_sum = operation(
        "bounded_exponent_sum",
        minimum,
        "d",
        s1=exponent_sum,
        s2=repair_exponent_limit,
    )
    factor_bias_scalar = constant("factor_bias_scalar", 8 << 7)
    factor_bias = operation("factor_bias", broadcast, "dst", src=factor_bias_scalar)
    factor_bits = operation(
        "factor_bits", add, "d", s1=bounded_exponent_sum, s2=factor_bias
    )
    zero_shift = constant("zero_shift", 0, shift_constant)
    # VFLOOR places one i32 result in each pair of 16-bit carrier lanes. The
    # native even-lane shuffle compacts their low halfwords without VPACK's
    # fixed-point rescaling.
    factor_shuffle = constant("factor_shuffle", 2, config_constant)
    factor_units: list[ValueRef] = []
    for unit_index in range(lane_count // 16):
        factor_bits_unit = temporary(f"factor_bits_unit_{unit_index}")
        emits.append(
            EmitRegisterSlice(
                source=factor_bits,
                result=factor_bits_unit,
                unit_offset=unit_index,
                unit_count=1,
            )
        )
        factor_i32 = operation(
            f"factor_i32_{unit_index}",
            floor_bf16,
            "dst",
            src=factor_bits_unit,
            shft=zero_shift,
        )
        factor_carrier = operation(
            f"factor_carrier_{unit_index}",
            shuffle,
            "dst",
            s1=factor_i32,
            s2=factor_i32,
            mod=factor_shuffle,
        )
        if lane_count == 16:
            factor_units.append(factor_carrier)
        else:
            factor_unit = temporary(f"factor_unit_{unit_index}")
            emits.append(
                EmitRegisterSlice(
                    source=factor_carrier,
                    result=factor_unit,
                    unit_count=1,
                )
            )
            factor_units.append(factor_unit)
    if lane_count == 16:
        factor = factor_units[0]
    else:
        factor = temporary("factor")
        emits.append(
            EmitRegisterConcat(
                sources=tuple(factor_units),
                result=factor,
                result_type=_I16X32_VECTOR,
            )
        )

    integer_control = constant(
        "integer_control",
        _U16_ELEMENTWISE_MULTIPLY_CONTROL,
        config_constant,
    )
    significand_products = operation(
        "significand_products",
        integer_multiply,
        "dst",
        s1=lhs_significand,
        s2=rhs_significand,
        acc=integer_control,
    )
    emits.append(
        EmitDescriptorOp(
            descriptor=set_srs_mode,
            immediates={"i": 1},
            form=DescriptorEmitForm.OP,
        )
    )
    products = operation(
        "products",
        integer_narrow,
        "dst",
        src=significand_products,
        su=zero_shift,
    )
    scaled_products = operation(
        "scaled_products",
        integer_multiply,
        "dst",
        s1=products,
        s2=factor,
        acc=integer_control,
    )
    repair_shift = constant("repair_shift", 16, shift_constant)
    repair_magnitude = operation(
        "repair_magnitude",
        integer_narrow,
        "dst",
        src=scaled_products,
        su=repair_shift,
    )

    sign_mask_scalar = constant("sign_mask_scalar", 0x8000)
    sign_mask = operation("sign_mask", broadcast, "dst", src=sign_mask_scalar)
    lhs_sign = operation("lhs_sign", bitwise_and, "d", s1=lhs, s2=sign_mask)
    rhs_sign = operation("rhs_sign", bitwise_and, "d", s1=rhs, s2=sign_mask)
    sign = operation("sign", add, "d", s1=lhs_sign, s2=rhs_sign)
    repair = operation("repair", bitwise_or, "d", s1=repair_magnitude, s2=sign)
    absolute_mask_scalar = constant("absolute_mask_scalar", 0x7FFF)
    absolute_mask = operation(
        "absolute_mask", broadcast, "dst", src=absolute_mask_scalar
    )
    direct_magnitude = operation(
        "direct_magnitude", bitwise_and, "d", s1=direct, s2=absolute_mask
    )
    repair_low = operation("repair_low", compare_zero, "cmp", s2=direct_magnitude)
    repair_lanes = temporary("repair_lanes")
    emits.append(
        EmitDescriptorOp(
            descriptor=complete_predicate,
            operands={"storage": repair_low},
            results={"dst": repair_lanes},
            result_types={"dst": DescriptorResultType()},
            immediates={"i": 0},
            form=DescriptorEmitForm.OP,
        )
    )
    emits.append(
        _op_emit(
            select,
            operands={"s1": direct, "s2": repair, "sel": repair_lanes},
            results={"d": result},
            result_types=None if result_type is None else {"d": result_type},
        )
    )

    return tuple(emits)


def _vector_multiply_bf16_origins_to_bf16_rule(
    lane_count: Literal[16, 32],
) -> DescriptorRule:
    """Multiplies BF16 vector/scalar origins and rounds exactly to BF16."""

    source_f32_type = {16: _F32X16_VECTOR, 32: _F32X32_VECTOR}[lane_count]
    source_bf16_type = {16: _BF16X16_VECTOR, 32: _BF16X32_VECTOR}[lane_count]

    return DescriptorRule(
        source_op=vector.vector_mulf,
        descriptor=_descriptor("amd.xdna.aie2p.multiply.bf16x32.configured"),
        source_nodes=(
            SourceNode.adjacent_unique_user(
                "narrow",
                source_op=vector.vector_fptrunc,
                parent_result=ValueRef.result("result"),
                node_operand=ValueRef.operand("input"),
                guards=(
                    Guard.value_type("input", source_f32_type),
                    Guard.value_type("result", source_bf16_type),
                ),
            ),
        ),
        priority=1,
        guards=(
            Guard.value_type("lhs", source_f32_type),
            Guard.exact_lane_origin_type("lhs", source_bf16_type),
            Guard.value_type("rhs", source_f32_type),
            Guard.exact_uniform_element_origin_type("rhs", _BF16),
            Guard.value_type("result", source_f32_type),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "ninf"),
        ),
        emit=_bf16_vector_scalar_product_emits(
            lane_count,
            lhs=ValueRef.exact_lane_origin_operand("lhs"),
            rhs_scalar=ValueRef.exact_uniform_element_origin_operand("rhs"),
            result=ValueRef.result("result", source_node="narrow"),
        ),
        report_key="exact_bf16_vector_scalar_product_to_bf16",
    )


def _bf16_extremum_guards(type_pattern: TypePattern) -> tuple[Guard, ...]:
    # The fused extrema preserve their first operand on unordered inputs and
    # order signed zeros. The source extrema agree with them when NaNs and zero
    # signs are explicitly outside the source contract.
    return (
        *_typed_guards(("lhs", "rhs", "result"), type_pattern),
        Guard.instance_flags_has_all("fastmath", "nnan"),
        Guard.instance_flags_has_all("fastmath", "nsz"),
    )


def emit_bf16_extremum(
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    operation: Literal["minimum", "maximum"],
    *,
    temporary_prefix: str = "",
) -> tuple[ContractEmit, ...]:
    """Emits a fused BF16 extremum over one native vector carrier."""

    descriptor = _descriptor(
        f"amd.xdna.aie2p.{'min.ge' if operation == 'minimum' else 'max.lt'}"
        ".bf16x32.native"
    )
    return (
        _op_emit(
            descriptor,
            operands={"s1": lhs, "s2": rhs},
            results={
                "d": result,
                "cmp": ValueRef.temporary(f"{temporary_prefix}comparison"),
            },
            result_types={
                "d": DescriptorResultType(),
                "cmp": DescriptorResultType(),
            },
        ),
    )


def _vector_extremum_bf16_rule(
    source_op: Op, operation: Literal["minimum", "maximum"]
) -> DescriptorRule:
    descriptor = _descriptor(
        f"amd.xdna.aie2p.{'min.ge' if operation == 'minimum' else 'max.lt'}"
        ".bf16x32.native"
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_bf16_extremum_guards(_BF16_VECTOR),
        emit=emit_bf16_extremum(
            ValueRef.operand("lhs"),
            ValueRef.operand("rhs"),
            ValueRef.result("result"),
            operation,
        ),
    )


def _scalar_extremum_bf16_rule(
    source_op: Op, operation: Literal["minimum", "maximum"]
) -> DescriptorRule:
    broadcast = _descriptor("amd.xdna.aie2p.splat.i16x32")
    descriptor = _descriptor(
        f"amd.xdna.aie2p.{'min.ge' if operation == 'minimum' else 'max.lt'}"
        ".bf16x32.native"
    )
    extract = _descriptor("amd.xdna.aie2p.extract.i16.immediate")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_bf16_extremum_guards(_BF16),
        emit=(
            *(
                _op_emit(
                    broadcast,
                    operands={"src": ValueRef.operand(operand)},
                    results={"dst": ValueRef.temporary(f"{operand}_vector")},
                    result_types={"dst": DescriptorResultType()},
                )
                for operand in ("lhs", "rhs")
            ),
            *emit_bf16_extremum(
                ValueRef.temporary("lhs_vector"),
                ValueRef.temporary("rhs_vector"),
                ValueRef.temporary("extremum_vector"),
                operation,
            ),
            EmitDescriptorOp(
                descriptor=extract,
                operands={"s1": ValueRef.temporary("extremum_vector")},
                results={"dst": ValueRef.result("result")},
                immediates={"idx": 0},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _vector_dot2f_bf16_rule(
    input_type: TypePattern,
    result_type: TypePattern,
    *,
    initial_accumulator: Literal["source", "zero"],
    report_key: str,
    rhs_form: Literal["packed", "interleaved"] = "packed",
) -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    clear = _descriptor("amd.xdna.aie2p.accumulator.clear.f32x64")
    move_to_accumulator = _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512")
    accumulate = _descriptor("amd.xdna.aie2p.accumulate.bf16x32.configured")
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )

    emits: list[ContractEmit] = []
    input_values = {
        operand_name: ValueRef.operand(operand_name) for operand_name in ("lhs", "rhs")
    }
    for lane_group, control in zip(
        ("even", "odd"), _BF16_DOT2_DEINTERLEAVE_CONTROLS, strict=True
    ):
        control_value = ValueRef.temporary(f"{lane_group}_control")
        emits.append(_constant_emit(config_constant, control_value, control))
        emits.extend(
            _op_emit(
                shuffle,
                operands={
                    "s1": input_values[operand_name],
                    "s2": input_values[operand_name],
                    "mod": control_value,
                },
                results={"dst": ValueRef.temporary(f"{operand_name}_{lane_group}")},
                result_types={"dst": DescriptorResultType()},
            )
            for operand_name in (
                ("lhs",) if rhs_form == "interleaved" else ("lhs", "rhs")
            )
        )

    # An interleave supplies the low sixteen even/odd lanes directly. Only
    # those accumulator lanes belong to the source result; carrier padding
    # does not participate in the returned dot product.
    rhs_values = {
        lane_group: (
            ValueRef.operand(lane_group, source_node="interleave")
            if rhs_form == "interleaved"
            else ValueRef.temporary(f"rhs_{lane_group}")
        )
        for lane_group in ("even", "odd")
    }
    emits.append(
        _op_emit(
            clear,
            results={"dst": ValueRef.temporary("zero_accumulator")},
            result_types={"dst": DescriptorResultType()},
        )
    )
    accumulator = ValueRef.temporary("zero_accumulator")
    guards = (
        Guard.value_type("lhs", input_type),
        Guard.value_type("rhs", input_type),
        Guard.value_type("acc", result_type),
        Guard.value_type("result", result_type),
    )
    if initial_accumulator == "zero":
        # Exact floating facts distinguish positive zero from negative zero.
        # The cleared accumulator already supplies every source and padding lane.
        guards += (Guard.value_float_equals("acc", 0.0),)
    else:
        accumulator = ValueRef.temporary("initial_accumulator")
        emits.extend(
            (
                *(
                    EmitRegisterSlice(
                        source=ValueRef.temporary("zero_accumulator"),
                        result=ValueRef.temporary(f"zero_accumulator_unit_{unit}"),
                        unit_offset=unit,
                        unit_count=1,
                    )
                    for unit in range(1, 4)
                ),
                _op_emit(
                    move_to_accumulator,
                    operands={"src": ValueRef.operand("acc")},
                    results={"dst": ValueRef.temporary("initial_accumulator_unit")},
                    result_types={"dst": DescriptorResultType()},
                ),
                EmitRegisterConcat(
                    sources=(
                        ValueRef.temporary("initial_accumulator_unit"),
                        ValueRef.temporary("zero_accumulator_unit_1"),
                        ValueRef.temporary("zero_accumulator_unit_2"),
                        ValueRef.temporary("zero_accumulator_unit_3"),
                    ),
                    result=ValueRef.temporary("initial_accumulator"),
                    result_type=_F32X64_ACCUMULATOR,
                ),
            )
        )
    emits.extend(
        (
            _constant_emit(
                config_constant,
                ValueRef.temporary("accumulate_control"),
                _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
            ),
            _op_emit(
                accumulate,
                operands={
                    "acc1": accumulator,
                    "s1": ValueRef.temporary("lhs_even"),
                    "s2": rhs_values["even"],
                    "acc": ValueRef.temporary("accumulate_control"),
                },
                results={"dst": ValueRef.temporary("even_accumulator")},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                accumulate,
                operands={
                    "acc1": ValueRef.temporary("even_accumulator"),
                    "s1": ValueRef.temporary("lhs_odd"),
                    "s2": rhs_values["odd"],
                    "acc": ValueRef.temporary("accumulate_control"),
                },
                results={"dst": ValueRef.temporary("result_accumulator")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterSlice(
                source=ValueRef.temporary("result_accumulator"),
                result=ValueRef.temporary("result_accumulator_unit"),
                unit_count=1,
            ),
            _op_emit(
                move_from_accumulator,
                operands={"src": ValueRef.temporary("result_accumulator_unit")},
                results={"dst": ValueRef.result("result")},
            ),
        )
    )
    return DescriptorRule(
        source_op=vector.vector_dot2f,
        descriptor=accumulate,
        source_nodes=(
            (
                SourceNode.adjacent_definition(
                    "interleave",
                    source_op=vector.vector_interleave,
                    parent_operand=ValueRef.operand("rhs"),
                    node_result=ValueRef.result("result"),
                    guards=(
                        Guard.value_type("even", Vector("bf16", lanes=16)),
                        Guard.value_type("odd", Vector("bf16", lanes=16)),
                        Guard.i64_range("axis", 0, 0),
                    ),
                ),
            )
            if rhs_form == "interleaved"
            else ()
        ),
        priority=1 if rhs_form == "interleaved" else 0,
        guards=guards,
        emit=tuple(emits),
        report_key=report_key,
    )


def _matrix_multiply_bf16bf16_m8n8k1_rule() -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    broadcast = _descriptor("amd.xdna.aie2p.broadcast.bf16x8.to.bf16x32")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    move = _descriptor("amd.xdna.aie2p.move.vector512")
    multiply = _descriptor(
        "amd.xdna.aie2p.matrix.accumulate.bf16bf16.m8n8k1.configured"
    )
    return DescriptorRule(
        source_op=vector.vector_mma,
        descriptor=multiply,
        guards=(
            Guard.value_type("lhs", _BF16X8_VECTOR),
            Guard.value_type("rhs", _BF16X8_VECTOR),
            Guard.value_type("init", _F32X64_ACCUMULATOR),
            Guard.value_type("result", _F32X64_ACCUMULATOR),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=broadcast,
                operands={"s1": ValueRef.operand("lhs")},
                results={"dst": ValueRef.temporary("lhs_broadcast")},
                result_types={"dst": DescriptorResultType()},
                immediates={"idx": 0},
                form=DescriptorEmitForm.OP,
            ),
            _constant_emit(
                config_constant,
                ValueRef.temporary("lhs_shuffle_even_control"),
                _BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS[0],
            ),
            _op_emit(
                shuffle,
                operands={
                    "s1": ValueRef.temporary("lhs_broadcast"),
                    "s2": ValueRef.temporary("lhs_broadcast"),
                    "mod": ValueRef.temporary("lhs_shuffle_even_control"),
                },
                results={"dst": ValueRef.temporary("lhs_rows_even")},
                result_types={"dst": DescriptorResultType()},
            ),
            _constant_emit(
                config_constant,
                ValueRef.temporary("lhs_shuffle_odd_control"),
                _BF16_OUTER_PRODUCT_SHUFFLE_CONTROLS[1],
            ),
            _op_emit(
                shuffle,
                operands={
                    "s1": ValueRef.temporary("lhs_broadcast"),
                    "s2": ValueRef.temporary("lhs_broadcast"),
                    "mod": ValueRef.temporary("lhs_shuffle_odd_control"),
                },
                results={"dst": ValueRef.temporary("lhs_rows_odd")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.temporary("lhs_rows_even"),
                    ValueRef.temporary("lhs_rows_odd"),
                ),
                result=ValueRef.temporary("lhs_rows"),
                result_type=_BF16X64_VECTOR,
            ),
            EmitDescriptorOp(
                descriptor=broadcast,
                operands={"s1": ValueRef.operand("rhs")},
                results={"dst": ValueRef.temporary("rhs_columns_low")},
                result_types={"dst": DescriptorResultType()},
                immediates={"idx": 0},
                form=DescriptorEmitForm.OP,
            ),
            _op_emit(
                move,
                operands={"src": ValueRef.temporary("rhs_columns_low")},
                results={"dst": ValueRef.temporary("rhs_columns_high")},
                result_types={"dst": DescriptorResultType()},
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.temporary("rhs_columns_low"),
                    ValueRef.temporary("rhs_columns_high"),
                ),
                result=ValueRef.temporary("rhs_columns"),
                result_type=_BF16X64_VECTOR,
            ),
            _constant_emit(
                config_constant,
                ValueRef.temporary("multiply_control"),
                _BF16_OUTER_PRODUCT_MULTIPLY_CONTROL,
            ),
            _op_emit(
                multiply,
                operands={
                    "acc1": ValueRef.operand("init"),
                    "s1": ValueRef.temporary("lhs_rows"),
                    "s2": ValueRef.temporary("rhs_columns"),
                    "acc": ValueRef.temporary("multiply_control"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
        report_key="bf16bf16_m8n8k1",
    )


def emit_f32x16_extremum(
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    operation: Literal["minimum", "maximum"],
    *,
    temporary_prefix: str,
    zero_vector: ValueRef | None = None,
) -> tuple[ContractEmit, ...]:
    """Selects packed F32 extrema through the signed integer data path."""

    signed_maximum = _descriptor("amd.xdna.aie2p.max.lt.signed.i32x16.native")
    signed_minimum = _descriptor("amd.xdna.aie2p.min.ge.signed.i32x16.native")
    sign_compare = _descriptor("amd.xdna.aie2p.cmp.lt.signed.i32x16.native")
    select = _descriptor("amd.xdna.aie2p.select.i32x16")

    def temporary(name: str) -> ValueRef:
        return ValueRef.temporary(f"{temporary_prefix}{name}")

    emits: list[ContractEmit] = []
    if zero_vector is None:
        zero = temporary("zero")
        zero_vector = temporary("zero_vector")
        emits.extend(
            (
                _constant_emit(
                    _descriptor("amd.xdna.aie2p.constant.i32.short"), zero, 0
                ),
                _op_emit(
                    _descriptor("amd.xdna.aie2p.splat.i32x16"),
                    operands={"src": zero},
                    results={"dst": zero_vector},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )

    # Signed integer maximum has the right ordering unless both floats are
    # negative; signed integer minimum has the right ordering in that remaining
    # quadrant. nnan excludes unordered encodings and nsz makes the two zero
    # encodings interchangeable, so the sign of signed_maximum identifies the
    # quadrant without changing the source contract.
    maximum = temporary("signed_maximum")
    minimum = temporary("signed_minimum")
    both_negative = temporary("both_negative")
    emits.extend(
        (
            _op_emit(
                signed_maximum,
                operands={"s1": lhs, "s2": rhs},
                results={
                    "d": maximum,
                    "cmp": temporary("maximum_comparison"),
                },
                result_types={
                    "d": DescriptorResultType(),
                    "cmp": DescriptorResultType(),
                },
            ),
            _op_emit(
                signed_minimum,
                operands={"s1": lhs, "s2": rhs},
                results={
                    "d": minimum,
                    "cmp": temporary("minimum_comparison"),
                },
                result_types={
                    "d": DescriptorResultType(),
                    "cmp": DescriptorResultType(),
                },
            ),
            _op_emit(
                sign_compare,
                operands={"s1": maximum, "s2": zero_vector},
                results={"cmp": both_negative},
                result_types={"cmp": DescriptorResultType()},
            ),
            _op_emit(
                select,
                operands={
                    "s1": maximum if operation == "maximum" else minimum,
                    "s2": minimum if operation == "maximum" else maximum,
                    "sel": both_negative,
                },
                results={"d": result},
                result_types={"d": DescriptorResultType()},
            ),
        )
    )
    return tuple(emits)


def _vector_extremum_f32x16_rule(
    source_op: Op, operation: Literal["minimum", "maximum"]
) -> DescriptorRule:
    select = _descriptor("amd.xdna.aie2p.select.i32x16")
    return DescriptorRule(
        source_op=source_op,
        descriptor=select,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), _F32X16_VECTOR),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        ),
        emit=emit_f32x16_extremum(
            ValueRef.operand("lhs"),
            ValueRef.operand("rhs"),
            ValueRef.result("result"),
            operation,
            temporary_prefix="extremum_",
        ),
        report_key=f"f32x16_packed_{operation}",
    )


def _float_matrix_accumulator_zero_rule() -> DescriptorRule:
    descriptor = _descriptor("amd.xdna.aie2p.accumulator.clear.f32x64")
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64"),
            Guard.value_type("result", _F32X64_ACCUMULATOR),
            Guard.value_float_equals("result", 0.0),
        ),
        emit=(
            _op_emit(
                descriptor,
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _float_matrix_accumulator_add_rule() -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    add = _descriptor("amd.xdna.aie2p.add.f32x64.configured")
    return DescriptorRule(
        source_op=vector.vector_addf,
        descriptor=add,
        guards=_typed_guards(("lhs", "rhs", "result"), _F32X64_ACCUMULATOR),
        emit=(
            _constant_emit(
                config_constant,
                ValueRef.temporary("add_control"),
                F32_ACCUMULATOR_ADD_CONTROL,
            ),
            _op_emit(
                add,
                operands={
                    "acc1": ValueRef.operand("lhs"),
                    "acc2": ValueRef.operand("rhs"),
                    "acc": ValueRef.temporary("add_control"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _float_vector_accumulator_binary_rule(
    source_op: Op,
    operation_descriptor_key: str,
) -> DescriptorRule:
    operation = _descriptor(operation_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=operation,
        guards=_typed_guards(("lhs", "rhs", "result"), _F32_VECTOR),
        emit=emit_f32_vector_accumulator_binary(
            ValueRef.operand("lhs"),
            ValueRef.operand("rhs"),
            ValueRef.result("result"),
            operation_descriptor_key,
            extract_scalar_result=False,
        ),
    )


def _float_scalar_accumulator_binary_rule(
    source_op: Op,
    operation_descriptor_key: str,
) -> DescriptorRule:
    operation = _descriptor(operation_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=operation,
        guards=_typed_guards(("lhs", "rhs", "result"), _F32),
        emit=emit_f32_scalar_accumulator_binary(
            ValueRef.operand("lhs"),
            ValueRef.operand("rhs"),
            ValueRef.result("result"),
            operation_descriptor_key,
        ),
    )


AIE2P_BF16_MATRIX_RULES = (_matrix_multiply_bf16bf16_m8n8k1_rule(),)

AIE2P_FLOATING_RULES = (
    _scalar_multiply_f16_rule(),
    _vector_multiply_bf16_origin_scale_rule(),
    _vector_multiply_bf16_origins_to_bf16_rule(16),
    _vector_multiply_bf16_origins_to_bf16_rule(32),
    *(
        _vector_extremum_f32x16_rule(source_op, operation)
        for source_op, operation in (
            (vector.vector_minnumf, "minimum"),
            (vector.vector_minimumf, "minimum"),
            (vector.vector_maxnumf, "maximum"),
            (vector.vector_maximumf, "maximum"),
        )
    ),
    *(
        _vector_extremum_bf16_rule(source_op, operation)
        for source_op, operation in (
            (vector.vector_minnumf, "minimum"),
            (vector.vector_minimumf, "minimum"),
            (vector.vector_maxnumf, "maximum"),
            (vector.vector_maximumf, "maximum"),
        )
    ),
    *(
        _scalar_extremum_bf16_rule(source_op, operation)
        for source_op, operation in (
            (scalar_arithmetic.scalar_minnumf, "minimum"),
            (scalar_arithmetic.scalar_minimumf, "minimum"),
            (scalar_arithmetic.scalar_maxnumf, "maximum"),
            (scalar_arithmetic.scalar_maximumf, "maximum"),
        )
    ),
    _float_matrix_accumulator_zero_rule(),
    _float_matrix_accumulator_add_rule(),
    *(
        _float_vector_accumulator_binary_rule(source_op, descriptor_key)
        for source_op, descriptor_key in (
            (vector.vector_addf, "amd.xdna.aie2p.add.f32x16.configured"),
            (vector.vector_subf, "amd.xdna.aie2p.sub.f32x16.configured"),
        )
    ),
    *(
        _float_scalar_accumulator_binary_rule(source_op, descriptor_key)
        for source_op, descriptor_key in (
            (
                scalar_arithmetic.scalar_addf,
                "amd.xdna.aie2p.add.f32x16.configured",
            ),
            (
                scalar_arithmetic.scalar_subf,
                "amd.xdna.aie2p.sub.f32x16.configured",
            ),
        )
    ),
    _vector_dot2f_bf16_rule(
        _BF16X32_VECTOR,
        _F32X16_VECTOR,
        initial_accumulator="zero",
        rhs_form="interleaved",
        report_key="bf16_dot2_interleaved_rhs_zero",
    ),
    _vector_dot2f_bf16_rule(
        _BF16X32_VECTOR,
        _F32X16_VECTOR,
        initial_accumulator="source",
        rhs_form="interleaved",
        report_key="bf16_dot2_interleaved_rhs",
    ),
    _vector_dot2f_bf16_rule(
        _BF16_DOT2_VECTOR,
        _F32_VECTOR,
        initial_accumulator="zero",
        report_key="bf16_dot2_zero",
    ),
    _vector_dot2f_bf16_rule(
        _BF16_DOT2_VECTOR,
        _F32_VECTOR,
        initial_accumulator="source",
        report_key="bf16_dot2",
    ),
)
