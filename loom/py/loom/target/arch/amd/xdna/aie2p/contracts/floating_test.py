# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P native floating-point contracts."""

import struct

from loom.dialect.scalar import arithmetic as scalar
from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.floating import AIE2P_FLOATING_RULES
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
    EmitRegisterSlice,
    Guard,
    Scalar,
    SourceNodeRelation,
    ValueRef,
    Vector,
)


def _bf16_origin_scale_rule():
    return next(
        rule
        for rule in AIE2P_FLOATING_RULES
        if rule.source_op is vector.vector_mulf
        and rule.report_key == "exact_bf16_origin_scale_1_4453125"
    )


def _bf16_vector_scalar_product_rule(lane_count: int = 16):
    return next(
        rule
        for rule in AIE2P_FLOATING_RULES
        if rule.source_op is vector.vector_mulf
        and rule.report_key == "exact_bf16_vector_scalar_product_to_bf16"
        and rule.guards[0].type_pattern == Vector("f32", lanes=lane_count)
    )


def _f32_extremum_rule(source_op):
    return next(
        rule
        for rule in AIE2P_FLOATING_RULES
        if rule.source_op is source_op
        and rule.report_key in ("f32x16_packed_minimum", "f32x16_packed_maximum")
    )


def _f32_bits(value: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return (0x80000000 if value < 0 else 0) | 0x7F800000


def _bf16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def test_bf16_origin_scale_uses_native_multiply_with_packed_underflow_repair() -> None:
    rule = _bf16_origin_scale_rule()
    assert rule.guards == (
        Guard.value_type("lhs", Vector("f32", lanes=16)),
        Guard.exact_lane_origin_type("lhs", Vector("bf16", lanes=16)),
        Guard.value_type("rhs", Vector("f32", lanes=16)),
        Guard.value_type("result", Vector("f32", lanes=16)),
        Guard.value_float_equals("rhs", 1.4453125),
        Guard.instance_flags_has_all("fastmath", "nnan"),
    )
    assert [
        emit.descriptor.key if isinstance(emit, EmitDescriptorOp) else "slice"
        for emit in rule.emit
    ] == [
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.multiply.bf16x32.configured",
        "slice",
        "amd.xdna.aie2p.move.accumulator512.to.vector512",
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.and.bits512",
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.multiply.i16x32.configured",
        "amd.xdna.aie2p.constant.i32.shift",
        "amd.xdna.aie2p.constant.i32.shift",
        "amd.xdna.aie2p.state.rounding.immediate",
        "amd.xdna.aie2p.state.srs-mode.immediate",
        "amd.xdna.aie2p.state.saturation.immediate",
        "amd.xdna.aie2p.narrow.trunc.signed.i16x32",
        "amd.xdna.aie2p.narrow.trunc.signed.i16x32",
        "amd.xdna.aie2p.add.i16x32",
        "amd.xdna.aie2p.constant.i32",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.and.bits512",
        "amd.xdna.aie2p.or.bits512",
        "amd.xdna.aie2p.constant.i32.mova",
        "amd.xdna.aie2p.shuffle.x.configured",
        "amd.xdna.aie2p.constant.i32.short",
        "amd.xdna.aie2p.splat.i16x32",
        "amd.xdna.aie2p.cmp.lt.unsigned.i16x32.el.low32",
        "amd.xdna.aie2p.select.i32x16.mask64",
    ]
    assert rule.emit[0].immediates == {"i": 0x3FB9}
    assert rule.emit[2].immediates == {"i": 60}
    multiply = rule.emit[3]
    assert isinstance(multiply, EmitDescriptorOp)
    assert multiply.operands["s1"] == ValueRef.exact_lane_origin_operand("lhs")
    assert isinstance(rule.emit[4], EmitRegisterSlice)
    assert rule.emit[9].immediates == {"i": 185 << 8}
    assert rule.emit[11].immediates == {"i": 90}
    assert rule.emit[13].immediates == {"i": 0}
    assert rule.emit[14].immediates == {"i": 15}
    assert rule.emit[15].immediates == {"i": 0}
    assert rule.emit[16].immediates == {"i": 1}
    assert rule.emit[17].immediates == {"i": 0}
    assert rule.emit[25].immediates == {"i": 18}
    assert rule.emit[27].immediates == {"i": 89}


def test_bf16_origin_scale_repair_matches_every_non_nan_encoding() -> None:
    rule = _bf16_origin_scale_rule()
    scale_bits = rule.emit[0].immediates["i"]
    repair_factor = rule.emit[9].immediates["i"]
    repair_limit = rule.emit[27].immediates["i"]
    scale = _bf16_value(scale_bits)
    assert scale == 1.4453125

    checked_count = 0
    for value_bits in range(1 << 16):
        exponent = value_bits & 0x7F80
        fraction = value_bits & 0x007F
        if exponent == 0x7F80 and fraction:
            continue

        expected = _f32_bits(_bf16_value(value_bits) * scale)
        expected_magnitude = expected & 0x7FFFFFFF
        direct = expected
        if 0 < expected_magnitude < 0x00800000:
            direct = expected & 0x80000000

        magnitude = value_bits & 0x7FFF
        wide_repair = magnitude * repair_factor
        low_half = ((wide_repair & 0xFFFF) * 2) & 0xFFFF
        high_half = ((wide_repair >> 15) | (value_bits & 0x8000)) & 0xFFFF
        repair = (high_half << 16) | low_half
        actual = repair if magnitude < repair_limit else direct
        assert actual == expected, f"BF16 input 0x{value_bits:04x}"
        checked_count += 1

    assert checked_count == 65282


def test_bf16_vector_scalar_product_owns_the_exact_roundtrip_graph() -> None:
    for lane_count in (16, 32):
        f32_vector = Vector("f32", lanes=lane_count)
        bf16_vector = Vector("bf16", lanes=lane_count)
        rule = _bf16_vector_scalar_product_rule(lane_count)
        assert rule.priority == 1
        assert rule.guards == (
            Guard.value_type("lhs", f32_vector),
            Guard.exact_lane_origin_type("lhs", bf16_vector),
            Guard.value_type("rhs", f32_vector),
            Guard.exact_uniform_element_origin_type("rhs", Scalar("bf16")),
            Guard.value_type("result", f32_vector),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "ninf"),
        )

        (narrow,) = rule.source_nodes
        assert (
            narrow.name,
            narrow.source_op,
            narrow.relation,
            narrow.parent_value,
            narrow.node_value,
            narrow.parent,
            narrow.guards,
        ) == (
            "narrow",
            vector.vector_fptrunc,
            SourceNodeRelation.ADJACENT_UNIQUE_USER,
            ValueRef.result("result"),
            ValueRef.operand("input"),
            "",
            (
                Guard.value_type("input", f32_vector),
                Guard.value_type("result", bf16_vector),
            ),
        )
        descriptor_emits = [
            emit for emit in rule.emit if isinstance(emit, EmitDescriptorOp)
        ]
        temporary_producers = {
            result.field: emit
            for emit in descriptor_emits
            for result in emit.results.values()
            if result == ValueRef.temporary(result.field)
        }
        assert temporary_producers["rhs"].operands == {
            "src": ValueRef.exact_uniform_element_origin_operand("rhs")
        }
        assert temporary_producers["raw_products"].operands["s1"] == (
            ValueRef.exact_lane_origin_operand("lhs")
        )
        final_select = descriptor_emits[-1]
        assert final_select.descriptor.key == "amd.xdna.aie2p.select.i16x32.mask64"
        assert final_select.results == {
            "d": ValueRef.result("result", source_node="narrow")
        }


def test_bf16_vector_scalar_product_exhausts_subnormal_rounding() -> None:
    rule = _bf16_vector_scalar_product_rule()
    constants = {
        result.field: emit.immediates["i"]
        for emit in rule.emit
        if isinstance(emit, EmitDescriptorOp) and emit.form is DescriptorEmitForm.CONST
        for result in emit.results.values()
    }
    exponent_limit = constants["repair_exponent_limit_scalar"] >> 7
    factor_exponent_bias = constants["factor_bias_scalar"] >> 7
    repair_shift = constants["repair_shift"]
    assert (exponent_limit, factor_exponent_bias, repair_shift) == (134, 8, 16)

    def round_even_shift(value: int, shift: int) -> int:
        quotient = value >> shift
        remainder = value & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        return quotient + (
            remainder > halfway or (remainder == halfway and quotient & 1)
        )

    checked_count = 0
    tie_count = 0
    for product in range(255 * 255 + 1):
        for exponent_sum in range(119, exponent_limit + 1):
            reference_shift = 135 - exponent_sum
            expected = round_even_shift(product, reference_shift)
            factor = 1 << (
                min(exponent_sum, exponent_limit) + factor_exponent_bias - 127
            )
            actual = round_even_shift(product * factor, repair_shift)
            assert actual == expected, (product, exponent_sum)
            remainder = product & ((1 << reference_shift) - 1)
            if remainder == 1 << (reference_shift - 1):
                tie_count += 1
            checked_count += 1

    # Smaller exponent sums encode the factor as a positive BF16 fraction.
    # Flooring that factor produces zero, which is also the correctly rounded
    # result even for the largest possible significand product.
    for exponent_sum in range(2, 119):
        reference_shift = 135 - exponent_sum
        assert round_even_shift(255 * 255, reference_shift) == 0
        checked_count += 1

    assert checked_count == 1_040_533
    assert tie_count == 65_025


def test_bf16_vector_extrema_cover_every_native_lane_width() -> None:
    for source_op, descriptor_key in (
        (vector.vector_minnumf, "amd.xdna.aie2p.min.ge.bf16x32.native"),
        (vector.vector_minimumf, "amd.xdna.aie2p.min.ge.bf16x32.native"),
        (vector.vector_maxnumf, "amd.xdna.aie2p.max.lt.bf16x32.native"),
        (vector.vector_maximumf, "amd.xdna.aie2p.max.lt.bf16x32.native"),
    ):
        rules = [
            rule
            for rule in AIE2P_FLOATING_RULES
            if rule.source_op is source_op and rule.descriptor.key == descriptor_key
        ]
        (rule,) = rules
        native_bf16 = Vector(
            "bf16", minimum_static_elements=1, maximum_static_elements=32
        )
        assert rule.guards == (
            Guard.value_type("lhs", native_bf16),
            Guard.value_type("rhs", native_bf16),
            Guard.value_type("result", native_bf16),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        )
        assert len(rule.emit) == 1
        extremum = rule.emit[0]
        assert extremum.descriptor.key == descriptor_key
        assert extremum.operands == {
            "s1": ValueRef.operand("lhs"),
            "s2": ValueRef.operand("rhs"),
        }
        assert extremum.results == {
            "d": ValueRef.result("result"),
            "cmp": ValueRef.temporary("comparison"),
        }
        # The unused mask remains a real fixed-register descriptor result.
        assert extremum.result_types == {
            "d": DescriptorResultType(),
            "cmp": DescriptorResultType(),
        }


def test_f32_vector_extrema_use_packed_signed_quadrants() -> None:
    cases = (
        (vector.vector_minnumf, "minimum"),
        (vector.vector_minimumf, "minimum"),
        (vector.vector_maxnumf, "maximum"),
        (vector.vector_maximumf, "maximum"),
    )
    for source_op, operation in cases:
        rule = _f32_extremum_rule(source_op)
        assert rule.guards == (
            Guard.value_type("lhs", Vector("f32", lanes=16)),
            Guard.value_type("rhs", Vector("f32", lanes=16)),
            Guard.value_type("result", Vector("f32", lanes=16)),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        )
        assert [emit.descriptor.key for emit in rule.emit] == [
            "amd.xdna.aie2p.constant.i32.short",
            "amd.xdna.aie2p.splat.i32x16",
            "amd.xdna.aie2p.max.lt.signed.i32x16.native",
            "amd.xdna.aie2p.min.ge.signed.i32x16.native",
            "amd.xdna.aie2p.cmp.lt.signed.i32x16.native",
            "amd.xdna.aie2p.select.i32x16",
        ]
        select = rule.emit[-1]
        assert select.operands == {
            "s1": ValueRef.temporary(
                f"extremum_signed_{'max' if operation == 'maximum' else 'min'}imum"
            ),
            "s2": ValueRef.temporary(
                f"extremum_signed_{'min' if operation == 'maximum' else 'max'}imum"
            ),
            "sel": ValueRef.temporary("extremum_both_negative"),
        }
        assert select.results == {"d": ValueRef.result("result")}


def test_f32_vector_extrema_signed_quadrants_match_numeric_order() -> None:
    values = (
        0xFF800000,  # -inf
        0xC0000000,  # -2
        0xBF800000,  # -1
        0x80800000,  # minimum negative normal
        0x80000001,  # minimum negative subnormal
        0x80000000,  # -0
        0x00000000,  # +0
        0x00000001,  # minimum positive subnormal
        0x00800000,  # minimum positive normal
        0x3F800000,  # +1
        0x40000000,  # +2
        0x7F800000,  # +inf
    )

    def signed(bits: int) -> int:
        return bits if bits < 0x80000000 else bits - 0x100000000

    def selected(lhs: int, rhs: int, maximum: bool) -> int:
        signed_maximum = max((lhs, rhs), key=signed)
        signed_minimum = min((lhs, rhs), key=signed)
        both_negative = signed(signed_maximum) < 0
        if maximum:
            return signed_minimum if both_negative else signed_maximum
        return signed_maximum if both_negative else signed_minimum

    for lhs in values:
        for rhs in values:
            lhs_value = struct.unpack("<f", struct.pack("<I", lhs))[0]
            rhs_value = struct.unpack("<f", struct.pack("<I", rhs))[0]
            for maximum in (False, True):
                actual = selected(lhs, rhs, maximum)
                actual_value = struct.unpack("<f", struct.pack("<I", actual))[0]
                expected = (max if maximum else min)(lhs_value, rhs_value)
                assert actual_value == expected


def test_bf16_scalar_extrema_broadcast_and_extract_raw_halfwords() -> None:
    for source_op, descriptor_key in (
        (scalar.scalar_minnumf, "amd.xdna.aie2p.min.ge.bf16x32.native"),
        (scalar.scalar_minimumf, "amd.xdna.aie2p.min.ge.bf16x32.native"),
        (scalar.scalar_maxnumf, "amd.xdna.aie2p.max.lt.bf16x32.native"),
        (scalar.scalar_maximumf, "amd.xdna.aie2p.max.lt.bf16x32.native"),
    ):
        rule = next(
            rule for rule in AIE2P_FLOATING_RULES if rule.source_op is source_op
        )
        assert rule.guards == (
            Guard.value_type("lhs", Scalar("bf16")),
            Guard.value_type("rhs", Scalar("bf16")),
            Guard.value_type("result", Scalar("bf16")),
            Guard.instance_flags_has_all("fastmath", "nnan"),
            Guard.instance_flags_has_all("fastmath", "nsz"),
        )
        assert [emit.descriptor.key for emit in rule.emit] == [
            "amd.xdna.aie2p.splat.i16x32",
            "amd.xdna.aie2p.splat.i16x32",
            descriptor_key,
            "amd.xdna.aie2p.extract.i16.immediate",
        ]
        assert rule.emit[0].operands == {"src": ValueRef.operand("lhs")}
        assert rule.emit[1].operands == {"src": ValueRef.operand("rhs")}
        assert rule.emit[2].result_types == {
            "d": DescriptorResultType(),
            "cmp": DescriptorResultType(),
        }
        assert rule.emit[3].immediates == {"idx": 0}
        assert rule.emit[3].results == {"dst": ValueRef.result("result")}
