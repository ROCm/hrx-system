# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P approximate nonlinear contracts."""

from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts import nonlinear
from loom.target.contracts import (
    EmitDescriptorOp,
    EmitRegisterConcat,
    GuardKind,
    SourceNodeRelation,
    ValueRef,
)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bf16_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _round_to_bf16(value: float) -> float:
    bits = _f32_bits(value)
    rounded_bits = bits + 0x7FFF + ((bits >> 16) & 1)
    return _bf16_from_bits((rounded_bits >> 16) & 0xFFFF)


def _evaluate_sine_polynomial(turns: float) -> float:
    coefficients = tuple(
        _bf16_from_bits(bits) for bits in nonlinear._SINE_POLYNOMIAL_BF16_BITS
    )
    argument = _round_to_bf16(turns)
    squared = _round_to_bf16(_f32(argument * argument))
    polynomial = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        product = _f32(squared * polynomial)
        polynomial = _round_to_bf16(_f32(product + coefficient))
    return _f32(argument * polynomial)


def _bf16_residual_limbs(value: float) -> tuple[float, float, float]:
    residual = _f32(value)
    limbs = []
    for _ in range(3):
        limb = _round_to_bf16(residual)
        limbs.append(limb)
        residual = _f32(residual - limb)
    return limbs[0], limbs[1], limbs[2]


def _evaluate_radians_to_turns(radians: float) -> float:
    input_limbs = _bf16_residual_limbs(radians)
    coefficient_limbs = tuple(
        _bf16_from_bits(bits) for bits in nonlinear._INVERSE_TWO_PI_BF16_BITS
    )
    products = tuple(
        _f32(input_limbs[input_index] * coefficient_limbs[coefficient_index])
        for input_index, coefficient_index in (
            (2, 0),
            (1, 1),
            (0, 2),
            (1, 0),
            (0, 1),
            (0, 0),
        )
    )
    result = products[0]
    for product in products[1:]:
        result = _f32(result + product)
    return result


def _evaluate_radian_trig(radians: float, *, cosine: bool) -> float:
    turns = _evaluate_radians_to_turns(radians)
    reduced = _f32(turns - _f32(float(round(turns))))
    absolute = abs(reduced)
    if cosine:
        signed_argument = _f32(0.25 - absolute)
        argument = abs(signed_argument)
        sign = math.copysign(1.0, signed_argument)
    else:
        argument = _f32(0.5 - absolute) if absolute > 0.25 else absolute
        sign = math.copysign(1.0, reduced)
    return math.copysign(_evaluate_sine_polynomial(argument), sign)


def test_sine_polynomial_covers_folded_quadrant() -> None:
    maximum_error = max(
        abs(
            _evaluate_sine_polynomial(index / 65536.0)
            - math.sin(2.0 * math.pi * index / 65536.0)
        )
        for index in range(16385)
    )
    assert maximum_error < 0.005
    assert _evaluate_sine_polynomial(0.0) == 0.0
    assert _evaluate_sine_polynomial(0.25) == 1.0


def test_radian_scale_preserves_f32_phase_over_context_domain() -> None:
    coefficient_limbs = tuple(
        _bf16_from_bits(bits) for bits in nonlinear._INVERSE_TWO_PI_BF16_BITS
    )
    reconstructed_coefficient = _f32(
        _f32(coefficient_limbs[0] + coefficient_limbs[1]) + coefficient_limbs[2]
    )
    assert reconstructed_coefficient == _f32(1.0 / (2.0 * math.pi))

    representative_radians = {
        _f32(float(value)) for value in range(-262144, 262145, 127)
    }
    witness_positions = {
        *range(2045, 2052),
        *range(262137, 262144),
        *(-value for value in range(2045, 2052)),
        *(-value for value in range(262137, 262144)),
    }
    representative_radians.update(_f32(float(value)) for value in witness_positions)
    for position in (0, 1, 77, 2048, 262143):
        for dimension in range(64):
            inverse_frequency = _f32(1.0 / (10000.0 ** (2.0 * dimension / 128.0)))
            representative_radians.add(_f32(_f32(float(position)) * inverse_frequency))

    for radians in representative_radians:
        assert (
            abs(_evaluate_radian_trig(radians, cosine=False) - _f32(math.sin(radians)))
            < 0.025
        )
        assert (
            abs(_evaluate_radian_trig(radians, cosine=True) - _f32(math.cos(radians)))
            < 0.025
        )

    for positions in (range(2045, 2052), range(262137, 262144)):
        turns = {_evaluate_radians_to_turns(_f32(float(value))) for value in positions}
        assert len(turns) == len(positions)


def test_radian_scale_emits_ordered_compensated_product() -> None:
    emits, _ = nonlinear._radians_to_turns_emits()
    descriptor_emits = [emit for emit in emits if isinstance(emit, EmitDescriptorOp)]
    descriptor_keys = [emit.descriptor.key for emit in descriptor_emits]
    assert descriptor_keys.count("amd.xdna.aie2p.convert.f32x16.to.bf16x16.low") == 3
    assert descriptor_keys.count("amd.xdna.aie2p.convert.bf16x16.to.f32x16.low") == 2
    assert descriptor_keys.count("amd.xdna.aie2p.sub.f32x16.configured") == 2
    assert descriptor_keys.count("amd.xdna.aie2p.splat.i32x16") == 1
    assert descriptor_keys.count("amd.xdna.aie2p.splat.i16x32") == 3

    product_emits = [
        emit
        for emit in descriptor_emits
        if emit.descriptor.key
        in (
            "amd.xdna.aie2p.multiply.bf16x32.configured",
            "amd.xdna.aie2p.accumulate.bf16x16.configured",
        )
    ]
    assert [emit.descriptor.key for emit in product_emits] == [
        "amd.xdna.aie2p.multiply.bf16x32.configured",
        *["amd.xdna.aie2p.accumulate.bf16x16.configured"] * 5,
    ]
    assert [(emit.operands["s1"], emit.operands["s2"]) for emit in product_emits] == [
        (
            ValueRef.temporary(f"radian_scale_limb_{input_index}_x32"),
            ValueRef.temporary(f"inverse_two_pi_limb_{coefficient_index}"),
        )
        for input_index, coefficient_index in (
            (2, 0),
            (1, 1),
            (0, 2),
            (1, 0),
            (0, 1),
            (0, 0),
        )
    ]
    for index, bits in enumerate(nonlinear._INVERSE_TWO_PI_BF16_BITS):
        coefficient_constants = [
            emit
            for emit in descriptor_emits
            if emit.results.get("dst")
            == ValueRef.temporary(f"inverse_two_pi_limb_{index}_bits")
        ]
        assert len(coefficient_constants) == 1
        assert coefficient_constants[0].immediates == {"i": bits}


def test_sine_polynomial_serializes_accumulator_horner_chain() -> None:
    emits = nonlinear._sine_polynomial_emits(
        ValueRef.operand("input"), ValueRef.result("result")
    )
    descriptor_emits = [emit for emit in emits if isinstance(emit, EmitDescriptorOp)]
    emits_by_result = {
        result: emit for emit in descriptor_emits for result in emit.results.values()
    }

    squared = ValueRef.temporary("trig_polynomial_argument_squared")
    polynomial = ValueRef.temporary("trig_polynomial_coefficient_3")
    control = ValueRef.temporary("trig_polynomial_control")
    one = ValueRef.temporary("trig_polynomial_one")
    for coefficient_index in (2, 1, 0):
        product = ValueRef.temporary(
            f"trig_polynomial_stage_{coefficient_index}_product_accumulator"
        )
        accumulator = ValueRef.temporary(
            f"trig_polynomial_stage_{coefficient_index}_accumulator"
        )
        next_polynomial = ValueRef.temporary(
            f"trig_polynomial_stage_{coefficient_index}"
        )
        product_emit = emits_by_result[product]
        assert product_emit.descriptor.key == (
            "amd.xdna.aie2p.multiply.bf16x32.configured"
        )
        assert product_emit.operands == {
            "s1": squared,
            "s2": polynomial,
            "acc": control,
        }

        accumulator_emit = emits_by_result[accumulator]
        assert accumulator_emit.descriptor.key == (
            "amd.xdna.aie2p.accumulate.bf16x32.configured"
        )
        assert accumulator_emit.operands == {
            "acc1": product,
            "s1": ValueRef.temporary(
                f"trig_polynomial_coefficient_{coefficient_index}"
            ),
            "s2": one,
            "acc": control,
        }
        polynomial = next_polynomial

    result_emit = emits_by_result[
        ValueRef.temporary("trig_polynomial_result_accumulator")
    ]
    assert result_emit.descriptor.key == ("amd.xdna.aie2p.multiply.bf16x32.configured")
    assert result_emit.operands == {
        "s1": ValueRef.temporary("trig_polynomial_argument_bf16x32"),
        "s2": polynomial,
        "acc": control,
    }


def test_trig_rules_use_native_bf16_mac() -> None:
    for report_key in (
        "native_bf16_mac_sin_radians",
        "native_bf16_mac_cos_radians",
        "native_bf16_mac_sin_turns",
        "native_bf16_mac_cos_turns",
    ):
        rules = [
            rule
            for rule in nonlinear.AIE2P_NONLINEAR_RULES
            if rule.report_key == report_key
        ]
        assert len(rules) == 1
        descriptor_keys = {
            emit.descriptor.key
            for emit in rules[0].emit
            if isinstance(emit, EmitDescriptorOp)
        }
        assert "amd.xdna.aie2p.accumulate.bf16x32.configured" in descriptor_keys
        assert "amd.xdna.aie2p.multiply.bf16x32.configured" in descriptor_keys


def test_native_bf16_nonlinear_results_bypass_f32_round_trip() -> None:
    fused_rules = [
        rule
        for rule in nonlinear.AIE2P_NONLINEAR_RULES
        if rule.report_key.endswith("_f32_to_bf16")
    ]
    scalar_operations = {
        scalar_math.scalar_expf,
        scalar_math.scalar_exp2f,
        scalar_math.scalar_tanhf,
    }
    vector_operations = {
        vector.vector_expf,
        vector.vector_exp2f,
        vector.vector_tanhf,
    }
    assert Counter(rule.source_op for rule in fused_rules) == Counter(
        {
            scalar_math.scalar_expf: 1,
            scalar_math.scalar_exp2f: 1,
            scalar_math.scalar_tanhf: 1,
            vector.vector_expf: 1,
            vector.vector_exp2f: 2,
            vector.vector_tanhf: 2,
        }
    )

    register_classes = defaultdict(set)
    for rule in fused_rules:
        assert rule.priority == 1
        assert not rule.target_query
        (narrow,) = rule.source_nodes
        scalar = rule.source_op in scalar_operations
        assert scalar or rule.source_op in vector_operations
        assert narrow.source_op is (
            scalar_conversion.scalar_fptrunc if scalar else vector.vector_fptrunc
        )
        assert narrow.relation is SourceNodeRelation.EXCLUSIVE_USER
        assert narrow.parent_value == ValueRef.result("result")
        assert narrow.node_value == ValueRef.operand("input")

        descriptor_keys = [
            emit.descriptor.key
            for emit in rule.emit
            if isinstance(emit, EmitDescriptorOp)
        ]
        assert "amd.xdna.aie2p.convert.bf16x16.to.f32x16" not in descriptor_keys
        if scalar:
            result_emit = rule.emit[-1]
            assert isinstance(result_emit, EmitDescriptorOp)
            assert result_emit.descriptor.key == (
                "amd.xdna.aie2p.extract.i16.immediate"
            )
            assert result_emit.results == {
                "dst": ValueRef.result("result", source_node="narrow")
            }
        else:
            result_emit = rule.emit[-1]
            assert isinstance(result_emit, EmitRegisterConcat)
            assert result_emit.sources[0] == result_emit.sources[1]
            assert result_emit.result == ValueRef.result("result", source_node="narrow")
            register_classes[rule.source_op].update(
                guard.register_class
                for guard in rule.guards
                if guard.kind is GuardKind.LOW_VALUE_REGISTER_CLASS
            )

    assert register_classes == {
        vector.vector_expf: set(),
        vector.vector_exp2f: {"aie2p.vec256", "aie2p.ebmll"},
        vector.vector_tanhf: {"aie2p.vec256", "aie2p.ebmll"},
    }
