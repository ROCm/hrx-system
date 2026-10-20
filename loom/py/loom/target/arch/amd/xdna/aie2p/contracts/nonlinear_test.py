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


def _bf16_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _round_to_bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
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
        polynomial = _round_to_bf16(_f32(coefficient + _f32(squared * polynomial)))
    return _f32(argument * polynomial)


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
