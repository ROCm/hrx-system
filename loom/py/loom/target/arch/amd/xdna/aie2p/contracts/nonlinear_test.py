# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMD XDNA AIE2P approximate nonlinear contracts."""

from __future__ import annotations

import math
import struct

from loom.target.arch.amd.xdna.aie2p.contracts import nonlinear
from loom.target.contracts import EmitDescriptorOp


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
