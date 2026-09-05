# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for exact AMD XDNA AIE2P scalar conversion contracts."""

from __future__ import annotations

import random
import struct

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.target.arch.amd.xdna.aie2p.contracts.conversion import (
    AIE2P_CONVERSION_RULES,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    ValueRef,
    compile_lower_rule_set,
)
from loom.target.contracts.kinds import SourceValueKind
from loom.target.low_descriptors import ConstraintKind, OperandRole


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _s32(value: int) -> int:
    value = _u32(value)
    return value - (1 << 32) if value & 0x80000000 else value


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _value(ref: ValueRef, input_value: int, temporaries: dict[str, int]) -> int:
    if ref.kind is SourceValueKind.OPERAND:
        assert ref.field == "input"
        return input_value
    assert ref.kind is SourceValueKind.TEMPORARY
    return temporaries[ref.field]


def _evaluate(rule: DescriptorRule, input_value: int) -> int:
    temporaries: dict[str, int] = {}
    result: int | None = None
    for emit in rule.emit:
        descriptor_key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        result_ref = next(iter(emit.results.values()))
        if emit.form is DescriptorEmitForm.CONST:
            value = _u32(emit.immediates["i"])
        else:
            operands = {
                name: _value(ref, input_value, temporaries)
                for name, ref in emit.operands.items()
            }
            if descriptor_key == "extend.signed.i8":
                value = operands["s0"] & 0xFF
                if value & 0x80:
                    value |= 0xFFFFFF00
            elif descriptor_key == "extend.signed.i16":
                value = operands["s0"] & 0xFFFF
                if value & 0x8000:
                    value |= 0xFFFF0000
            elif descriptor_key == "extend.unsigned.i8":
                value = operands["s0"] & 0xFF
            elif descriptor_key == "extend.unsigned.i16":
                value = operands["s0"] & 0xFFFF
            elif descriptor_key == "convert.signed.i32.to.f32":
                assert operands["m"] == 0
                value = _float_bits(float(_s32(operands["s0"])))
            elif descriptor_key == "and.i32":
                value = operands["s0"] & operands["s1"]
            elif descriptor_key == "or.i32":
                value = operands["s0"] | operands["s1"]
            elif descriptor_key == "add.i32":
                value = operands["s0"] + operands["s1"]
            elif descriptor_key == "sub.i32":
                value = operands["s0"] - operands["s1"]
            elif descriptor_key == "lshl.i32":
                shift = _s32(operands["s1"])
                assert -31 <= shift <= 31
                value = (
                    operands["s0"] << shift if shift >= 0 else operands["s0"] >> -shift
                )
            elif descriptor_key == "clz.i32":
                value = 32 - _u32(operands["s0"]).bit_length()
            elif descriptor_key == "cmp.eqz.i32":
                value = int(_u32(operands["s0"]) == 0)
            elif descriptor_key == "cmp.eq.i32":
                value = int(_u32(operands["s0"]) == _u32(operands["s1"]))
            elif descriptor_key == "cmp.slt.i32":
                value = int(_s32(operands["s0"]) < _s32(operands["s1"]))
            elif descriptor_key == "select.nonzero.i32":
                value = operands["s0"] if operands["s2"] != 0 else operands["s1"]
            else:
                raise AssertionError(f"unmodeled descriptor {descriptor_key}")
            value = _u32(value)

        if result_ref.kind is SourceValueKind.RESULT:
            result = value
        else:
            assert result_ref.kind is SourceValueKind.TEMPORARY
            temporaries[result_ref.field] = value
    assert result is not None
    return result


def _reference_f16_to_f32(input_bits: int) -> int:
    sign = (input_bits & 0x8000) << 16
    exponent = (input_bits >> 10) & 0x1F
    fraction = input_bits & 0x03FF
    if exponent == 0:
        if fraction == 0:
            return sign
        unbiased_exponent = -14
        while fraction & 0x0400 == 0:
            fraction <<= 1
            unbiased_exponent -= 1
        fraction &= 0x03FF
        return sign | ((unbiased_exponent + 127) << 23) | (fraction << 13)
    if exponent == 0x1F:
        return sign | 0x7F800000 | (fraction << 13)
    return sign | ((exponent + 112) << 23) | (fraction << 13)


def _rule(report_key: str) -> DescriptorRule:
    return next(
        rule for rule in AIE2P_CONVERSION_RULES if rule.report_key == report_key
    )


def test_conversion_programs_are_valid_compact_descriptor_data() -> None:
    for rule in AIE2P_CONVERSION_RULES:
        rule.validate(AIE2P_CORE_DESCRIPTOR_SET)
    compiled = compile_lower_rule_set(
        ContractFragment(
            name="amd.xdna.aie2p.conversion.test",
            descriptor_set=AIE2P_CORE_DESCRIPTOR_SET,
            cases=AIE2P_CONVERSION_RULES,
        ),
        dialect_ops={"scalar": ALL_SCALAR_OPS},
    )
    assert all(rule.emit_count > 0 for rule in compiled.rules)

    for report_key in (
        "native_signed_i8_to_binary32",
        "native_signed_i16_to_binary32",
        "native_signed_i32_to_binary32",
        "native_unsigned_i8_to_binary32",
        "native_unsigned_i16_to_binary32",
    ):
        rule = _rule(report_key)
        assert rule.emit[-2].descriptor.key == (
            "amd.xdna.aie2p.constant.i32.fx2flt-scale"
        )
        assert rule.emit[-1].descriptor.key == (
            "amd.xdna.aie2p.convert.signed.i32.to.f32"
        )


def test_native_integer_conversion_preserves_scale_and_status_state() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    scale = descriptors["amd.xdna.aie2p.constant.i32.fx2flt-scale"]
    assert scale.operands[0].reg_alts[0].reg_class == "aie2p.ms2"
    assert scale.constraints[0].kind is ConstraintKind.REMATERIALIZABLE

    convert = descriptors["amd.xdna.aie2p.convert.signed.i32.to.f32"]
    assert [operand.field_name for operand in convert.operands] == [
        "d0",
        "s0",
        "m",
        "implicit_def_srfpcnvfx2fl",
        "implicit_use_crfpcnvfx2flmask",
    ]
    assert [operand.reg_alts[0].reg_class for operand in convert.operands] == [
        "aie2p.er",
        "aie2p.er",
        "aie2p.ms2",
        "aie2p.state.srfpcnvfx2fl",
        "aie2p.state.crfpcnvfx2flmask",
    ]
    assert convert.operands[0].role is OperandRole.RESULT
    assert [
        (operand.read_stage, operand.ready_stage) for operand in convert.operands
    ] == [
        (0, 2),
        (1, 0),
        (1, 0),
        (0, 1),
        (1, 0),
    ]


def test_integer_to_f32_programs_match_binary32_oracle() -> None:
    signed_i8 = _rule("native_signed_i8_to_binary32")
    unsigned_i8 = _rule("native_unsigned_i8_to_binary32")
    for bits in range(1 << 8):
        signed = bits - 256 if bits & 0x80 else bits
        assert _evaluate(signed_i8, bits) == _float_bits(float(signed))
        assert _evaluate(unsigned_i8, bits) == _float_bits(float(bits))

    signed_i16 = _rule("native_signed_i16_to_binary32")
    unsigned_i16 = _rule("native_unsigned_i16_to_binary32")
    for bits in range(1 << 16):
        assert _evaluate(signed_i16, bits) == _float_bits(
            float(bits - (1 << 16) if bits & 0x8000 else bits)
        )
        assert _evaluate(unsigned_i16, bits) == _float_bits(float(bits))

    signed_i32 = _rule("native_signed_i32_to_binary32")
    values = [
        0,
        1,
        -1,
        2**24 - 1,
        2**24,
        2**24 + 1,
        2**31 - 1,
        -(2**31),
    ]
    random_source = random.Random(0xA1E2C0DE)
    values.extend(_s32(random_source.getrandbits(32)) for _ in range(20_000))
    for value in values:
        assert _evaluate(signed_i32, _u32(value)) == _float_bits(float(value))

    unsigned_i32 = _rule("exact_unsigned_i32_to_binary32")
    unsigned_values = [
        0,
        1,
        2**24 - 1,
        2**24,
        2**24 + 1,
        2**31 - 1,
        2**31,
        2**31 + 1,
        2**32 - 1,
    ]
    unsigned_values.extend(random_source.getrandbits(32) for _ in range(20_000))
    for value in unsigned_values:
        assert _evaluate(unsigned_i32, value) == _float_bits(float(value))


def test_16bit_float_widening_programs_match_exhaustive_oracles() -> None:
    f16 = _rule("exact_f16_to_binary32")
    bf16 = _rule("exact_bf16_to_binary32")
    for bits in range(1 << 16):
        assert _evaluate(f16, bits) == _reference_f16_to_f32(bits)
        assert _evaluate(bf16, bits) == bits << 16
