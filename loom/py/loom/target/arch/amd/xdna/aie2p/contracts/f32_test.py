# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for exact AMD XDNA AIE2P binary32 contracts."""

from __future__ import annotations

import random
import struct

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.target.arch.amd.xdna.aie2p.contracts.f32 import AIE2P_F32_RULES
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractFragment,
    DescriptorEmitForm,
    DescriptorResultType,
    ValueRef,
    compile_lower_rule_set,
)
from loom.target.contracts.kinds import SourceValueKind


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _s32(value: int) -> int:
    value = _u32(value)
    return value - (1 << 32) if value & 0x80000000 else value


def _leading_zeros(value: int) -> int:
    value = _u32(value)
    return 32 - value.bit_length()


def _logical_shift(value: int, amount: int) -> int:
    amount = _s32(amount)
    assert -31 <= amount <= 31
    if amount < 0:
        return _u32(value) >> -amount
    return _u32(value << amount)


def _value(
    ref: ValueRef,
    inputs: dict[str, int],
    temporaries: dict[str, int],
) -> int:
    if ref.kind is SourceValueKind.OPERAND:
        return inputs[ref.field]
    assert ref.kind is SourceValueKind.TEMPORARY
    return temporaries[ref.field]


def _evaluate_lane(lhs: int, rhs: int) -> int:
    inputs = {"lhs": _u32(lhs), "rhs": _u32(rhs)}
    temporaries: dict[str, int] = {}
    result: int | None = None
    for emit in AIE2P_F32_RULES[0].emit:
        descriptor_key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        result_ref = next(iter(emit.results.values()))
        if emit.form is DescriptorEmitForm.CONST:
            value = _u32(emit.immediates["i"])
        else:
            operands = {
                name: _value(ref, inputs, temporaries)
                for name, ref in emit.operands.items()
            }
            if descriptor_key == "add.i32.immediate":
                value = operands["s0"] + emit.immediates["imm"]
            elif descriptor_key == "add.i32":
                value = operands["s0"] + operands["s1"]
            elif descriptor_key == "sub.i32":
                value = operands["s0"] - operands["s1"]
            elif descriptor_key == "mul.i32":
                value = operands["s0"] * operands["s1"]
            elif descriptor_key == "and.i32":
                value = operands["s0"] & operands["s1"]
            elif descriptor_key == "or.i32":
                value = operands["s0"] | operands["s1"]
            elif descriptor_key == "xor.i32":
                value = operands["s0"] ^ operands["s1"]
            elif descriptor_key == "lshl.i32":
                value = _logical_shift(operands["s0"], operands["s1"])
            elif descriptor_key == "clz.i32":
                value = _leading_zeros(operands["s0"])
            elif descriptor_key == "cmp.eq.i32":
                value = int(operands["s0"] == operands["s1"])
            elif descriptor_key == "cmp.ult.i32":
                value = int(operands["s0"] < operands["s1"])
            elif descriptor_key == "cmp.slt.i32":
                value = int(_s32(operands["s0"]) < _s32(operands["s1"]))
            elif descriptor_key == "cmp.sge.i32":
                value = int(_s32(operands["s0"]) >= _s32(operands["s1"]))
            elif descriptor_key == "cmp.eqz.i32":
                value = int(operands["s0"] == 0)
            elif descriptor_key == "cmp.nez.i32":
                value = int(operands["s0"] != 0)
            elif descriptor_key == "select.zero.i32":
                value = operands["s0"] if operands["s2"] == 0 else operands["s1"]
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


def _float_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", _u32(bits)))[0]


def _float_bits(value: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return 0xFF800000 if value < 0 else 0x7F800000


def _reference_multiply(lhs: int, rhs: int) -> int:
    absolute_mask = 0x7FFFFFFF
    infinity = 0x7F800000
    quiet_bit = 0x00400000
    lhs_absolute = lhs & absolute_mask
    rhs_absolute = rhs & absolute_mask
    if lhs_absolute > infinity:
        return lhs | quiet_bit
    if rhs_absolute > infinity:
        return rhs | quiet_bit
    if (lhs_absolute == infinity and rhs_absolute == 0) or (
        rhs_absolute == infinity and lhs_absolute == 0
    ):
        return infinity | quiet_bit
    return _float_bits(_float_from_bits(lhs) * _float_from_bits(rhs))


def test_f32_multiply_program_is_compact_descriptor_data() -> None:
    for rule in AIE2P_F32_RULES:
        rule.validate(AIE2P_CORE_DESCRIPTOR_SET)
        assert rule.report_key == "exact_binary32"
        assert all(
            emit.form in (DescriptorEmitForm.CONST, DescriptorEmitForm.OP)
            for emit in rule.emit
        )
        assert all(
            (
                emit.form is DescriptorEmitForm.CONST
                if ".constant.i32" in emit.descriptor.key
                else emit.form is DescriptorEmitForm.OP
            )
            for emit in rule.emit
        )
        assert all(
            tuple(emit.result_types.values()) == (DescriptorResultType(),)
            for emit in rule.emit[:-1]
        )
        assert rule.emit[-1].results["d0"].kind is SourceValueKind.RESULT
    compiled = compile_lower_rule_set(
        ContractFragment(
            name="amd.xdna.aie2p.f32.multiply.test",
            descriptor_set=AIE2P_CORE_DESCRIPTOR_SET,
            cases=AIE2P_F32_RULES,
        ),
        dialect_ops={"scalar": ALL_SCALAR_OPS},
    )
    assert compiled.rules[0].emit_count == len(AIE2P_F32_RULES[0].emit)
    assert (
        sum(
            emit.descriptor.key == "amd.xdna.aie2p.mul.i32"
            for emit in AIE2P_F32_RULES[0].emit
        )
        == 4
    )


def test_f32_multiply_program_matches_binary32_oracle() -> None:
    edge_values = (
        0x00000000,
        0x80000000,
        0x00000001,
        0x007FFFFF,
        0x00800000,
        0x00800001,
        0x3EFFFFFF,
        0x3F000000,
        0x3F000001,
        0x3F7FFFFF,
        0x3F800000,
        0x3F800001,
        0x3FFFFFFF,
        0x40000000,
        0x40000001,
        0x7F7FFFFE,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0x7FC00000,
        0x7FFFFFFF,
        0xFF800001,
        0xFFFFFFFF,
    )
    pairs = [(lhs, rhs) for lhs in edge_values for rhs in edge_values]
    random_source = random.Random(0xA1E2F32)
    pairs.extend(
        (random_source.getrandbits(32), random_source.getrandbits(32))
        for _ in range(20_000)
    )
    for lhs, rhs in pairs:
        assert _evaluate_lane(lhs, rhs) == _reference_multiply(lhs, rhs), (
            f"{lhs:08x} * {rhs:08x}"
        )
