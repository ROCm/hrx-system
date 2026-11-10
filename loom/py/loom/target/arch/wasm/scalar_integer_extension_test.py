# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.arch.wasm.scalar_integer_extension import (
    scalar_integer_extension_rules,
)
from loom.target.contracts import GuardKind, ValueRef, descriptor_by_key


def _rules():
    return scalar_integer_extension_rules(
        lambda key: descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, key)
    )


def _widths(rule):
    widths = {
        guard.field: int(guard.type_pattern.elements[0][1:])
        for guard in rule.guards
        if guard.kind is GuardKind.VALUE_TYPE
    }
    return widths["input"], widths["result"]


def _signed_word(value):
    return int.from_bytes(value.to_bytes(4, "little"), "little", signed=True)


def _evaluate(rule, carrier):
    values = {ValueRef.operand("input"): carrier}
    for emit in rule.emit:
        key = emit.descriptor.key
        operands = {name: values[value] for name, value in emit.operands.items()}
        if key == "wasm.i32.const":
            result = emit.immediates["i32_value"]
        elif key == "wasm.i32.and":
            result = operands["lhs"] & operands["rhs"]
        elif key == "wasm.i32.shl":
            result = operands["lhs"] << (operands["rhs"] & 31)
        elif key == "wasm.i32.shr_s":
            result = _signed_word(operands["lhs"]) >> (operands["rhs"] & 31)
        elif key == "wasm.i64.extend_i32_s":
            result = _signed_word(operands["input"])
        elif key == "wasm.i64.extend_i32_u":
            result = operands["input"]
        else:
            raise AssertionError(f"unhandled instruction: {key}")
        width = 64 if key.startswith("wasm.i64.") else 32
        values[emit.results["dst"]] = result & ((1 << width) - 1)
    return values[ValueRef.result("result")]


def test_complete_signed_and_unsigned_family_with_nonnegative_specialization():
    ordinary = set()
    nonnegative = set()
    for rule in _rules():
        source, result = _widths(rule)
        signedness = rule.source_op.name.removeprefix("scalar.ext")
        range_guards = [
            guard for guard in rule.guards if guard.kind is GuardKind.VALUE_I64_RANGE
        ]
        if range_guards:
            assert signedness == "si"
            assert rule.priority == 1
            assert len(range_guards) == 1
            assert (range_guards[0].minimum, range_guards[0].maximum) == (
                0,
                (1 << (source - 1)) - 1,
            )
            nonnegative.add((source, result))
        else:
            assert rule.priority == 0
            ordinary.add((source, result, signedness))
    pairs = {(8, 16), (8, 32), (8, 64), (16, 32), (16, 64), (32, 64)}
    assert ordinary == {
        (source, result, sign) for source, result in pairs for sign in ("si", "ui")
    }
    assert nonnegative == pairs - {(32, 64)}
    assert len(_rules()) == len(ordinary) + len(nonnegative)


def test_recipes_ignore_unused_carrier_bits_and_preserve_sign_boundaries():
    for rule in _rules():
        source, result = _widths(rule)
        mask = (1 << source) - 1
        samples = (
            range(256)
            if source == 8
            else {
                value & mask
                for bit in range(source + 1)
                for value in (
                    (1 << bit) - 1,
                    1 << bit,
                    (1 << bit) + 1,
                    -(1 << bit),
                    0xA5A5A5A5,
                )
            }
        )
        signed = rule.source_op.name == "scalar.extsi"
        for sample in samples:
            numeric = int.from_bytes(
                sample.to_bytes(source // 8, "little"), "little", signed=signed
            )
            if rule.priority and numeric < 0:
                continue
            expected = numeric & ((1 << result) - 1)
            for upper_bits in (0, 0xA5A5A5A5, 0xFFFFFFFF):
                carrier = sample | (upper_bits & ~mask)
                actual = _evaluate(rule, carrier) & ((1 << result) - 1)
                assert actual == expected, (source, result, signed, hex(carrier))
