# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Byte-exact BFP storage interpretation, independent of numerical rounding."""

import random

from loom.target.arch.amd.xdna.aie2p.contracts.bfp import bfp_unpack_emits
from loom.target.contracts import DescriptorEmitForm, ValueRef


def _unpack(payload: bytes, padding: bytes) -> tuple[bytes, bytes]:
    low = ValueRef.temporary("low")
    high = ValueRef.temporary("high")
    result = ValueRef.result("result")
    values = {low: payload[:64], high: payload[64:] + padding}
    for emit in bfp_unpack_emits(low, high, result):
        key = emit.descriptor.key.removeprefix("amd.xdna.aie2p.")
        operands = {name: values[ref] for name, ref in emit.operands.items()}
        if emit.form is DescriptorEmitForm.CONST:
            value = emit.immediates["i"]
        elif key == "shift.bytes.x.configured":
            start = operands["shift"]
            value = (operands["s1"] + operands["s2"])[start : start + 64]
        elif key == "select.i32x16":
            value = bytes(
                operands["s2" if operands["sel"] & (1 << (index // 4)) else "s1"][index]
                for index in range(64)
            )
        elif key == "select.mask.i32":
            value = (operands["s0"] + emit.immediates["imm"]) & 0xFFFFFFFF
        elif key == "extract.i8.immediate":
            value = operands["s1"][emit.immediates["idx"]]
        elif key == "lshl.i32":
            value = (operands["s0"] << operands["s1"]) & 0xFFFFFFFF
        elif key == "or.i32":
            value = operands["s0"] | operands["s1"]
        elif key == "bfp.mantissas.from.vector512":
            value = (operands["src"], b"")
        elif key in (
            "bfp.exponents.low32.from.scalar",
            "bfp.exponents.high32.from.scalar",
        ):
            mantissas, exponents = operands["storage"]
            value = (mantissas, exponents + operands["src"].to_bytes(4, "little"))
        else:
            raise AssertionError(key)
        assert len(emit.results) == 1
        values[next(iter(emit.results.values()))] = value
    return values[result]


def _expected(payload: bytes) -> tuple[bytes, bytes]:
    return (
        bytes(value for index, value in enumerate(payload) if index % 9),
        payload[::9],
    )


def test_packed_bytes_preserve_every_payload_bit() -> None:
    # A one-bit witness at every input position detects dropped, duplicated,
    # swapped, and sign-extended exponent/mantissa bits independently.
    for bit in range(72 * 8):
        payload = (1 << bit).to_bytes(72, "little")
        assert _unpack(payload, bytes(56)) == _expected(payload)


def test_packed_bytes_ignore_the_unused_carrier_tail() -> None:
    random_source = random.Random(0xBF16)
    for _ in range(32):
        payload = random_source.randbytes(72)
        assert _unpack(payload, random_source.randbytes(56)) == _expected(payload)
    payload = bytes(range(72))
    assert _unpack(payload, bytes([0xFF]) * 56) == _expected(payload)
