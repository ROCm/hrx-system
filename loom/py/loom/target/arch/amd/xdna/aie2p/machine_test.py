# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.target.arch.amd.xdna.aie.machine import (
    decode_immediate,
    encode_immediate,
    validate_machine_table,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import CORE_ENCODING_TABLE
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE


def test_core_machine_table_is_structurally_complete() -> None:
    validate_machine_table(CORE_MACHINE_TABLE, CORE_ENCODING_TABLE)

    assert len(CORE_MACHINE_TABLE.atomic_unit_names) == 207
    assert len(CORE_MACHINE_TABLE.physical_registers) == 359
    assert len(CORE_MACHINE_TABLE.register_classes) == 368
    assert len(CORE_MACHINE_TABLE.register_adapters) == 58
    assert len(CORE_MACHINE_TABLE.immediates) == 21
    assert len(CORE_MACHINE_TABLE.forms) == 880
    assert (
        sum(
            len(register.atomic_units)
            for register in CORE_MACHINE_TABLE.physical_registers
        )
        == 701
    )
    assert (
        sum(
            len(register_class.candidates)
            for register_class in CORE_MACHINE_TABLE.register_classes
        )
        == 3426
    )
    assert (
        sum(
            len(adapter.register_encodings)
            for adapter in CORE_MACHINE_TABLE.register_adapters
        )
        == 1518
    )
    assert (
        len(
            {
                adapter.register_encodings
                for adapter in CORE_MACHINE_TABLE.register_adapters
            }
        )
        == 25
    )
    assert sum(len(form.ties) for form in CORE_MACHINE_TABLE.forms) == 386
    assert sum(len(form.implicit_defs) for form in CORE_MACHINE_TABLE.forms) == 387
    assert sum(len(form.implicit_uses) for form in CORE_MACHINE_TABLE.forms) == 840


def test_atomic_units_preserve_subregister_aliasing() -> None:
    registers = {
        register.name: register for register in CORE_MACHINE_TABLE.physical_registers
    }
    x0 = registers["x0"]
    assert x0.subregisters == ("wl0", "wh0")
    assert x0.subregister_indices == ("sub_256_lo", "sub_256_hi")
    assert set(x0.atomic_units) == (
        set(registers["wl0"].atomic_units) | set(registers["wh0"].atomic_units)
    )


def test_register_adapter_is_operand_local() -> None:
    adapters = {
        adapter.name: adapter for adapter in CORE_MACHINE_TABLE.register_adapters
    }
    destination_values = dict(adapters["OP_mAguDst"].register_encodings)
    source_values = dict(adapters["OP_mAguSrc"].register_encodings)

    assert destination_values["p1"] == 12
    assert source_values["p1"] == 18


def test_q_register_adapters_repair_source_encoder_aliasing() -> None:
    adapters = {
        adapter.name: adapter for adapter in CORE_MACHINE_TABLE.register_adapters
    }
    source_values = {"q0": 0, "q1": 3, "q2": 4, "q3": 7}
    architectural_values = {"q0": 0, "q1": 1, "q2": 2, "q3": 3}
    for name in ("OP_mQQsa", "OP_mQQsm", "OP_mQQss"):
        assert dict(adapters[name].register_encodings) == source_values
        assert dict(adapters[name].effective_register_encodings) == architectural_values


def test_all_immediate_domains_round_trip_boundaries() -> None:
    for immediate in CORE_MACHINE_TABLE.immediates:
        fixed_zero_bits = immediate.step.bit_length() - 1
        if immediate.is_negative:
            minimum = -(1 << (immediate.encoded_width_bits + fixed_zero_bits))
            maximum = -immediate.step
        elif immediate.is_signed:
            semantic_bits = immediate.encoded_width_bits + fixed_zero_bits
            minimum = -(1 << (semantic_bits - 1))
            maximum = (1 << (semantic_bits - 1)) - immediate.step
        else:
            minimum = 0
            maximum = (
                1 << (immediate.encoded_width_bits + fixed_zero_bits)
            ) - immediate.step
        for value in (minimum, maximum):
            assert (
                decode_immediate(immediate, encode_immediate(immediate, value)) == value
            )


def test_negative_immediate_rejects_zero() -> None:
    immediate = next(
        row for row in CORE_MACHINE_TABLE.immediates if row.name == "c12n_step4"
    )
    with pytest.raises(ValueError, match="outside"):
        encode_immediate(immediate, 0)
