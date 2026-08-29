# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE
from loom.target.low_descriptors import EffectKind, OperandRole, RegClassFlag


def test_core_descriptor_closure_is_complete() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    assert len(descriptor_set.physical_registers) == 359
    assert len(descriptor_set.reg_classes) == 9
    assert len(descriptor_set.descriptors) == 11
    assert tuple(row.name for row in descriptor_set.physical_registers) == tuple(
        row.name for row in CORE_MACHINE_TABLE.physical_registers
    )
    assert tuple(
        row.atomic_units for row in descriptor_set.physical_registers
    ) == tuple(row.atomic_units for row in CORE_MACHINE_TABLE.physical_registers)


def test_low_register_classes_retain_machine_candidate_order() -> None:
    machine_classes = {row.name: row for row in CORE_MACHINE_TABLE.register_classes}
    for register_class in AIE2P_CORE_DESCRIPTOR_SET.reg_classes:
        assert (
            register_class.physical_registers
            == machine_classes[register_class.name].candidates
        )
        assert RegClassFlag.PHYSICAL in register_class.flags
        assert RegClassFlag.EXPLICIT_PHYSICAL_REGISTERS in register_class.flags
        assert RegClassFlag.UNSPILLABLE in register_class.flags


def test_descriptor_encoding_ids_and_adapters_are_materialized() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_CORE_DESCRIPTOR_SET.descriptors
    }
    vector_add = descriptors["amd.xdna.aie2p.add.i32x16"]
    assert vector_add.encoding_id != 0
    assert [operand.field_name for operand in vector_add.operands] == [
        "d",
        "s1",
        "s2",
    ]
    assert all(operand.encoding_field_id != 0 for operand in vector_add.operands)
    assert all(operand.encoding_adapter_id != 0 for operand in vector_add.operands)
    assert vector_add.operands[0].ready_stage == 2
    assert vector_add.operands[1].read_stage == 1

    vector_load = descriptors["amd.xdna.aie2p.load.a.i8x64.indexed.immediate"]
    assert vector_load.operands[0].ready_stage == 7


def test_seed_schedule_contract_retains_endpoint_events_and_separations() -> None:
    descriptor_set = AIE2P_CORE_DESCRIPTOR_SET
    descriptors = {row.key: row for row in descriptor_set.descriptors}
    separations = {
        (row.producer_event, row.consumer_event): row.minimum_issue_separation_cycles
        for row in descriptor_set.event_separations
    }

    scalar_add = descriptors["amd.xdna.aie2p.add.i32.immediate"]
    scalar_write = next(
        row.write_event for row in scalar_add.operands if row.role is OperandRole.RESULT
    )
    scalar_read = next(
        row.read_event for row in scalar_add.operands if row.role is OperandRole.OPERAND
    )
    scalar_store = descriptors["amd.xdna.aie2p.store.scalar.indexed.immediate"]
    scalar_store_read = next(
        row.read_event for row in scalar_store.operands if row.field_name == "src"
    )
    assert separations[scalar_write, scalar_read] == 1
    assert separations[scalar_write, scalar_write] == 1
    assert separations[scalar_read, scalar_write] == 0
    assert separations[scalar_write, scalar_store_read] == 1

    vector_add = descriptors["amd.xdna.aie2p.add.i32x16"]
    vector_write = next(
        row.write_event for row in vector_add.operands if row.role is OperandRole.RESULT
    )
    vector_read = next(
        row.read_event for row in vector_add.operands if row.role is OperandRole.OPERAND
    )
    vector_load = descriptors["amd.xdna.aie2p.load.a.i8x64.indexed.immediate"]
    load_write = next(
        row.write_event
        for row in vector_load.operands
        if row.role is OperandRole.RESULT
    )
    vector_store = descriptors["amd.xdna.aie2p.store.i8x64.indexed.immediate"]
    vector_store_read = next(
        row.read_event for row in vector_store.operands if row.field_name == "src"
    )
    assert separations[load_write, vector_read] == 7
    assert separations[vector_write, vector_read] == 1
    assert separations[vector_write, vector_write] == 1
    assert separations[vector_read, vector_write] == 0
    assert separations[vector_write, vector_store_read] == 2

    memory_write = next(
        row.timing_event for row in vector_store.effects if row.kind is EffectKind.WRITE
    )
    memory_read = next(
        row.timing_event for row in vector_load.effects if row.kind is EffectKind.READ
    )
    assert separations[memory_write, memory_read] == 1
    assert vector_load.immediates[0].encoding_field_id != 0
    assert vector_load.immediates[0].encoding_id != 0
    assert vector_load.immediates[0].value_step == 64

    scalar_add = descriptors["amd.xdna.aie2p.add.i32.immediate"]
    assert scalar_add.immediates[0].value_step == 1
