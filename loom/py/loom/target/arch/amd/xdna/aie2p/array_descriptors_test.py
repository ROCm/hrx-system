# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.target.arch.amd.xdna.aie2p.array_descriptors import (
    AIE2P_ARRAY_DESCRIPTOR_SET,
)
from loom.target.low_descriptors import (
    DescriptorFlag,
    DescriptorOpKind,
    ImmediateFlag,
    ImmediateKind,
    RegClassFlag,
)


def test_array_descriptor_set_is_a_resident_topology_contract() -> None:
    descriptor_set = AIE2P_ARRAY_DESCRIPTOR_SET

    assert descriptor_set.key == "amd.xdna.aie2p.array"
    assert descriptor_set.feature_key == "amd.xdna.aie2p.array.v2"
    assert not descriptor_set.physical_registers
    assert not descriptor_set.physical_register_views
    assert not descriptor_set.register_parts
    assert not descriptor_set.timing_events
    assert not descriptor_set.event_separations
    assert all(
        RegClassFlag.VIRTUAL_ONLY in reg_class.flags
        and RegClassFlag.REFERENCE in reg_class.flags
        and RegClassFlag.UNSPILLABLE in reg_class.flags
        for reg_class in descriptor_set.reg_classes
    )


def test_topology_parameters_are_ssa_operands() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_ARRAY_DESCRIPTOR_SET.descriptors
    }

    constant = descriptors["amd.xdna.aie2p.array.constant.u32"]
    assert constant.op_kind is DescriptorOpKind.CONST
    assert constant.immediates[0].field_name == "value"

    group = descriptors["amd.xdna.aie2p.array.group"]
    assert [operand.field_name for operand in group.operands] == ["result", "lanes"]
    assert not group.immediates

    channel = descriptors["amd.xdna.aie2p.array.channel"]
    assert [operand.field_name for operand in channel.operands] == [
        "result",
        "sender",
        "receiver",
        "capacity",
    ]
    assert not channel.immediates

    location = descriptors["amd.xdna.aie2p.array.constrain.location"]
    assert [operand.field_name for operand in location.operands] == [
        "worker",
        "column",
        "row",
    ]
    assert not location.immediates


def test_worker_entry_is_an_explicit_symbolic_product_edge() -> None:
    worker = next(
        descriptor
        for descriptor in AIE2P_ARRAY_DESCRIPTOR_SET.descriptors
        if descriptor.key == "amd.xdna.aie2p.array.worker"
    )

    assert worker.asm_forms[0].mnemonic == "worker"
    entry = next(
        immediate for immediate in worker.immediates if immediate.field_name == "entry"
    )
    assert entry.kind is ImmediateKind.ORDINAL
    assert entry.flags == (ImmediateFlag.SYMBOLIC,)
    assert DescriptorFlag.SIDE_EFFECTING in worker.flags


def test_channels_are_typed_persistent_topology_edges() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_ARRAY_DESCRIPTOR_SET.descriptors
    }
    sender = descriptors["amd.xdna.aie2p.array.sender"]
    receiver = descriptors["amd.xdna.aie2p.array.receiver"]
    partition = descriptors["amd.xdna.aie2p.array.partition"]
    channel = descriptors["amd.xdna.aie2p.array.channel"]

    assert {alternative.reg_class for alternative in sender.operands[1].reg_alts} == {
        "aie2p.array.binding",
        "aie2p.array.worker",
    }
    assert {alternative.reg_class for alternative in receiver.operands[1].reg_alts} == {
        "aie2p.array.binding",
        "aie2p.array.worker",
    }
    assert [operand.field_name for operand in partition.operands] == [
        "result",
        "source",
        "lane",
        "lanes",
    ]
    assert DescriptorFlag.SIDE_EFFECTING in channel.flags


def test_asm_mnemonics_are_target_relative() -> None:
    assert {
        descriptor.asm_forms[0].mnemonic
        for descriptor in AIE2P_ARRAY_DESCRIPTOR_SET.descriptors
    } == {
        "binding",
        "channel",
        "constant.u32",
        "constrain.location",
        "group",
        "partition",
        "receiver",
        "sender",
        "worker",
    }
