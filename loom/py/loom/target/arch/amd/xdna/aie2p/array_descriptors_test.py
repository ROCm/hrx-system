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
    ImmediateFlag,
    ImmediateKind,
    OperandFlag,
    RegClassFlag,
)


def test_array_descriptor_set_exposes_only_logical_program_state() -> None:
    descriptor_set = AIE2P_ARRAY_DESCRIPTOR_SET

    assert descriptor_set.key == "amd.xdna.aie2p.array"
    assert len(descriptor_set.reg_classes) == 5
    assert len(descriptor_set.descriptors) == 8
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


def test_async_descriptors_preserve_explicit_token_dependencies() -> None:
    descriptors = {
        descriptor.key: descriptor
        for descriptor in AIE2P_ARRAY_DESCRIPTOR_SET.descriptors
    }
    invoke = descriptors["amd.xdna.aie2p.array.invoke"]
    await_ = descriptors["amd.xdna.aie2p.array.await"]

    assert invoke.operands[-1].flags == (OperandFlag.VARIADIC,)
    assert DescriptorFlag.SIDE_EFFECTING in invoke.flags
    assert len(await_.operands) == 2
    assert not await_.operands[0].flags
    assert await_.operands[-1].flags == (OperandFlag.VARIADIC,)
    assert DescriptorFlag.BARRIER in await_.flags
    assert DescriptorFlag.SIDE_EFFECTING in await_.flags


def test_asm_mnemonics_are_target_relative() -> None:
    assert {
        descriptor.asm_forms[0].mnemonic
        for descriptor in AIE2P_ARRAY_DESCRIPTOR_SET.descriptors
    } == {
        "await",
        "binding",
        "buffer",
        "constrain.location",
        "flow",
        "invoke",
        "transfer",
        "worker",
    }
