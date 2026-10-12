# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

import pytest

from loom.target.arch.amdgpu.descriptors.api import (
    _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS,
    _with_instruction_classes,
    _with_storage_lease_rows,
)
from loom.target.arch.amdgpu.descriptors.common import (
    _COUNTER_VMEM_LOAD,
    _COUNTER_X,
    _REG_PART_VGPR_HIGH16,
    _REG_PART_VGPR_LOW16,
    _REG_VGPR,
    _SCHEDULE_VMEM_LOAD,
)
from loom.target.arch.amdgpu.descriptors.contracts import (
    _amdgpu_contract_descriptor_from_overlay,
)
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
    AMDGPU_PROCESSOR_INFOS,
)
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    DescriptorSet,
    EffectKind,
    InstructionClass,
    OperandFlag,
    OperandRole,
    StorageLeaseAttachment,
    StorageLeaseFlag,
    StorageLeaseKind,
)

_NATIVE_TARGETS = (
    "cdna2",
    "cdna3",
    "cdna4",
    "gfx9_4_generic",
    "rdna3",
    "gfx11_generic",
    "rdna3_5",
    "rdna4m",
    "rdna4",
    "gfx12_generic",
    "gfx12_5_generic",
    "rdna4_gfx125x",
    "rdna4_gfx1250_a0",
    "rdna4_gfx1251",
)
_XCNT_TARGETS = (
    "gfx12_5_generic",
    "rdna4_gfx125x",
    "rdna4_gfx1250_a0",
    "rdna4_gfx1251",
)


@pytest.mark.parametrize("target", _NATIVE_TARGETS)
def test_asynchronous_continuations_have_one_preserved_vgpr_part(target: str) -> None:
    continuations = set()
    for descriptor in _descriptor_set(target).descriptors:
        sources = tuple(
            operand
            for operand in descriptor.operands
            if operand.role is OperandRole.OPERAND
            and OperandFlag.STORAGE_CONTINUATION in operand.flags
        )
        if not sources or not any(
            effect.kind is EffectKind.READ for effect in descriptor.effects
        ):
            continue
        continuations.add(descriptor.key)
        results = tuple(
            operand
            for operand in descriptor.operands
            if operand.role is OperandRole.RESULT
        )
        assert len(results) == len(sources) == 1, descriptor.key
        result, source = results[0], sources[0]
        assert descriptor.operands[:2] == (result, source), descriptor.key
        assert result.unit_count == source.unit_count == 1, descriptor.key
        assert len(result.reg_alts) == len(source.reg_alts) == 1, descriptor.key
        assert result.reg_alts[0].reg_class == _REG_VGPR, descriptor.key
        assert source.reg_alts[0].reg_class == _REG_VGPR, descriptor.key
        assert result.reg_alts[0].register_part == _REG_PART_VGPR_HIGH16
        assert source.reg_alts[0].register_part == _REG_PART_VGPR_LOW16
        assert descriptor.constraints == (Constraint(ConstraintKind.TIED, 0, 1),)

    expected = {"amdgpu.ds_load_u16_d16_hi"}
    if target in ("rdna3", "gfx11_generic", "rdna3_5", "rdna4m"):
        expected.update(
            {
                "amdgpu.buffer_load_b16_d16_hi",
                "amdgpu.buffer_load_b16_d16_hi_vaddr_offset",
                "amdgpu.global_load_b16_d16_hi",
                "amdgpu.global_load_b16_d16_hi_saddr",
            }
        )
    if target in ("cdna2", "cdna3", "cdna4", "gfx9_4_generic"):
        expected = set()
    assert continuations == expected, (target, continuations, expected)


def _descriptor_set(target: str) -> DescriptorSet:
    builder = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target]
    return _with_instruction_classes(
        replace(
            builder.base,
            descriptors=(
                *(
                    _amdgpu_contract_descriptor_from_overlay(overlay)
                    for overlay in builder.overlay_rows()
                ),
                *builder.extra_descriptors,
                *builder.base.descriptors,
            ),
        )
    )


def test_scalar_capture_target_contracts() -> None:
    # A new native family needs an explicit source-capture hazard audit. In
    # particular, gfx10.1 requires a separate VMEM-to-scalar-write contract.
    assert set(_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS) == set(_NATIVE_TARGETS)
    legacy_processors = tuple(
        row for row in AMDGPU_PROCESSOR_INFOS if row.processor.startswith("gfx101")
    )
    assert legacy_processors
    for processor in legacy_processors:
        assert not processor.descriptor_set.key
        assert not processor.flags & AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION


@pytest.mark.parametrize("target", ["rdna3", "gfx11_generic", "rdna3_5", "rdna4m"])
def test_buffer_scalar_capture_preserves_all_result_leases(target: str) -> None:
    descriptor_set = _with_storage_lease_rows(
        _descriptor_set(target),
        builder_flags=_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].flags,
    )
    descriptors = {
        descriptor.key: descriptor for descriptor in descriptor_set.descriptors
    }
    families = (
        *(
            (name, units, ("", "_off_zero", "_vaddr_offset"))
            for name, units in (
                ("dword", 1),
                ("b64", 2),
                ("b96", 3),
                ("b128", 4),
                ("u8", 1),
                ("i8", 1),
                ("u16", 1),
                ("i16", 1),
            )
        ),
        ("b16_d16", 1, ("", "_vaddr_offset")),
        ("b16_d16_hi", 1, ("", "_vaddr_offset")),
    )
    for family, units, suffixes in families:
        for suffix in suffixes:
            descriptor = descriptors[f"amdgpu.buffer_load_{family}{suffix}"]
            assert len(descriptor.storage_leases) == 1, descriptor.key
            lease = descriptor.storage_leases[0]
            assert lease.kind is StorageLeaseKind.RESULT_WRITE
            assert lease.attachment is StorageLeaseAttachment.RESULT
            assert lease.attachment_index == 0
            assert lease.unit_offset == 0
            assert lease.unit_count == units
            assert lease.release_class_id == _COUNTER_VMEM_LOAD
            assert lease.flags == (
                StorageLeaseFlag.STARTS_AT_ISSUE,
                StorageLeaseFlag.RELEASE_BEFORE_BOUNDARY,
                StorageLeaseFlag.RELEASE_FOR_PRESSURE,
            )


@pytest.mark.parametrize("target", _NATIVE_TARGETS)
def test_memory_completion_never_leases_source_registers(target: str) -> None:
    descriptor_set = _descriptor_set(target)
    flags = _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].flags
    captured = _with_storage_lease_rows(descriptor_set, builder_flags=flags)
    memory_count = 0
    for before, after in zip(
        descriptor_set.descriptors, captured.descriptors, strict=True
    ):
        assert before == replace(after, storage_leases=()), before.key
        for lease in after.storage_leases:
            if lease.kind is StorageLeaseKind.SOURCE_READ:
                assert target in _XCNT_TARGETS, after.key
                assert lease.release_class_id == _COUNTER_X, after.key
                assert lease.attachment is StorageLeaseAttachment.OPERAND
                assert lease.flags == (
                    StorageLeaseFlag.STARTS_AT_ISSUE,
                    StorageLeaseFlag.MAY_CARRY_ACROSS_BOUNDARY,
                )
        if InstructionClass.GLOBAL_MEMORY in after.instruction_classes:
            memory_count += 1
    assert memory_count > 100


@pytest.mark.parametrize("target", _XCNT_TARGETS)
def test_xcnt_captures_every_smem_and_vmem_input(target: str) -> None:
    descriptor_set = _with_storage_lease_rows(
        _descriptor_set(target),
        builder_flags=_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].flags,
    )
    capture_classes = {
        "amdgpu.smem.load",
        "amdgpu.smem.store",
        "amdgpu.vmem.load",
        "amdgpu.vmem.load.lds",
        "amdgpu.vmem.store",
        "amdgpu.vmem.atomic.return",
        "amdgpu.vmem.atomic.no_return",
        "amdgpu.flat.load",
        "amdgpu.flat.store",
        "amdgpu.flat.atomic.return",
        "amdgpu.flat.atomic.no_return",
        "amdgpu.cluster.load.lds",
    }
    for descriptor in descriptor_set.descriptors:
        inputs = tuple(
            operand
            for operand in descriptor.operands
            if operand.role
            in (OperandRole.OPERAND, OperandRole.RESOURCE, OperandRole.PREDICATE)
        )
        expected = (
            tuple(
                (index, operand.unit_count, _COUNTER_X)
                for index, operand in enumerate(inputs)
                if operand.unit_count
            )
            if descriptor.schedule_class in capture_classes
            else ()
        )
        assert (
            tuple(
                (lease.attachment_index, lease.unit_count, lease.release_class_id)
                for lease in descriptor.storage_leases
                if lease.kind is StorageLeaseKind.SOURCE_READ
            )
            == expected
        ), descriptor.key


@pytest.mark.parametrize("target", _NATIVE_TARGETS)
def test_memory_results_and_capture_are_independent(target: str) -> None:
    descriptor_set = _with_storage_lease_rows(
        _descriptor_set(target),
        builder_flags=_AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].flags,
    )
    buffer_count = 0
    for descriptor in descriptor_set.descriptors:
        if not (
            InstructionClass.BUFFER_LOAD in descriptor.instruction_classes
            and descriptor.schedule_class == _SCHEDULE_VMEM_LOAD
        ):
            continue
        buffer_count += 1
        results = tuple(
            operand
            for operand in descriptor.operands
            if operand.role is OperandRole.RESULT
        )
        assert len(results) == 1
        result_leases = tuple(
            lease
            for lease in descriptor.storage_leases
            if lease.kind is StorageLeaseKind.RESULT_WRITE
        )
        assert len(result_leases) == 1, descriptor.key
        lease = result_leases[0]
        assert lease.attachment is StorageLeaseAttachment.RESULT
        assert lease.attachment_index == 0
        assert lease.unit_offset == 0
        assert lease.unit_count == results[0].unit_count
        assert lease.release_class_id == _COUNTER_VMEM_LOAD
        source_leases = tuple(
            lease
            for lease in descriptor.storage_leases
            if lease.kind is StorageLeaseKind.SOURCE_READ
        )
        inputs = tuple(
            operand
            for operand in descriptor.operands
            if operand.role
            in (OperandRole.OPERAND, OperandRole.RESOURCE, OperandRole.PREDICATE)
        )
        expected = (
            tuple(
                (index, operand.unit_count, _COUNTER_X)
                for index, operand in enumerate(inputs)
                if operand.unit_count
            )
            if target in _XCNT_TARGETS
            else ()
        )
        assert (
            tuple(
                (lease.attachment_index, lease.unit_count, lease.release_class_id)
                for lease in source_leases
            )
            == expected
        ), descriptor.key
    assert buffer_count in (18, 26, 28)
