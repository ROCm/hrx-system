# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final AMDGPU descriptor metadata projection."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    EventSeparation,
    InstructionClass,
    ModelQuality,
    TimingEvent,
)

from .alignment import _with_operand_alignment_row
from .categories import _amdgpu_descriptor_category
from .common import (
    _SCHEDULE_CLUSTER_LOAD_LDS,
    _SCHEDULE_FLAT_ATOMIC_NO_RETURN,
    _SCHEDULE_FLAT_ATOMIC_RETURN,
    _SCHEDULE_FLAT_LOAD,
    _SCHEDULE_FLAT_STORE,
    _SCHEDULE_LDS_ATOMIC,
    _SCHEDULE_LDS_CROSSLANE,
    _SCHEDULE_LDS_LOAD,
    _SCHEDULE_LDS_STORE,
    _SCHEDULE_MATRIX,
    _SCHEDULE_MFMA,
    _SCHEDULE_MFMA_QUALIFIED_PREFIX,
    _SCHEDULE_SMEM_LOAD,
    _SCHEDULE_SMEM_STORE,
    _SCHEDULE_SWMMAC,
    _SCHEDULE_TENSOR_LOAD_LDS,
    _SCHEDULE_VMEM_ATOMIC_NO_RETURN,
    _SCHEDULE_VMEM_ATOMIC_RETURN,
    _SCHEDULE_VMEM_LOAD,
    _SCHEDULE_VMEM_LOAD_LDS,
    _SCHEDULE_VMEM_STORE,
    _SCHEDULE_WMMA,
    _SCHEDULE_WMMA_SCALE,
)
from .timing import (
    _SGPR_READ_EVENT,
    _STATE_READ_EVENT,
    _STATE_WRITE_EVENT,
    _VALU_SGPR_WRITE_EVENT,
    _with_state_read_before_clobber_timing_operand,
    _with_valu_sgpr_timing_operand,
)

_AMDGPU_SCHEDULE_INSTRUCTION_CLASSES = {
    _SCHEDULE_SMEM_LOAD: (InstructionClass.SCALAR_MEMORY,),
    _SCHEDULE_SMEM_STORE: (InstructionClass.SCALAR_MEMORY,),
    _SCHEDULE_VMEM_LOAD: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_VMEM_LOAD_LDS: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
        InstructionClass.LDSDMA,
    ),
    _SCHEDULE_VMEM_STORE: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_VMEM_ATOMIC_RETURN: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_VMEM_ATOMIC_NO_RETURN: (InstructionClass.GLOBAL_MEMORY,),
    _SCHEDULE_FLAT_LOAD: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
    ),
    _SCHEDULE_FLAT_STORE: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
    ),
    _SCHEDULE_FLAT_ATOMIC_RETURN: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
    ),
    _SCHEDULE_FLAT_ATOMIC_NO_RETURN: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
    ),
    _SCHEDULE_LDS_LOAD: (InstructionClass.LOCAL_MEMORY,),
    _SCHEDULE_LDS_STORE: (InstructionClass.LOCAL_MEMORY,),
    _SCHEDULE_LDS_ATOMIC: (InstructionClass.LOCAL_MEMORY,),
    _SCHEDULE_LDS_CROSSLANE: (InstructionClass.LOCAL_MEMORY,),
    _SCHEDULE_TENSOR_LOAD_LDS: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
        InstructionClass.LDSDMA,
    ),
    _SCHEDULE_CLUSTER_LOAD_LDS: (
        InstructionClass.GLOBAL_MEMORY,
        InstructionClass.LOCAL_MEMORY,
        InstructionClass.LDSDMA,
    ),
    _SCHEDULE_MFMA: (InstructionClass.MFMA,),
    _SCHEDULE_WMMA: (InstructionClass.WMMA,),
    _SCHEDULE_WMMA_SCALE: (InstructionClass.WMMA,),
    _SCHEDULE_SWMMAC: (InstructionClass.SWMMAC,),
}

_AMDGPU_KEY_INSTRUCTION_CLASSES = (
    ("amdgpu.global_load_", InstructionClass.GLOBAL_LOAD),
    ("amdgpu.global_store_", InstructionClass.GLOBAL_STORE),
    ("amdgpu.buffer_load_", InstructionClass.BUFFER_LOAD),
    ("amdgpu.buffer_store_", InstructionClass.BUFFER_STORE),
    ("amdgpu.flat_load_", InstructionClass.FLAT_MEMORY),
    ("amdgpu.flat_store_", InstructionClass.FLAT_MEMORY),
    ("amdgpu.flat_atomic_", InstructionClass.FLAT_MEMORY),
)


@cache
def _project_instruction_classes(
    key: str,
    semantic_tag: str | None,
    schedule_class: str,
    existing_instruction_classes: tuple[InstructionClass, ...],
) -> tuple[InstructionClass, ...]:
    instruction_classes = set(existing_instruction_classes)
    schedule_instruction_classes = _AMDGPU_SCHEDULE_INSTRUCTION_CLASSES.get(
        schedule_class, ()
    )
    if schedule_class.startswith(_SCHEDULE_MFMA_QUALIFIED_PREFIX):
        schedule_instruction_classes = (InstructionClass.MFMA,)
    elif schedule_class.startswith(f"{_SCHEDULE_MATRIX}."):
        semantic_tag = semantic_tag or ""
        if semantic_tag.startswith("matrix.wmma."):
            schedule_instruction_classes = (InstructionClass.WMMA,)
        elif semantic_tag.startswith("matrix.swmmac."):
            schedule_instruction_classes = (InstructionClass.SWMMAC,)
    instruction_classes.update(schedule_instruction_classes)
    semantic_tag = semantic_tag or ""
    if semantic_tag.startswith(("memory.stack.", "memory.private.")):
        instruction_classes.discard(InstructionClass.GLOBAL_MEMORY)
    for key_prefix, instruction_class in _AMDGPU_KEY_INSTRUCTION_CLASSES:
        if key.startswith(key_prefix):
            instruction_classes.add(instruction_class)
            break
    return tuple(
        instruction_class
        for instruction_class in InstructionClass
        if instruction_class in instruction_classes
    )


def _with_instruction_classes_descriptor(descriptor: Descriptor) -> Descriptor:
    instruction_classes = _project_instruction_classes(
        descriptor.key,
        descriptor.semantic_tag,
        descriptor.schedule_class,
        descriptor.instruction_classes,
    )
    if instruction_classes == descriptor.instruction_classes:
        return descriptor
    return replace(descriptor, instruction_classes=instruction_classes)


def _with_instruction_classes(descriptor_set: DescriptorSet) -> DescriptorSet:
    descriptors = tuple(
        _with_instruction_classes_descriptor(descriptor)
        for descriptor in descriptor_set.descriptors
    )
    if all(
        descriptor is existing
        for descriptor, existing in zip(
            descriptors,
            descriptor_set.descriptors,
            strict=True,
        )
    ):
        return descriptor_set
    return replace(descriptor_set, descriptors=descriptors)


def _finalize_amdgpu_descriptor(
    descriptor: Descriptor, *, with_valu_sgpr_timing: bool
) -> Descriptor:
    operands = []
    for operand in descriptor.operands:
        operand = _with_operand_alignment_row(operand)
        if with_valu_sgpr_timing:
            operand = _with_valu_sgpr_timing_operand(
                descriptor.key,
                descriptor.schedule_class,
                operand,
            )
        operands.append(_with_state_read_before_clobber_timing_operand(operand))
    projected_operands = tuple(operands)
    category = _amdgpu_descriptor_category(descriptor)
    instruction_classes = _project_instruction_classes(
        descriptor.key,
        descriptor.semantic_tag,
        descriptor.schedule_class,
        descriptor.instruction_classes,
    )
    if (
        projected_operands == descriptor.operands
        and category == descriptor.category
        and instruction_classes == descriptor.instruction_classes
    ):
        return descriptor
    return replace(
        descriptor,
        operands=projected_operands,
        category=category,
        instruction_classes=instruction_classes,
    )


def _finalize_amdgpu_descriptors(
    descriptor_set: DescriptorSet, *, valu_sgpr_separation_cycles: int
) -> DescriptorSet:
    with_valu_sgpr_timing = valu_sgpr_separation_cycles != 0
    descriptors = tuple(
        _finalize_amdgpu_descriptor(
            descriptor,
            with_valu_sgpr_timing=with_valu_sgpr_timing,
        )
        for descriptor in descriptor_set.descriptors
    )
    timing_events = descriptor_set.timing_events
    event_separations = descriptor_set.event_separations
    if with_valu_sgpr_timing:
        timing_events = (
            *timing_events,
            TimingEvent(_SGPR_READ_EVENT),
            TimingEvent(_VALU_SGPR_WRITE_EVENT),
        )
        event_separations = (
            *event_separations,
            EventSeparation(
                _VALU_SGPR_WRITE_EVENT,
                _SGPR_READ_EVENT,
                valu_sgpr_separation_cycles,
                ModelQuality.ESTIMATED,
            ),
        )
    return replace(
        descriptor_set,
        descriptors=descriptors,
        timing_events=(
            *timing_events,
            TimingEvent(_STATE_READ_EVENT),
            TimingEvent(_STATE_WRITE_EVENT),
        ),
        event_separations=(
            *event_separations,
            EventSeparation(
                _STATE_READ_EVENT,
                _STATE_WRITE_EVENT,
                0,
                ModelQuality.EXACT,
            ),
        ),
    )
