# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU register dependency and execution resource timing."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

from loom.target.low_descriptors import (
    Descriptor,
    DescriptorSet,
    EffectKind,
    EventSeparation,
    IssueUse,
    MemorySpace,
    ModelQuality,
    Operand,
    OperandFlag,
    OperandRole,
    Resource,
    ResourceKind,
    ScheduleClass,
    TimingEvent,
)

from .common import (
    _REG_SGPR,
    _SCHEDULE_LDS_ATOMIC,
    _SCHEDULE_LDS_CROSSLANE,
    _SCHEDULE_LDS_LOAD,
    _SCHEDULE_LDS_STORE,
    _SCHEDULE_VALU,
)


def _with_lds_service_timing(
    descriptor_set: DescriptorSet, bits_per_lane_per_cycle: int
) -> DescriptorSet:
    # Independent addresses still contend for the LDS execution pipeline.
    # Model the minimum service of one SIMD issue pass from packet width;
    # bank conflicts, wave64 replay, and other waves can increase this cost.
    # This bandwidth reservation is distinct from result availability and
    # register-source leases. Keep their existing latency/hazard models.
    service_resource = "amdgpu.lds.service"
    parents = {
        row.name: row
        for row in descriptor_set.schedule_classes
        if row.name
        in (
            _SCHEDULE_LDS_LOAD,
            _SCHEDULE_LDS_STORE,
            _SCHEDULE_LDS_ATOMIC,
            _SCHEDULE_LDS_CROSSLANE,
        )
    }
    variants: dict[tuple[str, int], ScheduleClass] = {}
    descriptors: list[Descriptor] = []
    for descriptor in descriptor_set.descriptors:
        parent = parents.get(descriptor.schedule_class)
        if parent is None:
            descriptors.append(descriptor)
            continue
        # An atomic's read and write effects describe the same packet width.
        # Cross-lane packets use the LDS pipeline without a memory footprint.
        width_bits = max(
            (
                effect.width_bits
                for effect in descriptor.effects
                if effect.memory_space == MemorySpace.WORKGROUP
                and effect.kind in (EffectKind.READ, EffectKind.WRITE)
            ),
            default=0,
        )
        cycles = max(
            1,
            (width_bits + bits_per_lane_per_cycle - 1) // bits_per_lane_per_cycle,
        )
        key = (parent.name, cycles)
        if key not in variants:
            variants[key] = replace(
                parent,
                name=f"{parent.name}.service{cycles}",
                issue_uses=(
                    *parent.issue_uses,
                    IssueUse(service_resource, cycles=cycles, units=1),
                ),
            )
        descriptors.append(replace(descriptor, schedule_class=variants[key].name))
    return replace(
        descriptor_set,
        descriptors=tuple(descriptors),
        resources=(
            *descriptor_set.resources,
            Resource(
                service_resource, capacity_per_cycle=1, kind=ResourceKind.PIPELINE
            ),
        ),
        schedule_classes=(*descriptor_set.schedule_classes, *variants.values()),
    )


def _with_valu_sgpr_timing(
    descriptor_set: DescriptorSet, separation_cycles: int
) -> DescriptorSet:
    # Compare masks and lane reads are produced by VALU into scalar registers.
    # Model their availability at operand endpoints so independent instructions
    # can fill the dependency gap. Class latency also controls completion and
    # critical-path priority, which have different scheduling consequences.
    descriptors = tuple(
        _with_valu_sgpr_timing_descriptor(descriptor)
        for descriptor in descriptor_set.descriptors
    )
    return replace(
        descriptor_set,
        descriptors=descriptors,
        timing_events=(
            *descriptor_set.timing_events,
            TimingEvent(_SGPR_READ_EVENT),
            TimingEvent(_VALU_SGPR_WRITE_EVENT),
        ),
        event_separations=(
            *descriptor_set.event_separations,
            EventSeparation(
                _VALU_SGPR_WRITE_EVENT,
                _SGPR_READ_EVENT,
                separation_cycles,
                ModelQuality.ESTIMATED,
            ),
        ),
    )


_SGPR_READ_EVENT = "amdgpu.sgpr.read"
_VALU_SGPR_WRITE_EVENT = "amdgpu.valu.sgpr.write"


@cache
def _with_valu_sgpr_timing_operand(
    descriptor_key: str,
    schedule_class: str,
    operand: Operand,
) -> Operand:
    has_sgpr = any(
        alternative.reg_class == _REG_SGPR for alternative in operand.reg_alts
    )
    reads_sgpr = has_sgpr and (
        operand.role
        in (
            OperandRole.OPERAND,
            OperandRole.OPERAND_RESULT,
            OperandRole.PREDICATE,
            OperandRole.RESOURCE,
        )
        or OperandFlag.STATE_READ in operand.flags
    )
    writes_sgpr = (
        has_sgpr
        and schedule_class == _SCHEDULE_VALU
        and (
            operand.role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT)
            or OperandFlag.STATE_WRITE in operand.flags
        )
    )
    if writes_sgpr and any(
        alternative.reg_class != _REG_SGPR for alternative in operand.reg_alts
    ):
        raise ValueError(
            f"AMDGPU descriptor '{descriptor_key}' result "
            f"'{operand.field_name}' requires distinct SGPR and VGPR forms "
            "to model VALU write timing"
        )
    read_event = _SGPR_READ_EVENT if reads_sgpr else operand.read_event
    write_event = _VALU_SGPR_WRITE_EVENT if writes_sgpr else operand.write_event
    if read_event == operand.read_event and write_event == operand.write_event:
        return operand
    return replace(operand, read_event=read_event, write_event=write_event)


def _with_valu_sgpr_timing_descriptor(descriptor: Descriptor) -> Descriptor:
    operands = tuple(
        _with_valu_sgpr_timing_operand(
            descriptor.key,
            descriptor.schedule_class,
            operand,
        )
        for operand in descriptor.operands
    )
    if all(
        operand is existing
        for operand, existing in zip(operands, descriptor.operands, strict=True)
    ):
        return descriptor
    return replace(descriptor, operands=operands)


def _with_state_read_before_clobber_timing(
    descriptor_set: DescriptorSet,
) -> DescriptorSet:
    """Models architectural state inputs as consumed when a packet issues.

    The shared wave-issue resource keeps packets from the same wave in distinct
    issue slots. A following packet may therefore overwrite architectural state
    without waiting for the reader's unrelated result latency.
    """
    descriptors = tuple(
        _with_state_read_before_clobber_timing_descriptor(descriptor)
        for descriptor in descriptor_set.descriptors
    )
    return replace(
        descriptor_set,
        descriptors=descriptors,
        timing_events=(
            *descriptor_set.timing_events,
            TimingEvent(_STATE_READ_EVENT),
            TimingEvent(_STATE_WRITE_EVENT),
        ),
        event_separations=(
            *descriptor_set.event_separations,
            EventSeparation(
                _STATE_READ_EVENT,
                _STATE_WRITE_EVENT,
                0,
                ModelQuality.EXACT,
            ),
        ),
    )


_STATE_READ_EVENT = "amdgpu.state.read"
_STATE_WRITE_EVENT = "amdgpu.state.write"


@cache
def _with_state_read_before_clobber_timing_operand(operand: Operand) -> Operand:
    read_event = (
        _STATE_READ_EVENT
        if OperandFlag.STATE_READ in operand.flags and operand.read_event is None
        else operand.read_event
    )
    write_event = (
        _STATE_WRITE_EVENT
        if OperandFlag.STATE_WRITE in operand.flags and operand.write_event is None
        else operand.write_event
    )
    if read_event == operand.read_event and write_event == operand.write_event:
        return operand
    return replace(operand, read_event=read_event, write_event=write_event)


def _with_state_read_before_clobber_timing_descriptor(
    descriptor: Descriptor,
) -> Descriptor:
    operands = tuple(
        _with_state_read_before_clobber_timing_operand(operand)
        for operand in descriptor.operands
    )
    if all(
        operand is existing
        for operand, existing in zip(operands, descriptor.operands, strict=True)
    ):
        return descriptor
    return replace(descriptor, operands=operands)
