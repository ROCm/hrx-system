# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU CDNA descriptor-set base data."""

from __future__ import annotations

from .common import *

_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.cdna4.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=106,
            full_register_part_mask=_REG_PART_SGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_AGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    ),
    register_parts=_AMDGPU_REGISTER_PARTS,
    resources=(
        *_common_scalar_vector_memory_resources(),
        Resource(_RESOURCE_MFMA, capacity_per_cycle=1, kind=ResourceKind.MATRIX),
        Resource(_RESOURCE_CONTROL, capacity_per_cycle=1, kind=ResourceKind.CONTROL),
    ),
    schedule_classes=(
        *_common_scalar_vector_memory_schedule_classes(
            smem_load_hazards=_SMEM_WAIT_HAZARDS,
            smem_store_hazards=_SMEM_WAIT_HAZARDS,
            vmem_load_hazards=_VMEM_LOAD_WAIT_HAZARDS,
            vmem_store_hazards=_VMEM_STORE_WAIT_HAZARDS,
            lds_load_hazards=_LDS_WAIT_HAZARDS,
            lds_store_hazards=_LDS_WAIT_HAZARDS,
            lds_atomic_hazards=_LDS_WAIT_HAZARDS,
            lds_crosslane_hazards=_LDS_WAIT_HAZARDS,
        ),
        ScheduleClass(
            _SCHEDULE_VMEM_LOAD_LDS,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=16,
            issue_uses=(
                IssueUse(_RESOURCE_VMEM_LOAD, cycles=1, units=1),
                IssueUse(_RESOURCE_LDS_STORE, cycles=1, units=1),
            ),
            hazards=_VMEM_LOAD_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.MAY_LOAD, ScheduleClassFlag.MAY_STORE),
            model_quality=ModelQuality.FALLBACK,
        ),
        ScheduleClass(
            _SCHEDULE_MFMA,
            latency_kind=LatencyKind.ESTIMATE,
            latency_cycles=32,
            issue_uses=(IssueUse(_RESOURCE_MFMA, cycles=1, units=1),),
            hazards=_matrix_hazards(_RESOURCE_MFMA),
            model_quality=ModelQuality.ESTIMATED,
        ),
        ScheduleClass(
            _SCHEDULE_WAIT_MEMORY,
            latency_kind=LatencyKind.VARIABLE,
            latency_cycles=1,
            issue_uses=(IssueUse(_RESOURCE_CONTROL, cycles=1, units=1),),
            hazards=_GFX950_MEMORY_WAIT_HAZARDS,
            flags=(ScheduleClassFlag.CONTROL,),
            model_quality=ModelQuality.FALLBACK,
        ),
    ),
)


_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE = _amdgpu_core_descriptor_set(
    key="amdgpu.cdna3.core",
    reg_classes=(
        RegClass(
            _REG_SGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=102,
            full_register_part_mask=_REG_PART_SGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_VGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
            full_register_part_mask=_REG_PART_VGPR_FULL32_MASK,
        ),
        RegClass(
            _REG_SCC,
            1,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_EXEC,
            64,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
        RegClass(
            _REG_AGPR,
            32,
            SpillSlotSpace.SCRATCH,
            flags=(RegClassFlag.PHYSICAL,),
            allocatable_count=256,
        ),
        RegClass(
            _REG_M0,
            32,
            SpillSlotSpace.PRIVATE,
            flags=(RegClassFlag.PHYSICAL, RegClassFlag.UNSPILLABLE),
            allocatable_count=1,
        ),
    ),
    register_parts=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.register_parts,
    resources=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.resources,
    schedule_classes=_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE.schedule_classes,
)


__all__ = (
    "_AMDGPU_CDNA3_CORE_DESCRIPTOR_SET_BASE",
    "_AMDGPU_CDNA4_CORE_DESCRIPTOR_SET_BASE",
)
