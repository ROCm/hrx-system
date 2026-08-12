# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared target-family selection for AMDGPU descriptor-derived tables."""

from __future__ import annotations

from dataclasses import dataclass

from loom.target.arch.amdgpu.descriptors import (
    amdgpu_core_descriptor_set_instruction_names_by_isa_key,
)
from loom.target.arch.amdgpu.target_info import (
    AmdgpuDescriptorSetInfo,
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
)


@dataclass(frozen=True, slots=True)
class AmdgpuTargetTableFamily:
    """One storage descriptor set and every view backed by its tables."""

    storage_info: AmdgpuDescriptorSetInfo
    view_infos: tuple[AmdgpuDescriptorSetInfo, ...]

    @property
    def descriptor_set_infos(self) -> tuple[AmdgpuDescriptorSetInfo, ...]:
        return (self.storage_info, *self.view_infos)

    @property
    def generator_targets(self) -> tuple[str, ...]:
        return tuple(info.generator_target for info in self.descriptor_set_infos)


def amdgpu_target_table_family(
    storage_generator_target: str,
) -> AmdgpuTargetTableFamily:
    """Resolves one independently selectable storage-target family."""

    descriptor_set_info = amdgpu_descriptor_set_info_by_generator_target(storage_generator_target)
    storage_info = amdgpu_descriptor_set_storage_info_by_generator_target(storage_generator_target)
    if storage_info != descriptor_set_info:
        raise ValueError(f"AMDGPU table target {storage_generator_target} is a view of storage target {storage_info.generator_target}")
    return AmdgpuTargetTableFamily(
        storage_info=storage_info,
        view_infos=amdgpu_descriptor_set_view_infos_by_storage_generator_target(storage_generator_target),
    )


def amdgpu_target_table_instruction_names_by_isa_key(
    family: AmdgpuTargetTableFamily,
) -> dict[str, set[str]]:
    """Returns the instruction facts needed by every target-table product."""

    instruction_names_by_isa_key = {isa_key: set(instruction_names) for isa_key, instruction_names in amdgpu_core_descriptor_set_instruction_names_by_isa_key(family.descriptor_set_infos).items()}
    for isa_info in family.storage_info.isa_infos:
        instruction_names_by_isa_key[isa_info.isa_xml_key].add("V_NOP")
    return instruction_names_by_isa_key
