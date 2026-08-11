# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared source model for AMDGPU descriptor-derived planning tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from loom.target.arch.amdgpu.descriptors import (
    amdgpu_core_descriptor_set_instruction_names_by_isa_key,
    build_amdgpu_core_descriptor_set_from_specs,
)
from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaFactSource,
    parse_amdgpu_isa_xml_paths_for_instructions,
)
from loom.target.arch.amdgpu.target_info import (
    AmdgpuDescriptorSetInfo,
    sorted_descriptor_set_infos,
)
from loom.target.low_descriptors import DescriptorSet


@dataclass(frozen=True, slots=True)
class AmdgpuPlanningTableInputs:
    """Materialized ISA and descriptor facts shared by planning generators."""

    # Descriptor-set metadata in stable ordinal order.
    descriptor_set_infos: tuple[AmdgpuDescriptorSetInfo, ...]
    # Parsed ISA facts keyed by the build-provided XML source name.
    isa_specs: Mapping[str, AmdgpuIsaFactSource]
    # Materialized descriptor sets keyed by their stable descriptor-set key.
    descriptor_sets_by_key: Mapping[str, DescriptorSet]


def _parse_isa_xml_paths(
    values: Sequence[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        key, separator, path_text = value.partition(":")
        if not separator or not key or not path_text:
            raise ValueError("AMDGPU planning-table --isa-xml entries must be key:path pairs")
        path = Path(path_text)
        existing_path = paths.get(key)
        if existing_path is not None:
            if existing_path != path:
                raise ValueError(f"AMDGPU planning-table ISA XML key '{key}' has conflicting paths '{existing_path}' and '{path}'")
            continue
        paths[key] = path
    return paths


def load_amdgpu_planning_table_inputs(
    isa_xml_arguments: Sequence[str],
    additional_instruction_names_by_isa_key: Mapping[str, Iterable[str]],
) -> AmdgpuPlanningTableInputs:
    """Loads one ISA corpus and materializes every registered descriptor set."""

    descriptor_set_infos = sorted_descriptor_set_infos()
    instruction_names_by_isa_key = {isa_key: set(instruction_names) for isa_key, instruction_names in (amdgpu_core_descriptor_set_instruction_names_by_isa_key(descriptor_set_infos).items())}
    for isa_key, additional_instruction_names in additional_instruction_names_by_isa_key.items():
        instruction_names_by_isa_key.setdefault(isa_key, set()).update(additional_instruction_names)
    isa_specs = parse_amdgpu_isa_xml_paths_for_instructions(
        _parse_isa_xml_paths(isa_xml_arguments),
        instruction_names_by_isa_key,
    )
    descriptor_sets_by_key: dict[str, DescriptorSet] = {}
    for info in descriptor_set_infos:
        descriptor_set = build_amdgpu_core_descriptor_set_from_specs(
            info.generator_target,
            isa_specs,
        )
        if descriptor_set.key != info.key:
            raise ValueError(f"AMDGPU descriptor-set builder '{info.generator_target}' produced '{descriptor_set.key}', expected '{info.key}'")
        descriptor_sets_by_key[info.key] = descriptor_set
    return AmdgpuPlanningTableInputs(
        descriptor_set_infos=descriptor_set_infos,
        isa_specs=MappingProxyType(isa_specs),
        descriptor_sets_by_key=MappingProxyType(descriptor_sets_by_key),
    )
