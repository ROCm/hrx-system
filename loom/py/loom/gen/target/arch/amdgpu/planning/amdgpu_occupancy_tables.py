# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU occupancy model rows -> compact C tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    _amdgpu_core_descriptor_set_bases,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
    AmdgpuOccupancyModelInfo,
    AmdgpuProcessorInfo,
    amdgpu_descriptor_set_ordinal,
    amdgpu_processor_occupancy_model,
    amdgpu_processor_ordinal,
    kernel_descriptor_profile_supports_wavefront_size,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
)
from loom.target.low_descriptors import RegClass, RegClassFlag  # noqa: E402


def _c_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _c_identifier(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char.isalnum():
            chars.append(char.upper())
        else:
            chars.append("_")
    return "".join(chars).strip("_")


@dataclass(frozen=True, slots=True)
class _AmdgpuOccupancyModelRow:
    descriptor_set_key: str
    wave_size: int
    model: AmdgpuOccupancyModelInfo
    processors: tuple[AmdgpuProcessorInfo, ...]


def _model_symbol_suffix(row: _AmdgpuOccupancyModelRow) -> str:
    descriptor_set_key = row.descriptor_set_key
    prefix = "amdgpu."
    suffix = ".core"
    if not descriptor_set_key.startswith(prefix) or not descriptor_set_key.endswith(suffix):
        raise ValueError(f"AMDGPU occupancy descriptor-set key '{descriptor_set_key}' must be a core key")
    descriptor_suffix = descriptor_set_key.removeprefix(prefix).removesuffix(suffix)
    return _c_identifier(f"{descriptor_suffix}_{row.processors[0].processor}_wave{row.wave_size}")


def _model_c_suffix(row: _AmdgpuOccupancyModelRow) -> str:
    return _camel_c_suffix(_model_symbol_suffix(row))


def _camel_c_suffix(value: str) -> str:
    return "".join(part[:1].upper() + part[1:].lower() for part in value.split("_") if part)


def _u16_expr(value: int) -> str:
    return f"UINT16_C({value})"


def _u32_expr(value: int) -> str:
    return f"UINT32_C({value})"


def _descriptor_reg_class_ids(descriptor_set_key: str) -> dict[str, int]:
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        if descriptor_set.key != descriptor_set_key:
            continue
        return {reg_class.name: index for index, reg_class in enumerate(descriptor_set.reg_classes)}
    raise ValueError(f"AMDGPU occupancy model references unknown descriptor set '{descriptor_set_key}'")


def _descriptor_reg_classes(descriptor_set_key: str) -> tuple[RegClass, ...]:
    for descriptor_set in _amdgpu_core_descriptor_set_bases():
        if descriptor_set.key == descriptor_set_key:
            return descriptor_set.reg_classes
    raise ValueError(f"AMDGPU occupancy model references unknown descriptor set '{descriptor_set_key}'")


def _round_up(value: int, multiple: int) -> int:
    if value == 0 or multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _wave_limit(
    pool_units: int,
    allocation_granularity: int,
    max_waves_per_simd: int,
    allocated_units: int,
) -> int:
    if allocated_units == 0:
        return max_waves_per_simd
    rounded_units = _round_up(allocated_units, allocation_granularity)
    if rounded_units == 0:
        return 0
    return min(pool_units // rounded_units, max_waves_per_simd)


def _pressure_cliffs(
    pool_units: int,
    allocation_granularity: int,
    max_waves_per_simd: int,
) -> tuple[tuple[int, int, int], ...]:
    previous_wave_limit = max_waves_per_simd
    cliffs: list[tuple[int, int, int]] = []
    while previous_wave_limit != 0:
        first_lower_rounded_unit = (pool_units // previous_wave_limit) + 1
        next_rounded_units = _round_up(
            first_lower_rounded_unit,
            allocation_granularity,
        )
        candidate = max(
            1,
            next_rounded_units - allocation_granularity + 1,
        )
        candidate_wave_limit = _wave_limit(
            pool_units,
            allocation_granularity,
            max_waves_per_simd,
            candidate,
        )
        if candidate_wave_limit >= previous_wave_limit:
            raise ValueError("AMDGPU occupancy cliff generation made no progress")
        _validate_u32(candidate, "AMDGPU occupancy pressure cliff")
        cliffs.append((candidate, previous_wave_limit, candidate_wave_limit))
        previous_wave_limit = candidate_wave_limit
    return tuple(cliffs)


def _validate_u32(value: int, description: str) -> None:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"{description} does not fit uint32")


def _materialize_models(
    processors: Sequence[AmdgpuProcessorInfo],
) -> tuple[_AmdgpuOccupancyModelRow, ...]:
    grouped_processors: dict[tuple[str, int, AmdgpuOccupancyModelInfo], list[AmdgpuProcessorInfo]] = {}
    for processor in processors:
        for wave_size in (32, 64):
            model = amdgpu_processor_occupancy_model(processor, wave_size)
            if model is None:
                continue
            key = (processor.descriptor_set.key, wave_size, model)
            grouped_processors.setdefault(key, []).append(processor)
    return tuple(
        _AmdgpuOccupancyModelRow(
            descriptor_set_key=descriptor_set_key,
            wave_size=wave_size,
            model=model,
            processors=tuple(model_processors),
        )
        for (
            descriptor_set_key,
            wave_size,
            model,
        ), model_processors in grouped_processors.items()
    )


def _validate_models(
    models: Sequence[_AmdgpuOccupancyModelRow],
    processors: Sequence[AmdgpuProcessorInfo],
) -> None:
    descriptor_set_keys = {info.key for info in sorted_descriptor_set_infos()}
    expected_processor_waves: set[tuple[str, int]] = set()
    for processor in processors:
        for wave_size in (32, 64):
            supports_wave = kernel_descriptor_profile_supports_wavefront_size(processor.kernel_descriptor.profile, wave_size)
            model = amdgpu_processor_occupancy_model(processor, wave_size)
            if not processor.descriptor_set.key:
                if model is not None:
                    raise ValueError(f"AMDGPU processor {processor.processor} has an occupancy model without a descriptor set")
                continue
            if supports_wave:
                expected_processor_waves.add((processor.processor, wave_size))
                if model is None:
                    raise ValueError(f"AMDGPU processor {processor.processor} is missing its wave{wave_size} occupancy model")
            elif model is not None:
                raise ValueError(f"AMDGPU processor {processor.processor} defines an unsupported wave{wave_size} occupancy model")

    covered_processor_waves: set[tuple[str, int]] = set()
    for model_row in models:
        model = model_row.model
        descriptor_set_key = model_row.descriptor_set_key
        wave_size = model_row.wave_size
        if descriptor_set_key not in descriptor_set_keys:
            raise ValueError(f"AMDGPU occupancy model references unknown descriptor set '{descriptor_set_key}'")
        if amdgpu_descriptor_set_ordinal(descriptor_set_key) >= (AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE):
            raise ValueError("AMDGPU occupancy descriptor-set ordinal overflows")
        if wave_size not in (32, 64):
            raise ValueError(f"AMDGPU occupancy wave size for {descriptor_set_key} must be 32 or 64")
        if not model_row.processors:
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} wave{wave_size} has no processors")
        for processor in model_row.processors:
            processor_wave = (processor.processor, wave_size)
            if processor_wave in covered_processor_waves:
                raise ValueError(f"AMDGPU occupancy model duplicates {processor.processor} wave{wave_size}")
            if processor.descriptor_set.key != descriptor_set_key:
                raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} references processor {processor.processor} from {processor.descriptor_set.key}")
            if amdgpu_processor_occupancy_model(processor, wave_size) != model:
                raise ValueError(f"AMDGPU occupancy model for {processor.processor} wave{wave_size} disagrees with processor facts")
            if processor.limits.max_workgroup_storage_bytes > model.domain.local_memory_bytes:
                raise ValueError(f"AMDGPU processor {processor.processor} permits more workgroup storage than its occupancy domain")
            covered_processor_waves.add(processor_wave)
        if model.max_waves_per_simd <= 0:
            raise ValueError(f"AMDGPU occupancy max waves for {descriptor_set_key} must be positive")
        _validate_u32(
            model.max_waves_per_simd,
            f"AMDGPU occupancy max waves for {descriptor_set_key}",
        )
        domain = model.domain
        if domain.simd_count <= 0:
            raise ValueError(f"AMDGPU occupancy SIMD count for {descriptor_set_key} must be positive")
        if domain.local_memory_bytes <= 0:
            raise ValueError(f"AMDGPU occupancy local memory for {descriptor_set_key} must be positive")
        if domain.local_memory_allocation_granularity <= 0:
            raise ValueError(f"AMDGPU occupancy local-memory granularity for {descriptor_set_key} must be positive")
        if domain.local_memory_bytes % domain.local_memory_allocation_granularity != 0:
            raise ValueError(f"AMDGPU occupancy local-memory granularity for {descriptor_set_key} must divide its capacity")
        if domain.max_barrier_workgroup_count <= 0:
            raise ValueError(f"AMDGPU occupancy barrier count for {descriptor_set_key} must be positive")
        _validate_u32(
            domain.simd_count,
            f"AMDGPU occupancy SIMD count for {descriptor_set_key}",
        )
        _validate_u32(
            domain.local_memory_bytes,
            f"AMDGPU occupancy local memory for {descriptor_set_key}",
        )
        _validate_u32(
            domain.local_memory_allocation_granularity,
            f"AMDGPU occupancy local-memory granularity for {descriptor_set_key}",
        )
        _validate_u32(
            domain.max_barrier_workgroup_count,
            f"AMDGPU occupancy barrier count for {descriptor_set_key}",
        )
        wave_slots = model.max_waves_per_simd * domain.simd_count
        _validate_u32(
            wave_slots,
            f"AMDGPU occupancy wave slots for {descriptor_set_key}",
        )
        if domain.max_barrier_workgroup_count > wave_slots:
            raise ValueError(f"AMDGPU occupancy barrier count for {descriptor_set_key} exceeds its wave slots")
        register_classes = [row.register_class for row in model.register_classes]
        if len(register_classes) > 0xFFFF:
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} has too many register classes")
        if len(register_classes) != len(set(register_classes)):
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} has duplicate register classes")
        missing_base_classes = sorted({"amdgpu.sgpr", "amdgpu.vgpr"} - set(register_classes))
        if missing_base_classes:
            missing = ", ".join(missing_base_classes)
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} is missing base register classes: {missing}")
        descriptor_reg_classes = _descriptor_reg_class_ids(descriptor_set_key)
        descriptor_reg_class_rows = {row.name: row for row in _descriptor_reg_classes(descriptor_set_key)}
        if len(descriptor_reg_class_rows) > 0xFFFF:
            raise ValueError(f"AMDGPU descriptor set {descriptor_set_key} has too many register classes for pressure-resource indexes")
        pressure_cliff_count = 0
        for row in model.register_classes:
            if row.register_class not in descriptor_reg_classes:
                raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} references unknown register class {row.register_class}")
            if row.pool_units <= 0:
                raise ValueError(f"AMDGPU occupancy pool for {row.register_class} must be positive")
            if row.allocation_granularity <= 0:
                raise ValueError(f"AMDGPU occupancy granularity for {row.register_class} must be positive")
            _validate_u32(row.pool_units, f"AMDGPU occupancy pool for {row.register_class}")
            _validate_u32(row.allocation_granularity, f"AMDGPU occupancy granularity for {row.register_class}")
            descriptor_reg_class = descriptor_reg_class_rows[row.register_class]
            if RegClassFlag.PHYSICAL not in descriptor_reg_class.flags:
                raise ValueError(f"AMDGPU occupancy register class {row.register_class} must use physical locations")
            if descriptor_reg_class.allocatable_count == 0:
                raise ValueError(f"AMDGPU occupancy register class {row.register_class} must have a finite physical capacity")
            maximum_allocated_units = _round_up(
                descriptor_reg_class.allocatable_count,
                row.allocation_granularity,
            )
            _validate_u32(
                maximum_allocated_units,
                f"AMDGPU occupancy register class {row.register_class} maximum rounded allocation",
            )
            if row.limits_occupancy and maximum_allocated_units > row.pool_units:
                raise ValueError(f"AMDGPU occupancy register class {row.register_class} permits a legal allocation with zero residency")
            if row.limits_occupancy:
                pressure_cliff_count += len(
                    _pressure_cliffs(
                        row.pool_units,
                        row.allocation_granularity,
                        model.max_waves_per_simd,
                    )
                )
            if pressure_cliff_count > 0xFFFFFFFF:
                raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} has too many register-class pressure cliffs")
        modeled_register_classes = set(register_classes)
        missing_spillable_classes = sorted(
            reg_class.name for reg_class in _descriptor_reg_classes(descriptor_set_key) if RegClassFlag.UNSPILLABLE not in reg_class.flags and reg_class.name not in modeled_register_classes
        )
        if missing_spillable_classes:
            missing = ", ".join(missing_spillable_classes)
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} is missing spillable descriptor register classes: {missing}")
        resources = [row.resource for row in model.resources]
        if len(resources) > 0xFFFF:
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} has too many pressure resources")
        if len(resources) != len(set(resources)):
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} has duplicate resources")
        resource_member_count = 0
        resource_cliff_count = 0
        for resource in model.resources:
            if not resource.resource:
                raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} has an empty resource name")
            if resource.pool_units <= 0:
                raise ValueError(f"AMDGPU occupancy resource pool for {resource.resource} must be positive")
            if resource.allocation_granularity <= 0:
                raise ValueError(f"AMDGPU occupancy resource granularity for {resource.resource} must be positive")
            _validate_u32(resource.pool_units, f"AMDGPU occupancy resource pool for {resource.resource}")
            _validate_u32(resource.allocation_granularity, f"AMDGPU occupancy resource granularity for {resource.resource}")
            if not resource.members:
                raise ValueError(f"AMDGPU occupancy resource {resource.resource} must have members")
            member_register_classes = [member.register_class for member in resource.members]
            if len(member_register_classes) != len(set(member_register_classes)):
                raise ValueError(f"AMDGPU occupancy resource {resource.resource} has duplicate members")
            maximum_resource_units = 0
            for member in resource.members:
                if member.register_class not in register_classes:
                    raise ValueError(f"AMDGPU occupancy resource {resource.resource} references unknown register class {member.register_class}")
                if member.contribution_granularity <= 0:
                    raise ValueError(f"AMDGPU occupancy resource {resource.resource} member {member.register_class} granularity must be positive")
                _validate_u32(
                    member.contribution_granularity,
                    f"AMDGPU occupancy resource {resource.resource} member {member.register_class} granularity",
                )
                member_capacity = descriptor_reg_class_rows[member.register_class].allocatable_count
                if member_capacity == 0:
                    raise ValueError(f"AMDGPU occupancy resource {resource.resource} member {member.register_class} must have a finite physical capacity")
                maximum_resource_units += _round_up(member_capacity, member.contribution_granularity)
                _validate_u32(
                    maximum_resource_units,
                    f"AMDGPU occupancy resource {resource.resource} maximum member contribution",
                )
            if maximum_resource_units > resource.pool_units:
                raise ValueError(f"AMDGPU occupancy resource {resource.resource} permits independently legal member allocations with zero residency")
            resource_member_count += len(resource.members)
            resource_cliff_count += len(
                _pressure_cliffs(
                    resource.pool_units,
                    resource.allocation_granularity,
                    model.max_waves_per_simd,
                )
            )
        if resource_member_count > 0xFFFF:
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} has too many pressure-resource members")
        if resource_cliff_count > 0xFFFF:
            raise ValueError(f"AMDGPU occupancy model for {descriptor_set_key} has too many pressure-resource cliffs")

    missing_processor_waves = sorted(expected_processor_waves - covered_processor_waves)
    extra_processor_waves = sorted(covered_processor_waves - expected_processor_waves)
    if missing_processor_waves or extra_processor_waves:
        details: list[str] = []
        if missing_processor_waves:
            details.append("missing " + ", ".join(f"{processor}/wave{wave_size}" for processor, wave_size in missing_processor_waves))
        if extra_processor_waves:
            details.append("unexpected " + ", ".join(f"{processor}/wave{wave_size}" for processor, wave_size in extra_processor_waves))
        raise ValueError("AMDGPU occupancy models do not cover processor wave modes: " + "; ".join(details))


def _emit_source(models: Sequence[_AmdgpuOccupancyModelRow]) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.planning.amdgpu_occupancy_tables"),
        "",
        '#include "loom/target/arch/amdgpu/planning/occupancy_model.h"',
        "",
        "// clang-format off",
    ]
    for model_row in models:
        model = model_row.model
        suffix = _model_c_suffix(model_row)
        descriptor_reg_classes = _descriptor_reg_class_ids(model_row.descriptor_set_key)
        descriptor_reg_class_rows = _descriptor_reg_classes(model_row.descriptor_set_key)
        register_class_indices = {row.register_class: index for index, row in enumerate(model.register_classes)}
        pressure_cliffs_by_register_class = {
            row.register_class: (
                _pressure_cliffs(
                    row.pool_units,
                    row.allocation_granularity,
                    model.max_waves_per_simd,
                )
                if row.limits_occupancy
                else ()
            )
            for row in model.register_classes
        }
        pressure_cliff_rows: list[tuple[int, int, int, int]] = []
        pressure_cliff_ranges: dict[str, tuple[int, int]] = {}
        for descriptor_reg_class in descriptor_reg_class_rows:
            cliffs = pressure_cliffs_by_register_class.get(descriptor_reg_class.name, ())
            pressure_cliff_ranges[descriptor_reg_class.name] = (len(pressure_cliff_rows), len(cliffs))
            pressure_cliff_rows.extend((descriptor_reg_classes[descriptor_reg_class.name], cliff_units, tier_before, tier_after) for cliff_units, tier_before, tier_after in cliffs)
        if pressure_cliff_rows:
            lines.extend(
                [
                    "",
                    f"static const loom_target_residency_cliff_t kAmdgpu{suffix}PressureCliffs[] = {{",
                ]
            )
            for descriptor_reg_class_id, cliff_units, tier_before, tier_after in pressure_cliff_rows:
                lines.extend(
                    [
                        "  {",
                        f"    .resource_id = {_u16_expr(descriptor_reg_class_id)},",
                        f"    .cliff_units = {_u32_expr(cliff_units)},",
                        f"    .tier_before = {_u32_expr(tier_before)},",
                        f"    .tier_after = {_u32_expr(tier_after)},",
                        "  },",
                    ]
                )
            lines.append("};")
            pressure_cliffs_initializer = f"kAmdgpu{suffix}PressureCliffs"
            pressure_cliff_count_initializer = f"IREE_ARRAYSIZE(kAmdgpu{suffix}PressureCliffs)"
        else:
            pressure_cliffs_initializer = "NULL"
            pressure_cliff_count_initializer = "0"
        lines.extend(
            [
                "",
                f"static const loom_target_residency_cliff_range_t kAmdgpu{suffix}PressureCliffRanges[] = {{",
            ]
        )
        for descriptor_reg_class in descriptor_reg_class_rows:
            start, count = pressure_cliff_ranges.get(descriptor_reg_class.name, (0, 0))
            descriptor_reg_class_id = descriptor_reg_classes[descriptor_reg_class.name]
            lines.append(f"  [{_u16_expr(descriptor_reg_class_id)}] = {{.start = {_u32_expr(start)}, .count = {_u32_expr(count)}}},")
        lines.append("};")
        lines.extend(
            [
                "",
                f"static const iree_string_view_t kAmdgpu{suffix}ResidencyDirectResourceNames[] = {{",
            ]
        )
        for descriptor_reg_class in descriptor_reg_class_rows:
            descriptor_reg_class_id = descriptor_reg_classes[descriptor_reg_class.name]
            lines.append(f'  [{_u16_expr(descriptor_reg_class_id)}] = IREE_SVL("{_c_string_literal(descriptor_reg_class.name)}"),')
        lines.append("};")
        lines.extend(
            [
                "",
                f"static const loom_amdgpu_occupancy_register_class_model_t kAmdgpu{suffix}RegisterClasses[] = {{",
            ]
        )
        for row in model.register_classes:
            lines.extend(
                [
                    "  {",
                    f'    .register_class = IREE_SVL("{_c_string_literal(row.register_class)}"),',
                    f"    .descriptor_reg_class_id = {_u16_expr(descriptor_reg_classes[row.register_class])},",
                    f"    .pool_units = {_u32_expr(row.pool_units)},",
                    f"    .allocation_granularity = {_u32_expr(row.allocation_granularity)},",
                    f"    .limits_occupancy = {'true' if row.limits_occupancy else 'false'},",
                    "  },",
                ]
            )
        lines.append("};")
        lines.extend(
            [
                "",
                f"static const uint16_t kAmdgpu{suffix}RegisterClassIndexByDescriptorRegClassId[] = {{",
            ]
        )
        for descriptor_reg_class in descriptor_reg_class_rows:
            register_class_index = register_class_indices.get(descriptor_reg_class.name)
            index_expr = "UINT16_MAX" if register_class_index is None else _u16_expr(register_class_index)
            lines.append(f"  [{_u16_expr(descriptor_reg_classes[descriptor_reg_class.name])}] = {index_expr},")
        lines.append("};")

        resource_member_rows: list[tuple[int, int, int]] = []
        resource_cliff_rows: list[tuple[int, int, int, int]] = []
        resource_rows: list[tuple[str, int, int, int, int, int, int]] = []
        resource_member_indices_by_register_class: dict[str, list[int]] = {}
        for resource_id, resource in enumerate(model.resources):
            member_start = len(resource_member_rows)
            for member in resource.members:
                member_index = len(resource_member_rows)
                resource_member_rows.append(
                    (
                        resource_id,
                        descriptor_reg_classes[member.register_class],
                        member.contribution_granularity,
                    )
                )
                resource_member_indices_by_register_class.setdefault(member.register_class, []).append(member_index)
            cliff_start = len(resource_cliff_rows)
            resource_cliffs = _pressure_cliffs(
                resource.pool_units,
                resource.allocation_granularity,
                model.max_waves_per_simd,
            )
            resource_cliff_rows.extend((resource_id, cliff_units, tier_before, tier_after) for cliff_units, tier_before, tier_after in resource_cliffs)
            resource_rows.append(
                (
                    resource.resource,
                    resource.pool_units,
                    resource.allocation_granularity,
                    member_start,
                    len(resource.members),
                    cliff_start,
                    len(resource_cliffs),
                )
            )

        resource_member_index_rows: list[int] = []
        resource_member_ranges: dict[str, tuple[int, int]] = {}
        for descriptor_reg_class in descriptor_reg_class_rows:
            member_indices = resource_member_indices_by_register_class.get(descriptor_reg_class.name, [])
            resource_member_ranges[descriptor_reg_class.name] = (
                len(resource_member_index_rows),
                len(member_indices),
            )
            resource_member_index_rows.extend(member_indices)

        if resource_rows:
            lines.extend(
                [
                    "",
                    f"static const loom_target_residency_derived_member_t kAmdgpu{suffix}PressureResourceMembers[] = {{",
                ]
            )
            for resource_id, descriptor_reg_class_id, contribution_granularity in resource_member_rows:
                lines.extend(
                    [
                        "  {",
                        f"    .resource_id = {_u16_expr(resource_id)},",
                        f"    .direct_resource_id = {_u16_expr(descriptor_reg_class_id)},",
                        f"    .contribution_granularity = {_u32_expr(contribution_granularity)},",
                        "  },",
                    ]
                )
            lines.append("};")
            lines.extend(
                [
                    "",
                    f"static const loom_target_residency_cliff_t kAmdgpu{suffix}PressureResourceCliffs[] = {{",
                ]
            )
            for resource_id, cliff_units, tier_before, tier_after in resource_cliff_rows:
                lines.extend(
                    [
                        "  {",
                        f"    .resource_id = {_u16_expr(resource_id)},",
                        f"    .cliff_units = {_u32_expr(cliff_units)},",
                        f"    .tier_before = {_u32_expr(tier_before)},",
                        f"    .tier_after = {_u32_expr(tier_after)},",
                        "  },",
                    ]
                )
            lines.append("};")
            lines.extend(
                [
                    "",
                    f"static const loom_target_residency_derived_resource_t kAmdgpu{suffix}PressureResources[] = {{",
                ]
            )
            for resource_name, pool_units, allocation_granularity, member_start, member_count, cliff_start, cliff_count in resource_rows:
                lines.extend(
                    [
                        "  {",
                        f'    .name = IREE_SVL("{_c_string_literal(resource_name)}"),',
                        f"    .pool_units = {_u32_expr(pool_units)},",
                        f"    .allocation_granularity = {_u32_expr(allocation_granularity)},",
                        f"    .member_start = {_u16_expr(member_start)},",
                        f"    .member_count = {_u16_expr(member_count)},",
                        f"    .cliff_start = {_u16_expr(cliff_start)},",
                        f"    .cliff_count = {_u16_expr(cliff_count)},",
                        "  },",
                    ]
                )
            lines.append("};")
            lines.extend(
                [
                    "",
                    f"static const uint16_t kAmdgpu{suffix}PressureResourceMemberIndicesByRegClass[] = {{",
                ]
            )
            lines.extend(f"  {_u16_expr(member_index)}," for member_index in resource_member_index_rows)
            lines.append("};")
            lines.extend(
                [
                    "",
                    f"static const loom_target_residency_derived_member_range_t kAmdgpu{suffix}PressureResourceMemberRangesByRegClass[] = {{",
                ]
            )
            for descriptor_reg_class in descriptor_reg_class_rows:
                start, count = resource_member_ranges[descriptor_reg_class.name]
                descriptor_reg_class_id = descriptor_reg_classes[descriptor_reg_class.name]
                lines.append(f"  [{_u16_expr(descriptor_reg_class_id)}] = {{.start = {_u16_expr(start)}, .count = {_u16_expr(count)}}},")
            lines.append("};")
            resource_initializer = f"kAmdgpu{suffix}PressureResources"
            resource_count_initializer = f"IREE_ARRAYSIZE(kAmdgpu{suffix}PressureResources)"
            resource_member_initializer = f"kAmdgpu{suffix}PressureResourceMembers"
            resource_member_count_initializer = f"IREE_ARRAYSIZE(kAmdgpu{suffix}PressureResourceMembers)"
            resource_cliff_initializer = f"kAmdgpu{suffix}PressureResourceCliffs"
            resource_cliff_count_initializer = f"IREE_ARRAYSIZE(kAmdgpu{suffix}PressureResourceCliffs)"
            resource_member_index_initializer = f"kAmdgpu{suffix}PressureResourceMemberIndicesByRegClass"
            resource_member_range_initializer = f"kAmdgpu{suffix}PressureResourceMemberRangesByRegClass"
        else:
            resource_initializer = "NULL"
            resource_count_initializer = "0"
            resource_member_initializer = "NULL"
            resource_member_count_initializer = "0"
            resource_cliff_initializer = "NULL"
            resource_cliff_count_initializer = "0"
            resource_member_index_initializer = "NULL"
            resource_member_range_initializer = "NULL"
        lines.extend(
            [
                "",
                f"static const loom_amdgpu_occupancy_model_t kAmdgpu{suffix}OccupancyModel = {{",
                f"  .descriptor_set_ordinal = {_u16_expr(amdgpu_descriptor_set_ordinal(model_row.descriptor_set_key))},",
                f"  .wave_size = {_u32_expr(model_row.wave_size)},",
                f"  .max_waves_per_simd = {_u32_expr(model.max_waves_per_simd)},",
                "  .domain = {",
                f"    .simd_count = {_u32_expr(model.domain.simd_count)},",
                f"    .local_memory_bytes = {_u32_expr(model.domain.local_memory_bytes)},",
                f"    .local_memory_allocation_granularity = {_u32_expr(model.domain.local_memory_allocation_granularity)},",
                f"    .max_barrier_workgroup_count = {_u32_expr(model.domain.max_barrier_workgroup_count)},",
                "  },",
                "  .residency_model = {",
                f"    .best_tier = {_u32_expr(model.max_waves_per_simd)},",
                "    .direct_resources = {",
                f"      .names = kAmdgpu{suffix}ResidencyDirectResourceNames,",
                f"      .cliffs = {pressure_cliffs_initializer},",
                f"      .cliff_count = {pressure_cliff_count_initializer},",
                f"      .cliff_ranges = kAmdgpu{suffix}PressureCliffRanges,",
                f"      .resource_count = IREE_ARRAYSIZE(kAmdgpu{suffix}ResidencyDirectResourceNames),",
                "    },",
                "    .derived_resources = {",
                f"      .resources = {resource_initializer},",
                f"      .resource_count = {resource_count_initializer},",
                f"      .members = {resource_member_initializer},",
                f"      .member_count = {resource_member_count_initializer},",
                f"      .cliffs = {resource_cliff_initializer},",
                f"      .cliff_count = {resource_cliff_count_initializer},",
                f"      .member_indices_by_direct_resource = {resource_member_index_initializer},",
                f"      .member_ranges_by_direct_resource = {resource_member_range_initializer},",
                "    },",
                "  },",
                f"  .register_classes = kAmdgpu{suffix}RegisterClasses,",
                f"  .register_class_count = IREE_ARRAYSIZE(kAmdgpu{suffix}RegisterClasses),",
                f"  .register_class_indices_by_descriptor_reg_class_id = kAmdgpu{suffix}RegisterClassIndexByDescriptorRegClassId,",
                f"  .descriptor_reg_class_count = IREE_ARRAYSIZE(kAmdgpu{suffix}RegisterClassIndexByDescriptorRegClassId),",
                "};",
            ]
        )
    model_by_processor_wave: dict[tuple[str, int], _AmdgpuOccupancyModelRow] = {}
    for model_row in models:
        for processor in model_row.processors:
            model_by_processor_wave[(processor.processor, model_row.wave_size)] = model_row
    lines.extend(
        [
            "",
            "const loom_amdgpu_occupancy_model_t* const",
            "    kLoomAmdgpuOccupancyModelsByProcessor[][LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_COUNT] = {",
        ]
    )
    for processor in sorted_processor_infos():
        wave_rows = {wave_size: model_by_processor_wave[(processor.processor, wave_size)] for wave_size in (32, 64) if (processor.processor, wave_size) in model_by_processor_wave}
        if not wave_rows:
            continue
        lines.append(f"  [{_u16_expr(amdgpu_processor_ordinal(processor.processor))}] = {{")
        for wave_size, model_row in wave_rows.items():
            wave_slot = "LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_32" if wave_size == 32 else "LOOM_AMDGPU_OCCUPANCY_WAVE_SLOT_64"
            lines.append(f"    [{wave_slot}] = &kAmdgpu{_model_c_suffix(model_row)}OccupancyModel,")
        lines.append("  },")
    lines.extend(
        [
            "};",
            "",
            "// clang-format on",
        ]
    )
    return "\n".join(lines) + "\n"


def write_occupancy_tables_to_path(source_path: Path) -> None:
    processors = sorted_processor_infos()
    models = _materialize_models(processors)
    _validate_models(models, processors)
    write_text_file(source_path, _emit_source(models))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU occupancy model C tables.")
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args(argv)
    write_occupancy_tables_to_path(args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
