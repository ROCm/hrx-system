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
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    _amdgpu_core_descriptor_set_bases,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
    AmdgpuOccupancyModelInfo,
    AmdgpuOccupancyRegisterClassInfo,
    amdgpu_descriptor_set_ordinal,
    sorted_descriptor_set_infos,
    sorted_occupancy_model_infos,
)


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


def _model_symbol_suffix(descriptor_set_key: str) -> str:
    prefix = "amdgpu."
    suffix = ".core"
    if not descriptor_set_key.startswith(prefix) or not descriptor_set_key.endswith(suffix):
        raise ValueError(f"AMDGPU occupancy descriptor-set key '{descriptor_set_key}' must be a core key")
    return _c_identifier(descriptor_set_key.removeprefix(prefix).removesuffix(suffix))


def _model_c_suffix(descriptor_set_key: str) -> str:
    return _camel_c_suffix(_model_symbol_suffix(descriptor_set_key))


def _resource_c_suffix(resource: str) -> str:
    prefix = "amdgpu."
    if resource.startswith(prefix):
        resource = resource.removeprefix(prefix)
    return _camel_c_suffix(_c_identifier(resource))


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
    register_class: AmdgpuOccupancyRegisterClassInfo,
    max_waves_per_simd: int,
) -> tuple[tuple[int, int, int], ...]:
    previous_wave_limit = max_waves_per_simd
    cliffs: list[tuple[int, int, int]] = []
    stop_candidate = min(
        register_class.pool_units + register_class.allocation_granularity,
        0xFFFFFFFF,
    )
    for candidate in range(1, stop_candidate + 1):
        candidate_wave_limit = _wave_limit(
            register_class.pool_units,
            register_class.allocation_granularity,
            max_waves_per_simd,
            candidate,
        )
        if candidate_wave_limit >= previous_wave_limit:
            continue
        cliffs.append((candidate, previous_wave_limit, candidate_wave_limit))
        previous_wave_limit = candidate_wave_limit
        if candidate_wave_limit == 0:
            break
    return tuple(cliffs)


def _validate_models(models: Sequence[AmdgpuOccupancyModelInfo]) -> None:
    descriptor_set_keys = {info.key for info in sorted_descriptor_set_infos()}
    model_keys = [info.descriptor_set_key for info in models]
    if len(model_keys) != len(set(model_keys)):
        raise ValueError("AMDGPU occupancy descriptor-set keys must be unique")
    for model in models:
        if model.descriptor_set_key not in descriptor_set_keys:
            raise ValueError(f"AMDGPU occupancy model references unknown descriptor set '{model.descriptor_set_key}'")
        if amdgpu_descriptor_set_ordinal(model.descriptor_set_key) >= (AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE):
            raise ValueError("AMDGPU occupancy descriptor-set ordinal overflows")
        if model.wave_size not in (32, 64):
            raise ValueError(f"AMDGPU occupancy wave size for {model.descriptor_set_key} must be 32 or 64")
        if model.max_waves_per_simd <= 0:
            raise ValueError(f"AMDGPU occupancy max waves for {model.descriptor_set_key} must be positive")
        register_classes = [row.register_class for row in model.register_classes]
        if len(register_classes) > 0x10000:
            raise ValueError(f"AMDGPU occupancy model for {model.descriptor_set_key} has too many register classes")
        if len(register_classes) != len(set(register_classes)):
            raise ValueError(f"AMDGPU occupancy model for {model.descriptor_set_key} has duplicate register classes")
        descriptor_reg_classes = _descriptor_reg_class_ids(model.descriptor_set_key)
        for row in model.register_classes:
            if row.register_class not in descriptor_reg_classes:
                raise ValueError(f"AMDGPU occupancy model for {model.descriptor_set_key} references unknown register class {row.register_class}")
            if row.pool_units <= 0:
                raise ValueError(f"AMDGPU occupancy pool for {row.register_class} must be positive")
            if row.allocation_granularity <= 0:
                raise ValueError(f"AMDGPU occupancy granularity for {row.register_class} must be positive")
        resources = [row.resource for row in model.resources]
        if len(resources) != len(set(resources)):
            raise ValueError(f"AMDGPU occupancy model for {model.descriptor_set_key} has duplicate resources")
        for resource in model.resources:
            if resource.pool_units <= 0:
                raise ValueError(f"AMDGPU occupancy resource pool for {resource.resource} must be positive")
            if resource.allocation_granularity <= 0:
                raise ValueError(f"AMDGPU occupancy resource granularity for {resource.resource} must be positive")
            if not resource.members:
                raise ValueError(f"AMDGPU occupancy resource {resource.resource} must have members")
            member_register_classes = [member.register_class for member in resource.members]
            if len(member_register_classes) != len(set(member_register_classes)):
                raise ValueError(f"AMDGPU occupancy resource {resource.resource} has duplicate members")
            for member in resource.members:
                if member.register_class not in register_classes:
                    raise ValueError(f"AMDGPU occupancy resource {resource.resource} references unknown register class {member.register_class}")
                if member.contribution_granularity <= 0:
                    raise ValueError(f"AMDGPU occupancy resource {resource.resource} member {member.register_class} granularity must be positive")


def _emit_source(models: Sequence[AmdgpuOccupancyModelInfo]) -> str:
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
    for model in models:
        suffix = _model_c_suffix(model.descriptor_set_key)
        descriptor_reg_classes = _descriptor_reg_class_ids(model.descriptor_set_key)
        register_class_indices = {row.register_class: index for index, row in enumerate(model.register_classes)}
        pressure_cliffs_by_register_class = {row.register_class: _pressure_cliffs(row, model.max_waves_per_simd) for row in model.register_classes}
        pressure_cliff_count = sum(len(cliffs) for cliffs in pressure_cliffs_by_register_class.values())
        for row in model.register_classes:
            cliffs = pressure_cliffs_by_register_class[row.register_class]
            register_class_suffix = _resource_c_suffix(row.register_class)
            if not cliffs:
                continue
            lines.extend(
                [
                    "",
                    f"static const loom_amdgpu_occupancy_pressure_cliff_model_t kAmdgpu{suffix}{register_class_suffix}PressureCliffs[] = {{",
                ]
            )
            for cliff_units, tier_before, tier_after in cliffs:
                lines.extend(
                    [
                        "  {",
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
                f"static const loom_amdgpu_occupancy_register_class_model_t kAmdgpu{suffix}RegisterClasses[] = {{",
            ]
        )
        for row in model.register_classes:
            cliffs = pressure_cliffs_by_register_class[row.register_class]
            register_class_suffix = _resource_c_suffix(row.register_class)
            if cliffs:
                pressure_cliffs_initializer = f"kAmdgpu{suffix}{register_class_suffix}PressureCliffs"
                pressure_cliff_count_initializer = f"IREE_ARRAYSIZE(kAmdgpu{suffix}{register_class_suffix}PressureCliffs)"
            else:
                pressure_cliffs_initializer = "NULL"
                pressure_cliff_count_initializer = "0"
            lines.extend(
                [
                    "  {",
                    f'    .register_class = IREE_SVL("{_c_string_literal(row.register_class)}"),',
                    f"    .descriptor_reg_class_id = {_u16_expr(descriptor_reg_classes[row.register_class])},",
                    f"    .pool_units = {_u32_expr(row.pool_units)},",
                    f"    .allocation_granularity = {_u32_expr(row.allocation_granularity)},",
                    f"    .pressure_cliffs = {pressure_cliffs_initializer},",
                    f"    .pressure_cliff_count = {pressure_cliff_count_initializer},",
                    "  },",
                ]
            )
        lines.append("};")
        for resource in model.resources:
            resource_suffix = _resource_c_suffix(resource.resource)
            lines.extend(
                [
                    "",
                    f"static const loom_amdgpu_occupancy_resource_member_model_t kAmdgpu{suffix}{resource_suffix}Members[] = {{",
                ]
            )
            for member in resource.members:
                lines.extend(
                    [
                        "  {",
                        f"    .register_class_index = {_u16_expr(register_class_indices[member.register_class])},",
                        f"    .contribution_granularity = {_u32_expr(member.contribution_granularity)},",
                        "  },",
                    ]
                )
            lines.append("};")
        if model.resources:
            lines.extend(
                [
                    "",
                    f"static const loom_amdgpu_occupancy_resource_model_t kAmdgpu{suffix}Resources[] = {{",
                ]
            )
            for resource in model.resources:
                resource_suffix = _resource_c_suffix(resource.resource)
                lines.extend(
                    [
                        "  {",
                        f'    .resource = IREE_SVL("{_c_string_literal(resource.resource)}"),',
                        f"    .pool_units = {_u32_expr(resource.pool_units)},",
                        f"    .allocation_granularity = {_u32_expr(resource.allocation_granularity)},",
                        f"    .members = kAmdgpu{suffix}{resource_suffix}Members,",
                        f"    .member_count = IREE_ARRAYSIZE(kAmdgpu{suffix}{resource_suffix}Members),",
                        "  },",
                    ]
                )
            lines.append("};")
            resource_initializer = f"kAmdgpu{suffix}Resources"
            resource_count_initializer = f"IREE_ARRAYSIZE(kAmdgpu{suffix}Resources)"
        else:
            resource_initializer = "NULL"
            resource_count_initializer = "0"
        lines.extend(
            [
                "",
                f"static const loom_amdgpu_occupancy_model_t kAmdgpu{suffix}OccupancyModel = {{",
                f"  .descriptor_set_ordinal = {_u16_expr(amdgpu_descriptor_set_ordinal(model.descriptor_set_key))},",
                f"  .wave_size = {_u32_expr(model.wave_size)},",
                f"  .max_waves_per_simd = {_u32_expr(model.max_waves_per_simd)},",
                f"  .pressure_cliff_count = {pressure_cliff_count},",
                f"  .register_classes = kAmdgpu{suffix}RegisterClasses,",
                f"  .register_class_count = IREE_ARRAYSIZE(kAmdgpu{suffix}RegisterClasses),",
                f"  .resources = {resource_initializer},",
                f"  .resource_count = {resource_count_initializer},",
                "};",
            ]
        )
    lines.extend(
        [
            "",
            "const loom_amdgpu_occupancy_model_t* const kLoomAmdgpuOccupancyModels[LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT] = {",
        ]
    )
    for model in models:
        suffix = _model_c_suffix(model.descriptor_set_key)
        lines.extend(
            [
                f"  [{_u16_expr(amdgpu_descriptor_set_ordinal(model.descriptor_set_key))}] = &kAmdgpu{suffix}OccupancyModel,",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "// clang-format on",
        ]
    )
    return "\n".join(lines) + "\n"


def write_occupancy_tables_to_path(source_path: Path) -> None:
    models = sorted_occupancy_model_infos()
    _validate_models(models)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(_emit_source(models), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU occupancy model C tables.")
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args(argv)
    write_occupancy_tables_to_path(args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
