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
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
    AmdgpuOccupancyModelInfo,
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
        for row in model.register_classes:
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


def _emit_header() -> str:
    guard = "LOOM_TARGET_ARCH_AMDGPU_PLANNING_OCCUPANCY_TABLES_H_"
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.planning.amdgpu_occupancy_tables"),
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
        '#include "iree/base/api.h"',
        '#include "loom/target/arch/amdgpu/target_info.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "typedef struct loom_amdgpu_occupancy_register_class_model_t {",
        "  // Stable target-low register-class name.",
        "  iree_string_view_t register_class;",
        "  // Occupancy register-file pool shared by resident waves.",
        "  uint32_t pool_units;",
        "  // Allocation granularity used by occupancy calculations.",
        "  uint32_t allocation_granularity;",
        "} loom_amdgpu_occupancy_register_class_model_t;",
        "",
        "typedef struct loom_amdgpu_occupancy_resource_member_model_t {",
        "  // Index into loom_amdgpu_occupancy_model_t::register_classes.",
        "  uint16_t register_class_index;",
        "  // Member contribution granularity applied before summing pressure.",
        "  uint32_t contribution_granularity;",
        "} loom_amdgpu_occupancy_resource_member_model_t;",
        "",
        "typedef struct loom_amdgpu_occupancy_resource_model_t {",
        "  // Stable target-low resource name.",
        "  iree_string_view_t resource;",
        "  // Occupancy resource pool shared by resident waves.",
        "  uint32_t pool_units;",
        "  // Allocation granularity used by occupancy calculations.",
        "  uint32_t allocation_granularity;",
        "  // Register-class members contributing to this resource.",
        "  const loom_amdgpu_occupancy_resource_member_model_t* members;",
        "  // Number of entries in members.",
        "  iree_host_size_t member_count;",
        "} loom_amdgpu_occupancy_resource_model_t;",
        "",
        "typedef struct loom_amdgpu_occupancy_model_t {",
        "  // Dense generated AMDGPU descriptor-set ordinal.",
        "  uint16_t descriptor_set_ordinal;",
        "  // AMDGPU wave size used by this model.",
        "  uint32_t wave_size;",
        "  // Maximum resident waves per SIMD.",
        "  uint32_t max_waves_per_simd;",
        "  // Register-class occupancy models in diagnostic order.",
        "  const loom_amdgpu_occupancy_register_class_model_t* register_classes;",
        "  // Number of entries in register_classes.",
        "  iree_host_size_t register_class_count;",
        "  // Derived occupancy resources in diagnostic order.",
        "  const loom_amdgpu_occupancy_resource_model_t* resources;",
        "  // Number of entries in resources.",
        "  iree_host_size_t resource_count;",
        "} loom_amdgpu_occupancy_model_t;",
        "",
        "// Returns the generated occupancy model for descriptor_set_ordinal.",
        "const loom_amdgpu_occupancy_model_t*",
        "loom_amdgpu_occupancy_model_for_descriptor_set_ordinal(",
        "    uint16_t descriptor_set_ordinal);",
        "",
        "#ifdef __cplusplus",
        '}  // extern "C"',
        "#endif",
        "",
        f"#endif  // {guard}",
    ]
    return "\n".join(lines) + "\n"


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
        '#include "loom/target/arch/amdgpu/planning/occupancy_tables.h"',
        "",
        "// clang-format off",
    ]
    for model in models:
        suffix = _model_c_suffix(model.descriptor_set_key)
        register_class_indices = {row.register_class: index for index, row in enumerate(model.register_classes)}
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
                    f"    .pool_units = {_u32_expr(row.pool_units)},",
                    f"    .allocation_granularity = {_u32_expr(row.allocation_granularity)},",
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
            "static const loom_amdgpu_occupancy_model_t* const kAmdgpuOccupancyModels[LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT] = {",
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
            "",
            "const loom_amdgpu_occupancy_model_t*",
            "loom_amdgpu_occupancy_model_for_descriptor_set_ordinal(",
            "    uint16_t descriptor_set_ordinal) {",
            "  if (descriptor_set_ordinal >= IREE_ARRAYSIZE(kAmdgpuOccupancyModels)) {",
            "    return NULL;",
            "  }",
            "  return kAmdgpuOccupancyModels[descriptor_set_ordinal];",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_occupancy_tables_to_paths(header_path: Path, source_path: Path) -> None:
    models = sorted_occupancy_model_infos()
    _validate_models(models)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(_emit_header(), encoding="utf-8")
    source_path.write_text(_emit_source(models), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU occupancy model C tables.")
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args(argv)
    write_occupancy_tables_to_paths(args.header, args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
