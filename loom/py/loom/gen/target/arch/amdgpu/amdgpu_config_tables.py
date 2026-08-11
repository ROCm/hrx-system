# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU selected-target C config fragment emission."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from loom.gen.support.c import CIdentifierCase, c_identifier
from loom.gen.support.generated_file import line_comment_header
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FIELD_NAMES,
)
from loom.target.arch.amdgpu.names import (
    amdgpu_descriptor_set_define,
    amdgpu_descriptor_set_ordinal_constant_name,
    amdgpu_encoding_table_symbol,
    amdgpu_low_descriptor_provider_symbol,
)
from loom.target.arch.amdgpu.target_info import (
    AmdgpuDescriptorSetInfo,
    AmdgpuProcessorInfo,
    AmdgpuTargetInfo,
    amdgpu_target_descriptor_set_key,
)


def _selected_descriptor_set_infos(
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
    processors: Sequence[AmdgpuProcessorInfo],
    targets: Sequence[AmdgpuTargetInfo],
) -> tuple[AmdgpuDescriptorSetInfo, ...]:
    processors_by_name = {info.processor: info for info in processors}
    descriptor_sets_by_key = {info.key: info for info in descriptor_sets}
    descriptor_set_keys: list[str] = []
    for target in targets:
        processor = processors_by_name.get(target.processor)
        if processor is None:
            raise ValueError(f"AMDGPU target '{target.target}' has no processor row '{target.processor}'")
        descriptor_set_key = amdgpu_target_descriptor_set_key(target, processor)
        if not descriptor_set_key:
            raise ValueError(f"AMDGPU target '{target.target}' has no descriptor-set key")
        if descriptor_set_key not in descriptor_sets_by_key:
            raise ValueError(f"AMDGPU target '{target.target}' references unknown descriptor set '{descriptor_set_key}'")
        if descriptor_set_key not in descriptor_set_keys:
            descriptor_set_keys.append(descriptor_set_key)
    if not descriptor_set_keys:
        raise ValueError("AMDGPU selected descriptor-set list must not be empty")
    return tuple(descriptor_sets_by_key[key] for key in descriptor_set_keys)


def _emit_no_selected_descriptor_set_guard(
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
    *,
    message: str,
) -> list[str]:
    conditions = [f"!defined({amdgpu_descriptor_set_define(info.key)})" for info in descriptor_sets]
    lines: list[str] = []
    for index, condition in enumerate(conditions):
        suffix = " && \\" if index + 1 < len(conditions) else ""
        prefix = "#if " if index == 0 else "    "
        lines.append(f"{prefix}{condition}{suffix}")
    lines.extend([f'#error "{message}"', "#endif", ""])
    return lines


def _emit_low_registry_tables(
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.amdgpu.amdgpu_config_tables",
        ),
        "",
        "// AMDGPU low descriptor registry X-macro rows.",
        "//",
        "// Define one or more of the documented macros before including this file:",
        "//   LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER_DECL(provider)",
        "//   LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER(provider)",
        "",
        *_emit_no_selected_descriptor_set_guard(
            descriptor_sets,
            message="Loom AMDGPU low descriptor registry requires at least one selected descriptor set.",
        ),
        "#ifdef LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER_DECL",
    ]
    for descriptor_set in descriptor_sets:
        define = amdgpu_descriptor_set_define(descriptor_set.key)
        provider = amdgpu_low_descriptor_provider_symbol(descriptor_set.key)
        lines.extend(
            [
                f"#if defined({define})",
                f"LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER_DECL({provider})",
                "#endif",
            ]
        )
    lines.extend(["#endif  // LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER_DECL", ""])

    lines.append("#ifdef LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER")
    for descriptor_set in descriptor_sets:
        define = amdgpu_descriptor_set_define(descriptor_set.key)
        provider = amdgpu_low_descriptor_provider_symbol(descriptor_set.key)
        lines.extend(
            [
                f"#if defined({define})",
                f"LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER({provider})",
                "#endif",
            ]
        )
    lines.extend(["#endif  // LOOM_AMDGPU_LOW_DESCRIPTOR_PROVIDER", ""])
    return "\n".join(lines)


def _emit_encoding_tables(
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.amdgpu.amdgpu_config_tables",
        ),
        "",
        "// AMDGPU encoding table X-macro rows.",
        "//",
        "// Define one or more of the documented macros before including this file:",
        "//   LOOM_AMDGPU_ENCODING_TABLE_DECL(descriptor_set_ordinal, table_fn)",
        "//   LOOM_AMDGPU_ENCODING_TABLE(descriptor_set_ordinal, table_fn)",
        "",
        *_emit_no_selected_descriptor_set_guard(
            descriptor_sets,
            message="Loom AMDGPU encoding table lookup requires at least one selected descriptor set.",
        ),
        "#ifdef LOOM_AMDGPU_ENCODING_TABLE_DECL",
    ]
    for descriptor_set in descriptor_sets:
        define = amdgpu_descriptor_set_define(descriptor_set.key)
        ordinal = amdgpu_descriptor_set_ordinal_constant_name(descriptor_set.key)
        table = amdgpu_encoding_table_symbol(descriptor_set)
        lines.extend(
            [
                f"#if defined({define})",
                f"LOOM_AMDGPU_ENCODING_TABLE_DECL({ordinal}, {table})",
                "#endif",
            ]
        )
    lines.extend(["#endif  // LOOM_AMDGPU_ENCODING_TABLE_DECL", ""])

    lines.append("#ifdef LOOM_AMDGPU_ENCODING_TABLE")
    for descriptor_set in descriptor_sets:
        define = amdgpu_descriptor_set_define(descriptor_set.key)
        ordinal = amdgpu_descriptor_set_ordinal_constant_name(descriptor_set.key)
        table = amdgpu_encoding_table_symbol(descriptor_set)
        lines.extend(
            [
                f"#if defined({define})",
                f"LOOM_AMDGPU_ENCODING_TABLE({ordinal}, {table})",
                "#endif",
            ]
        )
    lines.extend(["#endif  // LOOM_AMDGPU_ENCODING_TABLE", ""])
    return "\n".join(lines)


def _emit_encoding_field_ids() -> str:
    lines: list[str] = []
    for name in AMDGPU_ENCODING_FIELD_NAMES:
        field_id = AMDGPU_ENCODING_FIELD_IDS[name]
        suffix = c_identifier(name, case=CIdentifierCase.UPPER, empty="EMPTY")
        lines.append(f"LOOM_AMDGPU_ENCODING_FIELD(LOOM_AMDGPU_ENCODING_FIELD_{suffix}, {field_id})")
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_config_tables_to_paths(
    *,
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
    processors: Sequence[AmdgpuProcessorInfo],
    targets: Sequence[AmdgpuTargetInfo],
    low_registry_tables_path: Path,
    encoding_tables_path: Path,
    encoding_field_ids_path: Path,
) -> None:
    selected_descriptor_sets = _selected_descriptor_set_infos(
        descriptor_sets,
        processors,
        targets,
    )
    _write(
        low_registry_tables_path,
        _emit_low_registry_tables(selected_descriptor_sets),
    )
    _write(
        encoding_tables_path,
        _emit_encoding_tables(selected_descriptor_sets),
    )
    _write(encoding_field_ids_path, _emit_encoding_field_ids())
