# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU selected-target C config fragments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import CIdentifierCase, c_identifier  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.encoding import (  # noqa: E402
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FIELD_NAMES,
)
from loom.target.arch.amdgpu.names import (  # noqa: E402
    amdgpu_descriptor_set_define,
    amdgpu_descriptor_set_ordinal_constant_name,
    amdgpu_encoding_table_symbol,
    amdgpu_low_descriptor_provider_symbol,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AmdgpuDescriptorSetInfo,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
    sorted_target_record_infos,
)


def _selected_descriptor_set_infos() -> tuple[AmdgpuDescriptorSetInfo, ...]:
    processors_by_name = {info.processor: info for info in sorted_processor_infos()}
    descriptor_sets_by_key = {info.key: info for info in sorted_descriptor_set_infos()}
    descriptor_set_keys: list[str] = []
    for record in sorted_target_record_infos():
        processor = processors_by_name.get(record.processor)
        if processor is None:
            raise ValueError(f"AMDGPU target record '{record.processor}' has no processor row")
        descriptor_set_key = processor.descriptor_set.key
        if not descriptor_set_key:
            raise ValueError(f"AMDGPU target record '{record.processor}' has no descriptor-set key")
        if descriptor_set_key not in descriptor_sets_by_key:
            raise ValueError(f"AMDGPU target record '{record.processor}' references unknown descriptor set '{descriptor_set_key}'")
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


def _emit_encoding_field_ids_header() -> str:
    guard = "LOOM_TARGET_ARCH_AMDGPU_ENCODING_ENCODING_FIELD_IDS_H_"
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
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "// Stable target-owned encoding field identifiers used in",
        "// loom_amdgpu_encoding_field_value_t arrays.",
        "enum {",
    ]
    for name in AMDGPU_ENCODING_FIELD_NAMES:
        field_id = AMDGPU_ENCODING_FIELD_IDS[name]
        field_suffix = c_identifier(name, case=CIdentifierCase.UPPER, empty="EMPTY")
        lines.append(f"  LOOM_AMDGPU_ENCODING_FIELD_{field_suffix} = {field_id},")
    lines.extend(["};", "", f"#endif  // {guard}", ""])
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU selected-target C config fragments.")
    parser.add_argument(
        "--low-registry-tables",
        type=Path,
        help="Generated low descriptor registry X-macro table path.",
    )
    parser.add_argument(
        "--encoding-tables",
        type=Path,
        help="Generated encoding table X-macro table path.",
    )
    parser.add_argument(
        "--encoding-field-ids-header",
        type=Path,
        help="Generated encoding field-id header path.",
    )
    args = parser.parse_args(argv)

    if not (args.low_registry_tables or args.encoding_tables or args.encoding_field_ids_header):
        parser.error("at least one output flag is required")

    descriptor_sets = _selected_descriptor_set_infos()
    if args.low_registry_tables:
        _write(args.low_registry_tables, _emit_low_registry_tables(descriptor_sets))
    if args.encoding_tables:
        _write(args.encoding_tables, _emit_encoding_tables(descriptor_sets))
    if args.encoding_field_ids_header:
        _write(args.encoding_field_ids_header, _emit_encoding_field_ids_header())
    return 0


if __name__ == "__main__":
    sys.exit(main())
