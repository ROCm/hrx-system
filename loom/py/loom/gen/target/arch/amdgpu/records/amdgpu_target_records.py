# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU compiler target-record X-macro rows."""

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

from loom.gen.support.c import c_pascal_identifier  # noqa: E402
from loom.gen.support.c import c_string_arg as _c_string_arg  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
    AmdgpuDescriptorSetInfo,
    AmdgpuProcessorInfo,
    AmdgpuTargetRecordInfo,
    amdgpu_descriptor_set_ordinal,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
    sorted_target_record_infos,
)


@dataclass(frozen=True, slots=True)
class _AmdgpuTargetRecordRow:
    info: AmdgpuTargetRecordInfo
    processor: AmdgpuProcessorInfo
    descriptor_set: AmdgpuDescriptorSetInfo
    descriptor_set_ordinal: int


def _u16_expr(value: int) -> str:
    return f"UINT16_C({value})"


def _u32_expr(value: int) -> str:
    return f"UINT32_C({value})"


def _c_symbol_suffix(value: str) -> str:
    suffix = c_pascal_identifier(value)
    if not suffix:
        raise ValueError("empty AMDGPU target-record symbol suffix")
    return suffix


def _target_bundle_name(generator_target: str) -> str:
    return f"amdgpu-{generator_target.replace('_', '-')}"


def _lookup_processor(
    processors: dict[str, AmdgpuProcessorInfo],
    processor_name: str,
) -> AmdgpuProcessorInfo:
    processor = processors.get(processor_name)
    if processor is None:
        raise ValueError(f"AMDGPU target record '{processor_name}' does not name a known processor")
    if not processor.descriptor_set.key:
        raise ValueError(f"AMDGPU target record '{processor_name}' has no supported descriptor set")
    return processor


def _lookup_descriptor_set(
    descriptor_sets: dict[str, AmdgpuDescriptorSetInfo],
    processor: AmdgpuProcessorInfo,
) -> AmdgpuDescriptorSetInfo:
    descriptor_set = descriptor_sets.get(processor.descriptor_set.key)
    if descriptor_set is None:
        raise ValueError(f"AMDGPU target record '{processor.processor}' references unknown descriptor set '{processor.descriptor_set.key}'")
    return descriptor_set


def _materialize_rows(
    target_records: Sequence[AmdgpuTargetRecordInfo],
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> tuple[_AmdgpuTargetRecordRow, ...]:
    processors_by_name = {processor.processor: processor for processor in processors}
    descriptor_sets_by_key = {info.key: info for info in descriptor_sets}
    rows: list[_AmdgpuTargetRecordRow] = []
    for info in target_records:
        processor = _lookup_processor(processors_by_name, info.processor)
        descriptor_set = _lookup_descriptor_set(descriptor_sets_by_key, processor)
        rows.append(
            _AmdgpuTargetRecordRow(
                info=info,
                processor=processor,
                descriptor_set=descriptor_set,
                descriptor_set_ordinal=amdgpu_descriptor_set_ordinal(descriptor_set.key),
            )
        )
    return tuple(rows)


def _validate_target_record_infos(rows: Sequence[_AmdgpuTargetRecordRow]) -> None:
    if not rows:
        raise ValueError("AMDGPU target records must not be empty")
    enum_values = [row.info.enum_value for row in rows]
    if enum_values != list(range(1, len(enum_values) + 1)):
        raise ValueError("AMDGPU target record enum values must be a dense one-based range")
    processors = [row.info.processor for row in rows]
    if len(processors) != len(set(processors)):
        raise ValueError("AMDGPU target record processors must be unique")

    defaults_by_descriptor_set: dict[str, list[_AmdgpuTargetRecordRow]] = {}
    for row in rows:
        if row.descriptor_set_ordinal >= AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"AMDGPU target record '{row.info.processor}' has an invalid descriptor-set ordinal {row.descriptor_set_ordinal}")
        if row.info.default_for_descriptor_set:
            defaults_by_descriptor_set.setdefault(row.descriptor_set.key, []).append(row)
    for descriptor_set_key in {row.descriptor_set.key for row in rows}:
        defaults = defaults_by_descriptor_set.get(descriptor_set_key, [])
        if len(defaults) != 1:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set_key}' requires exactly one default target record, found {len(defaults)}")


def _descriptor_sets_from_rows(
    rows: Sequence[_AmdgpuTargetRecordRow],
) -> tuple[AmdgpuDescriptorSetInfo, ...]:
    descriptor_sets_by_key = {row.descriptor_set.key: row.descriptor_set for row in rows}
    return tuple(
        sorted(
            descriptor_sets_by_key.values(),
            key=lambda info: amdgpu_descriptor_set_ordinal(info.key),
        )
    )


def _default_record_rows_by_ordinal(
    rows: Sequence[_AmdgpuTargetRecordRow],
) -> dict[int, _AmdgpuTargetRecordRow]:
    return {row.descriptor_set_ordinal: row for row in rows if row.info.default_for_descriptor_set}


def _emit_tables(rows: Sequence[_AmdgpuTargetRecordRow]) -> str:
    descriptor_sets = _descriptor_sets_from_rows(rows)
    default_rows_by_ordinal = _default_record_rows_by_ordinal(rows)
    max_descriptor_set_ordinal = max(row.descriptor_set_ordinal for row in rows)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.records.amdgpu_target_records"),
        "",
        "// AMDGPU target-record X-macro rows.",
        "//",
        "// Define one or more of the documented macros before including this file:",
        "//   LOOM_AMDGPU_TARGET_DESCRIPTOR_SET(symbol_suffix, bundle_name,",
        "//       snapshot_name, descriptor_set_key, wavefront_size)",
        "//   LOOM_AMDGPU_TARGET_RECORD_INFO(record_suffix, target_kind, processor,",
        "//       descriptor_set_ordinal, bundle_suffix)",
        "//   LOOM_AMDGPU_TARGET_RECORD_DEFAULT(descriptor_set_ordinal, record_suffix)",
        "//   LOOM_AMDGPU_TARGET_RECORD_DEFAULT_ABSENT(descriptor_set_ordinal)",
        "",
        "#ifdef LOOM_AMDGPU_TARGET_DESCRIPTOR_SET",
    ]
    for descriptor_set in descriptor_sets:
        default_row = default_rows_by_ordinal[amdgpu_descriptor_set_ordinal(descriptor_set.key)]
        bundle_name = _target_bundle_name(descriptor_set.generator_target)
        lines.append(
            "LOOM_AMDGPU_TARGET_DESCRIPTOR_SET("
            f"{_c_symbol_suffix(descriptor_set.generator_target)}, "
            f"{_c_string_arg(bundle_name)}, "
            f"{_c_string_arg(bundle_name + '-low')}, "
            f"{_c_string_arg(descriptor_set.key)}, "
            f"{default_row.processor.wavefront.default_size})"
        )
    lines.extend(["#endif  // LOOM_AMDGPU_TARGET_DESCRIPTOR_SET", ""])

    lines.append("#ifdef LOOM_AMDGPU_TARGET_RECORD_INFO")
    lines.extend(
        (
            "LOOM_AMDGPU_TARGET_RECORD_INFO("
            f"{_c_symbol_suffix(row.info.processor)}, "
            f"{_u32_expr(row.info.enum_value)}, "
            f"{_c_string_arg(row.info.processor)}, "
            f"{_u16_expr(row.descriptor_set_ordinal)}, "
            f"{_c_symbol_suffix(row.descriptor_set.generator_target)})"
        )
        for row in rows
    )
    lines.extend(["#endif  // LOOM_AMDGPU_TARGET_RECORD_INFO", ""])

    lines.append("#if defined(LOOM_AMDGPU_TARGET_RECORD_DEFAULT) && defined(LOOM_AMDGPU_TARGET_RECORD_DEFAULT_ABSENT)")
    for descriptor_set_ordinal in range(max_descriptor_set_ordinal + 1):
        row = default_rows_by_ordinal.get(descriptor_set_ordinal)
        if row is None:
            lines.append(f"LOOM_AMDGPU_TARGET_RECORD_DEFAULT_ABSENT({_u16_expr(descriptor_set_ordinal)})")
        else:
            lines.append(f"LOOM_AMDGPU_TARGET_RECORD_DEFAULT({_u16_expr(descriptor_set_ordinal)}, {_c_symbol_suffix(row.info.processor)})")
    lines.append("#endif  // LOOM_AMDGPU_TARGET_RECORD_DEFAULT && LOOM_AMDGPU_TARGET_RECORD_DEFAULT_ABSENT")
    lines.append("")
    return "\n".join(lines)


def write_target_record_tables_to_path(tables_path: Path) -> None:
    rows = _materialize_rows(
        sorted_target_record_infos(),
        sorted_processor_infos(),
        sorted_descriptor_set_infos(),
    )
    _validate_target_record_infos(rows)
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    tables_path.write_text(_emit_tables(rows), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU compiler target-record X-macro rows.")
    parser.add_argument(
        "--tables",
        required=True,
        type=Path,
        help="Generated target-record X-macro table path.",
    )
    args = parser.parse_args(argv)

    write_target_record_tables_to_path(tables_path=args.tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
