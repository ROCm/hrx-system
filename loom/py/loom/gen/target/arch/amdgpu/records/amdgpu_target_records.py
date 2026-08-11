# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU compiler target-record X-macro row emission."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.gen.support.c import c_pascal_identifier
from loom.gen.support.c import c_string_arg as _c_string_arg
from loom.gen.support.generated_file import line_comment_header
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
    AmdgpuDescriptorSetInfo,
    AmdgpuProcessorInfo,
    AmdgpuTargetInfo,
    amdgpu_descriptor_set_ordinal,
    amdgpu_target_descriptor_set_key,
)


@dataclass(frozen=True, slots=True)
class _AmdgpuTargetRecordRow:
    info: AmdgpuTargetInfo
    processor: AmdgpuProcessorInfo
    descriptor_set: AmdgpuDescriptorSetInfo
    descriptor_set_ordinal: int


def _u16_expr(value: int) -> str:
    return f"UINT16_C({value})"


def _u32_expr(value: int) -> str:
    return f"UINT32_C({value})"


def _u64_expr(value: int) -> str:
    return f"UINT64_C({value})"


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
    target: AmdgpuTargetInfo,
    processor: AmdgpuProcessorInfo,
) -> AmdgpuDescriptorSetInfo:
    descriptor_set_key = amdgpu_target_descriptor_set_key(target, processor)
    descriptor_set = descriptor_sets.get(descriptor_set_key)
    if descriptor_set is None:
        raise ValueError(f"AMDGPU target record '{target.target}' references unknown descriptor set '{descriptor_set_key}'")
    return descriptor_set


def _materialize_rows(
    targets: Sequence[AmdgpuTargetInfo],
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> tuple[_AmdgpuTargetRecordRow, ...]:
    processors_by_name = {processor.processor: processor for processor in processors}
    descriptor_sets_by_key = {info.key: info for info in descriptor_sets}
    rows: list[_AmdgpuTargetRecordRow] = []
    for info in targets:
        processor = _lookup_processor(processors_by_name, info.processor)
        descriptor_set = _lookup_descriptor_set(descriptor_sets_by_key, info, processor)
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
    targets = [row.info.target for row in rows]
    if len(targets) != len(set(targets)):
        raise ValueError("AMDGPU target record identities must be unique")

    defaults_by_descriptor_set: dict[str, list[_AmdgpuTargetRecordRow]] = {}
    for row in rows:
        if row.processor.limits.max_workgroup_storage_bytes == 0:
            raise ValueError(f"AMDGPU target record '{row.info.target}' requires a max workgroup storage limit")
        if row.descriptor_set_ordinal >= AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE:
            raise ValueError(f"AMDGPU target record '{row.info.target}' has an invalid descriptor-set ordinal {row.descriptor_set_ordinal}")
        if row.info.default_for_descriptor_set:
            defaults_by_descriptor_set.setdefault(row.descriptor_set.key, []).append(row)
    for descriptor_set_key in {row.descriptor_set.key for row in rows}:
        defaults = defaults_by_descriptor_set.get(descriptor_set_key, [])
        if len(defaults) != 1:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set_key}' requires exactly one default target record, found {len(defaults)}")
        expected_limit = defaults[0].processor.limits.max_workgroup_storage_bytes
        for row in rows:
            if row.descriptor_set.key != descriptor_set_key:
                continue
            actual_limit = row.processor.limits.max_workgroup_storage_bytes
            if actual_limit != expected_limit:
                raise ValueError(
                    f"AMDGPU descriptor set '{descriptor_set_key}' has inconsistent max workgroup storage limits: "
                    f"default target record '{defaults[0].info.target}' has {expected_limit}, "
                    f"target record '{row.info.target}' has {actual_limit}"
                )


def _validate_target_record_coverage(
    rows: Sequence[_AmdgpuTargetRecordRow],
    processors: Sequence[AmdgpuProcessorInfo],
) -> None:
    expected_processors = {processor.processor for processor in processors if processor.descriptor_set.key}
    actual_base_processors = {row.info.processor for row in rows if row.info.target == row.info.processor}
    if actual_base_processors == expected_processors:
        return
    missing_processors = sorted(expected_processors - actual_base_processors)
    unexpected_processors = sorted(actual_base_processors - expected_processors)
    raise ValueError(f"AMDGPU target records do not match descriptor-backed processors: missing={missing_processors}, unexpected={unexpected_processors}")


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
        "//       snapshot_name, descriptor_set_key, descriptor_set_flags,",
        "//       wavefront_size, max_workgroup_storage_bytes)",
        "//   LOOM_AMDGPU_TARGET_RECORD_INFO(record_suffix, target_kind, target,",
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
            f"{_u64_expr(descriptor_set.flags)}, "
            f"{default_row.processor.wavefront.default_size}, "
            f"{_u64_expr(default_row.processor.limits.max_workgroup_storage_bytes)})"
        )
    lines.extend(["#endif  // LOOM_AMDGPU_TARGET_DESCRIPTOR_SET", ""])

    lines.append("#ifdef LOOM_AMDGPU_TARGET_RECORD_INFO")
    lines.extend(
        (
            "LOOM_AMDGPU_TARGET_RECORD_INFO("
            f"{_c_symbol_suffix(row.info.target)}, "
            f"{_u32_expr(row.info.enum_value)}, "
            f"{_c_string_arg(row.info.target)}, "
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
            lines.append(f"LOOM_AMDGPU_TARGET_RECORD_DEFAULT({_u16_expr(descriptor_set_ordinal)}, {_c_symbol_suffix(row.info.target)})")
    lines.append("#endif  // LOOM_AMDGPU_TARGET_RECORD_DEFAULT && LOOM_AMDGPU_TARGET_RECORD_DEFAULT_ABSENT")
    lines.append("")
    return "\n".join(lines)


def write_target_record_tables_to_path(
    tables_path: Path,
    *,
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
    processors: Sequence[AmdgpuProcessorInfo],
    targets: Sequence[AmdgpuTargetInfo],
) -> None:
    rows = _materialize_rows(targets, processors, descriptor_sets)
    _validate_target_record_infos(rows)
    _validate_target_record_coverage(rows, processors)
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    tables_path.write_text(_emit_tables(rows), encoding="utf-8")
