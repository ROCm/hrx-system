# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML encoding layouts -> compact C bit tables."""

from __future__ import annotations

import argparse
import re
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.encoding import (  # noqa: E402
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FORMAT_IDS,
)
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    AmdgpuIsaBitRange,
    AmdgpuIsaEncoding,
    AmdgpuIsaEncodingField,
    AmdgpuIsaFactSource,
    AmdgpuIsaInstruction,
    AmdgpuIsaOperandType,
    compose_amdgpu_isa_partitioned_field,
    parse_amdgpu_isa_xml_path,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AmdgpuDescriptorSetInfo,
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_ordinal,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
    validate_amdgpu_descriptor_set_isa_xml,
)


@dataclass(frozen=True, slots=True)
class _CompiledField:
    field: AmdgpuIsaEncodingField
    range_start: int
    range_count: int
    value_bit_count: int


@dataclass(frozen=True, slots=True)
class _CompiledFormat:
    encoding: AmdgpuIsaEncoding
    format_id: int
    field_start: int
    field_count: int
    word_count: int


def _bit_range(bit_offset: int, bit_count: int) -> AmdgpuIsaBitRange:
    return AmdgpuIsaBitRange(order=0, bit_count=bit_count, bit_offset=bit_offset)


def _field(name: str, *ranges: AmdgpuIsaBitRange) -> AmdgpuIsaEncodingField:
    return AmdgpuIsaEncodingField(
        name=name,
        is_conditional=False,
        ranges=ranges,
    )


def _rdna4_vop3p_supplemental_fields() -> tuple[AmdgpuIsaEncodingField, ...]:
    return (
        _field("INDEX_KEY_16BIT", _bit_range(11, 1)),
        _field("MATRIX_A_REUSE", _bit_range(13, 1)),
        _field("MATRIX_B_REUSE", _bit_range(14, 1)),
    )


def _rdna4_vop3px2_supplemental_encoding() -> AmdgpuIsaEncoding:
    return AmdgpuIsaEncoding(
        name="ENC_VOP3PX2",
        order=0,
        bit_count=128,
        identifier_mask=(0xFF << 24) | (0x1FF << 50) | (0xFF << 88),
        identifier_values=((0xCC << 24) | (0x100 << 50) | (0xCC << 88),),
        fields=(
            _field("MATRIX_B_SCALE_FMT", _bit_range(8, 2)),
            _field("MATRIX_A_SCALE", _bit_range(11, 1)),
            _field("MATRIX_A_REUSE", _bit_range(13, 1)),
            _field("MATRIX_B_REUSE", _bit_range(14, 1)),
            _field("X2ENCODING", _bit_range(16, 8)),
            _field("SCALE_SRC0", _bit_range(32, 9)),
            _field("SCALE_SRC1", _bit_range(41, 9)),
            _field("MATRIX_B_SCALE", _bit_range(59, 1)),
            _field("MATRIX_A_SCALE_FMT", _bit_range(61, 2)),
            _field("VDST", _bit_range(64, 8)),
            _field("MATRIX_A_FMT", _bit_range(75, 3)),
            _field("MATRIX_B_FMT", _bit_range(123, 2), _bit_range(78, 1)),
            _field("CLAMP", _bit_range(79, 1)),
            _field("OP", _bit_range(80, 8)),
            _field("SRC0", _bit_range(96, 9)),
            _field("SRC1", _bit_range(105, 9)),
            _field("SRC2", _bit_range(114, 9)),
        ),
    )


def _supplemental_fields_by_encoding(
    target: str,
) -> dict[str, tuple[AmdgpuIsaEncodingField, ...]]:
    if target in ("rdna4", "rdna4_gfx125x"):
        return {
            "ENC_VOP3P": _rdna4_vop3p_supplemental_fields(),
        }
    return {}


def _supplemental_encodings(
    target: str,
    _encodings: tuple[AmdgpuIsaEncoding, ...],
) -> tuple[AmdgpuIsaEncoding, ...]:
    if target in ("rdna4", "rdna4_gfx125x"):
        return (_rdna4_vop3px2_supplemental_encoding(),)
    return ()


def _supplement_encoding_fields(
    encoding: AmdgpuIsaEncoding,
    supplemental_fields: tuple[AmdgpuIsaEncodingField, ...],
) -> AmdgpuIsaEncoding:
    if not supplemental_fields:
        return encoding
    field_names = {field.name for field in encoding.fields}
    duplicate_names = sorted(field.name for field in supplemental_fields if field.name in field_names)
    if duplicate_names:
        duplicate_text = ", ".join(duplicate_names)
        raise ValueError(f"AMDGPU encoding '{encoding.name}' supplemental fields collide with XML fields: {duplicate_text}")
    return replace(encoding, fields=(*encoding.fields, *supplemental_fields))


def _with_supplemental_encodings(target: str, encodings: tuple[AmdgpuIsaEncoding, ...]) -> tuple[AmdgpuIsaEncoding, ...]:
    fields_by_encoding = _supplemental_fields_by_encoding(target)
    supplemental_encodings = _supplemental_encodings(target, encodings)
    supplemental_names = {encoding.name for encoding in supplemental_encodings}
    output = []
    for encoding in encodings:
        if encoding.name in supplemental_names:
            raise ValueError(f"AMDGPU encoding target '{target}' XML now defines supplemental encoding '{encoding.name}'")
        output.append(_supplement_encoding_fields(encoding, fields_by_encoding.get(encoding.name, ())))
    output.extend(supplemental_encodings)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class _EncodingTableView:
    descriptor_set_key: str
    table_prefix: str
    table_function: str


@dataclass(frozen=True, slots=True)
class _InlineF32Source:
    bit_pattern: int
    source: int


def _parse_view_headers(values: Sequence[str]) -> dict[str, Path]:
    view_headers: dict[str, Path] = {}
    for value in values:
        target, separator, path = value.partition("=")
        if not separator or not target or not path:
            raise ValueError("AMDGPU encoding --view-header must have form <target>=<path>")
        if target in view_headers:
            raise ValueError(f"duplicate AMDGPU encoding view header for {target}")
        view_headers[target] = Path(path)
    return view_headers


def _table_prefix_for_target(target: str) -> str:
    return "Amdgpu" + "".join(part.title() for part in target.split("_"))


def _table_function_for_target(target: str) -> str:
    return f"loom_amdgpu_{target}_encoding_table"


def _encoding_table_view_for_info(
    info: AmdgpuDescriptorSetInfo,
) -> _EncodingTableView:
    return _EncodingTableView(
        descriptor_set_key=info.key,
        table_prefix=_table_prefix_for_target(info.generator_target),
        table_function=_table_function_for_target(info.generator_target),
    )


def _view_infos_for_storage_target(
    storage_info: AmdgpuDescriptorSetInfo,
    view_headers: dict[str, Path],
) -> tuple[AmdgpuDescriptorSetInfo, ...]:
    view_infos = amdgpu_descriptor_set_view_infos_by_storage_generator_target(storage_info.generator_target)
    expected_view_targets = {info.generator_target for info in view_infos}
    unknown_view_headers = set(view_headers) - expected_view_targets
    if unknown_view_headers:
        unknown_targets = ", ".join(sorted(unknown_view_headers))
        raise ValueError(f"AMDGPU encoding target {storage_info.generator_target} cannot emit view headers for: {unknown_targets}")
    return view_infos


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def _word_count(bit_count: int) -> int:
    return (bit_count + 31) // 32


def _split_words(value: int, word_count: int) -> tuple[int, ...]:
    return tuple((value >> (32 * i)) & 0xFFFFFFFF for i in range(word_count))


def _range_value_bit_count(bit_range: AmdgpuIsaBitRange) -> int:
    return bit_range.padding_bit_count + bit_range.bit_count


def _compile_field_ranges(
    ranges: tuple[AmdgpuIsaBitRange, ...],
) -> tuple[tuple[tuple[AmdgpuIsaBitRange, int], ...], int]:
    source_bit_offset = 0
    compiled_ranges = []
    for bit_range in ranges:
        compiled_ranges.append((bit_range, source_bit_offset))
        source_bit_offset += _range_value_bit_count(bit_range)
    return tuple(compiled_ranges), source_bit_offset


def _encoding_fields_by_name(
    encoding: AmdgpuIsaEncoding,
) -> dict[str, AmdgpuIsaEncodingField]:
    return {field.name: field for field in encoding.fields}


def _add_encoding_field(
    fields: dict[str, AmdgpuIsaEncodingField],
    ambiguous_fields: set[str],
    field: AmdgpuIsaEncodingField,
) -> None:
    if field.name in ambiguous_fields:
        return
    existing_field = fields.get(field.name)
    if existing_field is None:
        fields[field.name] = field
        return
    if existing_field != field:
        del fields[field.name]
        ambiguous_fields.add(field.name)


def _partitioned_fields_by_encoding(
    encodings: tuple[AmdgpuIsaEncoding, ...],
    instructions: tuple[AmdgpuIsaInstruction, ...],
    operand_types: tuple[AmdgpuIsaOperandType, ...],
) -> dict[str, tuple[AmdgpuIsaEncodingField, ...]]:
    encodings_by_name = {encoding.name: encoding for encoding in encodings}
    operand_types_by_name = {operand_type.name: operand_type for operand_type in operand_types}
    fields_by_encoding: dict[str, dict[str, AmdgpuIsaEncodingField]] = {encoding.name: {} for encoding in encodings}
    ambiguous_fields_by_encoding: dict[str, set[str]] = {encoding.name: set() for encoding in encodings}
    for instruction in instructions:
        for instruction_encoding in instruction.encodings:
            encoding = encodings_by_name.get(instruction_encoding.encoding_name)
            if encoding is None:
                raise ValueError(f"AMDGPU instruction '{instruction.name}' references missing encoding '{instruction_encoding.encoding_name}'")
            base_fields = _encoding_fields_by_name(encoding)
            for operand in instruction_encoding.operands:
                if not operand.is_binary_microcode_required:
                    continue
                operand_type = operand_types_by_name.get(operand.operand_type)
                if operand_type is None:
                    raise ValueError(f"AMDGPU instruction '{instruction.name}' references missing operand type '{operand.operand_type}'")
                if not operand_type.is_partitioned:
                    continue
                if operand.field_name is None:
                    raise ValueError(f"AMDGPU instruction '{instruction.name}' encoding '{instruction_encoding.encoding_name}' has partitioned binary microcode operand without a field name")
                base_field = base_fields.get(operand.field_name)
                if base_field is None:
                    raise ValueError(f"AMDGPU instruction '{instruction.name}' binary microcode operand '{operand.field_name}' is not a field in encoding '{encoding.name}'")
                for operand_field in operand_type.fields:
                    if operand_field.name in base_fields:
                        continue
                    composed_field = compose_amdgpu_isa_partitioned_field(base_field, operand_field)
                    if composed_field is None:
                        continue
                    _add_encoding_field(
                        fields_by_encoding[encoding.name],
                        ambiguous_fields_by_encoding[encoding.name],
                        composed_field,
                    )
    return {encoding_name: tuple(sorted(fields.values(), key=lambda field: field.name)) for encoding_name, fields in fields_by_encoding.items() if fields}


def _emit_header(
    *,
    header_guard: str,
    table_function: str,
) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.encoding.amdgpu_encoding_tables"),
        "",
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        '#include "loom/target/arch/amdgpu/encoding/encoding.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
    ]
    lines.extend(
        [
            f"const loom_amdgpu_encoding_table_t* {table_function}(void);",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif",
            "",
            f"#endif  // {header_guard}",
        ]
    )
    return "\n".join(lines) + "\n"


def _emit_header_for_target(
    *,
    target: str,
) -> str:
    return _emit_header(
        header_guard=f"LOOM_TARGET_ARCH_AMDGPU_{target.upper()}_ENCODING_TABLES_H_",
        table_function=f"loom_amdgpu_{target}_encoding_table",
    )


def _compile_formats(
    encodings: tuple[AmdgpuIsaEncoding, ...],
    partitioned_fields_by_encoding: dict[str, tuple[AmdgpuIsaEncodingField, ...]],
) -> tuple[list[_CompiledFormat], list[_CompiledField], list[tuple[AmdgpuIsaBitRange, int]]]:
    compiled_formats: list[_CompiledFormat] = []
    compiled_fields: list[_CompiledField] = []
    compiled_ranges: list[tuple[AmdgpuIsaBitRange, int]] = []
    for encoding in encodings:
        try:
            format_id = AMDGPU_ENCODING_FORMAT_IDS[encoding.name]
        except KeyError as exc:
            raise ValueError(f"unmapped AMDGPU encoding format '{encoding.name}'") from exc
        if encoding.bit_count > 32 * 4:
            raise ValueError(f"AMDGPU encoding '{encoding.name}' has unsupported {encoding.bit_count}-bit packet width")
        field_start = len(compiled_fields)
        fields = list(encoding.fields)
        fields.extend(partitioned_fields_by_encoding.get(encoding.name, ()))
        for field in fields:
            try:
                AMDGPU_ENCODING_FIELD_IDS[field.name]
            except KeyError as exc:
                raise ValueError(f"AMDGPU encoding '{encoding.name}' references unmapped field '{field.name}'") from exc
            range_start = len(compiled_ranges)
            field_ranges, value_bit_count = _compile_field_ranges(field.ranges)
            if value_bit_count > 64:
                raise ValueError(f"AMDGPU encoding '{encoding.name}' field '{field.name}' has unsupported {value_bit_count}-bit source value")
            compiled_ranges.extend(field_ranges)
            compiled_fields.append(
                _CompiledField(
                    field=field,
                    range_start=range_start,
                    range_count=len(field_ranges),
                    value_bit_count=value_bit_count,
                )
            )
        compiled_formats.append(
            _CompiledFormat(
                encoding=encoding,
                format_id=format_id,
                field_start=field_start,
                field_count=len(fields),
                word_count=_word_count(encoding.bit_count),
            )
        )
    compiled_formats.sort(key=lambda entry: entry.format_id)
    return compiled_formats, compiled_fields, compiled_ranges


def _instruction_opcode(
    instructions: tuple[AmdgpuIsaInstruction, ...],
    *,
    instruction_name: str,
    encoding_name: str,
    condition_name: str | None,
) -> int:
    for instruction in instructions:
        if instruction.name != instruction_name:
            continue
        opcodes = {encoding.opcode for encoding in instruction.encodings if encoding.encoding_name == encoding_name and (condition_name is None or encoding.condition_name == condition_name)}
        if len(opcodes) == 1:
            return next(iter(opcodes))
        if opcodes:
            raise ValueError(f"AMDGPU instruction '{instruction_name}' has ambiguous {encoding_name}/{condition_name or '*'} opcodes")
        for encoding in instruction.encodings:
            if encoding.encoding_name == encoding_name:
                raise ValueError(f"AMDGPU instruction '{instruction_name}' has no {encoding_name}/{condition_name or '*'} encoding")
        raise ValueError(f"AMDGPU instruction '{instruction_name}' has no {encoding_name} encoding")
    raise ValueError(f"AMDGPU instruction '{instruction_name}' is missing")


def _derive_predefined_linear_range(
    spec: AmdgpuIsaFactSource,
    *,
    operand_type_name: str,
    base_name: str,
    name_pattern: re.Pattern[str],
    description: str,
) -> tuple[int, int]:
    operand_type = spec.operand_type_map().get(operand_type_name)
    if operand_type is None:
        raise ValueError(f"{spec.source_name}: unknown AMDGPU ISA operand type '{operand_type_name}'")
    base_value = spec.operand_predefined_value(operand_type_name, base_name)
    indexed_values: dict[int, int] = {}
    for predefined_value in operand_type.predefined_values:
        match = name_pattern.fullmatch(predefined_value.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in indexed_values:
            raise ValueError(f"{spec.source_name}: duplicate {description} predefined value index {index}")
        indexed_values[index] = predefined_value.value
    count = 0
    while count in indexed_values:
        actual_value = indexed_values[count]
        expected_value = base_value + count
        if actual_value != expected_value:
            raise ValueError(f"{spec.source_name}: {description} {count} has value {actual_value}, expected {expected_value}")
        count += 1
    if count == 0:
        raise ValueError(f"{spec.source_name}: no {description} predefined values")
    extra_indices = [index for index in indexed_values if index >= count]
    if extra_indices:
        extra_text = ", ".join(str(index) for index in sorted(extra_indices))
        raise ValueError(f"{spec.source_name}: non-contiguous {description} predefined value indices after {count - 1}: {extra_text}")
    return base_value, count


def _f32_bit_pattern(value: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", value))[0])


def _derive_predefined_f32_sources(
    spec: AmdgpuIsaFactSource,
    *,
    operand_type_name: str,
) -> tuple[_InlineF32Source, ...]:
    operand_type = spec.operand_type_map().get(operand_type_name)
    if operand_type is None:
        raise ValueError(f"{spec.source_name}: unknown AMDGPU ISA operand type '{operand_type_name}'")
    sources: list[_InlineF32Source] = []
    seen_bit_patterns: set[int] = set()
    for predefined_value in operand_type.predefined_values:
        if "." not in predefined_value.name:
            continue
        try:
            bit_pattern = _f32_bit_pattern(float(predefined_value.name))
        except ValueError:
            continue
        if bit_pattern in seen_bit_patterns:
            raise ValueError(f"{spec.source_name}: duplicate OPR_SRC inline f32 bit pattern 0x{bit_pattern:08x}")
        seen_bit_patterns.add(bit_pattern)
        sources.append(
            _InlineF32Source(
                bit_pattern=bit_pattern,
                source=predefined_value.value,
            )
        )
    if not sources:
        raise ValueError(f"{spec.source_name}: no OPR_SRC inline f32 predefined values")
    return tuple(sorted(sources, key=lambda source: source.bit_pattern))


def _emit_word_array(words: tuple[int, ...]) -> str:
    padded_words = words + (0,) * (4 - len(words))
    return "{" + ", ".join(f"UINT32_C(0x{word:08x})" for word in padded_words) + "}"


def _emit_source(
    *,
    target: str,
    descriptor_set_key: str,
    public_header: str,
    table_prefix: str,
    table_function: str,
    table_views: tuple[_EncodingTableView, ...] = (),
    encodings: tuple[AmdgpuIsaEncoding, ...],
    instructions: tuple[AmdgpuIsaInstruction, ...],
    operand_types: tuple[AmdgpuIsaOperandType, ...],
    source_literal: int,
    scalar_inline_u32_zero: int,
    scalar_inline_u32_count: int,
    inline_f32_sources: tuple[_InlineF32Source, ...],
    vector_source_vgpr0: int,
    vector_source_vgpr_count: int,
) -> str:
    encodings = _with_supplemental_encodings(target, encodings)
    compiled_formats, compiled_fields, compiled_ranges = _compile_formats(
        encodings,
        _partitioned_fields_by_encoding(encodings, instructions, operand_types),
    )
    v_mov_b32_opcode = _instruction_opcode(
        instructions,
        instruction_name="V_MOV_B32",
        encoding_name="ENC_VOP1",
        condition_name="default",
    )
    s_mov_b32_opcode = _instruction_opcode(
        instructions,
        instruction_name="S_MOV_B32",
        encoding_name="ENC_SOP1",
        condition_name=None,
    )
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.encoding.amdgpu_encoding_tables"),
        "",
        f'#include "{public_header}"',
        "",
        "#include <stdint.h>",
        "",
    ]
    lines.extend(f"const loom_amdgpu_encoding_table_t* {table_view.table_function}(void);" for table_view in table_views if table_view.table_function != table_function)
    if len(table_views) > 1:
        lines.append("")
    lines.append(f"static const loom_amdgpu_encoding_bit_range_t k{table_prefix}BitRanges[] = {{")
    for bit_range, source_bit_offset in compiled_ranges:
        lines.extend(
            [
                "    {",
                f"        .bit_offset = {bit_range.bit_offset},",
                f"        .bit_count = {bit_range.bit_count},",
                f"        .source_bit_offset = {source_bit_offset},",
                f"        .padding_bit_count = {bit_range.padding_bit_count},",
                f"        .padding_value = {bit_range.padding_value},",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            f"static const loom_amdgpu_encoding_field_layout_t k{table_prefix}Fields[] = {{",
        ]
    )
    for field in compiled_fields:
        lines.extend(
            [
                "    {",
                f"        .field_id = {AMDGPU_ENCODING_FIELD_IDS[field.field.name]},",
                f"        .range_start = {field.range_start},",
                f"        .range_count = {field.range_count},",
                f"        .value_bit_count = {field.value_bit_count},",
                f"        .flags = {1 if field.field.is_conditional else 0},",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            f"static const loom_amdgpu_encoding_format_layout_t k{table_prefix}Formats[] = {{",
        ]
    )
    for entry in compiled_formats:
        identifier_words = _split_words(entry.encoding.identifier_values[0], entry.word_count)
        identifier_mask_words = _split_words(entry.encoding.identifier_mask, entry.word_count)
        lines.extend(
            [
                "    {",
                f"        .format_id = {entry.format_id},",
                f"        .bit_count = {entry.encoding.bit_count},",
                f"        .word_count = {entry.word_count},",
                f"        .field_start = {entry.field_start},",
                f"        .field_count = {entry.field_count},",
                f"        .identifier_words = {_emit_word_array(identifier_words)},",
                f"        .identifier_mask_words = {_emit_word_array(identifier_mask_words)},",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            f"static const loom_amdgpu_encoding_inline_f32_source_t k{table_prefix}InlineF32Sources[] = {{",
        ]
    )
    for inline_source in inline_f32_sources:
        lines.extend(
            [
                "    {",
                f"        .bit_pattern = UINT32_C(0x{inline_source.bit_pattern:08x}),",
                f"        .source = UINT16_C({inline_source.source}),",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
        ]
    )
    if not table_views:
        table_views = (
            _EncodingTableView(
                descriptor_set_key=descriptor_set_key,
                table_prefix=table_prefix,
                table_function=table_function,
            ),
        )
    for table_view in table_views:
        lines.extend(
            [
                f"static const loom_amdgpu_encoding_table_t k{table_view.table_prefix}Table = {{",
                f"    .descriptor_set_ordinal = UINT16_C({amdgpu_descriptor_set_ordinal(table_view.descriptor_set_key)}),",
                f'    .descriptor_set_key = IREE_SVL("{table_view.descriptor_set_key}"),',
                f"    .s_mov_b32_opcode = {s_mov_b32_opcode},",
                f"    .v_mov_b32_opcode = {v_mov_b32_opcode},",
                f"    .source_literal = {source_literal},",
                f"    .scalar_inline_u32_zero = {scalar_inline_u32_zero},",
                f"    .scalar_inline_u32_count = {scalar_inline_u32_count},",
                f"    .inline_f32_sources = k{table_prefix}InlineF32Sources,",
                f"    .inline_f32_source_count = IREE_ARRAYSIZE(k{table_prefix}InlineF32Sources),",
                f"    .vector_source_vgpr0 = {vector_source_vgpr0},",
                f"    .vector_source_vgpr_count = {vector_source_vgpr_count},",
                f"    .formats = k{table_prefix}Formats,",
                f"    .format_count = IREE_ARRAYSIZE(k{table_prefix}Formats),",
                f"    .fields = k{table_prefix}Fields,",
                f"    .field_count = IREE_ARRAYSIZE(k{table_prefix}Fields),",
                f"    .bit_ranges = k{table_prefix}BitRanges,",
                f"    .bit_range_count = IREE_ARRAYSIZE(k{table_prefix}BitRanges),",
                "};",
                "",
                f"const loom_amdgpu_encoding_table_t* {table_view.table_function}(void) {{",
                f"  return &k{table_view.table_prefix}Table;",
                "}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--descriptor-set-key", required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--public-header", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument(
        "--view-header",
        action="append",
        default=[],
        help="Generated encoding-table view header as <target>=<path>.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    view_headers = _parse_view_headers(args.view_header)
    descriptor_set_info = amdgpu_descriptor_set_info_by_generator_target(args.target)
    if descriptor_set_info.key != args.descriptor_set_key:
        raise ValueError(f"AMDGPU encoding target {args.target} expects descriptor set '{descriptor_set_info.key}', found '{args.descriptor_set_key}'")
    storage_info = amdgpu_descriptor_set_storage_info_by_generator_target(args.target)
    if storage_info != descriptor_set_info:
        raise ValueError(f"AMDGPU encoding target {args.target} is a view of storage target {storage_info.generator_target}; generate the storage target with --view-header instead")
    view_infos = _view_infos_for_storage_target(descriptor_set_info, view_headers)
    spec = parse_amdgpu_isa_xml_path(args.xml)
    validate_amdgpu_descriptor_set_isa_xml(descriptor_set_info, spec)
    source_literal = spec.operand_predefined_value("OPR_SRC", "SRC_LITERAL")
    scalar_source_literal = spec.operand_predefined_value("OPR_SSRC", "SRC_LITERAL")
    if scalar_source_literal != source_literal:
        raise ValueError(f"{spec.source_name}: OPR_SRC and OPR_SSRC disagree on SRC_LITERAL")
    scalar_inline_u32_zero, scalar_inline_u32_count = _derive_predefined_linear_range(
        spec,
        operand_type_name="OPR_SSRC",
        base_name="0",
        name_pattern=re.compile(r"([0-9]+)"),
        description="OPR_SSRC inline integer",
    )
    inline_f32_sources = _derive_predefined_f32_sources(
        spec,
        operand_type_name="OPR_SRC",
    )
    vector_source_vgpr0, vector_source_vgpr_count = _derive_predefined_linear_range(
        spec,
        operand_type_name="OPR_SRC",
        base_name="v0",
        name_pattern=re.compile(r"v([0-9]+)"),
        description="OPR_SRC VGPR",
    )
    table_prefix = _table_prefix_for_target(args.target)
    table_function = _table_function_for_target(args.target)
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(
        _emit_header_for_target(
            target=args.target,
        ),
        encoding="utf-8",
    )
    for view_info in view_infos:
        view_header_path = view_headers.get(view_info.generator_target)
        if view_header_path is None:
            continue
        view_header_path.parent.mkdir(parents=True, exist_ok=True)
        view_header_path.write_text(
            _emit_header_for_target(
                target=view_info.generator_target,
            ),
            encoding="utf-8",
        )

    table_views: tuple[_EncodingTableView, ...] = ()
    if view_infos:
        table_views = (
            _encoding_table_view_for_info(descriptor_set_info),
            *(_encoding_table_view_for_info(view_info) for view_info in view_infos),
        )
    args.source.write_text(
        _emit_source(
            target=args.target,
            descriptor_set_key=args.descriptor_set_key,
            public_header=args.public_header,
            table_prefix=table_prefix,
            table_function=table_function,
            table_views=table_views,
            encodings=spec.encodings,
            instructions=spec.instructions,
            operand_types=spec.operand_types,
            source_literal=source_literal,
            scalar_inline_u32_zero=scalar_inline_u32_zero,
            scalar_inline_u32_count=scalar_inline_u32_count,
            inline_f32_sources=inline_f32_sources,
            vector_source_vgpr0=vector_source_vgpr0,
            vector_source_vgpr_count=vector_source_vgpr_count,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
