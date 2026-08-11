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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    amdgpu_core_descriptor_set_instruction_names_by_isa_key,
    build_amdgpu_core_descriptor_set_from_specs,
)
from loom.target.arch.amdgpu.encoding import (  # noqa: E402
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FORMAT_XML_NAMES_BY_ID,
    AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE,
    AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_COUNT,
    AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_OFFSET,
    AMDGPU_VOP3_ENCODING_FORMAT_NAMES,
    amdgpu_encoding_field_name,
    amdgpu_supplemental_encoding_format_names,
)
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    AmdgpuIsaBitRange,
    AmdgpuIsaEncoding,
    AmdgpuIsaEncodingField,
    AmdgpuIsaFactSource,
    AmdgpuIsaInstruction,
    AmdgpuIsaOperandType,
    AmdgpuIsaPartitionedOperandUse,
    compose_amdgpu_isa_partitioned_field,
    parse_amdgpu_isa_xml_paths_for_instructions,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AmdgpuDescriptorSetInfo,
    AmdgpuDescriptorSetIsaInfo,
    amdgpu_descriptor_set_info_by_generator_target,
    amdgpu_descriptor_set_ordinal,
    amdgpu_descriptor_set_storage_info_by_generator_target,
    amdgpu_descriptor_set_view_infos_by_storage_generator_target,
)
from loom.target.low_descriptors import Descriptor, DescriptorSet  # noqa: E402


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
        _field("MATRIX_A_FMT", _bit_range(11, 3)),
        _field("MATRIX_B_FMT", _bit_range(59, 2), _bit_range(14, 1)),
    )


def _gfx125x_vop3_supplemental_fields() -> tuple[AmdgpuIsaEncodingField, ...]:
    return (
        _field(
            "SCALE_SEL",
            _bit_range(
                AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_OFFSET,
                AMDGPU_GFX125X_VOP3_SCALE_SEL_BIT_COUNT,
            ),
        ),
    )


def _rdna4_vop3px2_supplemental_encoding() -> AmdgpuIsaEncoding:
    return AmdgpuIsaEncoding(
        name="ENC_VOP3PX2",
        order=0,
        bit_count=128,
        identifier_mask=(0xFF << 24) | (0x1FF << 50) | (0xFF << 88),
        identifier_values=((0xCC << 24) | (0x080 << 50) | (0xCC << 88),),
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
            _field("OPSEL_HI", _bit_range(123, 2), _bit_range(78, 1)),
            _field("CLAMP", _bit_range(79, 1)),
            _field("OP", _bit_range(80, 8)),
            _field("SRC0", _bit_range(96, 9)),
            _field("SRC1", _bit_range(105, 9)),
            _field("SRC2", _bit_range(114, 9)),
        ),
    )


def _gfx125x_vop1_vgpr_supplemental_encoding() -> AmdgpuIsaEncoding:
    # VSRC0[7] selects the high 16-bit half of v[0:127], not v[128:255].
    # Loom allocates packed 16-bit values in the low half of a whole VGPR, so
    # this supported form fixes the selector to zero and exposes the register
    # index as a seven-bit field.
    return AmdgpuIsaEncoding(
        name="ENC_VOP1_VGPR",
        order=0,
        bit_count=32,
        identifier_mask=(0x7F << 25) | (1 << 16) | (1 << 7),
        identifier_values=(0x3F << 25,),
        fields=(
            _field("OP", _bit_range(8, 8)),
            _field("VSRC0", _bit_range(0, 7)),
            _field("VDST", _bit_range(17, 8)),
        ),
    )


def _supplemental_fields_by_encoding(
    target: str,
) -> dict[str, tuple[AmdgpuIsaEncodingField, ...]]:
    fields_by_encoding: dict[str, tuple[AmdgpuIsaEncodingField, ...]] = {}
    if target in ("rdna4", "rdna4_gfx125x"):
        fields_by_encoding["ENC_VOP3P"] = _rdna4_vop3p_supplemental_fields()
    if target == "rdna4_gfx125x":
        fields_by_encoding["ENC_VOP3"] = _gfx125x_vop3_supplemental_fields()
    return fields_by_encoding


def _replacement_fields_by_encoding(
    target: str,
) -> dict[str, tuple[AmdgpuIsaEncodingField, ...]]:
    if target == "rdna4_gfx125x":
        return {
            "ENC_VOP3P": (_field("OP", _bit_range(16, 8)),),
        }
    return {}


_SUPPLEMENTAL_ENCODING_BUILDERS = {
    "ENC_VOP3PX2": _rdna4_vop3px2_supplemental_encoding,
    "ENC_VOP1_VGPR": _gfx125x_vop1_vgpr_supplemental_encoding,
}


def _supplemental_encodings(target: str) -> tuple[AmdgpuIsaEncoding, ...]:
    return tuple(_SUPPLEMENTAL_ENCODING_BUILDERS[format_name]() for format_name in amdgpu_supplemental_encoding_format_names(target))


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


def _replace_encoding_fields(
    encoding: AmdgpuIsaEncoding,
    replacement_fields: tuple[AmdgpuIsaEncodingField, ...],
) -> AmdgpuIsaEncoding:
    if not replacement_fields:
        return encoding
    replacements_by_name = {field.name: field for field in replacement_fields}
    existing_names = {field.name for field in encoding.fields}
    missing_names = sorted(replacements_by_name.keys() - existing_names)
    if missing_names:
        missing_text = ", ".join(missing_names)
        raise ValueError(f"AMDGPU encoding '{encoding.name}' cannot replace missing fields: {missing_text}")
    return replace(
        encoding,
        fields=tuple(replacements_by_name.get(field.name, field) for field in encoding.fields),
    )


def _with_supplemental_encodings(target: str, encodings: tuple[AmdgpuIsaEncoding, ...]) -> tuple[AmdgpuIsaEncoding, ...]:
    fields_by_encoding = _supplemental_fields_by_encoding(target)
    replacement_fields_by_encoding = _replacement_fields_by_encoding(target)
    supplemental_encodings = _supplemental_encodings(target)
    supplemental_names = {encoding.name for encoding in supplemental_encodings}
    output = []
    for encoding in encodings:
        if encoding.name in supplemental_names:
            raise ValueError(f"AMDGPU encoding target '{target}' XML now defines supplemental encoding '{encoding.name}'")
        encoding = _replace_encoding_fields(encoding, replacement_fields_by_encoding.get(encoding.name, ()))
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


@dataclass(frozen=True, slots=True)
class _EncodingFieldContract:
    field: AmdgpuIsaEncodingField
    value_bit_count: int
    ranges: tuple[tuple[AmdgpuIsaBitRange, int], ...]


@dataclass(frozen=True, slots=True)
class _DescriptorEncodingContract:
    descriptor_key: str
    format_id: int
    bit_count: int
    word_count: int
    identifier_seed: int
    fields: tuple[_EncodingFieldContract, ...]


@dataclass(frozen=True, slots=True)
class _EncodingContract:
    descriptors: tuple[_DescriptorEncodingContract, ...]
    source_literal: int
    scalar_source_literal: int
    scalar_inline_u32: tuple[int, int]
    inline_f32_sources: tuple[_InlineF32Source, ...]
    vector_source_vgprs: tuple[int, int]
    s_mov_b32_opcode: int
    v_mov_b32_opcode: int


def _parse_isa_xml_paths(
    values: Sequence[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition(":")
        if not separator or not key or not path:
            raise ValueError("AMDGPU encoding --isa-xml entries must be key:path pairs")
        if key in paths:
            raise ValueError(f"AMDGPU encoding ISA XML key '{key}' is duplicate")
        paths[key] = Path(path)
    return paths


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


def _encoding_with_field_default(
    encoding: AmdgpuIsaEncoding,
    field: AmdgpuIsaEncodingField,
    value: int,
) -> AmdgpuIsaEncoding:
    field_ranges, value_bit_count = _compile_field_ranges(field.ranges)
    if value < 0 or (value_bit_count < 64 and value >> value_bit_count):
        raise ValueError(f"AMDGPU encoding '{encoding.name}' field '{field.name}' default value {value} does not fit {value_bit_count} bits")
    identifier_values = []
    for identifier_value in encoding.identifier_values:
        for bit_range, source_bit_offset in field_ranges:
            if bit_range.padding_bit_count:
                padding_mask = (1 << bit_range.padding_bit_count) - 1
                padding_value = (value >> source_bit_offset) & padding_mask
                if padding_value != bit_range.padding_value:
                    raise ValueError(f"AMDGPU encoding '{encoding.name}' field '{field.name}' default value {value} violates its required source padding")
            value_bit_offset = source_bit_offset + bit_range.padding_bit_count
            range_mask = (1 << bit_range.bit_count) - 1
            range_value = (value >> value_bit_offset) & range_mask
            target_mask = range_mask << bit_range.bit_offset
            identifier_value = (identifier_value & ~target_mask) | range_value << bit_range.bit_offset
        identifier_values.append(identifier_value)
    return replace(encoding, identifier_values=tuple(identifier_values))


def _with_vop3_unused_source_defaults(
    encodings: tuple[AmdgpuIsaEncoding, ...],
    unused_source_value: int | None,
) -> tuple[AmdgpuIsaEncoding, ...]:
    if unused_source_value is None:
        return encodings
    output = []
    for encoding in encodings:
        if encoding.name not in AMDGPU_VOP3_ENCODING_FORMAT_NAMES:
            output.append(encoding)
            continue
        fields_by_name = _encoding_fields_by_name(encoding)
        if not all(name in fields_by_name for name in ("SRC0", "SRC1", "SRC2")):
            output.append(encoding)
            continue
        for field_name in ("SRC0", "SRC1", "SRC2"):
            encoding = _encoding_with_field_default(
                encoding,
                fields_by_name[field_name],
                unused_source_value,
            )
        output.append(encoding)
    return tuple(output)


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
    partitioned_operand_uses: tuple[AmdgpuIsaPartitionedOperandUse, ...],
    operand_types: tuple[AmdgpuIsaOperandType, ...],
) -> dict[str, tuple[AmdgpuIsaEncodingField, ...]]:
    encodings_by_name = {encoding.name: encoding for encoding in encodings}
    operand_types_by_name = {operand_type.name: operand_type for operand_type in operand_types}
    fields_by_encoding: dict[str, dict[str, AmdgpuIsaEncodingField]] = {encoding.name: {} for encoding in encodings}
    ambiguous_fields_by_encoding: dict[str, set[str]] = {encoding.name: set() for encoding in encodings}
    for use in partitioned_operand_uses:
        encoding = encodings_by_name.get(use.encoding_name)
        if encoding is None:
            raise ValueError(f"AMDGPU instruction '{use.instruction_name}' references missing encoding '{use.encoding_name}'")
        operand_type = operand_types_by_name.get(use.operand_type)
        if operand_type is None:
            raise ValueError(f"AMDGPU instruction '{use.instruction_name}' references missing operand type '{use.operand_type}'")
        if not operand_type.is_partitioned:
            raise ValueError(f"AMDGPU instruction '{use.instruction_name}' references non-partitioned operand type '{use.operand_type}' as partitioned")
        if use.field_name is None:
            raise ValueError(f"AMDGPU instruction '{use.instruction_name}' encoding '{use.encoding_name}' has partitioned binary microcode operand without a field name")
        base_fields = _encoding_fields_by_name(encoding)
        base_field = base_fields.get(use.field_name)
        if base_field is None:
            raise ValueError(f"AMDGPU instruction '{use.instruction_name}' binary microcode operand '{use.field_name}' is not a field in encoding '{encoding.name}'")
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


def _descriptor_encoding_field_names(descriptor: Descriptor) -> set[str]:
    field_names = {"OP"}
    field_names.update(amdgpu_encoding_field_name(operand.encoding_field_id) for operand in descriptor.operands if operand.encoding_field_id != 0)
    for immediate in descriptor.immediates:
        if immediate.encoding_field_id != 0:
            field_names.add(amdgpu_encoding_field_name(immediate.encoding_field_id))
        field_names.update(amdgpu_encoding_field_name(encoding_slice.encoding_field_id) for encoding_slice in immediate.encoding_slices if encoding_slice.encoding_field_id != 0)
    field_names.update(amdgpu_encoding_field_name(field_value.encoding_field_id) for field_value in descriptor.encoding_field_values if field_value.encoding_field_id != 0)
    return field_names


def _identifier_seed_without_fields(
    identifier_value: int,
    fields: Sequence[AmdgpuIsaEncodingField],
) -> int:
    identifier_seed = identifier_value
    for field in fields:
        for bit_range in field.ranges:
            field_mask = ((1 << bit_range.bit_count) - 1) << bit_range.bit_offset
            identifier_seed &= ~field_mask
    return identifier_seed


def _compile_encoding_contract(
    target: str,
    spec: AmdgpuIsaFactSource,
    descriptor_set: DescriptorSet,
    vop3_unused_source_value: int | None,
) -> _EncodingContract:
    encodings = _with_supplemental_encodings(target, spec.encodings)
    encodings = _with_vop3_unused_source_defaults(encodings, vop3_unused_source_value)
    compiled_formats, compiled_fields, compiled_ranges = _compile_formats(
        encodings,
        _partitioned_fields_by_encoding(
            encodings,
            spec.partitioned_operand_uses,
            spec.operand_types,
        ),
    )
    compiled_formats_by_id = {compiled_format.format_id: compiled_format for compiled_format in compiled_formats}
    descriptor_contracts = []
    for descriptor in descriptor_set.descriptors:
        if descriptor.encoding_format_id == 0:
            continue
        compiled_format = compiled_formats_by_id.get(descriptor.encoding_format_id)
        if compiled_format is None:
            raise ValueError(f"{spec.source_name}: AMDGPU encoding table is missing descriptor '{descriptor.key}' format id {descriptor.encoding_format_id}")
        field_names = _descriptor_encoding_field_names(descriptor)
        compiled_format_fields = compiled_fields[compiled_format.field_start : compiled_format.field_start + compiled_format.field_count]
        fields_by_name = {compiled_field.field.name: compiled_field for compiled_field in compiled_format_fields}
        missing_fields = sorted(field_names - fields_by_name.keys())
        if missing_fields:
            encoding_name = AMDGPU_ENCODING_FORMAT_XML_NAMES_BY_ID[compiled_format.format_id]
            raise ValueError(f"{spec.source_name}: AMDGPU encoding format '{encoding_name}' for descriptor '{descriptor.key}' is missing fields: {', '.join(missing_fields)}")
        field_contracts: list[_EncodingFieldContract] = []
        for field_name in sorted(field_names):
            compiled_field = fields_by_name[field_name]
            ranges = compiled_ranges[compiled_field.range_start : compiled_field.range_start + compiled_field.range_count]
            field_contracts.append(
                _EncodingFieldContract(
                    field=compiled_field.field,
                    value_bit_count=compiled_field.value_bit_count,
                    ranges=tuple(ranges),
                )
            )
        descriptor_contracts.append(
            _DescriptorEncodingContract(
                descriptor_key=descriptor.key,
                format_id=compiled_format.format_id,
                bit_count=compiled_format.encoding.bit_count,
                word_count=compiled_format.word_count,
                identifier_seed=_identifier_seed_without_fields(
                    compiled_format.encoding.identifier_values[0],
                    tuple(contract.field for contract in field_contracts),
                ),
                fields=tuple(field_contracts),
            )
        )
    return _EncodingContract(
        descriptors=tuple(descriptor_contracts),
        source_literal=spec.operand_predefined_value("OPR_SRC", "SRC_LITERAL"),
        scalar_source_literal=spec.operand_predefined_value("OPR_SSRC", "SRC_LITERAL"),
        scalar_inline_u32=_derive_predefined_linear_range(
            spec,
            operand_type_name="OPR_SSRC",
            base_name="0",
            name_pattern=re.compile(r"([0-9]+)"),
            description="OPR_SSRC inline integer",
        ),
        inline_f32_sources=_derive_predefined_f32_sources(
            spec,
            operand_type_name="OPR_SRC",
        ),
        vector_source_vgprs=_derive_predefined_linear_range(
            spec,
            operand_type_name="OPR_SRC",
            base_name="v0",
            name_pattern=re.compile(r"v([0-9]+)"),
            description="OPR_SRC VGPR",
        ),
        s_mov_b32_opcode=_instruction_opcode(
            spec.instructions,
            instruction_name="S_MOV_B32",
            encoding_name="ENC_SOP1",
            condition_name=None,
        ),
        v_mov_b32_opcode=_instruction_opcode(
            spec.instructions,
            instruction_name="V_MOV_B32",
            encoding_name="ENC_VOP1",
            condition_name="default",
        ),
    )


def _project_encoding_contract(
    contract: _EncodingContract,
    descriptor_keys: Sequence[str],
) -> _EncodingContract:
    descriptors_by_key = {descriptor.descriptor_key: descriptor for descriptor in contract.descriptors}
    descriptors: list[_DescriptorEncodingContract] = []
    for descriptor_key in descriptor_keys:
        descriptor = descriptors_by_key.get(descriptor_key)
        if descriptor is None:
            raise ValueError(f"AMDGPU encoding storage contract is missing view descriptor '{descriptor_key}'")
        descriptors.append(descriptor)
    return replace(contract, descriptors=tuple(descriptors))


def _validate_view_encoding_contract(
    storage_target: str,
    storage_spec: AmdgpuIsaFactSource,
    storage_contract: _EncodingContract,
    view_info: AmdgpuDescriptorSetInfo,
    view_descriptor_set: DescriptorSet,
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
) -> None:
    view_storage_contract = _project_encoding_contract(
        storage_contract,
        tuple(descriptor.key for descriptor in view_descriptor_set.descriptors if descriptor.encoding_format_id != 0),
    )
    for isa_info in view_info.isa_infos:
        member_spec = isa_specs[isa_info.isa_xml_key]
        member_contract = (
            view_storage_contract
            if member_spec is storage_spec
            else _compile_encoding_contract(
                storage_target,
                member_spec,
                view_descriptor_set,
                _vop3_unused_source_value(member_spec, isa_info),
            )
        )
        if member_contract != view_storage_contract:
            raise ValueError(
                f"AMDGPU encoding view '{view_info.key}' does not have a common encoding contract across storage ISA '{storage_spec.source_name}' and member ISA '{member_spec.source_name}'"
            )


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


def _vop3_unused_source_value(
    spec: AmdgpuIsaFactSource,
    isa_info: AmdgpuDescriptorSetIsaInfo,
) -> int | None:
    value_name = isa_info.vop3_unused_source_value
    if value_name is None:
        return None
    return spec.operand_predefined_value("OPR_SRC", value_name)


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
        if "." not in predefined_value.name and predefined_value.name != "0":
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
    partitioned_operand_uses: tuple[AmdgpuIsaPartitionedOperandUse, ...],
    instructions: tuple[AmdgpuIsaInstruction, ...],
    operand_types: tuple[AmdgpuIsaOperandType, ...],
    source_literal: int,
    scalar_inline_u32_zero: int,
    scalar_inline_u32_count: int,
    inline_f32_sources: tuple[_InlineF32Source, ...],
    vector_source_vgpr0: int,
    vector_source_vgpr_count: int,
    vop3_unused_source_value: int | None,
) -> str:
    encodings = _with_supplemental_encodings(target, encodings)
    encodings = _with_vop3_unused_source_defaults(encodings, vop3_unused_source_value)
    compiled_formats, compiled_fields, compiled_ranges = _compile_formats(
        encodings,
        _partitioned_fields_by_encoding(
            encodings,
            partitioned_operand_uses,
            operand_types,
        ),
    )
    maximum_format_field_count = max(
        (compiled_format.field_count for compiled_format in compiled_formats),
        default=0,
    )
    v_mov_b32_opcode = _instruction_opcode(
        instructions,
        instruction_name="V_MOV_B32",
        encoding_name="ENC_VOP1",
        condition_name="default",
    )
    v_nop_opcode = _instruction_opcode(
        instructions,
        instruction_name="V_NOP",
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
        "static_assert(",
        "    LOOM_AMDGPU_ENCODING_PACKET_FIELD_VALUE_CAPACITY >=",
        f"        {maximum_format_field_count},",
        '    "AMDGPU packet field workspace is too small for this target");',
        "",
    ]
    if target == "rdna4_gfx125x":
        lines.extend(
            [
                "static_assert(",
                "    LOOM_AMDGPU_VGPR_MSB_WINDOW_SIZE ==",
                f"        {AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE},",
                '    "gfx125x descriptor and native encoding VGPR windows disagree");',
                "",
            ]
        )
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
                f"    .v_nop_opcode = {v_nop_opcode},",
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
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
    )
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
    instruction_names_by_isa_key = {
        isa_key: set(instruction_names) for isa_key, instruction_names in (amdgpu_core_descriptor_set_instruction_names_by_isa_key((descriptor_set_info, *view_infos)).items())
    }
    for isa_info in descriptor_set_info.isa_infos:
        instruction_names_by_isa_key[isa_info.isa_xml_key].add("V_NOP")
    isa_specs = parse_amdgpu_isa_xml_paths_for_instructions(
        _parse_isa_xml_paths(args.isa_xml),
        instruction_names_by_isa_key,
    )
    if len(descriptor_set_info.isa_infos) != 1:
        raise ValueError(f"AMDGPU encoding storage target '{args.target}' must have one ISA member")
    storage_isa_info = descriptor_set_info.isa_infos[0]
    try:
        spec = isa_specs[storage_isa_info.isa_xml_key]
    except KeyError as exc:
        raise ValueError(f"AMDGPU encoding target '{args.target}' is missing ISA XML key '{storage_isa_info.isa_xml_key}'") from exc
    if spec.architecture_name != storage_isa_info.isa_architecture_name or spec.architecture_id != storage_isa_info.isa_architecture_id:
        raise ValueError(f"{spec.source_name}: AMDGPU encoding target '{args.target}' expects {storage_isa_info.isa_architecture_name} architecture id {storage_isa_info.isa_architecture_id}")
    storage_descriptor_set = build_amdgpu_core_descriptor_set_from_specs(
        args.target,
        isa_specs,
    )
    vop3_unused_source_value = _vop3_unused_source_value(spec, storage_isa_info)
    storage_contract = _compile_encoding_contract(
        args.target,
        spec,
        storage_descriptor_set,
        vop3_unused_source_value,
    )
    for view_info in view_infos:
        view_descriptor_set = build_amdgpu_core_descriptor_set_from_specs(
            view_info.generator_target,
            isa_specs,
        )
        _validate_view_encoding_contract(
            args.target,
            spec,
            storage_contract,
            view_info,
            view_descriptor_set,
            isa_specs,
        )
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
    if args.target == "rdna4_gfx125x" and vector_source_vgpr_count != AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE:
        raise ValueError(f"{spec.source_name}: OPR_SRC exposes {vector_source_vgpr_count} VGPRs; gfx125x S_SET_VGPR_MSB expects {AMDGPU_GFX125X_VGPR_MSB_WINDOW_SIZE}")
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
            partitioned_operand_uses=spec.partitioned_operand_uses,
            instructions=spec.instructions,
            operand_types=spec.operand_types,
            source_literal=source_literal,
            scalar_inline_u32_zero=scalar_inline_u32_zero,
            scalar_inline_u32_count=scalar_inline_u32_count,
            inline_f32_sources=inline_f32_sources,
            vector_source_vgpr0=vector_source_vgpr0,
            vector_source_vgpr_count=vector_source_vgpr_count,
            vop3_unused_source_value=vop3_unused_source_value,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
