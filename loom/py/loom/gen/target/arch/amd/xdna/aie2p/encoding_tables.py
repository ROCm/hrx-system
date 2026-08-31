# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates one coherent family of AIE2P target tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from loom.gen.support.c import c_string_literal
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.arch.amd.xdna.aie2p import machine_tables
from loom.gen.target.low.low_descriptors import generate_descriptor_set
from loom.target.arch.amd.xdna.aie.encoding import (
    BundleFieldEncoding,
    InstructionEncoding,
    InstructionFieldEncoding,
    encode_witness,
    instruction_fixed_pattern_overlaps,
    validate_encoding_table,
)
from loom.target.arch.amd.xdna.aie2p.array_descriptors import (
    AIE2P_ARRAY_DESCRIPTOR_SET,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import (
    CORE_ENCODING_TABLE,
    CORE_ENCODING_WITNESSES,
    LLVM_AIE_SOURCE_COMMIT,
    SLOT_BIT_COUNTS,
)
from loom.target.arch.amd.xdna.aie2p.core_machine_data import CORE_MACHINE_TABLE

_NATIVE_MAX_PACKET_SIZE = 16
_NATIVE_MAX_BUNDLE_SLOT_COUNT = 8

_MappingKey = tuple[tuple[int, int], ...]
_MappingRange = tuple[int, int, int]


def _hex_u64(value: int) -> str:
    return f"UINT64_C(0x{value:016x})"


def _byte_initializer(value: int) -> str:
    values = value.to_bytes(_NATIVE_MAX_PACKET_SIZE, "little")
    return "{" + ", ".join(f"0x{byte:02x}" for byte in values) + "}"


def _slot_c_id(slot: str) -> str:
    if slot not in SLOT_BIT_COUNTS:
        raise ValueError(f"unknown AIE2P physical slot '{slot}'")
    return f"LOOM_AIE2P_SLOT_{slot.upper()}"


def _mapping_key(
    field: InstructionFieldEncoding | BundleFieldEncoding,
) -> _MappingKey:
    return tuple((mapping.target_bit, mapping.value_bit) for mapping in field.mappings)


def _coalesce_mapping_ranges(mapping_key: _MappingKey) -> tuple[_MappingRange, ...]:
    ranges: list[_MappingRange] = []
    for target_bit, value_bit in mapping_key:
        if ranges:
            target_start, value_start, bit_count = ranges[-1]
            if target_bit == target_start + bit_count and value_bit == value_start + bit_count:
                ranges[-1] = (target_start, value_start, bit_count + 1)
                continue
        ranges.append((target_bit, value_bit, 1))
    return tuple(ranges)


def _mapping_value_bit_count(mapping_key: _MappingKey) -> int:
    return max(value_bit for _target_bit, value_bit in mapping_key) + 1


def _build_name_table(names: Sequence[str], table_name: str) -> tuple[dict[str, tuple[int, int]], list[str]]:
    rows: dict[str, tuple[int, int]] = {}
    offset = 0
    lines = [f"static const char {table_name}[] ="]
    for name in names:
        encoded_name = name.encode("ascii")
        if len(encoded_name) > 0xFF:
            raise ValueError(f"AIE2P table name '{name}' exceeds uint8 length")
        if offset > 0xFFFF:
            raise ValueError(f"{table_name} exceeds uint16 offsets")
        rows[name] = (offset, len(encoded_name))
        lines.append(f'    "{c_string_literal(name)}\\0"')
        offset += len(encoded_name) + 1
    if offset > 0x10000:
        raise ValueError(f"{table_name} exceeds uint16 addressable bytes")
    lines[-1] += ";"
    return rows, lines


def _emit_encoding_tables() -> str:
    validate_encoding_table(CORE_ENCODING_TABLE, SLOT_BIT_COUNTS)
    for witness in CORE_ENCODING_WITNESSES:
        actual_bytes = encode_witness(CORE_ENCODING_TABLE, witness)
        if actual_bytes != witness.expected_bytes:
            raise ValueError(f"{witness.symbol}: generated model produced {actual_bytes.hex()}, expected {witness.expected_bytes.hex()}")

    instructions = tuple(sorted(CORE_ENCODING_TABLE.instructions, key=lambda row: row.name))
    bundle_formats = tuple(sorted(CORE_ENCODING_TABLE.bundle_formats, key=lambda row: row.name))
    instruction_ids = {instruction.name: index for index, instruction in enumerate(instructions, start=1)}
    field_names = tuple(sorted({field.name for instruction in instructions for field in instruction.fields}))
    field_ids = {field_name: index for index, field_name in enumerate(field_names, start=1)}
    if len(instructions) > 0xFFFF:
        raise ValueError("instruction table exceeds uint16 identifiers")
    if len(bundle_formats) > 0xFFFF:
        raise ValueError("bundle format table exceeds uint16 identifiers")
    if len(field_names) > 0xFF:
        raise ValueError("instruction field table exceeds compact uint8 identifiers")

    all_fields = tuple(field for instruction in instructions for field in instruction.fields) + tuple(field for bundle in bundle_formats for field in bundle.fields)
    mapping_keys = tuple(sorted({_mapping_key(field) for field in all_fields}))
    if len(mapping_keys) > 0x100:
        raise ValueError("mapping pattern table exceeds compact uint8 identifiers")
    mapping_pattern_ids = {mapping_key: index for index, mapping_key in enumerate(mapping_keys)}

    mapping_range_lines: list[str] = []
    mapping_pattern_lines: list[str] = []
    mapping_range_count = 0
    for mapping_key in mapping_keys:
        ranges = _coalesce_mapping_ranges(mapping_key)
        if mapping_range_count > 0xFF:
            raise ValueError("mapping range table exceeds compact uint8 offsets")
        if len(ranges) > 0xFF:
            raise ValueError("mapping pattern exceeds compact uint8 range counts")
        value_bit_count = _mapping_value_bit_count(mapping_key)
        if value_bit_count > 64:
            raise ValueError("mapping pattern value width exceeds uint64")
        mapping_pattern_lines.append(f"    {{.bit_range_offset = {mapping_range_count}, .bit_range_count = {len(ranges)}, .value_bit_count = {value_bit_count}}},")
        for target_bit, value_bit, bit_count in ranges:
            mapping_range_lines.append(f"    {{.target_bit = {target_bit}, .value_bit = {value_bit}, .bit_count = {bit_count}}},")
        mapping_range_count += len(ranges)
    if mapping_range_count > 0x100:
        raise ValueError("mapping range table exceeds compact uint8 address space")

    instruction_field_keys = tuple(sorted({(field_ids[field.name], mapping_pattern_ids[_mapping_key(field)]) for instruction in instructions for field in instruction.fields}))
    if len(instruction_field_keys) > 0x100:
        raise ValueError("instruction field layout table exceeds compact uint8 identifiers")
    instruction_field_ids = {field_key: index for index, field_key in enumerate(instruction_field_keys)}
    instruction_field_lines = [f"    {{.field_id = {field_id}, .mapping_pattern_id = {mapping_pattern_id}}}," for field_id, mapping_pattern_id in instruction_field_keys]

    slot_order = tuple(SLOT_BIT_COUNTS)
    slot_indices = {slot: index for index, slot in enumerate(slot_order)}

    def instruction_layout_key(
        instruction: InstructionEncoding,
    ) -> tuple[int, int, tuple[int, ...]]:
        return (
            slot_indices[instruction.slot],
            instruction.fixed_mask,
            tuple(instruction_field_ids[(field_ids[field.name], mapping_pattern_ids[_mapping_key(field)])] for field in instruction.fields),
        )

    instruction_layout_keys = tuple(sorted({instruction_layout_key(instruction) for instruction in instructions}))
    if len(instruction_layout_keys) > 0xFFFF:
        raise ValueError("instruction layout table exceeds uint16 identifiers")
    instruction_layout_ids = {layout_key: index for index, layout_key in enumerate(instruction_layout_keys)}
    instruction_layout_fixed_mask_lines: list[str] = []
    instruction_layout_lines: list[str] = []
    instruction_layout_field_ref_lines: list[str] = []
    for slot_index, fixed_mask, field_refs in instruction_layout_keys:
        field_ref_offset = len(instruction_layout_field_ref_lines)
        if field_ref_offset > 0xFFFF:
            raise ValueError("instruction layout field refs exceed uint16 offsets")
        if len(field_refs) > 0xFF:
            raise ValueError("instruction layout exceeds compact uint8 field counts")
        instruction_layout_fixed_mask_lines.append(f"    {_hex_u64(fixed_mask)},")
        instruction_layout_lines.append(f"    {{.field_ref_offset = UINT16_C({field_ref_offset}), .field_count = {len(field_refs)}, .slot_id = {_slot_c_id(slot_order[slot_index])}}},")
        instruction_layout_field_ref_lines.extend(f"    {field_ref}," for field_ref in field_refs)
    if len(instruction_layout_field_ref_lines) > 0xFFFF:
        raise ValueError("instruction layout field refs exceed uint16 rows")

    instruction_name_rows, instruction_name_lines = _build_name_table(
        [instruction.name for instruction in instructions],
        "kLoomAie2pInstructionNames",
    )
    instruction_form_lines = ["    {0},"]
    instruction_fixed_value_lines = ["    UINT64_C(0),"]
    for instruction in instructions:
        name_offset, name_length = instruction_name_rows[instruction.name]
        layout_id = instruction_layout_ids[instruction_layout_key(instruction)]
        instruction_form_lines.append(
            f"    {{.name_offset = UINT16_C({name_offset}), .layout_id = UINT16_C({layout_id}), .name_length = {name_length}, .delay_slot_count = {instruction.delay_slot_count}}},"
        )
        instruction_fixed_value_lines.append(f"    {_hex_u64(instruction.fixed_value)},")

    bundle_name_rows, bundle_name_lines = _build_name_table(
        [bundle_format.name for bundle_format in bundle_formats],
        "kLoomAie2pBundleNames",
    )
    bundle_field_lines: list[str] = []
    bundle_layout_lines = ["    {0},"]
    for bundle_format in bundle_formats:
        field_offset = len(bundle_field_lines)
        if field_offset > 0xFFFF:
            raise ValueError("bundle field table exceeds uint16 offsets")
        if len(bundle_format.fields) > _NATIVE_MAX_BUNDLE_SLOT_COUNT:
            raise ValueError(f"{bundle_format.name} has {len(bundle_format.fields)} slots; native decoded-bundle capacity is {_NATIVE_MAX_BUNDLE_SLOT_COUNT}")
        bundle_field_lines.extend((f"    {{.slot_id = {_slot_c_id(field.slot)}, .mapping_pattern_id = {mapping_pattern_ids[_mapping_key(field)]}}},") for field in bundle_format.fields)
        name_offset, name_length = bundle_name_rows[bundle_format.name]
        bundle_layout_lines.extend(
            [
                "    {",
                f"        .fixed_mask = {_byte_initializer(bundle_format.fixed_mask)},",
                f"        .fixed_value = {_byte_initializer(bundle_format.fixed_value)},",
                f"        .name_offset = UINT16_C({name_offset}),",
                f"        .field_offset = UINT16_C({field_offset}),",
                f"        .name_length = {name_length},",
                f"        .bit_count = {bundle_format.bit_count},",
                f"        .field_count = {len(bundle_format.fields)},",
                "    },",
            ]
        )
    if len(bundle_field_lines) > 0xFFFF:
        raise ValueError("bundle field table exceeds uint16 rows")

    slot_bit_count_lines = [f"    [{_slot_c_id(slot)}] = {SLOT_BIT_COUNTS[slot]}," for slot in slot_order]
    search_order: list[int] = []
    search_range_lines: list[str] = []
    for slot in slot_order:
        offset = len(search_order)
        slot_instructions = tuple(instruction for instruction in instructions if instruction.slot == slot)
        if offset > 0xFFFF or len(slot_instructions) > 0xFFFF:
            raise ValueError(f"{slot}: instruction search range exceeds uint16")
        search_order.extend(instruction_ids[instruction.name] for instruction in slot_instructions)
        search_range_lines.extend(
            [
                f"    [{_slot_c_id(slot)}] = {{",
                f"        .offset = UINT16_C({offset}),",
                f"        .count = UINT16_C({len(slot_instructions)}),",
                "    },",
            ]
        )
    if len(search_order) > 0xFFFF:
        raise ValueError("instruction search-order table exceeds uint16")

    overlap_lines = [f"//   {left} <-> {right}" for left, right in instruction_fixed_pattern_overlaps(CORE_ENCODING_TABLE)]

    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.amd.xdna.aie2p.encoding_tables",
        ),
        "// Fixed-pattern instruction overlaps retained for candidate decoding:",
        *overlap_lines,
        "",
        "static_assert(LOOM_AIE2P_SLOT_INVALID == 0,",
        '              "AIE2P slot tables reserve index zero");',
        f"static_assert(LOOM_AIE2P_SLOT_COUNT == {len(SLOT_BIT_COUNTS) + 1},",
        '              "AIE2P slot table must cover every public identifier");',
        "static_assert(LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE ==",
        f"                  {_NATIVE_MAX_PACKET_SIZE}u,",
        '              "AIE2P packet storage must match generated tables");',
        "static_assert(LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT ==",
        f"                  {_NATIVE_MAX_BUNDLE_SLOT_COUNT}u,",
        '              "AIE2P decoded slot storage must match generated tables");',
        "",
        "static const iree_string_view_t",
        "    kLoomAie2pEncodingLlvmAieSourceCommit =",
        f'        IREE_SVL("{LLVM_AIE_SOURCE_COMMIT}");',
        "",
        "static const uint8_t",
        "    kLoomAie2pSlotBitCounts[LOOM_AIE2P_SLOT_COUNT] = {",
        *slot_bit_count_lines,
        "};",
        "",
        "static const iree_string_view_t kLoomAie2pEncodingFieldNames[] = {",
        '    IREE_SVL(""),',
        *(f'    IREE_SVL("{c_string_literal(field_name)}"),' for field_name in field_names),
        "};",
        "",
        "static const loom_aie2p_encoding_bit_range_t",
        "    kLoomAie2pEncodingBitRanges[] = {",
        *mapping_range_lines,
        "};",
        "",
        "static const loom_aie2p_encoding_mapping_pattern_t",
        "    kLoomAie2pEncodingMappingPatterns[] = {",
        *mapping_pattern_lines,
        "};",
        "",
        "static const loom_aie2p_instruction_field_layout_t",
        "    kLoomAie2pInstructionFieldLayouts[] = {",
        *instruction_field_lines,
        "};",
        "",
        "static const uint8_t kLoomAie2pInstructionLayoutFieldRefs[] = {",
        *instruction_layout_field_ref_lines,
        "};",
        "",
        "static const uint64_t kLoomAie2pInstructionLayoutFixedMasks[] = {",
        *instruction_layout_fixed_mask_lines,
        "};",
        "",
        "static const loom_aie2p_instruction_layout_t",
        "    kLoomAie2pInstructionLayouts[] = {",
        *instruction_layout_lines,
        "};",
        "",
        *instruction_name_lines,
        "",
        "static const uint64_t kLoomAie2pInstructionFixedValues[] = {",
        *instruction_fixed_value_lines,
        "};",
        "",
        "static const loom_aie2p_instruction_form_t",
        "    kLoomAie2pInstructionForms[] = {",
        *instruction_form_lines,
        "};",
        "",
        "static const loom_aie2p_bundle_field_layout_t",
        "    kLoomAie2pBundleFields[] = {",
        *bundle_field_lines,
        "};",
        "",
        *bundle_name_lines,
        "",
        "static const loom_aie2p_bundle_layout_t kLoomAie2pBundleLayouts[] = {",
        *bundle_layout_lines,
        "};",
        "",
        "static const uint16_t kLoomAie2pInstructionSearchOrder[] = {",
        *(f"    UINT16_C({instruction_id})," for instruction_id in search_order),
        "};",
        "",
        "static const loom_aie2p_instruction_search_range_t",
        "    kLoomAie2pInstructionSearchRanges[LOOM_AIE2P_SLOT_COUNT] = {",
        *search_range_lines,
        "};",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AIE2P target tables.")
    parser.add_argument(
        "--encoding-output",
        type=Path,
        help="Path to write the native AIE2P encoding table include.",
    )
    parser.add_argument(
        "--machine-output",
        type=Path,
        help="Path to write the native AIE2P machine table include.",
    )
    parser.add_argument(
        "--descriptor-header-output",
        type=Path,
        help="Path to write the AIE2P core Low descriptor header.",
    )
    parser.add_argument(
        "--descriptor-source-output",
        type=Path,
        help="Path to write the AIE2P core Low descriptor source.",
    )
    parser.add_argument(
        "--array-descriptor-header-output",
        type=Path,
        help="Path to write the AIE2P array Low descriptor header.",
    )
    parser.add_argument(
        "--array-descriptor-source-output",
        type=Path,
        help="Path to write the AIE2P array Low descriptor source.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generation inputs without writing an output file.",
    )
    args = parser.parse_args(argv)
    output_paths = (
        args.encoding_output,
        args.machine_output,
        args.descriptor_header_output,
        args.descriptor_source_output,
        args.array_descriptor_header_output,
        args.array_descriptor_source_output,
    )
    if args.check and any(path is not None for path in output_paths):
        parser.error("--check cannot be combined with output paths")
    if not args.check and any(path is None for path in output_paths):
        parser.error("all AIE2P target output paths are required")

    encoding_contents = _emit_encoding_tables()
    machine_contents = machine_tables.emit_tables(
        CORE_MACHINE_TABLE,
        CORE_ENCODING_TABLE,
    )
    generated_descriptors = generate_descriptor_set(AIE2P_CORE_DESCRIPTOR_SET)
    generated_array_descriptors = generate_descriptor_set(AIE2P_ARRAY_DESCRIPTOR_SET)
    if args.encoding_output is not None:
        write_text_file(args.encoding_output, encoding_contents)
        write_text_file(args.machine_output, machine_contents)
        write_text_file(args.descriptor_header_output, generated_descriptors.header)
        write_text_file(args.descriptor_source_output, generated_descriptors.source)
        write_text_file(
            args.array_descriptor_header_output,
            generated_array_descriptors.header,
        )
        write_text_file(
            args.array_descriptor_source_output,
            generated_array_descriptors.source,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
