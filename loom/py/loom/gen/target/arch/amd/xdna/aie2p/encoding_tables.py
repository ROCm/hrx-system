# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates package-owned AIE2P encoding, machine, and descriptor tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from loom.gen.support.c import c_string_literal
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.arch.amd.xdna.aie2p import machine_tables
from loom.gen.target.low.compiler import compile_descriptor_set
from loom.gen.target.low.low_descriptors import generate_descriptor_set
from loom.target.arch.amd.xdna.aie.encoding import (
    BundleFieldEncoding,
    EncodingTable,
    InstructionEncoding,
    InstructionFieldEncoding,
    encode_witness,
    instruction_fixed_pattern_overlaps,
    validate_encoding_table,
)
from loom.target.arch.amd.xdna.aie.machine import MachineOperandKind
from loom.target.arch.amd.xdna.aie2p.array_descriptors import (
    AIE2P_ARRAY_DESCRIPTOR_SET,
)
from loom.target.arch.amd.xdna.aie2p.configuration_descriptors import (
    AIE2P_CONFIGURATION_DESCRIPTOR_SET,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_ACCUMULATOR_1024_MOVE_SCHEDULE_CLASS_NAME,
    AIE2P_CORE_DESCRIPTOR_SET,
    AIE2P_FIFO_512_MOVE_SCHEDULE_CLASS_NAME,
    AIE2P_NATIVE_MOVE_SCHEDULE_CLASS_NAMES,
    AIE2P_SCALAR_MOVE_COMMON_SCHEDULE_CLASS_NAME,
    AIE2P_SCALAR_MOVE_CORE_ID_SCHEDULE_CLASS_NAME,
    AIE2P_SCALAR_MOVE_DESTINATION_SCHEDULE_CLASS_NAMES,
    AIE2P_SCALAR_MOVE_SOURCE_SCHEDULE_CLASS_NAMES,
)
from loom.target.arch.amd.xdna.aie2p.core_encoding_data import (
    CORE_ENCODING_TABLE,
    CORE_ENCODING_WITNESSES,
    LLVM_AIE_SOURCE_COMMIT,
    SLOT_BIT_COUNTS,
)
from loom.target.arch.amd.xdna.aie2p.core_machine_data import (
    BFP_STORAGE_FIELDS,
    CORE_MACHINE_TABLE,
    DIMENSION_FIELDS,
)
from loom.target.arch.amd.xdna.aie2p.core_move_topology import (
    AGGREGATE_MOVE_COMPONENT_FORMS,
    AGGREGATE_MOVE_RECIPES,
    SCALAR_MOVE_TOPOLOGY,
)
from loom.target.low_descriptors import DescriptorFlag

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


def _validate_singleton_bundle_formats(table: EncodingTable) -> None:
    """Proves that native packet placement can issue every instruction alone."""
    singleton_slots = {bundle_format.fields[0].slot for bundle_format in table.bundle_formats if len(bundle_format.fields) == 1}
    # AIE2P's exact slot signatures are not downward closed. Membership in a
    # larger format does not let an instruction start an otherwise empty packet.
    for slot in sorted({instruction.slot for instruction in table.instructions}):
        if slot not in singleton_slots:
            raise ValueError(f"AIE2P instruction slot '{slot}' has no exact singleton bundle format")


def _emit_encoding_tables() -> str:
    validate_encoding_table(CORE_ENCODING_TABLE, SLOT_BIT_COUNTS)
    _validate_singleton_bundle_formats(CORE_ENCODING_TABLE)
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
    bundle_layout_lines = ["    {.fixed_mask = {0}},"]
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


def _register_part_unit_masks(compiled) -> list[int]:
    """Projects semantic parts onto physical atomic-unit ordinals once."""
    spec = AIE2P_CORE_DESCRIPTOR_SET
    registers = {row.name: row for row in CORE_MACHINE_TABLE.physical_registers}
    classes = {row.name: row for row in spec.reg_classes}
    # Vector 128-bit halves share an indivisible 256-bit alias unit. Other
    # parts follow the named physical subregister hierarchy, not unit-ID order.
    paths = {
        "aie2p.mexa": tuple(path for _, path, _ in BFP_STORAGE_FIELDS),
        "aie2p.elpredicate": (("sub_l_even",), ("sub_l_odd",)),
        "aie2p.vec256": ((), ()),
        "aie2p.eldfiforeg": (("sub_lo_fifo",), ("sub_hi_fifo",)),
        "aie2p.mstfifo": (("sub_lo_fifo",), ("sub_hi_fifo",)),
        **{f"aie2p.{name.lower()}": tuple(path for _, path, _ in fields) for name, fields in DIMENSION_FIELDS.items()},
    }
    if any(len(row.atomic_units) > 32 for row in registers.values()):
        raise ValueError("AIE2P register units exceed uint32 ordinal masks")
    result = []
    for part in compiled.register_parts:
        if part.mask >> len(paths[part.reg_class]):
            raise ValueError(f"{part.name}: register part has no physical projection")
        masks = set()
        for name in classes[part.reg_class].physical_registers:
            register = registers[name]
            selected_units = set()
            for index, path in enumerate(paths[part.reg_class]):
                if not part.mask & (1 << index):
                    continue
                selected = register
                for step in path:
                    selected = registers[selected.subregisters[selected.subregister_indices.index(step)]]
                selected_units.update(selected.atomic_units)
            if not selected_units or not selected_units <= set(register.atomic_units):
                raise ValueError(f"{part.name}: invalid unit projection for {name}")
            masks.add(sum(1 << index for index, unit in enumerate(register.atomic_units) if unit in selected_units))
        if len(masks) != 1 or max(masks) > 255:
            raise ValueError(f"{part.name}: candidates require distinct or wider unit masks")
        result.append(masks.pop())
    return result


def _emit_register_tables() -> str:
    """Indexes native moves and register parts by their producer-owned IDs."""
    spec = AIE2P_CORE_DESCRIPTOR_SET
    compiled = compile_descriptor_set(
        spec,
        required_schedule_class_names=AIE2P_NATIVE_MOVE_SCHEDULE_CLASS_NAMES,
    )
    moves = tuple((ordinal, descriptor) for ordinal, descriptor in enumerate(spec.descriptors) if DescriptorFlag.ALLOCATION_MOVE in descriptor.flags)
    if not moves or len(moves) > 8:
        raise ValueError("AIE2P allocation moves must fit nonempty uint8 route masks")
    if moves[-1][0] > 0xFFFF:
        raise ValueError("AIE2P allocation move ordinals exceed uint16")
    register_ids = {register.name: ordinal for ordinal, register in enumerate(spec.physical_registers)}
    classes = {row.name: row for row in spec.reg_classes}
    masks = [[0] * len(register_ids) for _ in range(2)]
    for move_index, (_ordinal, descriptor) in enumerate(moves):
        for operand_index, operand in enumerate(descriptor.operands):
            register_class = classes[operand.reg_alts[0].reg_class]
            for register_name in register_class.physical_registers:
                masks[operand_index][register_ids[register_name]] |= 1 << move_index
    lines = [
        *line_comment_header("//", generator="loom.gen.target.arch.amd.xdna.aie2p.encoding_tables"),
        "static const uint16_t kMoveDescriptorOrdinals[] = {",
        *(f"    {ordinal}," for ordinal, _descriptor in moves),
        "};",
        "",
    ]
    for name, values in zip(("Destination", "Source"), masks, strict=True):
        lines.extend((f"static const uint8_t kMove{name}Masks[] = {{", *(f"    0x{value:02x}," for value in values), "};", ""))

    def category_table(categories: tuple[tuple[str, ...], ...], name: str) -> list[int]:
        values = [0xFF] * len(register_ids)
        for category, registers in enumerate(categories):
            for register in registers:
                register_id = register_ids[register]
                if values[register_id] != 0xFF:
                    raise ValueError(f"{register}: duplicate {name} move category")
                values[register_id] = category
        return values

    scalar_destination_categories = category_table(SCALAR_MOVE_TOPOLOGY.destination_categories, "destination")
    scalar_source_categories = category_table(SCALAR_MOVE_TOPOLOGY.source_categories, "source")
    lines.extend(
        (
            "static const uint8_t kMoveScalarDestinationCategories[] = {",
            *(f"    {value}," for value in scalar_destination_categories),
            "};",
            "",
            "static const uint8_t kMoveScalarSourceCategories[] = {",
            *(f"    {value}," for value in scalar_source_categories),
            "};",
            "",
        )
    )

    native_classes = {row.name: row for row in CORE_MACHINE_TABLE.register_classes}
    native_registers = {row.name: row for row in CORE_MACHINE_TABLE.physical_registers}
    projections_by_register: dict[str, tuple[int, tuple[int, ...]]] = {}
    for recipe_index, recipe in enumerate(AGGREGATE_MOVE_RECIPES):
        for class_name in recipe.machine_classes:
            for register_name in native_classes[class_name].candidates:
                register = native_registers[register_name]
                components = []
                for path in recipe.component_paths:
                    component = register
                    for step in path:
                        component = native_registers[component.subregisters[component.subregister_indices.index(step)]]
                    components.append(register_ids[component.name])
                projection = (recipe_index, tuple(components))
                previous = projections_by_register.setdefault(register_name, projection)
                if previous != projection:
                    raise ValueError(f"{register_name}: conflicting aggregate move projections")

    direct_route_first = 1
    scalar_route_first = direct_route_first + len(moves)
    scalar_route_count = len(SCALAR_MOVE_TOPOLOGY.destination_categories) * len(SCALAR_MOVE_TOPOLOGY.source_categories)
    aggregate_route_first = scalar_route_first + scalar_route_count
    route_limit = aggregate_route_first + len(AGGREGATE_MOVE_RECIPES)
    if route_limit > 256:
        raise ValueError("AIE2P move routes exceed uint8 identifiers")
    lines.extend(
        (
            "enum {",
            f"  kMoveRouteDirectFirst = {direct_route_first},",
            f"  kMoveRouteScalarFirst = {scalar_route_first},",
            f"  kMoveRouteScalarSourceCount = {len(SCALAR_MOVE_TOPOLOGY.source_categories)},",
            f"  kMoveRouteScalarDestinationCount = {len(SCALAR_MOVE_TOPOLOGY.destination_categories)},",
            f"  kMoveRouteAggregateFirst = {aggregate_route_first},",
            f"  kMoveRouteLimit = {route_limit},",
            "};",
            "",
        )
    )

    projection_indices = [0] * len(register_ids)
    projection_rows = [(0, 0, 0)]
    component_registers: list[int] = []
    for register_name, register_id in register_ids.items():
        projection = projections_by_register.get(register_name)
        if projection is None:
            continue
        recipe_index, components = projection
        if len(component_registers) > 0xFFFF:
            raise ValueError("AIE2P move projection components exceed uint16 offsets")
        projection_indices[register_id] = len(projection_rows)
        projection_rows.append(
            (
                len(component_registers),
                len(components),
                aggregate_route_first + recipe_index,
            )
        )
        component_registers.extend(components)
    if len(projection_rows) > 256:
        raise ValueError("AIE2P move projections exceed uint8 identifiers")
    lines.extend(
        (
            "static const uint8_t kMoveProjectionIndices[] = {",
            *(f"    {value}," for value in projection_indices),
            "};",
            "",
            "static const loom_aie2p_move_projection_t kMoveProjections[] = {",
            *(f"    {{{component_start}, {component_count}, {route_id}}}," for component_start, component_count, route_id in projection_rows),
            "};",
            "",
            "static const uint16_t kMoveProjectionComponents[] = {",
            *(f"    {value}," for value in component_registers),
            "};",
            "",
        )
    )

    descriptor_ordinals = {descriptor.key: ordinal for ordinal, descriptor in enumerate(spec.descriptors)}
    scalar_descriptor_ordinal = descriptor_ordinals["amd.xdna.aie2p.move.scalar"]
    event_descriptor_ordinal = descriptor_ordinals["amd.xdna.aie2p.move.accumulator512"]
    encoding_kinds = {
        "scalar": "LOOM_AIE2P_MOVE_STEP_ENCODING_DESCRIPTOR",
        "accumulator_1024": "LOOM_AIE2P_MOVE_STEP_ENCODING_ACCUMULATOR_1024",
        "fifo_512": "LOOM_AIE2P_MOVE_STEP_ENCODING_FIFO_512",
    }
    schedule_class_names = {
        "scalar": None,
        "accumulator_1024": AIE2P_ACCUMULATOR_1024_MOVE_SCHEDULE_CLASS_NAME,
        "fifo_512": AIE2P_FIFO_512_MOVE_SCHEDULE_CLASS_NAME,
    }

    instructions = tuple(sorted(CORE_ENCODING_TABLE.instructions, key=lambda row: row.name))
    instruction_ids = {instruction.name: index for index, instruction in enumerate(instructions, start=1)}
    field_ids = {
        name: index
        for index, name in enumerate(
            sorted({field.name for instruction in instructions for field in instruction.fields}),
            start=1,
        )
    }
    forms = {form.name: form for form in CORE_MACHINE_TABLE.forms}
    adapter_ids = {
        adapter.name: index
        for index, adapter in enumerate(
            sorted(CORE_MACHINE_TABLE.register_adapters, key=lambda row: row.name),
            start=1,
        )
    }

    aggregate_encoding_infos = []
    for recipe in AGGREGATE_MOVE_RECIPES:
        if recipe.component_kind == "scalar":
            aggregate_encoding_infos.append((0, 0, 0, 0, 0))
            continue
        form_name = AGGREGATE_MOVE_COMPONENT_FORMS[recipe.component_kind]
        form = forms[form_name]
        instruction = next(row for row in instructions if row.name == form_name)
        if (
            len(form.outputs) != 1
            or len(form.inputs) != 1
            or form.outputs[0].name != "dst"
            or form.inputs[0].name != "src"
            or form.outputs[0].kind is not MachineOperandKind.REGISTER_ADAPTER
            or form.inputs[0].kind is not MachineOperandKind.REGISTER_ADAPTER
            or {field.name for field in instruction.fields} != {"dst", "src"}
        ):
            raise ValueError(f"{form_name}: aggregate move encoding must have adapted dst/src fields")
        aggregate_encoding_infos.append(
            (
                instruction_ids[form_name],
                field_ids["dst"],
                field_ids["src"],
                adapter_ids[form.outputs[0].type_name],
                adapter_ids[form.inputs[0].type_name],
            )
        )
    lines.extend(
        (
            f"static const uint16_t kMoveScalarDescriptorOrdinal = {scalar_descriptor_ordinal};",
            f"static const uint16_t kMoveEventDescriptorOrdinal = {event_descriptor_ordinal};",
            "",
            "static const uint8_t kMoveAggregateStepCounts[] = {",
            *(f"    {len(recipe.component_paths)}," for recipe in AGGREGATE_MOVE_RECIPES),
            "};",
            "",
            "static const uint8_t kMoveAggregateEncodingKinds[] = {",
            *(f"    {encoding_kinds[recipe.component_kind]}," for recipe in AGGREGATE_MOVE_RECIPES),
            "};",
            "",
            "static const uint16_t kMoveAggregateScheduleClassIds[] = {",
            *(
                "    UINT16_MAX," if schedule_class_names[recipe.component_kind] is None else f"    {compiled.schedule_class_ids[schedule_class_names[recipe.component_kind]]},"
                for recipe in AGGREGATE_MOVE_RECIPES
            ),
            "};",
            "",
            "static const loom_aie2p_move_step_encoding_info_t",
            "    kMoveAggregateEncodingInfos[] = {",
            *(
                f"    {{{instruction_id}, {destination_field_id}, {source_field_id}, {destination_adapter_id}, {source_adapter_id}}},"
                for instruction_id, destination_field_id, source_field_id, destination_adapter_id, source_adapter_id in aggregate_encoding_infos
            ),
            "};",
            "",
        )
    )

    def schedule_class_id(name: str | None) -> str:
        return "UINT16_MAX" if name is None else str(compiled.schedule_class_ids[name])

    lines.extend(
        (
            f"static const uint16_t kMoveScalarCommonScheduleClassId = {compiled.schedule_class_ids[AIE2P_SCALAR_MOVE_COMMON_SCHEDULE_CLASS_NAME]};",
            "static const uint16_t kMoveScalarDestinationScheduleClassIds[] = {",
            *(f"    {schedule_class_id(name)}," for name in AIE2P_SCALAR_MOVE_DESTINATION_SCHEDULE_CLASS_NAMES),
            "};",
            "",
            "static const uint16_t kMoveScalarSourceScheduleClassIds[] = {",
            *(f"    {schedule_class_id(name)}," for name in AIE2P_SCALAR_MOVE_SOURCE_SCHEDULE_CLASS_NAMES),
            "};",
            "",
            f"static const uint16_t kMoveScalarCoreIdScheduleClassId = {compiled.schedule_class_ids[AIE2P_SCALAR_MOVE_CORE_ID_SCHEDULE_CLASS_NAME]};",
            "",
        )
    )
    lines.extend(("static const uint8_t kRegisterPartAtomicUnitMasks[] = {", *(f"    0x{mask:02x}," for mask in _register_part_unit_masks(compiled)), "};", ""))
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
        "--register-output",
        type=Path,
        help="Path to write the AIE2P register projection and move tables.",
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
    parser.add_argument("--configuration-descriptor-header-output", type=Path)
    parser.add_argument("--configuration-descriptor-source-output", type=Path)
    args = parser.parse_args(argv)
    output_paths = (
        args.encoding_output,
        args.machine_output,
        args.register_output,
        args.descriptor_header_output,
        args.descriptor_source_output,
        args.array_descriptor_header_output,
        args.array_descriptor_source_output,
        args.configuration_descriptor_header_output,
        args.configuration_descriptor_source_output,
    )
    if args.check and any(path is not None for path in output_paths):
        parser.error("--check cannot be combined with output paths")
    if not args.check and all(path is None for path in output_paths):
        parser.error("at least one AIE2P target output path is required")

    for path, generate in (
        (args.encoding_output, _emit_encoding_tables),
        (args.machine_output, lambda: machine_tables.emit_tables(CORE_MACHINE_TABLE, CORE_ENCODING_TABLE)),
        (args.register_output, _emit_register_tables),
    ):
        if args.check or path is not None:
            contents = generate()
            if path is not None:
                write_text_file(path, contents)

    for descriptor_set, header_path, source_path, schedule_classes in (
        (AIE2P_CORE_DESCRIPTOR_SET, args.descriptor_header_output, args.descriptor_source_output, AIE2P_NATIVE_MOVE_SCHEDULE_CLASS_NAMES),
        (AIE2P_ARRAY_DESCRIPTOR_SET, args.array_descriptor_header_output, args.array_descriptor_source_output, ()),
        (AIE2P_CONFIGURATION_DESCRIPTOR_SET, args.configuration_descriptor_header_output, args.configuration_descriptor_source_output, ()),
    ):
        if args.check or header_path is not None or source_path is not None:
            generated = generate_descriptor_set(descriptor_set, required_schedule_class_names=schedule_classes)
            if header_path is not None:
                write_text_file(header_path, generated.header)
            if source_path is not None:
                write_text_file(source_path, generated.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
