# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Emits compact native AIE2P register and machine-form tables."""

from __future__ import annotations

from collections.abc import Sequence

from loom.gen.support.c import c_string_literal
from loom.gen.support.generated_file import line_comment_header
from loom.target.arch.amd.xdna.aie.encoding import EncodingTable
from loom.target.arch.amd.xdna.aie.machine import (
    MachineOperand,
    MachineOperandKind,
    MachineTable,
    validate_machine_table,
)


def _build_string_table(strings: Sequence[str]) -> tuple[dict[str, int], list[str]]:
    offsets: dict[str, int] = {}
    offset = 0
    lines = ["static const char kLoomAie2pMachineStrings[] ="]
    for value in sorted(set(strings)):
        encoded_value = value.encode("ascii")
        if offset > 0xFFFF:
            raise ValueError("AIE2P machine string offsets exceed uint16")
        offsets[value] = offset
        lines.append(f'    "{c_string_literal(value)}\\0"')
        offset += len(encoded_value) + 1
    if offset > 0x10000:
        raise ValueError("AIE2P machine strings exceed 64 KiB")
    lines[-1] += ";"
    return offsets, lines


def _machine_strings(table: MachineTable) -> tuple[str, ...]:
    result = list(table.atomic_unit_names)
    for register in table.physical_registers:
        result.extend((register.name, register.assembly_name))
        result.extend(register.subregister_indices)
    for register_class in table.register_classes:
        result.append(register_class.name)
        result.extend(register_class.value_types)
    result.extend(adapter.name for adapter in table.register_adapters)
    result.extend(immediate.name for immediate in table.immediates)
    for form in table.forms:
        result.extend((form.name, form.assembly))
        for operand in (*form.outputs, *form.inputs):
            result.extend((operand.name, operand.type_name))
    return tuple(result)


def _operand_key(
    operand: MachineOperand,
    class_ids: dict[str, int],
    adapter_ids: dict[str, int],
    immediate_ids: dict[str, int],
) -> tuple[str, int, int]:
    if operand.kind is MachineOperandKind.REGISTER_CLASS:
        kind = 0
        type_id = class_ids[operand.type_name]
    elif operand.kind is MachineOperandKind.REGISTER_ADAPTER:
        kind = 1
        type_id = adapter_ids[operand.type_name]
    else:
        kind = 2
        type_id = immediate_ids[operand.type_name]
    if type_id > 0x3FFF:
        raise ValueError(f"{operand.type_name}: machine operand type exceeds 14 bits")
    return operand.name, kind, type_id


def emit_tables(table: MachineTable, encoding_table: EncodingTable) -> str:
    """Emits all native tables from one validated materialized machine model."""
    validate_machine_table(table, encoding_table)

    registers = tuple(sorted(table.physical_registers, key=lambda row: row.name))
    register_ids = {row.name: index for index, row in enumerate(registers)}
    register_classes = tuple(sorted(table.register_classes, key=lambda row: row.name))
    class_ids = {row.name: index for index, row in enumerate(register_classes, start=1)}
    adapters = tuple(sorted(table.register_adapters, key=lambda row: row.name))
    adapter_ids = {row.name: index for index, row in enumerate(adapters, start=1)}
    immediates = tuple(sorted(table.immediates, key=lambda row: row.name))
    immediate_ids = {row.name: index for index, row in enumerate(immediates, start=1)}
    forms = tuple(sorted(table.forms, key=lambda row: row.name))
    encoding_names = tuple(row.name for row in sorted(encoding_table.instructions, key=lambda row: row.name))
    if tuple(row.name for row in forms) != encoding_names:
        raise ValueError("AIE2P machine-form IDs do not align with encoding IDs")

    if len(table.atomic_unit_names) > 0x100:
        raise ValueError("AIE2P atomic unit IDs exceed uint8")
    if max(register.hardware_encoding for register in registers) > 0xFF:
        raise ValueError("AIE2P direct register encoding exceeds uint8")
    if any(immediate.encoded_width_bits >= 63 for immediate in immediates):
        raise ValueError("native AIE2P immediate arithmetic requires widths below 63")

    string_offsets, string_lines = _build_string_table(_machine_strings(table))

    atomic_unit_lines = [f"    UINT16_C({string_offsets[name]})," for name in table.atomic_unit_names]
    register_lines: list[str] = []
    register_atomic_units: list[int] = []
    subregister_lines: list[str] = []
    for register in registers:
        atomic_start = len(register_atomic_units)
        subregister_start = len(subregister_lines)
        register_atomic_units.extend(register.atomic_units)
        subregister_lines.extend(
            f"    {{.register_id = UINT16_C({register_ids[subregister]}), .index_name_offset = UINT16_C({string_offsets[index_name]})}},"
            for subregister, index_name in zip(
                register.subregisters,
                register.subregister_indices,
                strict=True,
            )
        )
        register_lines.append(
            "    {"
            f".name_offset = UINT16_C({string_offsets[register.name]}), "
            ".assembly_name_offset = "
            f"UINT16_C({string_offsets[register.assembly_name]}), "
            f".atomic_unit_start = UINT16_C({atomic_start}), "
            f".subregister_start = UINT16_C({subregister_start}), "
            f".hardware_encoding = UINT16_C({register.hardware_encoding}), "
            f".atomic_unit_count = {len(register.atomic_units)}, "
            f".subregister_count = {len(register.subregisters)}"
            "},"
        )
    if len(register_atomic_units) > 0xFFFF or len(subregister_lines) > 0xFFFF:
        raise ValueError("AIE2P register relation tables exceed uint16 offsets")

    layouts = tuple(sorted({row.layout for row in register_classes}, key=repr))
    layout_ids = {layout: index for index, layout in enumerate(layouts)}
    layout_lines = [
        "    {"
        f".register_size_bits = UINT16_C({layout.register_size_bits}), "
        f".alignment_bits = UINT16_C({layout.alignment_bits}), "
        f".spill_size_bits = UINT16_C({layout.spill_size_bits}), "
        f".spill_alignment_bits = UINT16_C({layout.spill_alignment_bits})"
        "},"
        for layout in layouts
    ]

    value_type_groups = tuple(sorted({row.value_types for row in register_classes}))
    value_type_group_ids = {value_types: index for index, value_types in enumerate(value_type_groups)}
    value_type_group_lines: list[str] = []
    value_type_name_lines: list[str] = []
    for value_types in value_type_groups:
        start = len(value_type_name_lines)
        value_type_group_lines.append(f"    {{.value_type_start = UINT16_C({start}), .value_type_count = {len(value_types)}}},")
        value_type_name_lines.extend(f"    UINT16_C({string_offsets[value_type]})," for value_type in value_types)

    class_lines = ["    {0},"]
    candidate_lines: list[str] = []
    for register_class in register_classes:
        candidate_start = len(candidate_lines)
        candidate_lines.extend(f"    UINT16_C({register_ids[candidate]})," for candidate in register_class.candidates)
        flags = int(register_class.is_allocatable) | (int(register_class.consider_in_pre_ra_scheduling) << 1) | (int(register_class.generate_pressure_set) << 2)
        class_lines.append(
            "    {"
            f".name_offset = UINT16_C({string_offsets[register_class.name]}), "
            f".candidate_start = UINT16_C({candidate_start}), "
            f".layout_id = {layout_ids[register_class.layout]}, "
            ".value_type_group_id = "
            f"{value_type_group_ids[register_class.value_types]}, "
            f".candidate_count = {len(register_class.candidates)}, "
            f".flags = {flags}"
            "},"
        )
    if len(candidate_lines) > 0xFFFF:
        raise ValueError("AIE2P register-class candidates exceed uint16 offsets")

    encoding_maps = tuple(sorted({tuple(sorted((register_ids[register], value) for register, value in adapter.effective_register_encodings)) for adapter in adapters}))
    if len(encoding_maps) > 0x100:
        raise ValueError("AIE2P register encoding maps exceed uint8 IDs")
    encoding_map_ids = {encoding_map: index for index, encoding_map in enumerate(encoding_maps)}
    encoding_map_lines: list[str] = []
    encoding_map_register_lines: list[str] = []
    encoding_map_value_lines: list[str] = []
    for encoding_map in encoding_maps:
        start = len(encoding_map_register_lines)
        encoding_map_lines.append(f"    {{.entry_start = UINT16_C({start}), .entry_count = {len(encoding_map)}}},")
        for register_id, value in encoding_map:
            encoding_map_register_lines.append(f"    UINT16_C({register_id}),")
            encoding_map_value_lines.append(f"    {value},")
    if len(encoding_map_register_lines) > 0xFFFF:
        raise ValueError("AIE2P register encoding entries exceed uint16 offsets")

    adapter_lines = ["    {0},"]
    for adapter in adapters:
        encoding_map = tuple(sorted((register_ids[register], value) for register, value in adapter.effective_register_encodings))
        adapter_lines.append(
            f"    {{.name_offset = UINT16_C({string_offsets[adapter.name]}), .register_class_id = UINT16_C({class_ids[adapter.register_class]}), .encoding_map_id = {encoding_map_ids[encoding_map]}}},"
        )

    immediate_lines = ["    {0},"]
    for immediate in immediates:
        flags = int(immediate.is_signed) | (int(immediate.is_negative) << 1) | (int(immediate.allows_symbol_reference) << 2)
        immediate_lines.append(
            "    {"
            f".step = UINT32_C({immediate.step}), "
            f".name_offset = UINT16_C({string_offsets[immediate.name]}), "
            f".semantic_width_bits = {immediate.semantic_width_bits}, "
            f".encoded_width_bits = {immediate.encoded_width_bits}, "
            f".flags = {flags}"
            "},"
        )

    operand_list_keys = tuple(
        sorted(
            {
                (
                    tuple(_operand_key(row, class_ids, adapter_ids, immediate_ids) for row in form.outputs),
                    tuple(_operand_key(row, class_ids, adapter_ids, immediate_ids) for row in form.inputs),
                )
                for form in forms
            }
        )
    )
    operand_list_ids = {operand_list: index for index, operand_list in enumerate(operand_list_keys)}
    operand_list_lines: list[str] = []
    operand_lines: list[str] = []
    for outputs, inputs in operand_list_keys:
        start = len(operand_lines)
        operand_list_lines.append(f"    {{.operand_start = UINT16_C({start}), .output_count = {len(outputs)}, .input_count = {len(inputs)}}},")
        for name, kind, type_id in (*outputs, *inputs):
            operand_lines.append(f"    {{.name_offset = UINT16_C({string_offsets[name]}), .type_and_kind = UINT16_C({type_id | (kind << 14)})}},")
    if len(operand_lines) > 0xFFFF:
        raise ValueError("AIE2P machine operands exceed uint16 offsets")

    register_list_keys = tuple(sorted({form.implicit_defs for form in forms} | {form.implicit_uses for form in forms}))
    if len(register_list_keys) > 0x100:
        raise ValueError("AIE2P implicit register lists exceed uint8 IDs")
    register_list_ids = {register_list: index for index, register_list in enumerate(register_list_keys)}
    register_list_lines: list[str] = []
    register_list_value_lines: list[str] = []
    for register_list in register_list_keys:
        start = len(register_list_value_lines)
        register_list_lines.append(f"    {{.register_start = UINT16_C({start}), .register_count = {len(register_list)}}},")
        register_list_value_lines.extend(f"    UINT16_C({register_ids[register]})," for register in register_list)

    def tie_key(form_index: int) -> tuple[tuple[int, int], ...]:
        form = forms[form_index]
        output_ordinals = {operand.name: index for index, operand in enumerate(form.outputs)}
        input_ordinals = {operand.name: index for index, operand in enumerate(form.inputs)}
        return tuple((output_ordinals[tie.definition], input_ordinals[tie.use]) for tie in form.ties)

    tie_keys_by_form = tuple(tie_key(index) for index in range(len(forms)))
    tie_list_keys = tuple(sorted(set(tie_keys_by_form)))
    if len(tie_list_keys) > 0x100:
        raise ValueError("AIE2P tie lists exceed uint8 IDs")
    tie_list_ids = {tie_list: index for index, tie_list in enumerate(tie_list_keys)}
    tie_list_lines: list[str] = []
    tie_lines: list[str] = []
    for tie_list in tie_list_keys:
        start = len(tie_lines)
        tie_list_lines.append(f"    {{.tie_start = UINT16_C({start}), .tie_count = {len(tie_list)}}},")
        tie_lines.extend(f"    {{.definition_ordinal = {definition}, .use_ordinal = {use}}}," for definition, use in tie_list)

    control_flow_ids = {
        None: 0,
        "branch_conditional_decrement": 1,
        "branch_conditional_nonzero": 2,
        "branch_conditional_zero": 3,
        "branch_direct": 4,
        "branch_indirect": 5,
        "call_direct": 6,
        "call_indirect": 7,
        "return": 8,
    }
    form_lines = ["    {0},"]
    for form_index, form in enumerate(forms):
        operand_key = (
            tuple(_operand_key(row, class_ids, adapter_ids, immediate_ids) for row in form.outputs),
            tuple(_operand_key(row, class_ids, adapter_ids, immediate_ids) for row in form.inputs),
        )
        form_lines.append(
            "    {"
            f".name_offset = UINT16_C({string_offsets[form.name]}), "
            f".assembly_offset = UINT16_C({string_offsets[form.assembly]}), "
            f".operand_list_id = UINT16_C({operand_list_ids[operand_key]}), "
            f".property_flags = UINT16_C({form.property_bits}), "
            ".implicit_def_list_id = "
            f"{register_list_ids[form.implicit_defs]}, "
            ".implicit_use_list_id = "
            f"{register_list_ids[form.implicit_uses]}, "
            f".tie_list_id = {tie_list_ids[tie_keys_by_form[form_index]]}, "
            f".control_flow_kind = {control_flow_ids[form.control_flow_kind]}"
            "},"
        )

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
        *string_lines,
        "",
        "static const uint16_t kLoomAie2pAtomicUnitNameOffsets[] = {",
        *atomic_unit_lines,
        "};",
        "",
        "static const loom_aie2p_physical_register_t",
        "    kLoomAie2pPhysicalRegisters[] = {",
        *register_lines,
        "};",
        "",
        "static const uint8_t kLoomAie2pPhysicalRegisterAtomicUnits[] = {",
        *(f"    {unit}," for unit in register_atomic_units),
        "};",
        "",
        "static const loom_aie2p_subregister_t kLoomAie2pSubregisters[] = {",
        *subregister_lines,
        "};",
        "",
        "static const loom_aie2p_register_layout_t",
        "    kLoomAie2pRegisterLayouts[] = {",
        *layout_lines,
        "};",
        "",
        "static const loom_aie2p_value_type_group_t",
        "    kLoomAie2pValueTypeGroups[] = {",
        *value_type_group_lines,
        "};",
        "",
        "static const uint16_t kLoomAie2pValueTypeNameOffsets[] = {",
        *value_type_name_lines,
        "};",
        "",
        "static const loom_aie2p_register_class_t",
        "    kLoomAie2pRegisterClasses[] = {",
        *class_lines,
        "};",
        "",
        "static const uint16_t kLoomAie2pRegisterClassCandidates[] = {",
        *candidate_lines,
        "};",
        "",
        "static const loom_aie2p_register_encoding_map_t",
        "    kLoomAie2pRegisterEncodingMaps[] = {",
        *encoding_map_lines,
        "};",
        "",
        "static const uint16_t kLoomAie2pRegisterEncodingMapRegisterIds[] = {",
        *encoding_map_register_lines,
        "};",
        "",
        "static const uint8_t kLoomAie2pRegisterEncodingMapValues[] = {",
        *encoding_map_value_lines,
        "};",
        "",
        "static const loom_aie2p_register_adapter_t",
        "    kLoomAie2pRegisterAdapters[] = {",
        *adapter_lines,
        "};",
        "",
        "static const loom_aie2p_immediate_t kLoomAie2pImmediates[] = {",
        *immediate_lines,
        "};",
        "",
        "static const loom_aie2p_operand_list_t kLoomAie2pOperandLists[] = {",
        *operand_list_lines,
        "};",
        "",
        "static const loom_aie2p_machine_operand_t kLoomAie2pMachineOperands[] = {",
        *operand_lines,
        "};",
        "",
        "static const loom_aie2p_register_list_t kLoomAie2pRegisterLists[] = {",
        *register_list_lines,
        "};",
        "",
        "static const uint16_t kLoomAie2pRegisterListValues[] = {",
        *register_list_value_lines,
        "};",
        "",
        "static const loom_aie2p_tie_list_t kLoomAie2pTieLists[] = {",
        *tie_list_lines,
        "};",
        "",
        "static const loom_aie2p_machine_tie_t kLoomAie2pMachineTies[] = {",
        *tie_lines,
        "};",
        "",
        "static const loom_aie2p_machine_form_t kLoomAie2pMachineForms[] = {",
        *form_lines,
        "};",
        "",
    ]
    return "\n".join(lines)
