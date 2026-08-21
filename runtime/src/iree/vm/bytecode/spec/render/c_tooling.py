# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private compact C tables used to inspect VM bytecode images."""

from __future__ import annotations

import json

from model.isa import Instruction
from model.module import Section
from model.schema import ScalarEncoding
from model.specification import Projection


def _c_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _require_u8(value: int, context: str) -> int:
    if not 0 <= value <= 0xFF:
        raise ValueError(f"{context}: {value} does not fit in u8")
    return value


def _require_u16(value: int, context: str) -> int:
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{context}: {value} does not fit in u16")
    return value


class _BStringPool:
    """Interns short strings into one packed [u8 length][bytes...] table."""

    def __init__(self, values: set[str]):
        self._values = tuple(sorted(values, key=lambda value: value.encode("utf-8")))
        self._offsets: dict[str, int] = {}
        byte_length = 0
        for value in self._values:
            encoded_value = value.encode("utf-8")
            if len(encoded_value) > 0xFF:
                raise ValueError(f"tooling B-string exceeds 255 bytes: {value!r}")
            self._offsets[value] = byte_length
            byte_length += 1 + len(encoded_value)
        if byte_length > 0xFFFF:
            raise ValueError(
                f"tooling B-string table exceeds 65535 bytes: {byte_length}"
            )
        self.byte_length = byte_length

    def offset(self, value: str) -> int:
        return self._offsets[value]

    def render(self, symbol_name: str) -> list[str]:
        lines = [
            f"static const uint8_t {symbol_name}",
            f"    [{self.byte_length}] =",
        ]
        for value in self._values:
            encoded_value = value.encode("utf-8")
            lines.append(f'    "\\x{len(encoded_value):02x}" {_c_string(value)}')
        lines.extend(("    ;", ""))
        return lines


def _instructions_by_domain(
    isa_projection: Projection,
) -> dict[str, tuple[Instruction, ...]]:
    return {
        domain.domain: tuple(
            sorted(
                (
                    entity
                    for entity in isa_projection.entities
                    if isinstance(entity, Instruction)
                    and entity.since.domain == domain.domain
                ),
                key=lambda instruction: instruction.opcode,
            )
        )
        for domain in sorted(isa_projection.domains, key=lambda value: value.page_id)
    }


def _collect_isa_strings(isa_projection: Projection) -> set[str]:
    values: set[str] = set()
    for entity in isa_projection.entities:
        if isinstance(entity, Instruction):
            values.add(entity.mnemonic)
            values.update(field.name for field in entity.fields)
    return values


def _collect_module_strings(module_projection: Projection) -> set[str]:
    return {
        section.entity_id.rsplit(".", 1)[-1]
        for section in module_projection.entities
        if isinstance(section, Section)
    }


def _render_preamble(product: str) -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED FILE: DO NOT EDIT.",
        f"// Private build-tree projection: {product}.",
        "// clang-format off",
        "",
    ]


def _render_opcode_map(instructions: tuple[Instruction, ...]) -> list[str]:
    local_index_by_opcode = {
        instruction.opcode: instruction_index + 1
        for instruction_index, instruction in enumerate(instructions)
    }
    values = [local_index_by_opcode.get(opcode, 0) for opcode in range(256)]
    lines = ["  {"]
    lines.extend(
        "    " + ", ".join(f"0x{value:02X}" for value in values[i : i + 16]) + ","
        for i in range(0, 256, 16)
    )
    lines.append("  },")
    return lines


def render_tooling_isa_tables(isa_projection: Projection) -> str:
    """Renders compact build-only instruction inspection tables."""

    entities_by_id = isa_projection.entity_map()
    domains = tuple(sorted(isa_projection.domains, key=lambda value: value.page_id))
    instructions_by_domain = _instructions_by_domain(isa_projection)
    string_pool = _BStringPool(_collect_isa_strings(isa_projection))

    lines = _render_preamble("instruction decoding tables")
    lines.extend(string_pool.render("iree_vm_bytecode_tooling_isa_string_table"))

    field_base_by_instruction_id: dict[str, int] = {}
    field_count = 0
    lines.extend(
        (
            "static const iree_vm_bytecode_tooling_field_descriptor_t",
            "    iree_vm_bytecode_tooling_fields[] = {",
        )
    )
    for domain in domains:
        for instruction in instructions_by_domain[domain.domain]:
            field_base_by_instruction_id[instruction.entity_id] = field_count
            for field in instruction.fields:
                encoding = entities_by_id[field.encoding_id]
                if not isinstance(encoding, ScalarEncoding):
                    raise ValueError(
                        f"{instruction.entity_id}.{field.name}: non-scalar encoding"
                    )
                _require_u8(
                    field.offset, f"{instruction.entity_id}.{field.name} offset"
                )
                _require_u8(
                    field.array_length,
                    f"{instruction.entity_id}.{field.name} array length",
                )
                encoding_name = field.encoding_id.rsplit(".", 1)[-1].upper()
                lines.extend(
                    (
                        f"  {{{string_pool.offset(field.name)}, {field.offset},",
                        f"   {field.array_length}, IREE_VM_BYTECODE_TOOLING_FIELD_ENCODING_{encoding_name},",
                        f"   IREE_VM_BYTECODE_TOOLING_FIELD_ROLE_{field.role.name}}},",
                    )
                )
                field_count += 1
    lines.extend(("};", ""))

    instruction_base_by_domain: dict[str, int] = {}
    instruction_count = 0
    lines.extend(
        (
            "static const iree_vm_bytecode_tooling_instruction_descriptor_t",
            "    iree_vm_bytecode_tooling_instructions[] = {",
        )
    )
    for domain in domains:
        instructions = instructions_by_domain[domain.domain]
        _require_u8(len(instructions), f"{domain.domain} instruction count")
        instruction_base_by_domain[domain.domain] = instruction_count
        for instruction in instructions:
            _require_u8(instruction.byte_length, f"{instruction.entity_id} byte length")
            _require_u8(len(instruction.fields), f"{instruction.entity_id} field count")
            _require_u16(
                field_base_by_instruction_id[instruction.entity_id],
                f"{instruction.entity_id} field base",
            )
            lines.extend(
                (
                    f"  {{{string_pool.offset(instruction.mnemonic)},",
                    f"   {field_base_by_instruction_id[instruction.entity_id]},",
                    f"   {instruction.byte_length}, {len(instruction.fields)}}},",
                )
            )
            instruction_count += 1
    lines.extend(("};", ""))

    lines.extend(
        (
            "static const uint8_t",
            "    iree_vm_bytecode_tooling_opcode_maps[][256] = {",
        )
    )
    for domain in domains:
        lines.extend(_render_opcode_map(instructions_by_domain[domain.domain]))
    lines.extend(("};", ""))

    lines.extend(
        (
            "static const iree_vm_bytecode_tooling_page_descriptor_t",
            "    iree_vm_bytecode_tooling_pages[] = {",
        )
    )
    for domain in domains:
        _require_u16(
            instruction_base_by_domain[domain.domain],
            f"{domain.domain} instruction base",
        )
        lines.extend(
            (
                f"  {{{instruction_base_by_domain[domain.domain]},",
                f"   0x{domain.page_id:02X},",
                f"   {len(instructions_by_domain[domain.domain])}}},",
            )
        )
    lines.extend(("};", ""))

    lines.extend(("// clang-format on", ""))
    return "\n".join(lines)


def render_tooling_module_tables(module_projection: Projection) -> str:
    """Renders compact build-only module inspection tables."""

    sections = tuple(
        sorted(
            (
                entity
                for entity in module_projection.entities
                if isinstance(entity, Section)
            ),
            key=lambda section: section.section_type,
        )
    )
    string_pool = _BStringPool(_collect_module_strings(module_projection))

    lines = _render_preamble("module container tables")
    lines.extend(string_pool.render("iree_vm_bytecode_tooling_module_string_table"))
    lines.extend(
        (
            "static const iree_vm_bytecode_tooling_section_descriptor_t",
            "    iree_vm_bytecode_tooling_sections[] = {",
        )
    )
    for section in sections:
        section_name = section.entity_id.rsplit(".", 1)[-1]
        lines.append(
            f"  {{{string_pool.offset(section_name)}, 0x{section.section_type:04X}}},"
        )
    lines.extend(("};", "", "// clang-format on", ""))
    return "\n".join(lines)
