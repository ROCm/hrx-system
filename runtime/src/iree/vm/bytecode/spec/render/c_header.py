# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Natural-layout C declarations projected from module and ISA facts."""

from __future__ import annotations

import re
import textwrap

from model.isa import Instruction, InstructionFamily
from model.module import ModuleFormat, Section
from model.schema import (
    NumericTable,
    NumericTableKind,
    ScalarEncoding,
    WireRecord,
    WireRecordLayout,
    selected_numeric_values,
    selected_record_layouts,
)
from model.specification import (
    ARCHITECTURAL_EXTENSION_PAGE_MAX,
    ARCHITECTURAL_EXTENSION_PAGE_MIN,
    CORE_PAGE_ID,
    RESERVED_EXPERIMENT_PAGE_ID,
    RESERVED_EXTENDED_ESCAPE_PAGE_ID,
    Projection,
)


def _c_identifier(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", value)


def _c_constant(value: str) -> str:
    return _c_identifier(value).upper()


def _guard(filename: str) -> str:
    return f"IREE_VM_BYTECODE_GENERATED_{_c_constant(filename)}_"


def _comment(text: str, *, indent: str = "") -> list[str]:
    prefix = f"{indent}// "
    return [
        f"{prefix}{line}"
        for line in textwrap.wrap(
            text,
            width=80 - len(prefix),
            break_long_words=False,
            break_on_hyphens=False,
        )
    ]


def _header_prologue(filename: str, description: str) -> list[str]:
    guard = _guard(filename)
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED FILE: DO NOT EDIT.",
        "// Regenerate with:",
        "//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate",
        f"// {description}",
        "// Multi-byte fields are little-endian and naturally aligned.",
        "// clang-format off",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
    ]


def _header_epilogue(filename: str) -> list[str]:
    guard = _guard(filename)
    return [
        f"#endif  // {guard}",
        "// clang-format on",
        "",
    ]


def _assertions_prologue(description: str, header_paths: tuple[str, ...]) -> list[str]:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED FILE: DO NOT EDIT.",
        "// Regenerate with:",
        "//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate",
        f"// {description}",
        "// clang-format off",
        "",
        '#include "iree/base/alignment.h"',
    ]
    lines.extend(
        f'#include "iree/vm/bytecode/{header_path}"' for header_path in header_paths
    )
    lines.append("")
    return lines


def _entity_stem(entity_id: str, marker: str) -> str:
    _, separator, tail = entity_id.partition(marker)
    if not separator:
        raise ValueError(f"{entity_id}: missing expected marker {marker!r}")
    return _c_identifier(tail)


def _append_enum_value(lines: list[str], name: str, value: str) -> None:
    declaration = f"  {name} = {value},"
    if len(declaration) <= 80:
        lines.append(declaration)
    else:
        lines.extend((f"  {name} =", f"      {value},"))


def _render_static_assert(expression: str, message: str) -> list[str]:
    declaration = f'static_assert({expression}, "{message}");'
    if len(declaration) <= 100:
        return [declaration]
    expression_line = f"static_assert({expression},"
    if len(expression_line) <= 100:
        return [expression_line, f'              "{message}");']
    return [
        "static_assert(",
        f"    {expression},",
        f'    "{message}");',
    ]


def _render_host_alignment_assert(
    type_name: str,
    wire_alignment: int,
) -> list[str]:
    return _render_static_assert(
        f"{wire_alignment} % iree_alignof({type_name}) == 0",
        "wire alignment",
    )


def _render_offset_assert(
    type_name: str,
    field_name: str,
    field_offset: int,
) -> list[str]:
    expression = f"offsetof({type_name}, {field_name}) == {field_offset}"
    if len(f"    {expression},") <= 100:
        return _render_static_assert(expression, "wire offset")
    return [
        "static_assert(",
        "    offsetof(",
        f"        {type_name},",
        f"        {field_name}) == {field_offset},",
        '    "wire offset");',
    ]


def _render_numeric_table(
    table: NumericTable,
    values,
    encodings: dict[str, ScalarEncoding],
    *,
    prefix: str,
) -> list[str]:
    encoding = encodings[table.encoding_id]
    marker = ".selector." if ".selector." in table.entity_id else ".numeric."
    stem = _entity_stem(table.entity_id, marker)
    type_stem = f"{stem}s" if table.table_kind == NumericTableKind.FLAGS else stem
    type_name = f"{prefix}_{type_stem}_t"
    lines = [*_comment(table.summary), f"typedef {encoding.c_type} {type_name};"]
    if table.table_kind == NumericTableKind.FLAGS:
        lines.append(f"enum {prefix}_{stem}_bits_e {{")
    else:
        lines.append("enum {")
    constant_prefix = _c_constant(f"{prefix}_{stem}")
    width = encoding.byte_length * 2
    for value in values:
        lines.extend(_comment(value.summary, indent="  "))
        lines.append(
            f"  {constant_prefix}_{_c_constant(value.name)} = "
            f"0x{value.value:0{width}X},"
        )
    lines.extend(("};", ""))
    lines.append("enum {")
    for value in values:
        _append_enum_value(
            lines,
            f"{constant_prefix}_{_c_constant(value.name)}_SINCE_MINOR",
            str(value.since.minor),
        )
    lines.extend(("};", ""))
    return lines


def _render_wire_record_declaration(
    record: WireRecord,
    layout: WireRecordLayout,
    encodings: dict[str, ScalarEncoding],
) -> list[str]:
    lines = _comment(record.summary)
    if layout.scalar_alias:
        field = layout.fields[0]
        lines.extend(_comment(field.description))
        lines.append(f"typedef {encodings[field.encoding_id].c_type} {record.c_type};")
    else:
        lines.append("typedef struct {")
        for field in layout.fields:
            encoding = encodings[field.encoding_id]
            suffix = f"[{field.array_length}]" if field.array_length != 1 else ""
            lines.extend(_comment(field.description, indent="  "))
            lines.append(f"  {encoding.c_type} {field.name}{suffix};")
        lines.append(f"}} {record.c_type};")
    lines.append("")
    return lines


def _render_wire_record_assertions(
    record: WireRecord,
    layout: WireRecordLayout,
) -> list[str]:
    lines = _render_static_assert(
        f"sizeof({record.c_type}) == {layout.byte_length}",
        "wire size",
    )
    lines.extend(
        _render_host_alignment_assert(
            record.c_type,
            layout.alignment,
        )
    )
    if not layout.scalar_alias:
        for field in layout.fields:
            lines.extend(
                _render_offset_assert(
                    record.c_type,
                    field.name,
                    field.offset,
                )
            )
    lines.append("")
    return lines


def _render_exact_byte_constants(
    records: list[WireRecord],
    layouts: dict[str, WireRecordLayout],
) -> list[str]:
    lines: list[str] = []
    for record in records:
        layout = layouts[record.entity_id]
        record_stem = _entity_stem(record.entity_id, ".record.")
        for field in layout.fields:
            expected_values = tuple(
                use.arguments[0]
                for use in field.validation
                if use.rule_id == "core.validation.field.exact_bytes"
                and len(use.arguments) == 1
                and isinstance(use.arguments[0], bytes)
            )
            if not expected_values:
                continue
            if len(expected_values) != 1:
                raise ValueError(
                    f"{record.entity_id}.{field.name}: multiple exact byte values"
                )
            expected = expected_values[0]
            constant = _c_constant(f"IREE_VM_BYTECODE_{record_stem}_{field.name}")
            encoded = "".join(f"\\x{value:02X}" for value in expected)
            lines.extend(
                (
                    *_comment(
                        f"Exact {record_stem}.{field.name} bytes; compare only "
                        "the declared length because the C literal has its normal "
                        "terminating zero."
                    ),
                    f'#define {constant}_BYTES "{encoded}"',
                    f"#define {constant}_LENGTH {len(expected)}",
                    "",
                )
            )
    return lines


def render_module_header(projection: Projection, filename: str) -> str:
    """Renders module-container constants and overlay-safe C record types."""

    lines = _header_prologue(
        filename,
        "Module-container constants and natural-layout wire declarations.",
    )
    formats = [
        entity for entity in projection.entities if isinstance(entity, ModuleFormat)
    ]
    if len(formats) != 1:
        raise ValueError("module projection must contain exactly one format")
    module_format = formats[0]
    if len(projection.versions) != 1:
        raise ValueError("module projection must select exactly one version domain")
    core_version = projection.versions[0]
    lines.extend(
        (
            "enum {",
            f"  IREE_VM_BYTECODE_CORE_MAJOR = {core_version.major},",
            f"  IREE_VM_BYTECODE_CORE_MINOR = {core_version.minor},",
            f"  IREE_VM_BYTECODE_IMAGE_ALIGNMENT = {module_format.image_alignment},",
            f"  IREE_VM_BYTECODE_SECTION_ALIGNMENT = {module_format.section_alignment},",
            "};",
            "",
            "typedef uint16_t iree_vm_bytecode_section_type_t;",
            "enum {",
        )
    )
    sections = sorted(
        (entity for entity in projection.entities if isinstance(entity, Section)),
        key=lambda value: value.section_type,
    )
    for section in sections:
        stem = _entity_stem(section.entity_id, ".section.")
        lines.append(
            f"  IREE_VM_BYTECODE_SECTION_{_c_constant(stem)} = "
            f"0x{section.section_type:04X},"
        )
    lines.extend(("};", ""))
    lines.append("enum {")
    for section in sections:
        stem = _c_constant(_entity_stem(section.entity_id, ".section."))
        _append_enum_value(
            lines,
            f"IREE_VM_BYTECODE_SECTION_{stem}_REQUIRED_FLAGS",
            f"0x{section.required_flags:04X}",
        )
        _append_enum_value(
            lines,
            f"IREE_VM_BYTECODE_SECTION_{stem}_SINCE_MINOR",
            str(section.since.minor),
        )
    lines.extend(("};", ""))

    encodings = {
        entity.entity_id: entity
        for entity in projection.entities
        if isinstance(entity, ScalarEncoding)
    }
    numeric_values = selected_numeric_values(projection)
    for table in (
        entity for entity in projection.entities if isinstance(entity, NumericTable)
    ):
        lines.extend(
            _render_numeric_table(
                table,
                numeric_values[table.entity_id],
                encodings,
                prefix="iree_vm_bytecode",
            )
        )

    selected_layouts = selected_record_layouts(projection)
    records = [
        entity for entity in projection.entities if isinstance(entity, WireRecord)
    ]
    lines.extend(_render_exact_byte_constants(records, selected_layouts))
    for record in records:
        lines.extend(
            _render_wire_record_declaration(
                record,
                selected_layouts[record.entity_id],
                encodings,
            )
        )
    lines.extend(_header_epilogue(filename))
    return "\n".join(lines)


def render_module_assertions(
    projection: Projection,
    header_path: str,
) -> str:
    """Renders one C translation unit enforcing module record layouts."""

    lines = _assertions_prologue(
        "Compile-time checks for module-container wire declarations.",
        (header_path,),
    )
    selected_layouts = selected_record_layouts(projection)
    records = [
        entity for entity in projection.entities if isinstance(entity, WireRecord)
    ]
    for record in records:
        layout = selected_layouts[record.entity_id]
        lines.extend(_render_wire_record_assertions(record, layout))
    lines.extend(("// clang-format on", ""))
    return "\n".join(lines)


def _instruction_alignment(
    instruction: Instruction,
    encodings: dict[str, ScalarEncoding],
) -> int:
    return max(
        (encodings[field.encoding_id].alignment for field in instruction.fields),
        default=1,
    )


def _render_instruction_record_declaration(
    instruction: Instruction,
    page_id: int,
    encodings: dict[str, ScalarEncoding],
) -> list[str]:
    record_stem = _c_identifier(instruction.mnemonic)
    type_name = f"iree_vm_isa_{record_stem}_record_t"
    lines = [
        *_comment(
            f"Page 0x{page_id:02X}, opcode 0x{instruction.opcode:02X}: "
            f"{instruction.summary}"
        ),
        "typedef struct {",
    ]
    if page_id != 0:
        lines.append("  // Architectural instruction page selector.")
        lines.append("  uint8_t page_u8;")
    lines.append("  // Page-local instruction opcode.")
    lines.append("  uint8_t opcode_u8;")
    for field in sorted(instruction.fields, key=lambda value: value.offset):
        encoding = encodings[field.encoding_id]
        suffix = f"[{field.array_length}]" if field.array_length != 1 else ""
        lines.extend(_comment(field.description, indent="  "))
        lines.append(f"  {encoding.c_type} {field.name}{suffix};")
    lines.append(f"}} {type_name};")
    lines.append("")
    return lines


def _render_instruction_record_assertions(
    instruction: Instruction,
    page_id: int,
    encodings: dict[str, ScalarEncoding],
) -> list[str]:
    record_stem = _c_identifier(instruction.mnemonic)
    type_name = f"iree_vm_isa_{record_stem}_record_t"
    lines = _render_static_assert(
        f"sizeof({type_name}) == {instruction.byte_length}",
        "wire size",
    )
    lines.extend(
        _render_host_alignment_assert(
            type_name,
            _instruction_alignment(instruction, encodings),
        )
    )
    if page_id != 0:
        lines.extend(
            _render_offset_assert(
                type_name,
                "page_u8",
                0,
            )
        )
    opcode_offset = 1 if page_id != 0 else 0
    lines.extend(
        _render_offset_assert(
            type_name,
            "opcode_u8",
            opcode_offset,
        )
    )
    for field in instruction.fields:
        lines.extend(
            _render_offset_assert(
                type_name,
                field.name,
                field.offset,
            )
        )
    lines.append("")
    return lines


def _domain_version(projection: Projection, domain_name: str):
    versions = {version.domain: version for version in projection.versions}
    version = versions.get(domain_name)
    if version is None:
        raise ValueError(f"ISA projection does not select {domain_name!r}")
    domains = {domain.domain: domain for domain in projection.domains}
    return version, domains[domain_name]


def render_isa_opcodes_header(
    projection: Projection,
    domain_name: str,
    filename: str,
) -> str:
    """Renders one instruction page's identity, version, and opcodes."""

    lines = _header_prologue(
        filename,
        f"{domain_name.upper()} ISA page identity and opcode constants.",
    )
    version, domain = _domain_version(projection, domain_name)
    lines.extend(
        (
            "typedef uint8_t iree_vm_isa_page_t;",
            "enum {",
            f"  IREE_VM_ISA_PAGE_{domain_name.upper()} = 0x{domain.page_id:02X},",
            "};",
            "",
            "enum {",
            f"  IREE_VM_ISA_{domain_name.upper()}_MAJOR = {version.major},",
            f"  IREE_VM_ISA_{domain_name.upper()}_MINOR = {version.minor},",
            "};",
            "",
        )
    )
    if domain.page_id == CORE_PAGE_ID:
        lines.extend(
            (
                "enum {",
                "  IREE_VM_ISA_ARCHITECTURAL_EXTENSION_PAGE_MIN = "
                f"0x{ARCHITECTURAL_EXTENSION_PAGE_MIN:02X},",
                "  IREE_VM_ISA_ARCHITECTURAL_EXTENSION_PAGE_MAX = "
                f"0x{ARCHITECTURAL_EXTENSION_PAGE_MAX:02X},",
                "  IREE_VM_ISA_RESERVED_EXPERIMENT_PAGE = "
                f"0x{RESERVED_EXPERIMENT_PAGE_ID:02X},",
                "  IREE_VM_ISA_RESERVED_EXTENDED_ESCAPE_PAGE = "
                f"0x{RESERVED_EXTENDED_ESCAPE_PAGE_ID:02X},",
                "};",
                "",
            )
        )

    instructions = sorted(
        (
            entity
            for entity in projection.entities
            if isinstance(entity, Instruction) and entity.since.domain == domain_name
        ),
        key=lambda value: value.opcode,
    )
    lines.extend(
        (
            f"typedef uint8_t iree_vm_isa_{domain_name}_opcode_t;",
            "enum {",
        )
    )
    for instruction in instructions:
        mnemonic = instruction.mnemonic.removeprefix(f"{domain_name}.")
        lines.append(
            f"  IREE_VM_ISA_{domain_name.upper()}_OPCODE_"
            f"{_c_constant(mnemonic)} = 0x{instruction.opcode:02X},"
        )
    lines.extend(("};", "", "enum {"))
    for instruction in instructions:
        mnemonic = instruction.mnemonic.removeprefix(f"{domain_name}.")
        _append_enum_value(
            lines,
            f"IREE_VM_ISA_{domain_name.upper()}_OPCODE_"
            f"{_c_constant(mnemonic)}_SINCE_MINOR",
            str(instruction.since.minor),
        )
    lines.extend(("};", ""))
    lines.extend(_header_epilogue(filename))
    return "\n".join(lines)


def _selector_consumers(projection: Projection) -> dict[str, set[str]]:
    selector_ids = {
        entity.entity_id
        for entity in projection.entities
        if isinstance(entity, NumericTable)
        and entity.table_kind == NumericTableKind.SELECTOR
    }
    consumers = {selector_id: set() for selector_id in selector_ids}
    for entity in projection.entities:
        if not isinstance(entity, Instruction):
            continue
        for selector_id in selector_ids.intersection(entity.referenced_entity_ids()):
            consumers[selector_id].add(entity.family_id)
    return consumers


def shared_selector_table_ids(
    projection: Projection,
    domain_name: str,
) -> tuple[str, ...]:
    """Returns selector tables shared by multiple families in one domain."""

    consumers = _selector_consumers(projection)
    return tuple(
        sorted(
            table.entity_id
            for table in projection.entities
            if isinstance(table, NumericTable)
            and table.table_kind == NumericTableKind.SELECTOR
            and table.since.domain == domain_name
            and len(consumers[table.entity_id]) > 1
        )
    )


def _render_selector_tables(
    lines: list[str],
    projection: Projection,
    table_ids: set[str],
) -> None:
    encodings = {
        entity.entity_id: entity
        for entity in projection.entities
        if isinstance(entity, ScalarEncoding)
    }
    numeric_values = selected_numeric_values(projection)
    for table in (
        entity for entity in projection.entities if isinstance(entity, NumericTable)
    ):
        if table.entity_id not in table_ids:
            continue
        lines.extend(
            _render_numeric_table(
                table,
                numeric_values[table.entity_id],
                encodings,
                prefix="iree_vm_isa",
            )
        )


def render_isa_shared_selectors_header(
    projection: Projection,
    domain_name: str,
    filename: str,
) -> str:
    """Renders selector constants used by multiple semantic families."""

    _domain_version(projection, domain_name)
    table_ids = set(shared_selector_table_ids(projection, domain_name))
    if not table_ids:
        raise ValueError(f"{domain_name}: no shared selector tables")
    lines = _header_prologue(
        filename,
        f"{domain_name.upper()} ISA selectors shared by multiple families.",
    )
    _render_selector_tables(lines, projection, table_ids)
    lines.extend(_header_epilogue(filename))
    return "\n".join(lines)


def render_isa_family_header(
    projection: Projection,
    family_id: str,
    filename: str,
) -> str:
    """Renders natural-layout instruction records for one semantic family."""

    family = projection.require_entity(family_id)
    if not isinstance(family, InstructionFamily):
        raise ValueError(f"{family_id}: not an instruction family")
    _, domain = _domain_version(projection, family.since.domain)
    lines = _header_prologue(
        filename,
        f"Natural-layout instruction records for {family_id}.",
    )
    encodings = {
        entity.entity_id: entity
        for entity in projection.entities
        if isinstance(entity, ScalarEncoding)
    }
    instructions = sorted(
        (
            entity
            for entity in projection.entities
            if isinstance(entity, Instruction) and entity.family_id == family_id
        ),
        key=lambda value: value.opcode,
    )
    if not instructions:
        raise ValueError(f"{family_id}: family has no instructions")
    consumers = _selector_consumers(projection)
    owned_selector_ids = {
        table_id
        for table_id, family_ids in consumers.items()
        if family_ids == {family_id}
    }
    _render_selector_tables(lines, projection, owned_selector_ids)
    for instruction in instructions:
        lines.extend(
            _render_instruction_record_declaration(
                instruction,
                domain.page_id,
                encodings,
            )
        )
    lines.extend(_header_epilogue(filename))
    return "\n".join(lines)


def render_isa_assertions(
    projection: Projection,
    domain_name: str,
    family_header_paths: tuple[str, ...],
) -> str:
    """Renders one C translation unit enforcing an ISA domain's record layouts."""

    _, domain = _domain_version(projection, domain_name)
    lines = _assertions_prologue(
        f"Compile-time checks for {domain_name.upper()} ISA wire declarations.",
        family_header_paths,
    )
    encodings = {
        entity.entity_id: entity
        for entity in projection.entities
        if isinstance(entity, ScalarEncoding)
    }
    families = {
        entity.entity_id: entity
        for entity in projection.entities
        if isinstance(entity, InstructionFamily) and entity.since.domain == domain_name
    }
    instructions = sorted(
        (
            entity
            for entity in projection.entities
            if isinstance(entity, Instruction) and entity.since.domain == domain_name
        ),
        key=lambda value: (families[value.family_id].document_order, value.opcode),
    )
    for instruction in instructions:
        lines.extend(
            _render_instruction_record_assertions(
                instruction,
                domain.page_id,
                encodings,
            )
        )
    lines.extend(("// clang-format on", ""))
    return "\n".join(lines)
