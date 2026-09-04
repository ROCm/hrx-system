# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders compact runtime C projections from the VM specification."""

from __future__ import annotations

from collections.abc import Iterable

from iree.vm.bytecode.spec import isa, module
from iree.vm.bytecode.spec.isa.core import rules as core_rules
from iree.vm.bytecode.spec.module import rules as module_rules
from iree.vm.bytecode.spec.schema import NumericKind
from iree.vm.bytecode.spec.specification import Specification

_COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

_GENERATED = """\
// Generated from the authoritative VM bytecode specification. Do not edit.
// clang-format off
"""


def _identifier(value: str) -> str:
    return value.replace(".", "_").upper()


def _instruction_c_type(instruction: isa.Instruction) -> str:
    return f"iree_vm_bytecode_{instruction.mnemonic.replace('.', '_')}_t"


def _c_field(field) -> str:
    suffix = f"[{field.element_count}]" if field.element_count != 1 else ""
    return f"  {field.encoding.c_type} {field.name}{suffix};"


def _header_prologue(guard: str) -> list[str]:
    return [
        _COPYRIGHT.rstrip(),
        "",
        _GENERATED.rstrip(),
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
    ]


def _header_epilogue(guard: str) -> list[str]:
    return ["", f"#endif  // {guard}", ""]


def _render_numeric_table(lines: list[str], table) -> None:
    stem = table.name.replace(".", "_")
    type_stem = f"{stem}s" if table.kind == NumericKind.FLAGS else stem
    lines.extend(
        [
            f"// {table.summary}",
            f"typedef {table.encoding.c_type} iree_vm_bytecode_{type_stem}_t;",
            f"enum iree_vm_bytecode_{stem}{'_bits' if table.kind == NumericKind.FLAGS else ''}_e {{",
        ]
    )
    prefix = f"IREE_VM_BYTECODE_{_identifier(table.name)}"
    width = table.encoding.byte_length * 2
    for value in table.values:
        lines.extend(
            [
                f"  // {value.summary}",
                f"  {prefix}_{_identifier(value.name)} = 0x{value.value:0{width}X},",
            ]
        )
    lines.extend(["};", "", "enum {"])
    lines.extend(
        f"  {prefix}_{_identifier(value.name)}_SINCE_MINOR = {value.since.minor},"
        for value in table.values
    )
    lines.extend(["};", ""])


def _render_exact_bytes(lines: list[str], record: module.WireRecord) -> None:
    for wire_field in record.fields:
        rule = wire_field.rule
        if rule.kind != module_rules.FieldRule.EXACT_BYTES:
            continue
        encoded = "".join(f"\\x{value:02X}" for value in rule.data)
        prefix = _identifier(f"{record.name}_{wire_field.field.name}")
        lines.extend(
            [
                f"// Exact {record.name}.{wire_field.field.name} bytes.",
                f'#define IREE_VM_BYTECODE_{prefix}_BYTES "{encoded}"',
                f"#define IREE_VM_BYTECODE_{prefix}_LENGTH {len(rule.data)}u",
                "",
            ]
        )


def render_module_header(specification: Specification) -> str:
    """Renders natural C overlays for selected module records."""

    guard = "IREE_VM_BYTECODE_WIRE_MODULE_H_"
    lines = _header_prologue(guard)
    lines.extend(
        [
            "enum {",
            f"  IREE_VM_BYTECODE_CORE_MAJOR = {specification.version.major},",
            f"  IREE_VM_BYTECODE_CORE_MINOR = {specification.version.minor},",
            f"  IREE_VM_BYTECODE_IMAGE_ALIGNMENT = {specification.module_format.image_alignment},",
            f"  IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT = {specification.module_format.minimum_section_alignment},",
            "};",
            "",
            "// Architectural module section identifier.",
            "typedef uint16_t iree_vm_bytecode_section_type_t;",
            "enum iree_vm_bytecode_section_type_e {",
        ]
    )
    for section in specification.module_format.sections:
        lines.append(
            f"  IREE_VM_BYTECODE_SECTION_{_identifier(section.name)} = "
            f"0x{section.section_type:04X},"
        )
    lines.extend(["};", "", "enum {"])
    for section in specification.module_format.sections:
        prefix = f"IREE_VM_BYTECODE_SECTION_{_identifier(section.name)}"
        lines.extend(
            [
                f"  {prefix}_REQUIRED_FLAGS = 0x{section.required_flags:04X},",
                f"  {prefix}_SINCE_MINOR = {section.since.minor},",
            ]
        )
    lines.extend(["};", ""])
    for table in specification.module_format.numeric_tables:
        _render_numeric_table(lines, table)
    for record in specification.module_format.records:
        _render_exact_bytes(lines, record)
    lines.extend(
        [
            "// Dense ordinal identifying one module record declaration.",
            "typedef uint8_t iree_vm_bytecode_module_record_ordinal_t;",
            "",
        ]
    )
    for ordinal, record in enumerate(specification.module_format.records):
        lines.append(
            f"#define IREE_VM_BYTECODE_MODULE_RECORD_{_identifier(record.name)} "
            f"((iree_vm_bytecode_module_record_ordinal_t){ordinal}u)"
        )
    lines.extend(
        [
            "#define IREE_VM_BYTECODE_MODULE_RECORD_COUNT "
            f"((iree_vm_bytecode_module_record_ordinal_t){len(specification.module_format.records)}u)",
            "",
        ]
    )
    for record in specification.module_format.records:
        lines.append(f"// {record.summary}")
        lines.append(f"typedef struct {record.c_type} {{")
        for wire_field in record.fields:
            lines.append(f"  // {wire_field.field.summary}")
            lines.append(_c_field(wire_field.field))
        lines.extend([f"}} {record.c_type};", ""])
    lines.extend(_header_epilogue(guard))
    return "\n".join(lines)


def render_core_header(specification: Specification) -> str:
    """Renders Core opcode constants and natural instruction overlays."""

    guard = "IREE_VM_BYTECODE_WIRE_CORE_H_"
    lines = _header_prologue(guard)
    for table in specification.selectors:
        _render_numeric_table(lines, table)
    lines.extend(
        [
            "// One-byte physical Core instruction opcode.",
            "typedef uint8_t iree_vm_bytecode_opcode_t;",
            "",
        ]
    )
    for instruction in sorted(specification.instructions, key=lambda item: item.opcode):
        lines.append(
            f"#define IREE_VM_BYTECODE_OPCODE_{_identifier(instruction.mnemonic)} "
            f"((iree_vm_bytecode_opcode_t)0x{instruction.opcode:02X}u)"
        )
    lines.extend(
        [
            "#define IREE_VM_BYTECODE_OPCODE_CAPACITY 256u",
            "",
        ]
    )
    for instruction in sorted(specification.instructions, key=lambda item: item.opcode):
        c_type = _instruction_c_type(instruction)
        lines.extend(
            [
                f"// {instruction.summary}",
                f"typedef struct {c_type} {{",
                "  // Physical instruction opcode.",
                "  uint8_t opcode;",
            ]
        )
        for instruction_field in instruction.fields:
            lines.append(f"  // {instruction_field.field.summary}")
            lines.append(_c_field(instruction_field.field))
        lines.extend([f"}} {c_type};", ""])
    lines.extend(_header_epilogue(guard))
    return "\n".join(lines)


def _static_assert(expression: str, message: str) -> str:
    return f'static_assert({expression}, "{message}");'


def _render_layout_assertions(
    lines, c_type, byte_length, alignment, field_offsets
) -> None:
    lines.extend(
        [
            _static_assert(f"sizeof({c_type}) == {byte_length}u", "wire size"),
            _static_assert(alignment, "wire alignment"),
        ]
    )
    lines.extend(
        _static_assert(f"offsetof({c_type}, {name}) == {offset}u", "wire offset")
        for name, offset in field_offsets
    )
    lines.append("")


def render_wire_assertions(specification: Specification) -> str:
    """Renders the sole compilation site for all wire-layout assertions."""

    lines = [
        _COPYRIGHT.rstrip(),
        "",
        _GENERATED.rstrip(),
        "",
        "#include <stddef.h>",
        "",
        '#include "iree/base/alignment.h"',
        '#include "iree/vm/bytecode/wire/core.h"',
        '#include "iree/vm/bytecode/wire/module.h"',
        "",
    ]
    for record in specification.module_format.records:
        _render_layout_assertions(
            lines,
            record.c_type,
            record.byte_length,
            f"iree_alignof({record.c_type}) == {record.alignment}u",
            (
                (wire_field.field.name, offset)
                for wire_field, offset in zip(
                    record.fields, record.field_offsets, strict=True
                )
            ),
        )
    for instruction in sorted(specification.instructions, key=lambda item: item.opcode):
        c_type = _instruction_c_type(instruction)
        _render_layout_assertions(
            lines,
            c_type,
            instruction.byte_length,
            f"4u % iree_alignof({c_type}) == 0u",
            (("opcode", 0),)
            + tuple(
                (instruction_field.field.name, offset)
                for instruction_field, offset in zip(
                    instruction.fields, instruction.field_offsets, strict=True
                )
            ),
        )
    return "\n".join(lines)


def _field_layouts(fields, offsets) -> dict[str, tuple[int, int]]:
    return {
        item.field.name: (offset, item.field.byte_length)
        for item, offset in zip(fields, offsets, strict=True)
    }


def _instruction_descriptor(instruction: isa.Instruction) -> str:
    if instruction.byte_length > 0xFF:
        raise ValueError(f"{instruction.mnemonic}: verification descriptor overflow")
    return (
        "    IREE_VM_BYTECODE_PACK_INSTRUCTION_VERIFICATION"
        f"(IREE_VM_BYTECODE_CONTROL_FLOW_"
        f"{instruction.control_flow.name}, {instruction.byte_length}u),"
    )


def _instruction_verification_shapes(specification: Specification):
    """Returns stable deduplicated verification shapes for Core instructions."""

    shape_indices = {}
    shapes = []
    for instruction in sorted(specification.instructions, key=lambda item: item.opcode):
        checks = _instruction_checks(instruction)
        shape_index = shape_indices.get(checks)
        if shape_index is None:
            shape_index = len(shapes)
            shape_indices[checks] = shape_index
            shapes.append([checks, []])
        shapes[shape_index][1].append(instruction)
    return shapes


def _c_u32(value: int) -> str:
    return f"UINT32_C(0x{value:08X})"


def _c_load_unsigned(offset: int, byte_length: int) -> str:
    if byte_length == 1:
        return f"record[{offset}u]"
    if byte_length in (2, 4, 8):
        return f"iree_unaligned_load_le_u{byte_length * 8}(record + {offset}u)"
    raise ValueError(f"unsupported generated verifier load width {byte_length}")


def _allowed_values_predicate(expression: str, values: Iterable[int]) -> str:
    values = sorted(set(values))
    ranges = []
    for value in values:
        if ranges and value == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], value)
        else:
            ranges.append((value, value))
    predicates = []
    for lower, upper in ranges:
        if lower == upper:
            predicates.append(f"{expression} == {_c_u32(lower)}")
        elif lower == 0:
            predicates.append(f"{expression} <= {_c_u32(upper)}")
        else:
            predicates.append(
                f"({expression} >= {_c_u32(lower)} && {expression} <= {_c_u32(upper)})"
            )
    return "(" + " || ".join(predicates) + ")"


def _direct_instruction_rule_c(
    kind,
    field_offset: int,
    field_byte_length: int,
    related_layouts: tuple[tuple[int, int], ...],
    values: tuple[int, ...],
    data,
) -> tuple[str | None, str | None]:
    """Returns direct C for one rule use without an intermediate encoding."""

    def related_load(index: int) -> str:
        offset, byte_length = related_layouts[index]
        return _c_load_unsigned(offset, byte_length)

    if kind == core_rules.FieldRule.ZERO:
        return (
            "iree_vm_bytecode_bytes_are_zero"
            f"(record + {field_offset}u, {field_byte_length}u)",
            None,
        )
    load = _c_load_unsigned(field_offset, field_byte_length)
    if kind == core_rules.FieldRule.REGISTER_VALUE:
        return f"{load} < context->function->value_register_count_u16", None
    if kind == core_rules.FieldRule.REGISTER_REF:
        return f"{load} < context->function->ref_register_count_u16", None
    if kind == core_rules.FieldRule.REGISTER_FUNCTION:
        return f"{load} < context->function->function_register_count_u16", None
    if kind in core_rules.DIRECT_TARGET_RULES:
        signed_load = f"(int{field_byte_length * 8}_t){load}"
        return (
            "iree_vm_bytecode_verify_control_target"
            f"(context, record_offset, record_length, {signed_load})",
            None,
        )
    if kind == core_rules.RecordRuleKind.SWITCH_TARGETS:
        base = related_load(0)
        return (
            "iree_vm_bytecode_range_fits_u32"
            f"({base}, {load}, "
            "context->function->switch_target_entry_count_u32)",
            None,
        )
    if kind == core_rules.FieldRule.ALLOWED_RANGE:
        return (
            f"{load} >= {_c_u32(values[0])} && {load} <= {_c_u32(values[1])}",
            None,
        )
    if kind == core_rules.FieldRule.ALLOWED_VALUES:
        return _allowed_values_predicate(load, values), None
    if kind == core_rules.FieldRule.SELECTOR:
        return _allowed_values_predicate(
            load, (value.value for value in data.values)
        ), None
    if kind == core_rules.FieldRule.GLOBAL_ORDINAL:
        return (
            "iree_vm_bytecode_verify_global_ordinal"
            f"(context, {load}, {_c_u32(values[0])})",
            None,
        )
    if kind == core_rules.FieldRule.LOCAL_BYTES_RANGE_BASE:
        length = related_load(0)
        return (
            "iree_vm_bytecode_range_fits_u32"
            f"({load}, {length}, context->function->local_byte_length_u16)",
            None,
        )
    if kind == core_rules.FieldRule.ABI_SLOT:
        count = (
            "iree_unaligned_load_le_u16"
            f"((const uint8_t*)context->signature + {values[0]}u)"
        )
        return f"{load} < iree_vm_bytecode_overflow_count({count})", None
    if kind == core_rules.FieldRule.LOCAL_BYTES_FIXED_BASE:
        length, alignment = values
        return (
            f"({load} & {_c_u32(alignment - 1)}) == 0 && "
            "iree_vm_bytecode_range_fits_u32"
            f"({load}, {_c_u32(length)}, "
            "context->function->local_byte_length_u16)",
            None,
        )
    if kind == core_rules.FieldRule.PACKED_SELECTORS:
        predicates = [f"({load} & {_c_u32(values[0])}) == 0"]
        for component in data:
            component_value = (
                f"(({load} >> {component.bit_offset}u) & "
                f"{_c_u32((1 << component.bit_length) - 1)})"
            )
            allowed_values = component.allowed_values or tuple(
                value.value for value in component.table.values
            )
            predicates.append(
                _allowed_values_predicate(component_value, allowed_values)
            )
        return " && ".join(predicates), None
    if kind == core_rules.FieldRule.REF_SLOT:
        return f"{load} < context->function->local_ref_count_u32", None
    if kind == core_rules.FieldRule.LOCAL_BYTES_RANGE_MEMORY_FORMAT:
        format_value = related_load(0)
        return (
            "iree_vm_bytecode_verify_local_memory_format_range"
            f"(context, {load}, {format_value})",
            None,
        )
    if kind == core_rules.FieldRule.RODATA_ORDINAL:
        return f"{load} < context->layout->rodata.count", None
    if kind == core_rules.FieldRule.CONSTANT_POOL_ORDINAL:
        return f"{load} < context->layout->constants.count", None
    if kind == core_rules.FieldRule.FUNCTION_LOCAL_ORDINAL:
        return f"{load} < context->function->local_function_count_u32", None
    if kind == core_rules.FieldRule.LOCAL_BYTES_REPEATED_BASE:
        count = related_load(0)
        element_length, alignment = values
        return (
            f"({load} & {_c_u32(alignment - 1)}) == 0 && "
            "iree_vm_bytecode_range_fits_u32"
            f"({load}, {count} * {_c_u32(element_length)}, "
            "context->function->local_byte_length_u16)",
            None,
        )
    if kind == core_rules.FieldRule.IMPORT_ORDINAL_OPTIONAL:
        return f"iree_vm_bytecode_verify_optional_import(context, {load})", None
    if kind == core_rules.FieldRule.RODATA_OFFSET:
        ordinal = related_load(0)
        return (
            f"iree_vm_bytecode_verify_rodata_offset(context, {ordinal}, {load})",
            None,
        )
    if kind == core_rules.FieldRule.RODATA_STATIC_OFFSET:
        ordinal = related_load(0)
        length = related_load(1)
        return (
            "iree_vm_bytecode_verify_rodata_static_offset"
            f"(context, {ordinal}, {load}, {length})",
            None,
        )
    if kind == core_rules.RecordRuleKind.CALL:
        return (
            "iree_vm_bytecode_verify_direct_call"
            f"(context, {load}, "
            f"{related_load(0)}, {related_load(1)})",
            None,
        )
    if kind == core_rules.RecordRuleKind.CALL_INDIRECT:
        return (
            "iree_vm_bytecode_verify_indirect_call"
            f"(context, {related_load(0)}, {related_load(1)})",
            None,
        )
    if kind == core_rules.RecordRuleKind.FIELDS_DISTINCT:
        return f"{load} != {related_load(0)}", None
    if kind == core_rules.RecordRuleKind.FUNCTION_ADDRESS:
        return (
            "iree_vm_bytecode_verify_function_address"
            f"(context, {load}, "
            f"{related_load(0)}, {related_load(1)})",
            None,
        )
    if kind == core_rules.RecordRuleKind.INTEGER_BITSTREAM_SHAPE:
        return (
            "iree_vm_bytecode_verify_integer_bitstream_shape"
            f"({load}, {related_load(0)}, {related_load(1)}, "
            f"{related_load(2)}, {related_load(3)}, "
            f"{str(values[0] == 0).lower()}, {_c_u32(values[1])})",
            None,
        )
    if kind == core_rules.RecordRuleKind.PACKED_SELECTOR_PAIRS:
        first, second = data
        first_value = (
            f"(({load} >> {first.bit_offset}u) & {_c_u32((1 << first.bit_length) - 1)})"
        )
        second_value = (
            f"(({load} >> {second.bit_offset}u) & "
            f"{_c_u32((1 << second.bit_length) - 1)})"
        )
        pair = f"({first_value} | ({second_value} << {first.bit_length}u))"
        allowed_pairs = (
            first_item | second_item << first.bit_length
            for first_item, second_item in zip(values[::2], values[1::2], strict=True)
        )
        return _allowed_values_predicate(pair, allowed_pairs), None
    if kind == core_rules.RecordRuleKind.VALUE_REGISTER_RANGE:
        count = related_load(0)
        return (
            f"{count} != 0 && iree_vm_bytecode_range_fits_u32"
            f"({load}, {count}, context->function->value_register_count_u16)",
            None,
        )
    if kind == core_rules.RecordRuleKind.VALUE_REGISTER_FORMAT_RANGE:
        format_value = related_load(0)
        return (
            "iree_vm_bytecode_verify_value_register_format_range"
            f"(context, {load}, {format_value}, {_c_u32(values[0])})",
            None,
        )
    raise ValueError(f"unhandled instruction verification rule {kind.name}")


def _instruction_checks(
    instruction: isa.Instruction,
) -> tuple[tuple[str | None, str | None], ...]:
    layouts_by_name = _field_layouts(instruction.fields, instruction.field_offsets)
    checks = []
    skipped_kinds = (
        core_rules.FieldRule.ANY_BITS,
        core_rules.FieldRule.CONSTRAINT_MEMBER,
        core_rules.FieldRule.LOCAL_BYTES_RANGE_LENGTH,
        core_rules.FieldRule.LOCAL_BYTES_REPEATED_COUNT,
    )
    for instruction_field, field_offset in zip(
        instruction.fields, instruction.field_offsets, strict=True
    ):
        rule = instruction_field.rule
        if rule.kind in skipped_kinds:
            continue
        checks.append(
            _direct_instruction_rule_c(
                rule.kind,
                field_offset,
                instruction_field.field.byte_length,
                tuple(layouts_by_name[name] for name in rule.fields),
                rule.values,
                rule.data,
            )
        )
    for rule in instruction.rules:
        field_offset, field_byte_length = layouts_by_name[rule.fields[0]]
        checks.append(
            _direct_instruction_rule_c(
                rule.kind,
                field_offset,
                field_byte_length,
                tuple(layouts_by_name[name] for name in rule.fields[1:]),
                rule.values,
                rule.data,
            )
        )
    return tuple(checks)


def render_layout_data(specification: Specification) -> str:
    """Renders the dense runtime section-layout table."""

    section_descriptors = [0] * (
        max(section.section_type for section in specification.module_format.sections)
        + 1
    )
    for section in specification.module_format.sections:
        section_descriptors[section.section_type] = (
            section.since.minor << 16 | section.required_flags
        )
    lines = [
        _COPYRIGHT.rstrip(),
        "",
        _GENERATED.rstrip(),
        "",
        "// Packed descriptors indexed by dense Core section type. Words encode",
        "// since_minor:u16 and required_flags:u16.",
        "static const uint32_t iree_vm_bytecode_section_layouts[] = {",
    ]
    lines.extend(f"    UINT32_C(0x{value:08X})," for value in section_descriptors)
    lines.extend(["};", ""])
    return "\n".join(lines)


def render_verifier_data(specification: Specification) -> str:
    """Renders the dense runtime instruction-verification table."""

    instruction_descriptors = ["    UINT16_C(0),"] * 256
    for instruction in specification.instructions:
        instruction_descriptors[instruction.opcode] = _instruction_descriptor(
            instruction
        )
    lines = [
        _COPYRIGHT.rstrip(),
        "",
        _GENERATED.rstrip(),
        "",
        '#include "iree/vm/bytecode/verifier_data.h"',
        "",
        "#define IREE_VM_BYTECODE_PACK_INSTRUCTION_VERIFICATION(\\",
        "    control_flow, byte_length)                        \\",
        "  (((uint16_t)(control_flow) << 12) | (uint16_t)(byte_length))",
        "",
        "const uint16_t iree_vm_bytecode_instruction_verification[256] = {",
    ]
    lines.extend(instruction_descriptors)
    lines.extend(
        [
            "};",
            "",
            "#undef IREE_VM_BYTECODE_PACK_INSTRUCTION_VERIFICATION",
            "",
        ]
    )
    return "\n".join(lines)


def render_instruction_verifier_cases(specification: Specification) -> str:
    """Renders direct deduplicated Core instruction validation cases."""

    shapes = _instruction_verification_shapes(specification)
    lines = [_COPYRIGHT.rstrip(), "", _GENERATED.rstrip(), ""]
    for checks, instructions in shapes:
        for instruction in instructions:
            lines.append(
                f"case IREE_VM_BYTECODE_OPCODE_{_identifier(instruction.mnemonic)}:"
            )
        lines.append(f"  // {', '.join(item.mnemonic for item in instructions)}")
        effects = []
        for predicate, effect in checks:
            if predicate:
                lines.append(f"  if (!({predicate})) return false;")
            if effect:
                effects.append(effect)
        lines.extend(f"  {effect}" for effect in effects)
        lines.append("  return true;")
    return "\n".join(lines) + "\n"


_MODULE_RELATIONSHIP_RULES = (
    module_rules.FieldRule.PAGE_MAJOR,
    module_rules.FieldRule.PAGE_REQUIRED_MINOR,
    module_rules.FieldRule.SECTION_BYTE_LENGTH,
    module_rules.FieldRule.SECTION_FLAGS,
    module_rules.FieldRule.SECTION_TYPE,
    module_rules.FieldRule.STRING_OFFSET,
    module_rules.FieldRule.SWITCH_TARGET,
)


def _module_ordinal_predicate(expression: str, domain) -> str:
    count = {
        module_rules.OrdinalDomain.STRING: "layout->strings.count",
        module_rules.OrdinalDomain.STRING_NONEMPTY: "layout->strings.count",
        module_rules.OrdinalDomain.REF_TYPE: "layout->ref_types.entry_count",
        module_rules.OrdinalDomain.SIGNATURE: "layout->signatures.count",
        module_rules.OrdinalDomain.CALLABLE_TYPE: "layout->callable_types.count",
        module_rules.OrdinalDomain.FUNCTION: "layout->functions.count",
    }[domain]
    predicate = f"{expression} < {count}"
    if domain == module_rules.OrdinalDomain.STRING_NONEMPTY:
        predicate += (
            f" && iree_vm_bytecode_string_at(&layout->strings, {expression}).size != 0"
        )
    return predicate


def _module_field_check(
    record: module.WireRecord,
    wire_field,
    field_offset: int,
    layouts_by_name: dict[str, tuple[int, int]],
) -> tuple[str, str | None] | None:
    """Returns a direct predicate and optional specialized failure statement."""

    rule = wire_field.rule
    kind = rule.kind
    if kind == module_rules.FieldRule.ANY_BITS or kind in _MODULE_RELATIONSHIP_RULES:
        return None
    field_byte_length = wire_field.field.byte_length
    if kind == module_rules.FieldRule.ZERO:
        return (
            "iree_vm_bytecode_bytes_are_zero"
            f"(record + {field_offset}u, {field_byte_length}u)",
            None,
        )
    if kind == module_rules.FieldRule.EXACT_BYTES:
        prefix = _identifier(f"{record.name}_{wire_field.field.name}")
        return (
            f"memcmp(record + {field_offset}u, "
            f"IREE_VM_BYTECODE_{prefix}_BYTES, "
            f"IREE_VM_BYTECODE_{prefix}_LENGTH) == 0",
            None,
        )

    load = _c_load_unsigned(field_offset, field_byte_length)
    if kind == module_rules.FieldRule.CORE_MAJOR:
        return (
            f"{load} == IREE_VM_BYTECODE_CORE_MAJOR",
            "return iree_make_status("
            'IREE_STATUS_INCOMPATIBLE, "bytecode Core major %" PRIu64 '
            f'" is unsupported", (uint64_t){load});',
        )
    if kind == module_rules.FieldRule.CORE_REQUIRED_MINOR:
        return (
            f"{load} <= IREE_VM_BYTECODE_CORE_MINOR",
            "return iree_make_status("
            'IREE_STATUS_INCOMPATIBLE, "bytecode Core minor %" PRIu64 '
            '" is newer than runtime minor %d", '
            f"(uint64_t){load}, IREE_VM_BYTECODE_CORE_MINOR);",
        )
    if kind == module_rules.FieldRule.ALLOWED_RANGE:
        return (
            f"{load} >= {_c_u32(rule.values[0])} && {load} <= {_c_u32(rule.values[1])}",
            None,
        )
    if kind == module_rules.FieldRule.ALLOWED_BITS:
        return f"({load} & ~{_c_u32(rule.values[0])}) == 0", None
    if kind == module_rules.FieldRule.MULTIPLE:
        return f"{load} % {_c_u32(rule.values[0])} == 0", None
    if kind == module_rules.FieldRule.BYTE_ALIGNMENT:
        return (
            f"{load} >= {_c_u32(rule.values[0])} && "
            f"iree_host_size_is_power_of_two((iree_host_size_t){load})",
            None,
        )
    if kind == module_rules.FieldRule.NONCORE_PAGE:
        return f"{load} >= UINT32_C(0xF0) && {load} <= UINT32_C(0xFD)", None
    if kind == module_rules.FieldRule.ORDINAL:
        return _module_ordinal_predicate(load, rule.data), None
    if kind == module_rules.FieldRule.ORDINAL_OR_NULL:
        return (
            f"{load} == {_c_u32(rule.values[0])} || "
            f"({_module_ordinal_predicate(load, rule.data)})",
            None,
        )
    if kind == module_rules.FieldRule.SIGNATURE_DESCRIPTOR:
        kind_offset, kind_byte_length = layouts_by_name[rule.fields[0]]
        kind_load = _c_load_unsigned(kind_offset, kind_byte_length)
        return (
            "iree_vm_bytecode_verify_signature_descriptor"
            f"(layout, {kind_load}, {load})",
            None,
        )
    raise ValueError(
        f"unhandled module verification rule {record.name}."
        f"{wire_field.field.name}: {kind.name}"
    )


def _module_verification_shapes(specification: Specification):
    """Returns stable deduplicated validation shapes for module records."""

    shape_ordinals = {}
    shapes = []
    for record in specification.module_format.records:
        layouts_by_name = _field_layouts(record.fields, record.field_offsets)
        checks = tuple(
            check
            for wire_field, field_offset in zip(
                record.fields, record.field_offsets, strict=True
            )
            if (
                check := _module_field_check(
                    record,
                    wire_field,
                    field_offset,
                    layouts_by_name,
                )
            )
            is not None
        )
        shape_ordinal = shape_ordinals.get(checks)
        if shape_ordinal is None:
            shape_ordinal = len(shapes)
            shape_ordinals[checks] = shape_ordinal
            shapes.append([checks, []])
        shapes[shape_ordinal][1].append(record)
    return shapes


def render_module_verifier_cases(specification: Specification) -> str:
    """Renders direct deduplicated module-record validation cases."""

    shapes = _module_verification_shapes(specification)
    lines = [_COPYRIGHT.rstrip(), "", _GENERATED.rstrip(), ""]
    for checks, records in shapes:
        for record in records:
            lines.append(
                f"case IREE_VM_BYTECODE_MODULE_RECORD_{_identifier(record.name)}:"
            )
        lines.append(f"  // {', '.join(record.name for record in records)}")
        for predicate, failure in checks:
            if failure:
                lines.append(f"  if (!({predicate})) {failure}")
            else:
                lines.append(f"  if (!({predicate})) break;")
        lines.append("  return iree_ok_status();")
    return "\n".join(lines) + "\n"


def _selector_values(specification: Specification, name: str):
    table = next(table for table in specification.selectors if table.name == name)
    values = tuple(sorted(table.values, key=lambda value: value.value))
    if tuple(value.value for value in values) != tuple(range(len(values))):
        raise ValueError(f"{name}: interpreter data requires dense selectors")
    return values


def _render_row_macro(lines: list[str], name: str, rows: Iterable[str]) -> None:
    rows = tuple(rows)
    lines.extend(
        [
            f"#define {name}_COUNT {len(rows)}u",
            f"#define {name}_ROWS(ROW) \\",
        ]
    )
    for index, row in enumerate(rows):
        continuation = " \\" if index + 1 != len(rows) else ""
        lines.append(f"  ROW({row}){continuation}")
    lines.append("")


def _render_instruction_macro(
    lines: list[str], name: str, instructions: Iterable[isa.Instruction]
) -> None:
    """Renders one X-macro list of physical instruction declarations."""

    instructions = tuple(instructions)
    lines.extend(
        [f"#define {name}_COUNT {len(instructions)}u", f"#define {name}(OP) \\"]
    )
    for index, instruction in enumerate(instructions):
        continuation = " \\" if index + 1 != len(instructions) else ""
        label = instruction.mnemonic.replace(".", "_")
        lines.append(
            f"  OP({_identifier(instruction.mnemonic)}, {label}, "
            f"{_instruction_c_type(instruction)}){continuation}"
        )
    lines.append("")


def render_interpreter_data(specification: Specification) -> str:
    """Renders complete Core dispatch and compact semantic selector data."""

    instructions = sorted(specification.instructions, key=lambda item: item.opcode)
    lines = [_COPYRIGHT.rstrip(), "", _GENERATED.rstrip(), ""]
    _render_instruction_macro(
        lines, "IREE_VM_BYTECODE_INTERPRETER_OPCODE_LIST", instructions
    )

    family_instructions = {
        family.name: tuple(
            instruction for instruction in instructions if instruction.family == family
        )
        for family in specification.families
    }
    integer_instructions = family_instructions["integer"]
    _render_instruction_macro(
        lines,
        "IREE_VM_BYTECODE_INTERPRETER_INTEGER_TOTAL_LIST",
        (
            instruction
            for instruction in integer_instructions
            if not instruction.failures
        ),
    )
    _render_instruction_macro(
        lines,
        "IREE_VM_BYTECODE_INTERPRETER_INTEGER_FALLIBLE_LIST",
        (instruction for instruction in integer_instructions if instruction.failures),
    )
    _render_instruction_macro(
        lines,
        "IREE_VM_BYTECODE_INTERPRETER_FLOAT_LIST",
        family_instructions["float"],
    )
    conversion_instructions = family_instructions["conversion"]
    _render_instruction_macro(
        lines,
        "IREE_VM_BYTECODE_INTERPRETER_CONVERSION_TOTAL_LIST",
        (
            instruction
            for instruction in conversion_instructions
            if not instruction.failures
        ),
    )
    _render_instruction_macro(
        lines,
        "IREE_VM_BYTECODE_INTERPRETER_CONVERSION_FALLIBLE_LIST",
        (
            instruction
            for instruction in conversion_instructions
            if instruction.failures
        ),
    )

    integer_rows = []
    for value in _selector_values(specification, "integer.convert"):
        source, destination = value.name.split(".to.")
        source_kind = source[0]
        if source_kind not in ("s", "u", "i") or destination[0] != "i":
            raise ValueError(f"integer.convert: invalid selector {value.name}")
        integer_rows.append(
            f"{_identifier(value.name)}, {value.value}u, {int(source[1:])}u, "
            f"{int(destination[1:])}u, {int(source_kind == 's')}u"
        )
    _render_row_macro(lines, "IREE_VM_BYTECODE_INTEGER_CONVERSION", integer_rows)

    float_formats = {"f8e4m3", "f8e5m2", "f16", "bf16", "f32", "f64"}
    for selector_name, macro_name in (
        ("float.extend", "IREE_VM_BYTECODE_FLOAT_EXTEND"),
        ("float.truncate", "IREE_VM_BYTECODE_FLOAT_TRUNCATE"),
        ("float.width", "IREE_VM_BYTECODE_FLOAT_WIDTH"),
    ):
        rows = []
        for value in _selector_values(specification, selector_name):
            source, destination = value.name.split(".to.")
            if source not in float_formats or destination not in float_formats:
                raise ValueError(f"{selector_name}: invalid selector {value.name}")
            rows.append(
                f"{_identifier(value.name)}, {value.value}u, "
                f"{_identifier(source)}, {_identifier(destination)}"
            )
        _render_row_macro(lines, macro_name, rows)

    integer_to_float_rows = []
    for value in _selector_values(specification, "integer.to.float"):
        source, destination = value.name.split(".to.")
        if source[0] not in ("s", "u") or destination not in float_formats:
            raise ValueError(f"integer.to.float: invalid selector {value.name}")
        integer_to_float_rows.append(
            f"{_identifier(value.name)}, {value.value}u, {int(source[1:])}u, "
            f"{int(source[0] == 's')}u, {_identifier(destination)}"
        )
    _render_row_macro(lines, "IREE_VM_BYTECODE_INTEGER_TO_FLOAT", integer_to_float_rows)

    float_to_integer_rows = []
    for value in _selector_values(specification, "float.to.integer"):
        source, destination = value.name.split(".to.")
        if source not in float_formats or destination[0] not in ("s", "u"):
            raise ValueError(f"float.to.integer: invalid selector {value.name}")
        float_to_integer_rows.append(
            f"{_identifier(value.name)}, {value.value}u, {_identifier(source)}, "
            f"{int(destination[1:])}u, {int(destination[0] == 's')}u"
        )
    _render_row_macro(lines, "IREE_VM_BYTECODE_FLOAT_TO_INTEGER", float_to_integer_rows)
    return "\n".join(lines)


def _c_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def render_disassembler_data(specification: Specification) -> str:
    """Renders separately linked direct indices into packed BSTRING data."""

    strings = [""]
    offsets = {"": 0}
    byte_length = 1

    def intern(value: str) -> int:
        nonlocal byte_length
        if value in offsets:
            return offsets[value]
        encoded = value.encode("utf-8")
        if len(encoded) > 0xFF:
            raise ValueError(f"disassembler string exceeds BSTRING limit: {value!r}")
        if byte_length + 1 + len(encoded) > 0xFFFF:
            raise ValueError("disassembler BSTRING table exceeds u16 offsets")
        offset = byte_length
        offsets[value] = offset
        strings.append(value)
        byte_length += 1 + len(encoded)
        return offset

    instruction_offsets = [0] * 256
    for instruction in specification.instructions:
        instruction_offsets[instruction.opcode] = intern(instruction.mnemonic)
    record_offsets = [
        intern(record.name) for record in specification.module_format.records
    ]

    lines = [
        _GENERATED.rstrip(),
        "",
        "static const uint8_t iree_vm_bytecode_disassembler_strings[] =",
    ]
    for value in strings:
        lines.append(f'    "\\x{len(value.encode("utf-8")):02x}" "{_c_string(value)}"')
    lines[-1] += ";"
    lines.extend(
        ["", "static const uint16_t iree_vm_bytecode_instruction_name_offsets[256] = {"]
    )
    lines.extend(f"    {offset}u," for offset in instruction_offsets)
    lines.extend(
        [
            "};",
            "",
            "static const uint16_t iree_vm_bytecode_module_record_name_offsets[] = {",
        ]
    )
    lines.extend(f"    {offset}u," for offset in record_offsets)
    lines.extend(["};", ""])
    return "\n".join(lines)
