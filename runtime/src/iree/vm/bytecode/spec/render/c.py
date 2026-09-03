# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders compact runtime C projections from the VM specification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

from iree.vm.bytecode.spec import isa, module
from iree.vm.bytecode.spec.isa.core import rules as core_rules
from iree.vm.bytecode.spec.module import rules as module_rules
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


def render_module_header(specification: Specification) -> str:
    """Renders natural C overlays for selected module records."""

    guard = "IREE_VM_BYTECODE_WIRE_MODULE_H_"
    lines = _header_prologue(guard)
    lines.extend(
        [
            "// Dense ordinal identifying one module record declaration.",
            "typedef uint8_t iree_vm_bytecode_module_record_ordinal_t;",
            "",
        ]
    )
    for ordinal, record in enumerate(specification.records):
        lines.append(
            f"#define IREE_VM_BYTECODE_MODULE_RECORD_{_identifier(record.name)} "
            f"((iree_vm_bytecode_module_record_ordinal_t){ordinal}u)"
        )
    lines.extend(
        [
            "#define IREE_VM_BYTECODE_MODULE_RECORD_COUNT "
            f"((iree_vm_bytecode_module_record_ordinal_t){len(specification.records)}u)",
            "",
        ]
    )
    for record in specification.records:
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
    for record in specification.records:
        lines.append(
            _static_assert(
                f"sizeof({record.c_type}) == {record.byte_length}u", "wire size"
            )
        )
        lines.append(
            _static_assert(
                f"iree_alignof({record.c_type}) == {record.alignment}u",
                "wire alignment",
            )
        )
        for wire_field, offset in zip(record.fields, record.field_offsets, strict=True):
            lines.append(
                _static_assert(
                    f"offsetof({record.c_type}, {wire_field.field.name}) == {offset}u",
                    "wire offset",
                )
            )
        lines.append("")
    for instruction in sorted(specification.instructions, key=lambda item: item.opcode):
        c_type = _instruction_c_type(instruction)
        lines.append(
            _static_assert(
                f"sizeof({c_type}) == {instruction.byte_length}u", "instruction size"
            )
        )
        lines.append(
            _static_assert(
                f"4u % iree_alignof({c_type}) == 0u", "instruction alignment"
            )
        )
        lines.append(
            _static_assert(f"offsetof({c_type}, opcode) == 0u", "opcode offset")
        )
        for instruction_field, offset in zip(
            instruction.fields, instruction.field_offsets, strict=True
        ):
            lines.append(
                _static_assert(
                    f"offsetof({c_type}, {instruction_field.field.name}) == {offset}u",
                    "instruction offset",
                )
            )
        lines.append("")
    return "\n".join(lines)


class _VerificationRule(NamedTuple):
    kind: str
    field_offset: int = 0
    field_length: int = 0
    auxiliary: int = 0
    parameter: int = 0
    label: str = ""


_INSTRUCTION_FIELD_RULE_KIND = {
    core_rules.FieldRule.ZERO: "ZERO",
    core_rules.FieldRule.REGISTER_VALUE: "VALUE_REGISTER",
    core_rules.FieldRule.CONTROL_TARGET_S16: "CONTROL_TARGET_S16",
    core_rules.FieldRule.CONTROL_TARGET_S32: "CONTROL_TARGET_S32",
}

_CONTROL_FLOW_C_NAME = {
    control_flow: f"IREE_VM_BYTECODE_CONTROL_FLOW_{control_flow.name}"
    for control_flow in isa.ControlFlow
}


def _append_parameter_words(parameters: list[int], values: Iterable[int]) -> int:
    base = len(parameters)
    parameters.extend(values)
    if len(parameters) > 0xFFFFFFFF:
        raise ValueError("verification parameter table exceeds u32")
    return base


def _instruction_rules(instruction: isa.Instruction) -> list[_VerificationRule]:
    rules = []
    offsets_by_name = dict(
        zip(
            (item.field.name for item in instruction.fields),
            instruction.field_offsets,
            strict=True,
        )
    )
    for instruction_field, offset in zip(
        instruction.fields, instruction.field_offsets, strict=True
    ):
        rule = instruction_field.rule
        if rule == core_rules.FieldRule.ANY_BITS:
            continue
        kind = _INSTRUCTION_FIELD_RULE_KIND[rule]
        rules.append(
            _VerificationRule(
                kind,
                offset,
                instruction_field.field.byte_length,
                label=f"{instruction.mnemonic}.{instruction_field.field.name}",
            )
        )
    for record_rule in instruction.rules:
        if record_rule.kind == core_rules.RecordRuleKind.RETURN_SIGNATURE:
            rules.append(
                _VerificationRule("RETURN_SIGNATURE", label=instruction.mnemonic)
            )
        elif record_rule.kind == core_rules.RecordRuleKind.SWITCH_TARGETS:
            count_field, base_field = record_rule.fields
            rules.append(
                _VerificationRule(
                    "SWITCH_TARGETS",
                    offsets_by_name[count_field],
                    auxiliary=offsets_by_name[base_field],
                    label=instruction.mnemonic,
                )
            )
        else:
            raise ValueError(f"{instruction.mnemonic}: unsupported instruction rule")
    return rules


def _module_rules(
    specification: Specification,
    record: module.WireRecord,
    parameters: list[int],
) -> list[_VerificationRule]:
    rules = []
    offsets_by_name = dict(
        zip(
            (item.field.name for item in record.fields),
            record.field_offsets,
            strict=True,
        )
    )
    for wire_field, offset in zip(record.fields, record.field_offsets, strict=True):
        rule = wire_field.rule
        label = f"{record.name}.{wire_field.field.name}"
        if rule == module_rules.FieldRule.ANY_BITS:
            continue
        if rule == module_rules.FieldRule.ZERO:
            rules.append(
                _VerificationRule(
                    "ZERO", offset, wire_field.field.byte_length, label=label
                )
            )
        elif rule == module_rules.FieldRule.CORE_MAJOR:
            rules.append(
                _VerificationRule(
                    "CORE_MAJOR",
                    offset,
                    2,
                    parameter=specification.version.major,
                    label=label,
                )
            )
        elif rule == module_rules.FieldRule.CORE_REQUIRED_MINOR:
            rules.append(
                _VerificationRule(
                    "CORE_REQUIRED_MINOR",
                    offset,
                    2,
                    parameter=specification.version.minor,
                    label=label,
                )
            )
        elif (
            isinstance(rule, module.FieldRuleUse)
            and rule.kind == module_rules.FieldRule.EXACT_BYTES
        ):
            expected = rule.data
            padded = expected + bytes((-len(expected)) % 4)
            words = (
                int.from_bytes(padded[index : index + 4], "little")
                for index in range(0, len(padded), 4)
            )
            parameter = _append_parameter_words(parameters, words)
            rules.append(
                _VerificationRule(
                    "EXACT_BYTES",
                    offset,
                    len(expected),
                    len(padded) // 4,
                    parameter,
                    label,
                )
            )
        elif (
            isinstance(rule, module.FieldRuleUse)
            and rule.kind == module_rules.FieldRule.ALLOWED_RANGE
        ):
            parameter = _append_parameter_words(parameters, rule.values)
            rules.append(
                _VerificationRule(
                    "ALLOWED_RANGE",
                    offset,
                    wire_field.field.byte_length,
                    2,
                    parameter,
                    label,
                )
            )
        elif (
            isinstance(rule, module.FieldRuleUse)
            and rule.kind == module_rules.FieldRule.SIGNATURE_DESCRIPTOR
        ):
            rules.append(
                _VerificationRule(
                    "SIGNATURE_DESCRIPTOR",
                    offsets_by_name[rule.fields[0]],
                    2,
                    offset,
                    label=label,
                )
            )
        else:
            raise ValueError(f"{label}: unsupported module field rule")
    return rules


class _InstructionVerificationDescriptor(NamedTuple):
    rule_base: int
    control_flow: isa.ControlFlow
    rule_count: int
    byte_length: int


def _instruction_descriptor(
    instruction: isa.Instruction, rule_base: int, rule_count: int
) -> _InstructionVerificationDescriptor:
    if rule_base > 0xFFFF or rule_count > 0xF or instruction.byte_length > 0xFF:
        raise ValueError(f"{instruction.mnemonic}: verification descriptor overflow")
    return _InstructionVerificationDescriptor(
        rule_base,
        instruction.control_flow,
        rule_count,
        instruction.byte_length,
    )


def _pack_record_descriptor(
    record: module.WireRecord, rule_base: int, rule_count: int
) -> int:
    if rule_base > 0xFFFF or rule_count > 0xFF or record.byte_length > 0xFF:
        raise ValueError(f"{record.name}: verification descriptor overflow")
    return (rule_base << 16) | (rule_count << 8) | record.byte_length


def render_verification_data(specification: Specification) -> str:
    """Renders pointer-free runtime verification tables."""

    instruction_descriptors: list[_InstructionVerificationDescriptor | None] = [
        None
    ] * 256
    rules: list[_VerificationRule] = []
    parameters: list[int] = []
    for instruction in specification.instructions:
        instruction_rules = _instruction_rules(instruction)
        instruction_descriptors[instruction.opcode] = _instruction_descriptor(
            instruction, len(rules), len(instruction_rules)
        )
        rules.extend(instruction_rules)

    record_descriptors = []
    for record in specification.records:
        record_rules = _module_rules(specification, record, parameters)
        record_descriptors.append(
            _pack_record_descriptor(record, len(rules), len(record_rules))
        )
        rules.extend(record_rules)

    lines = [
        _COPYRIGHT.rstrip(),
        "",
        _GENERATED.rstrip(),
        "",
        '#include "iree/base/alignment.h"',
        '#include "iree/vm/bytecode/verification.h"',
        '#include "iree/vm/bytecode/wire/module.h"',
        "",
        'static_assert(sizeof(iree_vm_bytecode_verification_rule_t) == 8u, "verification rule packing");',
        "",
        "#define IREE_VM_BYTECODE_PACK_INSTRUCTION_VERIFICATION(\\",
        "    rule_base, control_flow, rule_count, byte_length)       \\",
        "  (((uint32_t)(rule_base) << 16) |                          \\",
        "   ((uint32_t)(control_flow) << 12) |                       \\",
        "   ((uint32_t)(rule_count) << 8) | (uint32_t)(byte_length))",
        "",
        "const uint32_t iree_vm_bytecode_instruction_verification[256] = {",
    ]
    for descriptor in instruction_descriptors:
        if descriptor is None:
            lines.append("    UINT32_C(0),")
        else:
            lines.append(
                "    IREE_VM_BYTECODE_PACK_INSTRUCTION_VERIFICATION(%du, %s, %du, %du),"
                % (
                    descriptor.rule_base,
                    _CONTROL_FLOW_C_NAME[descriptor.control_flow],
                    descriptor.rule_count,
                    descriptor.byte_length,
                )
            )
    lines.extend(
        [
            "};",
            "",
            "#undef IREE_VM_BYTECODE_PACK_INSTRUCTION_VERIFICATION",
            "",
            "const uint32_t iree_vm_bytecode_module_record_verification[] = {",
        ]
    )
    lines.extend(f"    UINT32_C(0x{value:08X})," for value in record_descriptors)
    lines.extend(
        [
            "};",
            "",
            "const iree_vm_bytecode_verification_rule_t iree_vm_bytecode_verification_rules[] = {",
        ]
    )
    for rule in rules:
        lines.append(
            "    {UINT32_C(0x%08X), IREE_VM_BYTECODE_VERIFICATION_RULE_%s, %du, %du, %du},  // %s"
            % (
                rule.parameter,
                rule.kind,
                rule.field_offset,
                rule.field_length,
                rule.auxiliary,
                rule.label,
            )
        )
    lines.extend(
        [
            "};",
            f"const uint16_t iree_vm_bytecode_verification_rule_count = {len(rules)}u;",
            "",
            "const uint32_t iree_vm_bytecode_verification_parameters[] = {",
        ]
    )
    lines.extend(f"    UINT32_C(0x{value:08X})," for value in parameters)
    lines.extend(
        [
            "};",
            f"const uint16_t iree_vm_bytecode_verification_parameter_count = {len(parameters)}u;",
            "",
        ]
    )
    return "\n".join(lines)


def _c_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def render_tooling_data(specification: Specification) -> str:
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
            raise ValueError(f"tooling string exceeds BSTRING limit: {value!r}")
        if byte_length + 1 + len(encoded) > 0xFFFF:
            raise ValueError("tooling BSTRING table exceeds u16 offsets")
        offset = byte_length
        offsets[value] = offset
        strings.append(value)
        byte_length += 1 + len(encoded)
        return offset

    instruction_offsets = [0] * 256
    for instruction in specification.instructions:
        instruction_offsets[instruction.opcode] = intern(instruction.mnemonic)
    record_offsets = [intern(record.name) for record in specification.records]

    lines = [
        _GENERATED.rstrip(),
        "",
        "static const uint8_t iree_vm_bytecode_tooling_strings[] =",
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
