# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates the complete Loom-facing VM table family."""

from __future__ import annotations

import argparse
from pathlib import Path

from iree.vm.bytecode.spec import isa
from iree.vm.bytecode.spec.specification import SPECIFICATION

from loom.gen.ops.c_names import c_enum_name
from loom.gen.support.files import write_text_file
from loom.target.arch.vm.projection import SOURCE_LOWERINGS, SourceLowering

_COPYRIGHT = """\
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

_FIELD_ROLE_NAME = {role: f"LOOM_VM_INSTRUCTION_FIELD_ROLE_{role.name}" for role in isa.FieldRole}


def _identifier(value: str) -> str:
    return value.replace(".", "_").upper()


def _pack_instruction_descriptor(instruction: isa.Instruction, field_base: int) -> int:
    if field_base > 0xFFFF or len(instruction.fields) > 0xFF or instruction.byte_length > 0xFF:
        raise ValueError(f"{instruction.mnemonic}: Loom descriptor overflow")
    return (field_base << 16) | (len(instruction.fields) << 8) | instruction.byte_length


def _render_source_lowering(lowering: SourceLowering) -> str:
    source_op = c_enum_name(lowering.source_op)
    scalar_type = f"LOOM_SCALAR_TYPE_{lowering.scalar_type.name}"
    opcode = f"IREE_VM_BYTECODE_OPCODE_{_identifier(lowering.instruction.mnemonic)}"
    return f"    {{{source_op}, {scalar_type}, {opcode}}},"


def generate_source() -> str:
    """Returns data-only C tables for VM lowering and record emission."""

    descriptors = [0] * 256
    fields = []
    for instruction in sorted(SPECIFICATION.instructions, key=lambda item: item.opcode):
        descriptors[instruction.opcode] = _pack_instruction_descriptor(instruction, len(fields))
        fields.extend(
            (
                offset,
                field.field.byte_length,
                field.role,
                instruction.mnemonic,
            )
            for field, offset in zip(instruction.fields, instruction.field_offsets, strict=True)
        )

    lowerings = sorted(
        SOURCE_LOWERINGS,
        key=lambda lowering: (
            lowering.source_op_kind,
            lowering.scalar_type.value,
        ),
    )
    if len({(item.source_op, item.scalar_type) for item in lowerings}) != len(lowerings):
        raise ValueError("duplicate Loom VM source lowering")

    lines = [
        _COPYRIGHT.rstrip(),
        "",
        "// Generated from the authoritative VM and Loom declarations. Do not edit.",
        "// clang-format off",
        "",
        '#include "loom/target/arch/vm/tables.h"',
        "",
        '#include "iree/vm/bytecode/wire/core.h"',
        '#include "loom/ops/scalar/ops.h"',
        "",
        'static_assert(sizeof(loom_vm_instruction_field_t) == 4u, "VM field row packing");',
        'static_assert(sizeof(loom_vm_source_lowering_t) == 4u, "VM source lowering packing");',
        "",
        "const uint32_t loom_vm_instruction_descriptors[256] = {",
    ]
    lines.extend(f"    UINT32_C(0x{descriptor:08X})," for descriptor in descriptors)
    lines.extend(
        [
            "};",
            "",
            "const loom_vm_instruction_field_t loom_vm_instruction_fields[] = {",
        ]
    )
    for offset, byte_length, role, mnemonic in fields:
        role_name = _FIELD_ROLE_NAME[role]
        lines.append(f"    {{{offset}u, {byte_length}u, {role_name}, 0u}},  // {mnemonic}")
    lines.extend(
        [
            "};",
            f"const uint16_t loom_vm_instruction_field_count = {len(fields)}u;",
            "",
            "const loom_vm_source_lowering_t loom_vm_source_lowerings[] = {",
        ]
    )
    lines.extend(_render_source_lowering(lowering) for lowering in lowerings)
    lines.extend(
        [
            "};",
            f"const uint16_t loom_vm_source_lowering_count = {len(lowerings)}u;",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Loom VM target tables.")
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args()
    write_text_file(arguments.source, generate_source())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
