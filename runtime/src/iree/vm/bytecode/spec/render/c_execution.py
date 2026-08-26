# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Renders compact interpreter verification and dispatch rows."""

from __future__ import annotations

from collections.abc import Sequence

from execution import ExecutableInstruction
from model.isa import Instruction, InstructionFieldRole
from model.isa.validation import (
    ALLOWED_RANGE,
    ANY_BITS,
    CONSTANT_POOL_ORDINAL,
    GLOBAL_ORDINAL,
    REGISTER_REF,
    REGISTER_VALUE,
    RODATA_ORDINAL,
    SELECTOR,
    ZERO,
)
from model.schema import EntityReference, ScalarEncoding
from model.specification import Projection


def _opcode_token(mnemonic: str) -> str:
    return mnemonic.replace(".", "_").upper()


def _handler_label(mnemonic: str) -> str:
    return mnemonic.replace(".", "_")


def _require_field(
    instruction: Instruction,
    entities_by_id: dict[str, object],
    *,
    offset: int,
    byte_length: int,
    rule_id: str,
    roles: tuple[InstructionFieldRole, ...],
    array_length: int = 1,
    rule_arguments: tuple[object, ...] | None = None,
) -> None:
    fields = tuple(field for field in instruction.fields if field.offset == offset)
    if len(fields) != 1:
        raise ValueError(
            f"{instruction.mnemonic}: verification form requires one field "
            f"at byte {offset}"
        )
    field = fields[0]
    encoding = entities_by_id.get(field.encoding_id)
    if (
        not isinstance(encoding, ScalarEncoding)
        or encoding.byte_length != byte_length
        or field.array_length != array_length
        or field.role not in roles
        or tuple(rule.rule_id for rule in field.validation) != (rule_id,)
        or (
            rule_arguments is not None
            and tuple(field.validation[0].arguments) != rule_arguments
        )
    ):
        raise ValueError(
            f"{instruction.mnemonic}.{field.name}: field does not match its "
            "runtime verification form"
        )


def _validate_verification_form(
    instruction: Instruction,
    verification_form: str,
    entities_by_id: dict[str, object],
) -> None:
    verified_field_offsets: set[int] = set()
    result_or_operand = (
        InstructionFieldRole.RESULT,
        InstructionFieldRole.OPERAND,
    )

    def require_field(
        offset: int,
        byte_length: int,
        rule_id: str,
        roles: tuple[InstructionFieldRole, ...],
        *,
        array_length: int = 1,
        rule_arguments: tuple[object, ...] | None = None,
    ) -> None:
        _require_field(
            instruction,
            entities_by_id,
            offset=offset,
            byte_length=byte_length,
            rule_id=rule_id,
            roles=roles,
            array_length=array_length,
            rule_arguments=rule_arguments,
        )
        verified_field_offsets.add(offset)

    def require_value(offset: int) -> None:
        require_field(
            offset,
            1,
            REGISTER_VALUE.entity_id,
            result_or_operand,
        )

    def require_zero(offset: int, byte_length: int, *, array_length: int = 1) -> None:
        require_field(
            offset,
            byte_length,
            ZERO.entity_id,
            (InstructionFieldRole.PADDING,),
            array_length=array_length,
        )

    if verification_form in ("CONTROL_BLOCK", "CONTROL_RETURN"):
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: control record is not 4 bytes")
        require_zero(1, 1, array_length=3)
    elif verification_form in ("CONSTANT_ZERO", "CONSTANT_I32", "CONSTANT_I64"):
        require_value(1)
        require_zero(2, 2)
    elif verification_form == "CONSTANT_S16":
        require_value(1)
    elif verification_form in ("CONSTANT_POOL_LOAD_I32", "CONSTANT_POOL_LOAD_I64"):
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: constant-pool load is not 4 bytes"
            )
        require_value(1)
        require_field(
            2,
            2,
            CONSTANT_POOL_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "VALUE_UNARY_4":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: unary record is not 4 bytes")
        require_value(1)
        require_value(2)
        require_zero(3, 1)
    elif verification_form == "VALUE_SELECT":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: select record is not 8 bytes")
        for offset in range(1, 5):
            require_value(offset)
        require_zero(5, 1, array_length=3)
    elif verification_form in (
        "GLOBAL_VALUE_IMMUTABLE_LOAD",
        "GLOBAL_VALUE_IMMUTABLE_STORE",
        "GLOBAL_VALUE_MUTABLE_LOAD",
        "GLOBAL_VALUE_MUTABLE_STORE",
    ):
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: global record is not 4 bytes")
        require_value(1)
        require_field(
            2,
            2,
            GLOBAL_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form == "VALUE_BINARY_4":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: binary record is not 4 bytes")
        for offset in range(1, 4):
            require_value(offset)
    elif verification_form == "INTEGER_COMPARE":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: compare record is not 8 bytes")
        for offset in range(1, 4):
            require_value(offset)
        require_field(
            4,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference("core.selector.integer.compare"),),
        )
        require_zero(5, 1, array_length=3)
    elif verification_form == "INTEGER_LEA":
        if instruction.byte_length != 8:
            raise ValueError(f"{instruction.mnemonic}: LEA record is not 8 bytes")
        for offset in range(1, 4):
            require_value(offset)
        require_zero(5, 1)
    elif verification_form in (
        "INTEGER_CEILDIV_POW2_U32",
        "INTEGER_CEILDIV_POW2_U64",
    ):
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: power-of-two ceildiv record is not 4 bytes"
            )
        require_value(1)
        require_value(2)
        maximum_log2 = 31 if verification_form.endswith("U32") else 63
        require_field(
            3,
            1,
            ALLOWED_RANGE.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(0, maximum_log2),
        )
    elif verification_form == "BUFFER_RODATA_LOAD":
        if instruction.byte_length != 4:
            raise ValueError(f"{instruction.mnemonic}: rodata load is not 4 bytes")
        require_field(
            1,
            1,
            REGISTER_REF.entity_id,
            (InstructionFieldRole.RESULT,),
        )
        require_field(
            2,
            2,
            RODATA_ORDINAL.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
        )
    elif verification_form in (
        "CONVERSION_INTEGER",
        "CONVERSION_FLOAT_EXTEND",
        "CONVERSION_FLOAT_TO_INTEGER",
    ):
        if instruction.byte_length != 4:
            raise ValueError(
                f"{instruction.mnemonic}: conversion record is not 4 bytes"
            )
        require_value(1)
        require_value(2)
        selector_table_id = {
            "CONVERSION_INTEGER": "core.selector.integer.convert",
            "CONVERSION_FLOAT_EXTEND": "core.selector.float.extend",
            "CONVERSION_FLOAT_TO_INTEGER": "core.selector.float.to.integer",
        }[verification_form]
        require_field(
            3,
            1,
            SELECTOR.entity_id,
            (InstructionFieldRole.IMMEDIATE,),
            rule_arguments=(EntityReference(selector_table_id),),
        )
    else:
        raise ValueError(
            f"{instruction.mnemonic}: unknown runtime verification form "
            f"{verification_form!r}"
        )
    unchecked_validated_fields = tuple(
        field.name
        for field in instruction.fields
        if any(rule.rule_id != ANY_BITS.entity_id for rule in field.validation)
        and field.offset not in verified_field_offsets
    )
    if unchecked_validated_fields:
        raise ValueError(
            f"{instruction.mnemonic}: runtime verification form leaves "
            f"validated fields unchecked: {unchecked_validated_fields!r}"
        )


def render_execution_tables(
    isa_projection: Projection,
    executable_instructions: Sequence[ExecutableInstruction],
) -> str:
    """Renders one multi-include table for verifier and interpreter use."""

    entities_by_id = isa_projection.entity_map()
    instructions_by_mnemonic = {
        entity.mnemonic: entity
        for entity in isa_projection.entities
        if isinstance(entity, Instruction)
    }
    resolved: list[tuple[Instruction, ExecutableInstruction]] = []
    seen_mnemonics: set[str] = set()
    seen_opcodes: set[int] = set()
    for executable in executable_instructions:
        if executable.mnemonic in seen_mnemonics:
            raise ValueError(
                f"duplicate executable instruction {executable.mnemonic!r}"
            )
        instruction = instructions_by_mnemonic.get(executable.mnemonic)
        if instruction is None:
            raise ValueError(f"unknown executable instruction {executable.mnemonic!r}")
        if instruction.since.domain != "core":
            raise ValueError(
                f"executable instruction {executable.mnemonic!r} is not Core"
            )
        if instruction.opcode in seen_opcodes:
            raise ValueError(f"duplicate executable opcode 0x{instruction.opcode:02X}")
        if instruction.byte_length > 0xFF:
            raise ValueError(f"{executable.mnemonic}: record length does not fit in u8")
        _validate_verification_form(
            instruction, executable.verification_form, entities_by_id
        )
        seen_mnemonics.add(executable.mnemonic)
        seen_opcodes.add(instruction.opcode)
        resolved.append((instruction, executable))
    resolved.sort(key=lambda pair: pair[0].opcode)

    by_opcode = {
        instruction.opcode: (instruction, executable)
        for instruction, executable in resolved
    }
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        "// GENERATED FILE: DO NOT EDIT.",
        "// Private build-tree projection: executable Core instruction rows.",
        "// clang-format off",
        "",
        "#if defined(IREE_VM_BYTECODE_DEFINE_EXECUTION_INFO_ROWS)",
    ]
    for opcode in range(256):
        pair = by_opcode.get(opcode)
        if pair is None:
            lines.append("IREE_VM_BYTECODE_EXECUTION_INFO_ROW(0, NONE)")
        else:
            instruction, executable = pair
            lines.append(
                "IREE_VM_BYTECODE_EXECUTION_INFO_ROW("
                f"{instruction.byte_length}, {executable.verification_form})"
            )
    lines.extend(
        (
            "#elif defined(IREE_VM_BYTECODE_DEFINE_EXECUTABLE_OPCODE_LIST)",
            "#define IREE_VM_BYTECODE_EXECUTABLE_OPCODE_LIST(OP) \\",
        )
    )
    for index, (instruction, _) in enumerate(resolved):
        continuation = " \\" if index + 1 != len(resolved) else ""
        lines.append(
            f"  OP({_opcode_token(instruction.mnemonic)}, "
            f"{_handler_label(instruction.mnemonic)}){continuation}"
        )
    lines.extend(
        (
            "#else",
            '#error "define one VM bytecode execution-table projection"',
            "#endif",
            "",
            "// clang-format on",
            "",
        )
    )
    return "\n".join(lines)
