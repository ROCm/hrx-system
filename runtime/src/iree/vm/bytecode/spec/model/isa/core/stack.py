# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 frame-local byte-stack instructions."""

from __future__ import annotations

from model.isa import (
    FailureCase,
    Instruction,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
    RefNullPolicy,
    RefOwnership,
    RuntimeRefPolicy,
)
from model.isa.declarations import (
    core_instruction,
    instruction_field,
    ref_register,
    value_register,
    zero_padding,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import (
    ALLOWED_RANGE,
    ALLOWED_VALUES,
    ANY_BITS,
    LOCAL_BYTES_FIXED_BASE,
    LOCAL_BYTES_RANGE_BASE,
    LOCAL_BYTES_RANGE_LENGTH,
    LOCAL_BYTES_RANGE_MEMORY_FORMAT,
    LOCAL_BYTES_REPEATED_BASE,
    LOCAL_BYTES_REPEATED_COUNT,
    RODATA_ORDINAL,
    RODATA_STATIC_OFFSET,
    SELECTOR,
    VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT,
    ZERO,
)
from model.schema import I16, U8, U16, U32, EntityReference, FieldReference, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.stack",
    since=CORE_0,
    summary="Checked access to one frame-local byte array.",
    dependencies=("core.contract.machine",),
    document_order=10,
    normative_text=(
        "Each frame owns one fixed local byte array of 0 to 65,535 bytes whose "
        "entry contents are unspecified. A stack base is an unsigned byte offset, not a pointer "
        "or value-cell ordinal. Verification evaluates every static range in "
        "a mathematical type that cannot wrap. A zero-length range accepts "
        "base equal to the array length and execution forms no pointer for it. "
        "Lane memory format f = element_log2*4 + lane_log2 selects "
        "element_bytes = 1 << element_log2 and lane_count = 1 << lane_log2, "
        "where both logarithms are in 0..3 and bits 7:4 are zero. Lane zero "
        "uses the lowest addressed bytes and lowest value register. Lane "
        "access is little-endian and alignment-independent: loads zero-extend "
        "into complete value cells and stores use only the selected low bits. "
        "Fixed and repeated constant writers instead carry explicit natural-"
        "alignment requirements. No stack instruction allocates, grows, "
        "exposes, or retains the local array, changes ref ownership, or suspends."
    ),
)


def _field(
    name: str,
    offset: int,
    encoding_id: str,
    role: InstructionFieldRole,
    description: str,
    rule_id: str,
    arguments: tuple[object, ...] = (),
    *,
    array_length: int = 1,
):
    return instruction_field(
        name,
        offset,
        encoding_id,
        role,
        description,
        (RuleUse(rule_id, arguments),),
        array_length=array_length,
    )


def _memory_format(name: str, offset: int):
    table = SELECTOR_TABLES_BY_NAME["memory.format"]
    return _field(
        name,
        offset,
        U8.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Closed lane element-width and lane-count selector.",
        SELECTOR.entity_id,
        (EntityReference(table.entity_id),),
    )


def _memory_base(name: str, offset: int, format_field: str):
    return _field(
        name,
        offset,
        U16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Static base of an alignment-independent local-byte lane range.",
        LOCAL_BYTES_RANGE_MEMORY_FORMAT.entity_id,
        (FieldReference(format_field),),
    )


def _range_base(name: str, offset: int, length_field: str, description: str):
    return _field(
        name,
        offset,
        U16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        description,
        LOCAL_BYTES_RANGE_BASE.entity_id,
        (FieldReference(length_field),),
    )


def _range_length(name: str, offset: int):
    return _field(
        name,
        offset,
        U16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Exact byte length of the statically checked local range.",
        LOCAL_BYTES_RANGE_LENGTH.entity_id,
    )


def _lane_range_constraint(base_field: str, format_field: str):
    return RuleUse(
        VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT.entity_id,
        (FieldReference(base_field), FieldReference(format_field)),
    )


def _semantics(
    *,
    description: str,
    verification: tuple[str, ...],
    success: tuple[str, ...],
    assembly: tuple[str, ...],
    pseudocode: str,
    preconditions: tuple[str, ...] = (),
    failures: tuple[FailureCase, ...] = (),
    ownership: tuple[str, ...] = (),
    byte_length: int,
):
    return InstructionSemantics(
        description=description,
        verification=verification,
        preconditions=preconditions,
        success=(
            *success,
            f"The program counter advances by {byte_length} bytes.",
        ),
        failures=failures,
        ownership=ownership,
        assembly=assembly,
        pseudocode=pseudocode,
    )


STACK_LOAD = core_instruction(
    entity_id="core.instruction.stack.load",
    since=CORE_0,
    summary="Loads one fixed lane group from local bytes.",
    opcode=0xA8,
    mnemonic="stack.load",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "First destination value-register ordinal.",
        ),
        _memory_base("base_u16", 2, "format_u8"),
        _memory_format("format_u8", 4),
        zero_padding("zero_padding_u8", 5, 3),
    ),
    constraints=(_lane_range_constraint("dst_v8", "format_u8"),),
    semantics=_semantics(
        description=(
            "Loads the format-selected lane group from the statically selected "
            "local-byte range into consecutive value registers."
        ),
        verification=(
            "format_u8 must be an assigned memory.format selector.",
            "base_u16 plus the selected access length must fit local bytes, "
            "without an alignment requirement.",
            "dst_v8 plus the selected lane count must fit the value-register bank.",
            "Every zero_padding_u8 byte must equal zero.",
        ),
        success=(
            "Each destination receives one little-endian lane zero-extended "
            "to a complete 64-bit value cell, in increasing lane order.",
        ),
        assembly=(
            "%v<dst>.xN = stack.load #v<base> {format}",
            "%v4.x4 = stack.load #v128 {i32.x4}",
        ),
        pseudocode=(
            "format = verified_memory_format(format_u8);\n"
            "for (i = 0; i < format.lane_count; ++i) {\n"
            "  values[dst_v8 + i] = load_le(\n"
            "      format.element_bits,\n"
            "      local_bytes + base_u16 + i * format.element_bytes);\n"
            "}\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

STACK_STORE = core_instruction(
    entity_id="core.instruction.stack.store",
    since=CORE_0,
    summary="Stores one fixed lane group into local bytes.",
    opcode=0xA9,
    mnemonic="stack.store",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        zero_padding("zero_padding_u8", 1, 1),
        _memory_base("base_u16", 2, "format_u8"),
        value_register(
            "src_v8",
            4,
            InstructionFieldRole.OPERAND,
            "First source value-register ordinal.",
        ),
        _memory_format("format_u8", 5),
        _field(
            "zero_padding_u16",
            6,
            U16.entity_id,
            InstructionFieldRole.PADDING,
            "Canonical zero padding.",
            ZERO.entity_id,
        ),
    ),
    constraints=(_lane_range_constraint("src_v8", "format_u8"),),
    semantics=_semantics(
        description=(
            "Stores the low bits of consecutive source value registers into "
            "the format-selected local-byte lane group."
        ),
        verification=(
            "format_u8 must be an assigned memory.format selector.",
            "base_u16 plus the selected access length must fit local bytes, "
            "without an alignment requirement.",
            "src_v8 plus the selected lane count must fit the value-register bank.",
            "All padding bits must equal zero.",
        ),
        success=(
            "Each lane receives the selected low source bits in little-endian "
            "order, in increasing lane order.",
        ),
        assembly=("stack.store #v<base>, %v<src>.xN {format}",),
        pseudocode=(
            "format = verified_memory_format(format_u8);\n"
            "for (i = 0; i < format.lane_count; ++i) {\n"
            "  store_le(format.element_bits,\n"
            "           local_bytes + base_u16 + i * format.element_bytes,\n"
            "           values[src_v8 + i]);\n"
            "}\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)


def _indexed_stack_access(*, load: bool) -> Instruction:
    mnemonic = "stack.load.indexed" if load else "stack.store.indexed"
    register_name = "dst_v8" if load else "src_v8"
    register_role = (
        InstructionFieldRole.RESULT if load else InstructionFieldRole.OPERAND
    )
    fields = (
        (
            value_register(
                register_name,
                1,
                register_role,
                "First lane value-register ordinal.",
            ),
            _memory_base("base_u16", 2, "format_u8"),
            value_register(
                "index_v8",
                4,
                InstructionFieldRole.OPERAND,
                "Unsigned 64-bit dynamic index value.",
            ),
            _field(
                "scale_u8",
                5,
                U8.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Nonzero byte scale applied to index_v8.",
                ALLOWED_RANGE.entity_id,
                (1, 255),
            ),
            _memory_format("format_u8", 6),
            zero_padding("zero_padding_u8", 7, 1),
        )
        if load
        else (
            zero_padding("zero_padding_u8", 1, 1),
            _memory_base("base_u16", 2, "format_u8"),
            value_register(
                "index_v8",
                4,
                InstructionFieldRole.OPERAND,
                "Unsigned 64-bit dynamic index value.",
            ),
            _field(
                "scale_u8",
                5,
                U8.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Nonzero byte scale applied to index_v8.",
                ALLOWED_RANGE.entity_id,
                (1, 255),
            ),
            value_register(
                register_name,
                6,
                register_role,
                "First lane value-register ordinal.",
            ),
            _memory_format("format_u8", 7),
        )
    )
    action = "loads" if load else "stores"
    direction = "from" if load else "to"
    mutation = (
        "Every destination value register and every local byte remains unchanged."
        if load
        else "Every local byte and all VM register state remain unchanged."
    )
    execute = "load_lanes" if load else "store_lanes"
    execute_arguments = (
        "dst_v8, effective_u16, format" if load else "effective_u16, src_v8, format"
    )
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Dynamically indexes a checked local-byte lane {action[:-1]}.",
        opcode=0xAA if load else 0xAB,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=fields,
        constraints=(_lane_range_constraint(register_name, "format_u8"),),
        semantics=_semantics(
            description=(
                f"{action.capitalize()} one format-selected lane group {direction} "
                "effective_base = base_u16 + unsigned(index_v8) * scale_u8."
            ),
            verification=(
                "format_u8 must be assigned, scale_u8 must be in 1..255, and "
                "all register ordinals and padding must be valid.",
                "base_u16 plus the selected access length must fit local bytes; "
                "the complete lane register range must fit the value bank.",
            ),
            preconditions=(
                "index_v8 must not exceed UINT16_MAX and its scaled relative "
                "offset must fit after base_u16 and the selected access length.",
            ),
            success=(
                "All address checks finish before any destination publication. "
                "The lane transfer then follows the ordinary stack lane rules.",
            ),
            failures=(
                FailureCase(
                    "out_of_range",
                    "index_v8 exceeds UINT16_MAX or the scaled access is out of range.",
                    mutation,
                ),
            ),
            assembly=(
                ("%v<dst>.xN = stack.load.indexed #v<base>, %v<index> {scale, format}")
                if load
                else (
                    "stack.store.indexed #v<base>, %v<index>, "
                    "%v<src>.xN {scale, format}"
                ),
            ),
            pseudocode=(
                "format = verified_memory_format(format_u8);\n"
                "available_u32 = local_byte_length -\n"
                "    format.access_length - base_u16;\n"
                "index_u64 = values[index_v8];\n"
                "if (index_u64 > UINT16_MAX) fail(out_of_range);\n"
                "relative_u32 = u32(index_u64) * u32(scale_u8);\n"
                "if (relative_u32 > available_u32) fail(out_of_range);\n"
                "effective_u16 = u16(base_u16 + relative_u32);\n"
                f"{execute}({execute_arguments});\n"
                "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


STACK_LOAD_INDEXED = _indexed_stack_access(load=True)
STACK_STORE_INDEXED = _indexed_stack_access(load=False)

STACK_FILL = core_instruction(
    entity_id="core.instruction.stack.fill",
    since=CORE_0,
    summary="Repeats a low-byte pattern across a local-byte range.",
    opcode=0xAC,
    mnemonic="stack.fill",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        zero_padding("zero_padding_u8", 1, 1),
        _range_base(
            "target_base_u16",
            2,
            "length_u16",
            "Static local-byte target base.",
        ),
        _range_length("length_u16", 4),
        value_register(
            "pattern_v8",
            6,
            InstructionFieldRole.OPERAND,
            "Low-byte pattern source.",
        ),
        _field(
            "pattern_width_u8",
            7,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Pattern width in bytes.",
            ALLOWED_VALUES.entity_id,
            ((1, 2, 4, 8),),
        ),
    ),
    semantics=_semantics(
        description=(
            "Repeats the low one, two, four, or eight little-endian bytes of "
            "pattern_v8 across the exact target range; the final repetition "
            "may be partial."
        ),
        verification=(
            "The target range must fit local bytes and pattern_v8 must be valid.",
            "pattern_width_u8 must equal 1, 2, 4, or 8 and padding must be zero.",
        ),
        success=(
            "Exactly length_u16 bytes receive the repeated pattern. A zero "
            "length writes nothing and forms no local pointer.",
        ),
        assembly=("stack.fill #v<target>.x<length>, %v<pattern> {pattern_width}",),
        pseudocode=(
            "pattern_bits = values[pattern_v8];\n"
            "for (i = 0; i < length_u16; ++i) {\n"
            "  local_bytes[target_base_u16 + i] =\n"
            "      u8(pattern_bits >> (8 * (i % pattern_width_u8)));\n"
            "}\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

STACK_COPY = core_instruction(
    entity_id="core.instruction.stack.copy",
    since=CORE_0,
    summary="Moves one possibly overlapping local-byte range.",
    opcode=0xAD,
    mnemonic="stack.copy",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        zero_padding("zero_padding_u8", 1, 1),
        _range_base("target_u16", 2, "length_u16", "Static local-byte target base."),
        _range_base("source_u16", 4, "length_u16", "Static local-byte source base."),
        _range_length("length_u16", 6),
    ),
    semantics=_semantics(
        description=(
            "Moves one local-byte range to another with memmove semantics, "
            "including every overlap and equal-base case."
        ),
        verification=(
            "Both complete ranges must independently fit local bytes and "
            "zero_padding_u8 must equal zero.",
        ),
        success=(
            "The target receives the source bytes as if through a temporary "
            "copy. A zero length forms no pointer and performs no host call.",
        ),
        assembly=("stack.copy #v<target>.x<length>, #v<source>.x<length>",),
        pseudocode=(
            "if (length_u16 != 0) {\n"
            "  move_bytes(local_bytes + target_u16,\n"
            "             local_bytes + source_u16, length_u16);\n"
            "}\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

STACK_COMPARE = core_instruction(
    entity_id="core.instruction.stack.compare",
    since=CORE_0,
    summary="Lexicographically compares two local-byte ranges.",
    opcode=0xAE,
    mnemonic="stack.compare",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Canonical signed i32 comparison result.",
        ),
        _range_base("lhs_u16", 2, "length_u16", "Static left range base."),
        _range_base("rhs_u16", 4, "length_u16", "Static right range base."),
        _range_length("length_u16", 6),
    ),
    semantics=_semantics(
        description=(
            "Lexicographically compares equal-length local ranges as unsigned "
            "bytes and returns canonical i32 -1, 0, or +1."
        ),
        verification=(
            "Both complete ranges must independently fit local bytes and "
            "dst_v8 must be a valid value register.",
        ),
        success=(
            "dst_v8 receives UINT32_MAX for less, zero for equal, or one for "
            "greater, with its high 32 cell bits zero. Empty ranges compare equal.",
        ),
        assembly=("%v<dst> = stack.compare #v<lhs>.x<length>, #v<rhs>.x<length>",),
        pseudocode=(
            "ordering = length_u16 == 0 ? 0 : compare_unsigned_bytes(\n"
            "    local_bytes + lhs_u16, local_bytes + rhs_u16, length_u16);\n"
            "values[dst_v8] = ordering < 0 ? UINT32_MAX :\n"
            "                 ordering > 0 ? 1 : 0;\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

STACK_COPY_RODATA = core_instruction(
    entity_id="core.instruction.stack.copy.rodata",
    since=CORE_0,
    summary="Copies immutable module bytes into local bytes.",
    opcode=0xAF,
    mnemonic="stack.copy.rodata",
    byte_length=12,
    family_id=FAMILY.entity_id,
    fields=(
        zero_padding("zero_padding_u8", 1, 1),
        _range_base("target_u16", 2, "length_u16", "Static local-byte target base."),
        _field(
            "rodata_u16",
            4,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Direct module rodata-block ordinal.",
            RODATA_ORDINAL.entity_id,
        ),
        _range_length("length_u16", 6),
        _field(
            "source_offset_u32",
            8,
            U32.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Static source offset in the selected rodata block.",
            RODATA_STATIC_OFFSET.entity_id,
            (FieldReference("rodata_u16"), FieldReference("length_u16")),
        ),
    ),
    semantics=_semantics(
        description=(
            "Copies a statically selected immutable module-rodata range into "
            "the statically selected local-byte target without creating a buffer."
        ),
        verification=(
            "The target range must fit local bytes, rodata_u16 must be valid, "
            "and the source range must fit that rodata block.",
            "zero_padding_u8 must equal zero.",
        ),
        success=(
            "The target receives exactly length_u16 source bytes. A zero length "
            "accepts both one-past-end bases and forms neither pointer.",
        ),
        assembly=(
            "stack.copy.rodata #v<target>.x<length>, @rodata<ordinal>+<source_offset>",
        ),
        pseudocode=(
            "source = verified_rodata_blocks[rodata_u16];\n"
            "if (length_u16 != 0) {\n"
            "  copy_bytes(local_bytes + target_u16,\n"
            "             source.data + source_offset_u32, length_u16);\n"
            "}\n"
            "pc = pc + 12;"
        ),
        byte_length=12,
    ),
)


def _buffer_ref(offset: int):
    return ref_register(
        "buffer_r8",
        offset,
        InstructionFieldRole.OPERAND,
        "Required exact vm.buffer ref borrowed for this instruction.",
        RuntimeRefPolicy(
            "vm.buffer",
            RefNullPolicy.REQUIRED,
            RefOwnership.BORROW,
        ),
    )


def _buffer_copy(*, from_buffer: bool) -> Instruction:
    mnemonic = "stack.copy.from.buffer" if from_buffer else "stack.copy.to.buffer"
    fields = (
        (
            zero_padding("zero_padding_u8", 1, 1),
            _range_base(
                "target_u16", 2, "length_u16", "Static local-byte target base."
            ),
            _buffer_ref(4),
            value_register(
                "source_offset_v8",
                5,
                InstructionFieldRole.OPERAND,
                "Unsigned dynamic source offset in buffer_r8.",
            ),
            _range_length("length_u16", 6),
        )
        if from_buffer
        else (
            _buffer_ref(1),
            value_register(
                "target_offset_v8",
                2,
                InstructionFieldRole.OPERAND,
                "Unsigned dynamic target offset in buffer_r8.",
            ),
            zero_padding("zero_padding_u8", 3, 1),
            _range_base(
                "source_u16", 4, "length_u16", "Static local-byte source base."
            ),
            _range_length("length_u16", 6),
        )
    )
    access = "READ" if from_buffer else "WRITE"
    dynamic_offset = "source_offset_v8" if from_buffer else "target_offset_v8"
    destination = "local-byte target" if from_buffer else "buffer target"
    copy = (
        "copy_bytes(local_bytes + target_u16,\n"
        "             buffer.data + dynamic_offset_u64, length_u16);"
        if from_buffer
        else "copy_bytes(buffer.data + dynamic_offset_u64,\n"
        "             local_bytes + source_u16, length_u16);"
    )
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=(
            "Copies a checked vm.buffer range into local bytes."
            if from_buffer
            else "Copies local bytes into a checked vm.buffer range."
        ),
        opcode=0xB0 if from_buffer else 0xB1,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=fields,
        semantics=_semantics(
            description=(
                f"Copies between a statically verified local range and a "
                f"dynamically offset exact vm.buffer with {access} access."
            ),
            verification=(
                "The complete local range and both register ordinals must be valid.",
                "zero_padding_u8 must equal zero.",
            ),
            preconditions=(
                "buffer_r8 must hold a non-null exact vm.buffer granting "
                f"{access} access, and the dynamic offset plus length_u16 must "
                "fit its byte length. These checks also apply at zero length.",
            ),
            success=(
                "After every check succeeds, exactly length_u16 bytes are copied. "
                "The local frame and vm.buffer cannot alias; zero length forms "
                "no pointer.",
            ),
            failures=(
                FailureCase(
                    "failed_precondition",
                    "buffer_r8 contains canonical null.",
                    f"The {destination} remains unchanged.",
                ),
                FailureCase(
                    "invalid_argument",
                    "buffer_r8 contains a non-null ref of another type.",
                    f"The {destination} remains unchanged.",
                ),
                FailureCase(
                    "permission_denied",
                    f"The buffer does not grant {access} access.",
                    f"The {destination} remains unchanged.",
                ),
                FailureCase(
                    "out_of_range",
                    "The dynamic buffer range is outside buffer.length.",
                    f"The {destination} remains unchanged.",
                ),
            ),
            ownership=(
                "buffer_r8 is borrowed only for synchronous validation and "
                "copy; its ref count and register state do not change.",
            ),
            assembly=(
                (
                    "stack.copy.from.buffer #v<target>.x<length>, "
                    "%r<buffer>, %v<source_offset>"
                )
                if from_buffer
                else (
                    "stack.copy.to.buffer %r<buffer>, %v<target_offset>, "
                    "#v<source>.x<length>"
                ),
            ),
            pseudocode=(
                "buffer = check_deref(refs[buffer_r8], vm_buffer_type);\n"
                f"require_access(buffer, {access});\n"
                f"dynamic_offset_u64 = values[{dynamic_offset}];\n"
                "require checked_range(\n"
                "    dynamic_offset_u64, length_u16, buffer.length);\n"
                "if (length_u16 != 0) {\n"
                f"  {copy}\n"
                "}\n"
                "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


STACK_COPY_FROM_BUFFER = _buffer_copy(from_buffer=True)
STACK_COPY_TO_BUFFER = _buffer_copy(from_buffer=False)


def _repeated_constant(*, cell_byte_length: int, opcode: int) -> Instruction:
    cell_name = f"i{cell_byte_length * 8}"
    mnemonic = f"stack.const.s16.{cell_name}"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"Fills aligned {cell_name} cells from one signed immediate.",
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=(
            zero_padding("zero_padding_u8", 1, 1),
            _field(
                "target_u16",
                2,
                U16.entity_id,
                InstructionFieldRole.IMMEDIATE,
                f"{cell_byte_length}-byte-aligned local target base.",
                LOCAL_BYTES_REPEATED_BASE.entity_id,
                (
                    FieldReference("count_u16"),
                    cell_byte_length,
                    cell_byte_length,
                ),
            ),
            _field(
                "count_u16",
                4,
                U16.entity_id,
                InstructionFieldRole.IMMEDIATE,
                f"Number of {cell_name} cells to write.",
                LOCAL_BYTES_REPEATED_COUNT.entity_id,
            ),
            _field(
                "immediate_i16",
                6,
                I16.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Signed little-endian two's-complement fill immediate.",
                ANY_BITS.entity_id,
            ),
        ),
        semantics=_semantics(
            description=(
                f"Sign-extends immediate_i16 to {cell_name} and writes "
                f"count_u16 consecutive little-endian {cell_name} cells."
            ),
            verification=(
                f"target_u16 must be {cell_byte_length}-byte aligned and the "
                f"count_u16 * {cell_byte_length} extent must fit local bytes.",
                "zero_padding_u8 must equal zero; every immediate bit pattern is valid.",
            ),
            success=(
                "Every selected cell receives the same sign-extended immediate. "
                "A zero count writes nothing while retaining the base-alignment check.",
            ),
            assembly=(f"{mnemonic} #v<target>.x<count>, <immediate>",),
            pseudocode=(
                f"element = sext_u{cell_byte_length * 8}(\n"
                "    load_le(16, &record[6]), 16);\n"
                "for (i = 0; i < count_u16; ++i) {\n"
                f"  store_le({cell_byte_length * 8}, local_bytes + target_u16 +\n"
                f"           i * {cell_byte_length}, element);\n"
                "}\n"
                "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


STACK_CONST_S16_I32 = _repeated_constant(cell_byte_length=4, opcode=0xB2)
STACK_CONST_S16_I64 = _repeated_constant(cell_byte_length=8, opcode=0xB3)


def _fixed_pack(
    *,
    opcode: int,
    destination_bit_width: int,
    immediate_bit_width: int,
    lane_count: int,
) -> Instruction:
    destination_byte_length = destination_bit_width // 8
    immediate_byte_length = immediate_bit_width // 8
    result_byte_length = destination_byte_length * lane_count
    record_byte_length = 4 + immediate_byte_length * lane_count
    mnemonic = (
        f"stack.pack.i{destination_bit_width}.u{immediate_bit_width}.x{lane_count}"
    )
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=(
            f"Writes {lane_count} zero-extended u{immediate_bit_width} "
            f"immediates as i{destination_bit_width} cells."
        ),
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=record_byte_length,
        family_id=FAMILY.entity_id,
        fields=(
            zero_padding("zero_padding_u8", 1, 1),
            _field(
                "target_u16",
                2,
                U16.entity_id,
                InstructionFieldRole.IMMEDIATE,
                f"{destination_byte_length}-byte-aligned local target base.",
                LOCAL_BYTES_FIXED_BASE.entity_id,
                (result_byte_length, destination_byte_length),
            ),
            _field(
                "immediates_le",
                4,
                (U16 if immediate_bit_width == 16 else U32).entity_id,
                InstructionFieldRole.IMMEDIATE,
                f"Exact {lane_count}-element little-endian immediate payload.",
                ANY_BITS.entity_id,
                array_length=lane_count,
            ),
        ),
        semantics=_semantics(
            description=(
                f"Zero-extends each of {lane_count} encoded u{immediate_bit_width} "
                f"immediates to i{destination_bit_width} and writes consecutive "
                "little-endian destination cells."
            ),
            verification=(
                f"target_u16 must be {destination_byte_length}-byte aligned and "
                f"the fixed {result_byte_length}-byte target must fit local bytes.",
                "zero_padding_u8 must equal zero; every immediate bit pattern is valid.",
            ),
            success=(
                f"Exactly {lane_count} destination cells receive the "
                "zero-extended immediates in encoded order.",
            ),
            assembly=(
                f"{mnemonic} #v<target>, [u{immediate_bit_width}_0, "
                f"..., u{immediate_bit_width}_{lane_count - 1}]",
            ),
            pseudocode=(
                f"for (i = 0; i < {lane_count}; ++i) {{\n"
                f"  immediate = load_le({immediate_bit_width},\n"
                f"      &record[4 + i * {immediate_byte_length}]);\n"
                f"  store_le({destination_bit_width},\n"
                f"      local_bytes + target_u16 + i * {destination_byte_length},\n"
                "      immediate);\n"
                "}\n"
                f"pc = pc + {record_byte_length};"
            ),
            byte_length=record_byte_length,
        ),
    )


STACK_PACK_I32_U16_X2 = _fixed_pack(
    opcode=0xB4,
    destination_bit_width=32,
    immediate_bit_width=16,
    lane_count=2,
)
STACK_PACK_I32_U16_X4 = _fixed_pack(
    opcode=0xB5,
    destination_bit_width=32,
    immediate_bit_width=16,
    lane_count=4,
)
STACK_PACK_I32_U16_X8 = _fixed_pack(
    opcode=0xB6,
    destination_bit_width=32,
    immediate_bit_width=16,
    lane_count=8,
)
STACK_PACK_I64_U32_X2 = _fixed_pack(
    opcode=0xB7,
    destination_bit_width=64,
    immediate_bit_width=32,
    lane_count=2,
)
STACK_PACK_I64_U32_X4 = _fixed_pack(
    opcode=0xB8,
    destination_bit_width=64,
    immediate_bit_width=32,
    lane_count=4,
)
STACK_PACK_I64_U32_X8 = _fixed_pack(
    opcode=0xB9,
    destination_bit_width=64,
    immediate_bit_width=32,
    lane_count=8,
)

INSTRUCTIONS = (
    STACK_LOAD,
    STACK_STORE,
    STACK_LOAD_INDEXED,
    STACK_STORE_INDEXED,
    STACK_FILL,
    STACK_COPY,
    STACK_COMPARE,
    STACK_COPY_RODATA,
    STACK_COPY_FROM_BUFFER,
    STACK_COPY_TO_BUFFER,
    STACK_CONST_S16_I32,
    STACK_CONST_S16_I64,
    STACK_PACK_I32_U16_X2,
    STACK_PACK_I32_U16_X4,
    STACK_PACK_I32_U16_X8,
    STACK_PACK_I64_U32_X2,
    STACK_PACK_I64_U32_X4,
    STACK_PACK_I64_U32_X8,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
