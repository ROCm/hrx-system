# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checked access to frame-local byte storage."""

from __future__ import annotations

import enum
from typing import NamedTuple

from iree.vm.bytecode.spec.isa import (
    FailureCase,
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
    RecordRule,
    RuntimeRefPolicy,
    StateEffect,
)
from iree.vm.bytecode.spec.isa.core.rules import (
    FieldRule,
    RecordRuleKind,
    RefNullPolicy,
    RefOwnership,
    StateAccess,
    StateResource,
)
from iree.vm.bytecode.spec.schema import (
    I16,
    U8,
    U16,
    U32,
    Field,
    NumericKind,
    NumericTable,
    NumericValue,
)
from iree.vm.bytecode.spec.version import CORE_0


def _memory_format_values() -> tuple[NumericValue, ...]:
    values = []
    for element_log2, element_bits in enumerate((8, 16, 32, 64)):
        for lane_log2, lane_count in enumerate((1, 2, 4, 8)):
            values.append(
                NumericValue(
                    f"i{element_bits}.x{lane_count}",
                    element_log2 * 4 + lane_log2,
                    CORE_0,
                    f"Transfers {lane_count} {element_bits}-bit "
                    f"lane{'s' if lane_count != 1 else ''} spanning "
                    f"{(element_bits // 8) * lane_count} bytes.",
                )
            )
    return tuple(values)


MEMORY_FORMAT_SELECTOR = NumericTable(
    "memory.format",
    U8,
    NumericKind.SELECTOR,
    _memory_format_values(),
    CORE_0,
    (
        "Selects the integer lane width and lane count transferred between one "
        "value-register run and consecutive little-endian bytes. The value is "
        "element_log2 * 4 + lane_log2 for logarithms in 0..3; bits 7:4 are zero."
    ),
)

MEMORY_FORMAT_MAXIMUM_LANE_COUNT = 8

STACK_FAMILY = InstructionFamily(
    name="stack",
    since=CORE_0,
    summary="Checked access to one frame-local byte array.",
    contract=(
        "Each frame owns one fixed local byte array of 0 to 65,535 bytes whose "
        "entry contents are unspecified. A stack base is an unsigned byte offset, "
        "not a pointer or value-register ordinal. Verification evaluates every "
        "static range in a mathematical type that cannot wrap. A zero-length range "
        "accepts a base equal to the array length, and execution forms no pointer "
        "for it. Lane zero uses the lowest addressed bytes and lowest value register. "
        "Lane accesses are little-endian and alignment-independent: loads zero-extend "
        "into complete value cells and stores use only the selected low bits. Fixed "
        "and repeated constant writers carry explicit natural-alignment requirements. "
        "No stack instruction allocates, grows, exposes, or retains the local array, "
        "changes ref ownership, or suspends."
    ),
)


def _field(
    name: str,
    encoding,
    summary: str,
    role: FieldRole,
    rule,
    *,
    element_count: int = 1,
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, encoding, summary, element_count), role, rule)


def _value(name: str, role: FieldRole, summary: str) -> InstructionField:
    return _field(name, U8, summary, role, FieldRule.REGISTER_VALUE)


def _padding(name: str, encoding=U8, *, element_count: int = 1) -> InstructionField:
    return _field(
        name,
        encoding,
        "Canonical zero padding.",
        FieldRole.PADDING,
        FieldRule.ZERO,
        element_count=element_count,
    )


def _memory_format() -> InstructionField:
    return _field(
        "format_u8",
        U8,
        "Closed lane element-width and lane-count selector.",
        FieldRole.IMMEDIATE,
        FieldRuleUse(FieldRule.SELECTOR, data=MEMORY_FORMAT_SELECTOR),
    )


def _memory_base(name: str = "base_u16") -> InstructionField:
    return _field(
        name,
        U16,
        "Static base of an alignment-independent local-byte lane range.",
        FieldRole.IMMEDIATE,
        FieldRuleUse(
            FieldRule.LOCAL_BYTES_RANGE_MEMORY_FORMAT,
            fields=("format_u8",),
        ),
    )


def _range_base(name: str, length_name: str, summary: str) -> InstructionField:
    return _field(
        name,
        U16,
        summary,
        FieldRole.IMMEDIATE,
        FieldRuleUse(FieldRule.LOCAL_BYTES_RANGE_BASE, fields=(length_name,)),
    )


def _range_length(name: str = "length_u16") -> InstructionField:
    return _field(
        name,
        U16,
        "Exact byte length of the statically checked local range.",
        FieldRole.IMMEDIATE,
        FieldRule.LOCAL_BYTES_RANGE_LENGTH,
    )


def _value_format_range(register_name: str) -> RecordRule:
    return RecordRule(
        RecordRuleKind.VALUE_REGISTER_FORMAT_RANGE,
        fields=(register_name, "format_u8"),
        values=(MEMORY_FORMAT_MAXIMUM_LANE_COUNT,),
        summary=(
            f"{register_name} and format_u8 must select a complete in-range "
            "consecutive value-register run."
        ),
    )


class _LaneDirection(enum.Enum):
    LOAD = "load"
    STORE = "store"


class _LaneAddressing(enum.Enum):
    STATIC = "static"
    INDEXED = "indexed"


class _LaneTransferDefinition(NamedTuple):
    opcode: int
    direction: _LaneDirection
    addressing: _LaneAddressing


def _lane_transfer(definition: _LaneTransferDefinition) -> Instruction:
    opcode, direction, addressing = definition
    is_load = direction == _LaneDirection.LOAD
    is_indexed = addressing == _LaneAddressing.INDEXED
    mnemonic = f"stack.{direction.value}"
    if is_indexed:
        mnemonic += ".indexed"
    register_name = "destination_v8" if is_load else "source_v8"
    register = _value(
        register_name,
        FieldRole.RESULT if is_load else FieldRole.OPERAND,
        "First lane value-register ordinal.",
    )
    index_fields = (
        _value(
            "index_v8",
            FieldRole.OPERAND,
            "Unsigned 64-bit dynamic lane-group index.",
        ),
        _field(
            "scale_u8",
            U8,
            "Nonzero byte scale applied to index_v8.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(1, 255)),
        ),
    )
    if is_load:
        fields = (
            (
                register,
                _memory_base(),
                *index_fields,
                _memory_format(),
                _padding("zero_padding_u8"),
            )
            if is_indexed
            else (
                register,
                _memory_base(),
                _memory_format(),
                _padding("zero_padding_u8", element_count=3),
            )
        )
    else:
        fields = (
            (
                _padding("zero_padding_u8"),
                _memory_base(),
                *index_fields,
                register,
                _memory_format(),
            )
            if is_indexed
            else (
                _padding("zero_padding_u8"),
                _memory_base(),
                register,
                _memory_format(),
                _padding("zero_padding_u16", U16),
            )
        )

    operation = "Loads" if is_load else "Stores"
    direction_phrase = "from" if is_load else "to"
    address = "base_u16"
    preconditions = ()
    failures = ()
    if is_indexed:
        address = "base_u16 + unsigned(index_v8) * scale_u8"
        preconditions = (
            "index_v8 must not exceed UINT16_MAX and its scaled relative offset "
            "must fit after base_u16 and the selected access length.",
        )
        failures = (
            FailureCase(
                "out_of_range",
                "index_v8 exceeds UINT16_MAX or the scaled access is out of range.",
                (
                    "All destination registers and local bytes remain unchanged."
                    if is_load
                    else "All local bytes and VM registers remain unchanged."
                ),
            ),
        )

    if is_load:
        assembly = (
            "%v<destination>.xN = stack.load.indexed "
            "#v<base>, %v<index> {scale, format}"
            if is_indexed
            else "%v<destination>.xN = stack.load #v<base> {format}"
        )
        transfer = (
            "for (i = 0; i < format.lane_count; ++i) {\n"
            "  values[destination_v8 + i] = load_le(\n"
            "      format.element_bits, local_bytes + effective_base +\n"
            "      i * format.element_bytes);\n"
            "}"
        )
        success = (
            "Each destination receives one little-endian lane zero-extended to a "
            "complete 64-bit value cell, in increasing lane order.",
        )
    else:
        assembly = (
            "stack.store.indexed #v<base>, %v<index>, %v<source>.xN {scale, format}"
            if is_indexed
            else "stack.store #v<base>, %v<source>.xN {format}"
        )
        transfer = (
            "for (i = 0; i < format.lane_count; ++i) {\n"
            "  store_le(format.element_bits, local_bytes + effective_base +\n"
            "           i * format.element_bytes, values[source_v8 + i]);\n"
            "}"
        )
        success = (
            "Each lane receives the selected low source bits in little-endian "
            "order, in increasing lane order.",
        )

    address_code = "effective_base = base_u16;"
    if is_indexed:
        address_code = (
            "available_u32 = local_byte_length - format.access_length - base_u16;\n"
            "index_u64 = values[index_v8];\n"
            "if (index_u64 > UINT16_MAX) fail(out_of_range);\n"
            "relative_u32 = u32(index_u64) * u32(scale_u8);\n"
            "if (relative_u32 > available_u32) fail(out_of_range);\n"
            "effective_base = u16(base_u16 + relative_u32);"
        )
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=STACK_FAMILY,
        summary=(
            f"{operation} one {'dynamically indexed ' if is_indexed else ''}"
            "local-byte lane group."
        ),
        fields=fields,
        semantics=None,
        behavior=(
            f"{operation} one format-selected lane group {direction_phrase} "
            f"effective_base = {address}. All address checks finish before "
            "destination state changes."
        ),
        success=success,
        assembly=assembly,
        pseudocode=(
            "format = verified_memory_format(format_u8);\n"
            f"{address_code}\n{transfer}\npc = pc + 8;"
        ),
        rules=(_value_format_range(register_name),),
        state_effects=(
            StateEffect(
                StateAccess.READ if is_load else StateAccess.WRITE,
                StateResource.FRAME_LOCALS,
                ("base_u16",),
            ),
        ),
        preconditions=preconditions,
        failures=failures,
    )


STACK_LOAD, STACK_STORE, STACK_LOAD_INDEXED, STACK_STORE_INDEXED = tuple(
    _lane_transfer(definition)
    for definition in (
        _LaneTransferDefinition(0xA8, _LaneDirection.LOAD, _LaneAddressing.STATIC),
        _LaneTransferDefinition(0xA9, _LaneDirection.STORE, _LaneAddressing.STATIC),
        _LaneTransferDefinition(0xAA, _LaneDirection.LOAD, _LaneAddressing.INDEXED),
        _LaneTransferDefinition(0xAB, _LaneDirection.STORE, _LaneAddressing.INDEXED),
    )
)

STACK_FILL = Instruction(
    opcode=0xAC,
    mnemonic="stack.fill",
    since=CORE_0,
    family=STACK_FAMILY,
    summary="Repeats a low-byte pattern across a local-byte range.",
    fields=(
        _padding("zero_padding_u8"),
        _range_base("target_base_u16", "length_u16", "Static local-byte target base."),
        _range_length(),
        _value("pattern_v8", FieldRole.OPERAND, "Low-byte pattern source."),
        _field(
            "pattern_width_u8",
            U8,
            "Pattern width in bytes.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.ALLOWED_VALUES, values=(1, 2, 4, 8)),
        ),
    ),
    semantics=None,
    behavior=(
        "Repeats the low one, two, four, or eight little-endian bytes of "
        "pattern_v8 across the exact target range; the final repetition may be partial."
    ),
    success=(
        "Exactly length_u16 bytes receive the repeated pattern. A zero length "
        "writes nothing and forms no local pointer.",
    ),
    assembly="stack.fill #v<target>.x<length>, %v<pattern> {pattern_width}",
    pseudocode=(
        "pattern_bits = values[pattern_v8];\n"
        "for (i = 0; i < length_u16; ++i) {\n"
        "  local_bytes[target_base_u16 + i] =\n"
        "      u8(pattern_bits >> (8 * (i % pattern_width_u8)));\n"
        "}\n"
        "pc = pc + 8;"
    ),
    state_effects=(
        StateEffect(
            StateAccess.WRITE, StateResource.FRAME_LOCALS, ("target_base_u16",)
        ),
    ),
)

STACK_COPY = Instruction(
    opcode=0xAD,
    mnemonic="stack.copy",
    since=CORE_0,
    family=STACK_FAMILY,
    summary="Moves one possibly overlapping local-byte range.",
    fields=(
        _padding("zero_padding_u8"),
        _range_base("target_u16", "length_u16", "Static local-byte target base."),
        _range_base("source_u16", "length_u16", "Static local-byte source base."),
        _range_length(),
    ),
    semantics=None,
    behavior=(
        "Moves one local-byte range to another with memmove semantics, including "
        "every overlap and equal-base case."
    ),
    success=(
        "The target receives the source bytes as if through a temporary copy. A "
        "zero length forms no pointer and performs no host call.",
    ),
    assembly="stack.copy #v<target>.x<length>, #v<source>.x<length>",
    pseudocode=(
        "if (length_u16 != 0) {\n"
        "  move_bytes(local_bytes + target_u16,\n"
        "             local_bytes + source_u16, length_u16);\n"
        "}\n"
        "pc = pc + 8;"
    ),
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.FRAME_LOCALS, ("source_u16",)),
        StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("target_u16",)),
    ),
)

STACK_COMPARE = Instruction(
    opcode=0xAE,
    mnemonic="stack.compare",
    since=CORE_0,
    family=STACK_FAMILY,
    summary="Lexicographically compares two local-byte ranges.",
    fields=(
        _value(
            "destination_v8",
            FieldRole.RESULT,
            "Canonical signed i32 comparison result.",
        ),
        _range_base("left_u16", "length_u16", "Static left local-byte base."),
        _range_base("right_u16", "length_u16", "Static right local-byte base."),
        _range_length(),
    ),
    semantics=None,
    behavior=(
        "Lexicographically compares equal-length local ranges as unsigned bytes "
        "and returns canonical i32 -1, 0, or +1."
    ),
    success=(
        "destination_v8 receives UINT32_MAX for less, zero for equal, or one for "
        "greater, with its high 32 cell bits zero. Empty ranges compare equal.",
    ),
    assembly=(
        "%v<destination> = stack.compare #v<left>.x<length>, #v<right>.x<length>"
    ),
    pseudocode=(
        "ordering = length_u16 == 0 ? 0 : compare_unsigned_bytes(\n"
        "    local_bytes + left_u16, local_bytes + right_u16, length_u16);\n"
        "values[destination_v8] = ordering < 0 ? UINT32_MAX :\n"
        "                         ordering > 0 ? 1 : 0;\n"
        "pc = pc + 8;"
    ),
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.FRAME_LOCALS, ("left_u16",)),
        StateEffect(StateAccess.READ, StateResource.FRAME_LOCALS, ("right_u16",)),
    ),
)

STACK_COPY_RODATA = Instruction(
    opcode=0xAF,
    mnemonic="stack.copy.rodata",
    since=CORE_0,
    family=STACK_FAMILY,
    summary="Copies immutable module bytes into local bytes.",
    fields=(
        _padding("zero_padding_u8"),
        _range_base("target_u16", "length_u16", "Static local-byte target base."),
        _field(
            "rodata_u16",
            U16,
            "Direct module rodata-block ordinal.",
            FieldRole.IMMEDIATE,
            FieldRule.RODATA_ORDINAL,
        ),
        _range_length(),
        _field(
            "source_offset_u32",
            U32,
            "Static source offset in the selected rodata block.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(
                FieldRule.RODATA_STATIC_OFFSET,
                fields=("rodata_u16", "length_u16"),
            ),
        ),
    ),
    semantics=None,
    behavior=(
        "Copies a statically selected immutable module-rodata range into the "
        "statically selected local-byte target without creating a buffer."
    ),
    success=(
        "The target receives exactly length_u16 source bytes. A zero length "
        "accepts both one-past-end bases and forms neither pointer.",
    ),
    assembly=(
        "stack.copy.rodata #v<target>.x<length>, @rodata<ordinal>+<source_offset>"
    ),
    pseudocode=(
        "source = verified_rodata_blocks[rodata_u16];\n"
        "if (length_u16 != 0) {\n"
        "  copy_bytes(local_bytes + target_u16,\n"
        "             source.data + source_offset_u32, length_u16);\n"
        "}\n"
        "pc = pc + 12;"
    ),
    state_effects=(
        StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("target_u16",)),
    ),
)


def _buffer_ref() -> InstructionField:
    return InstructionField(
        Field(
            "buffer_r8",
            U8,
            "Required exact vm.buffer ref borrowed for this instruction.",
        ),
        FieldRole.OPERAND,
        FieldRuleUse(FieldRule.REGISTER_REF),
        RuntimeRefPolicy(
            "vm.buffer",
            RefNullPolicy.REQUIRED,
            RefOwnership.BORROW,
        ),
    )


class _BufferTransfer(enum.Enum):
    FROM_BUFFER = "from.buffer"
    TO_BUFFER = "to.buffer"


def _buffer_transfer(direction: _BufferTransfer) -> Instruction:
    from_buffer = direction == _BufferTransfer.FROM_BUFFER
    mnemonic = f"stack.copy.{direction.value}"
    if from_buffer:
        fields = (
            _padding("zero_padding_u8"),
            _range_base("target_u16", "length_u16", "Static local-byte target base."),
            _buffer_ref(),
            _value(
                "source_offset_v8",
                FieldRole.OPERAND,
                "Unsigned dynamic source offset in buffer_r8.",
            ),
            _range_length(),
        )
        access = "READ"
        offset_name = "source_offset_v8"
        unchanged = "The local-byte target remains unchanged."
        copy = (
            "copy_bytes(local_bytes + target_u16,\n"
            "             buffer.data + dynamic_offset_u64, length_u16);"
        )
        effects = (
            StateEffect(StateAccess.READ, StateResource.BUFFER, ("buffer_r8",)),
            StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("target_u16",)),
        )
        assembly = (
            "stack.copy.from.buffer #v<target>.x<length>, %r<buffer>, %v<source_offset>"
        )
    else:
        fields = (
            _buffer_ref(),
            _value(
                "target_offset_v8",
                FieldRole.OPERAND,
                "Unsigned dynamic target offset in buffer_r8.",
            ),
            _padding("zero_padding_u8"),
            _range_base("source_u16", "length_u16", "Static local-byte source base."),
            _range_length(),
        )
        access = "WRITE"
        offset_name = "target_offset_v8"
        unchanged = "The buffer target remains unchanged."
        copy = (
            "copy_bytes(buffer.data + dynamic_offset_u64,\n"
            "             local_bytes + source_u16, length_u16);"
        )
        effects = (
            StateEffect(StateAccess.READ, StateResource.FRAME_LOCALS, ("source_u16",)),
            StateEffect(StateAccess.WRITE, StateResource.BUFFER, ("buffer_r8",)),
        )
        assembly = (
            "stack.copy.to.buffer %r<buffer>, %v<target_offset>, #v<source>.x<length>"
        )
    return Instruction(
        opcode=0xB0 if from_buffer else 0xB1,
        mnemonic=mnemonic,
        since=CORE_0,
        family=STACK_FAMILY,
        summary=(
            "Copies a checked vm.buffer range into local bytes."
            if from_buffer
            else "Copies local bytes into a checked vm.buffer range."
        ),
        fields=fields,
        semantics=None,
        behavior=(
            "Copies between a statically verified local range and a dynamically "
            f"offset exact vm.buffer with {access} access. The dynamic checks also "
            "apply to a zero-length range."
        ),
        success=(
            "After every check succeeds, exactly length_u16 bytes are copied. The "
            "local frame and vm.buffer cannot alias; zero length forms no pointer.",
        ),
        assembly=assembly,
        pseudocode=(
            "buffer = check_deref(refs[buffer_r8], vm_buffer_type);\n"
            f"require_access(buffer, {access});\n"
            f"dynamic_offset_u64 = values[{offset_name}];\n"
            "require checked_range(dynamic_offset_u64, length_u16, buffer.length);\n"
            "if (length_u16 != 0) {\n"
            f"  {copy}\n"
            "}\n"
            "pc = pc + 8;"
        ),
        state_effects=effects,
        preconditions=(
            f"buffer_r8 holds a non-null exact vm.buffer granting {access} access, "
            "and the dynamic range fits its byte length.",
        ),
        failures=(
            FailureCase("failed_precondition", "buffer_r8 is null.", unchanged),
            FailureCase(
                "invalid_argument",
                "buffer_r8 contains a non-null ref of another type.",
                unchanged,
            ),
            FailureCase(
                "permission_denied",
                f"The buffer does not grant {access} access.",
                unchanged,
            ),
            FailureCase(
                "out_of_range",
                "The dynamic buffer range is outside buffer.length.",
                unchanged,
            ),
        ),
        ownership=(
            "buffer_r8 is borrowed only for synchronous validation and copy; its "
            "ref count and register state do not change.",
        ),
    )


STACK_COPY_FROM_BUFFER = _buffer_transfer(_BufferTransfer.FROM_BUFFER)
STACK_COPY_TO_BUFFER = _buffer_transfer(_BufferTransfer.TO_BUFFER)


def _repeated_constant(cell_byte_length: int, opcode: int) -> Instruction:
    cell_name = f"i{cell_byte_length * 8}"
    mnemonic = f"stack.const.s16.{cell_name}"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=STACK_FAMILY,
        summary=f"Fills aligned {cell_name} cells from one signed immediate.",
        fields=(
            _padding("zero_padding_u8"),
            _field(
                "target_u16",
                U16,
                f"{cell_byte_length}-byte-aligned local target base.",
                FieldRole.IMMEDIATE,
                FieldRuleUse(
                    FieldRule.LOCAL_BYTES_REPEATED_BASE,
                    fields=("count_u16",),
                    values=(cell_byte_length, cell_byte_length),
                ),
            ),
            _field(
                "count_u16",
                U16,
                f"Number of {cell_name} cells to write.",
                FieldRole.IMMEDIATE,
                FieldRule.LOCAL_BYTES_REPEATED_COUNT,
            ),
            _field(
                "immediate_i16",
                I16,
                "Signed little-endian two's-complement fill immediate.",
                FieldRole.IMMEDIATE,
                FieldRule.ANY_BITS,
            ),
        ),
        semantics=None,
        behavior=(
            f"Sign-extends immediate_i16 to {cell_name} and writes count_u16 "
            f"consecutive little-endian {cell_name} cells."
        ),
        success=(
            "Every selected cell receives the same sign-extended immediate. A zero "
            "count writes nothing while retaining the base-alignment check.",
        ),
        assembly=f"{mnemonic} #v<target>.x<count>, <immediate>",
        pseudocode=(
            f"element = sext_u{cell_byte_length * 8}(immediate_i16, 16);\n"
            "for (i = 0; i < count_u16; ++i) {\n"
            f"  store_le({cell_byte_length * 8}, local_bytes + target_u16 +\n"
            f"           i * {cell_byte_length}, element);\n"
            "}\n"
            "pc = pc + 8;"
        ),
        state_effects=(
            StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("target_u16",)),
        ),
    )


STACK_CONST_S16_I32 = _repeated_constant(4, 0xB2)
STACK_CONST_S16_I64 = _repeated_constant(8, 0xB3)


def _fixed_pack(
    opcode: int,
    destination_bit_width: int,
    immediate_bit_width: int,
    lane_count: int,
) -> Instruction:
    destination_byte_length = destination_bit_width // 8
    immediate_encoding = U16 if immediate_bit_width == 16 else U32
    target_byte_length = destination_byte_length * lane_count
    mnemonic = (
        f"stack.pack.i{destination_bit_width}.u{immediate_bit_width}.x{lane_count}"
    )
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=STACK_FAMILY,
        summary=(
            f"Writes {lane_count} zero-extended u{immediate_bit_width} immediates "
            f"as i{destination_bit_width} cells."
        ),
        fields=(
            _padding("zero_padding_u8"),
            _field(
                "target_u16",
                U16,
                f"{destination_byte_length}-byte-aligned local target base.",
                FieldRole.IMMEDIATE,
                FieldRuleUse(
                    FieldRule.LOCAL_BYTES_FIXED_BASE,
                    values=(target_byte_length, destination_byte_length),
                ),
            ),
            _field(
                f"immediates_u{immediate_bit_width}",
                immediate_encoding,
                f"Exact {lane_count}-element little-endian immediate payload.",
                FieldRole.IMMEDIATE,
                FieldRule.ANY_BITS,
                element_count=lane_count,
            ),
        ),
        semantics=None,
        behavior=(
            f"Zero-extends each encoded u{immediate_bit_width} immediate to "
            f"i{destination_bit_width} and writes consecutive little-endian cells."
        ),
        success=(
            f"Exactly {lane_count} destination cells receive the zero-extended "
            "immediates in encoded order.",
        ),
        assembly=(
            f"{mnemonic} #v<target>, [u{immediate_bit_width}_0, ..., "
            f"u{immediate_bit_width}_{lane_count - 1}]"
        ),
        pseudocode=(
            f"for (i = 0; i < {lane_count}; ++i) {{\n"
            f"  immediate = load_le({immediate_bit_width},\n"
            f"      &record[4 + i * {immediate_encoding.byte_length}]);\n"
            f"  store_le({destination_bit_width},\n"
            f"      local_bytes + target_u16 + i * {destination_byte_length},\n"
            "      immediate);\n"
            "}\n"
            f"pc = pc + {4 + immediate_encoding.byte_length * lane_count};"
        ),
        state_effects=(
            StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("target_u16",)),
        ),
    )


STACK_PACK_INSTRUCTIONS = tuple(
    _fixed_pack(*definition)
    for definition in (
        (0xB4, 32, 16, 2),
        (0xB5, 32, 16, 4),
        (0xB6, 32, 16, 8),
        (0xB7, 64, 32, 2),
        (0xB8, 64, 32, 4),
        (0xB9, 64, 32, 8),
    )
)

STACK_INSTRUCTIONS = (
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
    *STACK_PACK_INSTRUCTIONS,
)
