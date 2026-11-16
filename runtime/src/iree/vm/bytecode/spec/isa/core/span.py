# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checked access to synchronous borrowed argument storage."""

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
from iree.vm.bytecode.spec.isa.core.stack import MEMORY_FORMAT_SELECTOR
from iree.vm.bytecode.spec.schema import U8, Field
from iree.vm.bytecode.spec.version import CORE_0

SPAN_FAMILY = InstructionFamily(
    name="span",
    since=CORE_0,
    summary="Checked access to borrowed incoming arrays.",
    contract=(
        "A coordinate contains a nonzero 16-bit frame-local region ordinal and "
        "a 48-bit byte offset. Incoming signature pairs establish private native "
        "storage descriptors. The upper half of region ordinals identifies "
        "managed buffers pinned for the next child call. Pointer bits never "
        "enter guest registers. Each "
        "access checks region identity, shallow permissions, storage domain, and "
        "the complete extent before modifying state. Scalar bytes cannot be "
        "reinterpreted as reference cells. Reference coordinates use a fixed "
        "16-byte stride and resolve to native cells independently of host bitness. "
        "Borrowed spans cannot be returned, "
        "captured, or passed through a yielding call."
    ),
)


def _instruction(opcode: int, *, load: bool, references: bool) -> Instruction:
    direction = "load" if load else "store"
    mnemonic = "span." + ("ref." if references else "") + direction
    value_name = ("destination_" if load else "source_") + (
        "r8" if references else "v8"
    )
    value = InstructionField(
        Field(value_name, U8, "Reference cell or first scalar lane register."),
        FieldRole.RESULT if load else FieldRole.OPERAND,
        FieldRuleUse(
            FieldRule.REGISTER_REF if references else FieldRule.REGISTER_VALUE
        ),
        RuntimeRefPolicy(
            "dynamic",
            RefNullPolicy.NULLABLE,
            RefOwnership.REPLACE_COPY if load else RefOwnership.RETAIN,
        )
        if references
        else None,
    )
    address = InstructionField(
        Field("address_v8", U8, "Frame-relative region and byte coordinate."),
        FieldRole.OPERAND,
        FieldRuleUse(FieldRule.REGISTER_VALUE),
    )
    format_field = InstructionField(
        Field(
            "zero_padding_u8" if references else "format_u8",
            U8,
            "Canonical zero padding."
            if references
            else "Lane element width and count.",
        ),
        FieldRole.PADDING if references else FieldRole.IMMEDIATE,
        FieldRuleUse(FieldRule.ZERO)
        if references
        else FieldRuleUse(FieldRule.SELECTOR, data=MEMORY_FORMAT_SELECTOR),
    )
    extent = "16" if references else "format.byte_length"
    permission = "READ" if load else "WRITE"
    transfer = (
        "source = *cell;\n"
        "if (region.element == PTR && !(region.access & WRITE) || "
        "is_borrowed_only(source))\n"
        "  replace_borrow(refs[destination_r8], source);\n"
        "else\n"
        "  replace_retain(refs[destination_r8], source);"
        if load and references
        else "source = refs[source_r8];\n"
        "require(source == null || type(source) == region.ref_type);\n"
        "require(!is_borrowed_only(source));\n"
        "replace_retain(*cell, source);"
        if references
        else "for (lane = 0; lane < format.lane_count; ++lane)\n"
        "  values[destination_v8 + lane] = "
        "load_unsigned_le(data + lane * format.element_bytes, format.element_bytes);"
        if load
        else "for (lane = 0; lane < format.lane_count; ++lane)\n"
        "  store_le(data + lane * format.element_bytes, format.element_bytes, "
        "values[source_v8 + lane]);"
    )
    pseudocode = (
        ("" if references else "format = decode_memory_format(format_u8);\n")
        + "ordinal = values[address_v8] >> 48;\n"
        "offset = values[address_v8] & ((1 << 48) - 1);\n"
        "if (ordinal >= 0x8000) {\n"
        "  require(ordinal - 0x8000 < frame.span_buffer_count);\n"
        "  region = checked_buffer_region(frame.span_buffers[ordinal - 0x8000]);\n"
        "} else {\n"
        "  require(ordinal > 0 && ordinal <= frame.region_count);\n"
        "  region = frame.regions[ordinal - 1];\n"
        "}\n"
        f"require(region.domain == {'REFERENCES' if references else 'SCALARS'});\n"
        f"require(region.access & {permission});\n"
        f"require(offset <= region.byte_length && {extent} <= region.byte_length - offset);\n"
        + (
            "require(offset % 16 == 0);\ncell = &native_ref_cells(region.data)[offset / 16];\n"
            if references
            else "data = region.data + offset;\n"
        )
        + transfer
        + "\npc = pc + 4;"
    )
    register = "%r1" if references else "%v4.x2"
    spelling = mnemonic if references else mnemonic + ".i32"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=SPAN_FAMILY,
        summary=f"{direction.capitalize()}s borrowed {'reference cells' if references else 'scalar lanes'}.",
        fields=(value, address, format_field),
        semantics=None,
        behavior=(
            "Resolves the complete range before touching registers or storage. "
            "Scalar lanes use the same unaligned little-endian formats as local "
            "storage. Reference loads borrow read-only ptr elements and borrowed-only "
            "objects; other loads acquire an owner so later array mutation cannot "
            "invalidate the value. Reference stores validate the exact element type "
            "and acquire an owner before replacing the old cell. Borrowed-only "
            "objects cannot be captured by a store."
        ),
        success=(
            "The selected lanes or cell are transferred without changing other storage.",
        ),
        assembly=(
            f"{register} = {spelling} %v0" if load else f"{spelling} {register}, %v0"
        ),
        pseudocode=pseudocode,
        rules=()
        if references
        else (
            RecordRule(
                RecordRuleKind.VALUE_REGISTER_FORMAT_RANGE,
                fields=(value_name, "format_u8"),
                values=(8,),
                summary="The complete selected lane register run is in range.",
            ),
        ),
        state_effects=(
            StateEffect(
                StateAccess.READ if load else StateAccess.WRITE,
                StateResource.BORROWED_STORAGE,
                ("address_v8",),
            ),
        ),
        failures=(
            FailureCase(
                "out_of_range",
                "Region or complete byte extent is out of range.",
                "State is unchanged.",
            ),
            FailureCase(
                "permission_denied",
                "Access exceeds the signature's shallow bound.",
                "State is unchanged.",
            ),
            FailureCase(
                "invalid_argument",
                "Storage domain or stored reference type does not match.",
                "State is unchanged.",
            ),
        )
        + (
            ()
            if load or not references
            else (
                FailureCase(
                    "failed_precondition",
                    "A borrowed-only object cannot acquire an owner.",
                    "State is unchanged.",
                ),
            )
        ),
    )


SPAN_LOAD = _instruction(0xBA, load=True, references=False)
SPAN_STORE = _instruction(0xBB, load=False, references=False)
SPAN_REF_LOAD = _instruction(0xBC, load=True, references=True)
SPAN_REF_STORE = _instruction(0xBD, load=False, references=True)
SPAN_FROM_BUFFER = Instruction(
    opcode=0xBE,
    mnemonic="span.from.buffer",
    since=CORE_0,
    family=SPAN_FAMILY,
    summary="Pins a managed buffer for a synchronous child call.",
    fields=(
        InstructionField(
            Field("destination_v8", U8, "Call-scoped scalar storage coordinate."),
            FieldRole.RESULT,
            FieldRuleUse(FieldRule.REGISTER_VALUE),
        ),
        InstructionField(
            Field("buffer_r8", U8, "Exact vm.buffer retained for the child call."),
            FieldRole.OPERAND,
            FieldRuleUse(FieldRule.REGISTER_REF),
            RuntimeRefPolicy("vm.buffer", RefNullPolicy.REQUIRED, RefOwnership.RETAIN),
        ),
        InstructionField(
            Field("zero_padding_u8", U8, "Canonical zero padding."),
            FieldRole.PADDING,
            FieldRuleUse(FieldRule.ZERO),
        ),
    ),
    semantics=None,
    behavior=(
        "Acquires a private owner for a managed buffer and returns a coordinate "
        "in the upper half of the frame's region namespace. The coordinate remains "
        "valid until the next child call completes or this frame exits. Each access "
        "checks the buffer's liveness, permissions, and byte extent. Buffer bytes "
        "cannot contain reference cells. Pending owners are bounded by half the "
        "function's outgoing call value scratch count."
    ),
    success=("The buffer is pinned without copying its contents.",),
    assembly="%v0 = span.from.buffer %r0",
    pseudocode=(
        "buffer = require_exact_nonnull_buffer(refs[buffer_r8]);\n"
        "require(frame.span_buffer_count < frame.span_buffer_capacity);\n"
        "index = frame.span_buffer_count;\n"
        "frame.span_buffers[index] = retain(buffer);\n"
        "frame.span_buffer_count = index + 1;\n"
        "values[destination_v8] = (0x8000 + index) << 48;\n"
        "pc = pc + 4;"
    ),
    state_effects=(StateEffect(StateAccess.WRITE, StateResource.BORROWED_STORAGE),),
    failures=(
        FailureCase(
            "invalid_argument",
            "The ref is null or not a buffer.",
            "State is unchanged.",
        ),
        FailureCase(
            "resource_exhausted",
            "Pending buffer capacity is exhausted.",
            "State is unchanged.",
        ),
    ),
)
SPAN_INSTRUCTIONS = (
    SPAN_LOAD,
    SPAN_STORE,
    SPAN_REF_LOAD,
    SPAN_REF_STORE,
    SPAN_FROM_BUFFER,
)
