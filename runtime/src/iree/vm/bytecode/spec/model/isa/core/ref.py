# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 typed ref-state instructions."""

from __future__ import annotations

from model.isa import (
    Instruction,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
    RefNullPolicy,
    RefOwnership,
    RuntimeRefPolicy,
    StateEffect,
    StateResource,
)
from model.isa.declarations import (
    core_instruction,
    instruction_field,
    ref_register,
    state_read,
    state_write,
    value_register,
    zero_padding,
)
from model.isa.validation import FIELDS_DISTINCT, REF_SLOT, ZERO
from model.schema import U16, FieldReference, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.ref",
    since=CORE_0,
    summary="Generic typed ref-state and local-slot operations.",
    dependencies=("core.contract.machine",),
    document_order=9,
    normative_text=(
        "A ref register or local slot is canonical null, a non-null borrowed "
        "typed object, or a non-null owned typed object carrying one release "
        "obligation. Canonical null has both words zero. The actual descriptor "
        "moves with the object; generic ref operations never cast or dereference "
        "it. clear_ref first replaces a slot with canonical null and then "
        "releases the prior object only when it was owned. replace_ref installs "
        "an already-safe new state before releasing an old owner. retain_ref "
        "maps null to null and otherwise retains through the actual descriptor "
        "and creates an owner without querying the current refcount. Retain "
        "operations acquire the new owner before replacement; move operations "
        "snapshot and clear their independent source before replacement. "
        "Moving an internal borrow preserves its dominating lifetime. Frame "
        "unwind releases every owner and clears every borrow in the exact "
        "declared register and local-slot extents. All operations in this family "
        "are infallible after structural verification and never suspend."
    ),
)


def _policy(ownership: RefOwnership) -> RuntimeRefPolicy:
    return RuntimeRefPolicy("dynamic", RefNullPolicy.NULLABLE, ownership)


def _ref(
    name: str,
    offset: int,
    role: InstructionFieldRole,
    ownership: RefOwnership,
    description: str,
):
    return ref_register(
        name,
        offset,
        role,
        description,
        _policy(ownership),
    )


def _zero_u16():
    return instruction_field(
        "zero_padding_u16",
        2,
        U16.entity_id,
        InstructionFieldRole.PADDING,
        "Canonical zero padding.",
        (RuleUse(ZERO.entity_id),),
    )


def _slot():
    return instruction_field(
        "slot_u16",
        2,
        U16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Direct function-local ref-slot ordinal.",
        (RuleUse(REF_SLOT.entity_id),),
    )


def _semantics(
    description: str,
    verification: tuple[str, ...],
    success: tuple[str, ...],
    ownership: tuple[str, ...],
    assembly: tuple[str, ...],
    pseudocode: str,
) -> InstructionSemantics:
    return InstructionSemantics(
        description=description,
        verification=verification,
        preconditions=(),
        success=(*success, "The program counter advances by four bytes."),
        failures=(),
        ownership=ownership,
        assembly=assembly,
        pseudocode=pseudocode,
    )


REF_NULL = core_instruction(
    entity_id="core.instruction.ref.null",
    since=CORE_0,
    summary="Clears a ref register to canonical null.",
    opcode=0xC8,
    mnemonic="ref.null",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _ref(
            "dst_r8",
            1,
            InstructionFieldRole.RESULT,
            RefOwnership.CLEAR,
            "Destination ref-register ordinal to clear.",
        ),
        _zero_u16(),
    ),
    state_effects=(),
    semantics=_semantics(
        "Replaces dst_r8 with canonical null.",
        (
            "dst_r8 must be a valid ref-register ordinal.",
            "zero_padding_u16 must equal zero.",
        ),
        ("dst_r8 becomes canonical null.",),
        (
            "A previous owned destination is released through its actual "
            "descriptor; a previous borrow or null is only cleared.",
        ),
        ("%r<dst> = ref.null",),
        "clear_ref(&refs[dst_r8]);\npc = pc + 4;",
    ),
)

REF_COMPARE_NULL = core_instruction(
    entity_id="core.instruction.ref.compare.null",
    since=CORE_0,
    summary="Tests a ref register for canonical null.",
    opcode=0xC9,
    mnemonic="ref.compare.null",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Destination Boolean value-register ordinal.",
        ),
        _ref(
            "src_r8",
            2,
            InstructionFieldRole.OPERAND,
            RefOwnership.INSPECT,
            "Source ref-register ordinal inspected without dereference.",
        ),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    state_effects=(),
    semantics=_semantics(
        "Tests src_r8 for canonical null without inspecting its object.",
        (
            "dst_v8 and src_r8 must be valid registers in their respective banks.",
            "zero_padding_u8 must equal zero.",
        ),
        (
            "dst_v8 becomes canonical value one when src_r8 is null and "
            "canonical value zero otherwise.",
        ),
        ("src_r8 is neither retained, released, nor dereferenced.",),
        ("%v<dst> = ref.compare.null %r<src>",),
        ("values[dst_v8] = canonical_bool(refs[src_r8].is_null);\npc = pc + 4;"),
    ),
)

REF_COMPARE_EQ = core_instruction(
    entity_id="core.instruction.ref.compare.eq",
    since=CORE_0,
    summary="Tests exact typed object identity of two refs.",
    opcode=0xCA,
    mnemonic="ref.compare.eq",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Destination Boolean value-register ordinal.",
        ),
        _ref(
            "lhs_r8",
            2,
            InstructionFieldRole.OPERAND,
            RefOwnership.INSPECT,
            "Left ref-register ordinal.",
        ),
        _ref(
            "rhs_r8",
            3,
            InstructionFieldRole.OPERAND,
            RefOwnership.INSPECT,
            "Right ref-register ordinal.",
        ),
    ),
    state_effects=(),
    semantics=_semantics(
        (
            "Tests canonical null or exact non-null descriptor-and-object "
            "identity. Ownership state does not affect equality."
        ),
        (
            "dst_v8, lhs_r8, and rhs_r8 must be valid registers in their "
            "respective banks.",
        ),
        (
            "Two null refs compare equal. Null and non-null compare unequal. "
            "Two non-null refs compare equal only when both descriptor and "
            "object pointer are identical; dst_v8 receives canonical zero or one.",
        ),
        (
            "Neither source is retained, released, dereferenced, ordered, "
            "hashed, or exposed as integer bits.",
        ),
        ("%v<dst> = ref.compare.eq %r<lhs>, %r<rhs>",),
        (
            "lhs = refs[lhs_r8];\n"
            "rhs = refs[rhs_r8];\n"
            "equal = lhs.is_null ? rhs.is_null\n"
            "    : !rhs.is_null && lhs.descriptor == rhs.descriptor &&\n"
            "      lhs.object == rhs.object;\n"
            "values[dst_v8] = canonical_bool(equal);\n"
            "pc = pc + 4;"
        ),
    ),
)

REF_RETAIN = core_instruction(
    entity_id="core.instruction.ref.retain",
    since=CORE_0,
    summary="Creates an owned copy of a ref in another register.",
    opcode=0xCB,
    mnemonic="ref.retain",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _ref(
            "dst_r8",
            1,
            InstructionFieldRole.RESULT,
            RefOwnership.REPLACE_RETAIN,
            "Destination ref-register ordinal replaced by the new owner.",
        ),
        _ref(
            "src_r8",
            2,
            InstructionFieldRole.OPERAND,
            RefOwnership.RETAIN,
            "Source ref-register ordinal retained without mutation.",
        ),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    state_effects=(),
    semantics=_semantics(
        (
            "Creates an owned copy of src_r8 and replaces dst_r8 without "
            "changing the source. dst_r8 and src_r8 may alias."
        ),
        (
            "dst_r8 and src_r8 must be valid ref-register ordinals.",
            "zero_padding_u8 must equal zero.",
        ),
        (
            "A null source publishes null; a non-null source publishes one "
            "owned ref carrying its exact object and descriptor.",
        ),
        (
            "The source is snapshotted and retained before the previous "
            "destination owner is released. A borrowed self-retain becomes "
            "owned; an owned self-retain has no net refcount change.",
        ),
        ("%r<dst> = ref.retain %r<src>",),
        (
            "source = refs[src_r8];\n"
            "new_owner = retain_ref(source);\n"
            "replace_ref(&refs[dst_r8], new_owner);\n"
            "pc = pc + 4;"
        ),
    ),
)

REF_MOVE = core_instruction(
    entity_id="core.instruction.ref.move",
    since=CORE_0,
    summary="Destructively transfers one complete ref-register state.",
    opcode=0xCC,
    mnemonic="ref.move",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _ref(
            "dst_r8",
            1,
            InstructionFieldRole.RESULT,
            RefOwnership.REPLACE_MOVE,
            "Destination ref-register ordinal replaced by the moved state.",
        ),
        _ref(
            "src_r8",
            2,
            InstructionFieldRole.OPERAND,
            RefOwnership.MOVE,
            "Source ref-register ordinal cleared by the transfer.",
        ),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    constraints=(
        RuleUse(
            FIELDS_DISTINCT.entity_id,
            (FieldReference("dst_r8"), FieldReference("src_r8")),
        ),
    ),
    state_effects=(),
    semantics=_semantics(
        (
            "Transfers the complete null, borrowed, or owned state from src_r8 "
            "to a distinct dst_r8."
        ),
        (
            "dst_r8 and src_r8 must be valid, distinct ref-register ordinals.",
            "zero_padding_u8 must equal zero.",
        ),
        (
            "src_r8 becomes canonical null and dst_r8 receives the exact prior "
            "source state.",
        ),
        (
            "The source is snapshotted and cleared before replacing dst_r8. "
            "A borrow remains borrowed; an owner transfers exactly one release "
            "obligation; a previous destination owner is released.",
        ),
        ("%r<dst> = ref.move %r<src>",),
        (
            "source = refs[src_r8];\n"
            "refs[src_r8] = canonical_null_ref;\n"
            "replace_ref(&refs[dst_r8], source);\n"
            "pc = pc + 4;"
        ),
    ),
)

REF_DISCARD = core_instruction(
    entity_id="core.instruction.ref.discard",
    since=CORE_0,
    summary="Releases or clears one ref register.",
    opcode=0xCD,
    mnemonic="ref.discard",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _ref(
            "src_r8",
            1,
            InstructionFieldRole.OPERAND,
            RefOwnership.CLEAR,
            "Source ref-register ordinal to discard.",
        ),
        _zero_u16(),
    ),
    state_effects=(),
    semantics=_semantics(
        "Replaces src_r8 with canonical null.",
        (
            "src_r8 must be a valid ref-register ordinal.",
            "zero_padding_u16 must equal zero.",
        ),
        ("src_r8 becomes canonical null; discarding null is a no-op.",),
        (
            "A previous owned state is released through its private no-status "
            "descriptor teardown; a borrow or null is only cleared.",
        ),
        ("ref.discard %r<src>",),
        "clear_ref(&refs[src_r8]);\npc = pc + 4;",
    ),
)


def _stack_transfer(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    register_name: str,
    register_role: InstructionFieldRole,
    register_ownership: RefOwnership,
    register_description: str,
    description: str,
    success: tuple[str, ...],
    ownership: tuple[str, ...],
    state_effects: tuple[StateEffect, ...],
    assembly: str,
    pseudocode: str,
) -> Instruction:
    return core_instruction(
        entity_id=entity_id,
        since=CORE_0,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=4,
        family_id=FAMILY.entity_id,
        fields=(
            _ref(
                register_name,
                1,
                register_role,
                register_ownership,
                register_description,
            ),
            _slot(),
        ),
        state_effects=state_effects,
        semantics=_semantics(
            description,
            (
                f"{register_name} must be a valid ref-register ordinal.",
                "slot_u16 must be less than local_ref_count.",
            ),
            success,
            ownership,
            (assembly,),
            pseudocode,
        ),
    )


REF_STACK_LOAD_RETAIN = _stack_transfer(
    entity_id="core.instruction.ref.stack.load.retain",
    summary="Retains a local ref slot into a register.",
    opcode=0xCE,
    mnemonic="ref.stack.load.retain",
    register_name="dst_r8",
    register_role=InstructionFieldRole.RESULT,
    register_ownership=RefOwnership.REPLACE_RETAIN,
    register_description="Destination ref-register ordinal.",
    description=(
        "Creates one owned register copy of a local ref slot and leaves the "
        "slot unchanged."
    ),
    success=(
        "A null slot publishes null; a non-null borrowed or owned slot "
        "publishes one new owner to dst_r8.",
    ),
    ownership=(
        "The source is retained before replacing and possibly releasing an old "
        "destination owner.",
    ),
    state_effects=(state_read(StateResource.FRAME_LOCALS, "slot_u16"),),
    assembly="%r<dst> = ref.stack.load.retain #r<slot>",
    pseudocode=(
        "source = local_refs[slot_u16];\n"
        "new_owner = retain_ref(source);\n"
        "replace_ref(&refs[dst_r8], new_owner);\n"
        "pc = pc + 4;"
    ),
)

REF_STACK_LOAD_MOVE = _stack_transfer(
    entity_id="core.instruction.ref.stack.load.move",
    summary="Moves a local ref slot into a register.",
    opcode=0xCF,
    mnemonic="ref.stack.load.move",
    register_name="dst_r8",
    register_role=InstructionFieldRole.RESULT,
    register_ownership=RefOwnership.REPLACE_MOVE,
    register_description="Destination ref-register ordinal.",
    description=(
        "Transfers one complete local-slot state into dst_r8 and clears the local slot."
    ),
    success=(
        "slot_u16 becomes canonical null and dst_r8 receives its exact prior state.",
    ),
    ownership=(
        "A borrowed state remains borrowed; an owned state transfers exactly "
        "one release obligation; a previous destination owner is released.",
    ),
    state_effects=(
        state_read(StateResource.FRAME_LOCALS, "slot_u16"),
        state_write(StateResource.FRAME_LOCALS, "slot_u16"),
    ),
    assembly="%r<dst> = ref.stack.load.move #r<slot>",
    pseudocode=(
        "source = local_refs[slot_u16];\n"
        "local_refs[slot_u16] = canonical_null_ref;\n"
        "replace_ref(&refs[dst_r8], source);\n"
        "pc = pc + 4;"
    ),
)

REF_STACK_STORE_RETAIN = _stack_transfer(
    entity_id="core.instruction.ref.stack.store.retain",
    summary="Retains a ref register into a local slot.",
    opcode=0xD0,
    mnemonic="ref.stack.store.retain",
    register_name="src_r8",
    register_role=InstructionFieldRole.OPERAND,
    register_ownership=RefOwnership.RETAIN,
    register_description="Source ref-register ordinal retained without mutation.",
    description=(
        "Creates one owned local-slot copy of src_r8 and leaves the register unchanged."
    ),
    success=(
        "A null register stores null; a non-null borrowed or owned register "
        "stores one new owner in slot_u16.",
    ),
    ownership=(
        "The source is retained before replacing and possibly releasing an old "
        "slot owner.",
    ),
    state_effects=(state_write(StateResource.FRAME_LOCALS, "slot_u16"),),
    assembly="ref.stack.store.retain %r<src>, #r<slot>",
    pseudocode=(
        "source = refs[src_r8];\n"
        "new_owner = retain_ref(source);\n"
        "replace_ref(&local_refs[slot_u16], new_owner);\n"
        "pc = pc + 4;"
    ),
)

REF_STACK_STORE_MOVE = _stack_transfer(
    entity_id="core.instruction.ref.stack.store.move",
    summary="Moves a ref register into a local slot.",
    opcode=0xD1,
    mnemonic="ref.stack.store.move",
    register_name="src_r8",
    register_role=InstructionFieldRole.OPERAND,
    register_ownership=RefOwnership.MOVE,
    register_description="Source ref-register ordinal cleared by the transfer.",
    description=(
        "Transfers one complete ref-register state into slot_u16 and clears the "
        "source register."
    ),
    success=(
        "src_r8 becomes canonical null and slot_u16 receives its exact prior state.",
    ),
    ownership=(
        "A borrowed state remains borrowed; an owned state transfers exactly "
        "one release obligation; a previous slot owner is released.",
    ),
    state_effects=(state_write(StateResource.FRAME_LOCALS, "slot_u16"),),
    assembly="ref.stack.store.move %r<src>, #r<slot>",
    pseudocode=(
        "source = refs[src_r8];\n"
        "refs[src_r8] = canonical_null_ref;\n"
        "replace_ref(&local_refs[slot_u16], source);\n"
        "pc = pc + 4;"
    ),
)

REF_STACK_DISCARD = core_instruction(
    entity_id="core.instruction.ref.stack.discard",
    since=CORE_0,
    summary="Releases or clears one local ref slot.",
    opcode=0xD2,
    mnemonic="ref.stack.discard",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(zero_padding("zero_padding_u8", 1, 1), _slot()),
    state_effects=(state_write(StateResource.FRAME_LOCALS, "slot_u16"),),
    semantics=_semantics(
        "Replaces local ref slot_u16 with canonical null.",
        (
            "zero_padding_u8 must equal zero.",
            "slot_u16 must be less than local_ref_count.",
        ),
        ("slot_u16 becomes canonical null; discarding null is a no-op.",),
        (
            "A previous owned state is released through its actual descriptor; "
            "a borrow or null is only cleared.",
        ),
        ("ref.stack.discard #r<slot>",),
        "clear_ref(&local_refs[slot_u16]);\npc = pc + 4;",
    ),
)

INSTRUCTIONS = (
    REF_NULL,
    REF_COMPARE_NULL,
    REF_COMPARE_EQ,
    REF_RETAIN,
    REF_MOVE,
    REF_DISCARD,
    REF_STACK_LOAD_RETAIN,
    REF_STACK_LOAD_MOVE,
    REF_STACK_STORE_RETAIN,
    REF_STACK_STORE_MOVE,
    REF_STACK_DISCARD,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
