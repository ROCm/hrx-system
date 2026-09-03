# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generic typed ref-state and function-local ref-slot instructions."""

from iree.vm.bytecode.spec.isa import (
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
from iree.vm.bytecode.spec.schema import U8, U16, Field
from iree.vm.bytecode.spec.version import CORE_0

REF_FAMILY = InstructionFamily(
    name="ref",
    since=CORE_0,
    summary="Generic typed ref-state and function-local ref-slot operations.",
    contract=(
        "A ref register or local slot is canonical null, a non-null borrowed typed "
        "object, or a non-null owned typed object carrying one release obligation. "
        "Canonical null has both words zero. The actual descriptor moves with the "
        "object; generic ref operations never cast or dereference it. Clearing first "
        "installs canonical null and then releases only a prior owner. Replacement "
        "installs an already-safe state before releasing a prior owner. Retaining null "
        "produces null; retaining a live object creates an owner through its actual "
        "descriptor without querying the refcount. Retains acquire before replacement; "
        "moves snapshot and clear their distinct source before replacement. Moving an "
        "internal borrow preserves its dominating lifetime. Frame unwind releases every "
        "owner and clears every borrow in the exact declared register and local-slot "
        "extents. These operations are infallible after verification and never suspend."
    ),
)


def _field(
    name: str, encoding, summary: str, role: FieldRole, rule
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, encoding, summary), role, rule)


def _ref_register(
    name: str, role: FieldRole, ownership: RefOwnership, summary: str
) -> InstructionField:
    return InstructionField(
        Field(name, U8, summary),
        role,
        FieldRuleUse(FieldRule.REGISTER_REF),
        RuntimeRefPolicy("dynamic", RefNullPolicy.NULLABLE, ownership),
    )


def _value_register(name: str, role: FieldRole, summary: str) -> InstructionField:
    return _field(name, U8, summary, role, FieldRule.REGISTER_VALUE)


def _padding(encoding=U8) -> InstructionField:
    return _field(
        f"zero_padding_{encoding.name}",
        encoding,
        "Canonical zero padding.",
        FieldRole.PADDING,
        FieldRule.ZERO,
    )


def _local_slot() -> InstructionField:
    return _field(
        "slot_u16",
        U16,
        "Direct function-local ref-slot ordinal.",
        FieldRole.IMMEDIATE,
        FieldRule.REF_SLOT,
    )


REF_NULL = Instruction(
    opcode=0xC8,
    mnemonic="ref.null",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Clears a ref register to canonical null.",
    fields=(
        _ref_register(
            "destination_r8",
            FieldRole.RESULT,
            RefOwnership.CLEAR,
            "Ref-register ordinal replaced by canonical null.",
        ),
        _padding(U16),
    ),
    semantics=None,
    behavior="Replaces destination_r8 with canonical null.",
    success=("destination_r8 becomes canonical null.",),
    assembly="%r<destination> = ref.null",
    pseudocode="clear_ref(&refs[destination_r8]);\npc = pc + 4;",
    ownership=(
        "A previous owner is released through its actual descriptor; a previous borrow "
        "or null is only cleared.",
    ),
)

REF_COMPARE_NULL = Instruction(
    opcode=0xC9,
    mnemonic="ref.compare.null",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Tests a ref register for canonical null.",
    fields=(
        _value_register(
            "destination_v8",
            FieldRole.RESULT,
            "Value-register ordinal receiving canonical zero or one.",
        ),
        _ref_register(
            "source_r8",
            FieldRole.OPERAND,
            RefOwnership.INSPECT,
            "Ref-register ordinal inspected without dereference.",
        ),
        _padding(),
    ),
    semantics=None,
    behavior="Tests source_r8 for canonical null without inspecting its object.",
    success=(
        "destination_v8 becomes canonical one when source_r8 is null and zero "
        "otherwise.",
    ),
    assembly="%v<destination> = ref.compare.null %r<source>",
    pseudocode=(
        "values[destination_v8] = canonical_bool(refs[source_r8].is_null);\n"
        "pc = pc + 4;"
    ),
    ownership=("source_r8 is neither retained, released, nor dereferenced.",),
)

REF_COMPARE_EQ = Instruction(
    opcode=0xCA,
    mnemonic="ref.compare.eq",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Tests exact typed object identity of two refs.",
    fields=(
        _value_register(
            "destination_v8",
            FieldRole.RESULT,
            "Value-register ordinal receiving canonical zero or one.",
        ),
        _ref_register(
            "left_r8",
            FieldRole.OPERAND,
            RefOwnership.INSPECT,
            "Left ref-register ordinal.",
        ),
        _ref_register(
            "right_r8",
            FieldRole.OPERAND,
            RefOwnership.INSPECT,
            "Right ref-register ordinal.",
        ),
    ),
    semantics=None,
    behavior=(
        "Tests canonical null or exact non-null descriptor-and-object identity. "
        "Ownership state does not affect equality."
    ),
    success=(
        "Two null refs compare equal. Null and non-null compare unequal. Two non-null "
        "refs compare equal only when their descriptors and object pointers match; "
        "destination_v8 receives canonical zero or one.",
    ),
    assembly="%v<destination> = ref.compare.eq %r<left>, %r<right>",
    pseudocode=(
        "left = refs[left_r8];\n"
        "right = refs[right_r8];\n"
        "equal = left.is_null ? right.is_null\n"
        "    : !right.is_null && left.descriptor == right.descriptor &&\n"
        "      left.object == right.object;\n"
        "values[destination_v8] = canonical_bool(equal);\n"
        "pc = pc + 4;"
    ),
    ownership=(
        "Neither source is retained, released, dereferenced, ordered, hashed, or "
        "exposed as integer bits.",
    ),
)

REF_RETAIN = Instruction(
    opcode=0xCB,
    mnemonic="ref.retain",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Creates an owned copy of a ref in another register.",
    fields=(
        _ref_register(
            "destination_r8",
            FieldRole.RESULT,
            RefOwnership.REPLACE_RETAIN,
            "Ref-register ordinal replaced by the new owner.",
        ),
        _ref_register(
            "source_r8",
            FieldRole.OPERAND,
            RefOwnership.RETAIN,
            "Ref-register ordinal retained without mutation.",
        ),
        _padding(),
    ),
    semantics=None,
    behavior=(
        "Creates an owned copy of source_r8 and replaces destination_r8 without "
        "changing the source; both ordinals may alias."
    ),
    success=(
        "A null source publishes null; a non-null source publishes one owner carrying "
        "its exact object and descriptor.",
    ),
    assembly="%r<destination> = ref.retain %r<source>",
    pseudocode=(
        "source = refs[source_r8];\n"
        "new_owner = retain_ref(source);\n"
        "replace_ref(&refs[destination_r8], new_owner);\n"
        "pc = pc + 4;"
    ),
    ownership=(
        "The source is snapshotted and retained before a previous destination owner "
        "is released. A borrowed self-retain becomes owned; an owned self-retain has "
        "no net refcount change.",
    ),
)

REF_MOVE = Instruction(
    opcode=0xCC,
    mnemonic="ref.move",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Destructively transfers one complete ref-register state.",
    fields=(
        _ref_register(
            "destination_r8",
            FieldRole.RESULT,
            RefOwnership.REPLACE_MOVE,
            "Ref-register ordinal replaced by the moved state.",
        ),
        _ref_register(
            "source_r8",
            FieldRole.OPERAND,
            RefOwnership.MOVE,
            "Distinct ref-register ordinal cleared by the transfer.",
        ),
        _padding(),
    ),
    semantics=None,
    behavior=(
        "Transfers the complete null, borrowed, or owned state from source_r8 to a "
        "distinct destination_r8."
    ),
    success=(
        "source_r8 becomes canonical null and destination_r8 receives its exact prior "
        "state.",
    ),
    assembly="%r<destination> = ref.move %r<source>",
    pseudocode=(
        "source = refs[source_r8];\n"
        "refs[source_r8] = canonical_null_ref;\n"
        "replace_ref(&refs[destination_r8], source);\n"
        "pc = pc + 4;"
    ),
    rules=(
        RecordRule(
            RecordRuleKind.FIELDS_DISTINCT,
            ("destination_r8", "source_r8"),
            summary="The source and destination ref-register ordinals must differ.",
        ),
    ),
    ownership=(
        "The source is snapshotted and cleared before destination replacement. A "
        "borrow remains borrowed; an owner transfers exactly one release obligation; "
        "a previous destination owner is released.",
    ),
)

REF_DISCARD = Instruction(
    opcode=0xCD,
    mnemonic="ref.discard",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Releases or clears one ref register.",
    fields=(
        _ref_register(
            "source_r8",
            FieldRole.OPERAND,
            RefOwnership.CLEAR,
            "Ref-register ordinal to discard.",
        ),
        _padding(U16),
    ),
    semantics=None,
    behavior="Replaces source_r8 with canonical null.",
    success=("source_r8 becomes canonical null; discarding null is a no-op.",),
    assembly="ref.discard %r<source>",
    pseudocode="clear_ref(&refs[source_r8]);\npc = pc + 4;",
    ownership=(
        "A previous owner is released through its private no-status descriptor "
        "teardown; a borrow or null is only cleared.",
    ),
)


def _stack_transfer(
    opcode: int,
    mnemonic: str,
    summary: str,
    register_name: str,
    register_role: FieldRole,
    register_ownership: RefOwnership,
    register_summary: str,
    behavior: str,
    success: str,
    ownership: str,
    state_effects: tuple[StateEffect, ...],
    assembly: str,
    pseudocode: str,
) -> Instruction:
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=REF_FAMILY,
        summary=summary,
        fields=(
            _ref_register(
                register_name,
                register_role,
                register_ownership,
                register_summary,
            ),
            _local_slot(),
        ),
        semantics=None,
        behavior=behavior,
        success=(success,),
        assembly=assembly,
        pseudocode=pseudocode,
        state_effects=state_effects,
        ownership=(ownership,),
    )


REF_STACK_LOAD_RETAIN = _stack_transfer(
    0xCE,
    "ref.stack.load.retain",
    "Retains a local ref slot into a register.",
    "destination_r8",
    FieldRole.RESULT,
    RefOwnership.REPLACE_RETAIN,
    "Ref-register ordinal replaced by the new owner.",
    "Creates one owned register copy of a local ref slot and leaves the slot unchanged.",
    "A null slot publishes null; a non-null borrowed or owned slot publishes one new "
    "owner to destination_r8.",
    "The source is retained before replacing and possibly releasing a previous "
    "destination owner.",
    (StateEffect(StateAccess.READ, StateResource.FRAME_LOCALS, ("slot_u16",)),),
    "%r<destination> = ref.stack.load.retain #r<slot>",
    "source = local_refs[slot_u16];\n"
    "new_owner = retain_ref(source);\n"
    "replace_ref(&refs[destination_r8], new_owner);\n"
    "pc = pc + 4;",
)

REF_STACK_LOAD_MOVE = _stack_transfer(
    0xCF,
    "ref.stack.load.move",
    "Moves a local ref slot into a register.",
    "destination_r8",
    FieldRole.RESULT,
    RefOwnership.REPLACE_MOVE,
    "Ref-register ordinal replaced by the moved state.",
    "Transfers one complete local-slot state into destination_r8 and clears the slot.",
    "slot_u16 becomes canonical null and destination_r8 receives its exact prior state.",
    "A borrow remains borrowed; an owner transfers exactly one release obligation; a "
    "previous destination owner is released.",
    (
        StateEffect(StateAccess.READ, StateResource.FRAME_LOCALS, ("slot_u16",)),
        StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("slot_u16",)),
    ),
    "%r<destination> = ref.stack.load.move #r<slot>",
    "source = local_refs[slot_u16];\n"
    "local_refs[slot_u16] = canonical_null_ref;\n"
    "replace_ref(&refs[destination_r8], source);\n"
    "pc = pc + 4;",
)

REF_STACK_STORE_RETAIN = _stack_transfer(
    0xD0,
    "ref.stack.store.retain",
    "Retains a ref register into a local slot.",
    "source_r8",
    FieldRole.OPERAND,
    RefOwnership.RETAIN,
    "Ref-register ordinal retained without mutation.",
    "Creates one owned local-slot copy of source_r8 and leaves the register unchanged.",
    "A null register stores null; a non-null borrowed or owned register stores one new "
    "owner in slot_u16.",
    "The source is retained before replacing and possibly releasing a previous slot "
    "owner.",
    (StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("slot_u16",)),),
    "ref.stack.store.retain %r<source>, #r<slot>",
    "source = refs[source_r8];\n"
    "new_owner = retain_ref(source);\n"
    "replace_ref(&local_refs[slot_u16], new_owner);\n"
    "pc = pc + 4;",
)

REF_STACK_STORE_MOVE = _stack_transfer(
    0xD1,
    "ref.stack.store.move",
    "Moves a ref register into a local slot.",
    "source_r8",
    FieldRole.OPERAND,
    RefOwnership.MOVE,
    "Ref-register ordinal cleared by the transfer.",
    "Transfers one complete ref-register state into slot_u16 and clears source_r8.",
    "source_r8 becomes canonical null and slot_u16 receives its exact prior state.",
    "A borrow remains borrowed; an owner transfers exactly one release obligation; a "
    "previous slot owner is released.",
    (StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("slot_u16",)),),
    "ref.stack.store.move %r<source>, #r<slot>",
    "source = refs[source_r8];\n"
    "refs[source_r8] = canonical_null_ref;\n"
    "replace_ref(&local_refs[slot_u16], source);\n"
    "pc = pc + 4;",
)

REF_STACK_DISCARD = Instruction(
    opcode=0xD2,
    mnemonic="ref.stack.discard",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Releases or clears one function-local ref slot.",
    fields=(_padding(), _local_slot()),
    semantics=None,
    behavior="Replaces local ref slot_u16 with canonical null.",
    success=("slot_u16 becomes canonical null; discarding null is a no-op.",),
    assembly="ref.stack.discard #r<slot>",
    pseudocode="clear_ref(&local_refs[slot_u16]);\npc = pc + 4;",
    state_effects=(
        StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("slot_u16",)),
    ),
    ownership=(
        "A previous owner is released through its actual descriptor; a borrow or null "
        "is only cleared.",
    ),
)

REF_INSTRUCTIONS = (
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
