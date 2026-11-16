# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generic typed ref-state and function-local ref-slot instructions."""

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
from iree.vm.bytecode.spec.schema import U8, U16, U32, Field
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
        "copies acquire only for owned sources and preserve borrowed sources; moves "
        "snapshot and clear their distinct source before replacement. Copying or moving "
        "an internal borrow preserves its dominating lifetime. Frame unwind releases every "
        "owner and clears every borrow in the exact declared register and local-slot "
        "extents. A borrowed-only type has no owner prefix: internal copies and moves "
        "preserve this restriction, and ownership-producing instructions fail before "
        "touching the object or destination. Indexed slot accesses check the byte "
        "offset's cell alignment and frame range before touching any reference. "
        "Other ref operations are infallible after verification. No ref operation "
        "suspends."
    ),
)


_BORROWED_ONLY_RETAIN_FAILURES = (
    FailureCase(
        "failed_precondition",
        "The retained source has a borrowed-only type.",
        "No object, source, or destination is modified.",
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
    failures=_BORROWED_ONLY_RETAIN_FAILURES,
    assembly="%r<destination> = ref.retain %r<source>",
    pseudocode=(
        "source = refs[source_r8];\n"
        "require_can_retain(source);\n"
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
        failures=(
            _BORROWED_ONLY_RETAIN_FAILURES
            if register_ownership in (RefOwnership.RETAIN, RefOwnership.REPLACE_RETAIN)
            else ()
        ),
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
    "require_can_retain(source);\n"
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
    "require_can_retain(source);\n"
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

REF_SELECT = Instruction(
    opcode=0xD3,
    mnemonic="ref.select",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Copies the selected reference and its ownership state into a register.",
    fields=(
        _ref_register(
            "destination_r8",
            FieldRole.RESULT,
            RefOwnership.REPLACE_COPY,
            "Ref-register ordinal replaced by the selected reference state.",
        ),
        _value_register(
            "condition_v8",
            FieldRole.OPERAND,
            "Complete 64-bit truth-condition value-register ordinal.",
        ),
        _ref_register(
            "true_r8",
            FieldRole.OPERAND,
            RefOwnership.COPY,
            "Source selected when the condition is nonzero.",
        ),
        _ref_register(
            "false_r8",
            FieldRole.OPERAND,
            RefOwnership.COPY,
            "Source selected when the condition is zero.",
        ),
        InstructionField(
            Field("zero_padding_u8", U8, "Canonical zero padding.", 3),
            FieldRole.PADDING,
            FieldRuleUse(FieldRule.ZERO),
        ),
    ),
    semantics=None,
    behavior=(
        "Selects true_r8 when the complete condition is nonzero, otherwise false_r8, "
        "and copies its complete state into destination_r8. All ref ordinals may alias."
    ),
    success=(
        "A null selection publishes null; a non-null selection preserves its exact "
        "object, descriptor, and ownership state.",
    ),
    assembly="%r<destination> = ref.select %v<condition>, %r<true>, %r<false>",
    pseudocode=(
        "selected = values[condition_v8] != 0 ? refs[true_r8] : refs[false_r8];\n"
        "copy = is_owned_ref(selected) ? retain_ref(selected) : selected;\n"
        "replace_ref(&refs[destination_r8], copy);\n"
        "pc = pc + 8;"
    ),
    ownership=(
        "An owned selection is retained before releasing the destination. A borrowed "
        "selection preserves its existing anchor, which must cover destination "
        "replacement and all later uses. An aliased selection has no net ownership "
        "change; borrowed-only objects require no intrusive prefix access.",
    ),
)

REF_COPY = Instruction(
    opcode=0xD4,
    mnemonic="ref.copy",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Copies a ref register while preserving its ownership state.",
    fields=(
        _ref_register(
            "destination_r8",
            FieldRole.RESULT,
            RefOwnership.REPLACE_COPY,
            "Ref-register ordinal replaced by the copied state.",
        ),
        _ref_register(
            "source_r8",
            FieldRole.OPERAND,
            RefOwnership.COPY,
            "Ref-register ordinal copied without mutation.",
        ),
        _padding(),
    ),
    semantics=None,
    behavior=(
        "Copies source_r8 into destination_r8 without changing the source. Null "
        "stays null, a borrow stays borrowed, and an owner acquires one new owner. "
        "Both ordinals may alias."
    ),
    success=(
        "The destination carries the same object, exact descriptor, and ownership "
        "state as the source.",
    ),
    assembly="%r<destination> = ref.copy %r<source>",
    pseudocode=(
        "source = refs[source_r8];\n"
        "copy = is_owned_ref(source) ? retain_ref(source) : source;\n"
        "replace_ref(&refs[destination_r8], copy);\n"
        "pc = pc + 4;"
    ),
    ownership=(
        "An owned source is retained before releasing the previous destination. "
        "A borrowed source requires its anchor to cover both copies, including "
        "destination replacement. Self-copy preserves state with no net refcount "
        "change; a borrowed self-copy performs no retain or release.",
    ),
)

REF_STACK_LOAD_COPY = _stack_transfer(
    0xD5,
    "ref.stack.load.copy",
    "Copies a local ref slot into a register, preserving ownership state.",
    "destination_r8",
    FieldRole.RESULT,
    RefOwnership.REPLACE_COPY,
    "Ref-register ordinal replaced by the copied state.",
    "Copies one complete local-slot state into destination_r8 without changing the slot.",
    "The destination carries the same object, exact descriptor, and ownership state "
    "as the source slot.",
    "An owned slot acquires one owner before destination replacement. A borrowed slot "
    "performs no acquisition and requires its anchor to cover both copies, including "
    "destination replacement. Null copies null.",
    (StateEffect(StateAccess.READ, StateResource.FRAME_LOCALS, ("slot_u16",)),),
    "%r<destination> = ref.stack.load.copy #r<slot>",
    "source = local_refs[slot_u16];\n"
    "copy = is_owned_ref(source) ? retain_ref(source) : source;\n"
    "replace_ref(&refs[destination_r8], copy);\n"
    "pc = pc + 4;",
)

REF_STACK_STORE_COPY = _stack_transfer(
    0xD6,
    "ref.stack.store.copy",
    "Copies a ref register into a local slot, preserving ownership state.",
    "source_r8",
    FieldRole.OPERAND,
    RefOwnership.COPY,
    "Ref-register ordinal copied without mutation.",
    "Copies one complete ref-register state into slot_u16 without changing the register.",
    "The destination slot carries the same object, exact descriptor, and ownership "
    "state as the source register.",
    "An owned register acquires one owner before slot replacement. A borrowed register "
    "performs no acquisition and requires its anchor to cover both copies, including "
    "slot replacement. Null copies null.",
    (StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("slot_u16",)),),
    "ref.stack.store.copy %r<source>, #r<slot>",
    "source = refs[source_r8];\n"
    "copy = is_owned_ref(source) ? retain_ref(source) : source;\n"
    "replace_ref(&local_refs[slot_u16], copy);\n"
    "pc = pc + 4;",
)

REF_BORROW = Instruction(
    opcode=0xD7,
    mnemonic="ref.borrow",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Borrows a live reference into a distinct register.",
    fields=(
        _ref_register(
            "destination_r8",
            FieldRole.RESULT,
            RefOwnership.REPLACE_BORROW,
            "Ref-register ordinal replaced by the borrowed handle.",
        ),
        _ref_register(
            "source_r8",
            FieldRole.OPERAND,
            RefOwnership.BORROW,
            "Distinct live source whose owner covers the borrow.",
        ),
        _padding(),
    ),
    semantics=None,
    behavior=(
        "Copies the exact object and descriptor into a distinct borrowed carrier "
        "without retaining the source. Null stays canonical null. The compiler "
        "proves a live anchor through every use of the resulting borrow."
    ),
    success=("The source is unchanged and the destination holds null or a borrow.",),
    assembly="%r<destination> = ref.borrow %r<source>",
    pseudocode=(
        "source = refs[source_r8];\n"
        "borrowed = source.is_null ? canonical_null_ref : as_borrowed_ref(source);\n"
        "replace_ref(&refs[destination_r8], borrowed);\n"
        "pc = pc + 4;"
    ),
    rules=(
        RecordRule(
            RecordRuleKind.FIELDS_DISTINCT,
            ("destination_r8", "source_r8"),
            summary="The source and destination ref registers must differ.",
        ),
    ),
    ownership=(
        "The previous destination owner is released after installing the borrow. "
        "A distinct source carrier and a dominating live anchor are required. "
        "No ownership is acquired; clearing the destination never releases the source.",
    ),
)

REF_STACK_LOAD_BORROW = _stack_transfer(
    0xC7,
    "ref.stack.load.borrow",
    "Borrows a local ref slot into a register.",
    "destination_r8",
    FieldRole.RESULT,
    RefOwnership.REPLACE_BORROW,
    "Ref-register ordinal replaced by the borrowed handle.",
    "Borrows the exact object and descriptor from a live local slot without retaining.",
    "The slot is unchanged and the destination holds canonical null or a borrow.",
    "The compiler proves an anchor covering destination replacement and every use "
    "of the borrow. A previous destination owner is released; the source is not retained.",
    (StateEffect(StateAccess.READ, StateResource.FRAME_LOCALS, ("slot_u16",)),),
    "%r<destination> = ref.stack.load.borrow #r<slot>",
    "source = local_refs[slot_u16];\n"
    "borrowed = source.is_null ? canonical_null_ref : as_borrowed_ref(source);\n"
    "replace_ref(&refs[destination_r8], borrowed);\n"
    "pc = pc + 4;",
)

REF_STACK_STORE_BORROW = _stack_transfer(
    0xE6,
    "ref.stack.store.borrow",
    "Borrows a ref register into a local slot.",
    "source_r8",
    FieldRole.OPERAND,
    RefOwnership.BORROW,
    "Live source whose owner covers the borrowed slot.",
    "Borrows the exact object and descriptor from a live register without retaining.",
    "The register is unchanged and the slot holds canonical null or a borrow.",
    "The compiler proves an anchor covering slot replacement and every use of the "
    "borrow. A previous slot owner is released; the source is not retained.",
    (StateEffect(StateAccess.WRITE, StateResource.FRAME_LOCALS, ("slot_u16",)),),
    "ref.stack.store.borrow %r<source>, #r<slot>",
    "source = refs[source_r8];\n"
    "borrowed = source.is_null ? canonical_null_ref : as_borrowed_ref(source);\n"
    "replace_ref(&local_refs[slot_u16], borrowed);\n"
    "pc = pc + 4;",
)


def _indexed_stack_transfer(opcode: int, source: Instruction) -> Instruction:
    mnemonic = source.mnemonic.replace("ref.stack.load.", "ref.stack.load.indexed.")
    mnemonic = mnemonic.replace("ref.stack.store.", "ref.stack.store.indexed.")
    return source._replace(
        opcode=opcode,
        mnemonic=mnemonic,
        summary=source.summary + " Selects the slot by a checked byte offset.",
        fields=(
            source.fields[0],
            _value_register(
                "offset_v8",
                FieldRole.OPERAND,
                "Byte offset into frame-local ref cells.",
            ),
            _field(
                "zero_padding_tail_u8",
                U8,
                "Canonical zero padding.",
                FieldRole.PADDING,
                FieldRule.ZERO,
            ),
        ),
        behavior=(
            "values[offset_v8] must be divisible by the 16-byte ref cell size "
            "and select an existing local slot. "
            + source.behavior.replace("slot_u16", "the selected slot")
        ),
        success=tuple(
            text.replace("slot_u16", "the selected slot") for text in source.success
        ),
        assembly=source.assembly.replace(source.mnemonic, mnemonic).replace(
            "#r<slot>", "%v<offset>"
        ),
        pseudocode=(
            "slot = require_ref_cell_byte_offset(values[offset_v8]);\n"
            + source.pseudocode.replace("slot_u16", "slot")
        ),
        state_effects=tuple(
            effect._replace(resource_fields=("offset_v8",))
            for effect in source.state_effects
        ),
        failures=(
            FailureCase(
                "out_of_range",
                "The byte offset is unaligned or outside the local ref array.",
                "No reference state is modified.",
            ),
            *source.failures,
        ),
    )


REF_STACK_LOAD_INDEXED_RETAIN = _indexed_stack_transfer(0xE7, REF_STACK_LOAD_RETAIN)
REF_STACK_LOAD_INDEXED_MOVE = _indexed_stack_transfer(0xE8, REF_STACK_LOAD_MOVE)
REF_STACK_STORE_INDEXED_RETAIN = _indexed_stack_transfer(0xE9, REF_STACK_STORE_RETAIN)
REF_STACK_STORE_INDEXED_MOVE = _indexed_stack_transfer(0xEA, REF_STACK_STORE_MOVE)
REF_STACK_LOAD_INDEXED_COPY = _indexed_stack_transfer(0xEB, REF_STACK_LOAD_COPY)
REF_STACK_STORE_INDEXED_COPY = _indexed_stack_transfer(0xEC, REF_STACK_STORE_COPY)
REF_STACK_LOAD_INDEXED_BORROW = _indexed_stack_transfer(0xED, REF_STACK_LOAD_BORROW)
REF_STACK_STORE_INDEXED_BORROW = _indexed_stack_transfer(0xEE, REF_STACK_STORE_BORROW)

REF_STACK_DISCARD_RANGE = Instruction(
    opcode=0xEF,
    mnemonic="ref.stack.discard.range",
    since=CORE_0,
    family=REF_FAMILY,
    summary="Releases or clears a contiguous range of local ref slots.",
    fields=(
        _value_register(
            "offset_v8", FieldRole.OPERAND, "Byte offset into frame-local ref cells."
        ),
        _padding(U16),
        _field(
            "count_u32",
            U32,
            "Number of consecutive local ref slots.",
            FieldRole.IMMEDIATE,
            FieldRule.ANY_BITS,
        ),
    ),
    semantics=None,
    behavior="Checks a 16-byte-aligned logical ref offset and count, then replaces the entire range with canonical null in ascending slot order.",
    success=(
        "Each previous owner is released once; borrowed and null cells are only cleared.",
    ),
    assembly="ref.stack.discard.range %v<offset> {count = <count>}",
    pseudocode="base = require_ref_cell_byte_range(values[offset_v8], count_u32);\nfor i in 0..count_u32: clear_ref(&local_refs[base + i]);\npc = pc + 8;",
    state_effects=(
        StateEffect(
            StateAccess.WRITE, StateResource.FRAME_LOCALS, ("offset_v8", "count_u32")
        ),
    ),
    failures=(
        FailureCase(
            "out_of_range",
            "The byte offset is unaligned or the range is outside the local ref array.",
            "No reference state is modified.",
        ),
    ),
    ownership=(
        "A zero-length range touches no storage. Each cell is cleared before its previous owner is released.",
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
    REF_SELECT,
    REF_COPY,
    REF_STACK_LOAD_COPY,
    REF_STACK_STORE_COPY,
    REF_BORROW,
    REF_STACK_LOAD_BORROW,
    REF_STACK_STORE_BORROW,
    REF_STACK_LOAD_INDEXED_RETAIN,
    REF_STACK_LOAD_INDEXED_MOVE,
    REF_STACK_STORE_INDEXED_RETAIN,
    REF_STACK_STORE_INDEXED_MOVE,
    REF_STACK_LOAD_INDEXED_COPY,
    REF_STACK_STORE_INDEXED_COPY,
    REF_STACK_LOAD_INDEXED_BORROW,
    REF_STACK_STORE_INDEXED_BORROW,
    REF_STACK_DISCARD_RANGE,
)
