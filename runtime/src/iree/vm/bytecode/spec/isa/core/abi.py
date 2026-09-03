# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Physical overflow access for function call packets."""

from iree.vm.bytecode.spec.isa import (
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
    RuntimeRefPolicy,
    StateEffect,
)
from iree.vm.bytecode.spec.isa.core.rules import (
    FieldRule,
    RefNullPolicy,
    RefOwnership,
    StateAccess,
    StateResource,
)
from iree.vm.bytecode.spec.module.records import SIGNATURE_ROW
from iree.vm.bytecode.spec.schema import U8, U16, Field
from iree.vm.bytecode.spec.version import CORE_0

DIRECT_REGISTER_COUNT = 16

ABI_FAMILY = InstructionFamily(
    name="abi",
    since=CORE_0,
    summary="Physical overflow access for function call packets.",
    contract=(
        "A logical signature is source ordered, while verification derives independent "
        "value, ref, and function argument and result banks. Ordinals below "
        f"{DIRECT_REGISTER_COUNT} in each "
        "bank use direct registers; later ordinals use zero-based overflow slots. "
        "Value slots are naturally aligned 64-bit cells with register-exact bits. Ref "
        "slots carry complete null, borrowed, or owned states. Function slots carry "
        "complete 16-byte non-owning function values. Argument storage is read-only "
        "except for explicit consuming ref moves; result storage is private to the "
        "paused caller. Ref and function result slots begin canonical null, while value "
        "result bits are unspecified until stored. Call entry validates every dynamic "
        "argument and control.return validates every dynamic result before publication, "
        "so these operations perform no signature walk or type check. The caller and "
        "packet remain live across callee execution and suspension. These operations "
        "are infallible after verification and never suspend."
    ),
)

_SIGNATURE_FIELD_OFFSETS = {
    wire_field.field.name: offset
    for wire_field, offset in zip(
        SIGNATURE_ROW.fields, SIGNATURE_ROW.field_offsets, strict=True
    )
}


def _field(
    name: str, encoding, summary: str, role: FieldRole, rule
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, encoding, summary), role, rule)


def _value_register(name: str, role: FieldRole) -> InstructionField:
    return _field(
        name,
        U8,
        "Value-register side of the overflow transfer.",
        role,
        FieldRule.REGISTER_VALUE,
    )


def _function_register(name: str, role: FieldRole) -> InstructionField:
    return _field(
        name,
        U8,
        "Function-register side of the overflow transfer.",
        role,
        FieldRule.REGISTER_FUNCTION,
    )


def _ref_register(
    name: str,
    role: FieldRole,
    type_contract: str,
    ownership: RefOwnership,
) -> InstructionField:
    return InstructionField(
        Field(name, U8, "Ref-register side of the overflow transfer."),
        role,
        FieldRuleUse(FieldRule.REGISTER_REF),
        RuntimeRefPolicy(type_contract, RefNullPolicy.NULLABLE, ownership),
    )


def _slot(count_field_name: str, summary: str) -> InstructionField:
    return _field(
        "slot_u16",
        U16,
        summary,
        FieldRole.IMMEDIATE,
        FieldRuleUse(
            FieldRule.ABI_SLOT,
            values=(_SIGNATURE_FIELD_OFFSETS[count_field_name],),
        ),
    )


VALUE_ABI_ARGUMENT_LOAD = Instruction(
    opcode=0xC0,
    mnemonic="value.abi.argument.load",
    since=CORE_0,
    family=ABI_FAMILY,
    summary="Loads one exact value argument overflow cell.",
    fields=(
        _value_register("destination_v8", FieldRole.RESULT),
        _slot(
            "argument_value_count_u16",
            "Zero-based value argument overflow-slot ordinal.",
        ),
    ),
    semantics=None,
    behavior=(
        "Copies all 64 bits from one value argument overflow slot into destination_v8 "
        "without conversion, byte swapping, or scalar-width selection."
    ),
    success=("destination_v8 receives bits identical to the selected packet cell.",),
    assembly="%v<destination> = value.abi.argument.load <slot>",
    pseudocode=(
        "values[destination_v8] = frame.argument_value_slots[slot_u16];\npc = pc + 4;"
    ),
    state_effects=(
        StateEffect(
            StateAccess.READ, StateResource.INVOCATION_ARGUMENTS, ("slot_u16",)
        ),
    ),
)

VALUE_ABI_RESULT_STORE = Instruction(
    opcode=0xC1,
    mnemonic="value.abi.result.store",
    since=CORE_0,
    family=ABI_FAMILY,
    summary="Stores one exact value result overflow cell.",
    fields=(
        _value_register("source_v8", FieldRole.OPERAND),
        _slot(
            "result_value_count_u16",
            "Zero-based value result overflow-slot ordinal.",
        ),
    ),
    semantics=None,
    behavior="Copies all 64 bits of source_v8 into one private value result slot.",
    success=(
        "The selected result slot receives an exact copy of source_v8; repeated stores "
        "are valid and the last executed store wins.",
        "Public result variants remain untouched until successful root completion "
        "publishes the complete result list.",
    ),
    assembly="value.abi.result.store %v<source>, <slot>",
    pseudocode=(
        "frame.result_value_slots[slot_u16] = values[source_v8];\npc = pc + 4;"
    ),
    state_effects=(
        StateEffect(StateAccess.WRITE, StateResource.INVOCATION_RESULTS, ("slot_u16",)),
    ),
)


def _ref_abi_instruction(
    opcode: int,
    mnemonic: str,
    summary: str,
    register_name: str,
    register_role: FieldRole,
    type_contract: str,
    ownership_policy: RefOwnership,
    count_field_name: str,
    slot_summary: str,
    behavior: str,
    success: tuple[str, ...],
    ownership: tuple[str, ...],
    state_effects: tuple[StateEffect, ...],
    assembly: str,
    pseudocode: str,
) -> Instruction:
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=ABI_FAMILY,
        summary=summary,
        fields=(
            _ref_register(
                register_name, register_role, type_contract, ownership_policy
            ),
            _slot(count_field_name, slot_summary),
        ),
        semantics=None,
        behavior=behavior,
        success=success,
        assembly=assembly,
        pseudocode=pseudocode,
        state_effects=state_effects,
        ownership=ownership,
    )


REF_ABI_ARGUMENT_LOAD_BORROW = _ref_abi_instruction(
    0xC2,
    "ref.abi.argument.load.borrow",
    "Borrows one ref argument overflow slot into a register.",
    "destination_r8",
    FieldRole.RESULT,
    "signature.argument.ref",
    RefOwnership.BORROW,
    "argument_ref_count_u16",
    "Zero-based ref argument overflow-slot ordinal.",
    "Publishes a non-consuming internal borrow from one argument ref overflow slot "
    "into destination_r8 while leaving the packet source unchanged.",
    (
        "A null packet source publishes canonical null; a non-null source publishes "
        "its exact object and descriptor as a borrow.",
    ),
    (
        "The source is not retained. destination_r8 is replaced only after copying "
        "the source, and a previous destination owner is then released.",
        "The paused caller and argument packet dominate the borrow's complete callee "
        "lifetime, including nested suspension.",
    ),
    (StateEffect(StateAccess.READ, StateResource.INVOCATION_ARGUMENTS, ("slot_u16",)),),
    "%r<destination> = ref.abi.argument.load.borrow <slot>",
    "source = frame.argument_ref_slots[slot_u16];\n"
    "replace_ref(&refs[destination_r8], borrow_ref(source));\n"
    "pc = pc + 4;",
)

REF_ABI_ARGUMENT_LOAD_MOVE = _ref_abi_instruction(
    0xC3,
    "ref.abi.argument.load.move",
    "Consumes one ref argument overflow slot into a register.",
    "destination_r8",
    FieldRole.RESULT,
    "signature.argument.ref",
    RefOwnership.MOVE,
    "argument_ref_count_u16",
    "Zero-based ref argument overflow-slot ordinal.",
    "Transfers the complete null, borrowed, or owned packet state into destination_r8 "
    "and clears the packet source.",
    (
        "The selected argument slot becomes canonical null and destination_r8 receives "
        "its exact prior state; a second move therefore loads null.",
    ),
    (
        "The source is saved and cleared before destination replacement. An owner "
        "transfers one release obligation; an internal borrow remains borrowed under "
        "its existing dominating lifetime; a previous destination owner is released.",
    ),
    (
        StateEffect(
            StateAccess.READ, StateResource.INVOCATION_ARGUMENTS, ("slot_u16",)
        ),
        StateEffect(
            StateAccess.WRITE, StateResource.INVOCATION_ARGUMENTS, ("slot_u16",)
        ),
    ),
    "%r<destination> = ref.abi.argument.load.move <slot>",
    "source = frame.argument_ref_slots[slot_u16];\n"
    "frame.argument_ref_slots[slot_u16] = canonical_null_ref;\n"
    "replace_ref(&refs[destination_r8], source);\n"
    "pc = pc + 4;",
)

REF_ABI_RESULT_STORE_MOVE = _ref_abi_instruction(
    0xC4,
    "ref.abi.result.store.move",
    "Publishes one ref register into a result overflow slot.",
    "source_r8",
    FieldRole.OPERAND,
    "signature.result.ref",
    RefOwnership.PUBLISH_MOVE,
    "result_ref_count_u16",
    "Zero-based ref result overflow-slot ordinal.",
    "Promotes a borrowed source when necessary, moves the owner or canonical null "
    "into a caller-owned result slot, and clears source_r8.",
    (
        "source_r8 becomes canonical null and the result slot receives canonical null "
        "or an owner-backed exact source ref.",
        "Repeated stores replace the prior result state; control.return later validates "
        "the complete packet before publication.",
    ),
    (
        "A borrowed non-null source is retained and converted to owned before "
        "publication. An owned source transfers its existing obligation. A previous "
        "result owner is released only after the new state is safe.",
    ),
    (StateEffect(StateAccess.WRITE, StateResource.INVOCATION_RESULTS, ("slot_u16",)),),
    "ref.abi.result.store.move %r<source>, <slot>",
    "source = promote_borrow_to_owner(refs[source_r8]);\n"
    "old_result = frame.result_ref_slots[slot_u16];\n"
    "frame.result_ref_slots[slot_u16] = source;\n"
    "refs[source_r8] = canonical_null_ref;\n"
    "release_or_clear(old_result);\n"
    "pc = pc + 4;",
)


def _function_abi_instruction(
    opcode: int,
    mnemonic: str,
    summary: str,
    register_name: str,
    register_role: FieldRole,
    count_field_name: str,
    slot_summary: str,
    behavior: str,
    success: tuple[str, ...],
    state_effect: StateEffect,
    assembly: str,
    pseudocode: str,
) -> Instruction:
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=ABI_FAMILY,
        summary=summary,
        fields=(
            _function_register(register_name, register_role),
            _slot(count_field_name, slot_summary),
        ),
        semantics=None,
        behavior=behavior,
        success=success,
        assembly=assembly,
        pseudocode=pseudocode,
        state_effects=(state_effect,),
        ownership=("Function values own nothing and replacement requires no cleanup.",),
    )


FUNC_ABI_ARGUMENT_LOAD = _function_abi_instruction(
    0xC5,
    "func.abi.argument.load",
    "Loads one complete function argument overflow carrier.",
    "destination_f8",
    FieldRole.RESULT,
    "argument_function_count_u16",
    "Zero-based function argument overflow-slot ordinal.",
    "Copies one complete 16-byte function argument slot into destination_f8.",
    ("destination_f8 receives an exact copy of the selected argument slot.",),
    StateEffect(StateAccess.READ, StateResource.INVOCATION_ARGUMENTS, ("slot_u16",)),
    "%f<destination> = func.abi.argument.load <slot>",
    "functions[destination_f8] = frame.argument_function_slots[slot_u16];\n"
    "pc = pc + 4;",
)

FUNC_ABI_RESULT_STORE = _function_abi_instruction(
    0xC6,
    "func.abi.result.store",
    "Stores one complete function result overflow carrier.",
    "source_f8",
    FieldRole.OPERAND,
    "result_function_count_u16",
    "Zero-based function result overflow-slot ordinal.",
    "Copies one complete 16-byte function register into a private result slot.",
    (
        "The selected result slot receives an exact copy of source_f8; repeated stores "
        "are valid and the last executed store wins.",
        "control.return validates the complete result packet before publication.",
    ),
    StateEffect(StateAccess.WRITE, StateResource.INVOCATION_RESULTS, ("slot_u16",)),
    "func.abi.result.store %f<source>, <slot>",
    "frame.result_function_slots[slot_u16] = functions[source_f8];\npc = pc + 4;",
)

ABI_INSTRUCTIONS = (
    VALUE_ABI_ARGUMENT_LOAD,
    VALUE_ABI_RESULT_STORE,
    REF_ABI_ARGUMENT_LOAD_BORROW,
    REF_ABI_ARGUMENT_LOAD_MOVE,
    REF_ABI_RESULT_STORE_MOVE,
    FUNC_ABI_ARGUMENT_LOAD,
    FUNC_ABI_RESULT_STORE,
)
