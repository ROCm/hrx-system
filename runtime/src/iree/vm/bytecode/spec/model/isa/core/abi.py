# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 overflow ABI packet instructions."""

from __future__ import annotations

from model.isa import (
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
    function_register,
    instruction_field,
    ref_register,
    value_register,
)
from model.isa.validation import ABI_SLOT
from model.schema import U16, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.abi",
    since=CORE_0,
    summary="Physical overflow access for function call packets.",
    dependencies=("core.contract.machine",),
    document_order=0,
    normative_text=(
        "A logical function signature is source ordered, but verification "
        "derives independent value, ref, and function banks for arguments and "
        "results. Ordinals 0 through 15 in each bank use direct registers; "
        "ordinal N at or above 16 uses overflow slot N-16 in that bank. Other "
        "descriptor kinds do not consume a bank's ordinals. Value overflow "
        "slots are naturally aligned 64-bit invocation cells with register-"
        "exact bits. Ref overflow slots are complete null, borrowed, or owned "
        "typed ref states. Function overflow slots are complete 16-byte "
        "non-owning function carriers. Argument storage is read-only except "
        "for explicit consuming ref moves. Result storage is private to the "
        "paused caller and is cleared before callee entry. Call entry validates "
        "the complete argument packet and control.return validates the complete "
        "result packet, so these physical-bank handlers perform no signature "
        "walk or dynamic type check. The caller and packet remain live while a "
        "callee executes or suspends. ABI operations are infallible after "
        "verification and never suspend."
    ),
)


def _slot(packet_contract: str):
    return instruction_field(
        "slot_u16",
        2,
        U16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        f"Zero-based {packet_contract} overflow-slot ordinal.",
        (RuleUse(ABI_SLOT.entity_id, (packet_contract,)),),
    )


def _semantics(
    description: str,
    packet_contract: str,
    register_verification: str,
    success: tuple[str, ...],
    ownership: tuple[str, ...],
    assembly: str,
    pseudocode: str,
) -> InstructionSemantics:
    return InstructionSemantics(
        description=description,
        verification=(
            register_verification,
            f"slot_u16 must be less than the signature-derived {packet_contract} "
            "overflow count.",
        ),
        preconditions=(),
        success=(*success, "The program counter advances by four bytes."),
        failures=(),
        ownership=ownership,
        assembly=(assembly,),
        pseudocode=pseudocode,
    )


VALUE_ABI_ARGUMENT_LOAD = core_instruction(
    entity_id="core.instruction.value.abi.argument.load",
    since=CORE_0,
    summary="Loads one exact value argument overflow cell.",
    opcode=0xC0,
    mnemonic="value.abi.argument.load",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Destination value-register ordinal.",
        ),
        _slot("argument.value"),
    ),
    semantics=_semantics(
        (
            "Copies all 64 bits from one value argument overflow slot into "
            "dst_v8 without conversion, byte swapping, or scalar-width selection."
        ),
        "argument.value",
        "dst_v8 must be a valid value-register ordinal.",
        ("dst_v8 receives bits identical to argument_value_slots[slot_u16].",),
        (),
        "%v<dst> = value.abi.argument.load <slot>",
        ("values[dst_v8] = frame.argument_value_slots[slot_u16];\npc = pc + 4;"),
    ),
)

VALUE_ABI_RESULT_STORE = core_instruction(
    entity_id="core.instruction.value.abi.result.store",
    since=CORE_0,
    summary="Stores one exact value result overflow cell.",
    opcode=0xC1,
    mnemonic="value.abi.result.store",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "src_v8",
            1,
            InstructionFieldRole.OPERAND,
            "Source value-register ordinal.",
        ),
        _slot("result.value"),
    ),
    semantics=_semantics(
        ("Copies all 64 bits of src_v8 into one private value result overflow slot."),
        "result.value",
        "src_v8 must be a valid value-register ordinal.",
        (
            "result_value_slots[slot_u16] receives an exact copy of src_v8; "
            "repeated stores are valid and the last executed store wins.",
            "Public result variants remain untouched until successful root "
            "invocation completion publishes the complete result list.",
        ),
        (),
        "value.abi.result.store %v<src>, <slot>",
        ("frame.result_value_slots[slot_u16] = values[src_v8];\npc = pc + 4;"),
    ),
)


def _ref_abi_instruction(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    register_name: str,
    register_role: InstructionFieldRole,
    type_contract: str,
    ownership_policy: RefOwnership,
    packet_contract: str,
    description: str,
    success: tuple[str, ...],
    ownership: tuple[str, ...],
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
            ref_register(
                register_name,
                1,
                register_role,
                "Ref-register side of the overflow transfer.",
                RuntimeRefPolicy(
                    type_contract,
                    RefNullPolicy.NULLABLE,
                    ownership_policy,
                ),
            ),
            _slot(packet_contract),
        ),
        semantics=_semantics(
            description,
            packet_contract,
            f"{register_name} must be a valid ref-register ordinal.",
            success,
            ownership,
            assembly,
            pseudocode,
        ),
    )


REF_ABI_ARGUMENT_LOAD_BORROW = _ref_abi_instruction(
    entity_id="core.instruction.ref.abi.argument.load.borrow",
    summary="Borrows one ref argument overflow slot into a register.",
    opcode=0xC2,
    mnemonic="ref.abi.argument.load.borrow",
    register_name="dst_r8",
    register_role=InstructionFieldRole.RESULT,
    type_contract="signature.argument.ref",
    ownership_policy=RefOwnership.BORROW,
    packet_contract="argument.ref",
    description=(
        "Publishes a non-consuming internal borrow from one argument ref "
        "overflow slot into dst_r8 while leaving the packet source unchanged."
    ),
    success=(
        "A null packet source publishes canonical null; a non-null source "
        "publishes its exact object and descriptor as a borrow.",
    ),
    ownership=(
        "The source is not retained. dst_r8 is replaced only after the source "
        "is copied, and a previous owned destination is then released.",
        "The paused caller and argument packet dominate the borrow's complete "
        "callee lifetime, including nested callee suspension.",
    ),
    assembly="%r<dst> = ref.abi.argument.load.borrow <slot>",
    pseudocode=(
        "source = frame.argument_ref_slots[slot_u16];\n"
        "old_destination = refs[dst_r8];\n"
        "refs[dst_r8] = borrow_ref(source);\n"
        "release_or_clear(old_destination);\n"
        "pc = pc + 4;"
    ),
)

REF_ABI_ARGUMENT_LOAD_MOVE = _ref_abi_instruction(
    entity_id="core.instruction.ref.abi.argument.load.move",
    summary="Consumes one ref argument overflow slot into a register.",
    opcode=0xC3,
    mnemonic="ref.abi.argument.load.move",
    register_name="dst_r8",
    register_role=InstructionFieldRole.RESULT,
    type_contract="signature.argument.ref",
    ownership_policy=RefOwnership.MOVE,
    packet_contract="argument.ref",
    description=(
        "Transfers the complete null, borrowed, or owned state from one "
        "argument ref overflow slot into dst_r8 and clears the packet source."
    ),
    success=(
        "argument_ref_slots[slot_u16] becomes canonical null and dst_r8 "
        "receives its exact prior state. A second move therefore loads null.",
    ),
    ownership=(
        "The source is saved and cleared before the destination is replaced. "
        "An owner transfers one release obligation; an internal borrow remains "
        "borrowed under its existing dominating lifetime; a previous owned "
        "destination is released.",
    ),
    assembly="%r<dst> = ref.abi.argument.load.move <slot>",
    pseudocode=(
        "source = frame.argument_ref_slots[slot_u16];\n"
        "old_destination = refs[dst_r8];\n"
        "refs[dst_r8] = source;\n"
        "frame.argument_ref_slots[slot_u16] = canonical_null_ref;\n"
        "release_or_clear(old_destination);\n"
        "pc = pc + 4;"
    ),
)

REF_ABI_RESULT_STORE_MOVE = _ref_abi_instruction(
    entity_id="core.instruction.ref.abi.result.store.move",
    summary="Publishes one ref register into a result overflow slot.",
    opcode=0xC4,
    mnemonic="ref.abi.result.store.move",
    register_name="src_r8",
    register_role=InstructionFieldRole.OPERAND,
    type_contract="signature.result.ref",
    ownership_policy=RefOwnership.PUBLISH_MOVE,
    packet_contract="result.ref",
    description=(
        "Promotes a borrowed source when necessary, moves the resulting owner "
        "or canonical null into a caller-owned result slot, and clears src_r8."
    ),
    success=(
        "src_r8 becomes canonical null and result_ref_slots[slot_u16] receives "
        "canonical null or an owner-backed copy of the exact source ref.",
        "Repeated stores replace the prior result state. control.return later "
        "validates the complete result packet before publication.",
    ),
    ownership=(
        "A borrowed non-null source is retained and converted to owned before "
        "publication. An owned source transfers its existing obligation. The "
        "old result owner, if any, is released only after the new state is safe.",
    ),
    assembly="ref.abi.result.store.move %r<src>, <slot>",
    pseudocode=(
        "source = refs[src_r8];\n"
        "if (source.is_borrowed) {\n"
        "  retain(source.object, source.descriptor);\n"
        "  source.state = owned;\n"
        "}\n"
        "old_result = frame.result_ref_slots[slot_u16];\n"
        "frame.result_ref_slots[slot_u16] = source;\n"
        "refs[src_r8] = canonical_null_ref;\n"
        "release_or_clear(old_result);\n"
        "pc = pc + 4;"
    ),
)


def _function_abi_instruction(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    register_name: str,
    register_role: InstructionFieldRole,
    packet_contract: str,
    description: str,
    success: tuple[str, ...],
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
            function_register(
                register_name,
                1,
                register_role,
                "Function-register side of the overflow transfer.",
            ),
            _slot(packet_contract),
        ),
        semantics=_semantics(
            description,
            packet_contract,
            f"{register_name} must be a valid function-register ordinal.",
            success,
            (),
            assembly,
            pseudocode,
        ),
    )


FUNC_ABI_ARGUMENT_LOAD = _function_abi_instruction(
    entity_id="core.instruction.func.abi.argument.load",
    summary="Loads one complete function argument overflow carrier.",
    opcode=0xC5,
    mnemonic="func.abi.argument.load",
    register_name="dst_f8",
    register_role=InstructionFieldRole.RESULT,
    packet_contract="argument.function",
    description=(
        "Copies one complete 16-byte function argument overflow slot into dst_f8."
    ),
    success=("dst_f8 receives an exact copy of argument_function_slots[slot_u16].",),
    assembly="%f<dst> = func.abi.argument.load <slot>",
    pseudocode=(
        "functions[dst_f8] = frame.argument_function_slots[slot_u16];\npc = pc + 4;"
    ),
)

FUNC_ABI_RESULT_STORE = _function_abi_instruction(
    entity_id="core.instruction.func.abi.result.store",
    summary="Stores one complete function result overflow carrier.",
    opcode=0xC6,
    mnemonic="func.abi.result.store",
    register_name="src_f8",
    register_role=InstructionFieldRole.OPERAND,
    packet_contract="result.function",
    description=(
        "Copies one complete 16-byte function register into a private result "
        "overflow slot."
    ),
    success=(
        "result_function_slots[slot_u16] receives an exact copy of src_f8; "
        "repeated stores are valid and the last executed store wins.",
        "control.return validates the complete result packet before publication.",
    ),
    assembly="func.abi.result.store %f<src>, <slot>",
    pseudocode=(
        "frame.result_function_slots[slot_u16] = functions[src_f8];\npc = pc + 4;"
    ),
)

INSTRUCTIONS = (
    VALUE_ABI_ARGUMENT_LOAD,
    VALUE_ABI_RESULT_STORE,
    REF_ABI_ARGUMENT_LOAD_BORROW,
    REF_ABI_ARGUMENT_LOAD_MOVE,
    REF_ABI_RESULT_STORE_MOVE,
    FUNC_ABI_ARGUMENT_LOAD,
    FUNC_ABI_RESULT_STORE,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
