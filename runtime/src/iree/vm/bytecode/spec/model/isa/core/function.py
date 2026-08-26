# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 first-class function-value instructions."""

from __future__ import annotations

from model.isa import (
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
    StateResource,
)
from model.isa.declarations import (
    core_instruction,
    function_register,
    instruction_field,
    state_read,
    state_write,
    value_register,
    zero_padding,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import (
    CONSTRAINT_MEMBER,
    FUNCTION_ADDRESS,
    FUNCTION_LOCAL_ORDINAL,
    IMPORT_ORDINAL_OPTIONAL,
    SELECTOR,
    ZERO,
)
from model.schema import U8, U16, EntityReference, FieldReference, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.function",
    since=CORE_0,
    summary="Non-owning first-class function-value operations.",
    dependencies=("core.contract.machine",),
    document_order=6,
    normative_text=(
        "A function register is one complete 16-byte non-owning, program-bound "
        "function reference. Canonical null is all zero bits. Every function "
        "register and function-local cell begins canonical null when its frame "
        "is created; validated direct arguments then replace their signature "
        "prefix. A non-null value contains an absolute linked target, its "
        "canonical structural callable token, and its actual MAY_YIELD fact. "
        "Function values retain neither "
        "the program nor process; storage containing one is dominated by the "
        "process that retains the program. Operations preserve or construct "
        "the complete carrier and never splice its lanes. Function values need "
        "no retain, release, move, or discard operations because they own "
        "nothing. Every operation in this family is infallible after module "
        "verification and never suspends."
    ),
)


def _semantics(
    description: str,
    verification: tuple[str, ...],
    success: tuple[str, ...],
    assembly: tuple[str, ...],
    pseudocode: str,
    *,
    byte_length: int = 4,
) -> InstructionSemantics:
    return InstructionSemantics(
        description=description,
        verification=verification,
        preconditions=(),
        success=(
            *success,
            f"The program counter advances by {byte_length} bytes.",
        ),
        failures=(),
        ownership=(),
        assembly=assembly,
        pseudocode=pseudocode,
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


FUNC_NULL = core_instruction(
    entity_id="core.instruction.func.null",
    since=CORE_0,
    summary="Writes a canonical null function value.",
    opcode=0x20,
    mnemonic="func.null",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        function_register(
            "dst_f8",
            1,
            InstructionFieldRole.RESULT,
            "Destination function-register ordinal.",
        ),
        _zero_u16(),
    ),
    state_effects=(),
    semantics=_semantics(
        "Replaces dst_f8 with the all-zero canonical null function carrier.",
        (
            "dst_f8 must be a valid function-register ordinal.",
            "zero_padding_u16 must equal zero.",
        ),
        ("All 16 bytes of dst_f8 become zero.",),
        ("%f2 = func.null",),
        "functions[dst_f8] = canonical_null_function;\npc = pc + 4;",
    ),
)

FUNC_COMPARE_NULL = core_instruction(
    entity_id="core.instruction.func.compare.null",
    since=CORE_0,
    summary="Tests a complete function carrier for canonical null.",
    opcode=0x21,
    mnemonic="func.compare.null",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Destination Boolean value-register ordinal.",
        ),
        function_register(
            "src_f8",
            2,
            InstructionFieldRole.OPERAND,
            "Source function-register ordinal.",
        ),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    state_effects=(),
    semantics=_semantics(
        "Tests whether both 64-bit lanes of src_f8 are zero.",
        (
            "dst_v8 and src_f8 must be valid registers in their respective banks.",
            "zero_padding_u8 must equal zero.",
        ),
        (
            "dst_v8 becomes canonical value one when src_f8 is canonical null "
            "and canonical value zero otherwise.",
        ),
        ("%v1 = func.compare.null %f2",),
        ("values[dst_v8] = functions[src_f8].is_canonical_null ? 1 : 0;\npc = pc + 4;"),
    ),
)

FUNC_COPY = core_instruction(
    entity_id="core.instruction.func.copy",
    since=CORE_0,
    summary="Copies one complete function carrier.",
    opcode=0x22,
    mnemonic="func.copy",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        function_register(
            "dst_f8",
            1,
            InstructionFieldRole.RESULT,
            "Destination function-register ordinal.",
        ),
        function_register(
            "src_f8",
            2,
            InstructionFieldRole.OPERAND,
            "Source function-register ordinal.",
        ),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    state_effects=(),
    semantics=_semantics(
        "Copies all 16 bytes of src_f8 to dst_f8; the registers may alias.",
        (
            "dst_f8 and src_f8 must be valid function-register ordinals.",
            "zero_padding_u8 must equal zero.",
        ),
        ("dst_f8 receives an exact copy of the complete src_f8 carrier.",),
        ("%f5 = func.copy %f2",),
        ("source = functions[src_f8];\nfunctions[dst_f8] = source;\npc = pc + 4;"),
    ),
)

_CALL_TARGET_SELECTOR = SELECTOR_TABLES_BY_NAME["control.call.target"]
FUNC_ADDRESS = core_instruction(
    entity_id="core.instruction.func.address",
    since=CORE_0,
    summary="Materializes a verified local or imported function target.",
    opcode=0x23,
    mnemonic="func.address",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        function_register(
            "dst_f8",
            1,
            InstructionFieldRole.RESULT,
            "Destination function-register ordinal.",
        ),
        instruction_field(
            "target_kind_u8",
            2,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Local, required-import, or optional-import target kind.",
            (
                RuleUse(
                    SELECTOR.entity_id,
                    (EntityReference(_CALL_TARGET_SELECTOR.entity_id),),
                ),
            ),
        ),
        zero_padding("zero_padding_u8", 3, 1),
        instruction_field(
            "target_ordinal_u16",
            4,
            U16.entity_id,
            InstructionFieldRole.CONSTRAINT_MEMBER,
            "Direct ordinal in the domain selected by target_kind_u8.",
            (RuleUse(CONSTRAINT_MEMBER.entity_id, ("func.address",)),),
        ),
        instruction_field(
            "callable_type_ordinal_u16",
            6,
            U16.entity_id,
            InstructionFieldRole.CONSTRAINT_MEMBER,
            "Exact structural callable type expected for the target.",
            (RuleUse(CONSTRAINT_MEMBER.entity_id, ("func.address",)),),
        ),
    ),
    constraints=(
        RuleUse(
            FUNCTION_ADDRESS.entity_id,
            (
                FieldReference("target_kind_u8"),
                FieldReference("target_ordinal_u16"),
                FieldReference("callable_type_ordinal_u16"),
            ),
        ),
    ),
    state_effects=(),
    semantics=_semantics(
        (
            "Materializes a complete function carrier from immutable linked "
            "program state. A local target combines the current linked-module "
            "ordinal with its module-local function ordinal; a present import "
            "copies its resolved target. An absent optional import produces "
            "canonical null."
        ),
        (
            "dst_f8 must be a valid function-register ordinal and "
            "zero_padding_u8 must equal zero.",
            "target_kind_u8 and target_ordinal_u16 must name an existing local, "
            "required-import, or optional-import declaration with matching "
            "optionality.",
            "callable_type_ordinal_u16 must exactly match the selected "
            "declaration, including its permission for the target's actual "
            "MAY_YIELD fact.",
        ),
        (
            "A local or present imported target publishes its absolute linked "
            "target, canonical callable token, and actual behavior facts to dst_f8.",
            "An unresolved optional import publishes canonical null to dst_f8.",
            "Execution performs no lookup, signature walk, provider call, "
            "allocation, lock, or ownership operation.",
        ),
        (
            "%f4 = func.address local @function7, type @callable3",
            "%f5 = func.address optional_import @import2, type @callable3",
        ),
        (
            "target = linked_target(target_kind_u8, target_ordinal_u16);\n"
            "functions[dst_f8] = target.is_absent_optional\n"
            "    ? canonical_null_function\n"
            "    : make_function(target, callable_type_ordinal_u16);\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

FUNC_IMPORT_RESOLVED = core_instruction(
    entity_id="core.instruction.func.import.resolved",
    since=CORE_0,
    summary="Tests whether an optional import resolved.",
    opcode=0x24,
    mnemonic="func.import.resolved",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Destination Boolean value-register ordinal.",
        ),
        instruction_field(
            "import_ordinal_u16",
            2,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Direct optional-import ordinal.",
            (RuleUse(IMPORT_ORDINAL_OPTIONAL.entity_id),),
        ),
    ),
    state_effects=(),
    semantics=_semantics(
        (
            "Reads immutable linked state for an optional function import and "
            "turns expected absence into ordinary Boolean data."
        ),
        (
            "dst_v8 must be a valid value-register ordinal.",
            "import_ordinal_u16 must name an OPTIONAL import declaration; "
            "required imports are rejected because they are always present in "
            "a published program.",
        ),
        (
            "dst_v8 becomes canonical value one when the import is resolved "
            "and canonical value zero when it is absent.",
            "Expected absence does not construct an iree_status_t.",
        ),
        ("%v0 = func.import.resolved @import2",),
        (
            "values[dst_v8] = linked_imports[import_ordinal_u16].is_resolved\n"
            "    ? 1 : 0;\n"
            "pc = pc + 4;"
        ),
    ),
)


def _local_ordinal():
    return instruction_field(
        "local_ordinal_u16",
        2,
        U16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Direct function-local cell ordinal.",
        (RuleUse(FUNCTION_LOCAL_ORDINAL.entity_id),),
    )


FUNC_STACK_LOAD = core_instruction(
    entity_id="core.instruction.func.stack.load",
    since=CORE_0,
    summary="Loads one complete function-local cell.",
    opcode=0x25,
    mnemonic="func.stack.load",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        function_register(
            "dst_f8",
            1,
            InstructionFieldRole.RESULT,
            "Destination function-register ordinal.",
        ),
        _local_ordinal(),
    ),
    state_effects=(state_read(StateResource.FRAME_LOCALS, "local_ordinal_u16"),),
    semantics=_semantics(
        (
            "Copies one complete zero-initialized 16-byte function-local cell "
            "into dst_f8."
        ),
        (
            "dst_f8 must be a valid function-register ordinal.",
            "local_ordinal_u16 must be less than local_function_count.",
        ),
        (
            "dst_f8 receives the complete local carrier; an unwritten local "
            "therefore produces canonical null.",
        ),
        ("%f3 = func.stack.load 5",),
        ("functions[dst_f8] = function_locals[local_ordinal_u16];\npc = pc + 4;"),
    ),
)

FUNC_STACK_STORE = core_instruction(
    entity_id="core.instruction.func.stack.store",
    since=CORE_0,
    summary="Stores one complete function carrier into a local cell.",
    opcode=0x26,
    mnemonic="func.stack.store",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        function_register(
            "src_f8",
            1,
            InstructionFieldRole.OPERAND,
            "Source function-register ordinal.",
        ),
        _local_ordinal(),
    ),
    state_effects=(state_write(StateResource.FRAME_LOCALS, "local_ordinal_u16"),),
    semantics=_semantics(
        "Copies all 16 bytes of src_f8 into one function-local cell.",
        (
            "src_f8 must be a valid function-register ordinal.",
            "local_ordinal_u16 must be less than local_function_count.",
        ),
        (
            "The selected local receives an exact copy of src_f8; replacing a "
            "local has no cleanup effect.",
        ),
        ("func.stack.store %f3, 5",),
        ("function_locals[local_ordinal_u16] = functions[src_f8];\npc = pc + 4;"),
    ),
)

INSTRUCTIONS = (
    FUNC_NULL,
    FUNC_COMPARE_NULL,
    FUNC_COPY,
    FUNC_ADDRESS,
    FUNC_IMPORT_RESOLVED,
    FUNC_STACK_LOAD,
    FUNC_STACK_STORE,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
