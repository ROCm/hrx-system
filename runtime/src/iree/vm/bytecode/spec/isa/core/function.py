# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Non-owning first-class function-value instructions."""

import enum

from iree.vm.bytecode.spec.isa import (
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
    RecordRule,
    StateEffect,
)
from iree.vm.bytecode.spec.isa.core.control import CONTROL_CALL_TARGET_SELECTOR
from iree.vm.bytecode.spec.isa.core.rules import (
    FieldRule,
    RecordRuleKind,
    StateAccess,
    StateResource,
)
from iree.vm.bytecode.spec.schema import U8, U16, Field
from iree.vm.bytecode.spec.version import CORE_0

FUNCTION_FAMILY = InstructionFamily(
    name="function",
    since=CORE_0,
    summary="Non-owning first-class function-value operations.",
    contract=(
        "A function register is one complete 16-byte non-owning, program-bound "
        "function reference. Canonical null is all zero bits. A non-null carrier "
        "contains an absolute linked target, its canonical structural callable token, "
        "and its actual MAY_YIELD fact. Every register and function-local cell begins "
        "canonical null when its frame is created; validated direct arguments then "
        "replace their signature prefix. Value, raw local-byte, and ref operations "
        "cannot address function registers or function locals. Function values "
        "retain neither the program nor the process; containing storage is dominated "
        "by the process retaining "
        "the program. Operations preserve or construct the whole carrier and never "
        "splice its lanes. No retain, release, move, or discard operations are "
        "required. Every operation is infallible after verification and never suspends."
    ),
)


class FunctionStackAccess(enum.Enum):
    LOAD = "load"
    STORE = "store"


def _field(
    name: str, encoding, summary: str, role: FieldRole, rule
) -> InstructionField:
    if not isinstance(rule, FieldRuleUse):
        rule = FieldRuleUse(rule)
    return InstructionField(Field(name, encoding, summary), role, rule)


def _function_register(name: str, role: FieldRole, summary: str) -> InstructionField:
    return _field(name, U8, summary, role, FieldRule.REGISTER_FUNCTION)


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


FUNC_NULL = Instruction(
    opcode=0x20,
    mnemonic="func.null",
    since=CORE_0,
    family=FUNCTION_FAMILY,
    summary="Writes a canonical null function value.",
    fields=(
        _function_register(
            "destination_f8",
            FieldRole.RESULT,
            "Function-register ordinal receiving canonical null.",
        ),
        _padding(U16),
    ),
    semantics=None,
    behavior="Replaces destination_f8 with the all-zero function carrier.",
    success=("All 16 bytes of destination_f8 become zero.",),
    assembly="%f<destination> = func.null",
    pseudocode=("functions[destination_f8] = canonical_null_function;\npc = pc + 4;"),
)

FUNC_COMPARE_NULL = Instruction(
    opcode=0x21,
    mnemonic="func.compare.null",
    since=CORE_0,
    family=FUNCTION_FAMILY,
    summary="Tests a complete function carrier for canonical null.",
    fields=(
        _value_register(
            "destination_v8",
            FieldRole.RESULT,
            "Value-register ordinal receiving canonical zero or one.",
        ),
        _function_register(
            "source_f8",
            FieldRole.OPERAND,
            "Function-register ordinal tested for canonical null.",
        ),
        _padding(),
    ),
    semantics=None,
    behavior="Tests whether both 64-bit lanes of source_f8 are zero.",
    success=(
        "destination_v8 becomes canonical one when source_f8 is null and zero "
        "otherwise.",
    ),
    assembly="%v<destination> = func.compare.null %f<source>",
    pseudocode=(
        "values[destination_v8] = functions[source_f8].is_canonical_null ? 1 : 0;\n"
        "pc = pc + 4;"
    ),
)

FUNC_COPY = Instruction(
    opcode=0x22,
    mnemonic="func.copy",
    since=CORE_0,
    family=FUNCTION_FAMILY,
    summary="Copies one complete function carrier.",
    fields=(
        _function_register(
            "destination_f8",
            FieldRole.RESULT,
            "Function-register ordinal receiving the copied carrier.",
        ),
        _function_register(
            "source_f8",
            FieldRole.OPERAND,
            "Function-register ordinal providing the copied carrier.",
        ),
        _padding(),
    ),
    semantics=None,
    behavior="Reads and copies all 16 bytes; source and destination may alias.",
    success=("destination_f8 receives an exact copy of source_f8.",),
    assembly="%f<destination> = func.copy %f<source>",
    pseudocode=(
        "source = functions[source_f8];\n"
        "functions[destination_f8] = source;\n"
        "pc = pc + 4;"
    ),
)

FUNC_ADDRESS = Instruction(
    opcode=0x23,
    mnemonic="func.address",
    since=CORE_0,
    family=FUNCTION_FAMILY,
    summary="Materializes a verified local or imported function target.",
    fields=(
        _function_register(
            "destination_f8",
            FieldRole.RESULT,
            "Function-register ordinal receiving the materialized target.",
        ),
        _field(
            "target_kind_u8",
            U8,
            "Local, required-import, or optional-import target selector.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.SELECTOR, data=CONTROL_CALL_TARGET_SELECTOR),
        ),
        _padding(),
        _field(
            "target_ordinal_u16",
            U16,
            "Direct ordinal in the selected target table.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.CONSTRAINT_MEMBER,
        ),
        _field(
            "callable_type_ordinal_u16",
            U16,
            "Exact structural callable type expected for the target.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.CONSTRAINT_MEMBER,
        ),
    ),
    semantics=None,
    behavior=(
        "Materializes a complete carrier from immutable linked state. A local target "
        "combines current-module and function ordinals; a present import copies its "
        "resolved target; an absent optional import produces canonical null."
    ),
    success=(
        "A present target publishes its absolute target, canonical callable token, "
        "and actual behavior facts; an absent optional import publishes null.",
        "Execution performs no lookup, signature walk, provider call, allocation, "
        "lock, or ownership operation.",
    ),
    assembly=(
        "%f4 = func.address local @function7, type @callable3\n"
        "%f5 = func.address optional_import @import2, type @callable3"
    ),
    pseudocode=(
        "target = linked_target(target_kind_u8, target_ordinal_u16);\n"
        "functions[destination_f8] = target.is_absent_optional\n"
        "    ? canonical_null_function\n"
        "    : make_function(target, callable_type_ordinal_u16);\n"
        "pc = pc + 8;"
    ),
    rules=(
        RecordRule(
            RecordRuleKind.FUNCTION_ADDRESS,
            (
                "target_kind_u8",
                "target_ordinal_u16",
                "callable_type_ordinal_u16",
            ),
            summary=(
                "The selector and ordinal identify a matching local, required-import, "
                "or optional-import declaration whose structural callable type and "
                "MAY_YIELD behavior satisfy callable_type_ordinal_u16."
            ),
        ),
    ),
)

FUNC_IMPORT_RESOLVED = Instruction(
    opcode=0x24,
    mnemonic="func.import.resolved",
    since=CORE_0,
    family=FUNCTION_FAMILY,
    summary="Tests whether an optional import resolved.",
    fields=(
        _value_register(
            "destination_v8",
            FieldRole.RESULT,
            "Value-register ordinal receiving canonical zero or one.",
        ),
        _field(
            "import_ordinal_u16",
            U16,
            "Direct optional-import ordinal.",
            FieldRole.IMMEDIATE,
            FieldRule.IMPORT_ORDINAL_OPTIONAL,
        ),
    ),
    semantics=None,
    behavior=(
        "Turns expected absence in immutable linked state into ordinary Boolean data; "
        "required imports are rejected because published programs always resolve them."
    ),
    success=(
        "destination_v8 becomes canonical one when the import is resolved and zero "
        "when absent; no status is constructed.",
    ),
    assembly="%v<destination> = func.import.resolved @import2",
    pseudocode=(
        "values[destination_v8] = linked_imports[import_ordinal_u16].is_resolved\n"
        "    ? 1 : 0;\n"
        "pc = pc + 4;"
    ),
)


def _stack_access(opcode: int, access_kind: FunctionStackAccess) -> Instruction:
    is_load = access_kind == FunctionStackAccess.LOAD
    mnemonic = "func.stack.load" if is_load else "func.stack.store"
    register_name = "destination_f8" if is_load else "source_f8"
    register_summary = (
        "Function-register ordinal receiving the loaded local."
        if is_load
        else "Function-register ordinal providing the stored carrier."
    )
    role = FieldRole.RESULT if is_load else FieldRole.OPERAND
    access = StateAccess.READ if is_load else StateAccess.WRITE
    verb = "Loads" if is_load else "Stores"
    behavior = (
        "Copies all 16 bytes from one separate, zero-initialized function-local cell "
        "into the destination register."
        if is_load
        else "Copies all 16 bytes from the source register into one separate "
        "function-local cell."
    )
    success = (
        "destination_f8 receives the complete local carrier; an unwritten local "
        "therefore produces canonical null."
        if is_load
        else "The selected local receives an exact copy of source_f8; replacing a "
        "local has no cleanup effect."
    )
    assignment = (
        "functions[destination_f8] = function_locals[local_ordinal_u16];"
        if is_load
        else "function_locals[local_ordinal_u16] = functions[source_f8];"
    )
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=FUNCTION_FAMILY,
        summary=f"{verb} one complete function-local cell.",
        fields=(
            _function_register(
                register_name,
                role,
                register_summary,
            ),
            _field(
                "local_ordinal_u16",
                U16,
                "Direct function-local cell ordinal.",
                FieldRole.IMMEDIATE,
                FieldRule.FUNCTION_LOCAL_ORDINAL,
            ),
        ),
        semantics=access_kind,
        behavior=behavior,
        success=(success,),
        assembly=(
            "%f<destination> = func.stack.load <local_ordinal>"
            if is_load
            else "func.stack.store %f<source>, <local_ordinal>"
        ),
        pseudocode=f"{assignment}\npc = pc + 4;",
        state_effects=(
            StateEffect(access, StateResource.FRAME_LOCALS, ("local_ordinal_u16",)),
        ),
    )


FUNCTION_INSTRUCTIONS = (
    FUNC_NULL,
    FUNC_COMPARE_NULL,
    FUNC_COPY,
    FUNC_ADDRESS,
    FUNC_IMPORT_RESOLVED,
    _stack_access(0x25, FunctionStackAccess.LOAD),
    _stack_access(0x26, FunctionStackAccess.STORE),
)
