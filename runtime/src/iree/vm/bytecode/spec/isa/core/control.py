# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block, branch, call, suspension, return, and failure instructions."""

from __future__ import annotations

import enum

from iree.vm.bytecode.spec.isa import (
    ControlFlow,
    FailureCase,
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
    RecordRule,
    RuntimeRefPolicy,
    StateEffect,
    Suspension,
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
    I32,
    U8,
    U16,
    U32,
    Field,
    NumericKind,
    NumericTable,
    NumericValue,
)
from iree.vm.bytecode.spec.version import CORE_0


class BranchCondition(enum.Enum):
    ALWAYS = "always"
    NONZERO = "nonzero"
    ZERO = "zero"


def _selector(
    name: str, summary: str, values: tuple[tuple[str, int, str], ...]
) -> NumericTable:
    return NumericTable(
        name,
        U8,
        NumericKind.SELECTOR,
        tuple(
            NumericValue(value_name, value, CORE_0, meaning)
            for value_name, value, meaning in values
        ),
        CORE_0,
        summary,
    )


CONTROL_CALL_TARGET_SELECTOR = _selector(
    "control.call.target",
    "Selects the ordinal table used by a direct call or function address.",
    (
        ("local", 0, "The ordinal names a function in the current module."),
        (
            "required_import",
            1,
            "The ordinal names an import that must resolve during program creation.",
        ),
        (
            "optional_import",
            2,
            "The ordinal names an import whose absence is permitted.",
        ),
    ),
)

CONTROL_STATUS_SELECTOR = _selector(
    "control.status",
    "Selects one non-OK architectural status for control.fail.",
    tuple(
        (name, value, f"Produces the canonical {name} status code.")
        for name, value in (
            ("cancelled", 1),
            ("unknown", 2),
            ("invalid_argument", 3),
            ("deadline_exceeded", 4),
            ("not_found", 5),
            ("already_exists", 6),
            ("permission_denied", 7),
            ("resource_exhausted", 8),
            ("failed_precondition", 9),
            ("aborted", 10),
            ("out_of_range", 11),
            ("unimplemented", 12),
            ("internal", 13),
            ("unavailable", 14),
            ("data_loss", 15),
            ("unauthenticated", 16),
            ("incompatible", 18),
        )
    ),
)


CONTROL_FAMILY = InstructionFamily(
    name="control",
    since=CORE_0,
    summary="Block, branch, call, suspension, return, and failure operations.",
    contract=(
        "The first record of every function is control.block. A function never "
        "falls through the end of its bytecode: every sequential continuation "
        "has a following decoded record. Direct targets are signed four-byte-word "
        "displacements from the end of their record. Verification computes targets "
        "in widened signed arithmetic, rejects overflow or out-of-function addresses, "
        "and requires an exact decoded control.block target. Execution trusts the "
        "verified target without repeating those checks. Calls use the target "
        "signature's value, ref, and function prefixes plus its canonical overflow "
        "packet. Every call preflights all fallible checks before moving arguments, "
        "changing packet or frame state, or entering a provider. The parent remains "
        "parked on the call record until successful return publishes results. Any "
        "status failure is terminal to the invocation attempt and enters ordinary "
        "unwind; broader process state after failure is unspecified. Unreachable "
        "structurally valid records are permitted."
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


def _padding(encoding=U8, *, element_count: int = 1) -> InstructionField:
    return _field(
        f"zero_padding_{encoding.name}",
        encoding,
        "Canonical zero padding.",
        FieldRole.PADDING,
        FieldRule.ZERO,
        element_count=element_count,
    )


def _target(bit_width: int) -> InstructionField:
    encoding = I16 if bit_width == 16 else I32
    rule = (
        FieldRule.CONTROL_TARGET_S16
        if bit_width == 16
        else FieldRule.CONTROL_TARGET_S32
    )
    return _field(
        f"target_word_offset_s{bit_width}",
        encoding,
        "Signed four-byte-word displacement from the end of this record.",
        FieldRole.IMMEDIATE,
        rule,
    )


CONTROL_BLOCK = Instruction(
    opcode=0x01,
    mnemonic="control.block",
    since=CORE_0,
    family=CONTROL_FAMILY,
    summary="Marks the only legal direct control target.",
    fields=(_padding(element_count=3),),
    semantics=None,
    behavior=(
        "Marks a legal branch, switch, or yield destination without carrying a "
        "serialized block ordinal."
    ),
    success=("The program counter advances by four bytes.",),
    assembly="^bb3:\n  control.block",
    pseudocode="pc = pc + 4;",
    control_flow=ControlFlow.BLOCK,
)

CONTROL_RETURN = Instruction(
    opcode=0x02,
    mnemonic="control.return",
    since=CORE_0,
    family=CONTROL_FAMILY,
    summary="Validates, publishes, and returns function results.",
    fields=(_padding(element_count=3),),
    semantics=None,
    behavior=(
        "Completes the current function by validating every direct and overflow "
        "result against its private signature, transactionally publishing results, "
        "cleaning the frame, and returning to the parked parent or completing the "
        "root invocation."
    ),
    success=(
        "All result validation completes before caller-visible mutation.",
        "Direct value cells and function carriers copy exactly. Owned direct refs "
        "move; borrowed direct refs are retained and moved as owners; their sources "
        "become null and replaced caller owners are released.",
        "Overflow results remain in the already validated caller packet.",
        "Every remaining owned ref in the exact frame extent is released and the "
        "frame is popped.",
        "A nested parent advances past its parked call. A root invocation publishes "
        "its complete public result list and completes with OK.",
    ),
    assembly="control.return",
    pseudocode=(
        "validate_complete_result_packet(frame);\n"
        "publish_results(frame, parent_or_root);\n"
        "release_remaining_frame_owners(frame);\n"
        "pop_frame_and_complete_or_resume_parent();"
    ),
    control_flow=ControlFlow.RETURN,
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.INVOCATION_RESULTS),
        StateEffect(StateAccess.WRITE, StateResource.INVOCATION_RESULTS),
    ),
    preconditions=(
        "Every ref result satisfies its exact type and nullability contract.",
        "Every function result is canonical null or satisfies its callable contract.",
    ),
    failures=(
        FailureCase(
            "invalid_argument",
            "A ref or function result violates its private signature.",
            "No caller-visible result is changed before ordinary unwind.",
        ),
    ),
    ownership=(
        "Result publication promotes every escaping borrow to an owner. Frame "
        "cleanup releases all other owned refs; function values own nothing.",
    ),
)

CONTROL_YIELD_S32 = Instruction(
    opcode=0x03,
    mnemonic="control.yield.s32",
    since=CORE_0,
    family=CONTROL_FAMILY,
    summary="Suspends with one explicit wide continuation target.",
    fields=(_padding(element_count=3), _target(32)),
    semantics=None,
    behavior=(
        "Ends the physical block, durably records an explicit continuation block, "
        "publishes the suspended invocation, and invokes the optional level-triggered "
        "wake callback. There is no implicit next-record continuation."
    ),
    success=(
        "The frame program counter becomes the verified target before suspension is "
        "published and drive returns the non-status SUSPENDED outcome.",
    ),
    assembly="control.yield.s32 ^bb7",
    pseudocode=(
        "if (invocation_has_borrowed_public_arguments) fail(failed_precondition);\n"
        "frame.pc = direct_target(record_end, target_word_offset_s32);\n"
        "publish_suspended_invocation();\n"
        "invoke_wake_callback_if_nonnull();\n"
        "return SUSPENDED;"
    ),
    control_flow=ControlFlow.YIELD,
    suspension=Suspension.ALWAYS,
    preconditions=("The invocation has no non-null borrowed public ref argument.",),
    failures=(
        FailureCase(
            "failed_precondition",
            "The invocation has a non-null borrowed public ref argument.",
            "The program counter, suspension state, and wake state remain unchanged.",
        ),
    ),
    ownership=("Every frame-local ref remains live across suspension.",),
)


def _branch(
    opcode: int,
    condition: BranchCondition,
    bit_width: int,
) -> Instruction:
    mnemonic_component = {
        BranchCondition.ALWAYS: "",
        BranchCondition.NONZERO: ".if",
        BranchCondition.ZERO: ".unless",
    }[condition]
    mnemonic = f"control.branch{mnemonic_component}.s{bit_width}"
    fields: list[InstructionField] = []
    if condition == BranchCondition.ALWAYS:
        fields.append(_padding(element_count=3 if bit_width == 32 else 1))
    else:
        fields.append(
            _field(
                "condition_v8",
                U8,
                "Complete 64-bit branch condition value-register ordinal.",
                FieldRole.OPERAND,
                FieldRule.REGISTER_VALUE,
            )
        )
        if bit_width == 32:
            fields.append(_padding(U16))
    fields.append(_target(bit_width))

    if condition == BranchCondition.ALWAYS:
        summary = f"Branches unconditionally through an s{bit_width} displacement."
        behavior = "Unconditionally transfers to the verified direct target."
        success = "The program counter becomes the verified direct target."
        control_flow = ControlFlow.BRANCH
        pseudocode = "pc = direct_target(record_end, target_word_offset);"
        assembly = f"{mnemonic} ^bb8"
    else:
        comparison = "nonzero" if condition == BranchCondition.NONZERO else "zero"
        summary = f"Branches through an s{bit_width} displacement on {comparison}."
        behavior = (
            f"Transfers to the verified target when the complete condition is "
            f"{comparison}; otherwise continues at the following record."
        )
        success = "The program counter becomes the target or following record."
        control_flow = ControlFlow.CONDITIONAL_BRANCH
        pseudocode = (
            f"pc = condition_is_{comparison}(values[condition_v8])\n"
            "    ? direct_target(record_end, target_word_offset) : record_end;"
        )
        assembly = f"{mnemonic} %v<condition>, ^bb8"
    return Instruction(
        opcode=opcode,
        mnemonic=mnemonic,
        since=CORE_0,
        family=CONTROL_FAMILY,
        summary=summary,
        fields=tuple(fields),
        semantics=condition,
        behavior=behavior,
        success=(success,),
        assembly=assembly,
        pseudocode=pseudocode,
        control_flow=control_flow,
    )


CONTROL_SWITCH = Instruction(
    opcode=0x0A,
    mnemonic="control.switch",
    since=CORE_0,
    family=CONTROL_FAMILY,
    summary="Branches through a function-local dense target table.",
    fields=(
        _field(
            "selector_v8",
            U8,
            "Unsigned complete-cell zero-based table selector.",
            FieldRole.OPERAND,
            FieldRule.REGISTER_VALUE,
        ),
        _field(
            "target_count_u16",
            U16,
            "Number of entries in the function-local target-table slice.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.CONSTRAINT_MEMBER,
        ),
        _field(
            "target_base_u32",
            U32,
            "First entry in the function-local target table.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.CONSTRAINT_MEMBER,
        ),
    ),
    semantics=None,
    behavior=(
        "Treats the complete selector cell as unsigned, selects a verified "
        "function-local target when it is in range, and otherwise takes the "
        "sequential default path."
    ),
    success=(
        "An in-range selector branches to its table entry; an out-of-range selector "
        "advances by eight bytes.",
    ),
    assembly="control.switch %v<selector>, targets[<base>...<base+count>]",
    pseudocode=(
        "selector = bits_u64(values[selector_v8]);\n"
        "pc = selector < target_count_u16\n"
        "    ? function_target(target_base_u32 + selector) : record_end;"
    ),
    rules=(
        RecordRule(
            RecordRuleKind.SWITCH_TARGETS,
            ("target_count_u16", "target_base_u32"),
            summary=(
                "The base and count form an in-range function-local target slice, "
                "every entry names a decoded control.block, and a sequential default "
                "successor follows this record."
            ),
        ),
    ),
    control_flow=ControlFlow.SWITCH,
)

CONTROL_CALL = Instruction(
    opcode=0x0B,
    mnemonic="control.call",
    since=CORE_0,
    family=CONTROL_FAMILY,
    summary="Calls a local or linked imported function.",
    fields=(
        _field(
            "target_kind_u8",
            U8,
            "Local, required-import, or optional-import target selector.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.SELECTOR, data=CONTROL_CALL_TARGET_SELECTOR),
        ),
        _field(
            "target_ordinal_u16",
            U16,
            "Direct ordinal in the selected target table.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.CONSTRAINT_MEMBER,
        ),
        _field(
            "direct_ref_move_mask_u16",
            U16,
            "Ownership-transfer bits for direct ref arguments.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.CONSTRAINT_MEMBER,
        ),
        _padding(U16),
    ),
    semantics=None,
    behavior=(
        "Invokes a function in the current linked module or its resolved-import "
        "table using the target's exact three-bank callable packet."
    ),
    success=(
        "Direct values and function carriers copy into the child prefixes; each "
        "clear ref-mask bit borrows and each set bit moves the complete ref state.",
        "The parked parent advances by eight bytes only after successful return "
        "transactionally publishes results.",
    ),
    assembly=(
        "control.call @function7 {direct_ref_move_mask = 0x0005}\n"
        "control.call @import3 {direct_ref_move_mask = 0x0000}"
    ),
    pseudocode=(
        "target = preflight_linked_target(target_kind_u8, target_ordinal_u16);\n"
        "validate_borrow_and_argument_contracts(target);\n"
        "dispatch_preflighted_call(target, call_packet, direct_ref_move_mask_u16);"
    ),
    rules=(
        RecordRule(
            RecordRuleKind.CALL,
            (
                "target_kind_u8",
                "target_ordinal_u16",
                "direct_ref_move_mask_u16",
            ),
            summary=(
                "The selector and ordinal identify a matching local, required-import, "
                "or optional-import declaration; the move mask fits the direct ref "
                "arguments; caller registers and the outgoing packet cover the exact "
                "callable contract; and a sequential success continuation follows."
            ),
        ),
    ),
    control_flow=ControlFlow.CALL,
    suspension=Suspension.TARGET_DEPENDENT,
    state_effects=(StateEffect(StateAccess.UNKNOWN, StateResource.ANY),),
    preconditions=(
        "The selected target exists and its implementation can be entered.",
        "A target that may yield sees no non-null borrowed public invocation ref.",
        "Every direct and overflow ref or function argument satisfies the target's "
        "private callable contract.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "A yielding target is entered from an invocation with borrowed public refs.",
            "No argument, packet, frame, or provider state changes.",
        ),
        FailureCase(
            "invalid_argument",
            "A ref or function argument violates the target contract.",
            "No argument, packet, frame, or provider state changes.",
        ),
        FailureCase(
            "not_found",
            "The selected optional import is unresolved.",
            "No argument, packet, frame, or provider state changes.",
        ),
        FailureCase(
            "resource_exhausted",
            "A bytecode target's fixed frame does not fit the invocation stack.",
            "Target construction fails before argument moves or external work.",
        ),
        FailureCase(
            "target's non-OK status",
            "The entered target returns non-OK.",
            "The invocation unwinds and public result storage remains untouched.",
        ),
    ),
    ownership=(
        "Direct ref moves transfer their complete null, borrowed, or owned state. "
        "Borrows remain dominated by the parked parent; escaping results are promoted "
        "to owners during result publication.",
    ),
)

CONTROL_CALL_INDIRECT = Instruction(
    opcode=0x0C,
    mnemonic="control.call.indirect",
    since=CORE_0,
    family=CONTROL_FAMILY,
    summary="Calls a dynamic first-class function value.",
    fields=(
        _field(
            "target_f8",
            U8,
            "Dynamic first-class function target.",
            FieldRole.OPERAND,
            FieldRule.REGISTER_FUNCTION,
        ),
        _field(
            "callable_type_ordinal_u16",
            U16,
            "Exact expected structural callable type.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.CONSTRAINT_MEMBER,
        ),
        _field(
            "direct_ref_move_mask_u16",
            U16,
            "Ownership-transfer bits for direct ref arguments.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.CONSTRAINT_MEMBER,
        ),
        _padding(U16),
    ),
    semantics=None,
    behavior=(
        "Invokes a non-null function carrier in the current program after validating "
        "it against the encoded structural callable type."
    ),
    success=(
        "Argument transfer, target entry, suspension, and return publication use the "
        "same absolute-target path as a direct call.",
        "The parked parent advances by eight bytes only after successful return.",
    ),
    assembly=(
        "control.call.indirect %f4, type @callable3 {direct_ref_move_mask = 0x0002}"
    ),
    pseudocode=(
        "target = preflight_function_value(\n"
        "    functions[target_f8], callable_type_ordinal_u16);\n"
        "validate_borrow_and_argument_contracts(target);\n"
        "dispatch_preflighted_call(target, call_packet, direct_ref_move_mask_u16);"
    ),
    rules=(
        RecordRule(
            RecordRuleKind.CALL_INDIRECT,
            (
                "target_f8",
                "callable_type_ordinal_u16",
                "direct_ref_move_mask_u16",
            ),
            summary=(
                "The callable ordinal exists, the move mask fits its direct ref "
                "arguments, caller registers and the outgoing packet cover the exact "
                "contract, and a sequential success continuation follows."
            ),
        ),
    ),
    control_flow=ControlFlow.CALL,
    suspension=Suspension.TARGET_DEPENDENT,
    state_effects=(StateEffect(StateAccess.UNKNOWN, StateResource.ANY),),
    preconditions=(
        "The function carrier is non-null, belongs to the current program, names an "
        "in-range target, and carries the expected canonical callable token.",
        "Its actual suspension behavior is permitted and every ref or function "
        "argument satisfies the encoded callable contract.",
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "The target is null or a yielding target sees borrowed public refs.",
            "No argument, packet, frame, or provider state changes.",
        ),
        FailureCase(
            "invalid_argument",
            "Program identity, target bounds, callable token, suspension permission, "
            "or an argument contract is invalid.",
            "No argument, packet, frame, or provider state changes.",
        ),
        FailureCase(
            "resource_exhausted",
            "A bytecode target's fixed frame does not fit the invocation stack.",
            "Target construction fails before argument moves or external work.",
        ),
        FailureCase(
            "target's non-OK status",
            "The entered target returns non-OK.",
            "The invocation unwinds and public result storage remains untouched.",
        ),
    ),
    ownership=(
        "Direct ref transfer and result borrow promotion match control.call; function "
        "values themselves own nothing.",
    ),
)


def _diagnostic_message() -> InstructionField:
    return InstructionField(
        Field(
            "message_r8_nullable",
            U8,
            "Optional best-effort readable vm.buffer diagnostic message.",
        ),
        FieldRole.OPERAND,
        FieldRuleUse(FieldRule.REGISTER_REF),
        RuntimeRefPolicy(
            "vm.buffer",
            RefNullPolicy.NULLABLE,
            RefOwnership.DIAGNOSTIC_BORROW,
        ),
    )


CONTROL_ASSERT = Instruction(
    opcode=0x0D,
    mnemonic="control.assert",
    since=CORE_0,
    family=CONTROL_FAMILY,
    summary="Continues on true and fails with failed_precondition on false.",
    fields=(
        _field(
            "condition_v8",
            U8,
            "Complete 64-bit assertion condition value-register ordinal.",
            FieldRole.OPERAND,
            FieldRule.REGISTER_VALUE,
        ),
        _diagnostic_message(),
        _padding(),
    ),
    semantics=None,
    behavior=(
        "Continues when the complete condition cell is nonzero and otherwise "
        "terminates the invocation with failed_precondition and optional best-effort "
        "diagnostic bytes."
    ),
    success=(
        "A nonzero condition advances by four bytes without reading or mapping the "
        "diagnostic message.",
    ),
    assembly="control.assert %v4, %r2",
    pseudocode=(
        "if (values[condition_v8] != 0) {\n"
        "  pc = pc + 4;\n"
        "} else {\n"
        "  fail(failed_precondition, optional_message(message_r8_nullable));\n"
        "}"
    ),
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.BUFFER, ("message_r8_nullable",)),
    ),
    failures=(
        FailureCase(
            "failed_precondition",
            "condition_v8 contains the complete 64-bit zero value.",
            "Diagnostic context is captured before unwind; public results remain "
            "untouched.",
        ),
    ),
    ownership=(
        "The optional message is a diagnostic-only borrow and is not retained on the "
        "successful path. Wrong-type, unreadable, uncopyable, or unformattable "
        "diagnostic data is omitted without changing the failure status.",
    ),
)

CONTROL_FAIL = Instruction(
    opcode=0x0E,
    mnemonic="control.fail",
    since=CORE_0,
    family=CONTROL_FAMILY,
    summary="Terminates with one immediate architectural status.",
    fields=(
        _field(
            "status_u8",
            U8,
            "Assigned non-OK architectural status selector.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.SELECTOR, data=CONTROL_STATUS_SELECTOR),
        ),
        _diagnostic_message(),
        _padding(),
    ),
    semantics=None,
    behavior=(
        "Unconditionally terminates the invocation with the selected status and "
        "optional best-effort diagnostic bytes."
    ),
    success=(),
    assembly="control.fail invalid_argument, %r2",
    pseudocode=(
        "fail(status_from_architecture_selector(status_u8),\n"
        "     optional_message(message_r8_nullable));"
    ),
    control_flow=ControlFlow.FAIL,
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.BUFFER, ("message_r8_nullable",)),
    ),
    failures=(
        FailureCase(
            "status_u8",
            "The instruction is executed.",
            "Diagnostic context is captured before all frames unwind; public results "
            "remain untouched.",
        ),
    ),
    ownership=(
        "The optional message is a diagnostic-only borrow. Wrong-type, unreadable, "
        "uncopyable, or unformattable diagnostic data is omitted without changing "
        "the selected status; unwind releases all owned refs through their private "
        "no-status teardown paths.",
    ),
)

CONTROL_INSTRUCTIONS = (
    CONTROL_BLOCK,
    CONTROL_RETURN,
    CONTROL_YIELD_S32,
    _branch(0x04, BranchCondition.ALWAYS, 16),
    _branch(0x05, BranchCondition.ALWAYS, 32),
    _branch(0x06, BranchCondition.NONZERO, 16),
    _branch(0x07, BranchCondition.NONZERO, 32),
    _branch(0x08, BranchCondition.ZERO, 16),
    _branch(0x09, BranchCondition.ZERO, 32),
    CONTROL_SWITCH,
    CONTROL_CALL,
    CONTROL_CALL_INDIRECT,
    CONTROL_ASSERT,
    CONTROL_FAIL,
)
