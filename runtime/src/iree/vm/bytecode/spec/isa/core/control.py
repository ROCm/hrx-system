# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block, branch, suspension, and return instructions."""

from __future__ import annotations

import enum

from iree.vm.bytecode.spec.isa import (
    ControlFlow,
    FailureCase,
    FieldRole,
    Instruction,
    InstructionFamily,
    InstructionField,
    RecordRule,
    StateEffect,
    Suspension,
)
from iree.vm.bytecode.spec.isa.core.rules import (
    FieldRule,
    RecordRuleKind,
    StateAccess,
    StateResource,
)
from iree.vm.bytecode.spec.schema import I16, I32, U8, U16, U32, Field
from iree.vm.bytecode.spec.version import CORE_0


class BranchCondition(enum.Enum):
    ALWAYS = "always"
    NONZERO = "nonzero"
    ZERO = "zero"


CONTROL_FAMILY = InstructionFamily(
    name="control",
    since=CORE_0,
    summary="Block, branch, suspension, and return operations.",
    contract=(
        "The first record of every function is control.block. A function never "
        "falls through the end of its bytecode: every sequential continuation "
        "has a following decoded record. Direct targets are signed four-byte-word "
        "displacements from the end of their record. Verification computes targets "
        "in widened signed arithmetic, rejects overflow or out-of-function addresses, "
        "and requires an exact decoded control.block target. Execution trusts the "
        "verified target without repeating those checks. Unreachable structurally "
        "valid records are permitted."
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
        "Direct value cells and function carriers copy exactly; owned refs move and "
        "escaping borrowed refs are retained before publication.",
        "The frame releases all remaining owned refs and is popped.",
    ),
    assembly="control.return",
    pseudocode=(
        "validate_complete_result_packet(frame);\n"
        "publish_results(frame, parent_or_root);\n"
        "release_remaining_frame_owners(frame);\n"
        "pop_frame_and_complete_or_resume_parent();"
    ),
    rules=(RecordRule(RecordRuleKind.RETURN_SIGNATURE),),
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
        "Result publication promotes every escaping borrow to an owner; frame "
        "cleanup releases all other owned refs.",
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
        behavior = "Unconditionally transfers to the verified direct target."
        success = "The program counter becomes the verified direct target."
        control_flow = ControlFlow.BRANCH
        pseudocode = "pc = direct_target(record_end, target_word_offset);"
        assembly = f"{mnemonic} ^bb8"
    else:
        comparison = "nonzero" if condition == BranchCondition.NONZERO else "zero"
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
        summary=behavior,
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
            FieldRule.ANY_BITS,
        ),
        _field(
            "target_base_u32",
            U32,
            "First entry in the function-local target table.",
            FieldRole.CONSTRAINT_MEMBER,
            FieldRule.ANY_BITS,
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
        ),
    ),
    control_flow=ControlFlow.SWITCH,
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
)
