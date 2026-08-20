# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 control-flow, call, suspension, and failure instructions."""

from __future__ import annotations

from model.isa import (
    ControlFlow,
    FailureCase,
    Instruction,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
    RefNullPolicy,
    RefOwnership,
    RuntimeRefPolicy,
    Suspension,
)
from model.isa.declarations import (
    core_instruction,
    function_register,
    instruction_field,
    ref_register,
    value_register,
    zero_padding,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import (
    CONSTRAINT_MEMBER,
    CONTROL_RETURN_SIGNATURE,
    CONTROL_SWITCH_TARGETS,
    CONTROL_TARGET_RELATIVE_S16,
    CONTROL_TARGET_RELATIVE_S32,
    SELECTOR,
    ZERO,
)
from model.isa.validation import (
    CONTROL_CALL as CONTROL_CALL_RULE,
)
from model.isa.validation import (
    CONTROL_CALL_INDIRECT as CONTROL_CALL_INDIRECT_RULE,
)
from model.schema import (
    I16,
    I32,
    U8,
    U16,
    U32,
    EntityReference,
    FieldReference,
    RuleUse,
)
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.control",
    since=CORE_0,
    summary="Block, branch, call, suspension, return, and failure operations.",
    dependencies=("core.contract.machine",),
    document_order=3,
    normative_text=(
        "The first record of every function is control.block. A function never "
        "falls through the end of its bytecode: every possible sequential "
        "continuation has a following decoded record, so a final record must "
        "terminate, branch, yield, or return. Direct target fields are signed "
        "four-byte-word displacements from the end of their record. Verification "
        "computes them in widened signed arithmetic, rejects overflow or an "
        "out-of-function address, and requires an exact decoded control.block "
        "target. Execution trusts the verified target without repeating range, "
        "alignment, boundary, or opcode checks. Unreachable structurally valid "
        "records are permitted. Calls use the target signature's three direct "
        "register prefixes and canonical overflow packet. Every call preflight "
        "check completes before moving arguments, changing packet state, pushing "
        "a frame, or entering a provider. Common dispatch then enters the target "
        "module, which selects its implementation and owns any frame it pushes. "
        "A bytecode target pushes its fixed frame before consuming arguments or "
        "performing fallible work; immediate native targets may complete without "
        "a frame. The parent remains parked on the call "
        "record until successful return publishes results and advances it. Any "
        "status failure is terminal to that invocation attempt and uses ordinary "
        "unwind; the ISA does not specify broader process state after failure."
    ),
)


def _field(
    name: str,
    offset: int,
    encoding_id: str,
    role: InstructionFieldRole,
    description: str,
    rule,
    arguments: tuple[object, ...] = (),
):
    return instruction_field(
        name,
        offset,
        encoding_id,
        role,
        description,
        (RuleUse(rule.entity_id, arguments),),
    )


def _zero_u16(offset: int):
    return _field(
        "zero_padding_u16",
        offset,
        U16.entity_id,
        InstructionFieldRole.PADDING,
        "Canonical zero padding.",
        ZERO,
    )


def _target_s16():
    return _field(
        "target_rel16",
        2,
        I16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Signed four-byte-word displacement from the end of this record.",
        CONTROL_TARGET_RELATIVE_S16,
    )


def _target_s32():
    return _field(
        "target_rel32",
        4,
        I32.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Signed four-byte-word displacement from the end of this record.",
        CONTROL_TARGET_RELATIVE_S32,
    )


CONTROL_BLOCK = core_instruction(
    entity_id="core.instruction.control.block",
    since=CORE_0,
    summary="Marks the only legal direct control target.",
    opcode=0x01,
    mnemonic="control.block",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(zero_padding("zero_padding_u8", 1, 3),),
    control_flow=ControlFlow.BLOCK,
    semantics=InstructionSemantics(
        description=(
            "Marks a legal branch, switch, or yield destination without "
            "carrying a serialized block ordinal."
        ),
        verification=(
            "Every zero_padding_u8 byte must equal zero.",
            "The first decoded record of every function must be control.block; "
            "all direct targets must resolve to a decoded control.block.",
        ),
        preconditions=(),
        success=("The program counter advances by four bytes.",),
        failures=(),
        ownership=(),
        assembly=("^bb3:\n  control.block",),
        pseudocode="pc = pc + 4;",
    ),
)

CONTROL_RETURN = core_instruction(
    entity_id="core.instruction.control.return",
    since=CORE_0,
    summary="Validates, publishes, and returns function results.",
    opcode=0x02,
    mnemonic="control.return",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(zero_padding("zero_padding_u8", 1, 3),),
    constraints=(RuleUse(CONTROL_RETURN_SIGNATURE.entity_id),),
    control_flow=ControlFlow.RETURN,
    semantics=InstructionSemantics(
        description=(
            "Completes the current function by validating every direct and "
            "overflow result against its private signature, transactionally "
            "publishing direct results, cleaning the frame, and returning to "
            "the parked parent or completing the root invocation."
        ),
        verification=(
            "Every zero_padding_u8 byte must equal zero.",
            "The function's value, ref, and function extents must cover its "
            "signature-derived direct result prefixes and overflow packet.",
            "The record is a terminator with no sequential successor.",
        ),
        preconditions=(
            "Every direct and overflow ref result must satisfy its exact type "
            "and nullability contract.",
            "Every direct and overflow function result must be canonical null "
            "or satisfy current-program, target, callable-token, and MAY_YIELD "
            "requirements.",
        ),
        success=(
            "All result validation completes before caller-visible mutation.",
            "Direct value cells and function carriers copy exactly. Owned direct "
            "refs move; borrowed direct refs are retained and moved as owners; "
            "their sources become null and replaced caller owners are released.",
            "Overflow results remain in the already validated caller packet.",
            "Every remaining owned ref in the exact frame extent is released "
            "and the frame is popped.",
            "A nested parent advances past its parked call. A root invocation "
            "publishes its complete public result list and completes with OK.",
        ),
        failures=(
            FailureCase(
                "invalid_argument",
                "Any ref or function result violates its private signature.",
                "No caller-visible or public result is changed before ordinary unwind.",
            ),
        ),
        ownership=(
            "Result publication promotes every escaping borrow to an owner. "
            "Frame cleanup releases all other owned refs; function values own nothing.",
        ),
        assembly=("control.return",),
        pseudocode=(
            "validate_complete_result_packet(frame);\n"
            "publish_direct_results(frame, parent_or_root);\n"
            "release_remaining_frame_owners(frame);\n"
            "pop_frame_and_complete_or_resume_parent();"
        ),
    ),
)

CONTROL_YIELD_S32 = core_instruction(
    entity_id="core.instruction.control.yield.s32",
    since=CORE_0,
    summary="Suspends with one explicit wide continuation target.",
    opcode=0x03,
    mnemonic="control.yield.s32",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(zero_padding("zero_padding_u8", 1, 3), _target_s32()),
    control_flow=ControlFlow.YIELD,
    suspension=Suspension.ALWAYS,
    semantics=InstructionSemantics(
        description=(
            "Ends the physical block, durably records one explicit continuation "
            "block, publishes the suspended invocation, and issues its optional "
            "level-triggered wake callback when non-null. There is no implicit "
            "next-PC continuation."
        ),
        verification=(
            "Every zero_padding_u8 byte must equal zero.",
            "target_rel32 must resolve to a decoded control.block in the same function.",
            "Verification deliberately does not infer or check the function's "
            "conservative MAY_YIELD metadata.",
        ),
        preconditions=(
            "The public invocation must not have begun with any non-null borrowed ref.",
        ),
        success=(
            "The frame PC is set to the verified target and every suspension-"
            "visible state change is durable before any non-null wake callback runs.",
            "The invocation becomes suspended and drive returns the non-status "
            "SUSPENDED outcome. Resumption begins at the saved block.",
        ),
        failures=(
            FailureCase(
                "failed_precondition",
                "The public invocation has any non-null borrowed argument.",
                "The frame PC, suspension state, and wake state remain unchanged.",
            ),
        ),
        ownership=("Every frame-local ref state remains live across suspension.",),
        assembly=("control.yield.s32 ^bb7",),
        pseudocode=(
            "if (invocation_has_borrowed_public_arguments) {\n"
            "  fail(failed_precondition);\n"
            "}\n"
            "frame.pc = direct_target(record_pc, 8, target_rel32);\n"
            "publish_suspended_invocation();\n"
            "invoke_level_triggered_wake_callback_if_nonnull();\n"
            "return SUSPENDED;"
        ),
    ),
)


def _branch(
    *,
    opcode: int,
    mnemonic: str,
    width: int,
    condition: str | None,
) -> Instruction:
    is_wide = width == 32
    byte_length = 8 if is_wide else 4
    fields = []
    if condition is None:
        fields.append(zero_padding("zero_padding_u8", 1, 3 if is_wide else 1))
    else:
        fields.append(
            value_register(
                "condition_v8",
                1,
                InstructionFieldRole.OPERAND,
                "Complete 64-bit branch condition.",
            )
        )
        if is_wide:
            fields.append(_zero_u16(2))
    fields.append(_target_s32() if is_wide else _target_s16())

    if condition is None:
        description = "Unconditionally transfers to the verified direct target."
        pseudocode = "pc = direct_target(record_pc, record_size, target_rel);"
        control_flow = ControlFlow.BRANCH
        success = ("The program counter becomes the verified direct target.",)
        sequential_verification = ()
    else:
        comparison = "!= 0" if condition == "nonzero" else "== 0"
        description = (
            f"Transfers to the verified target when the complete condition is "
            f"{condition}; otherwise continues at the following record."
        )
        pseudocode = (
            f"pc = values[condition_v8] {comparison}\n"
            "    ? direct_target(record_pc, record_size, target_rel)\n"
            "    : record_pc + record_size;"
        )
        control_flow = ControlFlow.CONDITIONAL_BRANCH
        success = (
            "The program counter becomes the direct target when the predicate "
            "is true and the next record otherwise.",
        )
        sequential_verification = (
            "A decoded record must exist immediately after this instruction.",
        )

    target_name = "target_rel32" if is_wide else "target_rel16"
    verification = (
        f"{target_name} must resolve to a decoded control.block in this function.",
        *sequential_verification,
    )
    if condition is not None:
        verification = (
            "condition_v8 must be a valid value-register ordinal.",
            *verification,
        )
    if is_wide and condition is not None:
        verification = (*verification, "zero_padding_u16 must equal zero.")
    elif condition is None:
        verification = (
            *verification,
            "Every zero_padding_u8 byte must equal zero.",
        )

    predicate_word = "" if condition is None else " %v<condition>,"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=description,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=byte_length,
        family_id=FAMILY.entity_id,
        fields=tuple(fields),
        control_flow=control_flow,
        semantics=InstructionSemantics(
            description=description,
            verification=verification,
            preconditions=(),
            success=success,
            failures=(),
            ownership=(),
            assembly=(f"{mnemonic}{predicate_word} ^bb8",),
            pseudocode=pseudocode,
        ),
    )


CONTROL_BRANCH_S16 = _branch(
    opcode=0x04,
    mnemonic="control.branch.s16",
    width=16,
    condition=None,
)
CONTROL_BRANCH_S32 = _branch(
    opcode=0x05,
    mnemonic="control.branch.s32",
    width=32,
    condition=None,
)
CONTROL_BRANCH_IF_S16 = _branch(
    opcode=0x06,
    mnemonic="control.branch.if.s16",
    width=16,
    condition="nonzero",
)
CONTROL_BRANCH_IF_S32 = _branch(
    opcode=0x07,
    mnemonic="control.branch.if.s32",
    width=32,
    condition="nonzero",
)
CONTROL_BRANCH_UNLESS_S16 = _branch(
    opcode=0x08,
    mnemonic="control.branch.unless.s16",
    width=16,
    condition="zero",
)
CONTROL_BRANCH_UNLESS_S32 = _branch(
    opcode=0x09,
    mnemonic="control.branch.unless.s32",
    width=32,
    condition="zero",
)

CONTROL_SWITCH = core_instruction(
    entity_id="core.instruction.control.switch",
    since=CORE_0,
    summary="Branches through a function-local dense target table.",
    opcode=0x0A,
    mnemonic="control.switch",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "selector_v8",
            1,
            InstructionFieldRole.OPERAND,
            "Unsigned complete-cell zero-based table selector.",
        ),
        _field(
            "target_count_u16",
            2,
            U16.entity_id,
            InstructionFieldRole.CONSTRAINT_MEMBER,
            "Number of switch-target entries in this table slice.",
            CONSTRAINT_MEMBER,
            ("control.switch.targets",),
        ),
        _field(
            "target_base_u32",
            4,
            U32.entity_id,
            InstructionFieldRole.CONSTRAINT_MEMBER,
            "First entry relative to the owning function's target range.",
            CONSTRAINT_MEMBER,
            ("control.switch.targets",),
        ),
    ),
    constraints=(
        RuleUse(
            CONTROL_SWITCH_TARGETS.entity_id,
            (
                FieldReference("target_count_u16"),
                FieldReference("target_base_u32"),
            ),
        ),
    ),
    control_flow=ControlFlow.SWITCH,
    semantics=InstructionSemantics(
        description=(
            "Treats the complete selector cell as unsigned, selects a verified "
            "function-local target entry when it is below target_count_u16, and "
            "otherwise takes the sequential default path."
        ),
        verification=(
            "selector_v8 must be a valid value-register ordinal.",
            "target_base_u32 plus target_count_u16 must form an in-bounds slice "
            "of the function's switch-target entries without overflow.",
            "Every selected entry must target a decoded control.block in this function.",
            "A decoded record must exist immediately after this instruction.",
        ),
        preconditions=(),
        success=(
            "An in-range selector branches to its table entry; an out-of-range "
            "selector advances by eight bytes.",
        ),
        failures=(),
        ownership=(),
        assembly=("control.switch %v3, targets[12...17]",),
        pseudocode=(
            "selector = bits_u64(values[selector_v8]);\n"
            "if (selector < target_count_u16) {\n"
            "  entry = function_switch_targets[target_base_u32 + selector];\n"
            "  pc = function_bytecode_start + entry * 4;\n"
            "} else {\n"
            "  pc = record_pc + 8;\n"
            "}"
        ),
    ),
)


def _constraint_member(name: str, offset: int, description: str, kind: str):
    return _field(
        name,
        offset,
        U16.entity_id,
        InstructionFieldRole.CONSTRAINT_MEMBER,
        description,
        CONSTRAINT_MEMBER,
        (kind,),
    )


_CALL_TARGET_SELECTOR = SELECTOR_TABLES_BY_NAME["control.call.target"]
CONTROL_CALL = core_instruction(
    entity_id="core.instruction.control.call",
    since=CORE_0,
    summary="Calls a local or linked imported function.",
    opcode=0x0B,
    mnemonic="control.call",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        _field(
            "target_kind_u8",
            1,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Local, required-import, or optional-import target selector.",
            SELECTOR,
            (EntityReference(_CALL_TARGET_SELECTOR.entity_id),),
        ),
        _constraint_member(
            "target_ordinal_u16",
            2,
            "Direct ordinal in the target-kind-selected table.",
            "control.call",
        ),
        _constraint_member(
            "direct_ref_move_mask_u16",
            4,
            "Ownership-transfer bits for direct ref arguments.",
            "control.call",
        ),
        _zero_u16(6),
    ),
    constraints=(
        RuleUse(
            CONTROL_CALL_RULE.entity_id,
            (
                FieldReference("target_kind_u8"),
                FieldReference("target_ordinal_u16"),
                FieldReference("direct_ref_move_mask_u16"),
            ),
        ),
    ),
    control_flow=ControlFlow.CALL,
    suspension=Suspension.TARGET_DEPENDENT,
    semantics=InstructionSemantics(
        description=(
            "Invokes a function in the current linked module or its resolved-"
            "import table using the target's exact three-bank callable packet."
        ),
        verification=(
            "target_kind_u8 and target_ordinal_u16 must identify a valid local, "
            "required import, or optional import with matching optionality.",
            "Every move-mask bit above the direct ref argument count must be zero.",
            "The caller's register and outgoing-packet extents must cover every "
            "direct and overflow argument/result region of the target callable type.",
            "zero_padding_u16 must equal zero and a decoded success continuation "
            "must follow this instruction.",
            "Program linking must resolve every present import to an exactly "
            "compatible signature and MAY_YIELD contract.",
        ),
        preconditions=(
            "A yielding target requires that the public invocation began with no "
            "non-null borrowed ref.",
            "Every direct and overflow ref/function argument must satisfy the "
            "target's private contract.",
            "An optional imported target must be resolved.",
            "The target implementation must be enterable before any argument move.",
        ),
        success=(
            "Direct values and functions copy exactly into the child prefixes. "
            "Each clear ref-mask bit borrows without changing the parent; each "
            "set bit moves the complete ref state and clears the parent source."
            " Overflow cells already occupy the canonical packet.",
            "The parent remains parked while the child runs or suspends. "
            "Successful return publishes results and advances the parent by eight bytes.",
        ),
        failures=(
            FailureCase(
                "failed_precondition",
                "A yielding target is entered from an invocation with borrowed public refs.",
                "No argument, packet, or frame state changes.",
            ),
            FailureCase(
                "invalid_argument",
                "Any ref or function argument violates the target contract.",
                "No argument, packet, or frame state changes.",
            ),
            FailureCase(
                "not_found",
                "The selected optional import is unresolved.",
                "No argument, packet, or frame state changes.",
            ),
            FailureCase(
                "resource_exhausted",
                "A bytecode target's fixed frame does not fit the invocation stack.",
                "Target construction fails before argument moves or external work.",
            ),
            FailureCase(
                "target_status",
                "The entered target later returns any non-OK status.",
                "The invocation unwinds; public results remain untouched and "
                "the ISA does not specify broader program/process state.",
            ),
        ),
        ownership=(
            "Direct ref moves transfer their exact null/borrowed/owned state. "
            "Borrowed arguments remain dominated by the paused parent. Escaping "
            "results are promoted to owners by result publication.",
        ),
        assembly=(
            "control.call @function7 {direct_ref_move_mask = 0x0005}",
            "control.call @import3 {direct_ref_move_mask = 0x0000}",
        ),
        pseudocode=(
            "target = preflight_linked_target(target_kind_u8, target_ordinal_u16);\n"
            "validate_borrow_and_argument_contracts(target);\n"
            "dispatch_preflighted_call(\n"
            "    target, call_packet, direct_ref_move_mask_u16);"
        ),
    ),
)

CONTROL_CALL_INDIRECT = core_instruction(
    entity_id="core.instruction.control.call.indirect",
    since=CORE_0,
    summary="Calls a dynamic first-class function value.",
    opcode=0x0C,
    mnemonic="control.call.indirect",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        function_register(
            "target_f8",
            1,
            InstructionFieldRole.OPERAND,
            "Dynamic first-class function target.",
        ),
        _constraint_member(
            "callable_type_ordinal_u16",
            2,
            "Exact expected structural callable type.",
            "control.call.indirect",
        ),
        _constraint_member(
            "direct_ref_move_mask_u16",
            4,
            "Ownership-transfer bits for direct ref arguments.",
            "control.call.indirect",
        ),
        _zero_u16(6),
    ),
    constraints=(
        RuleUse(
            CONTROL_CALL_INDIRECT_RULE.entity_id,
            (
                FieldReference("target_f8"),
                FieldReference("callable_type_ordinal_u16"),
                FieldReference("direct_ref_move_mask_u16"),
            ),
        ),
    ),
    control_flow=ControlFlow.CALL,
    suspension=Suspension.TARGET_DEPENDENT,
    semantics=InstructionSemantics(
        description=(
            "Invokes a first-class function value in the current process after "
            "checking it against the encoded structural callable type."
        ),
        verification=(
            "target_f8 must be a valid function-register ordinal.",
            "callable_type_ordinal_u16 must name an available callable type and "
            "direct_ref_move_mask_u16 must fit its direct ref argument count.",
            "The caller's register and packet extents must cover that callable "
            "type and a decoded success continuation must follow this record.",
            "zero_padding_u16 must equal zero; verifier dataflow does not attempt "
            "to discover the runtime target.",
        ),
        preconditions=(
            "target_f8 must be non-null, belong to the current program, contain "
            "in-range module/function ordinals, and carry the canonical token "
            "for callable_type_ordinal_u16.",
            "The target's actual MAY_YIELD fact must be permitted, and a yielding "
            "target requires no borrowed public invocation refs.",
            "Every direct and overflow ref/function argument must satisfy the "
            "encoded callable contract before any mutation or provider entry.",
        ),
        success=(
            "Argument transfer, target entry, suspension, return publication, "
            "and unwind use the generic absolute-target call path. The parent "
            "advances by eight bytes only after successful return.",
        ),
        failures=(
            FailureCase(
                "failed_precondition",
                "target_f8 is null or a yielding target sees borrowed public refs.",
                "No argument, packet, frame, or provider state changes.",
            ),
            FailureCase(
                "invalid_argument",
                "Program identity, target bounds, callable token, effect, or "
                "argument contract is invalid.",
                "No argument, packet, frame, or provider state changes.",
            ),
            FailureCase(
                "resource_exhausted",
                "A bytecode target's fixed frame does not fit the invocation stack.",
                "Target construction fails before argument moves or external work.",
            ),
            FailureCase(
                "target_status",
                "The entered target later returns any non-OK status.",
                "The invocation unwinds; public results remain untouched and "
                "the ISA does not specify broader program/process state.",
            ),
        ),
        ownership=(
            "Direct ref-mask semantics and result borrow promotion are identical "
            "to control.call. Function values themselves own nothing.",
        ),
        assembly=(
            "control.call.indirect %f4, type @callable3 {"
            "direct_ref_move_mask = 0x0002}",
        ),
        pseudocode=(
            "target = preflight_function_value(\n"
            "    functions[target_f8], callable_type_ordinal_u16);\n"
            "validate_borrow_and_argument_contracts(target);\n"
            "dispatch_preflighted_call(\n"
            "    target, call_packet, direct_ref_move_mask_u16);"
        ),
    ),
)


def _diagnostic_message():
    return ref_register(
        "message_r8_nullable",
        2,
        InstructionFieldRole.OPERAND,
        "Optional best-effort readable vm.buffer diagnostic message.",
        RuntimeRefPolicy(
            "vm.buffer",
            RefNullPolicy.NULLABLE,
            RefOwnership.DIAGNOSTIC_BORROW,
        ),
    )


CONTROL_ASSERT = core_instruction(
    entity_id="core.instruction.control.assert",
    since=CORE_0,
    summary="Continues on true and fails with failed_precondition on false.",
    opcode=0x0D,
    mnemonic="control.assert",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "condition_v8",
            1,
            InstructionFieldRole.OPERAND,
            "Complete 64-bit assertion condition.",
        ),
        _diagnostic_message(),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    semantics=InstructionSemantics(
        description=(
            "Continues when the complete condition cell is nonzero and otherwise "
            "terminates the invocation with failed_precondition and optional "
            "best-effort diagnostic bytes."
        ),
        verification=(
            "condition_v8 and message_r8_nullable must be valid register ordinals.",
            "zero_padding_u8 must equal zero and a decoded sequential successor "
            "must follow this record.",
        ),
        preconditions=(),
        success=(
            "A nonzero condition advances the program counter by four bytes "
            "without inspecting the message ref.",
        ),
        failures=(
            FailureCase(
                "failed_precondition",
                "condition_v8 contains the complete 64-bit zero value.",
                "Optional diagnostics are captured before ordinary unwind; "
                "public result storage remains untouched.",
            ),
        ),
        ownership=(
            "The message is borrowed only while copying optional diagnostic "
            "bytes. Wrong-type, unreadable, uncopyable, or allocation-failed "
            "diagnostics are omitted without changing the status.",
        ),
        assembly=("control.assert %v4, %r2",),
        pseudocode=(
            "if (values[condition_v8] != 0) {\n"
            "  pc = record_pc + 4;\n"
            "} else {\n"
            "  fail(failed_precondition, optional_message(message_r8_nullable));\n"
            "}"
        ),
    ),
)

_STATUS_SELECTOR = SELECTOR_TABLES_BY_NAME["control.status"]
CONTROL_FAIL = core_instruction(
    entity_id="core.instruction.control.fail",
    since=CORE_0,
    summary="Terminates with one immediate architectural status.",
    opcode=0x0E,
    mnemonic="control.fail",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _field(
            "status_u8",
            1,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Assigned non-OK architectural status selector.",
            SELECTOR,
            (EntityReference(_STATUS_SELECTOR.entity_id),),
        ),
        _diagnostic_message(),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    control_flow=ControlFlow.FAIL,
    semantics=InstructionSemantics(
        description=(
            "Unconditionally terminates the invocation with status_u8 and "
            "optional best-effort diagnostic bytes."
        ),
        verification=(
            "status_u8 must be an assigned non-OK control.status selector.",
            "message_r8_nullable must be a valid ref-register ordinal and "
            "zero_padding_u8 must equal zero.",
            "The record is a terminator with no sequential successor.",
        ),
        preconditions=(),
        success=(),
        failures=(
            FailureCase(
                "status_u8",
                "The instruction is executed.",
                "Optional diagnostic context is captured before all live frames "
                "unwind; public results remain untouched and drive returns the "
                "invocation to idle before returning the terminal status.",
            ),
        ),
        ownership=(
            "The message is diagnostic-only borrowed state. Unwind releases all "
            "live owned refs through their private no-status teardown paths.",
        ),
        assembly=("control.fail invalid_argument, %r2",),
        pseudocode=(
            "fail(status_from_architecture_selector(status_u8),\n"
            "     optional_message(message_r8_nullable));"
        ),
    ),
)

INSTRUCTIONS = (
    CONTROL_BLOCK,
    CONTROL_RETURN,
    CONTROL_YIELD_S32,
    CONTROL_BRANCH_S16,
    CONTROL_BRANCH_S32,
    CONTROL_BRANCH_IF_S16,
    CONTROL_BRANCH_IF_S32,
    CONTROL_BRANCH_UNLESS_S16,
    CONTROL_BRANCH_UNLESS_S32,
    CONTROL_SWITCH,
    CONTROL_CALL,
    CONTROL_CALL_INDIRECT,
    CONTROL_ASSERT,
    CONTROL_FAIL,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
