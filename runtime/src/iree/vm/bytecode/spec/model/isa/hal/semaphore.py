# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL 0.0 timeline semaphore instructions."""

from __future__ import annotations

from model.isa import (
    FailureCase,
    InstructionFamily,
    InstructionFieldRole,
    InstructionRangeGroup,
    InstructionRangeMember,
    InstructionRangeStorage,
    InstructionSemantics,
    RefNullPolicy,
    RefOwnership,
    RuntimeRefPolicy,
    StateResource,
    Suspension,
)
from model.isa.declarations import (
    hal_instruction,
    instruction_field,
    state_allocate,
    state_read,
    state_synchronize,
    state_write,
    value_register,
    zero_padding,
)
from model.isa.hal.declarations import (
    hal_ref,
    hal_result_ref,
    required_ref_failures,
    zero_u16,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import (
    ALLOWED_BITS,
    COUNT_NONZERO_WHEN_SELECTOR,
    RANGE_BASE,
    RANGE_COUNT,
    SELECTOR,
)
from model.schema import U8, U16, U32, EntityReference, FieldReference, RuleUse
from model.specification import HAL_0

FAMILY = InstructionFamily(
    entity_id="hal.family.semaphore",
    since=HAL_0,
    summary="Timeline creation, query, group-scoped signal, and coroutine await.",
    dependencies=("hal.contract.abi",),
    document_order=16,
    normative_text=(
        "HAL semaphores are monotonically increasing timelines. Version zero "
        "admits program payloads from zero through 0x000000007FFFFFFE; larger "
        "values are reserved for HAL sticky-failure representation. Creation "
        "deliberately omits DEVICE_LOCAL and SINGLE_PRODUCER promises because a "
        "direct record cannot establish their whole-program preconditions. Host "
        "signals name the complete causal device group explicitly; there is no "
        "ambient frontier. semaphore.await is the HAL page's only suspending "
        "operation. It polls before considering timeout, converts a relative "
        "timeout to one absolute deadline only once, leaves its PC parked until "
        "terminal completion, and uses provider-owned callback arbitration in "
        "persistent invocation-stack storage. Suspension and pending resume are "
        "ordinary driving outcomes, never statuses. Provider terminal return "
        "quiesces every callback before releasing state or publishing a result."
    ),
)


def _value(name: str, offset: int, role: InstructionFieldRole, description: str):
    return value_register(name, offset, role, description)


def _selector(
    name: str,
    offset: int,
    selector_name: str,
    description: str,
):
    table = SELECTOR_TABLES_BY_NAME[selector_name]
    return instruction_field(
        name,
        offset,
        U8.entity_id,
        InstructionFieldRole.IMMEDIATE,
        description,
        (RuleUse(SELECTOR.entity_id, (EntityReference(table.entity_id),)),),
    )


def _range_field(
    name: str,
    offset: int,
    *,
    range_name: str,
    role: InstructionFieldRole,
    description: str,
):
    rule = RANGE_BASE if role == InstructionFieldRole.RANGE_BASE else RANGE_COUNT
    return instruction_field(
        name,
        offset,
        U16.entity_id,
        role,
        description,
        (RuleUse(rule.entity_id, (range_name,)),),
    )


HAL_SEMAPHORE_CREATE = hal_instruction(
    entity_id="hal.instruction.semaphore.create",
    since=HAL_0,
    summary="Creates one timeline semaphore on an explicit device.",
    opcode=0x06,
    mnemonic="hal.semaphore.create",
    byte_length=12,
    family_id=FAMILY.entity_id,
    fields=(
        hal_result_ref("dst_r8", 2, "hal.semaphore"),
        hal_ref("device_r8", 3, "hal.device"),
        _value(
            "affinity_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Complete u64 queue-affinity bitset.",
        ),
        _value(
            "initial_value_v8",
            5,
            InstructionFieldRole.OPERAND,
            "Initial architectural semaphore payload.",
        ),
        zero_u16("zero_padding_u16", 6),
        instruction_field(
            "flags_u32",
            8,
            U32.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "HOST_INTERRUPT, EXPORTABLE, and EXPORTABLE_TIMEPOINTS flag bits.",
            (RuleUse(ALLOWED_BITS.entity_id, (0x0000000E,)),),
        ),
    ),
    state_effects=(
        state_read(StateResource.HAL_DEVICE, "device_r8"),
        state_allocate(StateResource.HAL_SEMAPHORE, "dst_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Validates one explicit device and architectural initial payload, "
            "then transactionally creates an optionally interruptible/exportable "
            "timeline semaphore."
        ),
        verification=(
            "All register ordinals must be valid, zero_padding_u16 must be zero, "
            "and flags_u32 may contain only bits 1 through 3.",
        ),
        preconditions=(
            "device_r8 must contain a non-null exact hal.device and initial_value_v8 "
            "must be no greater than 0x000000007FFFFFFE.",
        ),
        success=(
            "A complete non-null semaphore owner replaces dst_r8 and the program "
            "counter advances by 12 bytes. Zero flags are valid.",
        ),
        failures=(
            *required_ref_failures(
                "hal.device", "No provider call occurs and dst_r8 remains unchanged."
            ),
            FailureCase(
                "out_of_range",
                "The initial value lies in HAL's reserved failure-payload domain.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "HAL cannot create the requested semaphore or returns another error.",
                "Any partial semaphore is released and dst_r8 remains unchanged.",
            ),
        ),
        ownership=(
            "device_r8 is borrowed for synchronous creation. The complete new "
            "semaphore owner replaces dst_r8 only after provider success.",
        ),
        assembly=(
            "%r<dst> = hal.semaphore.create %r<device>, %v<affinity>, "
            "%v<initial_value> {flags}",
        ),
        pseudocode=(
            "device = require_hal_ref(refs[device_r8], hal_device_type);\n"
            "initial_value = require_semaphore_payload(values[initial_value_v8]);\n"
            "semaphore = NULL;\n"
            "status = hal_semaphore_create(device, values[affinity_v8],\n"
            "    initial_value, flags_u32, &semaphore);\n"
            "if (status failed) { release_if_nonnull(semaphore); return status; }\n"
            "replace_ref(&refs[dst_r8], owned_ref(semaphore, hal_semaphore_type));\n"
            "pc = pc + 12;"
        ),
    ),
)

HAL_SEMAPHORE_QUERY = hal_instruction(
    entity_id="hal.instruction.semaphore.query",
    since=HAL_0,
    summary="Queries a timeline payload or propagates its sticky failure.",
    opcode=0x07,
    mnemonic="hal.semaphore.query",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _value(
            "dst_v8",
            2,
            InstructionFieldRole.RESULT,
            "Current architectural semaphore payload.",
        ),
        hal_ref("semaphore_r8", 3, "hal.semaphore"),
    ),
    state_effects=(state_read(StateResource.HAL_SEMAPHORE, "semaphore_r8"),),
    semantics=InstructionSemantics(
        description="Reads one timeline semaphore without blocking.",
        verification=("dst_v8 and semaphore_r8 must be valid register ordinals.",),
        preconditions=("semaphore_r8 must contain a non-null exact hal.semaphore.",),
        success=(
            "dst_v8 receives the current architectural payload and the program "
            "counter advances by four bytes.",
        ),
        failures=(
            *required_ref_failures(
                "hal.semaphore", "No provider call occurs and dst_v8 remains unchanged."
            ),
            FailureCase(
                "provider_status",
                "The semaphore has sticky failure or its provider query otherwise fails.",
                "dst_v8 remains unchanged.",
            ),
        ),
        ownership=("semaphore_r8 is borrowed for the synchronous query.",),
        assembly=("%v<dst> = hal.semaphore.query %r<semaphore>",),
        pseudocode=(
            "semaphore = require_hal_ref(refs[semaphore_r8], hal_semaphore_type);\n"
            "value = 0;\n"
            "status = hal_semaphore_query(semaphore, &value);\n"
            "if (status failed) return status;\n"
            "values[dst_v8] = value;\n"
            "pc = pc + 4;"
        ),
    ),
)

HAL_SEMAPHORE_SIGNAL = hal_instruction(
    entity_id="hal.instruction.semaphore.signal",
    since=HAL_0,
    summary="Publishes a host signal with explicit group causal state.",
    opcode=0x08,
    mnemonic="hal.semaphore.signal",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("group_r8", 2, "hal.device_group"),
        hal_ref("semaphore_r8", 3, "hal.semaphore"),
        _value(
            "payload_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Architectural semaphore signal payload.",
        ),
        zero_padding("zero_padding_u8", 5, 3),
    ),
    state_effects=(
        state_synchronize(StateResource.HAL_DEVICE_GROUP, "group_r8"),
        state_write(StateResource.HAL_SEMAPHORE, "semaphore_r8"),
        state_synchronize(StateResource.HAL_SEMAPHORE, "semaphore_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Publishes one host-originated timeline value together with the "
            "completed causal state of an explicit immutable device group."
        ),
        verification=(
            "Both ref registers and payload_v8 must be valid and every "
            "zero_padding_u8 byte must equal zero.",
        ),
        preconditions=(
            "group_r8 and semaphore_r8 must contain non-null exact objects and "
            "payload_v8 must be no greater than 0x000000007FFFFFFE.",
        ),
        success=(
            "The group composite operation publishes its completed causal state "
            "with the payload and the program counter advances by eight bytes.",
        ),
        failures=(
            *required_ref_failures(
                "hal.device_group",
                "No provider call occurs and all operands remain unchanged.",
            ),
            *required_ref_failures(
                "hal.semaphore",
                "No provider call occurs and all operands remain unchanged.",
            ),
            FailureCase(
                "out_of_range",
                "The payload lies in HAL's reserved failure-payload domain.",
                "No provider call occurs and all operands remain unchanged.",
            ),
            FailureCase(
                "provider_status",
                "Group-scoped signal returns any non-OK status.",
                "The provider defines any externally committed signal state; VM state is unchanged.",
            ),
        ),
        ownership=(
            "The device group and semaphore are borrowed for one synchronous "
            "composite HAL call; the VM owns no causal-frontier storage.",
        ),
        assembly=("hal.semaphore.signal %r<group>, %r<semaphore>, %v<payload>",),
        pseudocode=(
            "group = require_hal_ref(refs[group_r8], hal_device_group_type);\n"
            "semaphore = require_hal_ref(refs[semaphore_r8], hal_semaphore_type);\n"
            "payload = require_semaphore_payload(values[payload_v8]);\n"
            "status = hal_device_group_signal_semaphore(group, semaphore, payload);\n"
            "if (status failed) return status;\n"
            "pc = pc + 8;"
        ),
    ),
)

_AWAIT_RANGE_NAME = "timepoints"
_AWAIT_RANGE = InstructionRangeGroup(
    name=_AWAIT_RANGE_NAME,
    count_field="count_u16",
    members=(
        InstructionRangeMember(
            base_field="semaphore_base_u16",
            storage=InstructionRangeStorage.LOCAL_REFS,
            element_byte_length=1,
            element_alignment=1,
            runtime_ref_policy=RuntimeRefPolicy(
                "hal.semaphore", RefNullPolicy.REQUIRED, RefOwnership.BORROW
            ),
        ),
        InstructionRangeMember(
            base_field="payload_base_u16",
            storage=InstructionRangeStorage.LOCAL_BYTES,
            element_byte_length=8,
            element_alignment=8,
        ),
    ),
)

HAL_SEMAPHORE_AWAIT = hal_instruction(
    entity_id="hal.instruction.semaphore.await",
    since=HAL_0,
    summary="Polls or asynchronously awaits an ALL/ANY timepoint set.",
    opcode=0x09,
    mnemonic="hal.semaphore.await",
    byte_length=12,
    family_id=FAMILY.entity_id,
    fields=(
        _value(
            "dst_v8",
            2,
            InstructionFieldRole.RESULT,
            "UINT64_MAX for ALL or selected zero-based member index for ANY.",
        ),
        _selector(
            "mode_u8",
            3,
            "hal.semaphore.await.mode",
            "ALL=0 or ANY=1 completion selector.",
        ),
        _selector(
            "timeout_kind_u8",
            4,
            "hal.semaphore.await.timeout_kind",
            "RELATIVE=0 or ABSOLUTE=1 signed-nanosecond interpretation.",
        ),
        _value(
            "timeout_v8",
            5,
            InstructionFieldRole.OPERAND,
            "Exact two's-complement signed-i64 timeout bits.",
        ),
        _range_field(
            "semaphore_base_u16",
            6,
            range_name=_AWAIT_RANGE_NAME,
            role=InstructionFieldRole.RANGE_BASE,
            description="Base ref slot of exact non-null semaphore elements.",
        ),
        _range_field(
            "payload_base_u16",
            8,
            range_name=_AWAIT_RANGE_NAME,
            role=InstructionFieldRole.RANGE_BASE,
            description="Eight-byte-aligned local base of u64 payload elements.",
        ),
        _range_field(
            "count_u16",
            10,
            range_name=_AWAIT_RANGE_NAME,
            role=InstructionFieldRole.RANGE_COUNT,
            description="Shared semaphore and payload element count.",
        ),
    ),
    range_groups=(_AWAIT_RANGE,),
    constraints=(
        RuleUse(
            COUNT_NONZERO_WHEN_SELECTOR.entity_id,
            (
                FieldReference("mode_u8"),
                (1,),
                FieldReference("count_u16"),
            ),
        ),
    ),
    suspension=Suspension.CONDITIONAL,
    state_effects=(
        state_read(StateResource.FRAME_LOCALS, "semaphore_base_u16"),
        state_read(StateResource.FRAME_LOCALS, "payload_base_u16"),
        state_read(StateResource.HAL_SEMAPHORE, "semaphore_base_u16"),
        state_synchronize(StateResource.HAL_SEMAPHORE, "semaphore_base_u16"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Validates and polls one struct-of-arrays timepoint set, completing "
            "synchronously when possible or arming provider-owned asynchronous "
            "state in the invocation stack."
        ),
        verification=(
            "All registers and selectors must be valid; semaphore_base_u16 and "
            "payload_base_u16 share count_u16 and obey canonical empty-range, "
            "bounds, and eight-byte payload-alignment rules.",
            "ANY requires nonzero count. Empty ALL is valid.",
        ),
        preconditions=(
            "Every semaphore element must be non-null exact hal.semaphore and "
            "every payload must be no greater than 0x000000007FFFFFFE.",
            "A relative timeout must be nonnegative. The first execution converts "
            "a finite relative timeout to one saturating absolute monotonic deadline.",
            "Actual suspension requires no non-null borrowed public invocation argument.",
        ),
        success=(
            "Initial execution validates all dynamic operands before provider "
            "entry, polls every member, and gives sticky failures precedence over "
            "satisfaction in that deterministic scan.",
            "Satisfied ALL writes UINT64_MAX. Satisfied ANY writes the lowest "
            "satisfied index during initial polling; a later race may choose any "
            "member whose timepoint satisfied.",
            "Before terminal publication every callback source is quiescent and "
            "persistent state is released. dst_v8 is then written, the program "
            "counter advances by 12 bytes, and driving continues.",
            "If pending, the PC remains on this record and driving returns the "
            "non-status SUSPENDED outcome. A spurious resume may return SUSPENDED again.",
            "Every asynchronous completion publishes provider state before invoking "
            "the copied level-triggered wake callback when non-null; the callback "
            "never resumes the VM inline.",
        ),
        failures=(
            FailureCase(
                "failed_precondition",
                "A required semaphore is null or suspension is needed after borrowed public input.",
                "Before arming, no persistent state, callback, PC, or destination is changed.",
            ),
            FailureCase(
                "invalid_argument",
                "A non-null element has the wrong type or a relative timeout is negative.",
                "No provider state is armed and dst_v8 remains unchanged.",
            ),
            FailureCase(
                "out_of_range",
                "Any payload lies in HAL's reserved failure-payload domain.",
                "No provider state is armed and dst_v8 remains unchanged.",
            ),
            FailureCase(
                "deadline_exceeded",
                "One complete poll is unsatisfied at an expired deadline or the armed timer wins.",
                "All callbacks are quiescent, state is released, and dst_v8 remains unchanged.",
            ),
            FailureCase(
                "resource_exhausted",
                "Exact provider state does not fit the remaining invocation stack.",
                "No callback is armed and dst_v8 remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "A semaphore has sticky failure or provider poll/registration fails.",
                "Any armed callbacks are quiesced, state is released, and dst_v8 remains unchanged.",
            ),
            FailureCase(
                "cancellation_status",
                "Invocation cancellation wins asynchronous arbitration.",
                "Every callback is quiescent, state is released, and dst_v8 remains unchanged.",
            ),
        ),
        ownership=(
            "The suspended frames remain the lifetime anchors for copied native "
            "semaphore pointers. Provider state copies payloads, indices, deadline, "
            "wake callback, and callback registrations; callbacks never read VM ref slots.",
        ),
        assembly=(
            "%v<dst> = hal.semaphore.await {mode, timeout_kind}, %v<timeout>, "
            "local.refs[semaphore_base .. count], local.u64[payload_base .. count]",
        ),
        pseudocode=(
            "state = current_record_persistent_state_or_null();\n"
            "if (state == NULL) {\n"
            "  deadline = normalize_timeout(timeout_kind_u8,\n"
            "      signed_bits(values[timeout_v8]));\n"
            "  timepoints = validate_timepoint_range(semaphore_base_u16,\n"
            "      payload_base_u16, count_u16);\n"
            "  terminal = poll_all_before_timeout(timepoints, mode_u8, deadline);\n"
            "  if (terminal pending) {\n"
            "    require_no_borrowed_public_arguments();\n"
            "    state = invocation_stack_allocate_persistent(\n"
            "        checked_await_state_size(count_u16));\n"
            "    terminal = provider_arm(state, timepoints, mode_u8, deadline);\n"
            "  }\n"
            "} else {\n"
            "  terminal = provider_resume_or_cancel(state);\n"
            "}\n"
            "if (terminal pending) return SUSPENDED;\n"
            "quiesce_all_callback_sources_if_any(state);\n"
            "invocation_stack_release_persistent_if_any(state);\n"
            "if (terminal failed) return terminal.status;\n"
            "values[dst_v8] = terminal.selected_index_or_all;\n"
            "pc = pc + 12;"
        ),
    ),
)

INSTRUCTIONS = (
    HAL_SEMAPHORE_CREATE,
    HAL_SEMAPHORE_QUERY,
    HAL_SEMAPHORE_SIGNAL,
    HAL_SEMAPHORE_AWAIT,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
