# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL 0.0 device-group and executable-table instructions."""

from __future__ import annotations

from model.isa import (
    FailureCase,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
    RefNullPolicy,
    RefOwnership,
    RuntimeRefPolicy,
    StateResource,
)
from model.isa.declarations import (
    hal_instruction,
    instruction_field,
    ref_register,
    state_allocate,
    state_read,
    value_register,
    zero_padding,
)
from model.isa.validation import (
    LOCAL_BYTES_REPEATED_BASE,
    LOCAL_BYTES_REPEATED_COUNT,
    RODATA_EXECUTABLE_NAME_TABLE,
    STRING_ORDINAL_NONEMPTY,
)
from model.schema import U16, FieldReference, RuleUse
from model.specification import HAL_0

FAMILY = InstructionFamily(
    entity_id="hal.family.device",
    since=HAL_0,
    summary="Explicit device-group selection and executable-function tables.",
    dependencies=("hal.contract.abi",),
    document_order=14,
    normative_text=(
        "A hal.device_group is one host-created immutable device collection "
        "owning its member devices and topology/causal-domain state. Version "
        "zero exposes only count and indexed access; there is no implicit group "
        "or device constructor. A retained device object does not replace the "
        "host/process obligation to keep its enclosing group authority alive. "
        "Executable functions remain provider-local opaque tokens and therefore "
        "live only in an owned hal.executable_function_table that owns its loaded "
        "executable, ordered selected-token array, and dispatch metadata. An "
        "executable name-table rodata block is a dense little-endian u16 string-"
        "ordinal array with exact multiple-of-two length and nonempty strings. "
        "Empty tables and duplicate names are valid. A selected-name local array "
        "contains aligned u32 indices and preserves selection order and duplicates."
    ),
)


def _hal_ref(
    name: str,
    offset: int,
    type_contract: str,
    *,
    null_policy: RefNullPolicy = RefNullPolicy.REQUIRED,
    ownership: RefOwnership = RefOwnership.BORROW,
    role: InstructionFieldRole = InstructionFieldRole.OPERAND,
):
    return ref_register(
        name,
        offset,
        role,
        f"{null_policy.value} exact {type_contract} ref.",
        RuntimeRefPolicy(type_contract, null_policy, ownership),
    )


def _required_ref_failures(type_contract: str, atomicity: str):
    return (
        FailureCase(
            "failed_precondition",
            f"The required {type_contract} ref is canonical null.",
            atomicity,
        ),
        FailureCase(
            "invalid_argument",
            f"A non-null ref does not have exact {type_contract} type.",
            atomicity,
        ),
    )


HAL_DEVICE_GROUP_COUNT = hal_instruction(
    entity_id="hal.instruction.device.group.count",
    since=HAL_0,
    summary="Returns the exact immutable device-group count.",
    opcode=0x01,
    mnemonic="hal.device.group.count",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            2,
            InstructionFieldRole.RESULT,
            "Unsigned 64-bit device-count result.",
        ),
        _hal_ref("group_r8", 3, "hal.device_group"),
    ),
    state_effects=(),
    semantics=InstructionSemantics(
        description="Widens one immutable host device-group count to unsigned u64.",
        verification=(
            "dst_v8 and group_r8 must be valid value/ref register ordinals.",
        ),
        preconditions=("group_r8 must hold a non-null exact hal.device_group.",),
        success=(
            "dst_v8 receives the exact group count and the program counter "
            "advances by four bytes.",
        ),
        failures=_required_ref_failures(
            "hal.device_group", "dst_v8 and all other VM state remain unchanged."
        ),
        ownership=("group_r8 is borrowed for the synchronous count query.",),
        assembly=("%v<dst> = hal.device.group.count %r<group>",),
        pseudocode=(
            "group = require_hal_ref(refs[group_r8], hal_device_group_type);\n"
            "values[dst_v8] = u64(hal_device_group_count(group));\n"
            "pc = pc + 4;"
        ),
    ),
)

HAL_DEVICE_GROUP_GET = hal_instruction(
    entity_id="hal.instruction.device.group.get",
    since=HAL_0,
    summary="Retains and returns one device selected by index.",
    opcode=0x02,
    mnemonic="hal.device.group.get",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        _hal_ref(
            "dst_r8",
            2,
            "hal.device",
            null_policy=RefNullPolicy.RESULT_NONNULL,
            ownership=RefOwnership.REPLACE_OWNER,
            role=InstructionFieldRole.RESULT,
        ),
        _hal_ref("group_r8", 3, "hal.device_group"),
        value_register(
            "index_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Unsigned 64-bit device index.",
        ),
        zero_padding("zero_padding_u8", 5, 3),
    ),
    state_effects=(),
    semantics=InstructionSemantics(
        description=(
            "Checked-narrows the complete index to host size, obtains the "
            "group-owned device pointer, retains it, and transactionally publishes it."
        ),
        verification=(
            "dst_r8, group_r8, and index_v8 must be valid register ordinals and "
            "every zero_padding_u8 byte must equal zero.",
        ),
        preconditions=(
            "group_r8 must be a non-null exact hal.device_group and index_v8 must fit "
            "host size and be less than the immutable group count.",
        ),
        success=(
            "The selected device is retained before dst_r8 replacement, so "
            "dst_r8 may alias group_r8. The program counter advances by eight bytes.",
        ),
        failures=(
            *_required_ref_failures(
                "hal.device_group", "group_r8 and dst_r8 remain unchanged."
            ),
            FailureCase(
                "out_of_range",
                "index_v8 is not host-representable or is outside the group.",
                "group_r8 and dst_r8 remain unchanged.",
            ),
        ),
        ownership=(
            "A new ordinary device owner is retained before replacing and "
            "releasing dst_r8's prior state. Group authority remains host-owned.",
        ),
        assembly=("%r<dst> = hal.device.group.get %r<group>, %v<index>",),
        pseudocode=(
            "group = require_hal_ref(refs[group_r8], hal_device_group_type);\n"
            "index_u64 = values[index_v8];\n"
            "if (!fits_host_size(index_u64) ||\n"
            "    index_u64 >= hal_device_group_count(group)) {\n"
            "  fail(out_of_range);\n"
            "}\n"
            "device = hal_device_group_at(group, host_size(index_u64));\n"
            "retain(device);\n"
            "replace_ref(&refs[dst_r8], owned_ref(device, hal_device_type));\n"
            "pc = pc + 8;"
        ),
    ),
)

HAL_EXECUTABLE_LOAD = hal_instruction(
    entity_id="hal.instruction.executable.load",
    since=HAL_0,
    summary="Loads an ordered selected executable-function table.",
    opcode=0x0A,
    mnemonic="hal.executable.load",
    byte_length=16,
    family_id=FAMILY.entity_id,
    fields=(
        _hal_ref(
            "dst_r8",
            2,
            "hal.executable_function_table",
            null_policy=RefNullPolicy.RESULT_NONNULL,
            ownership=RefOwnership.REPLACE_OWNER,
            role=InstructionFieldRole.RESULT,
        ),
        _hal_ref("device_r8", 3, "hal.device"),
        value_register(
            "affinity_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Complete u64 queue-affinity bitset.",
        ),
        zero_padding("zero_padding0_u8", 5, 1),
        instruction_field(
            "resolver_string_u16",
            6,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Nonempty host-registered executable resolver name.",
            (RuleUse(STRING_ORDINAL_NONEMPTY.entity_id),),
        ),
        _hal_ref(
            "payload_vm_buffer_r8_nullable",
            8,
            "vm.buffer",
            null_policy=RefNullPolicy.NULLABLE,
        ),
        zero_padding("zero_padding1_u8", 9, 1),
        instruction_field(
            "name_table_u16",
            10,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Canonical executable-name-table rodata ordinal.",
            (RuleUse(RODATA_EXECUTABLE_NAME_TABLE.entity_id),),
        ),
        instruction_field(
            "selected_ordinal_base_u16",
            12,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Four-byte-aligned local base of selected u32 name indices.",
            (
                RuleUse(
                    LOCAL_BYTES_REPEATED_BASE.entity_id,
                    (FieldReference("selected_ordinal_count_u16"), 4, 4),
                ),
            ),
        ),
        instruction_field(
            "selected_ordinal_count_u16",
            14,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Number of selected u32 name indices.",
            (RuleUse(LOCAL_BYTES_REPEATED_COUNT.entity_id),),
        ),
    ),
    state_effects=(
        state_read(StateResource.HAL_DEVICE, "device_r8"),
        state_read(StateResource.BUFFER, "payload_vm_buffer_r8_nullable"),
        state_read(StateResource.FRAME_LOCALS, "selected_ordinal_base_u16"),
        state_allocate(StateResource.HAL_EXECUTABLE, "dst_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Invokes one host-registered resolver for an optional readable "
            "vm.buffer payload and returns an owned table of selected opaque "
            "function tokens in the exact requested order."
        ),
        verification=(
            "All registers and padding must be valid; resolver_string_u16 must "
            "be nonempty and name_table_u16 must decode as a canonical u16 "
            "table of nonempty string ordinals.",
            "The selected u32 index array must be naturally aligned and fit "
            "function-local bytes; zero count requires base zero.",
        ),
        preconditions=(
            "device_r8 must be a non-null exact hal.device. A non-null payload "
            "must be an exact open vm.buffer with READ. Every selected u32 index "
            "must be below name_table.count before provider entry.",
        ),
        success=(
            "The provider-returned table owns all executable and token state; a "
            "complete owner replaces dst_r8 and the program counter advances by "
            "16 bytes. Empty and duplicate selections retain their exact shape.",
        ),
        failures=(
            *_required_ref_failures(
                "hal.device", "dst_r8 and every input remain unchanged."
            ),
            FailureCase(
                "invalid_argument",
                "A non-null payload does not have exact vm.buffer type.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "failed_precondition",
                "A non-null payload's root is closed.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "permission_denied",
                "A non-null payload lacks READ access.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "out_of_range",
                "Any selected u32 index is outside the executable name table.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "resource_exhausted",
                "Required temporary adaptation storage does not fit invocation stack.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "The resolver or backend returns any non-OK status.",
                "Any partial table is released and dst_r8 remains unchanged.",
            ),
        ),
        ownership=(
            "device and optional payload are synchronously borrowed. The resolver "
            "must consume or copy payload bytes before returning. The completed "
            "table owner replaces dst_r8 only after provider success.",
        ),
        assembly=(
            "%r<dst> = hal.executable.load %r<device>, %v<affinity>, "
            "@resolver<string>, %r<payload>?, @rodata<name_table>, #selected_names",
        ),
        pseudocode=(
            "device = require_hal_ref(refs[device_r8], hal_device_type);\n"
            "payload = optional_hal_ref(\n"
            "    refs[payload_vm_buffer_r8_nullable], vm_buffer_type);\n"
            "if (payload != NULL) check_buffer_access(payload, READ);\n"
            "selected = local_u32_range(\n"
            "    selected_ordinal_base_u16, selected_ordinal_count_u16);\n"
            "for (index : selected) {\n"
            "  if (index >= name_table.count) fail(out_of_range);\n"
            "}\n"
            "table = NULL;\n"
            "status = hal_executable_function_table_load(\n"
            "    device, values[affinity_v8], strings[resolver_string_u16],\n"
            "    payload_bytes_or_empty(payload), name_table.strings,\n"
            "    selected, &table);\n"
            "if (status failed) { release_if_nonnull(table); return status; }\n"
            "replace_ref(&refs[dst_r8], owned_ref(\n"
            "    table, hal_executable_function_table_type));\n"
            "pc = pc + 16;"
        ),
    ),
)

INSTRUCTIONS = (
    HAL_DEVICE_GROUP_COUNT,
    HAL_DEVICE_GROUP_GET,
    HAL_EXECUTABLE_LOAD,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
