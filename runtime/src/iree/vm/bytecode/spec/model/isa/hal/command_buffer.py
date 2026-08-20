# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL 0.0 non-rerecordable command-buffer instructions."""

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
)
from model.isa.declarations import (
    hal_instruction,
    instruction_field,
    value_register,
    zero_padding,
)
from model.isa.hal.declarations import (
    hal_ref,
    hal_result_ref,
    required_ref_failures,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import (
    ALLOWED_BITS,
    ALLOWED_BITS_EXACTLY_ONE,
    ALLOWED_VALUES,
    ANY_BITS,
    LOCAL_BYTES_FIXED_BASE,
    LOCAL_BYTES_RANGE_BASE,
    LOCAL_BYTES_RANGE_LENGTH,
    PACKED_SELECTOR_INDEXED_ALLOWED_MASK,
    PACKED_SELECTORS,
    RANGE_BASE,
    RANGE_COUNT,
    SELECTOR,
    STRING_ORDINAL,
    ZERO,
)
from model.schema import U8, U16, U32, EntityReference, FieldReference, RuleUse
from model.specification import HAL_0

FAMILY = InstructionFamily(
    entity_id="hal.family.command_buffer",
    since=HAL_0,
    summary="Transactional command recording and permanent finalization.",
    dependencies=("hal.contract.abi",),
    document_order=13,
    normative_text=(
        "A command buffer is non-rerecordable. cmd.create returns an owned object "
        "already in recording state and cmd.finalize seals it permanently; there "
        "is no VM begin, reset, or rerecord transition. Host-provided command "
        "buffers may enter recording functions but must already be recording. "
        "The VM holds no shadow lifecycle or debug-nesting state. Before a record's "
        "first HAL call it completes every VM-owned ref/type/value/range/direct-or-"
        "slot/selector/adapter-storage check, so invalid VM input records nothing. "
        "A provider call may mutate the command buffer before returning failure; "
        "the ISA assigns no post-failure recording state and does not implicitly "
        "taint the process. Operand refs and temporary aggregates are borrowed only "
        "during the synchronous entry call. HAL copies immediate bytes/strings and "
        "captures every direct resource required for the command buffer's lifetime. "
        "Indirect buffer refs contain binding slots and capture no VM object."
    ),
)

_MEMORY_BARRIERS = "memory_barriers"
_BUFFER_BARRIERS = "buffer_barriers"
_DISPATCH_BINDINGS = "dispatch_bindings"


def _byte_member(base_field: str, width: int) -> InstructionRangeMember:
    return InstructionRangeMember(
        base_field=base_field,
        storage=InstructionRangeStorage.LOCAL_BYTES,
        element_byte_length=width,
        element_alignment=width,
    )


MEMORY_BARRIER_RANGE = InstructionRangeGroup(
    name=_MEMORY_BARRIERS,
    count_field="memory_count_u16",
    members=(
        _byte_member("memory_source_scope_base_u16", 4),
        _byte_member("memory_target_scope_base_u16", 4),
    ),
)

BUFFER_BARRIER_RANGE = InstructionRangeGroup(
    name=_BUFFER_BARRIERS,
    count_field="buffer_count_u16",
    members=(
        _byte_member("buffer_source_scope_base_u16", 4),
        _byte_member("buffer_target_scope_base_u16", 4),
        InstructionRangeMember(
            base_field="buffer_ref_base_u16",
            storage=InstructionRangeStorage.LOCAL_REFS,
            element_byte_length=1,
            element_alignment=1,
            runtime_ref_policy=RuntimeRefPolicy(
                "hal.buffer", RefNullPolicy.NULLABLE, RefOwnership.BORROW
            ),
        ),
        _byte_member("buffer_slot_base_u16", 4),
        _byte_member("buffer_offset_base_u16", 8),
        _byte_member("buffer_length_base_u16", 8),
    ),
)

DISPATCH_BINDING_RANGE = InstructionRangeGroup(
    name=_DISPATCH_BINDINGS,
    count_field="binding_count_u16",
    members=(
        InstructionRangeMember(
            base_field="binding_buffer_base_u16",
            storage=InstructionRangeStorage.LOCAL_REFS,
            element_byte_length=1,
            element_alignment=1,
            runtime_ref_policy=RuntimeRefPolicy(
                "hal.buffer", RefNullPolicy.NULLABLE, RefOwnership.BORROW
            ),
        ),
        _byte_member("binding_slot_base_u16", 4),
        _byte_member("binding_offset_base_u16", 8),
        _byte_member("binding_length_base_u16", 8),
    ),
)


def _value(name: str, offset: int, description: str):
    return value_register(name, offset, InstructionFieldRole.OPERAND, description)


def _direct_or_slot_ref(name: str, offset: int, description: str | None = None):
    return hal_ref(
        name,
        offset,
        "hal.buffer",
        null_policy=RefNullPolicy.DIRECT_OR_SLOT,
        description=description or "Direct exact hal.buffer or indirect binding slot.",
    )


def _zero_u32(name: str, offset: int):
    return instruction_field(
        name,
        offset,
        U32.entity_id,
        InstructionFieldRole.PADDING,
        "Canonical zero reserved flags.",
        (RuleUse(ZERO.entity_id),),
    )


def _flags(
    name: str,
    offset: int,
    allowed_mask: int,
    *,
    required_one_mask: int = 0,
):
    rule = (
        RuleUse(
            ALLOWED_BITS_EXACTLY_ONE.entity_id,
            (allowed_mask, required_one_mask),
        )
        if required_one_mask
        else RuleUse(ALLOWED_BITS.entity_id, (allowed_mask,))
    )
    return instruction_field(
        name,
        offset,
        U32.entity_id,
        InstructionFieldRole.IMMEDIATE,
        f"Architectural flag bits constrained by mask 0x{allowed_mask:08X}.",
        (rule,),
    )


def _range_field(
    name: str,
    offset: int,
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


def _local_fixed_base(name: str, offset: int, byte_length: int, description: str):
    return instruction_field(
        name,
        offset,
        U16.entity_id,
        InstructionFieldRole.IMMEDIATE,
        description,
        (RuleUse(LOCAL_BYTES_FIXED_BASE.entity_id, (byte_length, 4)),),
    )


def _local_byte_range_fields(base_offset: int) -> tuple:
    return (
        instruction_field(
            "constant_base_u16",
            base_offset,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Base of function-local dispatch-constant bytes.",
            (
                RuleUse(
                    LOCAL_BYTES_RANGE_BASE.entity_id,
                    (FieldReference("constant_count_u16"),),
                ),
            ),
        ),
        instruction_field(
            "constant_count_u16",
            base_offset + 2,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Exact dispatch-constant byte count.",
            (RuleUse(LOCAL_BYTES_RANGE_LENGTH.entity_id),),
        ),
    )


def _command_failures() -> tuple[FailureCase, ...]:
    return (
        *required_ref_failures(
            "hal.command_buffer", "No provider call occurs and VM state is unchanged."
        ),
        FailureCase(
            "provider_status",
            "The command-buffer provider entry returns any non-OK status.",
            "VM slots remain unchanged; the provider may already have mutated recording state.",
        ),
    )


def _direct_or_slot_failures() -> tuple[FailureCase, ...]:
    return (
        FailureCase(
            "invalid_argument",
            "A non-null direct buffer has the wrong exact descriptor, or direct/slot mode is malformed.",
            "No provider call occurs and recording state is unchanged.",
        ),
        FailureCase(
            "out_of_range",
            "A slot/offset/length cannot narrow or a direct live-buffer range is invalid.",
            "No provider call occurs and recording state is unchanged.",
        ),
    )


HAL_CMD_CREATE = hal_instruction(
    entity_id="hal.instruction.cmd.create",
    since=HAL_0,
    summary="Creates one command buffer and enters recording exactly once.",
    opcode=0x17,
    mnemonic="hal.cmd.create",
    byte_length=16,
    family_id=FAMILY.entity_id,
    fields=(
        hal_result_ref("dst_r8", 2, "hal.command_buffer"),
        hal_ref("device_r8", 3, "hal.device"),
        _flags("mode_u32", 4, 0x00000191),
        _flags("categories_u32", 8, 0x00000003),
        _value("affinity_v8", 12, "Complete u64 queue-affinity bitset."),
        _value(
            "binding_capacity_v8",
            13,
            "Number of indirect binding slots addressable at submission.",
        ),
        zero_padding("zero_padding_u8", 14, 2),
    ),
    semantics=InstructionSemantics(
        description=(
            "Creates one non-rerecordable command buffer and begins recording "
            "before transactionally publishing its owner."
        ),
        verification=(
            "Every register and padding byte must be valid; mode_u32 admits "
            "ONE_SHOT, ALLOW_INLINE_EXECUTION, RETAIN_PROFILE_METADATA, and "
            "RETAIN_DISPATCH_METADATA; categories_u32 admits transfer/dispatch.",
        ),
        preconditions=(
            "device_r8 must be a non-null exact hal.device. Binding capacity must "
            "fit host size and be at most 2^24. ALLOW_INLINE_EXECUTION requires "
            "ONE_SHOT and zero binding capacity.",
        ),
        success=(
            "Both create and begin return OK, then the complete recording command-"
            "buffer owner replaces dst_r8 and the program counter advances by 16 bytes.",
        ),
        failures=(
            *required_ref_failures(
                "hal.device", "No provider call occurs and dst_r8 remains unchanged."
            ),
            FailureCase(
                "out_of_range",
                "Binding capacity is not host-representable or exceeds 2^24.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "invalid_argument",
                "Inline mode lacks ONE_SHOT or has nonzero binding capacity.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "Create or begin returns any non-OK status.",
                "Any partially created command buffer is released and dst_r8 remains unchanged.",
            ),
        ),
        ownership=(
            "device_r8 is borrowed during create/begin. The new owner is published "
            "only after both provider calls succeed.",
        ),
        assembly=(
            "%r<dst> = hal.cmd.create %r<device>, %v<affinity>, "
            "%v<binding_capacity> {mode, categories}",
        ),
        pseudocode=(
            "device = require_hal_ref(refs[device_r8], hal_device_type);\n"
            "capacity = checked_binding_capacity(values[binding_capacity_v8]);\n"
            "check_command_buffer_mode(mode_u32, capacity);\n"
            "command_buffer = NULL;\n"
            "status = hal_command_buffer_create(device, mode_u32, categories_u32,\n"
            "    values[affinity_v8], capacity, &command_buffer);\n"
            "if (status succeeded) status = hal_command_buffer_begin(command_buffer);\n"
            "if (status failed) { release_if_nonnull(command_buffer); return status; }\n"
            "replace_ref(&refs[dst_r8], owned_ref(command_buffer, command_buffer_type));\n"
            "pc = pc + 16;"
        ),
    ),
)

HAL_CMD_FINALIZE = hal_instruction(
    entity_id="hal.instruction.cmd.finalize",
    since=HAL_0,
    summary="Permanently seals one recording command buffer without consuming it.",
    opcode=0x18,
    mnemonic="hal.cmd.finalize",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    semantics=InstructionSemantics(
        description="Ends recording permanently while retaining the VM ref owner.",
        verification=(
            "command_buffer_r8 must be a valid ref register and padding must be zero.",
        ),
        preconditions=(
            "command_buffer_r8 must be non-null exact hal.command_buffer in a HAL-valid "
            "recording state with balanced provider-managed debug groups.",
        ),
        success=(
            "Recording is permanently finalized, command_buffer_r8 remains unchanged, "
            "and the program counter advances by four bytes.",
        ),
        failures=_command_failures(),
        ownership=("command_buffer_r8 remains borrowed and unchanged.",),
        assembly=("hal.cmd.finalize %r<command_buffer>",),
        pseudocode=(
            "command_buffer = require_hal_ref(refs[command_buffer_r8], command_buffer_type);\n"
            "status = hal_command_buffer_end(command_buffer);\n"
            "if (status failed) return status;\n"
            "pc = pc + 4;"
        ),
    ),
)

HAL_CMD_EXECUTION_BARRIER = hal_instruction(
    entity_id="hal.instruction.cmd.execution.barrier",
    since=HAL_0,
    summary="Records independent global-memory and buffer-specific dependencies.",
    opcode=0x19,
    mnemonic="hal.cmd.execution.barrier",
    byte_length=32,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        _value("source_stage_v8", 3, "Unsigned source-stage mask."),
        _value("target_stage_v8", 4, "Unsigned target-stage mask."),
        zero_padding("zero_padding_u8", 5, 3),
        _zero_u32("flags_u32", 8),
        _range_field(
            "memory_source_scope_base_u16",
            12,
            _MEMORY_BARRIERS,
            InstructionFieldRole.RANGE_BASE,
            "Four-byte-aligned local base of u32 memory source scopes.",
        ),
        _range_field(
            "memory_target_scope_base_u16",
            14,
            _MEMORY_BARRIERS,
            InstructionFieldRole.RANGE_BASE,
            "Four-byte-aligned local base of u32 memory target scopes.",
        ),
        _range_field(
            "memory_count_u16",
            16,
            _MEMORY_BARRIERS,
            InstructionFieldRole.RANGE_COUNT,
            "Shared global-memory barrier row count.",
        ),
        _range_field(
            "buffer_source_scope_base_u16",
            18,
            _BUFFER_BARRIERS,
            InstructionFieldRole.RANGE_BASE,
            "Four-byte-aligned local base of u32 buffer source scopes.",
        ),
        _range_field(
            "buffer_target_scope_base_u16",
            20,
            _BUFFER_BARRIERS,
            InstructionFieldRole.RANGE_BASE,
            "Four-byte-aligned local base of u32 buffer target scopes.",
        ),
        _range_field(
            "buffer_ref_base_u16",
            22,
            _BUFFER_BARRIERS,
            InstructionFieldRole.RANGE_BASE,
            "Base ref slot of nullable direct buffers.",
        ),
        _range_field(
            "buffer_slot_base_u16",
            24,
            _BUFFER_BARRIERS,
            InstructionFieldRole.RANGE_BASE,
            "Four-byte-aligned local base of u32 indirect slots.",
        ),
        _range_field(
            "buffer_offset_base_u16",
            26,
            _BUFFER_BARRIERS,
            InstructionFieldRole.RANGE_BASE,
            "Eight-byte-aligned local base of u64 buffer offsets.",
        ),
        _range_field(
            "buffer_length_base_u16",
            28,
            _BUFFER_BARRIERS,
            InstructionFieldRole.RANGE_BASE,
            "Eight-byte-aligned local base of u64 buffer lengths.",
        ),
        _range_field(
            "buffer_count_u16",
            30,
            _BUFFER_BARRIERS,
            InstructionFieldRole.RANGE_COUNT,
            "Shared buffer-barrier row count.",
        ),
    ),
    range_groups=(MEMORY_BARRIER_RANGE, BUFFER_BARRIER_RANGE),
    semantics=InstructionSemantics(
        description=(
            "Records independent global-memory and direct-or-slot buffer dependency "
            "rows using one synchronous temporary adapter."
        ),
        verification=(
            "All scalar fields and padding must be valid; flags_u32 must be zero; "
            "the two independent range groups must obey bounds, alignment, and "
            "canonical empty-base rules.",
        ),
        preconditions=(
            "command_buffer_r8 must be non-null exact and recording. Stage masks "
            "must fit u32 and use only 0x0000003F; every scope must use only "
            "0x000003FF. Every buffer row must form a valid direct-or-slot ref.",
        ),
        success=(
            "HAL copies both barrier arrays into recording state and the program "
            "counter advances by 32 bytes. Empty lists and zero masks are valid.",
        ),
        failures=(
            *_command_failures(),
            *_direct_or_slot_failures(),
            FailureCase(
                "invalid_argument",
                "A stage or access-scope mask contains unknown bits.",
                "No provider call occurs and recording state is unchanged.",
            ),
            FailureCase(
                "resource_exhausted",
                "The exact native barrier rows do not fit invocation storage.",
                "No provider call occurs and recording state is unchanged.",
            ),
        ),
        ownership=(
            "The command buffer and direct buffers are borrowed only through the "
            "synchronous call; HAL copies rows and captures direct resources.",
        ),
        assembly=(
            "hal.cmd.execution.barrier %r<command_buffer>, %v<source_stages>, "
            "%v<target_stages>, #memory_barriers, #buffer_barriers",
        ),
        pseudocode=(
            "command_buffer = require_command_buffer(command_buffer_r8);\n"
            "source_stages = checked_stage_mask(values[source_stage_v8]);\n"
            "target_stages = checked_stage_mask(values[target_stage_v8]);\n"
            "adapter = prepare_barrier_rows(memory_barriers, buffer_barriers);\n"
            "status = hal_command_buffer_execution_barrier(command_buffer,\n"
            "    source_stages, target_stages, 0, adapter.memory_count,\n"
            "    adapter.memory_rows, adapter.buffer_count, adapter.buffer_rows);\n"
            "release_temporary_adapter(adapter);\n"
            "if (status failed) return status;\n"
            "pc = pc + 32;"
        ),
    ),
)

HAL_CMD_ADVISE_BUFFER = hal_instruction(
    entity_id="hal.instruction.cmd.advise.buffer",
    since=HAL_0,
    summary="Records an implementation-defined hint over one complete buffer ref.",
    opcode=0x1A,
    mnemonic="hal.cmd.advise.buffer",
    byte_length=16,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        _direct_or_slot_ref("buffer_r8_nullable", 3),
        _value("buffer_slot_v8", 4, "Unsigned direct-or-slot binding index."),
        _value("buffer_offset_v8", 5, "Unsigned buffer byte offset."),
        _value("buffer_length_v8", 6, "Unsigned buffer byte length."),
        zero_padding("zero_padding0_u8", 7, 1),
        _zero_u32("flags_u32", 8),
        _value("arg0_v8", 12, "Reserved zero dynamic advice argument zero."),
        _value("arg1_v8", 13, "Reserved zero dynamic advice argument one."),
        zero_padding("zero_padding1_u8", 14, 2),
    ),
    semantics=InstructionSemantics(
        description=(
            "Records a provider-defined no-flag advice operation; version zero "
            "requires both future dynamic arguments to be zero."
        ),
        verification=(
            "Every register and padding byte must be valid and flags_u32 must be zero.",
        ),
        preconditions=(
            "The command buffer must be non-null exact and recording; the buffer "
            "packet must form a complete direct-or-slot ref; arg0_v8 and arg1_v8 "
            "must both contain zero.",
        ),
        success=(
            "The provider records or deliberately ignores the hint and the program "
            "counter advances by 16 bytes.",
        ),
        failures=(
            *_command_failures(),
            *_direct_or_slot_failures(),
            FailureCase(
                "invalid_argument",
                "Either unassigned dynamic advice argument is nonzero.",
                "No provider call occurs and recording state is unchanged.",
            ),
        ),
        ownership=(
            "The command buffer and any direct buffer are borrowed for the call; "
            "HAL captures direct-resource state if it records the hint.",
        ),
        assembly=(
            "hal.cmd.advise.buffer %r<command_buffer>, "
            "%r<buffer>?/%v<slot>, %v<offset>, %v<length>, %v<arg0>, %v<arg1>",
        ),
        pseudocode=(
            "command_buffer = require_command_buffer(command_buffer_r8);\n"
            "buffer_ref = checked_direct_or_slot_ref(buffer_r8_nullable,\n"
            "    buffer_slot_v8, buffer_offset_v8, buffer_length_v8);\n"
            "if (values[arg0_v8] != 0 || values[arg1_v8] != 0) fail(invalid_argument);\n"
            "status = hal_command_buffer_advise_buffer(command_buffer,\n"
            "    buffer_ref, 0, 0, 0);\n"
            "if (status failed) return status;\n"
            "pc = pc + 16;"
        ),
    ),
)

HAL_CMD_FILL_BUFFER = hal_instruction(
    entity_id="hal.instruction.cmd.fill.buffer",
    since=HAL_0,
    summary="Records a repeating little-endian fill over one complete buffer ref.",
    opcode=0x1B,
    mnemonic="hal.cmd.fill.buffer",
    byte_length=16,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        _direct_or_slot_ref("target_buffer_r8_nullable", 3),
        _value("target_slot_v8", 4, "Unsigned target binding slot."),
        _value("target_offset_v8", 5, "Unsigned target byte offset."),
        _value("target_length_v8", 6, "Unsigned target byte length."),
        _value("pattern_v8", 7, "Low pattern bits in little-endian byte order."),
        instruction_field(
            "pattern_width_u8",
            8,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Repeating pattern width of one, two, or four bytes.",
            (RuleUse(ALLOWED_VALUES.entity_id, ((1, 2, 4),)),),
        ),
        zero_padding("zero_padding_u8", 9, 3),
        _zero_u32("flags_u32", 12),
    ),
    semantics=InstructionSemantics(
        description=(
            "Records one direct-or-slot fill using explicit low-byte extraction, "
            "making target bytes independent of host endianness."
        ),
        verification=(
            "All register/padding fields must be valid, pattern_width_u8 must be "
            "1, 2, or 4, and flags_u32 must be zero.",
        ),
        preconditions=(
            "The command buffer must be non-null exact and recording; the target "
            "packet must form a valid direct-or-slot ref; offset and length must "
            "both be multiples of pattern_width_u8.",
        ),
        success=(
            "HAL copies the pattern and captures any direct target, then the program "
            "counter advances by 16 bytes. Unused high pattern bits are ignored.",
        ),
        failures=(
            *_command_failures(),
            *_direct_or_slot_failures(),
            FailureCase(
                "invalid_argument",
                "Target offset or length is not a multiple of pattern width.",
                "No provider call occurs and recording state is unchanged.",
            ),
        ),
        ownership=(
            "The command/direct-buffer refs are borrowed synchronously; HAL copies "
            "the pattern bytes and captures the direct resource.",
        ),
        assembly=(
            "hal.cmd.fill.buffer %r<command_buffer>, %r<target>?/%v<slot>, "
            "%v<offset>, %v<length>, %v<pattern> {pattern_width}",
        ),
        pseudocode=(
            "command_buffer = require_command_buffer(command_buffer_r8);\n"
            "target = checked_direct_or_slot_ref(target_buffer_r8_nullable,\n"
            "    target_slot_v8, target_offset_v8, target_length_v8);\n"
            "require_multiple(target.offset, pattern_width_u8);\n"
            "require_multiple(target.length, pattern_width_u8);\n"
            "pattern = low_little_endian_bytes(values[pattern_v8], pattern_width_u8);\n"
            "status = hal_command_buffer_fill_buffer(command_buffer, target,\n"
            "    pattern, pattern_width_u8, 0);\n"
            "if (status failed) return status;\n"
            "pc = pc + 16;"
        ),
    ),
)

HAL_CMD_UPDATE_BUFFER = hal_instruction(
    entity_id="hal.instruction.cmd.update.buffer",
    since=HAL_0,
    summary="Copies readable host bytes immediately into recorded command state.",
    opcode=0x1C,
    mnemonic="hal.cmd.update.buffer",
    byte_length=16,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        hal_ref("source_vm_buffer_r8", 3, "vm.buffer"),
        _value("source_offset_v8", 4, "Unsigned source host-byte offset."),
        _direct_or_slot_ref("target_buffer_r8_nullable", 5),
        _value("target_slot_v8", 6, "Unsigned target binding slot."),
        _value("target_offset_v8", 7, "Unsigned target device-byte offset."),
        _value("target_length_v8", 8, "Exact source/target byte length."),
        zero_padding("zero_padding_u8", 9, 3),
        _zero_u32("flags_u32", 12),
    ),
    semantics=InstructionSemantics(
        description=(
            "Copies one readable core vm.buffer range synchronously into the "
            "command buffer's immediate update data for a direct-or-slot target."
        ),
        verification=(
            "All register/padding fields must be valid and flags_u32 must be zero.",
        ),
        preconditions=(
            "The command buffer and source must be non-null exact; source must be "
            "open, READ-capable, host-size representable, and contain the complete "
            "range. The target packet must be valid and its device-sized length "
            "must equal the selected source length.",
        ),
        success=(
            "HAL has copied all selected host bytes before returning; no source "
            "pointer remains in recorded state and the program counter advances by 16 bytes.",
        ),
        failures=(
            *_command_failures(),
            *required_ref_failures(
                "vm.buffer", "No provider call occurs and recording state is unchanged."
            ),
            *_direct_or_slot_failures(),
            FailureCase(
                "failed_precondition",
                "The source vm.buffer root is closed.",
                "No provider call occurs and recording state is unchanged.",
            ),
            FailureCase(
                "permission_denied",
                "The source vm.buffer lacks READ access.",
                "No provider call occurs and recording state is unchanged.",
            ),
            FailureCase(
                "out_of_range",
                "The source offset/length cannot fit host size or exceeds the source.",
                "No provider call occurs and recording state is unchanged.",
            ),
        ),
        ownership=(
            "The source and command/direct-target refs are borrowed only while HAL "
            "copies bytes and captures target state synchronously.",
        ),
        assembly=(
            "hal.cmd.update.buffer %r<command_buffer>, %r<source>, "
            "%v<source_offset>, %r<target>?/%v<slot>, %v<target_offset>, %v<length>",
        ),
        pseudocode=(
            "command_buffer = require_command_buffer(command_buffer_r8);\n"
            "source = require_hal_ref(refs[source_vm_buffer_r8], vm_buffer_type);\n"
            "source_offset = checked_host_size(values[source_offset_v8]);\n"
            "length = checked_host_size(values[target_length_v8]);\n"
            "check_vm_buffer_read_range(source, source_offset, length);\n"
            "target = checked_direct_or_slot_ref(target_buffer_r8_nullable,\n"
            "    target_slot_v8, target_offset_v8, target_length_v8);\n"
            "status = hal_command_buffer_update_buffer(command_buffer,\n"
            "    nonnull_empty_safe_data(source), source_offset, target, 0);\n"
            "if (status failed) return status;\n"
            "pc = pc + 16;"
        ),
    ),
)

HAL_CMD_COPY_BUFFER = hal_instruction(
    entity_id="hal.instruction.cmd.copy.buffer",
    since=HAL_0,
    summary="Records one byte-length copy between complete buffer refs.",
    opcode=0x1D,
    mnemonic="hal.cmd.copy.buffer",
    byte_length=16,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        _direct_or_slot_ref("source_buffer_r8_nullable", 3),
        _value("source_slot_v8", 4, "Unsigned source binding slot."),
        _value("source_offset_v8", 5, "Unsigned source device-byte offset."),
        _direct_or_slot_ref("target_buffer_r8_nullable", 6),
        _value("target_slot_v8", 7, "Unsigned target binding slot."),
        _value("target_offset_v8", 8, "Unsigned target device-byte offset."),
        _value("length_v8", 9, "Exact copy byte length."),
        zero_padding("zero_padding_u8", 10, 2),
        _zero_u32("flags_u32", 12),
    ),
    semantics=InstructionSemantics(
        description=(
            "Records equal-length direct-or-slot source and target ranges; HAL "
            "performs the definitive alias check when indirect bindings resolve."
        ),
        verification=(
            "All register/padding fields must be valid and flags_u32 must be zero.",
        ),
        preconditions=(
            "The command buffer must be non-null exact and recording. Length must "
            "fit device size; both packets must be valid direct-or-slot refs of that "
            "length. Resolved physical source/target ranges must be disjoint.",
        ),
        success=(
            "HAL captures direct buffers or records indirect slots and the program "
            "counter advances by 16 bytes.",
        ),
        failures=(
            *_command_failures(),
            *_direct_or_slot_failures(),
            FailureCase(
                "invalid_argument",
                "Definitively resolved source and target ranges overlap.",
                "HAL may discover this at recording/submission; provider recording state follows its returned status.",
            ),
        ),
        ownership=(
            "VM refs remain unchanged. HAL captures direct buffers; indirect slots "
            "carry no VM object lifetime.",
        ),
        assembly=(
            "hal.cmd.copy.buffer %r<command_buffer>, %r<source>?/%v<source_slot>, "
            "%v<source_offset>, %r<target>?/%v<target_slot>, "
            "%v<target_offset>, %v<length>",
        ),
        pseudocode=(
            "command_buffer = require_command_buffer(command_buffer_r8);\n"
            "length = checked_device_size(values[length_v8]);\n"
            "source = checked_direct_or_slot_ref_with_length(source_buffer_r8_nullable,\n"
            "    source_slot_v8, source_offset_v8, length);\n"
            "target = checked_direct_or_slot_ref_with_length(target_buffer_r8_nullable,\n"
            "    target_slot_v8, target_offset_v8, length);\n"
            "status = hal_command_buffer_copy_buffer(command_buffer, source, target, 0);\n"
            "if (status failed) return status;\n"
            "pc = pc + 16;"
        ),
    ),
)


def _collective_op_field():
    components = tuple(
        (
            component_name,
            bit_offset,
            8,
            EntityReference(SELECTOR_TABLES_BY_NAME[selector_name].entity_id),
            (),
        )
        for component_name, bit_offset, selector_name in (
            ("kind", 0, "hal.collective.kind"),
            ("reduction", 8, "hal.collective.reduction"),
            ("element", 16, "hal.collective.element"),
        )
    )
    return instruction_field(
        "op_u32",
        4,
        U32.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Packed collective kind, reduction, and element selectors.",
        (RuleUse(PACKED_SELECTORS.entity_id, (0xFF000000, components)),),
    )


HAL_CMD_COLLECTIVE = hal_instruction(
    entity_id="hal.instruction.cmd.collective",
    since=HAL_0,
    summary="Records one context-sensitive channel collective.",
    opcode=0x1E,
    mnemonic="hal.cmd.collective",
    byte_length=20,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        hal_ref("channel_r8", 3, "hal.channel"),
        _collective_op_field(),
        _value("param_v8", 8, "Collective root/peer packed u32 parameter."),
        _direct_or_slot_ref("send_buffer_r8_nullable", 9),
        _value("send_slot_v8", 10, "Unsigned send binding slot."),
        _value("send_offset_v8", 11, "Unsigned send device-byte offset."),
        _value("send_length_v8", 12, "Available send byte length."),
        _direct_or_slot_ref("recv_buffer_r8_nullable", 13),
        _value("recv_slot_v8", 14, "Unsigned receive binding slot."),
        _value("recv_offset_v8", 15, "Unsigned receive device-byte offset."),
        _value("recv_length_v8", 16, "Available receive byte length."),
        _value("element_count_v8", 17, "Logical collective element count."),
        zero_padding("zero_padding_u8", 18, 2),
    ),
    constraints=(
        RuleUse(
            PACKED_SELECTOR_INDEXED_ALLOWED_MASK.entity_id,
            (
                FieldReference("op_u32"),
                "kind",
                "reduction",
                (0x01, 0x3E, 0x01, 0x01, 0x3E, 0x3E, 0x01, 0x01, 0x01),
            ),
        ),
    ),
    semantics=InstructionSemantics(
        description=(
            "Queries immutable channel rank/count, derives context-dependent send "
            "and receive use/byte requirements from the verified packed operation, "
            "and records one collective."
        ),
        verification=(
            "Every register/padding field must be valid; op_u32's high byte must be "
            "zero; kind/reduction/element must be assigned selectors; ALL_REDUCE, "
            "REDUCE, and REDUCE_SCATTER require reductions 1..5 and every other "
            "kind requires NONE=0.",
        ),
        preconditions=(
            "The command buffer and channel must be non-null exact. param_v8 must "
            "fit u32 and element_count_v8 must fit device size. Root/peer ranks "
            "must satisfy the selected kind and checked byte-size multiplication "
            "must not overflow.",
            "Each contextually used side must form a valid direct-or-slot ref at "
            "least as large as its required bytes. An unused side must be the exact "
            "canonical all-zero ref/slot/offset/length packet.",
        ),
        success=(
            "HAL records the same contextual-side contract, captures channel and "
            "used direct resources, and the program counter advances by 20 bytes. "
            "A required null ref with slot zero remains valid indirect slot zero, "
            "including for zero elements.",
        ),
        failures=(
            *_command_failures(),
            *required_ref_failures(
                "hal.channel",
                "No collective call occurs and recording state is unchanged.",
            ),
            *_direct_or_slot_failures(),
            FailureCase(
                "invalid_argument",
                "An unused side is not canonical zero or ALL_TO_ALL count is not divisible by channel count.",
                "No collective call occurs and recording state is unchanged.",
            ),
            FailureCase(
                "out_of_range",
                "param/count/rank, checked byte multiplication, or required side length is invalid.",
                "No collective call occurs and recording state is unchanged.",
            ),
        ),
        ownership=(
            "The command buffer/channel/direct buffers are borrowed synchronously. "
            "HAL captures channel and used direct resources; unused and indirect "
            "sides capture no VM ref.",
        ),
        assembly=(
            "hal.cmd.collective %r<command_buffer>, %r<channel>, {op}, %v<param>, "
            "%r<send>?/%v<send_slot>, %v<send_offset>, %v<send_length>, "
            "%r<recv>?/%v<recv_slot>, %v<recv_offset>, %v<recv_length>, "
            "%v<element_count>",
        ),
        pseudocode=(
            "command_buffer = require_command_buffer(command_buffer_r8);\n"
            "channel = require_hal_ref(refs[channel_r8], hal_channel_type);\n"
            "op = verified_collective_op(op_u32);\n"
            "param = checked_u32(values[param_v8]);\n"
            "element_count = checked_device_size(values[element_count_v8]);\n"
            "rank, count = hal_channel_query_rank_and_count(channel);\n"
            "sides = resolve_collective_sides_and_lengths(\n"
            "    op, param, rank, count, element_count);\n"
            "send = sides.send_used ? checked_ref_at_least(send_packet, sides.send_bytes)\n"
            "    : require_canonical_zero_ref_packet(send_packet);\n"
            "recv = sides.recv_used ? checked_ref_at_least(recv_packet, sides.recv_bytes)\n"
            "    : require_canonical_zero_ref_packet(recv_packet);\n"
            "status = hal_command_buffer_collective(command_buffer, channel, op,\n"
            "    param, send, recv, element_count);\n"
            "if (status failed) return status;\n"
            "pc = pc + 20;"
        ),
    ),
)


def _barrier_before_field():
    table = SELECTOR_TABLES_BY_NAME["hal.cmd.dispatch.barrier_before"]
    return instruction_field(
        "barrier_before_u8",
        5,
        U8.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "NONE=0 or full device phase barrier ALL=1.",
        (RuleUse(SELECTOR.entity_id, (EntityReference(table.entity_id),)),),
    )


def _dispatch_binding_fields(base_offset: int) -> tuple:
    return (
        _range_field(
            "binding_buffer_base_u16",
            base_offset,
            _DISPATCH_BINDINGS,
            InstructionFieldRole.RANGE_BASE,
            "Base ref slot of nullable direct binding buffers.",
        ),
        _range_field(
            "binding_slot_base_u16",
            base_offset + 2,
            _DISPATCH_BINDINGS,
            InstructionFieldRole.RANGE_BASE,
            "Four-byte-aligned local base of u32 binding slots.",
        ),
        _range_field(
            "binding_offset_base_u16",
            base_offset + 4,
            _DISPATCH_BINDINGS,
            InstructionFieldRole.RANGE_BASE,
            "Eight-byte-aligned local base of u64 binding offsets.",
        ),
        _range_field(
            "binding_length_base_u16",
            base_offset + 6,
            _DISPATCH_BINDINGS,
            InstructionFieldRole.RANGE_BASE,
            "Eight-byte-aligned local base of u64 binding lengths.",
        ),
        _range_field(
            "binding_count_u16",
            base_offset + 8,
            _DISPATCH_BINDINGS,
            InstructionFieldRole.RANGE_COUNT,
            "Shared direct-or-slot binding row count.",
        ),
    )


def _dispatch_failures(*, indirect: bool) -> tuple[FailureCase, ...]:
    indirect_failures = (
        (
            *_direct_or_slot_failures(),
            FailureCase(
                "invalid_argument",
                "The indirect workgroup-count offset is not four-byte aligned.",
                "No barrier or dispatch call occurs and recording state is unchanged.",
            ),
        )
        if indirect
        else ()
    )
    return (
        *required_ref_failures(
            "hal.command_buffer",
            "No barrier/dispatch call occurs and recording state is unchanged.",
        ),
        *required_ref_failures(
            "hal.executable_function_table",
            "No barrier/dispatch call occurs and recording state is unchanged.",
        ),
        *indirect_failures,
        FailureCase(
            "invalid_argument",
            "A binding row has wrong direct-buffer type/mode or workgroup-size lanes mix zero/nonzero.",
            "No barrier or dispatch call occurs and recording state is unchanged.",
        ),
        FailureCase(
            "out_of_range",
            "The function ordinal or any binding/offset/length range is invalid.",
            "No barrier or dispatch call occurs and recording state is unchanged.",
        ),
        FailureCase(
            "resource_exhausted",
            "The exact native binding rows do not fit invocation storage.",
            "No barrier or dispatch call occurs and recording state is unchanged.",
        ),
        FailureCase(
            "provider_status",
            "The optional full barrier returns failure.",
            "Dispatch is not called; the provider defines any partial barrier recording.",
        ),
        FailureCase(
            "provider_status",
            "Dispatch returns failure after an optional barrier succeeded.",
            "The barrier may remain recorded and provider recording state is otherwise unspecified.",
        ),
    )


HAL_CMD_DISPATCH = hal_instruction(
    entity_id="hal.instruction.cmd.dispatch",
    since=HAL_0,
    summary="Records a static-count table dispatch with an optional fused barrier.",
    opcode=0x1F,
    mnemonic="hal.cmd.dispatch",
    byte_length=28,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        hal_ref(
            "function_table_r8",
            3,
            "hal.executable_function_table",
        ),
        _value("function_ordinal_v8", 4, "Unsigned selected function ordinal."),
        _barrier_before_field(),
        _local_fixed_base(
            "launch_base_u16",
            6,
            28,
            "Four-byte-aligned local base of seven u32 launch lanes.",
        ),
        *_local_byte_range_fields(8),
        *_dispatch_binding_fields(12),
        zero_padding("zero_padding_u8", 22, 2),
        _flags("flags_u32", 24, 0x00000020),
    ),
    range_groups=(DISPATCH_BINDING_RANGE,),
    semantics=InstructionSemantics(
        description=(
            "Preflights a complete static-count table dispatch, optionally records "
            "one fixed full device phase barrier, then records the dispatch."
        ),
        verification=(
            "All scalar/padding fields must be valid; launch_base_u16 addresses "
            "exactly seven aligned u32 lanes; constant and direct-or-slot binding "
            "ranges obey bounds/canonical-empty rules; barrier_before is NONE/ALL; "
            "flags_u32 may contain only ALLOW_INLINE_EXECUTION.",
        ),
        preconditions=(
            "Command buffer and function table must be non-null exact and recording-"
            "compatible; the function ordinal must fit host size and be in table. "
            "Each binding packet must be valid; workgroup sizes are all zero or all "
            "nonzero. Every VM check and adapter reservation finishes before barrier.",
        ),
        success=(
            "ALL records the fixed 0x1F stage / 0x33F memory-scope global barrier "
            "before dispatch even for an empty workgroup count. HAL copies constants "
            "and binding rows, captures direct resources, and the PC advances by 28 bytes.",
        ),
        failures=_dispatch_failures(indirect=False),
        ownership=(
            "Command buffer, table, constants, and direct buffers are borrowed only "
            "during recording; HAL copies/captures everything required afterward.",
        ),
        assembly=(
            "hal.cmd.dispatch %r<command_buffer>, %r<function_table>, "
            "%v<function_ordinal>, #launch7, #constants, #bindings "
            "{barrier_before, flags}",
        ),
        pseudocode=(
            "config = load_static_launch7(launch_base_u16);\n"
            "check_workgroup_size(config.workgroup_size);\n"
            "dispatch = preflight_command_dispatch(table, ordinal, constants, bindings);\n"
            "if (barrier_before_u8 == ALL) {\n"
            "  status = record_full_device_barrier(command_buffer);\n"
            "  if (status failed) { release_adapter(dispatch); return status; }\n"
            "}\n"
            "status = hal_command_buffer_dispatch_table(command_buffer, dispatch.table,\n"
            "    dispatch.ordinal, config, dispatch.constants, dispatch.bindings, flags_u32);\n"
            "release_adapter(dispatch);\n"
            "if (status failed) return status;\n"
            "pc = pc + 28;"
        ),
    ),
)

HAL_CMD_DISPATCH_INDIRECT_COUNT = hal_instruction(
    entity_id="hal.instruction.cmd.dispatch.indirect.count",
    since=HAL_0,
    summary="Records an indirect-count table dispatch with an optional fused barrier.",
    opcode=0x20,
    mnemonic="hal.cmd.dispatch.indirect.count",
    byte_length=32,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        hal_ref(
            "function_table_r8",
            3,
            "hal.executable_function_table",
        ),
        _value("function_ordinal_v8", 4, "Unsigned selected function ordinal."),
        _barrier_before_field(),
        _local_fixed_base(
            "launch_base_u16",
            6,
            16,
            "Four-byte-aligned local base of four u32 launch lanes.",
        ),
        _direct_or_slot_ref("workgroup_count_buffer_r8_nullable", 8),
        _value("workgroup_count_slot_v8", 9, "Unsigned indirect count binding slot."),
        _value(
            "workgroup_count_offset_v8",
            10,
            "Four-byte-aligned offset of exactly three u32 counts.",
        ),
        zero_padding("zero_padding0_u8", 11, 1),
        *_local_byte_range_fields(12),
        *_dispatch_binding_fields(16),
        zero_padding("zero_padding1_u8", 26, 2),
        _flags("flags_u32", 28, 0x00000023, required_one_mask=0x00000003),
    ),
    range_groups=(DISPATCH_BINDING_RANGE,),
    semantics=InstructionSemantics(
        description=(
            "Preflights a table dispatch whose three counts come from one exact "
            "direct-or-slot 12-byte range, optionally records a full phase barrier, "
            "then records the dispatch."
        ),
        verification=(
            "All scalar/padding fields must be valid; launch_base_u16 addresses "
            "exactly four aligned u32 lanes; constant/binding ranges are valid; "
            "barrier_before is NONE/ALL; flags admit bit 5 and exactly one of "
            "dynamic/static indirect-parameter bits 0 and 1.",
        ),
        preconditions=(
            "The common command-dispatch contract applies. The workgroup-count "
            "packet must be a valid direct-or-slot ref of exactly 12 bytes with "
            "four-byte-aligned offset. Dynamic parameters require an explicit "
            "dependency; barrier_before=ALL supplies the broad version-zero edge.",
        ),
        success=(
            "An optional fixed full barrier precedes the recorded indirect dispatch. "
            "HAL captures direct resources and the program counter advances by 32 bytes.",
        ),
        failures=_dispatch_failures(indirect=True),
        ownership=(
            "Every operand is borrowed synchronously. HAL captures table/direct "
            "resources and records indirect slots without VM lifetime.",
        ),
        assembly=(
            "hal.cmd.dispatch.indirect.count %r<command_buffer>, %r<function_table>, "
            "%v<function_ordinal>, #launch4, %r<count_buffer>?/%v<count_slot>, "
            "%v<count_offset>, #constants, #bindings {barrier_before, flags}",
        ),
        pseudocode=(
            "config = load_indirect_launch4(launch_base_u16);\n"
            "check_workgroup_size(config.workgroup_size);\n"
            "config.count_ref = checked_direct_or_slot_ref_with_length(\n"
            "    workgroup_count_buffer_r8_nullable, workgroup_count_slot_v8,\n"
            "    workgroup_count_offset_v8, 12);\n"
            "require_multiple(config.count_ref.offset, 4);\n"
            "dispatch = preflight_command_dispatch(table, ordinal, constants, bindings);\n"
            "if (barrier_before_u8 == ALL) {\n"
            "  status = record_full_device_barrier(command_buffer);\n"
            "  if (status failed) { release_adapter(dispatch); return status; }\n"
            "}\n"
            "status = hal_command_buffer_dispatch_table(command_buffer, dispatch.table,\n"
            "    dispatch.ordinal, config, dispatch.constants, dispatch.bindings, flags_u32);\n"
            "release_adapter(dispatch);\n"
            "if (status failed) return status;\n"
            "pc = pc + 32;"
        ),
    ),
)

HAL_CMD_DEBUG_GROUP_BEGIN = hal_instruction(
    entity_id="hal.instruction.cmd.debug.group.begin",
    since=HAL_0,
    summary="Begins one provider-managed diagnostic label group.",
    opcode=0x21,
    mnemonic="hal.cmd.debug.group.begin",
    byte_length=12,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        zero_padding("zero_padding0_u8", 3, 1),
        instruction_field(
            "label_string_u16",
            4,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Module string ordinal; empty labels are valid.",
            (RuleUse(STRING_ORDINAL.entity_id),),
        ),
        zero_padding("zero_padding1_u8", 6, 2),
        instruction_field(
            "color_u32",
            8,
            U32.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Little-endian RGBA8 channels; zero means unspecified.",
            (RuleUse(ANY_BITS.entity_id),),
        ),
    ),
    semantics=InstructionSemantics(
        description=(
            "Begins one nested diagnostic group with copied label bytes, explicit "
            "host-endian-independent RGBA extraction, and no source location."
        ),
        verification=(
            "command_buffer_r8, label_string_u16, and padding must be valid; "
            "color_u32 accepts every bit pattern.",
        ),
        preconditions=(
            "command_buffer_r8 must be non-null exact, recording, and accept another debug group.",
        ),
        success=(
            "HAL copies any label bytes, begins the provider-managed group, and the "
            "program counter advances by 12 bytes. Empty label and zero color are valid.",
        ),
        failures=_command_failures(),
        ownership=(
            "The command buffer and module string view are borrowed synchronously; "
            "HAL copies label bytes needed after return.",
        ),
        assembly=(
            "hal.cmd.debug.group.begin %r<command_buffer>, @label<string> {color}",
        ),
        pseudocode=(
            "command_buffer = require_command_buffer(command_buffer_r8);\n"
            "color = rgba8_from_little_endian_u32(color_u32);\n"
            "status = hal_command_buffer_begin_debug_group(command_buffer,\n"
            "    strings[label_string_u16], color, NULL);\n"
            "if (status failed) return status;\n"
            "pc = pc + 12;"
        ),
    ),
)

HAL_CMD_DEBUG_GROUP_END = hal_instruction(
    entity_id="hal.instruction.cmd.debug.group.end",
    since=HAL_0,
    summary="Ends the innermost provider-managed diagnostic group.",
    opcode=0x22,
    mnemonic="hal.cmd.debug.group.end",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("command_buffer_r8", 2, "hal.command_buffer"),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    semantics=InstructionSemantics(
        description="Closes the innermost HAL-managed debug group.",
        verification=(
            "command_buffer_r8 must be a valid ref register and padding must be zero.",
        ),
        preconditions=(
            "command_buffer_r8 must be non-null exact, recording, and have an open debug group.",
        ),
        success=(
            "HAL closes the group and the program counter advances by four bytes.",
        ),
        failures=_command_failures(),
        ownership=("command_buffer_r8 is borrowed and remains unchanged.",),
        assembly=("hal.cmd.debug.group.end %r<command_buffer>",),
        pseudocode=(
            "command_buffer = require_command_buffer(command_buffer_r8);\n"
            "status = hal_command_buffer_end_debug_group(command_buffer);\n"
            "if (status failed) return status;\n"
            "pc = pc + 4;"
        ),
    ),
)

INSTRUCTIONS = (
    HAL_CMD_CREATE,
    HAL_CMD_FINALIZE,
    HAL_CMD_EXECUTION_BARRIER,
    HAL_CMD_ADVISE_BUFFER,
    HAL_CMD_FILL_BUFFER,
    HAL_CMD_UPDATE_BUFFER,
    HAL_CMD_COPY_BUFFER,
    HAL_CMD_COLLECTIVE,
    HAL_CMD_DISPATCH,
    HAL_CMD_DISPATCH_INDIRECT_COUNT,
    HAL_CMD_DEBUG_GROUP_BEGIN,
    HAL_CMD_DEBUG_GROUP_END,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
