# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL 0.0 asynchronous queue-submission instructions."""

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
    StateEffect,
    StateResource,
)
from model.isa.declarations import (
    hal_instruction,
    instruction_field,
    state_allocate,
    state_read,
    state_release,
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
from model.isa.validation import (
    ALLOWED_BITS,
    ALLOWED_BITS_EXACTLY_ONE,
    ALLOWED_VALUES,
    LOCAL_BYTES_FIXED_BASE,
    LOCAL_BYTES_RANGE_BASE,
    LOCAL_BYTES_RANGE_LENGTH,
    RANGE_BASE,
    RANGE_COUNT,
    ZERO,
)
from model.schema import U8, U16, U32, FieldReference, RuleUse
from model.specification import HAL_0

FAMILY = InstructionFamily(
    entity_id="hal.family.queue",
    since=HAL_0,
    summary="Queue-ordered allocation, transfer, dispatch, execution, and barriers.",
    dependencies=("hal.contract.abi",),
    document_order=15,
    normative_text=(
        "Queue records synchronously submit or flush work and never suspend the "
        "VM invocation; submitted work may complete later. Submission order is "
        "not FIFO and creates no user-visible ordering beyond each operation's "
        "explicit wait and signal semaphore edges. Before provider entry the VM "
        "validates every scalar ref/value and complete counted range, computes "
        "the exact adapter size with checked host arithmetic, and reserves one "
        "temporary region in the invocation's LIFO stack. It then materializes "
        "borrowed native pointer/payload/row arrays, calls HAL, and releases the "
        "temporary region synchronously. HAL must copy every array and capture "
        "every non-device object or host byte range it needs after returning. "
        "The VM performs no per-element retain pass. The host/process keeps the "
        "selected device and enclosing group/topology authority alive across "
        "submitted work. Empty lists remain real operands: an empty barrier still "
        "calls HAL and neither implies prior work nor substitutes for flush."
    ),
)

_WAITS = "queue_waits"
_SIGNALS = "queue_signals"
_DIRECT_BINDINGS = "queue_direct_bindings"
_EXECUTE_BINDINGS = "queue_execute_bindings"


def _semaphore_range(name: str, prefix: str) -> InstructionRangeGroup:
    return InstructionRangeGroup(
        name=name,
        count_field=f"{prefix}_count_u16",
        members=(
            InstructionRangeMember(
                base_field=f"{prefix}_semaphore_base_u16",
                storage=InstructionRangeStorage.LOCAL_REFS,
                element_byte_length=1,
                element_alignment=1,
                runtime_ref_policy=RuntimeRefPolicy(
                    "hal.semaphore", RefNullPolicy.REQUIRED, RefOwnership.BORROW
                ),
            ),
            InstructionRangeMember(
                base_field=f"{prefix}_payload_base_u16",
                storage=InstructionRangeStorage.LOCAL_BYTES,
                element_byte_length=8,
                element_alignment=8,
            ),
        ),
    )


QUEUE_WAIT_RANGE = _semaphore_range(_WAITS, "wait")
QUEUE_SIGNAL_RANGE = _semaphore_range(_SIGNALS, "signal")


def _binding_range(name: str, *, nullable: bool) -> InstructionRangeGroup:
    return InstructionRangeGroup(
        name=name,
        count_field="binding_count_u16",
        members=(
            InstructionRangeMember(
                base_field="binding_buffer_base_u16",
                storage=InstructionRangeStorage.LOCAL_REFS,
                element_byte_length=1,
                element_alignment=1,
                runtime_ref_policy=RuntimeRefPolicy(
                    "hal.buffer",
                    RefNullPolicy.NULLABLE if nullable else RefNullPolicy.REQUIRED,
                    RefOwnership.BORROW,
                ),
            ),
            InstructionRangeMember(
                base_field="binding_offset_base_u16",
                storage=InstructionRangeStorage.LOCAL_BYTES,
                element_byte_length=8,
                element_alignment=8,
            ),
            InstructionRangeMember(
                base_field="binding_length_base_u16",
                storage=InstructionRangeStorage.LOCAL_BYTES,
                element_byte_length=8,
                element_alignment=8,
            ),
        ),
    )


QUEUE_DIRECT_BINDING_RANGE = _binding_range(_DIRECT_BINDINGS, nullable=False)
QUEUE_EXECUTE_BINDING_RANGE = _binding_range(_EXECUTE_BINDINGS, nullable=True)


def _value(name: str, offset: int, description: str):
    return value_register(name, offset, InstructionFieldRole.OPERAND, description)


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


def _semaphore_fields(wait_offset: int) -> tuple:
    signal_offset = wait_offset + 6
    return (
        _range_field(
            "wait_semaphore_base_u16",
            wait_offset,
            _WAITS,
            InstructionFieldRole.RANGE_BASE,
            "Base ref slot of exact non-null wait semaphores.",
        ),
        _range_field(
            "wait_payload_base_u16",
            wait_offset + 2,
            _WAITS,
            InstructionFieldRole.RANGE_BASE,
            "Eight-byte-aligned local base of u64 wait payloads.",
        ),
        _range_field(
            "wait_count_u16",
            wait_offset + 4,
            _WAITS,
            InstructionFieldRole.RANGE_COUNT,
            "Shared wait semaphore/payload count.",
        ),
        _range_field(
            "signal_semaphore_base_u16",
            signal_offset,
            _SIGNALS,
            InstructionFieldRole.RANGE_BASE,
            "Base ref slot of exact non-null signal semaphores.",
        ),
        _range_field(
            "signal_payload_base_u16",
            signal_offset + 2,
            _SIGNALS,
            InstructionFieldRole.RANGE_BASE,
            "Eight-byte-aligned local base of u64 signal payloads.",
        ),
        _range_field(
            "signal_count_u16",
            signal_offset + 4,
            _SIGNALS,
            InstructionFieldRole.RANGE_COUNT,
            "Shared signal semaphore/payload count.",
        ),
    )


def _queue_common_fields() -> tuple:
    return (
        hal_ref("device_r8", 2, "hal.device"),
        _value("affinity_v8", 3, "Complete u64 queue-affinity bitset."),
        *_semaphore_fields(4),
    )


def _flags(offset: int, allowed_mask: int, *, required_one_mask: int = 0):
    if required_one_mask:
        rule = RuleUse(
            ALLOWED_BITS_EXACTLY_ONE.entity_id,
            (allowed_mask, required_one_mask),
        )
    else:
        rule = RuleUse(ALLOWED_BITS.entity_id, (allowed_mask,))
    return instruction_field(
        "flags_u32",
        offset,
        U32.entity_id,
        InstructionFieldRole.IMMEDIATE,
        f"Architectural flag bits constrained by mask 0x{allowed_mask:08X}.",
        (rule,),
    )


def _zero_u32(offset: int):
    return instruction_field(
        "flags_u32",
        offset,
        U32.entity_id,
        InstructionFieldRole.PADDING,
        "Reserved flags; canonical zero.",
        (RuleUse(ZERO.entity_id),),
    )


def _common_queue_failures() -> tuple[FailureCase, ...]:
    return (
        *required_ref_failures(
            "hal.device", "No adapter storage is retained and no HAL call occurs."
        ),
        FailureCase(
            "failed_precondition",
            "Any wait or signal semaphore element is canonical null.",
            "No native list is exposed to HAL and no HAL call occurs.",
        ),
        FailureCase(
            "invalid_argument",
            "Any non-null wait or signal element has the wrong exact descriptor.",
            "No native list is exposed to HAL and no HAL call occurs.",
        ),
        FailureCase(
            "out_of_range",
            "Any wait or signal payload lies in HAL's reserved failure domain.",
            "No native list is exposed to HAL and no HAL call occurs.",
        ),
        FailureCase(
            "resource_exhausted",
            "The exact temporary native adapter rows do not fit invocation storage.",
            "No HAL call occurs and no VM state changes.",
        ),
    )


def _common_verification() -> str:
    return (
        "The device/affinity fields and both semaphore struct-of-arrays ranges "
        "must be valid; empty ranges require every owned base to be zero and "
        "nonempty payload arrays must be eight-byte aligned."
    )


def _common_preconditions() -> str:
    return (
        "device_r8 and every wait/signal semaphore must be non-null exact "
        "objects, and every payload must be no greater than 0x000000007FFFFFFE."
    )


def _queue_state_effects(
    *operation_effects: StateEffect,
) -> tuple[StateEffect, ...]:
    return (
        state_read(
            StateResource.FRAME_LOCALS,
            "wait_semaphore_base_u16",
            "wait_payload_base_u16",
            "signal_semaphore_base_u16",
            "signal_payload_base_u16",
        ),
        state_write(StateResource.HAL_QUEUE, "device_r8", "affinity_v8"),
        state_read(StateResource.HAL_SEMAPHORE, "wait_semaphore_base_u16"),
        state_synchronize(
            StateResource.HAL_SEMAPHORE,
            "wait_semaphore_base_u16",
        ),
        state_write(
            StateResource.HAL_SEMAPHORE,
            "signal_semaphore_base_u16",
        ),
        state_synchronize(
            StateResource.HAL_SEMAPHORE,
            "signal_semaphore_base_u16",
        ),
        *operation_effects,
    )


HAL_QUEUE_ALLOCA = hal_instruction(
    entity_id="hal.instruction.queue.alloca",
    since=HAL_0,
    summary="Returns one queue-ordered transient allocation.",
    opcode=0x0B,
    mnemonic="hal.queue.alloca",
    byte_length=28,
    family_id=FAMILY.entity_id,
    fields=(
        hal_result_ref("dst_r8", 2, "hal.buffer"),
        hal_ref("device_r8", 3, "hal.device"),
        _value("affinity_v8", 4, "Complete u64 queue-affinity bitset."),
        hal_ref(
            "pool_r8_nullable",
            5,
            "hal.pool",
            null_policy=RefNullPolicy.NULLABLE,
        ),
        *_semaphore_fields(6),
        _value("usage_v8", 18, "Unsigned HAL buffer-usage flag bits."),
        _value("access_v8", 19, "Unsigned HAL memory-access flag bits."),
        _value("memory_type_v8", 20, "Unsigned HAL memory-type flag bits."),
        _value(
            "memory_affinity_v8",
            21,
            "Complete u64 allocation-affinity bitset.",
        ),
        _value("min_alignment_v8", 22, "Minimum device-byte alignment."),
        _value("allocation_size_v8", 23, "Requested device-byte length."),
        _flags(24, 0x00000003),
    ),
    range_groups=(QUEUE_WAIT_RANGE, QUEUE_SIGNAL_RANGE),
    state_effects=_queue_state_effects(
        state_write(StateResource.HAL_POOL, "pool_r8_nullable"),
        state_allocate(StateResource.BUFFER, "dst_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Submits one queue-ordered transient allocation whose storage becomes "
            "usable only after the signal list is reached."
        ),
        verification=(
            _common_verification(),
            "All allocation parameter registers must be valid and flags_u32 may "
            "contain only INDETERMINATE_LIFETIME and ALLOW_POOL_WAIT_FRONTIER.",
        ),
        preconditions=(
            _common_preconditions(),
            "A non-null pool must be exact hal.pool. Allocation parameters must "
            "checked-narrow under the common packet masks and alignment rule.",
        ),
        success=(
            "A complete non-null transient buffer owner replaces dst_r8 and the "
            "program counter advances by 28 bytes. Its object may exist before "
            "its bytes become usable; the encoded signal edge establishes use.",
        ),
        failures=(
            *_common_queue_failures(),
            FailureCase(
                "invalid_argument",
                "A non-null pool has the wrong type or allocation flags/parameters are invalid.",
                "No HAL call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "out_of_range",
                "An allocation parameter cannot be represented in its HAL domain.",
                "No HAL call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "Queue allocation returns any non-OK status.",
                "Temporary storage is popped, any partial buffer is released, and dst_r8 is unchanged.",
            ),
        ),
        ownership=(
            "The device, optional pool, and semaphore elements are borrowed only "
            "for synchronous submission. HAL captures pool/provider state and "
            "list contents required later; dst_r8 receives the sole returned owner.",
        ),
        assembly=(
            "%r<dst> = hal.queue.alloca %r<device>, %v<affinity>, %r<pool>?, "
            "waits, signals, %v<usage>, %v<access>, %v<memory_type>, "
            "%v<memory_affinity>, %v<minimum_alignment>, %v<allocation_size> {flags}",
        ),
        pseudocode=(
            "device = require_hal_ref(refs[device_r8], hal_device_type);\n"
            "pool = optional_hal_ref(refs[pool_r8_nullable], hal_pool_type);\n"
            "params = checked_buffer_params(values[usage_v8], values[access_v8],\n"
            "    values[memory_type_v8], values[memory_affinity_v8],\n"
            "    values[min_alignment_v8]);\n"
            "size = checked_device_size(values[allocation_size_v8]);\n"
            "adapter = prepare_queue_adapter(waits, signals, 0);\n"
            "buffer = NULL;\n"
            "status = hal_device_queue_alloca(device, values[affinity_v8],\n"
            "    adapter.waits, adapter.signals, pool, params, size, flags_u32, &buffer);\n"
            "release_queue_adapter(adapter);\n"
            "if (status failed) { release_if_nonnull(buffer); return status; }\n"
            "replace_ref(&refs[dst_r8], owned_ref(buffer, hal_buffer_type));\n"
            "pc = pc + 28;"
        ),
    ),
)

HAL_QUEUE_DEALLOCA = hal_instruction(
    entity_id="hal.instruction.queue.dealloca",
    since=HAL_0,
    summary="Queues transient-storage reclamation after explicit waits.",
    opcode=0x0C,
    mnemonic="hal.queue.dealloca",
    byte_length=24,
    family_id=FAMILY.entity_id,
    fields=(
        *_queue_common_fields(),
        hal_ref("buffer_r8", 16, "hal.buffer"),
        zero_padding("zero_padding_u8", 17, 3),
        _flags(20, 0x00000001),
    ),
    range_groups=(QUEUE_WAIT_RANGE, QUEUE_SIGNAL_RANGE),
    state_effects=_queue_state_effects(
        state_release(StateResource.BUFFER, "buffer_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Makes transient bytes reusable after every wait and publishes "
            "signals only after target-visible reclamation effects."
        ),
        verification=(
            _common_verification(),
            "buffer_r8 and padding must be valid and flags_u32 may contain only PREFER_ORIGIN.",
        ),
        preconditions=(
            _common_preconditions(),
            "buffer_r8 must contain a non-null exact live hal.buffer compatible "
            "with queue deallocation.",
        ),
        success=(
            "Reclamation is queued, buffer_r8 remains unchanged, and the program "
            "counter advances by 24 bytes. After waits are reached the bytes are "
            "undefined even before signals publish.",
        ),
        failures=(
            *_common_queue_failures(),
            *required_ref_failures(
                "hal.buffer", "No HAL call occurs and buffer_r8 remains unchanged."
            ),
            FailureCase(
                "provider_status",
                "Queue deallocation returns any non-OK status.",
                "VM refs remain unchanged; HAL defines any externally committed queue state.",
            ),
        ),
        ownership=(
            "All refs are borrowed for submission and remain unchanged. HAL "
            "captures any origin, pool, allocator, or buffer state needed later.",
        ),
        assembly=(
            "hal.queue.dealloca %r<device>, %v<affinity>, waits, signals, "
            "%r<buffer> {flags}",
        ),
        pseudocode=(
            "buffer = require_hal_ref(refs[buffer_r8], hal_buffer_type);\n"
            "adapter = prepare_queue_adapter(waits, signals, 0);\n"
            "status = hal_device_queue_dealloca(device, affinity, adapter.waits,\n"
            "    adapter.signals, buffer, flags_u32);\n"
            "release_queue_adapter(adapter);\n"
            "if (status failed) return status;\n"
            "pc = pc + 24;"
        ),
    ),
)

HAL_QUEUE_FILL = hal_instruction(
    entity_id="hal.instruction.queue.fill",
    since=HAL_0,
    summary="Queues a repeating one-, two-, or four-byte pattern fill.",
    opcode=0x0D,
    mnemonic="hal.queue.fill",
    byte_length=28,
    family_id=FAMILY.entity_id,
    fields=(
        *_queue_common_fields(),
        hal_ref("target_buffer_r8", 16, "hal.buffer"),
        _value("target_offset_v8", 17, "Target device-byte offset."),
        _value("length_v8", 18, "Fill length in device bytes."),
        _value("pattern_v8", 19, "Low pattern bits in little-endian byte order."),
        instruction_field(
            "pattern_width_u8",
            20,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Repeating pattern width of one, two, or four bytes.",
            (RuleUse(ALLOWED_VALUES.entity_id, ((1, 2, 4),)),),
        ),
        zero_padding("zero_padding_u8", 21, 3),
        _zero_u32(24),
    ),
    range_groups=(QUEUE_WAIT_RANGE, QUEUE_SIGNAL_RANGE),
    state_effects=_queue_state_effects(
        state_write(StateResource.BUFFER, "target_buffer_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Queues a direct HAL-buffer fill using explicitly extracted little-"
            "endian pattern bytes independent of host endianness."
        ),
        verification=(
            _common_verification(),
            "All operation registers and padding must be valid, pattern_width_u8 "
            "must be 1, 2, or 4, and flags_u32 must be zero.",
        ),
        preconditions=(
            _common_preconditions(),
            "target_buffer_r8 must be non-null exact hal.buffer. Offset and length "
            "must fit device size, form a live target range, and both be multiples "
            "of pattern_width_u8. HAL requires transfer-target/write authority, "
            "visibility, and queue compatibility.",
        ),
        success=(
            "The fill and its explicit edges are submitted and the program counter "
            "advances by 28 bytes; VM operands remain unchanged.",
        ),
        failures=(
            *_common_queue_failures(),
            *required_ref_failures(
                "hal.buffer", "No HAL call occurs and VM state remains unchanged."
            ),
            FailureCase(
                "out_of_range",
                "Offset/length cannot narrow or the live range is invalid.",
                "No HAL call occurs and VM state remains unchanged.",
            ),
            FailureCase(
                "invalid_argument",
                "Offset or length is not a multiple of the pattern width.",
                "No HAL call occurs and VM state remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "Queue fill returns any non-OK status.",
                "VM state remains unchanged; HAL defines any externally committed queue state.",
            ),
        ),
        ownership=(
            "All refs and list rows are synchronously borrowed. HAL captures the "
            "target and list contents before return.",
        ),
        assembly=(
            "hal.queue.fill %r<device>, %v<affinity>, waits, signals, "
            "%r<target>, %v<target_offset>, %v<length>, %v<pattern> {pattern_width}",
        ),
        pseudocode=(
            "target = require_hal_ref(refs[target_buffer_r8], hal_buffer_type);\n"
            "offset = checked_device_size(values[target_offset_v8]);\n"
            "length = checked_device_size(values[length_v8]);\n"
            "check_hal_buffer_range(target, offset, length);\n"
            "require_multiple(offset, pattern_width_u8);\n"
            "require_multiple(length, pattern_width_u8);\n"
            "pattern = low_little_endian_bytes(values[pattern_v8], pattern_width_u8);\n"
            "adapter = prepare_queue_adapter(waits, signals, 0);\n"
            "status = hal_device_queue_fill(device, affinity, adapter.waits,\n"
            "    adapter.signals, target, offset, length, pattern, pattern_width_u8, 0);\n"
            "release_queue_adapter(adapter);\n"
            "if (status failed) return status;\n"
            "pc = pc + 28;"
        ),
    ),
)


def _transfer_instruction(
    *,
    entity_id: str,
    summary: str,
    opcode: int,
    mnemonic: str,
    source_field: str,
    source_type: str,
    target_field: str,
    target_type: str,
    contract: str,
    ownership: str,
    extra_failures: tuple[FailureCase, ...],
    state_effects: tuple[StateEffect, ...],
    pseudocode: str,
):
    return hal_instruction(
        entity_id=entity_id,
        since=HAL_0,
        summary=summary,
        opcode=opcode,
        mnemonic=mnemonic,
        byte_length=28,
        family_id=FAMILY.entity_id,
        fields=(
            *_queue_common_fields(),
            hal_ref(source_field, 16, source_type),
            _value("source_offset_v8", 17, "Source byte offset."),
            hal_ref(target_field, 18, target_type),
            _value("target_offset_v8", 19, "Target byte offset."),
            _value("length_v8", 20, "Exact transfer byte length."),
            zero_padding("zero_padding_u8", 21, 3),
            _zero_u32(24),
        ),
        range_groups=(QUEUE_WAIT_RANGE, QUEUE_SIGNAL_RANGE),
        state_effects=_queue_state_effects(*state_effects),
        semantics=InstructionSemantics(
            description=summary,
            verification=(
                _common_verification(),
                "All operation registers and padding must be valid and flags_u32 must be zero.",
            ),
            preconditions=(_common_preconditions(), contract),
            success=(
                "The exact transfer and its semaphore edges are captured during "
                "the synchronous call, VM operands remain unchanged, and the "
                "program counter advances by 28 bytes.",
            ),
            failures=(
                *_common_queue_failures(),
                FailureCase(
                    "failed_precondition",
                    f"The required {source_field} or {target_field} ref is canonical null.",
                    "No HAL call occurs and every VM operand remains unchanged.",
                ),
                FailureCase(
                    "invalid_argument",
                    f"A non-null operand does not have exact {source_type}/{target_type} type as encoded.",
                    "No HAL call occurs and every VM operand remains unchanged.",
                ),
                FailureCase(
                    "permission_denied",
                    "A file or buffer lacks operation-required read/write authority.",
                    "No HAL call occurs and every VM operand remains unchanged.",
                ),
                FailureCase(
                    "out_of_range",
                    "Any size cannot narrow or any source/target range is invalid.",
                    "No HAL call occurs and every VM operand remains unchanged.",
                ),
                *extra_failures,
                FailureCase(
                    "provider_status",
                    "The queue transfer returns any non-OK status.",
                    "VM state remains unchanged; HAL defines any externally committed queue state.",
                ),
            ),
            ownership=(ownership,),
            assembly=(
                f"{mnemonic} %r<device>, %v<affinity>, waits, signals, "
                "%r<source>, %v<source_offset>, %r<target>, "
                "%v<target_offset>, %v<length>",
            ),
            pseudocode=pseudocode,
        ),
    )


HAL_QUEUE_UPDATE = _transfer_instruction(
    entity_id="hal.instruction.queue.update",
    summary="Captures readable host bytes into a direct HAL-buffer range.",
    opcode=0x0E,
    mnemonic="hal.queue.update",
    source_field="source_vm_buffer_r8",
    source_type="vm.buffer",
    target_field="target_buffer_r8",
    target_type="hal.buffer",
    contract=(
        "The source must be an open READ-capable vm.buffer range. Length must fit "
        "host and device size; the target must be a live hal.buffer range with "
        "transfer-target/write authority, visibility, and queue compatibility."
    ),
    ownership=(
        "HAL copies the exact host bytes before returning and never retains the "
        "core buffer/data pointer. It captures the target and semaphore contents."
    ),
    extra_failures=(
        FailureCase(
            "failed_precondition",
            "The core source vm.buffer root is closed.",
            "No HAL call occurs and every VM operand remains unchanged.",
        ),
    ),
    state_effects=(
        state_read(StateResource.BUFFER, "source_vm_buffer_r8"),
        state_write(StateResource.BUFFER, "target_buffer_r8"),
    ),
    pseudocode=(
        "source = require_hal_ref(refs[source_vm_buffer_r8], vm_buffer_type);\n"
        "source_offset = checked_host_size(values[source_offset_v8]);\n"
        "length_u64 = values[length_v8];\n"
        "check_vm_buffer_read_range(source, source_offset, length_u64);\n"
        "target = require_hal_ref(refs[target_buffer_r8], hal_buffer_type);\n"
        "target_offset = checked_device_size(values[target_offset_v8]);\n"
        "length = checked_device_size(length_u64);\n"
        "check_hal_buffer_range(target, target_offset, length);\n"
        "adapter = prepare_queue_adapter(waits, signals, 0);\n"
        "status = hal_device_queue_update(device, affinity, adapter.waits,\n"
        "    adapter.signals, nonnull_empty_safe_data(source), source_offset,\n"
        "    target, target_offset, length, 0);\n"
        "release_queue_adapter(adapter);\n"
        "if (status failed) return status;\n"
        "pc = pc + 28;"
    ),
)

HAL_QUEUE_COPY = _transfer_instruction(
    entity_id="hal.instruction.queue.copy",
    summary="Queues one non-overlapping direct HAL-buffer copy.",
    opcode=0x0F,
    mnemonic="hal.queue.copy",
    source_field="source_buffer_r8",
    source_type="hal.buffer",
    target_field="target_buffer_r8",
    target_type="hal.buffer",
    contract=(
        "Both ranges must be live and disjoint. Source requires transfer-source/read "
        "authority; target requires transfer-target/write authority; both require "
        "device visibility and queue compatibility."
    ),
    ownership=(
        "HAL captures both buffers and semaphore contents before returning; every "
        "VM ref and local row remains unchanged."
    ),
    extra_failures=(
        FailureCase(
            "invalid_argument",
            "The source and target HAL-buffer ranges overlap.",
            "No HAL call occurs and every VM operand remains unchanged.",
        ),
    ),
    state_effects=(
        state_read(StateResource.BUFFER, "source_buffer_r8"),
        state_write(StateResource.BUFFER, "target_buffer_r8"),
    ),
    pseudocode=(
        "source = require_hal_ref(refs[source_buffer_r8], hal_buffer_type);\n"
        "target = require_hal_ref(refs[target_buffer_r8], hal_buffer_type);\n"
        "source_offset = checked_device_size(values[source_offset_v8]);\n"
        "target_offset = checked_device_size(values[target_offset_v8]);\n"
        "length = checked_device_size(values[length_v8]);\n"
        "check_hal_buffer_range(source, source_offset, length);\n"
        "check_hal_buffer_range(target, target_offset, length);\n"
        "require_disjoint(source, source_offset, target, target_offset, length);\n"
        "adapter = prepare_queue_adapter(waits, signals, 0);\n"
        "status = hal_device_queue_copy(device, affinity, adapter.waits,\n"
        "    adapter.signals, source, source_offset, target, target_offset, length, 0);\n"
        "release_queue_adapter(adapter);\n"
        "if (status failed) return status;\n"
        "pc = pc + 28;"
    ),
)

HAL_QUEUE_READ = _transfer_instruction(
    entity_id="hal.instruction.queue.read",
    summary="Queues a readable file range into a direct HAL-buffer range.",
    opcode=0x10,
    mnemonic="hal.queue.read",
    source_field="source_file_r8",
    source_type="hal.file",
    target_field="target_buffer_r8",
    target_type="hal.buffer",
    contract=(
        "The file must provide READ and contain the complete u64 source range. The "
        "target must contain the device-sized range and provide transfer-target/write "
        "authority, visibility, and queue compatibility."
    ),
    ownership=(
        "HAL captures file, buffer, and semaphore state until the transfer can no "
        "longer use them; VM refs and local rows remain unchanged."
    ),
    extra_failures=(),
    state_effects=(
        state_read(StateResource.IO_FILE, "source_file_r8"),
        state_write(StateResource.BUFFER, "target_buffer_r8"),
    ),
    pseudocode=(
        "source = require_hal_ref(refs[source_file_r8], hal_file_type);\n"
        "target = require_hal_ref(refs[target_buffer_r8], hal_buffer_type);\n"
        "source_offset = values[source_offset_v8];\n"
        "target_offset = checked_device_size(values[target_offset_v8]);\n"
        "length = checked_device_size(values[length_v8]);\n"
        "check_file_read_range(source, source_offset, length);\n"
        "check_hal_buffer_range(target, target_offset, length);\n"
        "adapter = prepare_queue_adapter(waits, signals, 0);\n"
        "status = hal_device_queue_read(device, affinity, adapter.waits,\n"
        "    adapter.signals, source, source_offset, target, target_offset, length, 0);\n"
        "release_queue_adapter(adapter);\n"
        "if (status failed) return status;\n"
        "pc = pc + 28;"
    ),
)

HAL_QUEUE_WRITE = _transfer_instruction(
    entity_id="hal.instruction.queue.write",
    summary="Queues a direct HAL-buffer range into a writable file range.",
    opcode=0x11,
    mnemonic="hal.queue.write",
    source_field="source_buffer_r8",
    source_type="hal.buffer",
    target_field="target_file_r8",
    target_type="hal.file",
    contract=(
        "The source must contain the device-sized range and provide transfer-source/read "
        "authority, visibility, and queue compatibility. The file must provide WRITE "
        "and contain the complete u64 target range; the operation never extends it."
    ),
    ownership=(
        "HAL captures buffer, file, and semaphore state until the transfer can no "
        "longer use them; VM refs and local rows remain unchanged."
    ),
    extra_failures=(),
    state_effects=(
        state_read(StateResource.BUFFER, "source_buffer_r8"),
        state_write(StateResource.IO_FILE, "target_file_r8"),
    ),
    pseudocode=(
        "source = require_hal_ref(refs[source_buffer_r8], hal_buffer_type);\n"
        "target = require_hal_ref(refs[target_file_r8], hal_file_type);\n"
        "source_offset = checked_device_size(values[source_offset_v8]);\n"
        "target_offset = values[target_offset_v8];\n"
        "length = checked_device_size(values[length_v8]);\n"
        "check_hal_buffer_range(source, source_offset, length);\n"
        "check_file_write_range(target, target_offset, length);\n"
        "adapter = prepare_queue_adapter(waits, signals, 0);\n"
        "status = hal_device_queue_write(device, affinity, adapter.waits,\n"
        "    adapter.signals, source, source_offset, target, target_offset, length, 0);\n"
        "release_queue_adapter(adapter);\n"
        "if (status failed) return status;\n"
        "pc = pc + 28;"
    ),
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
            "Base of a function-local byte range copied as dispatch constants.",
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


def _binding_fields(base_offset: int, range_name: str) -> tuple:
    return (
        _range_field(
            "binding_buffer_base_u16",
            base_offset,
            range_name,
            InstructionFieldRole.RANGE_BASE,
            "Base ref slot of binding buffers.",
        ),
        _range_field(
            "binding_offset_base_u16",
            base_offset + 2,
            range_name,
            InstructionFieldRole.RANGE_BASE,
            "Eight-byte-aligned local base of u64 binding offsets.",
        ),
        _range_field(
            "binding_length_base_u16",
            base_offset + 4,
            range_name,
            InstructionFieldRole.RANGE_BASE,
            "Eight-byte-aligned local base of u64 binding lengths.",
        ),
        _range_field(
            "binding_count_u16",
            base_offset + 6,
            range_name,
            InstructionFieldRole.RANGE_COUNT,
            "Shared binding-row count.",
        ),
    )


def _dispatch_failures(*, indirect: bool) -> tuple[FailureCase, ...]:
    indirect_ref_failures = (
        required_ref_failures(
            "hal.buffer", "No HAL call occurs and every VM operand remains unchanged."
        )
        if indirect
        else ()
    )
    indirect_alignment_failures = (
        (
            FailureCase(
                "invalid_argument",
                "The indirect count offset is not four-byte aligned.",
                "No HAL call occurs and every VM operand remains unchanged.",
            ),
        )
        if indirect
        else ()
    )
    return (
        *_common_queue_failures(),
        *required_ref_failures(
            "hal.executable_function_table",
            "No HAL call occurs and every VM operand remains unchanged.",
        ),
        *indirect_ref_failures,
        FailureCase(
            "failed_precondition",
            "Any direct binding buffer is canonical null.",
            "No native binding row is exposed to HAL and no HAL call occurs.",
        ),
        FailureCase(
            "invalid_argument",
            "Any non-null binding element has the wrong exact descriptor or workgroup-size lanes mix zero and nonzero values.",
            "No HAL call occurs and every VM operand remains unchanged.",
        ),
        FailureCase(
            "out_of_range",
            "The function ordinal, binding range, or narrowed offset/length is invalid.",
            "No HAL call occurs and every VM operand remains unchanged.",
        ),
        *indirect_alignment_failures,
        FailureCase(
            "provider_status",
            "Queue dispatch returns any non-OK status.",
            "VM state remains unchanged; HAL defines any externally committed queue state.",
        ),
    )


HAL_QUEUE_DISPATCH = hal_instruction(
    entity_id="hal.instruction.queue.dispatch",
    since=HAL_0,
    summary="Queues a static-count dispatch through an executable function table.",
    opcode=0x12,
    mnemonic="hal.queue.dispatch",
    byte_length=36,
    family_id=FAMILY.entity_id,
    fields=(
        *_queue_common_fields(),
        hal_ref(
            "function_table_r8",
            16,
            "hal.executable_function_table",
        ),
        _value("function_ordinal_v8", 17, "Unsigned selected function ordinal."),
        _local_fixed_base(
            "launch_base_u16",
            18,
            28,
            "Four-byte-aligned local base of seven u32 launch lanes.",
        ),
        *_local_byte_range_fields(20),
        *_binding_fields(24, _DIRECT_BINDINGS),
        _flags(32, 0x00000020),
    ),
    range_groups=(
        QUEUE_WAIT_RANGE,
        QUEUE_SIGNAL_RANGE,
        QUEUE_DIRECT_BINDING_RANGE,
    ),
    state_effects=_queue_state_effects(
        state_read(
            StateResource.FRAME_LOCALS,
            "launch_base_u16",
            "constant_base_u16",
            "binding_buffer_base_u16",
            "binding_offset_base_u16",
            "binding_length_base_u16",
        ),
        state_read(StateResource.BUFFER, "binding_buffer_base_u16"),
        state_write(StateResource.BUFFER, "binding_buffer_base_u16"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Reads exact static workgroup counts, optional workgroup sizes, dynamic "
            "local memory, constant bytes, and direct buffer bindings, then submits "
            "one table-plus-ordinal dispatch."
        ),
        verification=(
            _common_verification(),
            "launch_base_u16 must address exactly seven aligned u32 lanes; the "
            "constant byte range and direct-binding struct-of-arrays range must "
            "be in bounds and canonically empty; flags_u32 may contain only bit 5.",
        ),
        preconditions=(
            _common_preconditions(),
            "function_table_r8 must be a non-null exact function table and the "
            "complete ordinal must fit host size and be below its count.",
            "Every binding must be a non-null exact hal.buffer with a live checked "
            "device-sized range. Workgroup-size lanes must be all zero or all nonzero.",
        ),
        success=(
            "HAL captures the selected table function, launch configuration, "
            "constant bytes, direct bindings, and semaphore edges; the program "
            "counter advances by 36 bytes and VM operands remain unchanged.",
        ),
        failures=_dispatch_failures(indirect=False),
        ownership=(
            "The VM borrows table, buffers, constants, and adapter rows only for "
            "the synchronous call. HAL captures all execution resources it needs later.",
        ),
        assembly=(
            "hal.queue.dispatch %r<device>, %v<affinity>, waits, signals, "
            "%r<function_table>, %v<function_ordinal>, #launch7, #constants, "
            "#direct_bindings {flags}",
        ),
        pseudocode=(
            "config.workgroup_count = load_u32x3_le(local_bytes, launch_base_u16);\n"
            "config.workgroup_size = load_u32x3_le(local_bytes, launch_base_u16 + 12);\n"
            "config.dynamic_local_memory = load_u32_le(local_bytes, launch_base_u16 + 24);\n"
            "check_workgroup_size(config.workgroup_size);\n"
            "table, ordinal, constants, bindings = preflight_direct_dispatch();\n"
            "adapter = prepare_queue_adapter(waits, signals, binding_count_u16);\n"
            "status = hal_device_queue_dispatch_table(device, affinity, adapter.waits,\n"
            "    adapter.signals, table, ordinal, config, constants,\n"
            "    adapter.bindings, flags_u32);\n"
            "release_queue_adapter(adapter);\n"
            "if (status failed) return status;\n"
            "pc = pc + 36;"
        ),
    ),
)

HAL_QUEUE_DISPATCH_INDIRECT_COUNT = hal_instruction(
    entity_id="hal.instruction.queue.dispatch.indirect.count",
    since=HAL_0,
    summary="Queues a dispatch whose count is read from a direct HAL-buffer range.",
    opcode=0x13,
    mnemonic="hal.queue.dispatch.indirect.count",
    byte_length=40,
    family_id=FAMILY.entity_id,
    fields=(
        *_queue_common_fields(),
        hal_ref(
            "function_table_r8",
            16,
            "hal.executable_function_table",
        ),
        _value("function_ordinal_v8", 17, "Unsigned selected function ordinal."),
        _local_fixed_base(
            "launch_base_u16",
            18,
            16,
            "Four-byte-aligned local base of four u32 launch lanes.",
        ),
        hal_ref("workgroup_count_buffer_r8", 20, "hal.buffer"),
        _value(
            "workgroup_count_offset_v8",
            21,
            "Four-byte-aligned device offset of three u32 workgroup counts.",
        ),
        *_local_byte_range_fields(22),
        *_binding_fields(26, _DIRECT_BINDINGS),
        zero_u16("zero_padding_u16", 34),
        _flags(36, 0x00000023, required_one_mask=0x00000003),
    ),
    range_groups=(
        QUEUE_WAIT_RANGE,
        QUEUE_SIGNAL_RANGE,
        QUEUE_DIRECT_BINDING_RANGE,
    ),
    state_effects=_queue_state_effects(
        state_read(
            StateResource.FRAME_LOCALS,
            "launch_base_u16",
            "constant_base_u16",
            "binding_buffer_base_u16",
            "binding_offset_base_u16",
            "binding_length_base_u16",
        ),
        state_read(StateResource.BUFFER, "workgroup_count_buffer_r8"),
        state_read(StateResource.BUFFER, "binding_buffer_base_u16"),
        state_write(StateResource.BUFFER, "binding_buffer_base_u16"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Reads optional workgroup sizes and dynamic local memory from local "
            "bytes, and reads three workgroup counts from an exact direct "
            "HAL-buffer range under one explicit indirect-parameter mode."
        ),
        verification=(
            _common_verification(),
            "launch_base_u16 must address exactly four aligned u32 lanes; constant "
            "and direct-binding ranges must be valid and canonically empty; "
            "zero_padding_u16 must be zero; flags_u32 admits bit 5 and exactly one "
            "of dynamic/static indirect-parameter bits 0 and 1.",
        ),
        preconditions=(
            _common_preconditions(),
            "Function table, ordinal, bindings, and workgroup-size lanes obey the "
            "static dispatch contract. workgroup_count_buffer_r8 must be a non-null "
            "exact hal.buffer; its four-byte-aligned offset must select exactly 12 "
            "live bytes with indirect-parameter/read authority, visibility, and "
            "queue compatibility.",
        ),
        success=(
            "HAL captures the table function, indirect-count buffer, launch "
            "configuration, constants, direct bindings, and semaphore edges; the "
            "program counter advances by 40 bytes and VM operands remain unchanged.",
        ),
        failures=_dispatch_failures(indirect=True),
        ownership=(
            "Every VM ref and local range is borrowed only through synchronous "
            "submission. HAL captures the count buffer and all dispatch resources.",
        ),
        assembly=(
            "hal.queue.dispatch.indirect.count %r<device>, %v<affinity>, waits, "
            "signals, %r<function_table>, %v<function_ordinal>, #launch4, "
            "%r<count_buffer>, %v<count_offset>, #constants, #direct_bindings {flags}",
        ),
        pseudocode=(
            "config.workgroup_size = load_u32x3_le(local_bytes, launch_base_u16);\n"
            "config.dynamic_local_memory = load_u32_le(local_bytes, launch_base_u16 + 12);\n"
            "check_workgroup_size(config.workgroup_size);\n"
            "count_buffer = require_hal_ref(refs[workgroup_count_buffer_r8], hal_buffer_type);\n"
            "count_offset = checked_device_size(values[workgroup_count_offset_v8]);\n"
            "require_multiple(count_offset, 4);\n"
            "check_hal_buffer_range(count_buffer, count_offset, 12);\n"
            "table, ordinal, constants, bindings = preflight_direct_dispatch();\n"
            "adapter = prepare_queue_adapter(waits, signals, binding_count_u16);\n"
            "status = hal_device_queue_dispatch_table(device, affinity, adapter.waits,\n"
            "    adapter.signals, table, ordinal, indirect_config(count_buffer,\n"
            "    count_offset, config), constants, adapter.bindings, flags_u32);\n"
            "release_queue_adapter(adapter);\n"
            "if (status failed) return status;\n"
            "pc = pc + 40;"
        ),
    ),
)

HAL_QUEUE_EXECUTE = hal_instruction(
    entity_id="hal.instruction.queue.execute",
    since=HAL_0,
    summary="Submits one finalized command buffer with an optional binding table.",
    opcode=0x14,
    mnemonic="hal.queue.execute",
    byte_length=32,
    family_id=FAMILY.entity_id,
    fields=(
        *_queue_common_fields(),
        hal_ref("command_buffer_r8", 16, "hal.command_buffer"),
        zero_padding("zero_padding_u8", 17, 1),
        *_binding_fields(18, _EXECUTE_BINDINGS),
        zero_u16("zero_padding_u16", 26),
        _zero_u32(28),
    ),
    range_groups=(
        QUEUE_WAIT_RANGE,
        QUEUE_SIGNAL_RANGE,
        QUEUE_EXECUTE_BINDING_RANGE,
    ),
    state_effects=_queue_state_effects(
        state_read(StateResource.HAL_COMMAND_BUFFER, "command_buffer_r8"),
        state_read(StateResource.BUFFER),
        state_write(StateResource.BUFFER),
        state_synchronize(StateResource.BUFFER),
        state_read(
            StateResource.FRAME_LOCALS,
            "binding_buffer_base_u16",
            "binding_offset_base_u16",
            "binding_length_base_u16",
        ),
        state_read(StateResource.BUFFER, "binding_buffer_base_u16"),
        state_write(StateResource.BUFFER, "binding_buffer_base_u16"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Submits one non-null finalized command buffer with a nullable direct "
            "runtime binding table and explicit semaphore edges."
        ),
        verification=(
            _common_verification(),
            "command_buffer_r8, padding, and the nullable binding struct-of-arrays "
            "range must be valid; flags_u32 must be zero.",
        ),
        preconditions=(
            _common_preconditions(),
            "command_buffer_r8 must be non-null exact hal.command_buffer, finalized, "
            "and queue compatible. Each non-null binding must be exact hal.buffer "
            "with a live device-sized range; null entries preserve narrowed offset "
            "and length for command-buffer submission validation.",
            "An inline-execution command buffer requires an empty wait list.",
        ),
        success=(
            "HAL captures the command buffer, nullable binding rows, non-null "
            "buffers, and semaphore edges; the program counter advances by 32 "
            "bytes and VM storage remains unchanged.",
        ),
        failures=(
            *_common_queue_failures(),
            *required_ref_failures(
                "hal.command_buffer",
                "No HAL call occurs and every VM operand remains unchanged.",
            ),
            FailureCase(
                "invalid_argument",
                "A non-null binding has the wrong type or command/binding contract is invalid.",
                "No HAL call occurs and every VM operand remains unchanged.",
            ),
            FailureCase(
                "out_of_range",
                "A binding offset/length cannot narrow or selects an invalid live range.",
                "No HAL call occurs and every VM operand remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "Queue execute returns any non-OK status.",
                "VM state remains unchanged; HAL defines any externally committed queue state.",
            ),
        ),
        ownership=(
            "All input refs and rows remain VM-owned. HAL copies the table and "
            "captures the command buffer and every required non-null buffer before return.",
        ),
        assembly=(
            "hal.queue.execute %r<device>, %v<affinity>, waits, signals, "
            "%r<command_buffer>, #binding_table",
        ),
        pseudocode=(
            "command_buffer = require_hal_ref(refs[command_buffer_r8], command_buffer_type);\n"
            "bindings = preflight_nullable_execute_bindings();\n"
            "adapter = prepare_queue_adapter(waits, signals, binding_count_u16);\n"
            "status = hal_device_queue_execute(device, affinity, adapter.waits,\n"
            "    adapter.signals, command_buffer, adapter.bindings, 0);\n"
            "release_queue_adapter(adapter);\n"
            "if (status failed) return status;\n"
            "pc = pc + 32;"
        ),
    ),
)

HAL_QUEUE_BARRIER = hal_instruction(
    entity_id="hal.instruction.queue.barrier",
    since=HAL_0,
    summary="Joins explicit waits and publishes explicit signals without commands.",
    opcode=0x15,
    mnemonic="hal.queue.barrier",
    byte_length=20,
    family_id=FAMILY.entity_id,
    fields=(*_queue_common_fields(), _zero_u32(16)),
    range_groups=(QUEUE_WAIT_RANGE, QUEUE_SIGNAL_RANGE),
    state_effects=_queue_state_effects(),
    semantics=InstructionSemantics(
        description=(
            "Submits one explicit semaphore-only fan-in/fan-out operation, even "
            "when both lists are empty."
        ),
        verification=(
            _common_verification(),
            "flags_u32 must be canonical zero.",
        ),
        preconditions=(_common_preconditions(),),
        success=(
            "HAL captures both lists, including the empty/empty operation, and "
            "the program counter advances by 20 bytes. No implicit prior work or "
            "FIFO edge is acquired.",
        ),
        failures=(
            *_common_queue_failures(),
            FailureCase(
                "provider_status",
                "Queue barrier returns any non-OK status.",
                "VM state remains unchanged; HAL defines any externally committed queue state.",
            ),
        ),
        ownership=(
            "Semaphore refs and adapter rows are borrowed only for synchronous submission.",
        ),
        assembly=("hal.queue.barrier %r<device>, %v<affinity>, waits, signals",),
        pseudocode=(
            "adapter = prepare_queue_adapter(waits, signals, 0);\n"
            "status = hal_device_queue_barrier(device, affinity, adapter.waits,\n"
            "    adapter.signals, 0);\n"
            "release_queue_adapter(adapter);\n"
            "if (status failed) return status;\n"
            "pc = pc + 20;"
        ),
    ),
)

HAL_QUEUE_FLUSH = hal_instruction(
    entity_id="hal.instruction.queue.flush",
    since=HAL_0,
    summary="Flushes locally pending submissions for one device/affinity selection.",
    opcode=0x16,
    mnemonic="hal.queue.flush",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        hal_ref("device_r8", 2, "hal.device"),
        _value("affinity_v8", 3, "Complete u64 queue-affinity bitset."),
    ),
    state_effects=(state_write(StateResource.HAL_QUEUE, "device_r8", "affinity_v8"),),
    semantics=InstructionSemantics(
        description=(
            "Asks one explicit device/affinity selection to submit locally pending "
            "work without waiting for completion or creating an ordering edge."
        ),
        verification=("device_r8 and affinity_v8 must be valid register ordinals.",),
        preconditions=("device_r8 must contain a non-null exact hal.device.",),
        success=(
            "The flush request returns OK and the program counter advances by four "
            "bytes; completion remains observable only through encoded semaphores.",
        ),
        failures=(
            *required_ref_failures(
                "hal.device", "No HAL call occurs and VM state remains unchanged."
            ),
            FailureCase(
                "provider_status",
                "Queue flush returns any non-OK status.",
                "VM state remains unchanged; HAL defines any externally committed submission state.",
            ),
        ),
        ownership=("device_r8 is borrowed for the synchronous flush call.",),
        assembly=("hal.queue.flush %r<device>, %v<affinity>",),
        pseudocode=(
            "device = require_hal_ref(refs[device_r8], hal_device_type);\n"
            "status = hal_device_queue_flush(device, values[affinity_v8]);\n"
            "if (status failed) return status;\n"
            "pc = pc + 4;"
        ),
    ),
)

INSTRUCTIONS = (
    HAL_QUEUE_ALLOCA,
    HAL_QUEUE_DEALLOCA,
    HAL_QUEUE_FILL,
    HAL_QUEUE_UPDATE,
    HAL_QUEUE_COPY,
    HAL_QUEUE_READ,
    HAL_QUEUE_WRITE,
    HAL_QUEUE_DISPATCH,
    HAL_QUEUE_DISPATCH_INDIRECT_COUNT,
    HAL_QUEUE_EXECUTE,
    HAL_QUEUE_BARRIER,
    HAL_QUEUE_FLUSH,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
