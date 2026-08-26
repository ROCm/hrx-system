# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HAL 0.0 long-lived buffer and scoped-mapping instructions."""

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
from model.isa.validation import ALLOWED_VALUES
from model.schema import U8, RuleUse
from model.specification import HAL_0

FAMILY = InstructionFamily(
    entity_id="hal.family.buffer",
    since=HAL_0,
    summary="Long-lived device allocation and deterministic scoped mapping.",
    dependencies=("hal.contract.abi",),
    document_order=12,
    normative_text=(
        "hal.buffer names device-visible storage and is exact-type unrelated to "
        "the core CPU-addressable vm.buffer even when both reach one allocation. "
        "Long-lived allocation ownership ends through ordinary final release; "
        "there is no synchronous free opcode. Scoped mapping publishes a core "
        "vm.buffer root containing its native mapping record and provider kind "
        "after the hot buffer base. It begins open with exact static READ/WRITE "
        "rights, requested length/range pointer, and no root link. Proper views "
        "cache direct starts and retain the mapped root. Explicit unmap closes "
        "the root synchronously before provider flush/unmap, immediately turning "
        "every root alias and view into a byte-inaccessible tombstone while "
        "preserving ref operations and immutable length. Final release of an "
        "abandoned open mapping closes hot fields and invokes a no-status HAL "
        "abandon primitive; abandonment guarantees cleanup, not publication of "
        "host writes. Programs requiring writable completion execute unmap and "
        "observe its status. Races between byte access and unmap are externally "
        "unsynchronized C misuse; pointer/access atomics cannot make them safe."
    ),
)


def _value(name: str, offset: int, description: str):
    return value_register(
        name,
        offset,
        InstructionFieldRole.OPERAND,
        description,
    )


HAL_BUFFER_ALLOCATE = hal_instruction(
    entity_id="hal.instruction.buffer.allocate",
    since=HAL_0,
    summary="Allocates one long-lived HAL buffer on an explicit device.",
    opcode=0x03,
    mnemonic="hal.buffer.allocate",
    byte_length=12,
    family_id=FAMILY.entity_id,
    fields=(
        hal_result_ref("dst_r8", 2, "hal.buffer"),
        hal_ref("device_r8", 3, "hal.device"),
        _value("usage_v8", 4, "Unsigned HAL buffer-usage flag bits."),
        _value("access_v8", 5, "Unsigned HAL memory-access flag bits."),
        _value("memory_type_v8", 6, "Unsigned HAL memory-type flag bits."),
        _value("affinity_v8", 7, "Complete u64 queue-affinity bitset."),
        _value(
            "min_alignment_v8",
            8,
            "Unsigned minimum allocation alignment in bytes.",
        ),
        _value(
            "allocation_size_v8",
            9,
            "Unsigned requested allocation size in device bytes.",
        ),
        zero_u16("zero_padding_u16", 10),
    ),
    state_effects=(
        state_read(StateResource.HAL_DEVICE, "device_r8"),
        state_allocate(StateResource.BUFFER, "dst_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Checked-narrows the common allocation packet and calls the explicit "
            "device's current allocator to create one long-lived HAL buffer."
        ),
        verification=(
            "dst_r8, device_r8, and all six value-register fields must be valid; "
            "zero_padding_u16 must equal zero.",
        ),
        preconditions=(
            "device_r8 must be a non-null exact hal.device. Usage, access, and "
            "memory-type values must fit and use only masks 0x1F0F3F03, 0x001F, "
            "and 0x0000007F; access 0x0020 is forbidden. Alignment and size must "
            "fit device size and nonzero alignment must be a power of two.",
        ),
        success=(
            "A complete non-null result owner replaces dst_r8 and the program "
            "counter advances by 12 bytes. Zero packet fields retain HAL defaults "
            "and a zero-byte allocation still returns a non-null object.",
        ),
        failures=(
            *required_ref_failures(
                "hal.device", "No allocator call occurs and dst_r8 remains unchanged."
            ),
            FailureCase(
                "out_of_range",
                "A packet field cannot be represented by its HAL ABI domain.",
                "No allocator call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "invalid_argument",
                "Flags contain unknown/disallowed bits or alignment is not a power of two.",
                "No allocator call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "The HAL allocator returns any non-OK status.",
                "Any partial buffer is released and dst_r8 remains unchanged.",
            ),
        ),
        ownership=(
            "device_r8 is borrowed only for allocator entry. The completed HAL "
            "buffer owner replaces dst_r8 after success.",
        ),
        assembly=(
            "%r<dst> = hal.buffer.allocate %r<device>, %v<usage>, %v<access>, "
            "%v<memory_type>, %v<affinity>, %v<minimum_alignment>, "
            "%v<allocation_size>",
        ),
        pseudocode=(
            "device = require_hal_ref(refs[device_r8], hal_device_type);\n"
            "params = checked_buffer_params(values[usage_v8], values[access_v8],\n"
            "    values[memory_type_v8], values[affinity_v8],\n"
            "    values[min_alignment_v8]);\n"
            "allocation_size = checked_device_size(values[allocation_size_v8]);\n"
            "buffer = NULL;\n"
            "status = hal_allocator_allocate_buffer(hal_device_allocator(device),\n"
            "    params, allocation_size, &buffer);\n"
            "if (status failed) { release_if_nonnull(buffer); return status; }\n"
            "replace_ref(&refs[dst_r8], owned_ref(buffer, hal_buffer_type));\n"
            "pc = pc + 12;"
        ),
    ),
)

HAL_BUFFER_MAP = hal_instruction(
    entity_id="hal.instruction.buffer.map",
    since=HAL_0,
    summary="Creates one deterministic scoped CPU mapping.",
    opcode=0x04,
    mnemonic="hal.buffer.map",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        hal_result_ref("dst_r8", 2, "vm.buffer"),
        hal_ref("source_buffer_r8", 3, "hal.buffer"),
        _value("source_offset_v8", 4, "Unsigned HAL-buffer source offset."),
        _value("source_length_v8", 5, "Unsigned requested mapping length."),
        instruction_field(
            "access_u8",
            6,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Static mapping access: READ=1, WRITE=2, or READ_WRITE=3.",
            (RuleUse(ALLOWED_VALUES.entity_id, ((1, 2, 3),)),),
        ),
        zero_padding("zero_padding_u8", 7, 1),
    ),
    state_effects=(
        state_read(StateResource.BUFFER, "source_buffer_r8"),
        state_synchronize(StateResource.BUFFER, "source_buffer_r8"),
        state_allocate(StateResource.BUFFER, "dst_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Maps one exact HAL-buffer range in fixed SCOPED mode, invalidates "
            "READ-capable noncoherent mappings before publication, and wraps the "
            "prepared mapping in one owned core vm.buffer root."
        ),
        verification=(
            "All registers must be valid, access_u8 must be exactly 1, 2, or 3, "
            "and zero_padding_u8 must be zero.",
        ),
        preconditions=(
            "source_buffer_r8 must be a non-null exact hal.buffer and both source "
            "values must fit device size. HAL validates usage, requested access, "
            "support, and the live source range.",
        ),
        success=(
            "The published vm.buffer owns the complete prepared native mapping, "
            "uses exactly access_u8 rights and requested range length/start, "
            "replaces dst_r8, and advances the program counter by eight bytes.",
        ),
        failures=(
            *required_ref_failures(
                "hal.buffer", "No mapping is prepared and dst_r8 remains unchanged."
            ),
            FailureCase(
                "out_of_range",
                "Offset or length cannot be represented in the HAL device-size domain.",
                "No provider call occurs and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "provider_status",
                "Map, required invalidate, or cleanup returns a non-OK status.",
                "Any prepared mapping is synchronously unmapped and dst_r8 remains unchanged.",
            ),
            FailureCase(
                "resource_exhausted",
                "The mapped vm.buffer root cannot be allocated.",
                "The prepared mapping is synchronously unmapped and dst_r8 remains unchanged.",
            ),
        ),
        ownership=(
            "source_buffer_r8 is borrowed for synchronous map entry. A prepared "
            "mapping retains all HAL owners needed for address validity. Its new "
            "vm.buffer owner replaces dst_r8 only after complete construction.",
        ),
        assembly=(
            "%r<dst> = hal.buffer.map %r<source>, %v<offset>, %v<length> {access}",
        ),
        pseudocode=(
            "source = require_hal_ref(refs[source_buffer_r8], hal_buffer_type);\n"
            "offset = checked_device_size(values[source_offset_v8]);\n"
            "length = checked_device_size(values[source_length_v8]);\n"
            "mapping = zero_mapping();\n"
            "status = hal_buffer_map_range(source, SCOPED, access_u8,\n"
            "    offset, length, &mapping);\n"
            "if (status succeeded && access_u8 contains READ) {\n"
            "  status = hal_buffer_mapping_invalidate_range(mapping, 0, length);\n"
            "}\n"
            "if (status failed) {\n"
            "  return join_status(status,\n"
            "      hal_buffer_unmap_range_if_prepared(&mapping));\n"
            "}\n"
            "mapped_root = create_mapped_vm_buffer(mapping, access_u8, length);\n"
            "if (mapped_root == NULL) {\n"
            "  return join_status(resource_exhausted,\n"
            "      hal_buffer_unmap_range(&mapping));\n"
            "}\n"
            "replace_ref(&refs[dst_r8], owned_ref(mapped_root, vm_buffer_type));\n"
            "pc = pc + 8;"
        ),
    ),
)

HAL_BUFFER_UNMAP = hal_instruction(
    entity_id="hal.instruction.buffer.unmap",
    since=HAL_0,
    summary="Consumes and synchronously closes one mapped vm.buffer root.",
    opcode=0x05,
    mnemonic="hal.buffer.unmap",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        ref_register(
            "mapped_buffer_r8",
            2,
            InstructionFieldRole.OPERAND,
            "Required mapped-root vm.buffer consumed after successful preflight.",
            RuntimeRefPolicy(
                "vm.buffer",
                RefNullPolicy.REQUIRED,
                RefOwnership.CONSUME,
            ),
        ),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    state_effects=(
        state_write(StateResource.BUFFER, "mapped_buffer_r8"),
        state_synchronize(StateResource.BUFFER, "mapped_buffer_r8"),
        state_release(StateResource.BUFFER, "mapped_buffer_r8"),
    ),
    semantics=InstructionSemantics(
        description=(
            "Preflights an exact open mapped root, consumes its ref, immediately "
            "closes the hot data/access fields, flushes writable noncoherent "
            "mappings when required, and unmaps synchronously even after flush failure."
        ),
        verification=(
            "mapped_buffer_r8 must be a valid ref register and zero_padding_u8 zero.",
        ),
        preconditions=(
            "The ref must contain an exact non-null vm.buffer that is the open "
            "mapped-provider root itself, not a subspan, and has a live mapping record.",
        ),
        success=(
            "mapped_buffer_r8 is canonical null, the root and all aliases are "
            "closed tombstones, native unmap is complete, and the program counter "
            "advances by four bytes.",
        ),
        failures=(
            *required_ref_failures(
                "vm.buffer",
                "No provider call occurs and the ref/object remain unchanged.",
            ),
            FailureCase(
                "failed_precondition",
                "The object is a view, non-mapped root, closed root, or lacks a live mapping.",
                "No provider call occurs and the ref/object remain unchanged.",
            ),
            FailureCase(
                "provider_status",
                "Required flush or native unmap returns a non-OK status after preflight.",
                "The source remains null, root remains closed, all aliases remain tombstones, "
                "and unmap has been attempted exactly once.",
            ),
        ),
        ownership=(
            "Preflight leaves ownership untouched. After it succeeds, the exact "
            "owned or borrowed ref state moves out and the register clears. Final "
            "release occurs after native unmap; a consumed borrow performs no release.",
        ),
        assembly=("hal.buffer.unmap %r<mapped>",),
        pseudocode=(
            "mapped_root = preflight_open_mapped_root(refs[mapped_buffer_r8]);\n"
            "mapped_ref = move_ref(&refs[mapped_buffer_r8]);\n"
            "old_access = mapped_root.flags & (READ | WRITE);\n"
            "mapped_root.data = NULL;\n"
            "mapped_root.flags &= ~(READ | WRITE);\n"
            "status = ok;\n"
            "if (old_access contains WRITE && mapping requires flush) {\n"
            "  status = hal_buffer_mapping_flush_range(\n"
            "      mapping, 0, mapped_root.length);\n"
            "}\n"
            "status = join_status(status, hal_buffer_unmap_range(mapping));\n"
            "release_or_clear(mapped_ref);\n"
            "if (status succeeded) pc = pc + 4;\n"
            "return status;"
        ),
    ),
)

INSTRUCTIONS = (HAL_BUFFER_ALLOCATE, HAL_BUFFER_MAP, HAL_BUFFER_UNMAP)
ENTITIES = (FAMILY, *INSTRUCTIONS)
