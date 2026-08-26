# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core 0.0 vm.buffer instructions."""

from __future__ import annotations

from model.isa import (
    FailureCase,
    Instruction,
    InstructionFamily,
    InstructionFieldRole,
    InstructionSemantics,
    RefNullPolicy,
    RefOwnership,
    RuntimeRefPolicy,
    StateResource,
)
from model.isa.declarations import (
    core_instruction,
    instruction_field,
    ref_register,
    state_allocate,
    state_read,
    state_write,
    value_register,
    zero_padding,
)
from model.isa.selectors import SELECTOR_TABLES_BY_NAME
from model.isa.validation import (
    ALLOWED_VALUES,
    ANY_BITS,
    PACKED_SELECTOR_ALLOWED_PAIRS,
    PACKED_SELECTOR_TARGET_SUPPORTED,
    PACKED_SELECTORS,
    RODATA_OFFSET,
    RODATA_ORDINAL,
    SELECTOR,
    VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT,
    ZERO,
)
from model.schema import U8, U16, U32, EntityReference, FieldReference, RuleUse
from model.specification import CORE_0

FAMILY = InstructionFamily(
    entity_id="core.family.buffer",
    since=CORE_0,
    summary="Checked byte storage, views, bulk access, and atomics.",
    dependencies=("core.contract.machine",),
    document_order=1,
    normative_text=(
        "vm.buffer is the exact core ref type for fixed-length CPU-addressable "
        "byte storage. Every object has an offset-zero intrusive refcount, an "
        "immutable host-representable length, a current READ/WRITE access mask, "
        "a stable pointer for each nonzero usable range, and no-status final release. "
        "A proper subspan caches its direct byte start and immutable rights and "
        "retains a flattened root. A closable mapped root may synchronously "
        "clear its access mask and data pointer; the root and every view then "
        "remain ref-operable tombstones with stable length but no byte access "
        "or new subspan. Heap and rodata roots need no separate liveness check. "
        "Canonical null is never an empty buffer; a non-null zero-length buffer "
        "has ordinary identity and may have a null data pointer. Zero-length "
        "byte operations still check type, non-nullness, liveness, access, and "
        "one-past-end offsets, but form no pointer and make no host byte call. "
        "Every fallible check completes before destination or storage mutation. "
        "Unsynchronized conflicting byte access has ordinary source-language "
        "undefined guest-visible results but cannot escape the checked buffer. "
        "The VM adds no object lock. Execution is little-endian: multi-byte "
        "lanes and atomic carriers place their least-significant byte at the "
        "lowest address, and version zero has no byte-swapping path. Atomic "
        "carriers are naturally aligned raw i32 or i64 storage and require "
        "READ|WRITE. Integer arithmetic wraps; "
        "floating arithmetic uses the core floating profile and float.minmax "
        "rules. The CPU may strengthen atomic ordering or scope but never "
        "weaken it, and unsupported carriers make the image invalid. No buffer "
        "instruction suspends."
    ),
)


def _field(
    name: str,
    offset: int,
    encoding_id: str,
    role: InstructionFieldRole,
    description: str,
    rule_id: str,
    arguments: tuple[object, ...] = (),
):
    return instruction_field(
        name,
        offset,
        encoding_id,
        role,
        description,
        (RuleUse(rule_id, arguments),),
    )


def _buffer_ref(name: str, offset: int):
    return ref_register(
        name,
        offset,
        InstructionFieldRole.OPERAND,
        "Required exact vm.buffer ref borrowed for this instruction.",
        RuntimeRefPolicy(
            "vm.buffer",
            RefNullPolicy.REQUIRED,
            RefOwnership.BORROW,
        ),
    )


def _buffer_result(name: str, offset: int, ownership: RefOwnership):
    return ref_register(
        name,
        offset,
        InstructionFieldRole.RESULT,
        "Non-null exact vm.buffer result replacing the destination ref.",
        RuntimeRefPolicy(
            "vm.buffer",
            RefNullPolicy.RESULT_NONNULL,
            ownership,
        ),
    )


def _u16_zero(name: str, offset: int):
    return _field(
        name,
        offset,
        U16.entity_id,
        InstructionFieldRole.PADDING,
        "Canonical zero padding.",
        ZERO.entity_id,
    )


def _memory_format(offset: int):
    table = SELECTOR_TABLES_BY_NAME["memory.format"]
    return _field(
        "format_u8",
        offset,
        U8.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Closed lane element-width and lane-count selector.",
        SELECTOR.entity_id,
        (EntityReference(table.entity_id),),
    )


def _lane_range_constraint(register_field: str):
    return RuleUse(
        VALUE_REGISTER_RANGE_FROM_MEMORY_FORMAT.entity_id,
        (FieldReference(register_field), FieldReference("format_u8")),
    )


def _semantics(
    *,
    description: str,
    verification: tuple[str, ...],
    success: tuple[str, ...],
    assembly: tuple[str, ...],
    pseudocode: str,
    byte_length: int,
    preconditions: tuple[str, ...] = (),
    failures: tuple[FailureCase, ...] = (),
    ownership: tuple[str, ...] = (),
):
    return InstructionSemantics(
        description=description,
        verification=verification,
        preconditions=preconditions,
        success=(
            *success,
            f"The program counter advances by {byte_length} bytes.",
        ),
        failures=failures,
        ownership=ownership,
        assembly=assembly,
        pseudocode=pseudocode,
    )


def _required_buffer_failures(
    mutation_boundary: str,
    *,
    access: str | None = None,
    range_description: str | None = None,
    require_alignment: bool = False,
) -> tuple[FailureCase, ...]:
    failures = [
        FailureCase(
            "failed_precondition",
            "A required buffer ref is canonical null.",
            mutation_boundary,
        ),
        FailureCase(
            "invalid_argument",
            "A non-null required buffer ref has a descriptor other than vm.buffer.",
            mutation_boundary,
        ),
    ]
    if access is not None:
        failures.extend(
            (
                FailureCase(
                    "failed_precondition",
                    "The buffer root is closed, directly or through a view.",
                    mutation_boundary,
                ),
                FailureCase(
                    "permission_denied",
                    f"The buffer does not grant {access} access.",
                    mutation_boundary,
                ),
            )
        )
    if range_description is not None:
        failures.append(
            FailureCase(
                "out_of_range",
                range_description,
                mutation_boundary,
            )
        )
    if require_alignment:
        failures.append(
            FailureCase(
                "failed_precondition",
                "The concrete atomic host address is not carrier-aligned.",
                mutation_boundary,
            )
        )
    return tuple(failures)


BUFFER_ALLOCATE = core_instruction(
    entity_id="core.instruction.buffer.allocate",
    since=CORE_0,
    summary="Allocates one zeroed READ|WRITE vm.buffer.",
    opcode=0xD8,
    mnemonic="buffer.allocate",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _buffer_result("dst_r8", 1, RefOwnership.REPLACE_OWNER),
        value_register(
            "length_v8",
            2,
            InstructionFieldRole.OPERAND,
            "Unsigned 64-bit requested byte length.",
        ),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    state_effects=(state_allocate(StateResource.BUFFER, "dst_r8"),),
    semantics=_semantics(
        description=(
            "Allocates one distinct non-null buffer with immutable READ|WRITE "
            "access and zeroes every visible payload byte before publication."
        ),
        verification=(
            "dst_r8 and length_v8 must be valid register ordinals and "
            "zero_padding_u8 must equal zero.",
        ),
        preconditions=(
            "The complete u64 length, object-plus-payload size, and alignment "
            "must be host-representable and allocation must succeed.",
        ),
        success=(
            "dst_r8 releases its prior state and receives the sole owner of the "
            "new buffer. A zero length still produces a distinct object.",
        ),
        failures=(
            FailureCase(
                "resource_exhausted",
                "The request is not host-representable or allocation fails.",
                "dst_r8 remains unchanged and no partial object is published.",
            ),
        ),
        ownership=(
            "The new object's initial owner transfers to dst_r8 only after "
            "complete construction; replacement releases dst_r8's prior state.",
        ),
        assembly=("%r<dst> = buffer.allocate %v<length>",),
        pseudocode=(
            "length_u64 = values[length_v8];\n"
            "if (!fits_host_size(length_u64) ||\n"
            "    !checked_buffer_allocation_size(length_u64)) {\n"
            "  fail(resource_exhausted);\n"
            "}\n"
            "new_buffer = allocate_zeroed_rw_buffer(\n"
            "    process.host_allocator, host_size(length_u64));\n"
            "if (new_buffer == NULL) fail(resource_exhausted);\n"
            "replace_ref(&refs[dst_r8], owned_ref(new_buffer, vm_buffer_type));\n"
            "pc = pc + 4;"
        ),
        byte_length=4,
    ),
)

BUFFER_LENGTH = core_instruction(
    entity_id="core.instruction.buffer.length",
    since=CORE_0,
    summary="Returns a buffer's immutable byte length.",
    opcode=0xD9,
    mnemonic="buffer.length",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Unsigned 64-bit byte-length result.",
        ),
        _buffer_ref("buffer_r8", 2),
        zero_padding("zero_padding_u8", 3, 1),
    ),
    state_effects=(),
    semantics=_semantics(
        description=(
            "Returns the exact immutable logical byte length without inspecting "
            "payload bytes or requiring access rights. A closed tombstone keeps "
            "its length."
        ),
        verification=(
            "dst_v8 and buffer_r8 must be valid register ordinals and padding "
            "must equal zero.",
        ),
        preconditions=("buffer_r8 must contain a non-null exact vm.buffer.",),
        success=("dst_v8 receives buffer.length as a complete unsigned u64 cell.",),
        failures=_required_buffer_failures("dst_v8 remains unchanged."),
        ownership=("buffer_r8 is inspected without a retain or ownership change.",),
        assembly=("%v<dst> = buffer.length %r<buffer>",),
        pseudocode=(
            "buffer = check_deref(refs[buffer_r8], vm_buffer_type);\n"
            "values[dst_v8] = u64(buffer.length);\n"
            "pc = pc + 4;"
        ),
        byte_length=4,
    ),
)

BUFFER_SUBSPAN = core_instruction(
    entity_id="core.instruction.buffer.subspan",
    since=CORE_0,
    summary="Materializes an owned buffer view over one exact range.",
    opcode=0xDA,
    mnemonic="buffer.subspan",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        _buffer_result("dst_r8", 1, RefOwnership.REPLACE_OWNER),
        _buffer_ref("buffer_r8", 2),
        value_register(
            "offset_v8",
            3,
            InstructionFieldRole.OPERAND,
            "Unsigned 64-bit source byte offset.",
        ),
        value_register(
            "length_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Unsigned 64-bit view byte length.",
        ),
        zero_padding("zero_padding_u8", 5, 3),
    ),
    state_effects=(
        state_read(StateResource.BUFFER, "buffer_r8"),
        state_allocate(StateResource.BUFFER, "dst_r8"),
    ),
    semantics=_semantics(
        description=(
            "Creates an independently owned capability over an exact source "
            "range with inherited immutable access. A whole-source range "
            "retains the source object; a proper subrange allocates a view, "
            "flattens nested views, caches its direct start, and retains the root."
        ),
        verification=(
            "All four register ordinals must be valid and every padding byte zero.",
        ),
        preconditions=(
            "buffer_r8 must contain a non-null exact open vm.buffer and "
            "offset_v8 plus length_v8 must fit its immutable length. The root "
            "must be open even for zero-length and whole-source ranges.",
        ),
        success=(
            "Construction completes before dst_r8 replacement, so source and "
            "destination may alias. The result inherits, but cannot alter, rights.",
        ),
        failures=(
            *_required_buffer_failures(
                "buffer_r8 and dst_r8 remain unchanged.",
            ),
            FailureCase(
                "failed_precondition",
                "The buffer root is closed, directly or through a view.",
                "buffer_r8 and dst_r8 remain unchanged.",
            ),
            FailureCase(
                "out_of_range",
                "The requested source range is out of bounds.",
                "buffer_r8 and dst_r8 remain unchanged.",
            ),
            FailureCase(
                "resource_exhausted",
                "A proper view descriptor cannot be represented or allocated.",
                "buffer_r8 and dst_r8 remain unchanged.",
            ),
        ),
        ownership=(
            "A whole view is retained before replacement. A proper view owns "
            "one retained root reference. The completed result then replaces dst_r8.",
        ),
        assembly=("%r<dst> = buffer.subspan %r<buffer>, %v<offset>, %v<length>",),
        pseudocode=(
            "source = check_deref(refs[buffer_r8], vm_buffer_type);\n"
            "offset_u64 = values[offset_v8];\n"
            "length_u64 = values[length_v8];\n"
            "check_buffer_open(source);\n"
            "require checked_range(offset_u64, length_u64, source.length);\n"
            "if (offset_u64 == 0 && length_u64 == source.length) {\n"
            "  result = retain_ref(refs[buffer_r8]);\n"
            "} else {\n"
            "  result = create_owned_flattened_buffer_view_ref(\n"
            "      process.host_allocator, source, offset_u64, length_u64,\n"
            "      source.access);\n"
            "  if (result == NULL) fail(resource_exhausted);\n"
            "}\n"
            "replace_ref(&refs[dst_r8], result);\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)


def _buffer_lane_access(*, load: bool) -> Instruction:
    mnemonic = "buffer.load" if load else "buffer.store"
    value_field = "dst_v8" if load else "src_v8"
    fields = (
        (
            value_register(
                "dst_v8",
                1,
                InstructionFieldRole.RESULT,
                "First destination value-register ordinal.",
            ),
            _buffer_ref("buffer_r8", 2),
            value_register(
                "base_v8",
                3,
                InstructionFieldRole.OPERAND,
                "Unsigned 64-bit address base.",
            ),
            value_register(
                "index_v8",
                4,
                InstructionFieldRole.OPERAND,
                "Unsigned 64-bit scaled index.",
            ),
            _field(
                "scale_u8",
                5,
                U8.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Unsigned scale including the useful zero case.",
                ANY_BITS.entity_id,
            ),
            _memory_format(6),
            zero_padding("zero_padding_u8", 7, 1),
        )
        if load
        else (
            _buffer_ref("buffer_r8", 1),
            value_register(
                "base_v8",
                2,
                InstructionFieldRole.OPERAND,
                "Unsigned 64-bit address base.",
            ),
            value_register(
                "index_v8",
                3,
                InstructionFieldRole.OPERAND,
                "Unsigned 64-bit scaled index.",
            ),
            _field(
                "scale_u8",
                4,
                U8.entity_id,
                InstructionFieldRole.IMMEDIATE,
                "Unsigned scale including the useful zero case.",
                ANY_BITS.entity_id,
            ),
            value_register(
                "src_v8",
                5,
                InstructionFieldRole.OPERAND,
                "First source value-register ordinal.",
            ),
            _memory_format(6),
            zero_padding("zero_padding_u8", 7, 1),
        )
    )
    access = "READ" if load else "WRITE"
    mutation = (
        "Every destination value register remains unchanged."
        if load
        else "Every buffer byte remains unchanged."
    )
    action = "load_lanes(dst_v8, buffer.data + offset_u64, format);"
    if not load:
        action = "store_lanes(buffer.data + offset_u64, src_v8, format);"
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=f"{('Loads from' if load else 'Stores to')} a checked buffer lane group.",
        opcode=0xDB if load else 0xDC,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=fields,
        constraints=(_lane_range_constraint(value_field),),
        state_effects=(
            state_read(StateResource.BUFFER, "buffer_r8")
            if load
            else state_write(StateResource.BUFFER, "buffer_r8"),
        ),
        semantics=_semantics(
            description=(
                "Transfers a memory.format lane group at unsigned(base_v8) + "
                "unsigned(index_v8) * scale_u8 using checked u64 arithmetic and "
                "alignment-independent little-endian accesses."
            ),
            verification=(
                "format_u8 must be assigned, the complete lane register range "
                "must fit the value bank, and every named register and padding "
                "field must be valid.",
            ),
            preconditions=(
                f"buffer_r8 must contain an exact non-null open vm.buffer with "
                f"{access} access; scaled address arithmetic must not overflow "
                "and the complete access must fit buffer.length.",
            ),
            success=(
                (
                    "Base and index are snapshotted before publication, so a "
                    "destination lane may alias either address register. Each "
                    "loaded lane is zero-extended to a complete value cell."
                )
                if load
                else (
                    "Each lane stores the selected low source bits after every "
                    "address and access check succeeds."
                ),
            ),
            failures=_required_buffer_failures(
                mutation,
                access=access,
                range_description=(
                    "Scaled address arithmetic overflows u64 or the complete "
                    "lane range is outside buffer.length."
                ),
            ),
            ownership=("buffer_r8 is borrowed without a ref-count change.",),
            assembly=(
                (
                    "%v<dst>.xN = buffer.load %r<buffer>, %v<base>, "
                    "%v<index> {scale, format}"
                )
                if load
                else (
                    "buffer.store %r<buffer>, %v<base>, %v<index>, "
                    "%v<src>.xN {scale, format}"
                ),
            ),
            pseudocode=(
                "base_u64 = values[base_v8];\n"
                "index_u64 = values[index_v8];\n"
                "format = verified_memory_format(format_u8);\n"
                "buffer = check_deref(refs[buffer_r8], vm_buffer_type);\n"
                "offset_u64 = checked_buffer_offset(\n"
                "    base_u64, index_u64, scale_u8);\n"
                "require checked_range(\n"
                "    offset_u64, format.access_length, buffer.length);\n"
                f"check_buffer_access(buffer, {access});\n"
                f"{action}\n"
                "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


BUFFER_LOAD = _buffer_lane_access(load=True)
BUFFER_STORE = _buffer_lane_access(load=False)


def _packed_selector(
    name: str,
    offset: int,
    zero_mask: int,
    components: tuple[tuple[str, int, int, str, tuple[int, ...]], ...],
):
    normalized_components = tuple(
        (
            component_name,
            bit_offset,
            bit_length,
            EntityReference(SELECTOR_TABLES_BY_NAME[selector_name].entity_id),
            allowed_values,
        )
        for (
            component_name,
            bit_offset,
            bit_length,
            selector_name,
            allowed_values,
        ) in components
    )
    return _field(
        name,
        offset,
        U8.entity_id,
        InstructionFieldRole.IMMEDIATE,
        "Packed closed-selector components and canonical zero bits.",
        PACKED_SELECTORS.entity_id,
        (zero_mask, normalized_components),
    )


_ATOMIC_ALL_KINDS = tuple(range(16))
_ATOMIC_REDUCE_KINDS = tuple(range(2, 16))
_ORDERING_PAIRS = (
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (3, 0),
    (3, 1),
    (4, 0),
    (4, 1),
    (4, 4),
)


def _atomic_apply_selectors(
    selector0_offset: int,
    selector1_offset: int,
    *,
    reduce: bool,
):
    return (
        _packed_selector(
            "selector0_u8",
            selector0_offset,
            0x70,
            (
                (
                    "kind",
                    0,
                    4,
                    "buffer.atomic.kind",
                    _ATOMIC_REDUCE_KINDS if reduce else (),
                ),
                ("carrier", 7, 1, "buffer.atomic.carrier", ()),
            ),
        ),
        _packed_selector(
            "selector1_u8",
            selector1_offset,
            0xC0,
            (
                ("ordering", 0, 3, "buffer.atomic.ordering", ()),
                ("scope", 3, 3, "buffer.atomic.scope", ()),
            ),
        ),
    )


def _atomic_carrier_constraint():
    return RuleUse(
        PACKED_SELECTOR_TARGET_SUPPORTED.entity_id,
        (FieldReference("selector0_u8"), "carrier"),
    )


def _atomic_failures(mutation_boundary: str):
    return _required_buffer_failures(
        mutation_boundary,
        access="READ|WRITE",
        range_description="The complete carrier range is outside buffer.length.",
        require_alignment=True,
    )


def _atomic_apply(*, reduce: bool) -> Instruction:
    mnemonic = "buffer.atomic.reduce" if reduce else "buffer.atomic.rmw"
    selectors = _atomic_apply_selectors(
        4 if reduce else 5, 5 if reduce else 6, reduce=reduce
    )
    fields = (
        (
            _buffer_ref("buffer_r8", 1),
            value_register(
                "offset_v8",
                2,
                InstructionFieldRole.OPERAND,
                "Unsigned carrier byte offset.",
            ),
            value_register(
                "operand_v8",
                3,
                InstructionFieldRole.OPERAND,
                "Low carrier-width update operand.",
            ),
            *selectors,
            _u16_zero("zero_padding_u16", 6),
        )
        if reduce
        else (
            value_register(
                "old_v8",
                1,
                InstructionFieldRole.RESULT,
                "Carrier value immediately preceding the committed update.",
            ),
            _buffer_ref("buffer_r8", 2),
            value_register(
                "offset_v8",
                3,
                InstructionFieldRole.OPERAND,
                "Unsigned carrier byte offset.",
            ),
            value_register(
                "operand_v8",
                4,
                InstructionFieldRole.OPERAND,
                "Low carrier-width update operand.",
            ),
            *selectors,
            zero_padding("zero_padding_u8", 7, 1),
        )
    )
    result_text = (
        "The selected update commits atomically without publishing its old value."
        if reduce
        else (
            "old_v8 receives the carrier value immediately before the committed "
            "update; i32 clears the high 32 cell bits and i64 fills the cell."
        )
    )
    mutation = (
        "No buffer byte or VM register changes."
        if reduce
        else "No buffer byte changes and old_v8 remains unchanged."
    )
    return core_instruction(
        entity_id=f"core.instruction.{mnemonic}",
        since=CORE_0,
        summary=(
            "Atomically reduces one raw buffer carrier."
            if reduce
            else "Atomically updates and returns one raw buffer carrier."
        ),
        opcode=0xDD if reduce else 0xDE,
        mnemonic=mnemonic,
        byte_length=8,
        family_id=FAMILY.entity_id,
        fields=fields,
        constraints=(_atomic_carrier_constraint(),),
        state_effects=(
            state_read(StateResource.BUFFER, "buffer_r8"),
            state_write(StateResource.BUFFER, "buffer_r8"),
        ),
        semantics=_semantics(
            description=(
                "Atomically applies the selected integer or floating update to "
                "one i32/i64 raw storage carrier with the selected ordering and "
                "scope. Reduce excludes exchange kinds; RMW accepts all kinds."
            ),
            verification=(
                "Every register and padding field must be valid; packed selector "
                "bits must name assigned kind, carrier, ordering, and scope values.",
                "The target contract must support the selected carrier. "
                "buffer.atomic.reduce kind must be one of 2..15.",
            ),
            preconditions=(
                "buffer_r8 must be a non-null exact open vm.buffer with READ|WRITE; "
                "the complete carrier must fit and its concrete host address "
                "must be naturally aligned.",
            ),
            success=(result_text,),
            failures=_atomic_failures(mutation),
            ownership=("buffer_r8 is borrowed without a ref-count change.",),
            assembly=(
                (
                    "buffer.atomic.reduce %r<buffer>, %v<offset>, %v<operand> "
                    "{carrier, kind, ordering, scope}"
                )
                if reduce
                else (
                    "%v<old> = buffer.atomic.rmw %r<buffer>, %v<offset>, "
                    "%v<operand> {carrier, kind, ordering, scope}"
                ),
            ),
            pseudocode=(
                "offset_u64 = values[offset_v8];\n"
                "operand_bits = low_carrier_bits(values[operand_v8]);\n"
                "address = check_atomic_buffer_address(\n"
                "    refs[buffer_r8], offset_u64, carrier);\n"
                + (
                    "raw_atomic_apply(address, operand_bits,\n"
                    "                 kind, ordering, scope);\n"
                    if reduce
                    else (
                        "old_bits = raw_atomic_apply(\n"
                        "    address, operand_bits, kind, ordering, scope);\n"
                        "values[old_v8] = canonicalize_carrier(old_bits, carrier);\n"
                    )
                )
                + "pc = pc + 8;"
            ),
            byte_length=8,
        ),
    )


BUFFER_ATOMIC_REDUCE = _atomic_apply(reduce=True)
BUFFER_ATOMIC_RMW = _atomic_apply(reduce=False)

BUFFER_ATOMIC_CMPXCHG = core_instruction(
    entity_id="core.instruction.buffer.atomic.cmpxchg",
    since=CORE_0,
    summary="Performs strong exact-bit atomic compare-exchange.",
    opcode=0xDF,
    mnemonic="buffer.atomic.cmpxchg",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "old_v8",
            1,
            InstructionFieldRole.RESULT,
            "Actual carrier value observed by the comparison attempt.",
        ),
        _buffer_ref("buffer_r8", 2),
        value_register(
            "offset_v8",
            3,
            InstructionFieldRole.OPERAND,
            "Unsigned carrier byte offset.",
        ),
        value_register(
            "expected_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Low carrier-width expected bit pattern.",
        ),
        value_register(
            "replacement_v8",
            5,
            InstructionFieldRole.OPERAND,
            "Low carrier-width replacement bit pattern.",
        ),
        _packed_selector(
            "selector0_u8",
            6,
            0x40,
            (
                ("success_ordering", 0, 3, "buffer.atomic.ordering", ()),
                ("failure_ordering", 3, 3, "buffer.atomic.ordering", ()),
                ("carrier", 7, 1, "buffer.atomic.carrier", ()),
            ),
        ),
        _packed_selector(
            "selector1_u8",
            7,
            0xF8,
            (("scope", 0, 3, "buffer.atomic.scope", ()),),
        ),
    ),
    constraints=(
        _atomic_carrier_constraint(),
        RuleUse(
            PACKED_SELECTOR_ALLOWED_PAIRS.entity_id,
            (
                FieldReference("selector0_u8"),
                "success_ordering",
                "failure_ordering",
                _ORDERING_PAIRS,
            ),
        ),
    ),
    state_effects=(
        state_read(StateResource.BUFFER, "buffer_r8"),
        state_write(StateResource.BUFFER, "buffer_r8"),
    ),
    semantics=_semantics(
        description=(
            "Performs one strong, non-spurious exact-bit compare-exchange on a "
            "raw i32/i64 carrier and returns the actual value observed whether "
            "the comparison succeeds or fails."
        ),
        verification=(
            "Every register must be valid; selector components and zero bits "
            "must be canonical and the carrier target-supported.",
            "Failure ordering must be one of the nine encoded pairs: it cannot "
            "be release/acq_rel or stronger than success ordering.",
        ),
        preconditions=(
            "buffer_r8 must be a non-null exact open vm.buffer with READ|WRITE; "
            "the complete carrier must fit and its concrete host address must "
            "be naturally aligned.",
        ),
        success=(
            "All value inputs are snapshotted before old_v8 publication. The "
            "comparison and conditional replacement use exact low carrier bits, "
            "and old_v8 receives the observed carrier in canonical cell form.",
        ),
        failures=_atomic_failures(
            "No buffer byte changes and old_v8 remains unchanged."
        ),
        ownership=("buffer_r8 is borrowed without a ref-count change.",),
        assembly=(
            "%v<old> = buffer.atomic.cmpxchg %r<buffer>, %v<offset>, "
            "%v<expected>, %v<replacement> "
            "{carrier, success_ordering, failure_ordering, scope}",
        ),
        pseudocode=(
            "offset_u64 = values[offset_v8];\n"
            "expected_bits = low_carrier_bits(values[expected_v8]);\n"
            "replacement_bits = low_carrier_bits(values[replacement_v8]);\n"
            "address = check_atomic_buffer_address(\n"
            "    refs[buffer_r8], offset_u64, carrier);\n"
            "old_bits = raw_atomic_compare_exchange_strong(\n"
            "    address, expected_bits, replacement_bits,\n"
            "    success_ordering, failure_ordering, scope);\n"
            "values[old_v8] = canonicalize_carrier(old_bits, carrier);\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

BUFFER_FILL = core_instruction(
    entity_id="core.instruction.buffer.fill",
    since=CORE_0,
    summary="Repeats a byte pattern across a checked writable range.",
    opcode=0xE0,
    mnemonic="buffer.fill",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        _buffer_ref("buffer_r8", 1),
        value_register(
            "offset_v8", 2, InstructionFieldRole.OPERAND, "Unsigned byte offset."
        ),
        value_register(
            "length_v8", 3, InstructionFieldRole.OPERAND, "Unsigned byte length."
        ),
        value_register(
            "pattern_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Low-byte repeating pattern.",
        ),
        _field(
            "pattern_width_u8",
            5,
            U8.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Pattern width in bytes.",
            ALLOWED_VALUES.entity_id,
            ((1, 2, 4, 8),),
        ),
        _u16_zero("zero_padding_u16", 6),
    ),
    state_effects=(state_write(StateResource.BUFFER, "buffer_r8"),),
    semantics=_semantics(
        description=(
            "Repeats the low one, two, four, or eight little-endian bytes of "
            "pattern_v8 across the exact writable range; the final repetition "
            "may be partial."
        ),
        verification=(
            "All register ordinals must be valid, pattern_width_u8 must equal "
            "1, 2, 4, or 8, and padding must be zero.",
        ),
        preconditions=(
            "buffer_r8 must be an exact non-null open vm.buffer with WRITE, and "
            "offset_v8 plus length_v8 must fit its byte length.",
        ),
        success=(
            "The complete pattern and range are snapshotted and checked before "
            "exactly length_v8 bytes are written. Zero length forms no pointer.",
        ),
        failures=_required_buffer_failures(
            "Every target byte remains unchanged.",
            access="WRITE",
            range_description="The requested target range is out of bounds.",
        ),
        ownership=("buffer_r8 is borrowed without a ref-count change.",),
        assembly=(
            "buffer.fill %r<buffer>, %v<offset>, %v<length>, %v<pattern> "
            "{pattern_width}",
        ),
        pseudocode=(
            "buffer = check_deref(refs[buffer_r8], vm_buffer_type);\n"
            "offset_u64 = values[offset_v8];\n"
            "length_u64 = values[length_v8];\n"
            "pattern_bits = values[pattern_v8];\n"
            "require checked_range(offset_u64, length_u64, buffer.length);\n"
            "check_buffer_access(buffer, WRITE);\n"
            "if (length_u64 != 0) {\n"
            "  fill_repeating_le_pattern(buffer.data + offset_u64,\n"
            "      length_u64, pattern_bits, pattern_width_u8);\n"
            "}\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

BUFFER_COPY = core_instruction(
    entity_id="core.instruction.buffer.copy",
    since=CORE_0,
    summary="Moves one checked buffer range to another.",
    opcode=0xE1,
    mnemonic="buffer.copy",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        _buffer_ref("target_r8", 1),
        value_register(
            "target_offset_v8",
            2,
            InstructionFieldRole.OPERAND,
            "Unsigned target byte offset.",
        ),
        _buffer_ref("source_r8", 3),
        value_register(
            "source_offset_v8",
            4,
            InstructionFieldRole.OPERAND,
            "Unsigned source byte offset.",
        ),
        value_register(
            "length_v8", 5, InstructionFieldRole.OPERAND, "Unsigned byte length."
        ),
        _u16_zero("zero_padding_u16", 6),
    ),
    state_effects=(
        state_read(StateResource.BUFFER, "source_r8"),
        state_write(StateResource.BUFFER, "target_r8"),
    ),
    semantics=_semantics(
        description=(
            "Moves one checked source range to one checked target range with "
            "memmove semantics across identical objects, flattened views, and "
            "distinct wrappers over overlapping external storage."
        ),
        verification=(
            "Every register ordinal must be valid and zero_padding_u16 must be zero.",
        ),
        preconditions=(
            "Both refs must be exact non-null open vm.buffers; source requires "
            "READ, target requires WRITE, and both complete ranges must fit.",
        ),
        success=(
            "Every check completes before mutation. The target receives the "
            "source bytes as if through a temporary copy; zero length forms no pointer.",
        ),
        failures=(
            *_required_buffer_failures(
                "Every target byte remains unchanged.",
                range_description="Either complete source or target range is out of bounds.",
            ),
            FailureCase(
                "failed_precondition",
                "The source or target root is closed.",
                "Every target byte remains unchanged.",
            ),
            FailureCase(
                "permission_denied",
                "The source lacks READ or target lacks WRITE access.",
                "Every target byte remains unchanged.",
            ),
        ),
        ownership=("Both buffer refs are borrowed without ref-count changes.",),
        assembly=(
            "buffer.copy %r<target>, %v<target_offset>, %r<source>, "
            "%v<source_offset>, %v<length>",
        ),
        pseudocode=(
            "target = check_deref(refs[target_r8], vm_buffer_type);\n"
            "source = check_deref(refs[source_r8], vm_buffer_type);\n"
            "target_offset_u64 = values[target_offset_v8];\n"
            "source_offset_u64 = values[source_offset_v8];\n"
            "length_u64 = values[length_v8];\n"
            "require checked_range(target_offset_u64, length_u64, target.length);\n"
            "require checked_range(source_offset_u64, length_u64, source.length);\n"
            "check_buffer_access(target, WRITE);\n"
            "check_buffer_access(source, READ);\n"
            "if (length_u64 != 0) {\n"
            "  move_bytes(target.data + target_offset_u64,\n"
            "             source.data + source_offset_u64, length_u64);\n"
            "}\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

BUFFER_COMPARE = core_instruction(
    entity_id="core.instruction.buffer.compare",
    since=CORE_0,
    summary="Lexicographically compares two checked buffer ranges.",
    opcode=0xE2,
    mnemonic="buffer.compare",
    byte_length=8,
    family_id=FAMILY.entity_id,
    fields=(
        value_register(
            "dst_v8",
            1,
            InstructionFieldRole.RESULT,
            "Canonical signed i32 comparison result.",
        ),
        _buffer_ref("lhs_r8", 2),
        value_register(
            "lhs_offset_v8",
            3,
            InstructionFieldRole.OPERAND,
            "Unsigned left byte offset.",
        ),
        _buffer_ref("rhs_r8", 4),
        value_register(
            "rhs_offset_v8",
            5,
            InstructionFieldRole.OPERAND,
            "Unsigned right byte offset.",
        ),
        value_register(
            "length_v8", 6, InstructionFieldRole.OPERAND, "Unsigned byte length."
        ),
        zero_padding("zero_padding_u8", 7, 1),
    ),
    state_effects=(
        state_read(StateResource.BUFFER, "lhs_r8"),
        state_read(StateResource.BUFFER, "rhs_r8"),
    ),
    semantics=_semantics(
        description=(
            "Lexicographically compares equal-length checked ranges as unsigned "
            "bytes and returns canonical signed i32 -1, 0, or +1."
        ),
        verification=(
            "Every register ordinal must be valid and zero_padding_u8 must be zero.",
        ),
        preconditions=(
            "Both refs must be exact non-null open vm.buffers granting READ and "
            "both complete ranges must fit.",
        ),
        success=(
            "All value operands and checks precede publication. dst_v8 receives "
            "UINT32_MAX, zero, or one with high 32 cell bits zero; empty ranges "
            "compare equal.",
        ),
        failures=(
            *_required_buffer_failures(
                "dst_v8 remains unchanged.",
                range_description="Either complete comparison range is out of bounds.",
            ),
            FailureCase(
                "failed_precondition",
                "Either buffer root is closed.",
                "dst_v8 remains unchanged.",
            ),
            FailureCase(
                "permission_denied",
                "Either buffer lacks READ access.",
                "dst_v8 remains unchanged.",
            ),
        ),
        ownership=("Both buffer refs are borrowed without ref-count changes.",),
        assembly=(
            "%v<dst> = buffer.compare %r<lhs>, %v<lhs_offset>, %r<rhs>, "
            "%v<rhs_offset>, %v<length>",
        ),
        pseudocode=(
            "lhs = check_deref(refs[lhs_r8], vm_buffer_type);\n"
            "rhs = check_deref(refs[rhs_r8], vm_buffer_type);\n"
            "lhs_offset_u64 = values[lhs_offset_v8];\n"
            "rhs_offset_u64 = values[rhs_offset_v8];\n"
            "length_u64 = values[length_v8];\n"
            "require checked_range(lhs_offset_u64, length_u64, lhs.length);\n"
            "require checked_range(rhs_offset_u64, length_u64, rhs.length);\n"
            "check_buffer_access(lhs, READ);\n"
            "check_buffer_access(rhs, READ);\n"
            "ordering = length_u64 == 0 ? 0 : compare_unsigned_bytes(\n"
            "    lhs.data + lhs_offset_u64, rhs.data + rhs_offset_u64,\n"
            "    length_u64);\n"
            "values[dst_v8] = ordering < 0 ? UINT32_MAX :\n"
            "                 ordering > 0 ? 1 : 0;\n"
            "pc = pc + 8;"
        ),
        byte_length=8,
    ),
)

BUFFER_COPY_RODATA = core_instruction(
    entity_id="core.instruction.buffer.copy.rodata",
    since=CORE_0,
    summary="Copies module rodata into a checked writable buffer range.",
    opcode=0xE3,
    mnemonic="buffer.copy.rodata",
    byte_length=12,
    family_id=FAMILY.entity_id,
    fields=(
        _buffer_ref("target_r8", 1),
        value_register(
            "target_offset_v8",
            2,
            InstructionFieldRole.OPERAND,
            "Unsigned target byte offset.",
        ),
        zero_padding("zero_padding0_u8", 3, 1),
        _field(
            "rodata_u16",
            4,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Direct module rodata-block ordinal.",
            RODATA_ORDINAL.entity_id,
        ),
        value_register(
            "length_v8", 6, InstructionFieldRole.OPERAND, "Unsigned byte length."
        ),
        zero_padding("zero_padding1_u8", 7, 1),
        _field(
            "source_offset_u32",
            8,
            U32.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Statically in-range or one-past-end rodata offset.",
            RODATA_OFFSET.entity_id,
            (FieldReference("rodata_u16"),),
        ),
    ),
    state_effects=(state_write(StateResource.BUFFER, "target_r8"),),
    semantics=_semantics(
        description=(
            "Copies a dynamically sized immutable module-rodata range into a "
            "dynamically offset writable buffer range."
        ),
        verification=(
            "All register ordinals and padding must be valid; rodata_u16 must "
            "exist and source_offset_u32 must not exceed its length.",
        ),
        preconditions=(
            "target_r8 must be an exact non-null open vm.buffer with WRITE; "
            "both dynamic source and target ranges must fit.",
        ),
        success=(
            "Every check precedes mutation. The immutable image and writable "
            "buffer cannot alias; zero length forms no pointer.",
        ),
        failures=_required_buffer_failures(
            "Every target byte remains unchanged.",
            access="WRITE",
            range_description="The dynamic source or target range is out of bounds.",
        ),
        ownership=("target_r8 is borrowed without a ref-count change.",),
        assembly=(
            "buffer.copy.rodata %r<target>, %v<target_offset>, "
            "@rodata<ordinal>+<source_offset>, %v<length>",
        ),
        pseudocode=(
            "target = check_deref(refs[target_r8], vm_buffer_type);\n"
            "target_offset_u64 = values[target_offset_v8];\n"
            "length_u64 = values[length_v8];\n"
            "source = verified_rodata_blocks[rodata_u16];\n"
            "require checked_range(target_offset_u64, length_u64, target.length);\n"
            "require checked_range(source_offset_u32, length_u64, source.length);\n"
            "check_buffer_access(target, WRITE);\n"
            "if (length_u64 != 0) {\n"
            "  copy_bytes(target.data + target_offset_u64,\n"
            "             source.data + source_offset_u32, length_u64);\n"
            "}\n"
            "pc = pc + 12;"
        ),
        byte_length=12,
    ),
)

BUFFER_RODATA_LOAD = core_instruction(
    entity_id="core.instruction.buffer.rodata.load",
    since=CORE_0,
    summary="Borrows a module-owned read-only rodata buffer view.",
    opcode=0xE4,
    mnemonic="buffer.rodata.load",
    byte_length=4,
    family_id=FAMILY.entity_id,
    fields=(
        _buffer_result("dst_r8", 1, RefOwnership.REPLACE_BORROW),
        _field(
            "rodata_u16",
            2,
            U16.entity_id,
            InstructionFieldRole.IMMEDIATE,
            "Direct module rodata-block ordinal.",
            RODATA_ORDINAL.entity_id,
        ),
    ),
    state_effects=(),
    semantics=_semantics(
        description=(
            "Publishes an internal borrow of the complete immutable READ-only "
            "vm.buffer view prebuilt and owned by the loaded module."
        ),
        verification=(
            "dst_r8 must be valid and rodata_u16 must name an existing block.",
        ),
        success=(
            "dst_r8 releases its prior state and receives the canonical borrowed "
            "view without an atomic retain.",
        ),
        ownership=(
            "The loaded module holds the view's initial owner; the view "
            "independently retains image storage. Borrow promotion creates an "
            "ordinary owner, so escaped views outlive module teardown without "
            "forming an ownership cycle.",
        ),
        assembly=("%r<dst> = buffer.rodata.load @rodata<ordinal>",),
        pseudocode=(
            "view = loaded_module.rodata_views[rodata_u16];\n"
            "old_destination = refs[dst_r8];\n"
            "refs[dst_r8] = borrowed_ref(view, vm_buffer_type);\n"
            "release_or_clear(old_destination);\n"
            "pc = pc + 4;"
        ),
        byte_length=4,
    ),
)

INSTRUCTIONS = (
    BUFFER_ALLOCATE,
    BUFFER_LENGTH,
    BUFFER_SUBSPAN,
    BUFFER_LOAD,
    BUFFER_STORE,
    BUFFER_ATOMIC_REDUCE,
    BUFFER_ATOMIC_RMW,
    BUFFER_ATOMIC_CMPXCHG,
    BUFFER_FILL,
    BUFFER_COPY,
    BUFFER_COMPARE,
    BUFFER_COPY_RODATA,
    BUFFER_RODATA_LOAD,
)
ENTITIES = (FAMILY, *INSTRUCTIONS)
