# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checked byte storage, views, bulk access, and atomics."""

from __future__ import annotations

import enum

from iree.vm.bytecode.spec.isa import (
    FailureCase,
    FieldRole,
    FieldRuleUse,
    Instruction,
    InstructionFamily,
    InstructionField,
    PackedSelectorComponent,
    RecordRule,
    RuntimeRefPolicy,
    StateEffect,
)
from iree.vm.bytecode.spec.isa.core.rules import (
    FieldRule,
    RecordRuleKind,
    RefNullPolicy,
    RefOwnership,
    StateAccess,
    StateResource,
)
from iree.vm.bytecode.spec.isa.core.stack import (
    MEMORY_FORMAT_MAXIMUM_LANE_COUNT,
    MEMORY_FORMAT_SELECTOR,
)
from iree.vm.bytecode.spec.schema import (
    U8,
    U16,
    U32,
    Field,
    NumericKind,
    NumericTable,
    NumericValue,
    is_name,
    require,
)
from iree.vm.bytecode.spec.version import CORE_0


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


BUFFER_ATOMIC_KIND_SELECTOR = _selector(
    "buffer.atomic.kind",
    (
        "Selects the exact replacement function for an atomic carrier. Integer "
        "arithmetic wraps at carrier width; floating operations inherit the "
        "selected-width floating profile and float.minmax rules."
    ),
    (
        ("exchange.integer", 0, "Replaces the carrier bits with operand bits."),
        (
            "exchange.float",
            1,
            "Replaces the f32/f64 carrier bits with operand bits.",
        ),
        ("add.integer", 2, "Replaces with unsigned modular old+operand."),
        ("add.float", 3, "Replaces with selected-width IEEE old+operand."),
        ("subtract.integer", 4, "Replaces with unsigned modular old-operand."),
        ("and.integer", 5, "Replaces with the bitwise old AND operand."),
        ("or.integer", 6, "Replaces with the bitwise old OR operand."),
        ("xor.integer", 7, "Replaces with the bitwise old XOR operand."),
        (
            "minimum.signed",
            8,
            "Replaces with the two's-complement signed minimum.",
        ),
        (
            "maximum.signed",
            9,
            "Replaces with the two's-complement signed maximum.",
        ),
        ("minimum.unsigned", 10, "Replaces with the unsigned minimum."),
        ("maximum.unsigned", 11, "Replaces with the unsigned maximum."),
        ("minimum.float", 12, "Replaces with float.minmax minimum."),
        ("maximum.float", 13, "Replaces with float.minmax maximum."),
        ("minnum.float", 14, "Replaces with float.minmax minnum."),
        ("maxnum.float", 15, "Replaces with float.minmax maxnum."),
    ),
)

BUFFER_ATOMIC_CARRIER_SELECTOR = _selector(
    "buffer.atomic.carrier",
    "Selects the naturally aligned raw-storage carrier width.",
    (
        ("i32", 0, "Uses a four-byte carrier and low 32 value-cell bits."),
        ("i64", 1, "Uses an eight-byte carrier and the complete value cell."),
    ),
)

BUFFER_ATOMIC_ORDERING_SELECTOR = _selector(
    "buffer.atomic.ordering",
    (
        "Selects the minimum C11-style synchronization ordering. A target may "
        "strengthen but never weaken it."
    ),
    (
        ("relaxed", 0, "Guarantees atomicity without inter-operation ordering."),
        ("acquire", 1, "Applies acquire ordering to the operation's read."),
        ("release", 2, "Applies release ordering to the operation's write."),
        (
            "acq_rel",
            3,
            "Applies acquire ordering to the read and release ordering to the write.",
        ),
        (
            "seq_cst",
            4,
            "Participates in one sequentially consistent total order.",
        ),
    ),
)

BUFFER_ATOMIC_SCOPE_SELECTOR = _selector(
    "buffer.atomic.scope",
    (
        "Selects the minimum synchronization domain. CPU interpretation may "
        "strengthen a narrower domain to process/system scope."
    ),
    (
        ("thread", 0, "Requires ordering only within the current thread."),
        (
            "subgroup",
            1,
            "Requires ordering among invocations in one execution subgroup.",
        ),
        (
            "workgroup",
            2,
            "Requires ordering among invocations in one workgroup.",
        ),
        ("device", 3, "Requires ordering among agents on one logical device."),
        (
            "system",
            4,
            "Requires ordering across every participating system agent.",
        ),
    ),
)

BUFFER_SELECTORS = (
    BUFFER_ATOMIC_KIND_SELECTOR,
    BUFFER_ATOMIC_CARRIER_SELECTOR,
    BUFFER_ATOMIC_ORDERING_SELECTOR,
    BUFFER_ATOMIC_SCOPE_SELECTOR,
)

BUFFER_FAMILY = InstructionFamily(
    name="buffer",
    since=CORE_0,
    summary="Checked byte storage, views, bulk access, and atomics.",
    contract=(
        "vm.buffer is the exact Core ref type for fixed-length CPU-addressable "
        "byte storage. Every object has an offset-zero intrusive refcount, an "
        "immutable host-representable length, a current READ/WRITE access mask, "
        "a stable pointer for every nonzero usable range, and no-status final "
        "release. A proper subspan caches its direct byte start and immutable "
        "rights and retains a flattened root. A closable mapped root synchronously "
        "clears its access mask and data pointer when closed; the root and every "
        "view remain ref-operable tombstones with stable lengths but allow no byte "
        "access or new subspan. Heap and rodata roots need no liveness branch. "
        "Canonical null is not an empty buffer; a non-null zero-length buffer has "
        "ordinary identity and may have a null data pointer. Zero-length byte "
        "operations still check type, non-nullness, liveness, access, and the "
        "one-past-end offset, but form no pointer and call no host byte primitive. "
        "All fallible checks finish before destination or storage mutation. "
        "Unsynchronized conflicting byte access has ordinary source-language "
        "undefined behavior. This is intentionally not a VM safety property: "
        "trusted host programs and GPU-like kernels may corrupt their own byte "
        "buffers, while isolation between workloads is a process or machine "
        "concern. The VM adds no object lock. Execution is little-endian and "
        "Core 0.0 has no byte-swapping path. Atomic carriers are naturally aligned "
        "raw i32 or i64 storage requiring READ|WRITE. Integer arithmetic wraps; "
        "floating arithmetic uses the Core floating profile and float.minmax "
        "rules. A CPU may strengthen atomic ordering or scope but never weaken it. "
        "Unsupported carriers make the module incompatible at creation. No buffer "
        "instruction suspends."
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


def _value(name: str, role: FieldRole, summary: str) -> InstructionField:
    return _field(name, U8, summary, role, FieldRule.REGISTER_VALUE)


def _buffer_ref(name: str) -> InstructionField:
    return InstructionField(
        Field(name, U8, "Required exact vm.buffer ref borrowed for this instruction."),
        FieldRole.OPERAND,
        FieldRuleUse(FieldRule.REGISTER_REF),
        RuntimeRefPolicy("vm.buffer", RefNullPolicy.REQUIRED, RefOwnership.BORROW),
    )


def _buffer_result(name: str, ownership: RefOwnership) -> InstructionField:
    return InstructionField(
        Field(name, U8, "Non-null exact vm.buffer result replacing this ref register."),
        FieldRole.RESULT,
        FieldRuleUse(FieldRule.REGISTER_REF),
        RuntimeRefPolicy("vm.buffer", RefNullPolicy.RESULT_NONNULL, ownership),
    )


def _padding(name: str, encoding=U8, *, element_count: int = 1) -> InstructionField:
    return _field(
        name,
        encoding,
        "Canonical zero padding.",
        FieldRole.PADDING,
        FieldRule.ZERO,
        element_count=element_count,
    )


def _memory_format() -> InstructionField:
    return _field(
        "format_u8",
        U8,
        "Closed lane element-width and lane-count selector.",
        FieldRole.IMMEDIATE,
        FieldRuleUse(FieldRule.SELECTOR, data=MEMORY_FORMAT_SELECTOR),
    )


def _value_format_range(register_name: str) -> RecordRule:
    return RecordRule(
        RecordRuleKind.VALUE_REGISTER_FORMAT_RANGE,
        fields=(register_name, "format_u8"),
        values=(MEMORY_FORMAT_MAXIMUM_LANE_COUNT,),
        summary=(
            f"{register_name} and format_u8 must select a complete in-range "
            "consecutive value-register run."
        ),
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
            FailureCase("out_of_range", range_description, mutation_boundary)
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


BUFFER_ALLOCATE = Instruction(
    opcode=0xD8,
    mnemonic="buffer.allocate",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Allocates one zeroed READ|WRITE vm.buffer.",
    fields=(
        _buffer_result("destination_r8", RefOwnership.REPLACE_OWNER),
        _value(
            "length_v8",
            FieldRole.OPERAND,
            "Unsigned 64-bit requested byte length.",
        ),
        _field(
            "minimum_alignment_log2_u8",
            U8,
            "Base-two logarithm of the required payload-byte alignment.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.ALLOWED_RANGE, values=(0, 63)),
        ),
    ),
    semantics=None,
    behavior=(
        "Allocates one distinct non-null buffer with immutable READ|WRITE access "
        "and zeros every visible payload byte before publication."
    ),
    success=(
        "destination_r8 releases its prior state and receives the sole owner of "
        "the new buffer. Zero length still produces a distinct object.",
    ),
    assembly=("%r<destination> = buffer.allocate %v<length> {minimum_alignment_log2}"),
    pseudocode=(
        "length_u64 = values[length_v8];\n"
        "alignment_u64 = 1 << minimum_alignment_log2_u8;\n"
        "if (!fits_host_size(length_u64) ||\n"
        "    !fits_host_size(alignment_u64) ||\n"
        "    !checked_buffer_allocation_size(length_u64, alignment_u64)) {\n"
        "  fail(resource_exhausted);\n"
        "}\n"
        "new_buffer = allocate_zeroed_rw_buffer(\n"
        "    process.host_allocator, host_size(length_u64),\n"
        "    host_size(alignment_u64));\n"
        "if (new_buffer == NULL) fail(resource_exhausted);\n"
        "replace_ref(&refs[destination_r8], owned_ref(new_buffer, vm_buffer_type));\n"
        "pc = pc + 4;"
    ),
    state_effects=(
        StateEffect(StateAccess.ALLOCATE, StateResource.BUFFER, ("destination_r8",)),
    ),
    preconditions=(
        "The complete u64 length, object-plus-payload size, and alignment must "
        "be host-representable and allocation must succeed.",
    ),
    failures=(
        FailureCase(
            "resource_exhausted",
            "The request is not host-representable or allocation fails.",
            "destination_r8 remains unchanged and no partial object is published.",
        ),
    ),
    ownership=(
        "The new object's initial owner transfers to destination_r8 only after "
        "complete construction; replacement releases destination_r8's prior state.",
    ),
)

BUFFER_LENGTH = Instruction(
    opcode=0xD9,
    mnemonic="buffer.length",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Returns a buffer's immutable byte length.",
    fields=(
        _value(
            "destination_v8",
            FieldRole.RESULT,
            "Unsigned 64-bit byte-length result.",
        ),
        InstructionField(
            Field(
                "buffer_r8",
                U8,
                "Nullable exact vm.buffer ref borrowed for this instruction.",
            ),
            FieldRole.OPERAND,
            FieldRuleUse(FieldRule.REGISTER_REF),
            RuntimeRefPolicy("vm.buffer", RefNullPolicy.NULLABLE, RefOwnership.BORROW),
        ),
        _padding("zero_padding_u8"),
    ),
    semantics=None,
    behavior=(
        "Returns the exact immutable logical byte length without inspecting payload "
        "bytes or requiring access rights. Canonical null returns zero; a closed "
        "tombstone retains its length."
    ),
    success=(
        "destination_v8 receives zero for canonical null or buffer.length for a "
        "non-null buffer as a complete unsigned u64 cell.",
    ),
    assembly="%v<destination> = buffer.length %r<buffer>",
    pseudocode=(
        "buffer = check_optional_deref(refs[buffer_r8], vm_buffer_type);\n"
        "values[destination_v8] = buffer == NULL ? 0 : u64(buffer.length);\n"
        "pc = pc + 4;"
    ),
    preconditions=("A non-null buffer_r8 must contain an exact vm.buffer.",),
    failures=(
        FailureCase(
            "invalid_argument",
            "A non-null buffer_r8 has a descriptor other than vm.buffer.",
            "destination_v8 remains unchanged.",
        ),
    ),
    ownership=(
        "A non-null buffer_r8 is inspected without a retain or ownership change.",
    ),
)

BUFFER_SUBSPAN = Instruction(
    opcode=0xDA,
    mnemonic="buffer.subspan",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Materializes an owned buffer view over one exact range.",
    fields=(
        _buffer_result("destination_r8", RefOwnership.REPLACE_OWNER),
        _buffer_ref("buffer_r8"),
        _value(
            "offset_v8",
            FieldRole.OPERAND,
            "Unsigned 64-bit source byte offset.",
        ),
        _value(
            "length_v8",
            FieldRole.OPERAND,
            "Unsigned 64-bit view byte length.",
        ),
        _padding("zero_padding_u8", element_count=3),
    ),
    semantics=None,
    behavior=(
        "Creates an independently owned capability over an exact source range with "
        "inherited immutable access. A whole-source range retains the source object; "
        "a proper subrange allocates a view, flattens nested views, caches its direct "
        "start, and retains the root."
    ),
    success=(
        "Construction finishes before destination_r8 replacement, so source and "
        "destination may alias. The result inherits but cannot alter source rights.",
    ),
    assembly=("%r<destination> = buffer.subspan %r<buffer>, %v<offset>, %v<length>"),
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
        "replace_ref(&refs[destination_r8], result);\n"
        "pc = pc + 8;"
    ),
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.BUFFER, ("buffer_r8",)),
        StateEffect(StateAccess.ALLOCATE, StateResource.BUFFER, ("destination_r8",)),
    ),
    preconditions=(
        "buffer_r8 must contain a non-null exact open vm.buffer and offset_v8 "
        "plus length_v8 must fit its immutable length. The root must be open even "
        "for zero-length and whole-source ranges.",
    ),
    failures=(
        *_required_buffer_failures("buffer_r8 and destination_r8 remain unchanged."),
        FailureCase(
            "failed_precondition",
            "The buffer root is closed, directly or through a view.",
            "buffer_r8 and destination_r8 remain unchanged.",
        ),
        FailureCase(
            "out_of_range",
            "The requested source range is out of bounds.",
            "buffer_r8 and destination_r8 remain unchanged.",
        ),
        FailureCase(
            "resource_exhausted",
            "A proper view descriptor cannot be represented or allocated.",
            "buffer_r8 and destination_r8 remain unchanged.",
        ),
    ),
    ownership=(
        "A whole view is retained before replacement. A proper view owns one "
        "retained root reference. The completed result then replaces destination_r8.",
    ),
)


class _LaneDirection(enum.Enum):
    LOAD = "load"
    STORE = "store"


def _buffer_lane_access(direction: _LaneDirection) -> Instruction:
    is_load = direction == _LaneDirection.LOAD
    register_name = "destination_v8" if is_load else "source_v8"
    register = _value(
        register_name,
        FieldRole.RESULT if is_load else FieldRole.OPERAND,
        "First lane value-register ordinal.",
    )
    fields = (
        (
            register,
            _buffer_ref("buffer_r8"),
            _value("base_v8", FieldRole.OPERAND, "Unsigned 64-bit address base."),
            _value("index_v8", FieldRole.OPERAND, "Unsigned 64-bit scaled index."),
            _field(
                "scale_u8",
                U8,
                "Unsigned scale including zero.",
                FieldRole.IMMEDIATE,
                FieldRule.ANY_BITS,
            ),
            _memory_format(),
            _padding("zero_padding_u8"),
        )
        if is_load
        else (
            _buffer_ref("buffer_r8"),
            _value("base_v8", FieldRole.OPERAND, "Unsigned 64-bit address base."),
            _value("index_v8", FieldRole.OPERAND, "Unsigned 64-bit scaled index."),
            _field(
                "scale_u8",
                U8,
                "Unsigned scale including zero.",
                FieldRole.IMMEDIATE,
                FieldRule.ANY_BITS,
            ),
            register,
            _memory_format(),
            _padding("zero_padding_u8"),
        )
    )
    access = "READ" if is_load else "WRITE"
    mutation = (
        "Every destination value register remains unchanged."
        if is_load
        else "Every buffer byte remains unchanged."
    )
    transfer = (
        "load_lanes(destination_v8, buffer.data + offset_u64, format);"
        if is_load
        else "store_lanes(buffer.data + offset_u64, source_v8, format);"
    )
    return Instruction(
        opcode=0xDB if is_load else 0xDC,
        mnemonic=f"buffer.{direction.value}",
        since=CORE_0,
        family=BUFFER_FAMILY,
        summary=f"{'Loads from' if is_load else 'Stores to'} a checked buffer lane group.",
        fields=fields,
        semantics=None,
        behavior=(
            "Transfers a memory.format lane group at unsigned(base_v8) + "
            "unsigned(index_v8) * scale_u8 using checked u64 arithmetic and "
            "alignment-independent little-endian accesses."
        ),
        success=(
            (
                "Base and index are snapshotted before publication, so a "
                "destination lane may alias either address register. Each loaded "
                "lane is zero-extended to a complete value cell."
            )
            if is_load
            else (
                "Each lane stores the selected low source bits after every address "
                "and access check succeeds."
            ),
        ),
        assembly=(
            (
                "%v<destination>.xN = buffer.load %r<buffer>, %v<base>, "
                "%v<index> {scale, format}"
            )
            if is_load
            else (
                "buffer.store %r<buffer>, %v<base>, %v<index>, "
                "%v<source>.xN {scale, format}"
            )
        ),
        pseudocode=(
            "base_u64 = values[base_v8];\n"
            "index_u64 = values[index_v8];\n"
            "format = verified_memory_format(format_u8);\n"
            "buffer = check_deref(refs[buffer_r8], vm_buffer_type);\n"
            "offset_u64 = checked_buffer_offset(base_u64, index_u64, scale_u8);\n"
            "require checked_range(\n"
            "    offset_u64, format.access_length, buffer.length);\n"
            f"check_buffer_access(buffer, {access});\n"
            f"{transfer}\n"
            "pc = pc + 8;"
        ),
        rules=(_value_format_range(register_name),),
        state_effects=(
            StateEffect(
                StateAccess.READ if is_load else StateAccess.WRITE,
                StateResource.BUFFER,
                ("buffer_r8",),
            ),
        ),
        preconditions=(
            f"buffer_r8 must contain an exact non-null open vm.buffer with {access} "
            "access; scaled address arithmetic must not overflow and the complete "
            "access must fit buffer.length.",
        ),
        failures=_required_buffer_failures(
            mutation,
            access=access,
            range_description=(
                "Scaled address arithmetic overflows u64 or the complete lane "
                "range is outside buffer.length."
            ),
        ),
        ownership=("buffer_r8 is borrowed without a ref-count change.",),
    )


BUFFER_LOAD = _buffer_lane_access(_LaneDirection.LOAD)
BUFFER_STORE = _buffer_lane_access(_LaneDirection.STORE)


def _packed_component(
    name: str,
    bit_offset: int,
    bit_length: int,
    table: NumericTable,
    allowed_values: tuple[int, ...] = (),
) -> PackedSelectorComponent:
    values = allowed_values or tuple(value.value for value in table.values)
    valid = is_name(name) and table.kind == NumericKind.SELECTOR
    valid &= 0 <= bit_offset and 0 < bit_length <= 4 and bit_offset + bit_length <= 8
    valid &= set(values) <= {value.value for value in table.values}
    valid &= all(0 <= value < 1 << bit_length for value in values)
    require(valid, f"{name}: malformed packed selector component")
    return PackedSelectorComponent(name, bit_offset, bit_length, table, allowed_values)


def _packed_selector(
    name: str,
    components: tuple[PackedSelectorComponent, ...],
) -> InstructionField:
    component_mask = 0
    for component in components:
        mask = ((1 << component.bit_length) - 1) << component.bit_offset
        require(not component_mask & mask, f"{name}: overlapping selector components")
        component_mask |= mask
    return _field(
        name,
        U8,
        "Packed closed-selector components and canonical zero bits.",
        FieldRole.IMMEDIATE,
        FieldRuleUse(
            FieldRule.PACKED_SELECTORS,
            values=(0xFF ^ component_mask,),
            data=components,
        ),
    )


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


class _AtomicApplyForm(enum.Enum):
    REDUCE = "reduce"
    RMW = "rmw"


def _atomic_apply_selectors(
    form: _AtomicApplyForm,
) -> tuple[tuple[InstructionField, ...], PackedSelectorComponent]:
    is_reduce = form == _AtomicApplyForm.REDUCE
    kind = _packed_component(
        "kind",
        0,
        4,
        BUFFER_ATOMIC_KIND_SELECTOR,
        _ATOMIC_REDUCE_KINDS if is_reduce else (),
    )
    carrier = _packed_component("carrier", 7, 1, BUFFER_ATOMIC_CARRIER_SELECTOR)
    return (
        (
            _packed_selector(
                "selector0_u8",
                (kind, carrier),
            ),
            _packed_selector(
                "selector1_u8",
                (
                    _packed_component(
                        "ordering", 0, 3, BUFFER_ATOMIC_ORDERING_SELECTOR
                    ),
                    _packed_component("scope", 3, 3, BUFFER_ATOMIC_SCOPE_SELECTOR),
                ),
            ),
        ),
        carrier,
    )


def _atomic_carrier_rule(component: PackedSelectorComponent) -> RecordRule:
    return RecordRule(
        RecordRuleKind.ATOMIC_CARRIER_REQUIREMENT,
        fields=("selector0_u8",),
        data=(component,),
        summary=(
            "The carrier component of selector0_u8 contributes one static "
            "requirement checked against the selected target contract."
        ),
    )


def _atomic_failures(mutation_boundary: str) -> tuple[FailureCase, ...]:
    return _required_buffer_failures(
        mutation_boundary,
        access="READ|WRITE",
        range_description="The complete carrier range is outside buffer.length.",
        require_alignment=True,
    )


def _atomic_apply(form: _AtomicApplyForm) -> Instruction:
    is_reduce = form == _AtomicApplyForm.REDUCE
    selectors, carrier_component = _atomic_apply_selectors(form)
    fields = (
        (
            _buffer_ref("buffer_r8"),
            _value(
                "offset_v8",
                FieldRole.OPERAND,
                "Unsigned carrier byte offset.",
            ),
            _value(
                "operand_v8",
                FieldRole.OPERAND,
                "Low carrier-width update operand.",
            ),
            *selectors,
            _padding("zero_padding_u16", U16),
        )
        if is_reduce
        else (
            _value(
                "old_v8",
                FieldRole.RESULT,
                "Carrier value immediately preceding the committed update.",
            ),
            _buffer_ref("buffer_r8"),
            _value(
                "offset_v8",
                FieldRole.OPERAND,
                "Unsigned carrier byte offset.",
            ),
            _value(
                "operand_v8",
                FieldRole.OPERAND,
                "Low carrier-width update operand.",
            ),
            *selectors,
            _padding("zero_padding_u8"),
        )
    )
    mutation = (
        "No buffer byte or VM register changes."
        if is_reduce
        else "No buffer byte changes and old_v8 remains unchanged."
    )
    result = (
        "The selected update commits atomically without publishing its old value."
        if is_reduce
        else (
            "old_v8 receives the carrier value immediately before the committed "
            "update; i32 clears the high 32 cell bits and i64 fills the cell."
        )
    )
    return Instruction(
        opcode=0xDD if is_reduce else 0xDE,
        mnemonic=f"buffer.atomic.{form.value}",
        since=CORE_0,
        family=BUFFER_FAMILY,
        summary=(
            "Atomically reduces one raw buffer carrier."
            if is_reduce
            else "Atomically updates and returns one raw buffer carrier."
        ),
        fields=fields,
        semantics=None,
        behavior=(
            "Atomically applies the selected integer or floating update to one "
            "i32/i64 raw-storage carrier with the selected ordering and scope. "
            "Reduce excludes exchange kinds; RMW accepts all kinds."
        ),
        success=(result,),
        assembly=(
            (
                "buffer.atomic.reduce %r<buffer>, %v<offset>, %v<operand> "
                "{carrier, kind, ordering, scope}"
            )
            if is_reduce
            else (
                "%v<old> = buffer.atomic.rmw %r<buffer>, %v<offset>, "
                "%v<operand> {carrier, kind, ordering, scope}"
            )
        ),
        pseudocode=(
            "offset_u64 = values[offset_v8];\n"
            "operand_bits = low_carrier_bits(values[operand_v8]);\n"
            "address = check_atomic_buffer_address(\n"
            "    refs[buffer_r8], offset_u64, carrier);\n"
            + (
                "raw_atomic_apply(address, operand_bits,\n"
                "                 kind, ordering, scope);\n"
                if is_reduce
                else (
                    "old_bits = raw_atomic_apply(\n"
                    "    address, operand_bits, kind, ordering, scope);\n"
                    "values[old_v8] = canonicalize_carrier(old_bits, carrier);\n"
                )
            )
            + "pc = pc + 8;"
        ),
        rules=(_atomic_carrier_rule(carrier_component),),
        state_effects=(
            StateEffect(StateAccess.READ, StateResource.BUFFER, ("buffer_r8",)),
            StateEffect(StateAccess.WRITE, StateResource.BUFFER, ("buffer_r8",)),
        ),
        preconditions=(
            "buffer_r8 must be a non-null exact open vm.buffer with READ|WRITE; "
            "the complete carrier must fit and its concrete host address must be "
            "naturally aligned.",
        ),
        failures=_atomic_failures(mutation),
        ownership=("buffer_r8 is borrowed without a ref-count change.",),
    )


BUFFER_ATOMIC_REDUCE = _atomic_apply(_AtomicApplyForm.REDUCE)
BUFFER_ATOMIC_RMW = _atomic_apply(_AtomicApplyForm.RMW)


_CMPXCHG_SUCCESS_ORDERING = _packed_component(
    "success_ordering", 0, 3, BUFFER_ATOMIC_ORDERING_SELECTOR
)
_CMPXCHG_FAILURE_ORDERING = _packed_component(
    "failure_ordering", 3, 3, BUFFER_ATOMIC_ORDERING_SELECTOR
)
_CMPXCHG_CARRIER = _packed_component("carrier", 7, 1, BUFFER_ATOMIC_CARRIER_SELECTOR)


BUFFER_ATOMIC_CMPXCHG = Instruction(
    opcode=0xDF,
    mnemonic="buffer.atomic.cmpxchg",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Performs strong exact-bit atomic compare-exchange.",
    fields=(
        _value(
            "old_v8",
            FieldRole.RESULT,
            "Actual carrier value observed by the comparison attempt.",
        ),
        _buffer_ref("buffer_r8"),
        _value(
            "offset_v8",
            FieldRole.OPERAND,
            "Unsigned carrier byte offset.",
        ),
        _value(
            "expected_v8",
            FieldRole.OPERAND,
            "Low carrier-width expected bit pattern.",
        ),
        _value(
            "replacement_v8",
            FieldRole.OPERAND,
            "Low carrier-width replacement bit pattern.",
        ),
        _packed_selector(
            "selector0_u8",
            (
                _CMPXCHG_SUCCESS_ORDERING,
                _CMPXCHG_FAILURE_ORDERING,
                _CMPXCHG_CARRIER,
            ),
        ),
        _packed_selector(
            "selector1_u8",
            (_packed_component("scope", 0, 3, BUFFER_ATOMIC_SCOPE_SELECTOR),),
        ),
    ),
    semantics=None,
    behavior=(
        "Performs one strong, non-spurious exact-bit compare-exchange on a raw "
        "i32/i64 carrier and returns the actual value observed whether the "
        "comparison succeeds or fails."
    ),
    success=(
        "All value inputs are snapshotted before old_v8 publication. The comparison "
        "and conditional replacement use exact low carrier bits, and old_v8 receives "
        "the observed carrier in canonical cell form.",
    ),
    assembly=(
        "%v<old> = buffer.atomic.cmpxchg %r<buffer>, %v<offset>, "
        "%v<expected>, %v<replacement> "
        "{carrier, success_ordering, failure_ordering, scope}"
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
    rules=(
        _atomic_carrier_rule(_CMPXCHG_CARRIER),
        RecordRule(
            RecordRuleKind.PACKED_SELECTOR_PAIRS,
            fields=("selector0_u8",),
            values=tuple(value for pair in _ORDERING_PAIRS for value in pair),
            data=(_CMPXCHG_SUCCESS_ORDERING, _CMPXCHG_FAILURE_ORDERING),
            summary=(
                "The success and failure ordering components must name one of the "
                "nine legal pairs; failure cannot be release/acq_rel or stronger "
                "than success."
            ),
        ),
    ),
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.BUFFER, ("buffer_r8",)),
        StateEffect(StateAccess.WRITE, StateResource.BUFFER, ("buffer_r8",)),
    ),
    preconditions=(
        "buffer_r8 must be a non-null exact open vm.buffer with READ|WRITE; the "
        "complete carrier must fit and its concrete host address must be naturally "
        "aligned.",
    ),
    failures=_atomic_failures("No buffer byte changes and old_v8 remains unchanged."),
    ownership=("buffer_r8 is borrowed without a ref-count change.",),
)

BUFFER_FILL = Instruction(
    opcode=0xE0,
    mnemonic="buffer.fill",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Repeats a byte pattern across a checked writable range.",
    fields=(
        _buffer_ref("buffer_r8"),
        _value("offset_v8", FieldRole.OPERAND, "Unsigned byte offset."),
        _value("length_v8", FieldRole.OPERAND, "Unsigned byte length."),
        _value("pattern_v8", FieldRole.OPERAND, "Low-byte repeating pattern."),
        _field(
            "pattern_width_u8",
            U8,
            "Pattern width in bytes.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.ALLOWED_VALUES, values=(1, 2, 4, 8)),
        ),
        _padding("zero_padding_u16", U16),
    ),
    semantics=None,
    behavior=(
        "Repeats the low one, two, four, or eight little-endian bytes of pattern_v8 "
        "across the exact writable range; the final repetition may be partial."
    ),
    success=(
        "The complete pattern and range are snapshotted and checked before exactly "
        "length_v8 bytes are written. Zero length forms no pointer.",
    ),
    assembly=(
        "buffer.fill %r<buffer>, %v<offset>, %v<length>, %v<pattern> {pattern_width}"
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
    state_effects=(
        StateEffect(StateAccess.WRITE, StateResource.BUFFER, ("buffer_r8",)),
    ),
    preconditions=(
        "buffer_r8 must be an exact non-null open vm.buffer with WRITE, and "
        "offset_v8 plus length_v8 must fit its byte length.",
    ),
    failures=_required_buffer_failures(
        "Every target byte remains unchanged.",
        access="WRITE",
        range_description="The requested target range is out of bounds.",
    ),
    ownership=("buffer_r8 is borrowed without a ref-count change.",),
)

BUFFER_COPY = Instruction(
    opcode=0xE1,
    mnemonic="buffer.copy",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Moves one checked buffer range to another.",
    fields=(
        _buffer_ref("target_r8"),
        _value("target_offset_v8", FieldRole.OPERAND, "Unsigned target byte offset."),
        _buffer_ref("source_r8"),
        _value("source_offset_v8", FieldRole.OPERAND, "Unsigned source byte offset."),
        _value("length_v8", FieldRole.OPERAND, "Unsigned byte length."),
        _padding("zero_padding_u16", U16),
    ),
    semantics=None,
    behavior=(
        "Moves one checked source range to one checked target range with memmove "
        "semantics across identical objects, flattened views, and distinct wrappers "
        "over overlapping external storage."
    ),
    success=(
        "Every check finishes before mutation. The target receives the source bytes "
        "as if through a temporary copy; zero length forms no pointer.",
    ),
    assembly=(
        "buffer.copy %r<target>, %v<target_offset>, %r<source>, "
        "%v<source_offset>, %v<length>"
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
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.BUFFER, ("source_r8",)),
        StateEffect(StateAccess.WRITE, StateResource.BUFFER, ("target_r8",)),
    ),
    preconditions=(
        "Both refs must be exact non-null open vm.buffers; source requires READ, "
        "target requires WRITE, and both complete ranges must fit.",
    ),
    failures=(
        *_required_buffer_failures(
            "Every target byte remains unchanged.",
            range_description=(
                "Either complete source or target range is out of bounds."
            ),
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
)

BUFFER_COMPARE = Instruction(
    opcode=0xE2,
    mnemonic="buffer.compare",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Lexicographically compares two checked buffer ranges.",
    fields=(
        _value(
            "destination_v8",
            FieldRole.RESULT,
            "Canonical signed i32 comparison result.",
        ),
        _buffer_ref("left_r8"),
        _value("left_offset_v8", FieldRole.OPERAND, "Unsigned left byte offset."),
        _buffer_ref("right_r8"),
        _value("right_offset_v8", FieldRole.OPERAND, "Unsigned right byte offset."),
        _value("length_v8", FieldRole.OPERAND, "Unsigned byte length."),
        _padding("zero_padding_u8"),
    ),
    semantics=None,
    behavior=(
        "Lexicographically compares equal-length checked ranges as unsigned bytes "
        "and returns canonical signed i32 -1, 0, or +1."
    ),
    success=(
        "All value operands and checks precede publication. destination_v8 receives "
        "UINT32_MAX, zero, or one with high 32 cell bits zero; empty ranges compare "
        "equal.",
    ),
    assembly=(
        "%v<destination> = buffer.compare %r<left>, %v<left_offset>, "
        "%r<right>, %v<right_offset>, %v<length>"
    ),
    pseudocode=(
        "left = check_deref(refs[left_r8], vm_buffer_type);\n"
        "right = check_deref(refs[right_r8], vm_buffer_type);\n"
        "left_offset_u64 = values[left_offset_v8];\n"
        "right_offset_u64 = values[right_offset_v8];\n"
        "length_u64 = values[length_v8];\n"
        "require checked_range(left_offset_u64, length_u64, left.length);\n"
        "require checked_range(right_offset_u64, length_u64, right.length);\n"
        "check_buffer_access(left, READ);\n"
        "check_buffer_access(right, READ);\n"
        "ordering = length_u64 == 0 ? 0 : compare_unsigned_bytes(\n"
        "    left.data + left_offset_u64, right.data + right_offset_u64,\n"
        "    length_u64);\n"
        "values[destination_v8] = ordering < 0 ? UINT32_MAX :\n"
        "                         ordering > 0 ? 1 : 0;\n"
        "pc = pc + 8;"
    ),
    state_effects=(
        StateEffect(StateAccess.READ, StateResource.BUFFER, ("left_r8",)),
        StateEffect(StateAccess.READ, StateResource.BUFFER, ("right_r8",)),
    ),
    preconditions=(
        "Both refs must be exact non-null open vm.buffers granting READ, and both "
        "complete ranges must fit.",
    ),
    failures=(
        *_required_buffer_failures(
            "destination_v8 remains unchanged.",
            range_description="Either complete comparison range is out of bounds.",
        ),
        FailureCase(
            "failed_precondition",
            "Either buffer root is closed.",
            "destination_v8 remains unchanged.",
        ),
        FailureCase(
            "permission_denied",
            "Either buffer lacks READ access.",
            "destination_v8 remains unchanged.",
        ),
    ),
    ownership=("Both buffer refs are borrowed without ref-count changes.",),
)

BUFFER_COPY_RODATA = Instruction(
    opcode=0xE3,
    mnemonic="buffer.copy.rodata",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Copies module rodata into a checked writable buffer range.",
    fields=(
        _buffer_ref("target_r8"),
        _value("target_offset_v8", FieldRole.OPERAND, "Unsigned target byte offset."),
        _padding("zero_padding0_u8"),
        _field(
            "rodata_u16",
            U16,
            "Direct module rodata-block ordinal.",
            FieldRole.IMMEDIATE,
            FieldRule.RODATA_ORDINAL,
        ),
        _value("length_v8", FieldRole.OPERAND, "Unsigned byte length."),
        _padding("zero_padding1_u8"),
        _field(
            "source_offset_u32",
            U32,
            "Statically in-range or one-past-end rodata offset.",
            FieldRole.IMMEDIATE,
            FieldRuleUse(FieldRule.RODATA_OFFSET, fields=("rodata_u16",)),
        ),
    ),
    semantics=None,
    behavior=(
        "Copies a dynamically sized immutable module-rodata range into a "
        "dynamically offset writable buffer range."
    ),
    success=(
        "Every check precedes mutation. The immutable image and writable buffer "
        "cannot alias; zero length forms no pointer.",
    ),
    assembly=(
        "buffer.copy.rodata %r<target>, %v<target_offset>, "
        "@rodata<ordinal>+<source_offset>, %v<length>"
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
    state_effects=(
        StateEffect(StateAccess.WRITE, StateResource.BUFFER, ("target_r8",)),
    ),
    preconditions=(
        "target_r8 must be an exact non-null open vm.buffer with WRITE; both "
        "dynamic source and target ranges must fit.",
    ),
    failures=_required_buffer_failures(
        "Every target byte remains unchanged.",
        access="WRITE",
        range_description="The dynamic source or target range is out of bounds.",
    ),
    ownership=("target_r8 is borrowed without a ref-count change.",),
)

BUFFER_RODATA_LOAD = Instruction(
    opcode=0xE4,
    mnemonic="buffer.rodata.load",
    since=CORE_0,
    family=BUFFER_FAMILY,
    summary="Borrows a module-owned read-only rodata buffer view.",
    fields=(
        _buffer_result("destination_r8", RefOwnership.REPLACE_BORROW),
        _field(
            "rodata_u16",
            U16,
            "Direct module rodata-block ordinal.",
            FieldRole.IMMEDIATE,
            FieldRule.RODATA_ORDINAL,
        ),
    ),
    semantics=None,
    behavior=(
        "Publishes an internal borrow of the complete immutable READ-only vm.buffer "
        "view prebuilt and owned by the loaded module."
    ),
    success=(
        "destination_r8 releases its prior state and receives the canonical "
        "borrowed view without an atomic retain.",
    ),
    assembly="%r<destination> = buffer.rodata.load @rodata<ordinal>",
    pseudocode=(
        "view = loaded_module.rodata_views[rodata_u16];\n"
        "old_destination = refs[destination_r8];\n"
        "refs[destination_r8] = borrowed_ref(view, vm_buffer_type);\n"
        "release_or_clear(old_destination);\n"
        "pc = pc + 4;"
    ),
    ownership=(
        "The loaded module holds the view's initial owner; the view independently "
        "retains image storage. Borrow promotion creates an ordinary owner, so an "
        "escaped view outlives module teardown without forming an ownership cycle.",
    ),
)

BUFFER_INSTRUCTIONS = (
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
