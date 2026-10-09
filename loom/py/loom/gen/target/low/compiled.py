# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiled low descriptor intermediate data model."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

from loom.gen.support.string_pool import CStringPool
from loom.gen.target.low.validation import validate_u16_table_count
from loom.target.low_descriptors import (
    AsmOperandSegmentDelimiter,
    AsmResultValueType,
    Constraint,
    Descriptor,
    DescriptorSet,
    Effect,
    EncodingFieldValue,
    EnumDomain,
    EnumValue,
    EventSeparation,
    Hazard,
    Immediate,
    ImmediateEncodingSlice,
    InstructionClass,
    IssueUse,
    NativeAsmValueKind,
    Operand,
    OperandFormImmediateAction,
    OperandFormMatchKind,
    PhysicalRegister,
    PressureDelta,
    RegClass,
    RegClassAltFlag,
    RegisterPackingResource,
    RegisterPackingResourceFlag,
    RegisterPart,
    Resource,
    ScheduleClass,
    StorageLease,
    TimingEvent,
)


@dataclass(frozen=True, slots=True)
class CompiledPhysicalRegisterView:
    physical_register_id: int
    reg_class_id: int
    unit_candidate_ordinal_start: int
    unit_count: int
    # Lowest unit position in the class's aggregate-preserving search order.
    packing_rank: int


@dataclass(frozen=True, slots=True)
class CompiledRegisterPackingResource:
    # Authored resource capacity and member contributions.
    source: RegisterPackingResource
    # Immutable facts derived from all member register classes.
    flags: tuple[RegisterPackingResourceFlag, ...]
    # First row in the packed register-class contribution table.
    member_start: int
    # Number of register-class contribution rows.
    member_count: int


@dataclass(frozen=True, slots=True)
class CompiledRegisterPackingResourceMember:
    reg_class_id: int
    register_unit_count: int
    resource_unit_count: int


@dataclass(frozen=True, slots=True)
class CompiledResourceCalendar:
    # First occupancy slot, shared by resources in one contention group.
    slot_start: int
    # Power-of-two ring length minus one.
    slot_mask: int
    # Required stage-zero units common to every schedule class in this group.
    minimum_issue_units: int


@dataclass(frozen=True, slots=True)
class DescriptorAllowlist:
    keys: tuple[str, ...] = ()
    semantic_tags: tuple[str, ...] = ()
    mnemonics: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not self.keys and not self.semantic_tags and not self.mnemonics


@dataclass(frozen=True, slots=True)
class GeneratedDescriptorSet:
    header: str
    source: str


@dataclass(frozen=True, slots=True)
class GeneratedDescriptorSetFamily:
    """C artifacts emitted from one compiled storage set and its views."""

    # Shared C source implementing every descriptor-set view.
    source: str
    # Public C headers paired positionally with the requested view specs.
    view_headers: tuple[str, ...]


@dataclass(slots=True)
class CompiledPhysicalRegisterCandidateLookup:
    # First row in the packed reverse candidate table.
    ordinal_start: int
    # Physical-register ID represented by the first row.
    register_base: int
    # Number of physical-register IDs covered by the range.
    register_count: int


@dataclass(slots=True)
class CompiledPhysicalRegisterViewLookup:
    # First row in the packed view ordinal table.
    ordinal_start: int
    # Register-class ID represented by the first row.
    class_base: int
    # Number of register-class IDs covered by the range.
    class_count: int


@dataclass(slots=True)
class CompiledDescriptorSet:
    spec: DescriptorSet
    # Selected, validated descriptors before compact runtime projections.
    source_descriptors: list[Descriptor]
    # Runtime projections paired positionally with `source_descriptors`.
    descriptors: list[Descriptor]
    instruction_classes: list[tuple[InstructionClass, ...]]
    reg_classes: list[RegClass]
    physical_registers: list[PhysicalRegister]
    physical_register_candidate_ids: list[int]
    # Class-local semantic ordinals in aggregate-preserving search order.
    physical_register_allocation_ordinals: list[int]
    physical_register_candidate_starts: list[int]
    # Semantic ordinals indexed by physical ID, with 0xFFFF for nonmembers.
    physical_register_candidate_ordinals: list[int]
    # Reverse candidate ranges paired positionally with register classes.
    physical_register_candidate_lookups: list[CompiledPhysicalRegisterCandidateLookup]
    physical_register_atomic_units: list[int]
    physical_register_atomic_unit_starts: list[int]
    physical_register_views: list[CompiledPhysicalRegisterView]
    # View ordinals indexed by class ID, with 0xFFFFFFFF for absent views.
    physical_register_view_ordinals: list[int]
    # Class-ID intervals paired positionally with physical registers.
    physical_register_view_lookups: list[CompiledPhysicalRegisterViewLookup]
    physical_register_view_unit_candidate_ordinals: list[int]
    register_packing_resources: list[CompiledRegisterPackingResource]
    register_packing_resource_members: list[CompiledRegisterPackingResourceMember]
    register_parts: list[RegisterPart]
    resources: list[Resource]
    # Ring layouts paired with resources; shared groups have identical layouts.
    resource_calendars: list[CompiledResourceCalendar]
    # Total occupancy slots across distinct resource calendars.
    resource_calendar_slot_count: int
    # Earlier issue distance retained before the native issue high-water mark.
    resource_calendar_lookback_cycles: int
    schedule_classes: list[ScheduleClass]
    timing_events: list[TimingEvent]
    event_separations: list[EventSeparation]
    enum_domains: list[EnumDomain]
    reg_class_ids: dict[str, int]
    register_part_ids: dict[str, int]
    resource_ids: dict[str, int]
    schedule_class_ids: dict[str, int]
    timing_event_ids: dict[str, int]
    enum_domain_ids: dict[str, int]
    string_pool: CStringPool
    reg_class_alts: list[tuple[int | None, int | None, tuple[RegClassAltFlag, ...], int, int]]
    # Descriptor-local sequences of indices into the shared operand row tables.
    operand_indices: list[int]
    operands: list[Operand]
    operand_source_value_indices: list[int | None]
    operand_alt_starts: list[int]
    operand_rematerializable: list[bool]
    immediates: list[Immediate]
    immediate_encoding_slices: list[ImmediateEncodingSlice]
    immediate_encoding_slice_starts: list[int]
    enum_values: list[EnumValue]
    immediate_enum_domain_ids: list[int | None]
    immediate_attribute_masks: list[int]
    effects: list[Effect]
    constraints: list[Constraint]
    storage_leases: list[StorageLease]
    storage_lease_labels: list[tuple[str, int]]
    issue_uses: list[IssueUse]
    hazards: list[Hazard]
    pressure_deltas: list[PressureDelta]
    feature_mask_words: list[int]
    encoding_field_values: list[EncodingFieldValue]
    operand_forms: list[CompiledOperandForm]
    operand_form_matches: list[CompiledOperandFormMatch]
    operand_form_operand_indices: list[int]
    descriptor_rows: list[dict[str, int]]
    descriptor_refs: list[tuple[str, int]]
    schedule_alternative_rows: list[tuple[int, int]]
    canonical_asm_form_ordinals: list[int | None]
    asm_forms: list[CompiledAsmForm]
    asm_table_storage: CompiledAsmTableStorage
    schedule_rows: list[dict[str, int]]
    enum_domain_rows: list[dict[str, int]]


@dataclass(frozen=True, slots=True)
class DescriptorSetView:
    spec: DescriptorSet
    # View-owned classes at shared storage indices; None marks an absent class.
    reg_classes: tuple[RegClass | None, ...]
    descriptors: tuple[Descriptor, ...]
    instruction_classes: tuple[tuple[InstructionClass, ...], ...]
    storage_descriptor_ordinals: tuple[int, ...]
    descriptor_ordinal_count: int
    descriptor_membership_words: tuple[int, ...]
    descriptor_refs: list[tuple[str, int]]
    schedule_alternative_rows: list[tuple[int, int]]
    descriptor_rows: list[dict[str, int]]
    canonical_asm_form_ordinals: list[int | None]
    asm_forms: list[CompiledAsmForm]
    operand_forms: list[CompiledOperandForm]
    # Structural descriptor rows use shared storage ordinals.
    uses_storage_descriptor_tables: bool
    # View-owned descriptor rows use shared storage ordinals.
    uses_storage_descriptor_view_tables: bool
    # Descriptor-key references use shared storage ordinals.
    uses_storage_descriptor_ref_tables: bool
    uses_storage_asm_form_tables: bool
    uses_storage_operand_form_tables: bool
    uses_storage_schedule_alternative_tables: bool

    @property
    def descriptor_count(self) -> int:
        return len(self.storage_descriptor_ordinals)

    @property
    def runtime_descriptor_ordinals(self) -> tuple[int, ...]:
        if self.uses_storage_descriptor_tables:
            return self.storage_descriptor_ordinals
        return tuple(range(self.descriptor_count))


@dataclass(frozen=True, slots=True)
class CompiledAsmImmediate:
    immediate_index: int
    name_label: str | None
    name: str | None


@dataclass(frozen=True, slots=True)
class CompiledNativeAsmValue:
    kind: NativeAsmValueKind
    index: int
    bit_width: int
    target_format_id: int
    literal_label: str | None
    literal: str | None


@dataclass(frozen=True, slots=True)
class CompiledAsmOperandSegment:
    delimiter: AsmOperandSegmentDelimiter
    operand_count: int
    has_variadic_operand: bool


@dataclass(slots=True)
class CompiledAsmForm:
    descriptor_ordinal: int
    mnemonic_label: str
    mnemonic: str
    native_assembly_mnemonic_label: str | None
    native_assembly_mnemonic: str | None
    result_indices: tuple[int, ...]
    operand_indices: tuple[int, ...]
    operand_segments: tuple[CompiledAsmOperandSegment, ...]
    result_value_types: tuple[AsmResultValueType | None, ...]
    # Complete immediate order: explicit assembly fields, then descriptor suffix.
    immediates: tuple[CompiledAsmImmediate, ...]
    # Number of authored fields at the start of the complete immediate order.
    explicit_immediate_count: int
    # An explicit alias or default selects named syntax for the entire form.
    has_named_immediates: bool
    # At least one field in the named dictionary has no descriptor default.
    requires_named_immediates: bool
    # Native immediate projections own spelling instead of generic suffixes.
    native_owns_immediate_syntax: bool
    native_assembly_values: tuple[CompiledNativeAsmValue, ...]
    # Index assigned when this form's formatting spans enter shared storage.
    layout_index: int = 0


@dataclass(frozen=True, slots=True)
class CompiledAsmLayout:
    """Exact emitted formatting spans, independent of assembly identity."""

    # First result operand index in the shared operand-index table.
    result_operand_index_start: int
    # First exact semantic result type, or None when no types are declared.
    result_value_type_start: int | None
    # First input operand index in the shared operand-index table.
    operand_index_start: int
    # First delimited input operand segment.
    operand_segment_start: int
    # First row of the complete immediate order, including implicit fields.
    immediate_start: int
    # First native assembly value row.
    native_assembly_value_start: int
    # Number of result operand indices.
    result_operand_index_count: int
    # Number of input operand indices.
    operand_index_count: int
    # Number of delimited input operand segments.
    operand_segment_count: int
    # Number of explicit spelling rows before the implicit descriptor suffix.
    explicit_immediate_count: int
    # Number of native assembly value rows.
    native_assembly_value_count: int
    # Explicit aliases or defaults require named syntax for the entire form.
    has_named_immediates: bool
    # The named dictionary cannot be omitted after any positional prefix.
    requires_named_immediates: bool
    # Native immediate projections suppress generic immediate suffixes.
    native_owns_immediate_syntax: bool


def append_interned_sequence[RowT: Hashable](
    sequence: Sequence[RowT],
    rows: list[RowT],
    starts: dict[tuple[RowT, ...], int],
) -> tuple[int, bool]:
    """Interns an exact non-empty row sequence and returns its start and novelty."""

    key = tuple(sequence)
    if not key:
        return 0, False
    start = starts.get(key)
    if start is not None:
        return start, False
    start = len(rows)
    starts[key] = start
    rows.extend(key)
    return start, True


@dataclass(slots=True)
class CompiledAsmTableStorage:
    """Interned assembly-form table rows shared across storage and views."""

    # Exact formatting layouts addressed by each form's layout index.
    layouts: list[CompiledAsmLayout] = field(default_factory=list)
    # Result and operand indices addressed by assembly-form spans.
    operand_indices: list[int] = field(default_factory=list)
    # Delimited operand groups addressed by assembly-form spans.
    operand_segments: list[CompiledAsmOperandSegment] = field(default_factory=list)
    # Exact semantic result types addressed by assembly-form spans.
    result_value_types: list[AsmResultValueType | None] = field(default_factory=list)
    # Immediate projections addressed by assembly-form spans.
    immediates: list[CompiledAsmImmediate] = field(default_factory=list)
    # Native assembly values addressed by assembly-form spans.
    native_values: list[CompiledNativeAsmValue] = field(default_factory=list)
    # Indices of formatting layouts already in storage.
    _layout_indices: dict[CompiledAsmLayout, int] = field(default_factory=dict, init=False, repr=False)
    # Starts of exact operand-index sequences already in storage.
    _operand_index_starts: dict[tuple[int, ...], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # Starts of exact operand-segment sequences already in storage.
    _operand_segment_starts: dict[tuple[CompiledAsmOperandSegment, ...], int] = field(default_factory=dict, init=False, repr=False)
    # Starts of exact result-type sequences already in storage.
    _result_value_type_starts: dict[tuple[AsmResultValueType | None, ...], int] = field(default_factory=dict, init=False, repr=False)
    # Starts of exact immediate sequences already in storage.
    _immediate_starts: dict[tuple[CompiledAsmImmediate, ...], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # Starts of exact native-value sequences already in storage.
    _native_value_starts: dict[tuple[CompiledNativeAsmValue, ...], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def append_forms(self, asm_forms: Sequence[CompiledAsmForm]) -> None:
        """Interns each form's exact table spans into this storage."""

        for asm_form in asm_forms:
            result_index_start, _ = append_interned_sequence(
                asm_form.result_indices,
                self.operand_indices,
                self._operand_index_starts,
            )
            result_value_type_start = None
            if asm_form.result_value_types:
                result_value_type_start, _ = append_interned_sequence(
                    asm_form.result_value_types,
                    self.result_value_types,
                    self._result_value_type_starts,
                )
            operand_index_start, _ = append_interned_sequence(
                asm_form.operand_indices,
                self.operand_indices,
                self._operand_index_starts,
            )
            operand_segment_start, _ = append_interned_sequence(
                asm_form.operand_segments,
                self.operand_segments,
                self._operand_segment_starts,
            )
            immediate_start, _ = append_interned_sequence(
                asm_form.immediates,
                self.immediates,
                self._immediate_starts,
            )
            native_assembly_value_start, _ = append_interned_sequence(
                asm_form.native_assembly_values,
                self.native_values,
                self._native_value_starts,
            )
            layout = CompiledAsmLayout(
                result_operand_index_start=result_index_start,
                result_value_type_start=result_value_type_start,
                operand_index_start=operand_index_start,
                operand_segment_start=operand_segment_start,
                immediate_start=immediate_start,
                native_assembly_value_start=native_assembly_value_start,
                result_operand_index_count=len(asm_form.result_indices),
                operand_index_count=len(asm_form.operand_indices),
                operand_segment_count=len(asm_form.operand_segments),
                explicit_immediate_count=asm_form.explicit_immediate_count,
                native_assembly_value_count=len(asm_form.native_assembly_values),
                has_named_immediates=asm_form.has_named_immediates,
                requires_named_immediates=asm_form.requires_named_immediates,
                native_owns_immediate_syntax=asm_form.native_owns_immediate_syntax,
            )
            layout_index = self._layout_indices.get(layout)
            if layout_index is None:
                layout_index = len(self.layouts)
                self._layout_indices[layout] = layout_index
                self.layouts.append(layout)
            asm_form.layout_index = layout_index

        # Views can append new spans after the storage descriptors are compiled.
        # Validate their shared compact address space at its owning boundary.
        for table_name, rows in (
            ("layout", self.layouts),
            ("operand index", self.operand_indices),
            ("operand segment", self.operand_segments),
            ("result value type", self.result_value_types),
            ("immediate", self.immediates),
            ("native value", self.native_values),
        ):
            validate_u16_table_count(len(rows), f"asm {table_name}")


@dataclass(frozen=True, slots=True)
class CompiledOperandFormMatch:
    source_operand_index: int
    source_packet_operand_index: int
    match_kind: OperandFormMatchKind
    match_i64: int


@dataclass(slots=True)
class CompiledOperandForm:
    replacement_descriptor_ordinal: int
    source_immediate_index: int
    replacement_immediate_index: int
    immediate_match_index: int
    immediate_action: OperandFormImmediateAction
    match_start: int
    match_count: int
    operand_map_start: int
    operand_map_count: int
