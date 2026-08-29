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
    RegisterPart,
    Resource,
    ScheduleClass,
    StorageLease,
    TimingEvent,
)


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
    physical_register_candidate_starts: list[int]
    physical_register_atomic_units: list[int]
    physical_register_atomic_unit_starts: list[int]
    register_parts: list[RegisterPart]
    resources: list[Resource]
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
    reg_class_alts: list[tuple[int | None, tuple[RegClassAltFlag, ...]]]
    operands: list[Operand]
    operand_source_value_indices: list[int | None]
    operand_alt_starts: list[int]
    operand_rematerializable: list[bool]
    immediates: list[Immediate]
    immediate_encoding_slices: list[ImmediateEncodingSlice]
    immediate_encoding_slice_starts: list[int]
    enum_values: list[EnumValue]
    immediate_enum_domain_ids: list[int | None]
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
    canonical_asm_form_ordinals: list[int | None]
    asm_forms: list[CompiledAsmForm]
    asm_table_storage: CompiledAsmTableStorage
    schedule_rows: list[dict[str, int]]
    enum_domain_rows: list[dict[str, int]]


@dataclass(frozen=True, slots=True)
class DescriptorSetView:
    spec: DescriptorSet
    descriptors: tuple[Descriptor, ...]
    instruction_classes: tuple[tuple[InstructionClass, ...], ...]
    descriptor_ordinals: tuple[int, ...]
    descriptor_refs: list[tuple[str, int]]
    descriptor_rows: list[dict[str, int]]
    canonical_asm_form_ordinals: list[int | None]
    asm_forms: list[CompiledAsmForm]
    operand_forms: list[CompiledOperandForm]
    # Structural descriptor rows are a prefix of the storage table.
    uses_storage_descriptor_tables: bool
    # View-owned descriptor rows are a prefix of the storage table.
    uses_storage_descriptor_view_tables: bool
    uses_storage_asm_form_tables: bool
    uses_storage_operand_form_tables: bool

    @property
    def descriptor_count(self) -> int:
        return len(self.descriptor_ordinals)


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
    immediates: tuple[CompiledAsmImmediate, ...]
    native_assembly_values: tuple[CompiledNativeAsmValue, ...]
    result_index_start: int = 0
    result_value_type_start: int | None = None
    operand_index_start: int = 0
    operand_segment_start: int = 0
    immediate_start: int = 0
    native_assembly_value_start: int = 0


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
            asm_form.result_index_start, _ = append_interned_sequence(
                asm_form.result_indices,
                self.operand_indices,
                self._operand_index_starts,
            )
            if asm_form.result_value_types:
                asm_form.result_value_type_start, _ = append_interned_sequence(
                    asm_form.result_value_types,
                    self.result_value_types,
                    self._result_value_type_starts,
                )
            asm_form.operand_index_start, _ = append_interned_sequence(
                asm_form.operand_indices,
                self.operand_indices,
                self._operand_index_starts,
            )
            asm_form.operand_segment_start, _ = append_interned_sequence(
                asm_form.operand_segments,
                self.operand_segments,
                self._operand_segment_starts,
            )
            asm_form.immediate_start, _ = append_interned_sequence(
                asm_form.immediates,
                self.immediates,
                self._immediate_starts,
            )
            asm_form.native_assembly_value_start, _ = append_interned_sequence(
                asm_form.native_assembly_values,
                self.native_values,
                self._native_value_starts,
            )


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
