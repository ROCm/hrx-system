# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiled low descriptor intermediate data model."""

from __future__ import annotations

from dataclasses import dataclass

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
    Hazard,
    Immediate,
    ImmediateEncodingSlice,
    InstructionClass,
    IssueUse,
    NativeAsmValueKind,
    Operand,
    OperandFormImmediateAction,
    OperandFormMatchKind,
    PressureDelta,
    RegClass,
    RegClassAltFlag,
    RegisterPart,
    Resource,
    ScheduleClass,
    StorageLease,
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
    descriptors: list[Descriptor]
    instruction_classes: list[tuple[InstructionClass, ...]]
    reg_classes: list[RegClass]
    register_parts: list[RegisterPart]
    resources: list[Resource]
    schedule_classes: list[ScheduleClass]
    enum_domains: list[EnumDomain]
    reg_class_ids: dict[str, int]
    register_part_ids: dict[str, int]
    resource_ids: dict[str, int]
    schedule_class_ids: dict[str, int]
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
    asm_operand_indices: list[int]
    asm_operand_segments: list[CompiledAsmOperandSegment]
    asm_result_value_types: list[AsmResultValueType | None]
    asm_immediates: list[CompiledAsmImmediate]
    native_asm_values: list[CompiledNativeAsmValue]
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
    uses_storage_descriptor_tables: bool
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
