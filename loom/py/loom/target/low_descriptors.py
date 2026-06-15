# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python source schema for target-low descriptor sets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loom.stable_id import stable_id_from_string

LOW_DESCRIPTOR_SET_ABI_VERSION = 26
LOW_DESCRIPTOR_ENCODING_ID_NONE = (2**16) - 1
LOW_DESCRIPTOR_SET_ORDINAL_NONE = (2**16) - 1


def descriptor_stable_id(key: str) -> int:
    """Returns the deterministic non-zero 63-bit identity for a descriptor key."""
    return stable_id_from_string(key)


def _validate_metadata_key(kind: str, key: str) -> None:
    if not key:
        raise ValueError(f"{kind} key must not be empty")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    if any(char not in allowed for char in key):
        raise ValueError(
            f"{kind} key {key!r} must contain only lowercase letters, "
            "digits, '.', '_', or '-'"
        )


class CEnum(Enum):
    @property
    def c_name(self) -> str:
        return str(self.value)


class OperandRole(CEnum):
    RESULT = "LOOM_LOW_OPERAND_ROLE_RESULT"
    OPERAND = "LOOM_LOW_OPERAND_ROLE_OPERAND"
    OPERAND_RESULT = "LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT"
    PREDICATE = "LOOM_LOW_OPERAND_ROLE_PREDICATE"
    RESOURCE = "LOOM_LOW_OPERAND_ROLE_RESOURCE"
    IMPLICIT = "LOOM_LOW_OPERAND_ROLE_IMPLICIT"


class OperandAddressMapKind(CEnum):
    DIRECT = "LOOM_LOW_OPERAND_ADDRESS_MAP_DIRECT"
    LOW_SUBSET = "LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET"
    TARGET_STATE = "LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE"


class OperandFlag(CEnum):
    IMPLICIT = "LOOM_LOW_OPERAND_FLAG_IMPLICIT"
    TIED = "LOOM_LOW_OPERAND_FLAG_TIED"
    EARLY_CLOBBER = "LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER"
    OPTIONAL = "LOOM_LOW_OPERAND_FLAG_OPTIONAL"
    STATE_READ = "LOOM_LOW_OPERAND_FLAG_STATE_READ"
    STATE_WRITE = "LOOM_LOW_OPERAND_FLAG_STATE_WRITE"
    SCHEDULE_ONLY_STATE = "LOOM_LOW_OPERAND_FLAG_SCHEDULE_ONLY_STATE"


class RegClassAltFlag(CEnum):
    PREFERRED = "LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED"
    IMMEDIATE = "LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE"
    PHYSICAL_ONLY = "LOOM_LOW_REG_CLASS_ALT_FLAG_PHYSICAL_ONLY"


class RegClassFlag(CEnum):
    VIRTUAL_ONLY = "LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY"
    PHYSICAL = "LOOM_LOW_REG_CLASS_FLAG_PHYSICAL"
    REFERENCE = "LOOM_LOW_REG_CLASS_FLAG_REFERENCE"
    UNSPILLABLE = "LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE"


class SpillSlotSpace(CEnum):
    STACK = "LOOM_LOW_SPILL_SLOT_SPACE_STACK"
    SCRATCH = "LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH"
    PRIVATE = "LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE"
    LDS = "LOOM_LOW_SPILL_SLOT_SPACE_LDS"


class ImmediateKind(CEnum):
    SIGNED = "LOOM_LOW_IMMEDIATE_KIND_SIGNED"
    UNSIGNED = "LOOM_LOW_IMMEDIATE_KIND_UNSIGNED"
    ORDINAL = "LOOM_LOW_IMMEDIATE_KIND_ORDINAL"
    ENUM = "LOOM_LOW_IMMEDIATE_KIND_ENUM"


class ImmediateFlag(CEnum):
    SYMBOLIC = "LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC"
    RELATIVE = "LOOM_LOW_IMMEDIATE_FLAG_RELATIVE"
    DEFAULT_VALUE = "LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE"


class EffectKind(CEnum):
    READ = "LOOM_LOW_EFFECT_KIND_READ"
    WRITE = "LOOM_LOW_EFFECT_KIND_WRITE"
    CALL = "LOOM_LOW_EFFECT_KIND_CALL"
    BARRIER = "LOOM_LOW_EFFECT_KIND_BARRIER"
    COUNTER = "LOOM_LOW_EFFECT_KIND_COUNTER"
    CONVERGENT = "LOOM_LOW_EFFECT_KIND_CONVERGENT"
    CONTROL = "LOOM_LOW_EFFECT_KIND_CONTROL"


class MemorySpace(CEnum):
    NONE = "LOOM_LOW_MEMORY_SPACE_NONE"
    GENERIC = "LOOM_LOW_MEMORY_SPACE_GENERIC"
    GLOBAL = "LOOM_LOW_MEMORY_SPACE_GLOBAL"
    WORKGROUP = "LOOM_LOW_MEMORY_SPACE_WORKGROUP"
    STACK = "LOOM_LOW_MEMORY_SPACE_STACK"
    VM_REF = "LOOM_LOW_MEMORY_SPACE_VM_REF"
    WASM_MEMORY = "LOOM_LOW_MEMORY_SPACE_WASM_MEMORY"


class EffectFlag(CEnum):
    ORDERED = "LOOM_LOW_EFFECT_FLAG_ORDERED"
    DEPENDENCY = "LOOM_LOW_EFFECT_FLAG_DEPENDENCY"


class StorageLeaseKind(CEnum):
    SOURCE_READ = "LOOM_LOW_STORAGE_LEASE_SOURCE_READ"
    RESULT_WRITE = "LOOM_LOW_STORAGE_LEASE_RESULT_WRITE"


class StorageLeaseAttachment(CEnum):
    OPERAND = "LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND"
    RESULT = "LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT"


class StorageLeaseReleaseScope(CEnum):
    PROGRESS_CLASS = "LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS"


class StorageLeaseFlag(CEnum):
    STARTS_AT_ISSUE = "LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE"
    RELEASE_BEFORE_BOUNDARY = "LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY"
    MAY_CARRY_ACROSS_BOUNDARY = "LOOM_LOW_STORAGE_LEASE_FLAG_MAY_CARRY_ACROSS_BOUNDARY"


class ConstraintKind(CEnum):
    TIED = "LOOM_LOW_CONSTRAINT_KIND_TIED"
    COMMUTABLE = "LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE"
    DESTRUCTIVE = "LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE"
    EARLY_CLOBBER = "LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER"
    REMATERIALIZABLE = "LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE"
    FOLDABLE = "LOOM_LOW_CONSTRAINT_KIND_FOLDABLE"


class LatencyKind(CEnum):
    EXACT = "LOOM_LOW_LATENCY_KIND_EXACT"
    ESTIMATE = "LOOM_LOW_LATENCY_KIND_ESTIMATE"
    VARIABLE = "LOOM_LOW_LATENCY_KIND_VARIABLE"


class ModelQuality(CEnum):
    EXACT = "LOOM_LOW_MODEL_QUALITY_EXACT"
    CALIBRATED = "LOOM_LOW_MODEL_QUALITY_CALIBRATED"
    ESTIMATED = "LOOM_LOW_MODEL_QUALITY_ESTIMATED"
    FALLBACK = "LOOM_LOW_MODEL_QUALITY_FALLBACK"


class ScheduleClassFlag(CEnum):
    MAY_LOAD = "LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD"
    MAY_STORE = "LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE"
    MAY_CALL = "LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL"
    CONTROL = "LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL"


class ResourceKind(CEnum):
    SCALAR_ALU = "LOOM_LOW_RESOURCE_KIND_SCALAR_ALU"
    VECTOR_ALU = "LOOM_LOW_RESOURCE_KIND_VECTOR_ALU"
    MATRIX = "LOOM_LOW_RESOURCE_KIND_MATRIX"
    LOAD = "LOOM_LOW_RESOURCE_KIND_LOAD"
    STORE = "LOOM_LOW_RESOURCE_KIND_STORE"
    CONTROL = "LOOM_LOW_RESOURCE_KIND_CONTROL"
    ADDRESS = "LOOM_LOW_RESOURCE_KIND_ADDRESS"


class HazardKind(CEnum):
    MIN_DISTANCE = "LOOM_LOW_HAZARD_KIND_MIN_DISTANCE"
    WAIT_COUNTER = "LOOM_LOW_HAZARD_KIND_WAIT_COUNTER"
    BYPASS = "LOOM_LOW_HAZARD_KIND_BYPASS"
    FUSION = "LOOM_LOW_HAZARD_KIND_FUSION"


class HazardReferenceKind(CEnum):
    RESOURCE = "LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE"
    COUNTER = "LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER"
    TARGET = "LOOM_LOW_HAZARD_REFERENCE_KIND_TARGET"


class DescriptorFlag(CEnum):
    SIDE_EFFECTING = "LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING"
    TERMINATOR = "LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR"
    DEAD_REMOVABLE = "LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE"
    PSEUDO = "LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO"


class DescriptorAsmSurface(Enum):
    """Low-asm product surface promised by a descriptor row.

    This is generator-only metadata. It classifies whether a descriptor is
    directly authorable in low asm, intentionally structural/generated-only, or
    otherwise outside the user-facing asm surface. The C descriptor table gets
    the resulting canonical asm form rows, not this policy layer.
    """

    AUTHORABLE = "authorable"
    STRUCTURAL = "structural"
    GENERATED_ONLY = "generated_only"


class OperandFormMatchKind(CEnum):
    ALL_EQUAL_I64 = "LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_I64"
    ALL_EQUAL_EXACT_I64 = "LOOM_LOW_OPERAND_FORM_MATCH_ALL_EQUAL_EXACT_I64"


class OperandFormImmediateAction(CEnum):
    NONE = "LOOM_LOW_OPERAND_FORM_IMMEDIATE_NONE"
    SET_MATCHED_I64 = "LOOM_LOW_OPERAND_FORM_IMMEDIATE_SET_MATCHED_I64"
    ADD_MATCHED_I64 = "LOOM_LOW_OPERAND_FORM_IMMEDIATE_ADD_MATCHED_I64"


class NativeAsmValueKind(CEnum):
    LITERAL = "LOOM_LOW_NATIVE_ASM_VALUE_KIND_LITERAL"
    RESULT = "LOOM_LOW_NATIVE_ASM_VALUE_KIND_RESULT"
    OPERAND = "LOOM_LOW_NATIVE_ASM_VALUE_KIND_OPERAND"
    IMMEDIATE_I64 = "LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_I64"
    IMMEDIATE_UNSIGNED_HEX = "LOOM_LOW_NATIVE_ASM_VALUE_KIND_IMMEDIATE_UNSIGNED_HEX"


@dataclass(frozen=True, slots=True)
class DescriptorCategory:
    """Stable category for grouping related descriptors inside a set.

    Categories are generator-facing metadata. They provide a semantic shard
    boundary for large target-owned descriptor sets without deriving file layout
    from incidental descriptor order.
    """

    key: str
    doc: str = ""

    def __post_init__(self) -> None:
        _validate_metadata_key("descriptor category", self.key)


@dataclass(frozen=True, slots=True)
class RegClass:
    name: str
    alloc_unit_bits: int
    spill_slot_space: SpillSlotSpace
    flags: tuple[RegClassFlag, ...] = ()
    target_bank_id: int = 0
    allocatable_count: int = 0
    alias_set_id: int = 0
    spill_class: str | None = None
    full_register_part_mask: int = 1


@dataclass(frozen=True, slots=True)
class RegisterPart:
    name: str
    reg_class: str
    mask: int


@dataclass(frozen=True, slots=True)
class RegClassAlt:
    reg_class: str | None
    flags: tuple[RegClassAltFlag, ...] = (RegClassAltFlag.PREFERRED,)


@dataclass(frozen=True, slots=True)
class Operand:
    field_name: str
    role: OperandRole
    reg_alts: tuple[RegClassAlt, ...]
    flags: tuple[OperandFlag, ...] = ()
    unit_count: int = 1
    address_map_kind: OperandAddressMapKind = OperandAddressMapKind.DIRECT
    addressable_unit_count: int = 0
    encoding_field_id: int = 0
    data_format_id: int = 0
    register_part: str | None = None
    read_stage: int = 0
    ready_stage: int = 0


@dataclass(frozen=True, slots=True)
class ImmediateEncodingSlice:
    encoding_field_id: int
    source_bit_offset: int
    bit_count: int


@dataclass(frozen=True, slots=True)
class Immediate:
    field_name: str
    kind: ImmediateKind
    flags: tuple[ImmediateFlag, ...] = ()
    bit_width: int = 0
    encoding_field_id: int = 0
    encoding_slices: tuple[ImmediateEncodingSlice, ...] = ()
    enum_domain: str | None = None
    encoding_id: int = 0
    signed_min: int = 0
    unsigned_max: int = 0
    default_value: int = 0


@dataclass(frozen=True, slots=True)
class EncodingFieldValue:
    encoding_field_id: int
    value: int


@dataclass(frozen=True, slots=True)
class AsmImmediate:
    field_name: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class NativeAsmValue:
    kind: NativeAsmValueKind
    field_name: str | None = None
    literal: str | None = None
    bit_width: int = 0


@dataclass(frozen=True, slots=True)
class AsmForm:
    mnemonic: str | None = None
    native_assembly_mnemonic: str | None = None
    results: tuple[str, ...] = ()
    operands: tuple[str, ...] = ()
    immediates: tuple[AsmImmediate, ...] = ()
    native_assembly_values: tuple[NativeAsmValue, ...] = ()


@dataclass(frozen=True, slots=True)
class EnumValue:
    token: str
    value: int


@dataclass(frozen=True, slots=True)
class EnumDomain:
    name: str
    values: tuple[EnumValue, ...]


@dataclass(frozen=True, slots=True)
class Effect:
    kind: EffectKind
    memory_space: MemorySpace = MemorySpace.NONE
    scope_id: int = 0
    flags: tuple[EffectFlag, ...] = ()
    counter_id: int = 0
    width_bits: int = 0


@dataclass(frozen=True, slots=True)
class StorageLease:
    kind: StorageLeaseKind
    attachment: StorageLeaseAttachment
    attachment_index: int
    unit_offset: int
    unit_count: int
    release_scope: StorageLeaseReleaseScope
    release_class_id: int
    release_class_name: str
    release_action_id: int
    release_action_name: str
    release_reason_id: int
    release_reason_name: str
    flags: tuple[StorageLeaseFlag, ...] = ()


@dataclass(frozen=True, slots=True)
class Constraint:
    kind: ConstraintKind
    lhs_operand_index: int
    rhs_operand_index: int | None = None
    flags: tuple[CEnum, ...] = ()


@dataclass(frozen=True, slots=True)
class OperandFormMatch:
    source_operand: str
    match_kind: OperandFormMatchKind
    match_i64: int = 0


@dataclass(frozen=True, slots=True)
class OperandForm:
    replacement_descriptor: str
    matches: tuple[OperandFormMatch, ...]
    immediate_action: OperandFormImmediateAction = OperandFormImmediateAction.NONE
    immediate_field: str | None = None
    immediate_source_operand: str | None = None

    def __post_init__(self) -> None:
        _validate_metadata_key("replacement descriptor", self.replacement_descriptor)
        if not self.matches:
            raise ValueError("operand form must have at least one match predicate")
        match_source_operands = {match.source_operand for match in self.matches}
        if len(match_source_operands) != len(self.matches):
            raise ValueError("operand form match predicates must use unique operands")
        if self.immediate_field is not None:
            _validate_metadata_key("immediate field", self.immediate_field)
        if self.immediate_action is OperandFormImmediateAction.NONE:
            if (
                self.immediate_field is not None
                or self.immediate_source_operand is not None
            ):
                raise ValueError(
                    "operand form without an immediate action must not name "
                    "immediate fields or sources"
                )
        elif self.immediate_field is None or self.immediate_source_operand is None:
            raise ValueError(
                "operand form immediate action must name an immediate field and source"
            )
        elif self.immediate_source_operand not in match_source_operands:
            raise ValueError(
                "operand form immediate source must name one matched source operand"
            )


@dataclass(frozen=True, slots=True)
class IssueUse:
    resource: str
    cycles: int
    units: int
    stage: int = 0


@dataclass(frozen=True, slots=True)
class PressureDelta:
    reg_class: str
    delta: int


@dataclass(frozen=True, slots=True)
class Resource:
    name: str
    capacity_per_cycle: int
    kind: ResourceKind
    flags: tuple[CEnum, ...] = ()
    contention_group_id: int = 0


@dataclass(frozen=True, slots=True)
class Hazard:
    kind: HazardKind
    resource: str | None = None
    counter_id: int | None = None
    target_id: int | None = None
    producer_stage: int = 0
    consumer_stage: int = 0
    distance: int = 0
    flags: tuple[CEnum, ...] = ()


@dataclass(frozen=True, slots=True)
class ScheduleClass:
    name: str
    latency_kind: LatencyKind
    model_quality: ModelQuality
    latency_cycles: int = 0
    issue_uses: tuple[IssueUse, ...] = ()
    hazards: tuple[Hazard, ...] = ()
    flags: tuple[ScheduleClassFlag, ...] = ()
    pressure_deltas: tuple[PressureDelta, ...] = ()


@dataclass(frozen=True, slots=True)
class Descriptor:
    key: str
    mnemonic: str | None
    semantic_tag: str | None
    operands: tuple[Operand, ...]
    schedule_class: str
    immediates: tuple[Immediate, ...] = ()
    encoding_field_values: tuple[EncodingFieldValue, ...] = ()
    asm_forms: tuple[AsmForm, ...] = ()
    asm_surface: DescriptorAsmSurface = DescriptorAsmSurface.AUTHORABLE
    asm_surface_reason: str = ""
    effects: tuple[Effect, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    storage_leases: tuple[StorageLease, ...] = ()
    feature_mask_words: tuple[int, ...] = ()
    encoding_format_id: int = 0
    encoding_id: int = 0
    flags: tuple[DescriptorFlag, ...] = ()
    operand_forms: tuple[OperandForm, ...] = ()
    category: DescriptorCategory | None = None


@dataclass(frozen=True, slots=True)
class DescriptorSet:
    key: str
    target_key: str | None
    feature_key: str | None
    c_header_path: Path
    c_source_path: Path
    header_guard: str
    public_header: str
    function_name: str
    c_table_prefix: str
    c_enum_prefix: str
    generator_version: int
    reg_classes: tuple[RegClass, ...]
    resources: tuple[Resource, ...]
    schedule_classes: tuple[ScheduleClass, ...]
    descriptors: tuple[Descriptor, ...]
    descriptor_set_ordinal: int | None = None
    register_parts: tuple[RegisterPart, ...] = ()
    enum_domains: tuple[EnumDomain, ...] = ()
    categories: tuple[DescriptorCategory, ...] = ()
    default_category: DescriptorCategory | None = None
    requires_explicit_asm_surface: bool = False

    def __post_init__(self) -> None:
        if (
            self.default_category is not None
            and self.default_category not in self.categories
        ):
            raise ValueError(
                f"DescriptorSet '{self.key}': default_category "
                f"'{self.default_category.key}' is not declared in categories"
            )
        for descriptor in self.descriptors:
            if (
                descriptor.category is not None
                and self.categories
                and descriptor.category not in self.categories
            ):
                raise ValueError(
                    f"DescriptorSet '{self.key}': descriptor "
                    f"'{descriptor.key}' category '{descriptor.category.key}' "
                    "is not declared in categories"
                )


def target_relative_name(target_key: str | None, name: str) -> str:
    """Returns `name` relative to its target namespace when it is under one."""

    if not target_key:
        return name
    prefix = f"{target_key}."
    if name.startswith(prefix):
        return name.removeprefix(prefix)
    return name


def descriptor_set_relative_name(descriptor_set: DescriptorSet, name: str) -> str:
    """Returns `name` relative to `descriptor_set`'s target namespace."""

    return target_relative_name(descriptor_set.target_key, name)
