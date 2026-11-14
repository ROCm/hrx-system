# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU dense target reference constants and maps."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support import c_arrays  # noqa: E402
from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.low import attr_indices  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    amdgpu_common_reg_class_ids,
    amdgpu_core_descriptor_set_instruction_names_by_isa_key,
    amdgpu_descriptor_ref_immediates,
    amdgpu_descriptor_ref_keys,
    amdgpu_immediate_encoding_id_items,
    build_amdgpu_core_descriptor_sets_from_specs,
)
from loom.target.arch.amdgpu.descriptors.cdna import (  # noqa: E402
    AMDGPU_MFMA_TIMINGS_BY_DESCRIPTOR_SET,
)
from loom.target.arch.amdgpu.descriptors.common import (  # noqa: E402
    _REG_EXEC,
    _SCHEDULE_VMEM_CACHE_INVALIDATE,
)
from loom.target.arch.amdgpu.descriptors.memory import (  # noqa: E402
    _FLAT_LOAD_DESCRIPTOR_KEYS,
    _FLAT_STORE_DESCRIPTOR_KEYS,
)
from loom.target.arch.amdgpu.encoding import (  # noqa: E402
    AMDGPU_DPP_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FIELD_IDS,
    AMDGPU_ENCODING_FORMAT_FLAT,
    AMDGPU_ENCODING_FORMAT_IDS,
    AMDGPU_ENCODING_FORMAT_MUBUF,
    AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
    AMDGPU_ENCODING_FORMAT_VBUFFER,
    AMDGPU_ENCODING_FORMAT_VFLAT,
    AMDGPU_ENCODING_FORMAT_VGLOBAL,
    AMDGPU_ENCODING_FORMAT_VOP1_SDWA,
    AMDGPU_ENCODING_FORMAT_VSCRATCH,
)
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    parse_amdgpu_isa_xml_paths_for_instructions,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_STORE_DATA_WAIT_STATES,
    AmdgpuDescriptorSetInfo,
    amdgpu_descriptor_set_ordinal,
    amdgpu_descriptor_set_supported_target_contract_keys,
    sorted_descriptor_set_infos,
)
from loom.target.low_descriptors import (  # noqa: E402
    ConstraintKind,
    Descriptor,
    DescriptorSet,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    OperandAddressMapKind,
    OperandFlag,
    OperandRole,
    Resource,
    ResourceFlag,
    ResourceKind,
    ScheduleClass,
    StorageLeaseAttachment,
    StorageLeaseKind,
    target_relative_name,
)

_SYSTEM_MEMORY_GLOBAL_LOAD_DESCRIPTOR_KEYS = (
    "amdgpu.global_load_b32_saddr",
    "amdgpu.global_load_b64_saddr",
)

_SPILL_LOWERING_SCRATCH_DESCRIPTOR_KEYS = (
    "amdgpu.scratch_load_b32_offset_only",
    "amdgpu.scratch_load_b64_offset_only",
    "amdgpu.scratch_load_b128_offset_only",
    "amdgpu.scratch_load_b32_vaddr",
    "amdgpu.scratch_load_b64_vaddr",
    "amdgpu.scratch_load_b128_vaddr",
    "amdgpu.scratch_store_b32_offset_only",
    "amdgpu.scratch_store_b64_offset_only",
    "amdgpu.scratch_store_b128_offset_only",
    "amdgpu.scratch_store_b32_vaddr",
    "amdgpu.scratch_store_b64_vaddr",
    "amdgpu.scratch_store_b128_vaddr",
)

_SPILL_LOWERING_HELPER_DESCRIPTOR_KEYS = (
    "amdgpu.v_mov_b32_copy",
    "amdgpu.v_readfirstlane_b32",
    "amdgpu.s_mov_b64_exec_read",
    "amdgpu.s_mov_b64_exec.full",
    "amdgpu.s_mov_b64_exec",
)

_SPILL_LOWERING_M0_DESCRIPTOR_KEY = "amdgpu.s_mov_b32_m0.imm"

_SPILL_LOWERING_DESCRIPTOR_SHAPES = {
    "amdgpu.scratch_load_b32_offset_only": (0, 1),
    "amdgpu.scratch_load_b64_offset_only": (0, 1),
    "amdgpu.scratch_load_b128_offset_only": (0, 1),
    "amdgpu.scratch_load_b32_vaddr": (1, 1),
    "amdgpu.scratch_load_b64_vaddr": (1, 1),
    "amdgpu.scratch_load_b128_vaddr": (1, 1),
    "amdgpu.scratch_store_b32_offset_only": (1, 0),
    "amdgpu.scratch_store_b64_offset_only": (1, 0),
    "amdgpu.scratch_store_b128_offset_only": (1, 0),
    "amdgpu.scratch_store_b32_vaddr": (2, 0),
    "amdgpu.scratch_store_b64_vaddr": (2, 0),
    "amdgpu.scratch_store_b128_vaddr": (2, 0),
    "amdgpu.v_mov_b32_copy": (1, 1),
    "amdgpu.v_readfirstlane_b32": (1, 1),
    "amdgpu.s_mov_b64_exec_read": (0, 1),
    "amdgpu.s_mov_b64_exec.full": (0, 0),
    "amdgpu.s_mov_b64_exec": (1, 0),
    "amdgpu.s_mov_b32_m0.imm": (0, 1),
}

_VECTOR_MEMORY_ENCODING_FORMAT_IDS = frozenset(
    (
        AMDGPU_ENCODING_FORMAT_MUBUF,
        AMDGPU_ENCODING_FORMAT_IDS["ENC_MTBUF"],
        AMDGPU_ENCODING_FORMAT_VBUFFER,
        AMDGPU_ENCODING_FORMAT_FLAT,
        AMDGPU_ENCODING_FORMAT_IDS["ENC_FLAT_GLBL"],
        AMDGPU_ENCODING_FORMAT_IDS["ENC_FLAT_GLOBAL"],
        AMDGPU_ENCODING_FORMAT_IDS["ENC_FLAT_SCRATCH"],
        AMDGPU_ENCODING_FORMAT_VFLAT,
        AMDGPU_ENCODING_FORMAT_VGLOBAL,
        AMDGPU_ENCODING_FORMAT_VSCRATCH,
    )
)

_IMAGE_MEMORY_ENCODING_FORMAT_IDS = frozenset(
    (
        AMDGPU_ENCODING_FORMAT_IDS["ENC_MIMG"],
        AMDGPU_ENCODING_FORMAT_IDS["ENC_VIMAGE"],
        AMDGPU_ENCODING_FORMAT_IDS["ENC_VSAMPLE"],
        AMDGPU_ENCODING_FORMAT_IDS["MIMG_NSA1"],
        AMDGPU_ENCODING_FORMAT_IDS["MIMG_NSA2"],
        AMDGPU_ENCODING_FORMAT_IDS["MIMG_NSA3"],
    )
)

_MATRIX_MEMORY_ENCODING_FORMAT_IDS = (
    _VECTOR_MEMORY_ENCODING_FORMAT_IDS | _IMAGE_MEMORY_ENCODING_FORMAT_IDS | frozenset(AMDGPU_ENCODING_FORMAT_IDS[name] for name in ("ENC_DS", "ENC_VDS", "ENC_EXP", "ENC_VEXPORT"))
)

_TRANSCENDENTAL_DESCRIPTOR_KEYS = frozenset(
    (
        "amdgpu.v_exp_f32",
        "amdgpu.v_log_f32",
        "amdgpu.v_sin_f32",
        "amdgpu.v_cos_f32",
        "amdgpu.v_sqrt_f32",
        "amdgpu.v_rsq_f32",
        "amdgpu.v_rcp_f32",
    )
)

_READFIRSTLANE_DESCRIPTOR_KEYS = frozenset(
    (
        "amdgpu.v_readfirstlane_b32",
        "amdgpu.v_readlane_b32.src1_inline",
    )
)

_XCNT_IMPLICIT_DRAIN_DESCRIPTOR_KEYS = frozenset(("amdgpu.s_trap",))

_XCNT_IMPLICIT_DRAIN_DESCRIPTOR_KEY_PREFIXES = (
    "amdgpu.s_getreg_b32",
    "amdgpu.s_setreg",
    "amdgpu.s_sendmsg",
    "amdgpu.s_barrier_wait",
    "amdgpu.s_barrier_signal",
)

_DST_SEL_ENCODING_FIELD_ID = AMDGPU_ENCODING_FIELD_IDS["DST_SEL"]
_DESTINATION_OP_SEL_ENCODING_FIELD_IDS = frozenset(
    (
        AMDGPU_ENCODING_FIELD_IDS["OPSEL"],
        AMDGPU_ENCODING_FIELD_IDS["OP_SEL"],
    )
)
_DESTINATION_OP_SEL_MASK = 1 << 3
_LITERAL_ENCODING_FIELD_ID = AMDGPU_ENCODING_FIELD_IDS["LITERAL"]
_REL32_SYMBOL_IMMEDIATE_SLOT = 0
_REL32_BYTE_OFFSET_IMMEDIATE_SLOT = 1
_DESCRIPTOR_IMMEDIATE_SLOT_NONE = (1 << 8) - 1
_REL32_DESCRIPTOR_KEYS = (
    "amdgpu.s_add_u32.rhs_symbol_rel32_lo",
    "amdgpu.s_addc_u32.rhs_symbol_rel32_hi",
)
_COMPACT_DESCRIPTOR_ORDINAL_NONE = (1 << 16) - 1


@dataclass(frozen=True, slots=True)
class _DescriptorSetRefTable:
    descriptor_set_ordinal: int
    descriptor_set_key: str
    descriptor_ordinals: list[int | None]
    descriptor_traits: list[tuple[str, ...]]
    memory_properties: list[str]
    immediate_slots: list[tuple[int | None, int | None, int | None, int | None]]
    reg_class_traits: list[tuple[str, ...]]
    agpr_reg_class_id: int


@dataclass(frozen=True, slots=True)
class _DescriptorSetRefTableSymbols:
    descriptor_ordinals: str
    descriptor_traits: str
    memory_properties: str
    immediate_slot_refs: str
    reg_class_traits: str


@dataclass(frozen=True, slots=True)
class _DescriptorTraitContext:
    descriptor_set: DescriptorSet
    resources: Mapping[str, Resource]
    schedule_classes: Mapping[str, ScheduleClass]
    dot_identities: Mapping[tuple[int, int], int]


def _parse_isa_xml_argument(value: str) -> tuple[str, Path]:
    key, separator, path = value.partition(":")
    if not separator or not key or not path:
        raise ValueError("AMDGPU target-ref --isa-xml entries must be key:path pairs")
    return key, Path(path)


def _parse_isa_xml_paths(values: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        key, path = _parse_isa_xml_argument(value)
        if key in paths:
            if paths[key] != path:
                raise ValueError(f"AMDGPU target-ref ISA XML key '{key}' has conflicting paths '{paths[key]}' and '{path}'")
            continue
        paths[key] = path
    return paths


def select_target_ref_descriptor_set_infos(
    descriptor_set_keys: Sequence[str],
) -> tuple[AmdgpuDescriptorSetInfo, ...]:
    if not descriptor_set_keys:
        raise ValueError("AMDGPU target-ref source generation requires at least one --descriptor-set")
    infos_by_key = {info.key: info for info in sorted_descriptor_set_infos()}
    infos = []
    for descriptor_set_key in descriptor_set_keys:
        try:
            infos.append(infos_by_key[descriptor_set_key])
        except KeyError as exc:
            raise ValueError(f"AMDGPU target-ref generator got unknown descriptor set '{descriptor_set_key}'") from exc
    return tuple(infos)


def _materialization_descriptor_set_infos(
    selected_infos: Sequence[AmdgpuDescriptorSetInfo],
) -> tuple[AmdgpuDescriptorSetInfo, ...]:
    """Adds portable views that determine exact descriptor storage order."""

    selected_keys = {info.key for info in selected_infos}
    for info in sorted_descriptor_set_infos():
        if info.member_generator_targets and selected_keys.intersection(amdgpu_descriptor_set_supported_target_contract_keys(info)):
            selected_keys.add(info.key)
    return tuple(info for info in sorted_descriptor_set_infos() if info.key in selected_keys)


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def _descriptor_ref_constant_name(key: str) -> str:
    return f"LOOM_AMDGPU_DESCRIPTOR_REF_{_c_identifier(target_relative_name('amdgpu', key))}"


def _reg_class_id_constant_name(reg_class_name: str) -> str:
    return f"LOOM_AMDGPU_REG_CLASS_ID_{_c_identifier(target_relative_name('amdgpu', reg_class_name))}"


def _immediate_encoding_id_constant_name(name: str) -> str:
    return f"LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_{_c_identifier(name)}"


def _u16_literal(value: int) -> str:
    return f"UINT16_C({value})"


def _descriptor_set_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}DescriptorRefOrdinals"


def _descriptor_set_trait_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}DescriptorTraits"


def _descriptor_set_memory_property_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}DescriptorMemoryProperties"


def _descriptor_set_immediate_slot_ref_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}DescriptorImmediateSlotRefs"


def _reg_class_trait_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}RegClassTraits"


def _descriptor_by_key(descriptor_set: DescriptorSet, key: str) -> Descriptor:
    for descriptor in descriptor_set.descriptors:
        if descriptor.key == key:
            return descriptor
    raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' is missing descriptor '{key}' required by target lowering")


def _validate_canonical_asm_operand_count(
    descriptor_set: DescriptorSet,
    descriptor_key: str,
    accepted_operand_counts: tuple[int, ...],
) -> None:
    descriptor = _descriptor_by_key(descriptor_set, descriptor_key)
    if len(descriptor.asm_forms) != 1:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' descriptor '{descriptor_key}' must have exactly one canonical asm form for target lowering; found {len(descriptor.asm_forms)}")
    operand_count = len(descriptor.asm_forms[0].operands)
    if operand_count not in accepted_operand_counts:
        accepted = ", ".join(str(count) for count in accepted_operand_counts)
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' descriptor '{descriptor_key}' canonical asm form has {operand_count} operand(s); expected one of: {accepted}")


def _validate_rel32_descriptor_contract(
    descriptor_set: DescriptorSet,
    descriptor_key: str,
) -> None:
    descriptor = _descriptor_by_key(descriptor_set, descriptor_key)
    if descriptor.encoding_format_id != AMDGPU_ENCODING_FORMAT_SOP2_LITERAL:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' rel32 descriptor '{descriptor_key}' must use SOP2 literal encoding")
    result_count = sum(operand.role is OperandRole.RESULT for operand in descriptor.operands)
    if result_count == 0 or result_count >= len(descriptor.operands) or descriptor.operands[result_count].role is not OperandRole.OPERAND or descriptor.operands[result_count].field_name != "lhs":
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' rel32 descriptor '{descriptor_key}' must place the PC lhs operand after its results")
    if len(descriptor.immediates) != 2:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' rel32 descriptor '{descriptor_key}' must have symbol and byte_offset immediates")
    symbol = descriptor.immediates[_REL32_SYMBOL_IMMEDIATE_SLOT]
    if symbol.kind is not ImmediateKind.ORDINAL or ImmediateFlag.SYMBOLIC not in symbol.flags or ImmediateFlag.RELATIVE not in symbol.flags or symbol.encoding_field_id != _LITERAL_ENCODING_FIELD_ID:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' rel32 descriptor '{descriptor_key}' symbol immediate must be a symbolic relative literal")
    byte_offset = descriptor.immediates[_REL32_BYTE_OFFSET_IMMEDIATE_SLOT]
    if byte_offset.field_name != "byte_offset" or byte_offset.kind is not ImmediateKind.UNSIGNED or byte_offset.encoding_field_id != 0:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' rel32 descriptor '{descriptor_key}' byte_offset immediate must be an unencoded unsigned value")


def _validate_lowering_descriptor_contracts(descriptor_set: DescriptorSet) -> None:
    for descriptor_key in _SYSTEM_MEMORY_GLOBAL_LOAD_DESCRIPTOR_KEYS:
        _validate_canonical_asm_operand_count(descriptor_set, descriptor_key, (2, 3))
    for descriptor_key in _FLAT_LOAD_DESCRIPTOR_KEYS:
        _validate_canonical_asm_operand_count(descriptor_set, descriptor_key, (1, 2))
    for descriptor_key in _FLAT_STORE_DESCRIPTOR_KEYS:
        _validate_canonical_asm_operand_count(descriptor_set, descriptor_key, (2, 3))
    _validate_spill_lowering_descriptor_contracts(descriptor_set)
    for descriptor_key in _REL32_DESCRIPTOR_KEYS:
        _validate_rel32_descriptor_contract(descriptor_set, descriptor_key)


def _validate_spill_lowering_descriptor_contracts(
    descriptor_set: DescriptorSet,
) -> None:
    requires_m0_materialization = False
    descriptor_keys = {descriptor.key for descriptor in descriptor_set.descriptors}
    for descriptor_key in _SPILL_LOWERING_SCRATCH_DESCRIPTOR_KEYS:
        if descriptor_key.endswith("_offset_only") and descriptor_key not in descriptor_keys:
            continue
        descriptor = _descriptor_by_key(descriptor_set, descriptor_key)
        _validate_spill_lowering_descriptor_shape(descriptor_set, descriptor)
        offset_slot = _descriptor_address_offset_immediate_slot(descriptor_set, descriptor)
        if offset_slot is None:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' spill descriptor '{descriptor_key}' has no address offset immediate")
        offset = descriptor.immediates[offset_slot]
        if offset.kind not in (
            ImmediateKind.SIGNED,
            ImmediateKind.UNSIGNED,
            ImmediateKind.ORDINAL,
        ):
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' spill descriptor '{descriptor_key}' address offset immediate has unsupported kind {offset.kind.name.lower()}")
        if offset.kind is ImmediateKind.SIGNED and not (offset.signed_min <= 0 <= offset.unsigned_max):
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' spill descriptor '{descriptor_key}' address offset immediate does not encode zero")
        requires_m0_materialization |= any(operand.role is OperandRole.RESOURCE and OperandFlag.IMPLICIT in operand.flags for operand in descriptor.operands)
    for descriptor_key in _SPILL_LOWERING_HELPER_DESCRIPTOR_KEYS:
        descriptor = _descriptor_by_key(descriptor_set, descriptor_key)
        _validate_spill_lowering_descriptor_shape(descriptor_set, descriptor)
    if requires_m0_materialization:
        descriptor = _descriptor_by_key(descriptor_set, _SPILL_LOWERING_M0_DESCRIPTOR_KEY)
        _validate_spill_lowering_descriptor_shape(descriptor_set, descriptor)
        immediate_slot = _descriptor_immediate_slot(
            descriptor_set,
            descriptor,
            "imm32",
            lambda immediate: immediate.field_name == "imm32",
        )
        if immediate_slot is None:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' spill helper '{_SPILL_LOWERING_M0_DESCRIPTOR_KEY}' has no imm32 immediate")


def _validate_spill_lowering_descriptor_shape(
    descriptor_set: DescriptorSet,
    descriptor: Descriptor,
) -> None:
    expected_operand_count, expected_result_count = _SPILL_LOWERING_DESCRIPTOR_SHAPES[descriptor.key]
    explicit_operand_count = sum(
        operand.role
        in (
            OperandRole.OPERAND,
            OperandRole.PREDICATE,
            OperandRole.RESOURCE,
        )
        and OperandFlag.IMPLICIT not in operand.flags
        for operand in descriptor.operands
    )
    result_count = sum(operand.role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT) for operand in descriptor.operands)
    if explicit_operand_count != expected_operand_count or result_count != expected_result_count:
        raise ValueError(
            f"AMDGPU descriptor set '{descriptor_set.key}' spill descriptor "
            f"'{descriptor.key}' has {explicit_operand_count} explicit operand(s) "
            f"and {result_count} result(s); expected {expected_operand_count} "
            f"operand(s) and {expected_result_count} result(s)"
        )


def _descriptor_issue_resources(
    context: _DescriptorTraitContext,
    descriptor: Descriptor,
) -> tuple[Resource, ...]:
    try:
        schedule_class = context.schedule_classes[descriptor.schedule_class]
    except KeyError as exc:
        raise ValueError(f"AMDGPU descriptor set '{context.descriptor_set.key}' descriptor '{descriptor.key}' references missing schedule class '{descriptor.schedule_class}'") from exc
    resources: list[Resource] = []
    for issue_use in schedule_class.issue_uses:
        try:
            resource = context.resources[issue_use.resource]
        except KeyError as exc:
            raise ValueError(f"AMDGPU descriptor set '{context.descriptor_set.key}' schedule class '{schedule_class.name}' references missing resource '{issue_use.resource}'") from exc
        resources.append(resource)
    return tuple(resources)


def _descriptor_trait_context(descriptor_set: DescriptorSet) -> _DescriptorTraitContext:
    dot_encodings = sorted(
        {
            (descriptor.encoding_format_id, descriptor.encoding_id)
            for descriptor in descriptor_set.descriptors
            if descriptor_set.key in AMDGPU_MFMA_TIMINGS_BY_DESCRIPTOR_SET and (descriptor.semantic_tag or "").startswith("dot.")
        }
    )
    if len(dot_encodings) > 6:
        raise ValueError(f"CDNA descriptor set '{descriptor_set.key}' exceeds the six native DOT forwarding channels")
    return _DescriptorTraitContext(
        descriptor_set=descriptor_set,
        resources={resource.name: resource for resource in descriptor_set.resources},
        schedule_classes={schedule_class.name: schedule_class for schedule_class in descriptor_set.schedule_classes},
        dot_identities={encoding: ordinal for ordinal, encoding in enumerate(dot_encodings, 1)},
    )


def _descriptor_address_source_is_retained(descriptor: Descriptor) -> bool:
    address_packet_indices: set[int] = set()
    packet_operand_index = 0
    for operand in descriptor.operands:
        if operand.role not in (
            OperandRole.OPERAND,
            OperandRole.PREDICATE,
            OperandRole.RESOURCE,
        ):
            continue
        if operand.field_name in ("addr", "vaddr"):
            address_packet_indices.add(packet_operand_index)
        packet_operand_index += 1
    return any(
        lease.kind is StorageLeaseKind.SOURCE_READ and lease.attachment is StorageLeaseAttachment.OPERAND and lease.attachment_index in address_packet_indices for lease in descriptor.storage_leases
    )


def _descriptor_writes_exec(descriptor: Descriptor) -> bool:
    return any(OperandFlag.STATE_WRITE in operand.flags and any(reg_alt.reg_class == _REG_EXEC for reg_alt in operand.reg_alts) for operand in descriptor.operands)


def _descriptor_trait_names(
    context: _DescriptorTraitContext,
    descriptor: Descriptor,
) -> tuple[str, ...]:
    trait_names: list[str] = []
    issue_resources = _descriptor_issue_resources(context, descriptor)
    matrix_coexecution_sources = tuple(resource for resource in issue_resources if ResourceFlag.MATRIX_COEXECUTION_SOURCE in resource.flags)
    if len(matrix_coexecution_sources) > 1:
        raise ValueError(f"AMDGPU descriptor set '{context.descriptor_set.key}' descriptor '{descriptor.key}' must use at most one matrix coexecution source resource")
    if matrix_coexecution_sources and matrix_coexecution_sources[0].kind != ResourceKind.MATRIX:
        raise ValueError(f"AMDGPU descriptor set '{context.descriptor_set.key}' descriptor '{descriptor.key}' matrix coexecution source resource must be a matrix resource")
    if matrix_coexecution_sources:
        source_families = tuple(
            instruction_class
            for instruction_class in (
                InstructionClass.WMMA,
                InstructionClass.SWMMAC,
            )
            if instruction_class in descriptor.instruction_classes
        )
        if len(source_families) != 1:
            raise ValueError(
                f"AMDGPU descriptor set '{context.descriptor_set.key}' descriptor '{descriptor.key}' matrix coexecution source must belong to exactly one WMMA or SWMMAC instruction class"
            )
    uses_vector_alu = any(resource.kind == ResourceKind.VECTOR_ALU for resource in issue_resources)
    uses_matrix = any(resource.kind == ResourceKind.MATRIX for resource in issue_resources)
    # The final physical dependency owner retains five native issue slots.
    # Each instruction contributes at most two output spans and one separately
    # described store payload; wide outputs remain one contiguous span.
    physical_outputs = sum(
        (operand.role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT) or OperandFlag.STATE_WRITE in operand.flags)
        and any(alternative.reg_class in ("amdgpu.sgpr", "amdgpu.vgpr", "amdgpu.agpr", "amdgpu.exec", "amdgpu.vcc", "amdgpu.m0") for alternative in operand.reg_alts)
        for operand in descriptor.operands
    )
    if physical_outputs > 2:
        raise ValueError(f"AMDGPU descriptor '{descriptor.key}' exceeds the two-result native instruction contract")
    if uses_vector_alu:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU")
    if context.dot_identities and (descriptor.semantic_tag or "").startswith("dot."):
        results = tuple(operand for operand in descriptor.operands if operand.role is OperandRole.RESULT)
        inputs = tuple(operand for operand in descriptor.operands if operand.role is OperandRole.OPERAND and OperandFlag.IMPLICIT not in operand.flags)
        if (
            not uses_vector_alu
            or len(results) != 1
            or tuple(operand.field_name for operand in inputs) != ("lhs", "rhs", "acc")
            or any(operand.unit_count != 1 for operand in (*results, *inputs))
            or any(alternative.reg_class != "amdgpu.vgpr" for alternative in results[0].reg_alts)
        ):
            raise ValueError(f"CDNA DOT forwarding requires one DWORD result and lhs/rhs/acc inputs: {descriptor.key}")
        identity = context.dot_identities[(descriptor.encoding_format_id, descriptor.encoding_id)]
        trait_names.append(f"({identity}u << LOOM_AMDGPU_DESCRIPTOR_DOT_IDENTITY_SHIFT)")
    if any(resource.kind == ResourceKind.SCALAR_ALU for resource in issue_resources):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU")
    if uses_matrix:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX")
        timings = AMDGPU_MFMA_TIMINGS_BY_DESCRIPTOR_SET.get(context.descriptor_set.key)
        if timings is not None:
            timing = timings.get(descriptor.key)
            if timing is None or timing.execution_passes not in (2, 4, 8, 16):
                raise ValueError(f"AMDGPU matrix descriptor '{descriptor.key}' has no qualified execution pass count in '{context.descriptor_set.key}'")
            trait_names.append(f"({timing.execution_passes}u << LOOM_AMDGPU_DESCRIPTOR_MATRIX_PASSES_SHIFT)")
            semantic = descriptor.semantic_tag or ""
            if not semantic.startswith(("matrix.mfma.", "matrix.smfmac.")):
                raise ValueError(f"MFMA pipeline classification requires matrix semantics: {descriptor.key}")
            # Exact forwarding floors are indexed by their fixed physical
            # result shapes in the native wait planner, without a range search.
            floor_width = 4 if timing.execution_passes == 2 else 2 if semantic.startswith("matrix.mfma.f64.") and timing.execution_passes == 4 else 0
            if floor_width:
                results = tuple(operand for operand in descriptor.operands if operand.role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT))
                if len(results) != 1 or results[0].unit_count != floor_width:
                    raise ValueError(f"MFMA exact forwarding floor requires one {floor_width}-register result: {descriptor.key}")
            results = tuple((index, operand) for index, operand in enumerate(descriptor.operands) if operand.role is OperandRole.RESULT)
            accumulators = tuple((index, operand) for index, operand in enumerate(descriptor.operands) if operand.field_name == "acc")
            if (
                len(results) != 1
                or len(accumulators) != 1
                or results[0][1].unit_count != accumulators[0][1].unit_count
                or not any(
                    constraint.kind in (ConstraintKind.TIED, ConstraintKind.PREFER_TIED) and {constraint.lhs_operand_index, constraint.rhs_operand_index} == {results[0][0], accumulators[0][0]}
                    for constraint in descriptor.constraints
                )
            ):
                raise ValueError(f"MFMA input retention requires a same-width accumulator and result: {descriptor.key}")
            if semantic.startswith("matrix.mfma.f64."):
                trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_DGEMM")
            elif not semantic.endswith(".f32"):
                trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_XDL")
    if descriptor.encoding_format_id in _VECTOR_MEMORY_ENCODING_FORMAT_IDS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY")
    if descriptor.encoding_format_id == AMDGPU_ENCODING_FORMAT_IDS["ENC_SMEM"]:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY")
    if descriptor.encoding_format_id in _MATRIX_MEMORY_ENCODING_FORMAT_IDS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_MEMORY_CONSUMER")
        if descriptor.encoding_format_id not in (AMDGPU_ENCODING_FORMAT_IDS["ENC_EXP"], AMDGPU_ENCODING_FORMAT_IDS["ENC_VEXPORT"]):
            trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY_INTERLOCK")
    if descriptor.semantic_tag in ("float.fma.f64", "float.fmac.f64"):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_FMA64")
    if descriptor.key in _TRANSCENDENTAL_DESCRIPTOR_KEYS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL")
    if descriptor.encoding_format_id in AMDGPU_DPP_ENCODING_FORMAT_IDS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP")
    if (descriptor.semantic_tag or "").startswith("lane.permlane"):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_PERMLANE")
    if descriptor.key in _READFIRSTLANE_DESCRIPTOR_KEYS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE")
    if (descriptor.semantic_tag or "").startswith("float.div.fmas."):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_DIV_FMAS")
    if descriptor.encoding_format_id == AMDGPU_ENCODING_FORMAT_VOP1_SDWA:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA")
    if descriptor.key in _XCNT_IMPLICIT_DRAIN_DESCRIPTOR_KEYS or descriptor.key.startswith(_XCNT_IMPLICIT_DRAIN_DESCRIPTOR_KEY_PREFIXES):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_XCNT_IMPLICIT_DRAIN")
    if any(ResourceFlag.VECTOR_ISSUE in resource.flags for resource in issue_resources) or InstructionClass.LDSDMA in descriptor.instruction_classes:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE")
    if matrix_coexecution_sources:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_COEXECUTION_SOURCE")
    if uses_vector_alu and any(field.encoding_field_id in _DESTINATION_OP_SEL_ENCODING_FIELD_IDS and field.value & _DESTINATION_OP_SEL_MASK for field in descriptor.encoding_field_values):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_DESTINATION_SELECTION_FORWARDING")
    if _descriptor_address_source_is_retained(descriptor):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_ADDRESS_SOURCE_RETAINED")
    if _descriptor_writes_exec(descriptor):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_WRITES_EXEC")
    if descriptor.semantic_tag in ("float.add.pk2.f32", "float.mul.pk2.f32", "float.fma.pk2.f32"):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_PACKED_F32")
    if descriptor.semantic_tag in ("integer.shl.u64", "integer.shr.u64", "integer.shr.s64"):
        shift_operands = tuple(operand for operand in descriptor.operands if operand.field_name == "shift" and operand.addressable_unit_count == 255)
        if shift_operands:
            operands = tuple(operand for operand in descriptor.operands if operand.role == OperandRole.OPERAND and OperandFlag.IMPLICIT not in operand.flags)
            if (
                len(shift_operands) != 1
                or len(operands) != 2
                or operands[0] != shift_operands[0]
                or shift_operands[0].unit_count != 1
                or shift_operands[0].address_map_kind is not OperandAddressMapKind.LOW_SUBSET
            ):
                raise ValueError(f"CDNA wide shift count must be the first one-register operand: {descriptor.key}")
            trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_SHIFT64_ALLOCATION")
    return tuple(trait_names)


def _descriptor_vmem_result_order_class_name(descriptor: Descriptor) -> str:
    # Invalidation contributes to the ordered VMEM load counter without writing
    # a register. It must not make later ordinary loads appear out of order.
    if descriptor.schedule_class == _SCHEDULE_VMEM_CACHE_INVALIDATE:
        return "LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER"
    has_result = any(operand.role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT) for operand in descriptor.operands)
    if not has_result:
        return "LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE"
    if descriptor.encoding_format_id in _IMAGE_MEMORY_ENCODING_FORMAT_IDS:
        return "LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN"
    if descriptor.encoding_format_id in _VECTOR_MEMORY_ENCODING_FORMAT_IDS:
        return "LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER"
    return "LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE"


def _descriptor_memory_properties(info: AmdgpuDescriptorSetInfo, descriptor: Descriptor) -> str:
    order_class = _descriptor_vmem_result_order_class_name(descriptor)
    if not info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_STORE_DATA_WAIT_STATES:
        return order_class
    if descriptor.encoding_format_id not in _VECTOR_MEMORY_ENCODING_FORMAT_IDS:
        return order_class
    # CDNA2 ISA 4.5, table 9 requires one slot before wide-payload reuse;
    # CDNA3/4 table 11 requires a second slot before VALU reuse. SGPR-offset
    # buffers are exempt on CDNA2. Later targets retain LLVM's one-slot VALU
    # correction for those buffers. DATA/VDATA includes 64-bit CAS payloads.
    is_cdna2 = info.generator_target == "cdna2"
    data_fields = (AMDGPU_ENCODING_FIELD_IDS["DATA"], AMDGPU_ENCODING_FIELD_IDS["VDATA"])
    for index, operand in enumerate(descriptor.operands):
        if operand.encoding_field_id not in data_fields or operand.role not in (OperandRole.OPERAND, OperandRole.OPERAND_RESULT) or operand.unit_count <= 2:
            continue
        if index >= 7:
            raise ValueError(f"wide store payload index does not fit memory properties: {descriptor.key}")
        has_scalar_offset = any(operand.encoding_field_id == AMDGPU_ENCODING_FIELD_IDS["SOFFSET"] for operand in descriptor.operands)
        if is_cdna2 and has_scalar_offset:
            return order_class
        other_cycles = int(not has_scalar_offset)
        second_valu_cycle = int(not is_cdna2 and not has_scalar_offset)
        return f"{order_class} | (({index} + 1) << 3) | ({other_cycles} << 6) | ({second_valu_cycle} << 7)"
    return order_class


def _descriptor_immediate_slot(
    descriptor_set: DescriptorSet,
    descriptor: Descriptor,
    slot_name: str,
    predicate: Callable[[Immediate], bool],
) -> int | None:
    slot_index: int | None = None
    for index, immediate in enumerate(descriptor.immediates):
        if not predicate(immediate):
            continue
        if slot_index is not None:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' descriptor '{descriptor.key}' has multiple {slot_name} immediates")
        if index >= _DESCRIPTOR_IMMEDIATE_SLOT_NONE:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' descriptor '{descriptor.key}' {slot_name} immediate slot {index} does not fit compact target refs")
        slot_index = index
    return slot_index


def _descriptor_sdwa_dst_sel_immediate_slot(
    descriptor_set: DescriptorSet,
    descriptor: Descriptor,
    trait_names: tuple[str, ...],
) -> int | None:
    slot_index = _descriptor_immediate_slot(
        descriptor_set,
        descriptor,
        "SDWA dst_sel",
        lambda immediate: immediate.encoding_field_id == _DST_SEL_ENCODING_FIELD_ID,
    )
    if "LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA" in trait_names and slot_index is None:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' SDWA descriptor '{descriptor.key}' has no dst_sel immediate")
    return slot_index


def _descriptor_literal_immediate_slot(
    descriptor_set: DescriptorSet,
    descriptor: Descriptor,
) -> int | None:
    return _descriptor_immediate_slot(
        descriptor_set,
        descriptor,
        "literal",
        lambda immediate: immediate.encoding_field_id == _LITERAL_ENCODING_FIELD_ID,
    )


def _descriptor_address_offset_immediate_slot(
    descriptor_set: DescriptorSet,
    descriptor: Descriptor,
) -> int | None:
    return _descriptor_immediate_slot(
        descriptor_set,
        descriptor,
        "address offset",
        lambda immediate: immediate.field_name == "offset",
    )


def _descriptor_cache_coherence_immediate_slot(
    descriptor_set: DescriptorSet,
    descriptor: Descriptor,
) -> int | None:
    return _descriptor_immediate_slot(
        descriptor_set,
        descriptor,
        "cache coherence",
        lambda immediate: immediate.field_name in ("glc", "scope"),
    )


def _validate_descriptor_trait_keys() -> None:
    descriptor_ref_keys = frozenset(amdgpu_descriptor_ref_keys())
    trait_descriptor_keys = _TRANSCENDENTAL_DESCRIPTOR_KEYS | _READFIRSTLANE_DESCRIPTOR_KEYS
    for descriptor_key in sorted(trait_descriptor_keys):
        if descriptor_key not in descriptor_ref_keys:
            raise ValueError(f"AMDGPU descriptor trait key '{descriptor_key}' is not listed as a descriptor ref")


def _reg_class_trait_names(reg_class_name: str) -> tuple[str, ...]:
    trait_names: list[str] = []
    if reg_class_name == "amdgpu.agpr":
        trait_names.append("LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR")
    if reg_class_name == "amdgpu.m0":
        trait_names.append("LOOM_AMDGPU_REG_CLASS_TRAIT_M0")
    if reg_class_name == "amdgpu.vcc":
        trait_names.append("LOOM_AMDGPU_REG_CLASS_TRAIT_VCC")
    return tuple(trait_names)


def _materialize_descriptor_ref_tables(
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
    descriptor_sets_by_key: Mapping[str, DescriptorSet],
) -> list[_DescriptorSetRefTable]:
    _validate_descriptor_trait_keys()
    descriptor_ref_keys = amdgpu_descriptor_ref_keys()
    descriptor_set_tables: list[_DescriptorSetRefTable] = []
    for descriptor_set_info in descriptor_set_infos:
        try:
            descriptor_set = descriptor_sets_by_key[descriptor_set_info.key]
        except KeyError as exc:
            raise ValueError(f"AMDGPU target-ref generation is missing materialized descriptor set '{descriptor_set_info.key}'") from exc
        if descriptor_set.key != descriptor_set_info.key:
            raise ValueError(f"AMDGPU target-ref generation expected descriptor set '{descriptor_set_info.key}', got '{descriptor_set.key}'")
        if len(descriptor_set.descriptors) > _COMPACT_DESCRIPTOR_ORDINAL_NONE:
            raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' has {len(descriptor_set.descriptors)} descriptors, which do not fit compact target-ref ordinals")
        _validate_lowering_descriptor_contracts(descriptor_set)
        descriptor_ordinals = {descriptor.key: ordinal for ordinal, descriptor in enumerate(descriptor_set.descriptors)}
        trait_context = _descriptor_trait_context(descriptor_set)
        descriptor_traits = [_descriptor_trait_names(trait_context, descriptor) for descriptor in descriptor_set.descriptors]
        memory_properties = [_descriptor_memory_properties(descriptor_set_info, descriptor) for descriptor in descriptor_set.descriptors]
        immediate_slots = [
            (
                _descriptor_sdwa_dst_sel_immediate_slot(descriptor_set, descriptor, trait_names),
                _descriptor_literal_immediate_slot(descriptor_set, descriptor),
                _descriptor_address_offset_immediate_slot(descriptor_set, descriptor),
                _descriptor_cache_coherence_immediate_slot(descriptor_set, descriptor),
            )
            for descriptor, trait_names in zip(descriptor_set.descriptors, descriptor_traits, strict=True)
        ]
        agpr_reg_class_ids = [index for index, reg_class in enumerate(descriptor_set.reg_classes) if reg_class.name == "amdgpu.agpr"]
        if len(agpr_reg_class_ids) > 1 or (agpr_reg_class_ids and agpr_reg_class_ids[0] >= 255):
            raise ValueError(f"AMDGPU accumulator class must have a unique byte-sized ID: {descriptor_set.key}")
        if descriptor_set.key in AMDGPU_MFMA_TIMINGS_BY_DESCRIPTOR_SET and not agpr_reg_class_ids:
            raise ValueError(f"MFMA descriptor set requires an accumulator register class: {descriptor_set.key}")
        descriptor_set_tables.append(
            _DescriptorSetRefTable(
                descriptor_set_ordinal=amdgpu_descriptor_set_ordinal(descriptor_set_info.key),
                descriptor_set_key=descriptor_set_info.key,
                descriptor_ordinals=[descriptor_ordinals.get(key) for key in descriptor_ref_keys],
                descriptor_traits=descriptor_traits,
                memory_properties=memory_properties,
                immediate_slots=immediate_slots,
                reg_class_traits=[_reg_class_trait_names(reg_class.name) for reg_class in descriptor_set.reg_classes],
                agpr_reg_class_id=agpr_reg_class_ids[0] if agpr_reg_class_ids else 255,
            )
        )
    descriptor_set_tables.sort(key=lambda table: table.descriptor_set_ordinal)
    return descriptor_set_tables


def _native_immediate_slots(descriptor_immediates: Sequence[tuple[str, tuple[Immediate, ...]]]) -> dict[str, int]:
    fields = {
        "amdgpu.s_waitcnt": ("WAITCNT", ("vmcnt", "lgkmcnt")),
        "amdgpu.s_mov_b64.symbol": ("SYMBOL_ADDRESS", ("symbol", "byte_offset")),
    }
    slots: dict[str, int] = {}
    for key, immediates in descriptor_immediates:
        if key not in fields:
            continue
        prefix, names = fields[key]
        ordinals = {immediate.field_name: index for index, immediate in enumerate(immediates)}
        for name in names:
            if name not in ordinals:
                raise ValueError(f"AMDGPU native assembly descriptor '{key}' is missing immediate '{name}'")
            constant = f"LOOM_AMDGPU_{prefix}_{name.upper()}_IMMEDIATE_SLOT"
            ordinal = ordinals[name]
            if constant in slots and slots[constant] != ordinal:
                raise ValueError(f"AMDGPU native assembly immediate '{key}.{name}' has inconsistent descriptor slots across targets")
            slots[constant] = ordinal
    return slots


def _emit_tables_header(descriptor_sets: Sequence[DescriptorSet]) -> str:
    descriptor_ref_keys = amdgpu_descriptor_ref_keys()
    # Shared readers need the complete declared schema, while selected ISA
    # materializations additionally constrain the positions they can expose.
    descriptor_immediates = (*amdgpu_descriptor_ref_immediates(), *((descriptor.key, descriptor.immediates) for descriptor_set in descriptor_sets for descriptor in descriptor_set.descriptors))
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.refs.amdgpu_target_refs"),
        "",
        "#ifndef LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_TABLES_H_",
        "#define LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_TABLES_H_",
        "",
        "#include <stdint.h>",
        "",
        '#include "loom/codegen/low/immediate_fields.h"',
        "",
        "#define LOOM_AMDGPU_DESCRIPTOR_ORDINAL_NONE UINT16_MAX",
        "#define LOOM_AMDGPU_DESCRIPTOR_REF_NONE UINT16_MAX",
        f"#define LOOM_AMDGPU_DESCRIPTOR_REF_COUNT {_u16_literal(len(descriptor_ref_keys))}",
        f"#define LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT {_u16_literal(len(sorted_descriptor_set_infos()))}",
        f"#define LOOM_AMDGPU_REL32_SYMBOL_IMMEDIATE_SLOT {_u16_literal(_REL32_SYMBOL_IMMEDIATE_SLOT)}",
        f"#define LOOM_AMDGPU_REL32_BYTE_OFFSET_IMMEDIATE_SLOT {_u16_literal(_REL32_BYTE_OFFSET_IMMEDIATE_SLOT)}",
        "",
    ]
    lines.extend(f"#define {name} {_u16_literal(index)}" for name, index in _native_immediate_slots(descriptor_immediates).items())
    lines.append("")
    lines.extend(f"#define {_descriptor_ref_constant_name(key)} {_u16_literal(index)}" for index, key in enumerate(descriptor_ref_keys))
    lines.append("")
    descriptor_ref_key_set = frozenset(descriptor_ref_keys)
    lines.extend(
        attr_indices.emit_attr_accessors(
            "LOOM_AMDGPU",
            ((key, immediates) for key, immediates in descriptor_immediates if key in descriptor_ref_key_set),
            target_key="amdgpu",
        )
    )
    lines.extend(f"#define {_reg_class_id_constant_name(reg_class_name)} {reg_class_id}u" for reg_class_name, reg_class_id in amdgpu_common_reg_class_ids())
    lines.append("")
    lines.extend(f"#define {_immediate_encoding_id_constant_name(name)} {encoding_id}u" for name, encoding_id in amdgpu_immediate_encoding_id_items())
    lines.extend(
        [
            "",
            "#endif  // LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_TABLES_H_",
        ]
    )
    return "\n".join(lines) + "\n"


def _emit_source(
    *,
    public_header: str,
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
    descriptor_sets_by_key: Mapping[str, DescriptorSet],
) -> str:
    descriptor_set_tables = _materialize_descriptor_ref_tables(
        descriptor_set_infos,
        descriptor_sets_by_key,
    )
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.refs.amdgpu_target_refs"),
        "",
        f'#include "{public_header}"',
        "",
    ]
    immediate_slot_values: list[tuple[int | None, ...]] = []
    immediate_slot_refs_by_value: dict[tuple[int | None, ...], int] = {}
    immediate_slot_refs_by_descriptor_set_key: dict[str, list[int]] = {}
    for descriptor_set_table in descriptor_set_tables:
        immediate_slot_refs: list[int] = []
        for value in descriptor_set_table.immediate_slots:
            immediate_slot_ref = immediate_slot_refs_by_value.get(value)
            if immediate_slot_ref is None:
                immediate_slot_ref = len(immediate_slot_values)
                if immediate_slot_ref >= 256:
                    raise ValueError("AMDGPU target refs have more than 256 distinct immediate-slot rows")
                immediate_slot_refs_by_value[value] = immediate_slot_ref
                immediate_slot_values.append(value)
            immediate_slot_refs.append(immediate_slot_ref)
        immediate_slot_refs_by_descriptor_set_key[descriptor_set_table.descriptor_set_key] = immediate_slot_refs

    lines.append("const loom_amdgpu_descriptor_immediate_slots_t kLoomAmdgpuDescriptorImmediateSlotValues[] = {")
    for sdwa_dst_sel_slot, literal_slot, address_offset_slot, cache_coherence_slot in immediate_slot_values:
        lines.extend(
            [
                "    {",
                f"        .sdwa_dst_sel = {'LOOM_AMDGPU_DESCRIPTOR_IMMEDIATE_SLOT_NONE' if sdwa_dst_sel_slot is None else f'UINT8_C({sdwa_dst_sel_slot})'},",
                f"        .literal = {'LOOM_AMDGPU_DESCRIPTOR_IMMEDIATE_SLOT_NONE' if literal_slot is None else f'UINT8_C({literal_slot})'},",
                f"        .address_offset = {'LOOM_AMDGPU_DESCRIPTOR_IMMEDIATE_SLOT_NONE' if address_offset_slot is None else f'UINT8_C({address_offset_slot})'},",
                f"        .cache_coherence = {'LOOM_AMDGPU_DESCRIPTOR_IMMEDIATE_SLOT_NONE' if cache_coherence_slot is None else f'UINT8_C({cache_coherence_slot})'},",
                "    },",
            ]
        )
    lines.extend(["};", ""])

    array_emitter = c_arrays.StaticArrayEmitter(lines)
    symbols_by_descriptor_set_key: dict[str, _DescriptorSetRefTableSymbols] = {}
    for descriptor_set_table in descriptor_set_tables:
        table_name = _descriptor_set_table_name(descriptor_set_table.descriptor_set_key)
        descriptor_ordinal_table_symbol = array_emitter.append_value_array(
            "uint16_t",
            table_name,
            ["LOOM_AMDGPU_DESCRIPTOR_ORDINAL_NONE" if descriptor_ordinal is None else _u16_literal(descriptor_ordinal) for descriptor_ordinal in descriptor_set_table.descriptor_ordinals],
            emit_empty=True,
        )
        trait_table_name = _descriptor_set_trait_table_name(descriptor_set_table.descriptor_set_key)
        descriptor_trait_table_symbol = array_emitter.append_value_array(
            "loom_amdgpu_descriptor_traits_t",
            trait_table_name,
            [" | ".join(trait_names) if trait_names else "0" for trait_names in descriptor_set_table.descriptor_traits],
            emit_empty=True,
        )
        memory_property_table_name = _descriptor_set_memory_property_table_name(descriptor_set_table.descriptor_set_key)
        memory_property_table_symbol = array_emitter.append_value_array(
            "uint8_t",
            memory_property_table_name,
            descriptor_set_table.memory_properties,
            emit_empty=True,
        )
        immediate_slot_ref_table_name = _descriptor_set_immediate_slot_ref_table_name(descriptor_set_table.descriptor_set_key)
        immediate_slot_ref_table_symbol = array_emitter.append_value_array(
            "uint8_t",
            immediate_slot_ref_table_name,
            [f"UINT8_C({value})" for value in immediate_slot_refs_by_descriptor_set_key[descriptor_set_table.descriptor_set_key]],
            emit_empty=True,
        )
        reg_class_trait_table_name = _reg_class_trait_table_name(descriptor_set_table.descriptor_set_key)
        reg_class_trait_table_symbol = array_emitter.append_value_array(
            "loom_amdgpu_reg_class_traits_t",
            reg_class_trait_table_name,
            [" | ".join(trait_names) if trait_names else "0" for trait_names in descriptor_set_table.reg_class_traits],
            emit_empty=True,
        )
        symbols_by_descriptor_set_key[descriptor_set_table.descriptor_set_key] = _DescriptorSetRefTableSymbols(
            descriptor_ordinals=descriptor_ordinal_table_symbol,
            descriptor_traits=descriptor_trait_table_symbol,
            memory_properties=memory_property_table_symbol,
            immediate_slot_refs=immediate_slot_ref_table_symbol,
            reg_class_traits=reg_class_trait_table_symbol,
        )

    lines.append("const uint16_t* const kLoomAmdgpuDescriptorRefOrdinalTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_table in descriptor_set_tables:
        symbols = symbols_by_descriptor_set_key[descriptor_set_table.descriptor_set_key]
        lines.append(f"    [{_u16_literal(descriptor_set_table.descriptor_set_ordinal)}] = {symbols.descriptor_ordinals},")
    lines.append("};")
    lines.append("")
    lines.append("const loom_amdgpu_descriptor_traits_t* const kLoomAmdgpuDescriptorTraitTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_table in descriptor_set_tables:
        symbols = symbols_by_descriptor_set_key[descriptor_set_table.descriptor_set_key]
        lines.append(f"    [{_u16_literal(descriptor_set_table.descriptor_set_ordinal)}] = {symbols.descriptor_traits},")
    lines.append("};")
    lines.append("")
    lines.append("const uint8_t* const kLoomAmdgpuDescriptorMemoryPropertyTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_table in descriptor_set_tables:
        symbols = symbols_by_descriptor_set_key[descriptor_set_table.descriptor_set_key]
        lines.append(f"    [{_u16_literal(descriptor_set_table.descriptor_set_ordinal)}] = {symbols.memory_properties},")
    lines.append("};")
    lines.append("")
    lines.append("const uint8_t* const kLoomAmdgpuDescriptorImmediateSlotRefTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_table in descriptor_set_tables:
        symbols = symbols_by_descriptor_set_key[descriptor_set_table.descriptor_set_key]
        lines.append(f"    [{_u16_literal(descriptor_set_table.descriptor_set_ordinal)}] = {symbols.immediate_slot_refs},")
    lines.append("};")
    lines.append("")
    lines.append("const loom_amdgpu_reg_class_traits_t* const kLoomAmdgpuRegClassTraitTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_table in descriptor_set_tables:
        symbols = symbols_by_descriptor_set_key[descriptor_set_table.descriptor_set_key]
        lines.append(f"    [{_u16_literal(descriptor_set_table.descriptor_set_ordinal)}] = {symbols.reg_class_traits},")
    lines.append("};")
    lines.append("")
    lines.append("const uint8_t kLoomAmdgpuAgprRegClassIds[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    lines.extend(f"    [{_u16_literal(table.descriptor_set_ordinal)}] = {table.agpr_reg_class_id}," for table in descriptor_set_tables)
    lines.append("};")
    return "\n".join(lines) + "\n"


def generate_target_ref_outputs(
    *,
    public_header: str,
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
    descriptor_sets_by_key: Mapping[str, DescriptorSet],
    header_path: Path,
    source_path: Path,
) -> None:
    """Emits target-reference tables from an existing descriptor corpus."""

    write_text_file(header_path, _emit_tables_header(tuple(descriptor_sets_by_key[info.key] for info in descriptor_set_infos)))
    write_text_file(
        source_path,
        _emit_source(
            public_header=public_header,
            descriptor_set_infos=descriptor_set_infos,
            descriptor_sets_by_key=descriptor_sets_by_key,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU dense target reference constants and maps.")
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated target-ref header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated target-ref source path.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/target/arch/amdgpu/refs/target_refs.h",
        help="Public include path for the generated header.",
    )
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as key:path.",
    )
    parser.add_argument(
        "--descriptor-set",
        action="append",
        default=[],
        help="Descriptor-set key to materialize into the generated source.",
    )
    args = parser.parse_args(argv)

    descriptor_set_infos = select_target_ref_descriptor_set_infos(args.descriptor_set)
    materialization_infos = _materialization_descriptor_set_infos(descriptor_set_infos)
    isa_specs = parse_amdgpu_isa_xml_paths_for_instructions(
        _parse_isa_xml_paths(args.isa_xml),
        amdgpu_core_descriptor_set_instruction_names_by_isa_key(materialization_infos),
    )
    descriptor_sets_by_target = build_amdgpu_core_descriptor_sets_from_specs(
        tuple(info.generator_target for info in materialization_infos),
        isa_specs,
    )
    generate_target_ref_outputs(
        public_header=args.public_header,
        descriptor_set_infos=descriptor_set_infos,
        descriptor_sets_by_key={info.key: descriptor_sets_by_target[info.generator_target] for info in materialization_infos},
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
