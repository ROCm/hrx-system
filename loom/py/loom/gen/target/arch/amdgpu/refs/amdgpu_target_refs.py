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

from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    amdgpu_common_reg_class_ids,
    amdgpu_core_descriptor_set_instruction_names_by_isa_key,
    amdgpu_descriptor_ref_keys,
    amdgpu_immediate_encoding_id_items,
    build_amdgpu_core_descriptor_set_from_specs,
)
from loom.target.arch.amdgpu.descriptors.memory import (  # noqa: E402
    _FLAT_LOAD_DESCRIPTOR_KEYS,
    _FLAT_STORE_DESCRIPTOR_KEYS,
)
from loom.target.arch.amdgpu.encoding import (  # noqa: E402
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
    AmdgpuIsaFactSource,
    parse_amdgpu_isa_xml_paths_for_instructions,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AmdgpuDescriptorSetInfo,
    amdgpu_descriptor_set_ordinal,
    sorted_descriptor_set_infos,
)
from loom.target.low_descriptors import (  # noqa: E402
    Descriptor,
    DescriptorSet,
    Immediate,
    ImmediateFlag,
    ImmediateKind,
    InstructionClass,
    OperandFlag,
    OperandRole,
    Resource,
    ResourceFlag,
    ResourceKind,
    ScheduleClass,
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

_DPP_DESCRIPTOR_KEYS = frozenset(
    (
        "amdgpu.v_mov_b32_dpp",
        "amdgpu.v_mov_b32_dpp16",
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
_REL32_DESCRIPTOR_KEYS = (
    "amdgpu.s_add_u32.rhs_symbol_rel32_lo",
    "amdgpu.s_addc_u32.rhs_symbol_rel32_hi",
)


@dataclass(frozen=True, slots=True)
class _DescriptorSetRefTable:
    descriptor_set_ordinal: int
    descriptor_set_key: str
    descriptor_ordinals: list[int | None]
    descriptor_refs: list[str | None]
    descriptor_traits: list[tuple[str, ...]]
    vmem_result_order_classes: list[str]
    sdwa_dst_sel_immediate_slots: list[int | None]
    literal_immediate_slots: list[int | None]
    address_offset_immediate_slots: list[int | None]
    reg_class_traits: list[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _DescriptorTraitContext:
    descriptor_set: DescriptorSet
    resources: Mapping[str, Resource]
    schedule_classes: Mapping[str, ScheduleClass]


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


def _select_descriptor_set_infos(
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


def _descriptor_ref_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}DescriptorRefsByOrdinal"


def _descriptor_set_trait_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}DescriptorTraits"


def _descriptor_set_vmem_result_order_class_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}DescriptorVmemResultOrderClasses"


def _descriptor_set_immediate_slot_table_name(key: str) -> str:
    suffix = target_relative_name("amdgpu", key.removesuffix(".core"))
    c_suffix = "".join(part[:1].upper() + part[1:] for part in suffix.split(".") if part)
    return f"kAmdgpu{c_suffix}DescriptorImmediateSlots"


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
    if len(descriptor.operands) < 2 or descriptor.operands[0].role is not OperandRole.RESULT or descriptor.operands[1].role is not OperandRole.OPERAND:
        raise ValueError(f"AMDGPU descriptor set '{descriptor_set.key}' rel32 descriptor '{descriptor_key}' must begin with one result and the PC lhs operand")
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
    for descriptor_key in _SPILL_LOWERING_SCRATCH_DESCRIPTOR_KEYS:
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
    return _DescriptorTraitContext(
        descriptor_set=descriptor_set,
        resources={resource.name: resource for resource in descriptor_set.resources},
        schedule_classes={schedule_class.name: schedule_class for schedule_class in descriptor_set.schedule_classes},
    )


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
    if uses_vector_alu:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU")
    if any(resource.kind == ResourceKind.SCALAR_ALU for resource in issue_resources):
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU")
    if uses_matrix:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX")
    if descriptor.encoding_format_id in _VECTOR_MEMORY_ENCODING_FORMAT_IDS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY")
    if descriptor.key in _TRANSCENDENTAL_DESCRIPTOR_KEYS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL")
    if descriptor.key in _DPP_DESCRIPTOR_KEYS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP")
    if descriptor.key in _READFIRSTLANE_DESCRIPTOR_KEYS:
        trait_names.append("LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE")
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
    return tuple(trait_names)


def _descriptor_vmem_result_order_class_name(descriptor: Descriptor) -> str:
    has_result = any(operand.role in (OperandRole.RESULT, OperandRole.OPERAND_RESULT) for operand in descriptor.operands)
    if not has_result:
        return "LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE"
    if descriptor.encoding_format_id in _IMAGE_MEMORY_ENCODING_FORMAT_IDS:
        return "LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN"
    if descriptor.encoding_format_id in _VECTOR_MEMORY_ENCODING_FORMAT_IDS:
        return "LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER"
    return "LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE"


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


def _validate_descriptor_trait_keys() -> None:
    descriptor_ref_keys = frozenset(amdgpu_descriptor_ref_keys())
    trait_descriptor_keys = _TRANSCENDENTAL_DESCRIPTOR_KEYS | _DPP_DESCRIPTOR_KEYS | _READFIRSTLANE_DESCRIPTOR_KEYS
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
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
) -> list[_DescriptorSetRefTable]:
    _validate_descriptor_trait_keys()
    descriptor_ref_keys = amdgpu_descriptor_ref_keys()
    descriptor_ref_key_set = frozenset(descriptor_ref_keys)
    descriptor_set_tables: list[_DescriptorSetRefTable] = []
    for descriptor_set_info in descriptor_set_infos:
        descriptor_set = build_amdgpu_core_descriptor_set_from_specs(
            descriptor_set_info.generator_target,
            isa_specs,
        )
        if descriptor_set.key != descriptor_set_info.key:
            raise ValueError(f"AMDGPU descriptor-set builder '{descriptor_set_info.generator_target}' produced '{descriptor_set.key}', expected '{descriptor_set_info.key}'")
        _validate_lowering_descriptor_contracts(descriptor_set)
        descriptor_ordinals = {descriptor.key: ordinal for ordinal, descriptor in enumerate(descriptor_set.descriptors)}
        descriptor_refs = [_descriptor_ref_constant_name(descriptor.key) if descriptor.key in descriptor_ref_key_set else None for descriptor in descriptor_set.descriptors]
        trait_context = _descriptor_trait_context(descriptor_set)
        descriptor_traits = [_descriptor_trait_names(trait_context, descriptor) for descriptor in descriptor_set.descriptors]
        vmem_result_order_classes = [_descriptor_vmem_result_order_class_name(descriptor) for descriptor in descriptor_set.descriptors]
        sdwa_dst_sel_immediate_slots = [
            _descriptor_sdwa_dst_sel_immediate_slot(
                descriptor_set,
                descriptor,
                trait_names,
            )
            for descriptor, trait_names in zip(
                descriptor_set.descriptors,
                descriptor_traits,
                strict=True,
            )
        ]
        literal_immediate_slots = [_descriptor_literal_immediate_slot(descriptor_set, descriptor) for descriptor in descriptor_set.descriptors]
        address_offset_immediate_slots = [_descriptor_address_offset_immediate_slot(descriptor_set, descriptor) for descriptor in descriptor_set.descriptors]
        descriptor_set_tables.append(
            _DescriptorSetRefTable(
                descriptor_set_ordinal=amdgpu_descriptor_set_ordinal(descriptor_set_info.key),
                descriptor_set_key=descriptor_set_info.key,
                descriptor_ordinals=[descriptor_ordinals.get(key) for key in descriptor_ref_keys],
                descriptor_refs=descriptor_refs,
                descriptor_traits=descriptor_traits,
                vmem_result_order_classes=vmem_result_order_classes,
                sdwa_dst_sel_immediate_slots=sdwa_dst_sel_immediate_slots,
                literal_immediate_slots=literal_immediate_slots,
                address_offset_immediate_slots=address_offset_immediate_slots,
                reg_class_traits=[_reg_class_trait_names(reg_class.name) for reg_class in descriptor_set.reg_classes],
            )
        )
    descriptor_set_tables.sort(key=lambda table: table.descriptor_set_ordinal)
    return descriptor_set_tables


def _emit_tables_header() -> str:
    descriptor_ref_keys = amdgpu_descriptor_ref_keys()
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
        "#define LOOM_AMDGPU_DESCRIPTOR_REF_NONE UINT16_MAX",
        f"#define LOOM_AMDGPU_DESCRIPTOR_REF_COUNT {_u16_literal(len(descriptor_ref_keys))}",
        f"#define LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT {_u16_literal(len(sorted_descriptor_set_infos()))}",
        f"#define LOOM_AMDGPU_REL32_SYMBOL_IMMEDIATE_SLOT {_u16_literal(_REL32_SYMBOL_IMMEDIATE_SLOT)}",
        f"#define LOOM_AMDGPU_REL32_BYTE_OFFSET_IMMEDIATE_SLOT {_u16_literal(_REL32_BYTE_OFFSET_IMMEDIATE_SLOT)}",
        "",
    ]
    lines.extend(f"#define {_descriptor_ref_constant_name(key)} {_u16_literal(index)}" for index, key in enumerate(descriptor_ref_keys))
    lines.append("")
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
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
) -> str:
    descriptor_set_tables = _materialize_descriptor_ref_tables(isa_specs, descriptor_set_infos)
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
    for descriptor_set_table in descriptor_set_tables:
        table_name = _descriptor_set_table_name(descriptor_set_table.descriptor_set_key)
        lines.append(f"static const uint32_t {table_name}[] = {{")
        for descriptor_ordinal in descriptor_set_table.descriptor_ordinals:
            if descriptor_ordinal is None:
                lines.append("    LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,")
            else:
                lines.append(f"    {descriptor_ordinal}u,")
        lines.append("};")
        lines.append("")
        descriptor_ref_table_name = _descriptor_ref_table_name(descriptor_set_table.descriptor_set_key)
        lines.append(f"static const uint16_t {descriptor_ref_table_name}[] = {{")
        for descriptor_ref in descriptor_set_table.descriptor_refs:
            if descriptor_ref is None:
                lines.append("    LOOM_AMDGPU_DESCRIPTOR_REF_NONE,")
            else:
                lines.append(f"    {descriptor_ref},")
        lines.append("};")
        lines.append("")
        trait_table_name = _descriptor_set_trait_table_name(descriptor_set_table.descriptor_set_key)
        lines.append(f"static const loom_amdgpu_descriptor_traits_t {trait_table_name}[] = {{")
        for trait_names in descriptor_set_table.descriptor_traits:
            if trait_names:
                lines.append(f"    {' | '.join(trait_names)},")
            else:
                lines.append("    0,")
        lines.append("};")
        lines.append("")
        vmem_result_order_class_table_name = _descriptor_set_vmem_result_order_class_table_name(descriptor_set_table.descriptor_set_key)
        lines.append(f"static const uint8_t {vmem_result_order_class_table_name}[] = {{")
        lines.extend(f"    {order_class}," for order_class in descriptor_set_table.vmem_result_order_classes)
        lines.append("};")
        lines.append("")
        immediate_slot_table_name = _descriptor_set_immediate_slot_table_name(descriptor_set_table.descriptor_set_key)
        lines.append(f"static const loom_amdgpu_descriptor_immediate_slots_t {immediate_slot_table_name}[] = {{")
        for (
            sdwa_dst_sel_slot,
            literal_slot,
            address_offset_slot,
        ) in zip(
            descriptor_set_table.sdwa_dst_sel_immediate_slots,
            descriptor_set_table.literal_immediate_slots,
            descriptor_set_table.address_offset_immediate_slots,
            strict=True,
        ):
            sdwa_dst_sel_expr = "LOOM_LOW_ID_NONE" if sdwa_dst_sel_slot is None else f"UINT16_C({sdwa_dst_sel_slot})"
            literal_expr = "LOOM_LOW_ID_NONE" if literal_slot is None else f"UINT16_C({literal_slot})"
            address_offset_expr = "LOOM_LOW_ID_NONE" if address_offset_slot is None else f"UINT16_C({address_offset_slot})"
            lines.append(f"    {{.sdwa_dst_sel = {sdwa_dst_sel_expr}, .literal = {literal_expr}, .address_offset = {address_offset_expr}}},")
        lines.append("};")
        lines.append("")
        reg_class_trait_table_name = _reg_class_trait_table_name(descriptor_set_table.descriptor_set_key)
        lines.append(f"static const loom_amdgpu_reg_class_traits_t {reg_class_trait_table_name}[] = {{")
        for trait_names in descriptor_set_table.reg_class_traits:
            if trait_names:
                lines.append(f"    {' | '.join(trait_names)},")
            else:
                lines.append("    0,")
        lines.append("};")
        lines.append("")

    tables_by_ordinal = {table.descriptor_set_ordinal: table.descriptor_set_key for table in descriptor_set_tables}
    lines.append("const uint32_t* const kLoomAmdgpuDescriptorRefOrdinalTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_ordinal, descriptor_set_key in tables_by_ordinal.items():
        table_expr = _descriptor_set_table_name(descriptor_set_key)
        lines.append(f"    [{_u16_literal(descriptor_set_ordinal)}] = {table_expr},")
    lines.append("};")
    lines.append("")
    lines.append("const uint16_t* const kLoomAmdgpuDescriptorRefByOrdinalTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_ordinal, descriptor_set_key in tables_by_ordinal.items():
        table_expr = _descriptor_ref_table_name(descriptor_set_key)
        lines.append(f"    [{_u16_literal(descriptor_set_ordinal)}] = {table_expr},")
    lines.append("};")
    lines.append("")
    lines.append("const loom_amdgpu_descriptor_traits_t* const kLoomAmdgpuDescriptorTraitTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_ordinal, descriptor_set_key in tables_by_ordinal.items():
        table_expr = _descriptor_set_trait_table_name(descriptor_set_key)
        lines.append(f"    [{_u16_literal(descriptor_set_ordinal)}] = {table_expr},")
    lines.append("};")
    lines.append("")
    lines.append("const uint8_t* const kLoomAmdgpuDescriptorVmemResultOrderClassTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_ordinal, descriptor_set_key in tables_by_ordinal.items():
        table_expr = _descriptor_set_vmem_result_order_class_table_name(descriptor_set_key)
        lines.append(f"    [{_u16_literal(descriptor_set_ordinal)}] = {table_expr},")
    lines.append("};")
    lines.append("")
    lines.append("const loom_amdgpu_descriptor_immediate_slots_t* const kLoomAmdgpuDescriptorImmediateSlotTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_ordinal, descriptor_set_key in tables_by_ordinal.items():
        table_expr = _descriptor_set_immediate_slot_table_name(descriptor_set_key)
        lines.append(f"    [{_u16_literal(descriptor_set_ordinal)}] = {table_expr},")
    lines.append("};")
    lines.append("")
    lines.append("const loom_amdgpu_reg_class_traits_t* const kLoomAmdgpuRegClassTraitTables[LOOM_AMDGPU_TARGET_REF_DESCRIPTOR_SET_ORDINAL_COUNT] = {")
    for descriptor_set_ordinal, descriptor_set_key in tables_by_ordinal.items():
        table_expr = _reg_class_trait_table_name(descriptor_set_key)
        lines.append(f"    [{_u16_literal(descriptor_set_ordinal)}] = {table_expr},")
    lines.append("};")
    return "\n".join(lines) + "\n"


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

    descriptor_set_infos = _select_descriptor_set_infos(args.descriptor_set)
    isa_specs = parse_amdgpu_isa_xml_paths_for_instructions(
        _parse_isa_xml_paths(args.isa_xml),
        amdgpu_core_descriptor_set_instruction_names_by_isa_key(descriptor_set_infos),
    )
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(
        _emit_tables_header(),
        encoding="utf-8",
    )
    args.source.write_text(
        _emit_source(
            public_header=args.public_header,
            isa_specs=isa_specs,
            descriptor_set_infos=descriptor_set_infos,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
