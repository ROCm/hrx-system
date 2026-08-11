# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU descriptor-derived VOPD component tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.c import c_string_arg as _c_string_arg  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_planning_table_inputs import (  # noqa: E402
    AmdgpuPlanningTableInputs,
    load_amdgpu_planning_table_inputs,
)
from loom.gen.target.low.validation import operand_role_is_packet_input  # noqa: E402
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    AmdgpuIsaFactSource,
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
)
from loom.target.arch.amdgpu.names import amdgpu_c_identifier_fragment  # noqa: E402
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION,
    AmdgpuDescriptorSetInfo,
    amdgpu_descriptor_set_ordinal,
    sorted_descriptor_set_infos,
)
from loom.target.low_descriptors import ConstraintKind, Descriptor  # noqa: E402

_UINT16_MAX = 0xFFFF
_VOPD_COMPONENT_REGISTER_UNIT_COUNT = 1

_DESCRIPTOR_SET_GROUP_RDNA_VOPD = "rdna_vopd"
_DESCRIPTOR_SET_GROUP_GFX11_GFX12 = "gfx11_gfx12"
_DESCRIPTOR_SET_GROUP_RDNA4_GFX125X = "rdna4_gfx125x"

_FORM_BINARY_VGPR = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR"
_FORM_CNDMASK_VCC = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_CNDMASK_VCC"
_FORM_FMAAK_LITERAL = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAAK_LITERAL"
_FORM_FMAMK_LITERAL = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAMK_LITERAL"
_FORM_INLINE_MOV = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV"
_FORM_REGISTER_MOV = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_REGISTER_MOV"
_FORM_TIED_ACCUMULATE = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE"

_COMPONENT_FLAG_NONE = "LOOM_AMDGPU_VOPD_COMPONENT_FLAG_NONE"
_COMPONENT_FLAG_COMMUTABLE_SOURCES = "LOOM_AMDGPU_VOPD_COMPONENT_FLAG_COMMUTABLE_SOURCES"
_COMPONENT_FLAG_DUAL_MOV_SRC2_CACHE = "LOOM_AMDGPU_VOPD_COMPONENT_FLAG_DUAL_MOV_SRC2_CACHE"
_COMPONENT_FLAGS_NONE: frozenset[str] = frozenset()

_LANE_XY = "LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY"
_LANE_X = "LOOM_AMDGPU_VOPD_COMPONENT_LANE_X"
_LANE_Y = "LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y"

_PAIR_ANY = "LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY"
_PAIR_MIXED_OPCODE = "LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE"
_PAIR_SAME_OPCODE = "LOOM_AMDGPU_VOPD_COMPONENT_PAIR_SAME_OPCODE"
_PAIR_REASON_UNKNOWN = "LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN"

_SOURCE_BINARY = "LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY"
_SOURCE_NONE = "LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_NONE"
_SOURCE_SRC0 = "LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_SRC0"
_SOURCE_VSRC1 = "LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_VSRC1"

_PLACEMENT_COMPONENT_FIRST = "LOOM_LOW_PLACEMENT_PAIR_COMPONENT_FIRST"
_PLACEMENT_COMPONENT_SECOND = "LOOM_LOW_PLACEMENT_PAIR_COMPONENT_SECOND"
_PLACEMENT_VALUE_OPERAND = "LOOM_LOW_PLACEMENT_PAIR_VALUE_OPERAND"
_PLACEMENT_VALUE_RESULT = "LOOM_LOW_PLACEMENT_PAIR_VALUE_RESULT"
_PLACEMENT_DIFFERENT_MASKED = "LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION"
_PLACEMENT_DISJOINT = "LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE"

_XML_INSTRUCTION_DERIVED = object()


@dataclass(frozen=True, slots=True)
class _VopdComponentDefinition:
    descriptor_key: str
    descriptor_set_group: str
    op: str
    op_value: int
    op_name: str
    assembly_mnemonic: str
    form: str
    lane_mask: str
    pairing_mask: str
    source_register_mask: str
    descriptor_affinity_eligible: bool = True
    same_op_reason: str = "LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN"
    same_op_reason_name: str = ""
    numeric_minmax_mnemonic: str = ""
    operand_layout: tuple[int, int, int] = (0, 0, 0)
    xml_instruction_name: str | None = None
    xml_instruction_names_by_isa_key: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _VopdComponentRule:
    component: _VopdComponentDefinition
    descriptor_set_keys: tuple[str, ...]
    flags: frozenset[str] = _COMPONENT_FLAGS_NONE


@dataclass(frozen=True, slots=True)
class _VopdComponentDescriptorLookupRange:
    descriptor_set_key: str
    descriptor_set_ordinal: int
    first_descriptor_lookup: int
    descriptor_lookup_count: int


@dataclass(frozen=True, slots=True)
class _VopdPairAffinityRange:
    descriptor_set_key: str
    descriptor_set_ordinal: int
    first_pair_affinity: int
    pair_affinity_count: int


@dataclass(frozen=True, slots=True)
class _VopdPairAffinityRow:
    first_descriptor_ordinal: int
    second_descriptor_ordinal: int
    priority: int
    placement_recipe_index: int


@dataclass(frozen=True, slots=True)
class _VopdPairPlacementValueRef:
    component: str
    kind: str
    index: int
    unit_offset: int = 0


@dataclass(frozen=True, slots=True)
class _VopdPairPlacementRelation:
    result: _VopdPairPlacementValueRef
    source: _VopdPairPlacementValueRef
    kind: str
    location_mask: int
    unit_count: int = 1


@dataclass(frozen=True, slots=True)
class _VopdPairPlacementRecipe:
    alternatives: tuple[tuple[_VopdPairPlacementRelation, ...], ...]
    packet_savings: int = 1


@dataclass(frozen=True, slots=True)
class _VopdComponentTables:
    rules: tuple[_VopdComponentRule, ...]
    descriptor_lookup_ranges: tuple[_VopdComponentDescriptorLookupRange, ...]
    descriptor_lookup_rows: tuple[int, ...]
    pair_affinity_ranges: tuple[_VopdPairAffinityRange, ...]
    pair_affinity_rows: tuple[_VopdPairAffinityRow, ...]
    pair_placement_recipes: tuple[_VopdPairPlacementRecipe, ...]


def _descriptor_set_supports_vopd(info: AmdgpuDescriptorSetInfo) -> bool:
    return bool(info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION)


def _descriptor_set_storage_generator_target(
    info: AmdgpuDescriptorSetInfo,
) -> str:
    return info.storage_generator_target or info.generator_target


def _descriptor_set_keys_for_group(group: str, descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo]) -> tuple[str, ...]:
    def select(storage_generator_targets: set[str]) -> tuple[str, ...]:
        return tuple(info.key for info in descriptor_set_infos if (_descriptor_set_storage_generator_target(info) in storage_generator_targets and _descriptor_set_supports_vopd(info)))

    if group == _DESCRIPTOR_SET_GROUP_RDNA_VOPD:
        return tuple(info.key for info in descriptor_set_infos if _descriptor_set_supports_vopd(info))
    if group == _DESCRIPTOR_SET_GROUP_GFX11_GFX12:
        return select({"rdna3", "rdna3_5", "rdna4m", "rdna4"})
    if group == _DESCRIPTOR_SET_GROUP_RDNA4_GFX125X:
        return select(
            {
                "gfx12_5_generic",
                "rdna4_gfx1251",
                "rdna4_gfx125x",
            }
        )
    raise ValueError(f"unknown AMDGPU VOPD descriptor-set group '{group}'")


def _xml_instruction_name(
    component: _VopdComponentDefinition,
    isa_xml_key: str,
) -> str | None:
    for isa_key, instruction_name in component.xml_instruction_names_by_isa_key:
        if isa_key == isa_xml_key:
            return instruction_name
    return component.xml_instruction_name


def _vopd_instruction_opcodes(
    instruction: AmdgpuIsaInstruction,
) -> tuple[int, ...]:
    return tuple(encoding.opcode for encoding in instruction.encodings if encoding.encoding_name in ("VOPDXY", "VOPDXY_INST_LITERAL"))


def _lane_mask_from_instruction_encoding(
    encoding: AmdgpuIsaInstructionEncoding,
) -> str | None:
    lanes: set[str] = set()
    for operand in encoding.operands:
        if operand.field_name == "VDSTX":
            lanes.add("x")
        elif operand.field_name == "VDSTY":
            lanes.add("y")
    if lanes == {"x", "y"}:
        raise ValueError("single AMDGPU VOPD encoding cannot write both lanes")
    if lanes == {"x"}:
        return "LOOM_AMDGPU_VOPD_COMPONENT_LANE_X"
    if lanes == {"y"}:
        return "LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y"
    return None


def _lane_mask_from_instruction(instruction: AmdgpuIsaInstruction) -> str:
    lanes: set[str] = set()
    for encoding in instruction.encodings:
        if encoding.encoding_name not in ("VOPDXY", "VOPDXY_INST_LITERAL"):
            continue
        lane_mask = _lane_mask_from_instruction_encoding(encoding)
        if lane_mask == "LOOM_AMDGPU_VOPD_COMPONENT_LANE_X":
            lanes.add("x")
        elif lane_mask == "LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y":
            lanes.add("y")
    if lanes == {"x", "y"}:
        return "LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY"
    if lanes == {"x"}:
        return "LOOM_AMDGPU_VOPD_COMPONENT_LANE_X"
    if lanes == {"y"}:
        return "LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y"
    raise ValueError(f"AMDGPU VOPD instruction '{instruction.name}' has no VOPD destination lane")


def _validate_xml_instruction(
    component: _VopdComponentDefinition,
    info: AmdgpuDescriptorSetInfo,
    isa_xml_key: str,
    spec: AmdgpuIsaFactSource,
) -> None:
    instruction_name = _xml_instruction_name(component, isa_xml_key)
    if instruction_name is None:
        return
    try:
        (instruction,) = spec.select_instructions((instruction_name,))
    except ValueError as exc:
        raise ValueError(f"AMDGPU VOPD component '{component.descriptor_key}' expects missing instruction '{instruction_name}' in descriptor set '{info.key}'") from exc

    opcodes = _vopd_instruction_opcodes(instruction)
    if not opcodes:
        raise ValueError(f"AMDGPU VOPD instruction '{instruction_name}' has no VOPD encodings")
    bad_opcodes = tuple(opcode for opcode in opcodes if opcode != component.op_value)
    if bad_opcodes:
        raise ValueError(f"AMDGPU VOPD instruction '{instruction_name}' uses opcode(s) {sorted(set(opcodes))}, expected {component.op_value}")

    lane_mask = _lane_mask_from_instruction(instruction)
    if lane_mask != component.lane_mask:
        raise ValueError(f"AMDGPU VOPD instruction '{instruction_name}' has lane mask {lane_mask}, expected {component.lane_mask}")


def _validate_uint16(owner: str, field_name: str, value: int) -> None:
    if value < 0 or value > _UINT16_MAX:
        raise ValueError(f"{owner} has {field_name} {value}, expected uint16_t")


def _vopd_constant_suffix(name: str) -> str:
    return amdgpu_c_identifier_fragment(name).upper()


def _component(
    name: str,
    op_value: int,
    *,
    descriptor_key: str | None = None,
    descriptor_set_group: str = _DESCRIPTOR_SET_GROUP_RDNA_VOPD,
    assembly_mnemonic: str | None = None,
    numeric_minmax_mnemonic: str = "",
    form: str = _FORM_BINARY_VGPR,
    lane_mask: str = _LANE_XY,
    pairing_mask: str = _PAIR_ANY,
    source_register_mask: str = _SOURCE_BINARY,
    descriptor_affinity_eligible: bool = True,
    operand_layout: tuple[int, int, int] = (0, 0, 0),
    xml_instruction_name: str | None | object = _XML_INSTRUCTION_DERIVED,
    xml_instruction_names_by_isa_key: tuple[tuple[str, str], ...] = (),
) -> _VopdComponentDefinition:
    constant_suffix = _vopd_constant_suffix(name)
    same_op_reason = "LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN"
    same_op_reason_name = ""
    if pairing_mask == _PAIR_ANY:
        same_op_reason = f"LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_{constant_suffix}"
        same_op_reason_name = f"dual_{name}"

    resolved_xml_instruction_name: str | None
    if xml_instruction_name is _XML_INSTRUCTION_DERIVED:
        resolved_xml_instruction_name = f"V_DUAL_{constant_suffix}"
    else:
        resolved_xml_instruction_name = xml_instruction_name

    return _VopdComponentDefinition(
        descriptor_key=descriptor_key or f"amdgpu.v_{name}",
        descriptor_set_group=descriptor_set_group,
        op=f"LOOM_AMDGPU_VOPD_OP_{constant_suffix}",
        op_value=op_value,
        op_name=name,
        assembly_mnemonic=assembly_mnemonic or f"v_dual_{name}",
        numeric_minmax_mnemonic=numeric_minmax_mnemonic,
        form=form,
        lane_mask=lane_mask,
        pairing_mask=pairing_mask,
        source_register_mask=source_register_mask,
        descriptor_affinity_eligible=descriptor_affinity_eligible,
        same_op_reason=same_op_reason,
        same_op_reason_name=same_op_reason_name,
        operand_layout=operand_layout,
        xml_instruction_name=resolved_xml_instruction_name,
        xml_instruction_names_by_isa_key=xml_instruction_names_by_isa_key,
    )


def _component_definitions() -> tuple[_VopdComponentDefinition, ...]:
    return (
        _component("fmac_f32", 0, form=_FORM_TIED_ACCUMULATE, operand_layout=(0, 1, 2)),
        _component("fmaak_f32", 1, form=_FORM_FMAAK_LITERAL),
        _component("fmamk_f32", 2, form=_FORM_FMAMK_LITERAL),
        _component("mul_f32", 3),
        _component("add_f32", 4),
        _component("sub_f32", 5),
        _component("subrev_f32", 6),
        _component("mov_b32", 8, form=_FORM_INLINE_MOV, source_register_mask=_SOURCE_NONE),
        _component(
            "mov_b32",
            8,
            descriptor_key="amdgpu.v_mov_b32_copy",
            form=_FORM_REGISTER_MOV,
            source_register_mask=_SOURCE_SRC0,
            descriptor_affinity_eligible=False,
        ),
        _component(
            "cndmask_b32",
            9,
            descriptor_key="amdgpu.v_cndmask_b32.vcc",
            form=_FORM_CNDMASK_VCC,
        ),
        _component(
            "max_f32",
            10,
            numeric_minmax_mnemonic="v_dual_max_num_f32",
            xml_instruction_names_by_isa_key=(
                ("rdna3", "V_DUAL_MAX_F32"),
                ("rdna4", "V_DUAL_MAX_NUM_F32"),
            ),
        ),
        _component(
            "min_f32",
            11,
            numeric_minmax_mnemonic="v_dual_min_num_f32",
            xml_instruction_names_by_isa_key=(
                ("rdna3", "V_DUAL_MIN_F32"),
                ("rdna4", "V_DUAL_MIN_NUM_F32"),
            ),
        ),
        _component(
            "dot2_f32_f16",
            12,
            assembly_mnemonic="v_dual_dot2acc_f32_f16",
            form=_FORM_TIED_ACCUMULATE,
            operand_layout=(2, 0, 1),
            xml_instruction_name="V_DUAL_DOT2ACC_F32_F16",
        ),
        _component(
            "dot2_f32_bf16",
            13,
            assembly_mnemonic="v_dual_dot2acc_f32_bf16",
            form=_FORM_TIED_ACCUMULATE,
            operand_layout=(2, 0, 1),
            xml_instruction_name="V_DUAL_DOT2ACC_F32_BF16",
        ),
        _component(
            "add_u32",
            16,
            assembly_mnemonic="v_dual_add_nc_u32",
            lane_mask=_LANE_Y,
            pairing_mask=_PAIR_MIXED_OPCODE,
            xml_instruction_name="V_DUAL_ADD_NC_U32",
        ),
        _component(
            "lshlrev_b32",
            17,
            lane_mask=_LANE_Y,
            pairing_mask=_PAIR_MIXED_OPCODE,
        ),
        _component(
            "and_b32",
            18,
            descriptor_set_group=_DESCRIPTOR_SET_GROUP_GFX11_GFX12,
            lane_mask=_LANE_Y,
            pairing_mask=_PAIR_MIXED_OPCODE,
        ),
        _component(
            "max_i32",
            23,
            descriptor_set_group=_DESCRIPTOR_SET_GROUP_RDNA4_GFX125X,
            lane_mask=_LANE_Y,
            pairing_mask=_PAIR_MIXED_OPCODE,
            xml_instruction_name=None,
        ),
        _component(
            "min_i32",
            24,
            descriptor_set_group=_DESCRIPTOR_SET_GROUP_RDNA4_GFX125X,
            lane_mask=_LANE_Y,
            pairing_mask=_PAIR_MIXED_OPCODE,
            xml_instruction_name=None,
        ),
        _component(
            "sub_u32",
            20,
            descriptor_set_group=_DESCRIPTOR_SET_GROUP_RDNA4_GFX125X,
            assembly_mnemonic="v_dual_sub_nc_u32",
            lane_mask=_LANE_Y,
            pairing_mask=_PAIR_MIXED_OPCODE,
            xml_instruction_name=None,
        ),
        _component(
            "lshrrev_b32",
            21,
            descriptor_set_group=_DESCRIPTOR_SET_GROUP_RDNA4_GFX125X,
            lane_mask=_LANE_Y,
            pairing_mask=_PAIR_MIXED_OPCODE,
            xml_instruction_name=None,
        ),
        _component(
            "ashrrev_i32",
            22,
            descriptor_set_group=_DESCRIPTOR_SET_GROUP_RDNA4_GFX125X,
            lane_mask=_LANE_Y,
            pairing_mask=_PAIR_MIXED_OPCODE,
            xml_instruction_name=None,
        ),
    )


def amdgpu_vopd_instruction_names_by_isa_key() -> dict[str, tuple[str, ...]]:
    """Returns XML instruction facts consumed by VOPD table validation."""

    descriptor_set_infos = sorted_descriptor_set_infos()
    descriptor_set_infos_by_key = _descriptor_set_infos_by_key(descriptor_set_infos)
    names_by_isa_key: dict[str, set[str]] = {}
    for component in _component_definitions():
        descriptor_set_keys = _descriptor_set_keys_for_group(
            component.descriptor_set_group,
            descriptor_set_infos,
        )
        for descriptor_set_key in descriptor_set_keys:
            info = descriptor_set_infos_by_key[descriptor_set_key]
            for isa_info in info.isa_infos:
                instruction_name = _xml_instruction_name(
                    component,
                    isa_info.isa_xml_key,
                )
                if instruction_name is not None:
                    names_by_isa_key.setdefault(isa_info.isa_xml_key, set()).add(instruction_name)
    return {isa_key: tuple(sorted(instruction_names)) for isa_key, instruction_names in sorted(names_by_isa_key.items())}


def _descriptor_set_infos_by_key(
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
) -> dict[str, AmdgpuDescriptorSetInfo]:
    return {info.key: info for info in descriptor_set_infos}


def _validate_component_definition(
    component: _VopdComponentDefinition,
    descriptor_set_keys: Sequence[str],
    descriptor_set_infos_by_key: Mapping[str, AmdgpuDescriptorSetInfo],
    descriptors_by_set_key: Mapping[str, Mapping[str, Descriptor]],
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
) -> None:
    owner = f"AMDGPU VOPD component '{component.descriptor_key}'"
    _validate_uint16(owner, "opcode", component.op_value)
    if not descriptor_set_keys:
        raise ValueError(f"{owner} has no target descriptor sets")

    for descriptor_set_key in descriptor_set_keys:
        info = descriptor_set_infos_by_key.get(descriptor_set_key)
        if info is None:
            raise ValueError(f"{owner} references unknown descriptor set '{descriptor_set_key}'")
        if not _descriptor_set_supports_vopd(info):
            raise ValueError(f"{owner} references non-VOPD descriptor set '{descriptor_set_key}'")
        descriptors = descriptors_by_set_key[descriptor_set_key]
        if component.descriptor_key not in descriptors:
            raise ValueError(f"{owner} references descriptor set '{descriptor_set_key}' where the scalar component descriptor is absent")
        for isa_info in info.isa_infos:
            _validate_xml_instruction(
                component,
                info,
                isa_info.isa_xml_key,
                isa_specs[isa_info.isa_xml_key],
            )


def _descriptor_commutes_packet_operands(
    descriptor: Descriptor,
    lhs_packet_operand_index: int,
    rhs_packet_operand_index: int,
) -> bool:
    packet_operand_indices = tuple(index for index, operand in enumerate(descriptor.operands) if operand_role_is_packet_input(operand.role))
    packet_operand_count = len(packet_operand_indices)
    if lhs_packet_operand_index >= packet_operand_count or rhs_packet_operand_index >= packet_operand_count:
        raise ValueError(f"AMDGPU VOPD descriptor '{descriptor.key}' source operands ({lhs_packet_operand_index}, {rhs_packet_operand_index}) exceed packet operand count {packet_operand_count}")
    lhs_descriptor_operand_index = packet_operand_indices[lhs_packet_operand_index]
    rhs_descriptor_operand_index = packet_operand_indices[rhs_packet_operand_index]
    expected_pair = frozenset((lhs_descriptor_operand_index, rhs_descriptor_operand_index))
    return any(
        constraint.kind is ConstraintKind.COMMUTABLE and constraint.rhs_operand_index is not None and frozenset((constraint.lhs_operand_index, constraint.rhs_operand_index)) == expected_pair
        for constraint in descriptor.constraints
    )


def _component_flags_for_descriptor_set(
    component: _VopdComponentDefinition,
    descriptor_set_key: str,
    descriptor_set_info: AmdgpuDescriptorSetInfo,
    descriptors_by_set_key: Mapping[str, Mapping[str, Descriptor]],
) -> frozenset[str]:
    flags: set[str] = set()
    if component.source_register_mask == _SOURCE_BINARY:
        src0_index = _component_source_operand_index(component, _SOURCE_SRC0)
        vsrc1_index = _component_source_operand_index(component, _SOURCE_VSRC1)
        if _descriptor_commutes_packet_operands(
            descriptors_by_set_key[descriptor_set_key][component.descriptor_key],
            src0_index,
            vsrc1_index,
        ):
            flags.add(_COMPONENT_FLAG_COMMUTABLE_SOURCES)
    if component.form == _FORM_REGISTER_MOV and descriptor_set_info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE:
        flags.add(_COMPONENT_FLAG_DUAL_MOV_SRC2_CACHE)
    return frozenset(flags)


def _component_rules(
    component: _VopdComponentDefinition,
    descriptor_set_keys: Sequence[str],
    descriptor_set_infos_by_key: Mapping[str, AmdgpuDescriptorSetInfo],
    descriptors_by_set_key: Mapping[str, Mapping[str, Descriptor]],
) -> tuple[_VopdComponentRule, ...]:
    descriptor_set_keys_by_flags: dict[frozenset[str], list[str]] = {}
    for descriptor_set_key in descriptor_set_keys:
        flags = _component_flags_for_descriptor_set(
            component,
            descriptor_set_key,
            descriptor_set_infos_by_key[descriptor_set_key],
            descriptors_by_set_key,
        )
        descriptor_set_keys_by_flags.setdefault(flags, []).append(descriptor_set_key)
    return tuple(
        _VopdComponentRule(
            component=component,
            descriptor_set_keys=tuple(group_descriptor_set_keys),
            flags=flags,
        )
        for flags, group_descriptor_set_keys in descriptor_set_keys_by_flags.items()
    )


def _component_info_key(
    component: _VopdComponentDefinition,
) -> tuple[object, ...]:
    return (
        component.op,
        component.op_value,
        component.op_name,
        component.same_op_reason,
        component.same_op_reason_name,
        component.assembly_mnemonic,
        component.numeric_minmax_mnemonic,
        component.lane_mask,
        component.pairing_mask,
    )


def _canonical_component_infos(
    rules: Sequence[_VopdComponentRule],
) -> tuple[tuple[_VopdComponentDefinition, ...], dict[int, int]]:
    components: list[_VopdComponentDefinition] = []
    info_index_by_op_value: dict[int, int] = {}
    for rule in rules:
        component = rule.component
        info_index = info_index_by_op_value.get(component.op_value)
        if info_index is None:
            info_index_by_op_value[component.op_value] = len(components)
            components.append(component)
            continue
        canonical_component = components[info_index]
        if _component_info_key(component) != _component_info_key(canonical_component):
            raise ValueError(f"AMDGPU VOPD opcode {component.op_value} has inconsistent native facts in '{canonical_component.descriptor_key}' and '{component.descriptor_key}'")
    return tuple(components), info_index_by_op_value


def _component_flags_initializer(flags: frozenset[str]) -> str:
    if not flags:
        return _COMPONENT_FLAG_NONE
    return " | ".join(sorted(flags))


def _descriptor_lookup_rows_for_set(
    descriptor_set_key: str,
    descriptor_keys: Sequence[str],
    rules: Sequence[_VopdComponentRule],
) -> tuple[int, ...]:
    rule_by_descriptor_key: dict[str, int] = {}
    for rule_index, rule in enumerate(rules):
        if descriptor_set_key not in rule.descriptor_set_keys:
            continue
        previous_index = rule_by_descriptor_key.setdefault(
            rule.component.descriptor_key,
            rule_index + 1,
        )
        if previous_index != rule_index + 1:
            raise ValueError(f"AMDGPU VOPD descriptor '{rule.component.descriptor_key}' has multiple component rows for descriptor set '{descriptor_set_key}'")
    return tuple(rule_by_descriptor_key.get(descriptor_key, 0) for descriptor_key in descriptor_keys)


def _component_can_use_lane(
    component: _VopdComponentDefinition,
    lane_mask: str,
) -> bool:
    return component.lane_mask in (lane_mask, _LANE_XY)


def _component_can_pair(
    component: _VopdComponentDefinition,
    pairing_mask: str,
) -> bool:
    return component.pairing_mask in (pairing_mask, _PAIR_ANY)


def _components_can_pair(
    first_component: _VopdComponentDefinition,
    second_component: _VopdComponentDefinition,
) -> bool:
    if not _component_can_use_lane(first_component, _LANE_X):
        return False
    if not _component_can_use_lane(second_component, _LANE_Y):
        return False
    if first_component.op_value == second_component.op_value:
        has_same_reason = first_component.same_op_reason != _PAIR_REASON_UNKNOWN
        return _component_can_pair(first_component, _PAIR_SAME_OPCODE) and has_same_reason
    first_can_pair_mixed = _component_can_pair(first_component, _PAIR_MIXED_OPCODE)
    second_can_pair_mixed = _component_can_pair(
        second_component,
        _PAIR_MIXED_OPCODE,
    )
    return first_can_pair_mixed and second_can_pair_mixed


def _component_has_source(
    component: _VopdComponentDefinition,
    source_mask: str,
) -> bool:
    return component.source_register_mask in (source_mask, _SOURCE_BINARY)


def _component_source_operand_index(
    component: _VopdComponentDefinition,
    source_mask: str,
) -> int:
    if not _component_has_source(component, source_mask):
        raise ValueError(f"AMDGPU VOPD component '{component.descriptor_key}' has no {source_mask} source")
    if component.form == _FORM_TIED_ACCUMULATE:
        _accumulator_index, src0_index, vsrc1_index = component.operand_layout
        return src0_index if source_mask == _SOURCE_SRC0 else vsrc1_index
    if component.form in (
        _FORM_BINARY_VGPR,
        _FORM_CNDMASK_VCC,
        _FORM_FMAAK_LITERAL,
        _FORM_FMAMK_LITERAL,
    ):
        return 0 if source_mask == _SOURCE_SRC0 else 1
    if component.form == _FORM_REGISTER_MOV:
        return 0
    raise ValueError(f"AMDGPU VOPD component '{component.descriptor_key}' has unsupported source form '{component.form}'")


def _component_operand_count(component: _VopdComponentDefinition) -> int:
    if component.form == _FORM_TIED_ACCUMULATE:
        return max(component.operand_layout) + 1
    if component.form in (
        _FORM_BINARY_VGPR,
        _FORM_FMAAK_LITERAL,
        _FORM_FMAMK_LITERAL,
    ):
        return 2
    if component.form == _FORM_CNDMASK_VCC:
        return 3
    if component.form == _FORM_INLINE_MOV:
        return 0
    if component.form == _FORM_REGISTER_MOV:
        return 1
    raise ValueError(f"AMDGPU VOPD component '{component.descriptor_key}' has unsupported operand form '{component.form}'")


def _validate_pair_placement_ref(
    owner: str,
    field_name: str,
    ref: _VopdPairPlacementValueRef,
    first_component: _VopdComponentDefinition,
    second_component: _VopdComponentDefinition,
) -> None:
    component = first_component if ref.component == _PLACEMENT_COMPONENT_FIRST else second_component
    value_count = 1 if ref.kind == _PLACEMENT_VALUE_RESULT else _component_operand_count(component)
    if ref.index >= value_count:
        raise ValueError(f"{owner} {field_name} references {component.descriptor_key} {ref.kind} {ref.index} outside count {value_count}")


def _placement_result_ref(component: str) -> _VopdPairPlacementValueRef:
    return _VopdPairPlacementValueRef(
        component=component,
        kind=_PLACEMENT_VALUE_RESULT,
        index=0,
    )


def _placement_source_ref(
    pair_component: str,
    component: _VopdComponentDefinition,
    source_mask: str,
) -> _VopdPairPlacementValueRef:
    return _VopdPairPlacementValueRef(
        component=pair_component,
        kind=_PLACEMENT_VALUE_OPERAND,
        index=_component_source_operand_index(component, source_mask),
    )


def _source_mask_for_component_orientation(
    source_mask: str,
    component_flags: frozenset[str],
) -> str:
    if _COMPONENT_FLAG_COMMUTABLE_SOURCES not in component_flags:
        return source_mask
    if source_mask == _SOURCE_SRC0:
        return _SOURCE_VSRC1
    if source_mask == _SOURCE_VSRC1:
        return _SOURCE_SRC0
    raise ValueError(f"unsupported AMDGPU VOPD source mask '{source_mask}'")


def _component_sources_are_commutable(
    component_flags: frozenset[str],
) -> bool:
    return _COMPONENT_FLAG_COMMUTABLE_SOURCES in component_flags


def _components_share_source_cache(
    first_component: _VopdComponentDefinition,
    second_component: _VopdComponentDefinition,
    source_mask: str,
    first_component_flags: frozenset[str],
    second_component_flags: frozenset[str],
) -> bool:
    if source_mask != _SOURCE_SRC0:
        return True
    has_separate_caches = _COMPONENT_FLAG_DUAL_MOV_SRC2_CACHE in first_component_flags and _COMPONENT_FLAG_DUAL_MOV_SRC2_CACHE in second_component_flags
    return not (has_separate_caches and first_component.op_value == second_component.op_value)


def _pair_placement_relations(
    first_component: _VopdComponentDefinition,
    second_component: _VopdComponentDefinition,
    first_component_flags: frozenset[str],
    second_component_flags: frozenset[str],
) -> tuple[_VopdPairPlacementRelation, ...]:
    first_result = _placement_result_ref(_PLACEMENT_COMPONENT_FIRST)
    second_result = _placement_result_ref(_PLACEMENT_COMPONENT_SECOND)
    relations: list[_VopdPairPlacementRelation] = [
        _VopdPairPlacementRelation(
            result=first_result,
            source=second_result,
            kind=_PLACEMENT_DIFFERENT_MASKED,
            location_mask=0x1,
        )
    ]
    for source_mask in (_SOURCE_SRC0, _SOURCE_VSRC1):
        first_source_mask = _source_mask_for_component_orientation(
            source_mask,
            first_component_flags,
        )
        second_source_mask = _source_mask_for_component_orientation(
            source_mask,
            second_component_flags,
        )
        first_has_source = _component_has_source(first_component, first_source_mask)
        second_has_source = _component_has_source(
            second_component,
            second_source_mask,
        )
        if (
            first_has_source
            and second_has_source
            and _components_share_source_cache(
                first_component,
                second_component,
                source_mask,
                first_component_flags,
                second_component_flags,
            )
        ):
            relations.append(
                _VopdPairPlacementRelation(
                    result=_placement_source_ref(
                        _PLACEMENT_COMPONENT_FIRST,
                        first_component,
                        first_source_mask,
                    ),
                    source=_placement_source_ref(
                        _PLACEMENT_COMPONENT_SECOND,
                        second_component,
                        second_source_mask,
                    ),
                    kind=_PLACEMENT_DIFFERENT_MASKED,
                    location_mask=0x3,
                )
            )
        if second_has_source:
            relations.append(
                _VopdPairPlacementRelation(
                    result=first_result,
                    source=_placement_source_ref(
                        _PLACEMENT_COMPONENT_SECOND,
                        second_component,
                        second_source_mask,
                    ),
                    kind=_PLACEMENT_DISJOINT,
                    location_mask=0,
                )
            )
        if first_has_source:
            relations.append(
                _VopdPairPlacementRelation(
                    result=second_result,
                    source=_placement_source_ref(
                        _PLACEMENT_COMPONENT_FIRST,
                        first_component,
                        first_source_mask,
                    ),
                    kind=_PLACEMENT_DISJOINT,
                    location_mask=0,
                )
            )
    return tuple(relations)


def _pair_placement_recipe(
    first_component: _VopdComponentDefinition,
    second_component: _VopdComponentDefinition,
    first_component_flags: frozenset[str] = _COMPONENT_FLAGS_NONE,
    second_component_flags: frozenset[str] = _COMPONENT_FLAGS_NONE,
) -> _VopdPairPlacementRecipe:
    first_aligned_flags = first_component_flags - {_COMPONENT_FLAG_COMMUTABLE_SOURCES}
    second_aligned_flags = second_component_flags - {_COMPONENT_FLAG_COMMUTABLE_SOURCES}
    aligned_relations = _pair_placement_relations(
        first_component,
        second_component,
        first_aligned_flags,
        second_aligned_flags,
    )
    alternatives = [aligned_relations]
    first_sources_are_commutable = _component_sources_are_commutable(first_component_flags)
    second_sources_are_commutable = _component_sources_are_commutable(second_component_flags)
    if first_sources_are_commutable:
        crossed_relations = _pair_placement_relations(
            first_component,
            second_component,
            first_component_flags,
            second_aligned_flags,
        )
    elif second_sources_are_commutable:
        crossed_relations = _pair_placement_relations(
            first_component,
            second_component,
            first_aligned_flags,
            second_component_flags,
        )
    else:
        crossed_relations = aligned_relations
    if frozenset(crossed_relations) != frozenset(aligned_relations):
        alternatives.append(crossed_relations)
    return _VopdPairPlacementRecipe(alternatives=tuple(alternatives))


def _pair_affinity_rows_for_set(
    rules: Sequence[_VopdComponentRule],
    descriptor_lookup_rows: Sequence[int],
    placement_recipes: list[_VopdPairPlacementRecipe],
    placement_recipe_indices: dict[_VopdPairPlacementRecipe, int],
) -> tuple[_VopdPairAffinityRow, ...]:
    rows: list[_VopdPairAffinityRow] = []
    for first_ordinal, first_rule_index_plus_one in enumerate(descriptor_lookup_rows):
        if first_rule_index_plus_one == 0:
            continue
        first_rule = rules[first_rule_index_plus_one - 1]
        if not first_rule.component.descriptor_affinity_eligible:
            continue
        for second_ordinal, second_rule_index_plus_one in enumerate(descriptor_lookup_rows):
            if second_rule_index_plus_one == 0:
                continue
            second_rule = rules[second_rule_index_plus_one - 1]
            if not second_rule.component.descriptor_affinity_eligible:
                continue
            if not _components_can_pair(first_rule.component, second_rule.component):
                continue
            placement_recipe = _pair_placement_recipe(
                first_rule.component,
                second_rule.component,
                first_rule.flags,
                second_rule.flags,
            )
            placement_recipe_index = placement_recipe_indices.get(placement_recipe)
            if placement_recipe_index is None:
                placement_recipe_index = len(placement_recipes)
                placement_recipes.append(placement_recipe)
                placement_recipe_indices[placement_recipe] = placement_recipe_index
            rows.append(
                _VopdPairAffinityRow(
                    first_descriptor_ordinal=first_ordinal,
                    second_descriptor_ordinal=second_ordinal,
                    priority=2 if first_ordinal == second_ordinal else 1,
                    placement_recipe_index=placement_recipe_index,
                )
            )
    return tuple(rows)


def _validate_vopd_component_tables(tables: _VopdComponentTables) -> None:
    component_infos, _ = _canonical_component_infos(tables.rules)
    reason_ops: dict[str, int] = {}
    for component in component_infos:
        if component.same_op_reason == _PAIR_REASON_UNKNOWN:
            continue
        existing_op = reason_ops.setdefault(component.same_op_reason, component.op_value)
        if existing_op != component.op_value:
            raise ValueError(f"AMDGPU VOPD pair reason '{component.same_op_reason}' maps to opcodes {existing_op} and {component.op_value}")
    rule_count = len(tables.rules)
    lookup_count = len(tables.descriptor_lookup_rows)
    lookup_ranges_by_set_key = {row.descriptor_set_key: row for row in tables.descriptor_lookup_ranges}
    for row in tables.descriptor_lookup_ranges:
        owner = f"AMDGPU VOPD descriptor lookup range '{row.descriptor_set_key}'"
        remaining_lookup_count = lookup_count - row.first_descriptor_lookup
        lookup_start_oob = row.first_descriptor_lookup > lookup_count
        lookup_count_oob = row.descriptor_lookup_count > remaining_lookup_count
        if lookup_start_oob or lookup_count_oob:
            raise ValueError(f"{owner} is out of bounds")
        _validate_uint16(
            owner,
            "first descriptor lookup",
            row.first_descriptor_lookup,
        )
        _validate_uint16(
            owner,
            "descriptor lookup count",
            row.descriptor_lookup_count,
        )
        lookup_stop = row.first_descriptor_lookup + row.descriptor_lookup_count
        for descriptor_index_plus_one in tables.descriptor_lookup_rows[row.first_descriptor_lookup : lookup_stop]:
            if descriptor_index_plus_one == 0:
                continue
            if descriptor_index_plus_one > rule_count:
                raise ValueError(f"{owner} references an out-of-bounds rule")
    placement_relation_count = 0
    for recipe_index, recipe in enumerate(tables.pair_placement_recipes):
        owner = f"AMDGPU VOPD pair-placement recipe {recipe_index}"
        if not recipe.alternatives:
            raise ValueError(f"{owner} is empty")
        relation_count = len(recipe.alternatives[0])
        if relation_count == 0:
            raise ValueError(f"{owner} has an empty alternative")
        if any(len(alternative) != relation_count for alternative in recipe.alternatives):
            raise ValueError(f"{owner} alternatives have inconsistent relation counts")
        _validate_uint16(owner, "first relation", placement_relation_count)
        _validate_uint16(owner, "relation count", relation_count)
        _validate_uint16(owner, "alternative count", len(recipe.alternatives))
        if recipe.packet_savings <= 0:
            raise ValueError(f"{owner} has no packet savings")
        _validate_uint16(owner, "packet savings", recipe.packet_savings)
        relations = (relation for alternative in recipe.alternatives for relation in alternative)
        for relation in relations:
            for field_name, ref in (
                ("result", relation.result),
                ("source", relation.source),
            ):
                if ref.component not in (
                    _PLACEMENT_COMPONENT_FIRST,
                    _PLACEMENT_COMPONENT_SECOND,
                ):
                    raise ValueError(f"{owner} has unknown {field_name} component")
                if ref.kind not in (
                    _PLACEMENT_VALUE_OPERAND,
                    _PLACEMENT_VALUE_RESULT,
                ):
                    raise ValueError(f"{owner} has unknown {field_name} value kind")
                _validate_uint16(owner, f"{field_name} index", ref.index)
                _validate_uint16(owner, f"{field_name} unit offset", ref.unit_offset)
            if relation.unit_count <= 0:
                raise ValueError(f"{owner} has an empty relation")
            _validate_uint16(owner, "unit count", relation.unit_count)
            for field_name, ref in (
                ("result", relation.result),
                ("source", relation.source),
            ):
                if ref.unit_offset + relation.unit_count > _VOPD_COMPONENT_REGISTER_UNIT_COUNT:
                    raise ValueError(f"{owner} {field_name} range exceeds one VOPD register unit")
            if relation.kind == _PLACEMENT_DIFFERENT_MASKED:
                if relation.location_mask <= 0:
                    raise ValueError(f"{owner} has an empty location mask")
            elif relation.kind == _PLACEMENT_DISJOINT:
                if relation.location_mask != 0:
                    raise ValueError(f"{owner} disjoint relation has a location mask")
            else:
                raise ValueError(f"{owner} has unknown relation kind")
        placement_relation_count += relation_count * len(recipe.alternatives)
    if placement_relation_count > _UINT16_MAX:
        raise ValueError("AMDGPU VOPD pair-placement relations exceed uint16_t")
    pair_affinity_count = len(tables.pair_affinity_rows)
    for row in tables.pair_affinity_ranges:
        owner = f"AMDGPU VOPD pair-affinity range '{row.descriptor_set_key}'"
        lookup_range = lookup_ranges_by_set_key.get(row.descriptor_set_key)
        if lookup_range is None:
            raise ValueError(f"{owner} has no descriptor lookup range")
        remaining_pair_count = pair_affinity_count - row.first_pair_affinity
        pair_start_oob = row.first_pair_affinity > pair_affinity_count
        pair_count_oob = row.pair_affinity_count > remaining_pair_count
        if pair_start_oob or pair_count_oob:
            raise ValueError(f"{owner} is out of bounds")
        _validate_uint16(owner, "first pair affinity", row.first_pair_affinity)
        _validate_uint16(owner, "pair affinity count", row.pair_affinity_count)
        pair_stop = row.first_pair_affinity + row.pair_affinity_count
        for pair in tables.pair_affinity_rows[row.first_pair_affinity : pair_stop]:
            _validate_uint16(
                owner,
                "first descriptor ordinal",
                pair.first_descriptor_ordinal,
            )
            _validate_uint16(
                owner,
                "second descriptor ordinal",
                pair.second_descriptor_ordinal,
            )
            _validate_uint16(owner, "priority", pair.priority)
            if pair.placement_recipe_index >= len(tables.pair_placement_recipes):
                raise ValueError(f"{owner} references an out-of-bounds placement recipe")
            _validate_uint16(
                owner,
                "placement recipe index + 1",
                pair.placement_recipe_index + 1,
            )
            descriptor_lookup_count = lookup_range.descriptor_lookup_count
            first_descriptor_oob = pair.first_descriptor_ordinal >= descriptor_lookup_count
            second_descriptor_oob = pair.second_descriptor_ordinal >= descriptor_lookup_count
            if first_descriptor_oob or second_descriptor_oob:
                raise ValueError(f"{owner} references an out-of-bounds descriptor ordinal")
            lookup_start = lookup_range.first_descriptor_lookup
            first_rule_index_plus_one = tables.descriptor_lookup_rows[lookup_start + pair.first_descriptor_ordinal]
            second_rule_index_plus_one = tables.descriptor_lookup_rows[lookup_start + pair.second_descriptor_ordinal]
            if first_rule_index_plus_one == 0 or second_rule_index_plus_one == 0:
                raise ValueError(f"{owner} references a non-component descriptor")
            first_component = tables.rules[first_rule_index_plus_one - 1].component
            second_component = tables.rules[second_rule_index_plus_one - 1].component
            recipe = tables.pair_placement_recipes[pair.placement_recipe_index]
            for alternative in recipe.alternatives:
                for relation in alternative:
                    _validate_pair_placement_ref(
                        owner,
                        "result",
                        relation.result,
                        first_component,
                        second_component,
                    )
                    _validate_pair_placement_ref(
                        owner,
                        "source",
                        relation.source,
                        first_component,
                        second_component,
                    )


def _materialize_vopd_component_tables(
    inputs: AmdgpuPlanningTableInputs,
) -> _VopdComponentTables:
    descriptor_set_infos = inputs.descriptor_set_infos
    isa_specs = inputs.isa_specs
    infos_by_key = _descriptor_set_infos_by_key(descriptor_set_infos)
    descriptor_sets_by_key = inputs.descriptor_sets_by_key
    descriptor_keys_by_set_key = {key: tuple(descriptor.key for descriptor in descriptor_set.descriptors) for key, descriptor_set in descriptor_sets_by_key.items()}
    descriptors_by_set_key = {key: {descriptor.key: descriptor for descriptor in descriptor_set.descriptors} for key, descriptor_set in descriptor_sets_by_key.items()}
    components = _component_definitions()

    rules: list[_VopdComponentRule] = []
    descriptor_sets_by_ordinal = sorted(
        descriptor_set_infos,
        key=lambda info: amdgpu_descriptor_set_ordinal(info.key),
    )
    for component in components:
        descriptor_set_keys = _descriptor_set_keys_for_group(
            component.descriptor_set_group,
            descriptor_sets_by_ordinal,
        )
        _validate_component_definition(
            component,
            descriptor_set_keys,
            infos_by_key,
            descriptors_by_set_key,
            isa_specs,
        )
        rules.extend(
            _component_rules(
                component,
                descriptor_set_keys,
                infos_by_key,
                descriptors_by_set_key,
            )
        )

    descriptor_lookup_ranges: list[_VopdComponentDescriptorLookupRange] = []
    descriptor_lookup_rows: list[int] = []
    pair_affinity_ranges: list[_VopdPairAffinityRange] = []
    pair_affinity_rows: list[_VopdPairAffinityRow] = []
    pair_placement_recipes: list[_VopdPairPlacementRecipe] = []
    pair_placement_recipe_indices: dict[_VopdPairPlacementRecipe, int] = {}
    for info in descriptor_sets_by_ordinal:
        if not _descriptor_set_supports_vopd(info):
            continue
        descriptor_set_ordinal = amdgpu_descriptor_set_ordinal(info.key)
        if descriptor_set_ordinal < 0:
            raise ValueError(f"AMDGPU descriptor set '{info.key}' has invalid ordinal {descriptor_set_ordinal}")
        set_lookup_rows = _descriptor_lookup_rows_for_set(
            info.key,
            descriptor_keys_by_set_key[info.key],
            rules,
        )
        descriptor_lookup_ranges.append(
            _VopdComponentDescriptorLookupRange(
                descriptor_set_key=info.key,
                descriptor_set_ordinal=descriptor_set_ordinal,
                first_descriptor_lookup=len(descriptor_lookup_rows),
                descriptor_lookup_count=len(set_lookup_rows),
            )
        )
        descriptor_lookup_rows.extend(set_lookup_rows)
        set_pair_affinity_rows = _pair_affinity_rows_for_set(
            rules,
            set_lookup_rows,
            pair_placement_recipes,
            pair_placement_recipe_indices,
        )
        pair_affinity_ranges.append(
            _VopdPairAffinityRange(
                descriptor_set_key=info.key,
                descriptor_set_ordinal=descriptor_set_ordinal,
                first_pair_affinity=len(pair_affinity_rows),
                pair_affinity_count=len(set_pair_affinity_rows),
            )
        )
        pair_affinity_rows.extend(set_pair_affinity_rows)

    tables = _VopdComponentTables(
        rules=tuple(rules),
        descriptor_lookup_ranges=tuple(descriptor_lookup_ranges),
        descriptor_lookup_rows=tuple(descriptor_lookup_rows),
        pair_affinity_ranges=tuple(pair_affinity_ranges),
        pair_affinity_rows=tuple(pair_affinity_rows),
        pair_placement_recipes=tuple(pair_placement_recipes),
    )
    _validate_vopd_component_tables(tables)
    return tables


def _generated_header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator=("loom.gen.target.arch.amdgpu.descriptors.amdgpu_vopd_component_tables"),
        ),
        "",
    ]


def _component_info_initializer(
    index: int,
    component: _VopdComponentDefinition,
) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_COMPONENT_INFO_RULE(",
            f"    {index}, {component.op}, {component.same_op_reason},",
            f"    {_c_string_arg(component.op_name)},",
            f"    {_c_string_arg(component.same_op_reason_name)},",
            f"    {_c_string_arg(component.assembly_mnemonic)},",
            f"    {_c_string_arg(component.numeric_minmax_mnemonic)},",
            f"    {component.lane_mask}, {component.pairing_mask})",
        ]
    )


def _component_rule_initializer(
    index: int,
    info_index: int,
    rule: _VopdComponentRule,
) -> str:
    component = rule.component
    accumulator_index, src0_index, vsrc1_index = component.operand_layout
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_COMPONENT_RULE(",
            f"    {index}, {info_index},",
            f"    {component.form},",
            f"    {accumulator_index}, {src0_index}, {vsrc1_index},",
            f"    {component.source_register_mask},",
            f"    {_component_flags_initializer(rule.flags)})",
        ]
    )


def _component_reason_initializer(
    index: int,
    component: _VopdComponentDefinition,
) -> str | None:
    if component.same_op_reason == _PAIR_REASON_UNKNOWN:
        return None
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_COMPONENT_REASON_RULE(",
            f"    {component.same_op_reason},",
            f"    {index})",
        ]
    )


def _descriptor_lookup_range_initializer(row: _VopdComponentDescriptorLookupRange) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_COMPONENT_DESCRIPTOR_LOOKUP_RANGE(",
            f"    {row.descriptor_set_ordinal}, {row.first_descriptor_lookup},",
            f"    {row.descriptor_lookup_count})",
        ]
    )


def _descriptor_lookup_row_initializer(rule_index_plus_one: int) -> str:
    return f"LOOM_AMDGPU_VOPD_COMPONENT_DESCRIPTOR_LOOKUP({rule_index_plus_one})"


def _pair_affinity_range_initializer(row: _VopdPairAffinityRange) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_PAIR_AFFINITY_RANGE(",
            f"    {row.descriptor_set_ordinal}, {row.first_pair_affinity},",
            f"    {row.pair_affinity_count})",
        ]
    )


def _pair_affinity_row_initializer(row: _VopdPairAffinityRow) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_PAIR_AFFINITY(",
            f"    {row.first_descriptor_ordinal},",
            f"    {row.second_descriptor_ordinal}, {row.priority},",
            f"    {row.placement_recipe_index + 1})",
        ]
    )


def _pair_placement_recipe_initializer(
    recipe_index: int,
    first_relation: int,
    recipe: _VopdPairPlacementRecipe,
) -> str:
    relation_count = len(recipe.alternatives[0])
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_PAIR_PLACEMENT_RECIPE(",
            f"    {recipe_index}, {first_relation}, {relation_count},",
            f"    {len(recipe.alternatives)}, {recipe.packet_savings})",
        ]
    )


def _pair_placement_relation_initializer(
    relation: _VopdPairPlacementRelation,
) -> str:
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_PAIR_PLACEMENT_RELATION(",
            f"    {relation.result.component}, {relation.result.kind},",
            f"    {relation.result.index}, {relation.result.unit_offset},",
            f"    {relation.source.component}, {relation.source.kind},",
            f"    {relation.source.index}, {relation.source.unit_offset},",
            f"    {relation.unit_count}, {relation.kind},",
            f"    0x{relation.location_mask:X})",
        ]
    )


def _emit_component_rules(tables: _VopdComponentTables) -> str:
    component_infos, info_index_by_op_value = _canonical_component_infos(tables.rules)
    reason_initializers = [initializer for index, component in enumerate(component_infos) if (initializer := _component_reason_initializer(index, component)) is not None]
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_component_info_initializer(index, component) for index, component in enumerate(component_infos)),
                *(
                    _component_rule_initializer(
                        index,
                        info_index_by_op_value[rule.component.op_value],
                        rule,
                    )
                    for index, rule in enumerate(tables.rules)
                ),
                *reason_initializers,
            ]
        )
        + "\n"
    )


def _emit_descriptor_lookup_ranges(tables: _VopdComponentTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_descriptor_lookup_range_initializer(row) for row in tables.descriptor_lookup_ranges),
            ]
        )
        + "\n"
    )


def _emit_descriptor_lookup_rows(tables: _VopdComponentTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_descriptor_lookup_row_initializer(row) for row in tables.descriptor_lookup_rows),
            ]
        )
        + "\n"
    )


def _emit_pair_affinity_ranges(tables: _VopdComponentTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_pair_affinity_range_initializer(row) for row in tables.pair_affinity_ranges),
            ]
        )
        + "\n"
    )


def _emit_pair_affinity_rows(tables: _VopdComponentTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_pair_affinity_row_initializer(row) for row in tables.pair_affinity_rows),
            ]
        )
        + "\n"
    )


def _emit_pair_placement_recipes(tables: _VopdComponentTables) -> str:
    recipe_initializers: list[str] = []
    relation_initializers: list[str] = []
    first_relation = 0
    for recipe_index, recipe in enumerate(tables.pair_placement_recipes):
        recipe_initializers.append(
            _pair_placement_recipe_initializer(
                recipe_index,
                first_relation,
                recipe,
            )
        )
        relation_initializers.extend(_pair_placement_relation_initializer(relation) for alternative in recipe.alternatives for relation in alternative)
        first_relation += len(recipe.alternatives[0]) * len(recipe.alternatives)
    return (
        "\n".join(
            [
                *_generated_header(),
                *recipe_initializers,
                *relation_initializers,
            ]
        )
        + "\n"
    )


def _write_output(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def generate_vopd_component_table_outputs(
    inputs: AmdgpuPlanningTableInputs,
    *,
    component_rules_path: Path | None = None,
    descriptor_lookup_ranges_path: Path | None = None,
    descriptor_lookups_path: Path | None = None,
    pair_affinity_ranges_path: Path | None = None,
    pair_affinities_path: Path | None = None,
    pair_placement_recipes_path: Path | None = None,
) -> None:
    """Generates the requested VOPD component table fragments."""

    tables = _materialize_vopd_component_tables(inputs)
    if component_rules_path is not None:
        _write_output(component_rules_path, _emit_component_rules(tables))
    if descriptor_lookup_ranges_path is not None:
        _write_output(
            descriptor_lookup_ranges_path,
            _emit_descriptor_lookup_ranges(tables),
        )
    if descriptor_lookups_path is not None:
        _write_output(
            descriptor_lookups_path,
            _emit_descriptor_lookup_rows(tables),
        )
    if pair_affinity_ranges_path is not None:
        _write_output(
            pair_affinity_ranges_path,
            _emit_pair_affinity_ranges(tables),
        )
    if pair_affinities_path is not None:
        _write_output(pair_affinities_path, _emit_pair_affinity_rows(tables))
    if pair_placement_recipes_path is not None:
        _write_output(
            pair_placement_recipes_path,
            _emit_pair_placement_recipes(tables),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU descriptor-derived VOPD table fragments.")
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
    )
    parser.add_argument(
        "--component-rules",
        type=Path,
        help="Generated VOPD component rule row fragment path.",
    )
    parser.add_argument(
        "--descriptor-lookup-ranges",
        type=Path,
        help="Generated VOPD descriptor-set lookup range fragment path.",
    )
    parser.add_argument(
        "--descriptor-lookups",
        type=Path,
        help="Generated VOPD descriptor-ordinal lookup fragment path.",
    )
    parser.add_argument(
        "--pair-affinity-ranges",
        type=Path,
        help="Generated VOPD pair-affinity range fragment path.",
    )
    parser.add_argument(
        "--pair-affinities",
        type=Path,
        help="Generated VOPD pair-affinity row fragment path.",
    )
    parser.add_argument(
        "--pair-placement-recipes",
        type=Path,
        help="Generated VOPD pair-placement recipe fragment path.",
    )
    args = parser.parse_args(argv)
    requested_outputs = (
        args.component_rules,
        args.descriptor_lookup_ranges,
        args.descriptor_lookups,
        args.pair_affinity_ranges,
        args.pair_affinities,
        args.pair_placement_recipes,
    )
    if not any(path is not None for path in requested_outputs):
        parser.error("at least one output path is required")

    generate_vopd_component_table_outputs(
        load_amdgpu_planning_table_inputs(
            args.isa_xml,
            amdgpu_vopd_instruction_names_by_isa_key(),
        ),
        component_rules_path=args.component_rules,
        descriptor_lookup_ranges_path=args.descriptor_lookup_ranges,
        descriptor_lookups_path=args.descriptor_lookups,
        pair_affinity_ranges_path=args.pair_affinity_ranges,
        pair_affinities_path=args.pair_affinities,
        pair_placement_recipes_path=args.pair_placement_recipes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
