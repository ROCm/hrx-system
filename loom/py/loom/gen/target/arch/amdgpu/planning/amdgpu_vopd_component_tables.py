# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU VOPD component planning tables."""

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
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    amdgpu_descriptor_ref_keys,
    build_amdgpu_core_descriptor_set_from_spec,
)
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    AmdgpuIsaFactSource,
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
    parse_amdgpu_isa_xml_path,
)
from loom.target.arch.amdgpu.names import amdgpu_c_identifier_fragment  # noqa: E402
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION,
    AmdgpuDescriptorSetInfo,
    amdgpu_descriptor_set_ordinal,
    sorted_descriptor_set_infos,
)
from loom.target.low_descriptors import target_relative_name  # noqa: E402

_UINT16_MAX = 0xFFFF

_DESCRIPTOR_SET_GROUP_RDNA_VOPD = "rdna_vopd"
_DESCRIPTOR_SET_GROUP_GFX11_GFX12 = "gfx11_gfx12"
_DESCRIPTOR_SET_GROUP_RDNA4_GFX125X = "rdna4_gfx125x"

_FORM_BINARY_VGPR = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_BINARY_VGPR"
_FORM_FMAAK_LITERAL = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAAK_LITERAL"
_FORM_FMAMK_LITERAL = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_FMAMK_LITERAL"
_FORM_INLINE_MOV = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_INLINE_MOV"
_FORM_TIED_ACCUMULATE = "LOOM_AMDGPU_VOPD_COMPONENT_FORM_TIED_ACCUMULATE"

_LANE_XY = "LOOM_AMDGPU_VOPD_COMPONENT_LANE_XY"
_LANE_Y = "LOOM_AMDGPU_VOPD_COMPONENT_LANE_Y"

_PAIR_ANY = "LOOM_AMDGPU_VOPD_COMPONENT_PAIR_ANY"
_PAIR_MIXED_OPCODE = "LOOM_AMDGPU_VOPD_COMPONENT_PAIR_MIXED_OPCODE"

_SOURCE_BINARY = "LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_BINARY"
_SOURCE_NONE = "LOOM_AMDGPU_VOPD_COMPONENT_SOURCE_NONE"

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
    same_op_reason: str = "LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN"
    same_op_reason_name: str = ""
    rdna4_assembly_mnemonic: str = ""
    operand_layout: tuple[int, int, int] = (0, 0, 0)
    xml_instruction_name: str | None = None
    xml_instruction_names_by_isa_key: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _VopdComponentRule:
    component: _VopdComponentDefinition
    descriptor_ref: str
    descriptor_set_keys: tuple[str, ...]
    descriptor_set_mask: str


@dataclass(frozen=True, slots=True)
class _VopdComponentTables:
    rules: tuple[_VopdComponentRule, ...]


def _parse_isa_xml_argument(value: str) -> tuple[str, Path]:
    key, separator, path = value.partition(":")
    if not separator or not key or not path:
        raise ValueError("AMDGPU VOPD --isa-xml entries must be key:path pairs")
    return key, Path(path)


def _parse_isa_xml_arguments(values: Sequence[str]) -> dict[str, AmdgpuIsaFactSource]:
    paths: dict[str, Path] = {}
    specs: dict[str, AmdgpuIsaFactSource] = {}
    for value in values:
        key, path = _parse_isa_xml_argument(value)
        existing_path = paths.get(key)
        if existing_path is not None:
            if existing_path != path:
                raise ValueError(f"AMDGPU VOPD ISA XML key '{key}' has conflicting paths '{existing_path}' and '{path}'")
            continue
        paths[key] = path
        specs[key] = parse_amdgpu_isa_xml_path(path)
    return specs


def _descriptor_ref_constant_name(key: str) -> str:
    ref_name = amdgpu_c_identifier_fragment(target_relative_name("amdgpu", key))
    return f"LOOM_AMDGPU_DESCRIPTOR_REF_{ref_name}"


def _descriptor_set_ordinal_constant_name(key: str) -> str:
    ordinal_name = amdgpu_c_identifier_fragment(target_relative_name("amdgpu", key))
    if ordinal_name.endswith("_CORE"):
        ordinal_name = ordinal_name[: -len("_CORE")]
    return f"LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_{ordinal_name}"


def _descriptor_set_bit_expr(key: str) -> str:
    return f"LOOM_AMDGPU_VOPD_DESCRIPTOR_SET_BIT({_descriptor_set_ordinal_constant_name(key)})"


def _descriptor_set_mask_expr(keys: Sequence[str]) -> str:
    if not keys:
        raise ValueError("AMDGPU VOPD component row has no descriptor sets")
    return " | ".join(_descriptor_set_bit_expr(key) for key in keys)


def _descriptor_set_supports_vopd(info: AmdgpuDescriptorSetInfo) -> bool:
    return bool(info.flags & AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION)


def _descriptor_set_keys_for_group(group: str, descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo]) -> tuple[str, ...]:
    def select(keys: set[str]) -> tuple[str, ...]:
        return tuple(info.key for info in descriptor_set_infos if info.key in keys and _descriptor_set_supports_vopd(info))

    if group == _DESCRIPTOR_SET_GROUP_RDNA_VOPD:
        return tuple(info.key for info in descriptor_set_infos if _descriptor_set_supports_vopd(info))
    if group == _DESCRIPTOR_SET_GROUP_GFX11_GFX12:
        return select({"amdgpu.rdna3.core", "amdgpu.rdna4.core"})
    if group == _DESCRIPTOR_SET_GROUP_RDNA4_GFX125X:
        return select({"amdgpu.rdna4.gfx125x.core"})
    raise ValueError(f"unknown AMDGPU VOPD descriptor-set group '{group}'")


def _xml_instruction_name(component: _VopdComponentDefinition, info: AmdgpuDescriptorSetInfo) -> str | None:
    for isa_key, instruction_name in component.xml_instruction_names_by_isa_key:
        if isa_key == info.isa_xml_key:
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
    spec: AmdgpuIsaFactSource,
) -> None:
    instruction_name = _xml_instruction_name(component, info)
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
    rdna4_assembly_mnemonic: str = "",
    form: str = _FORM_BINARY_VGPR,
    lane_mask: str = _LANE_XY,
    pairing_mask: str = _PAIR_ANY,
    source_register_mask: str = _SOURCE_BINARY,
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
        rdna4_assembly_mnemonic=rdna4_assembly_mnemonic,
        form=form,
        lane_mask=lane_mask,
        pairing_mask=pairing_mask,
        source_register_mask=source_register_mask,
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
            "max_f32",
            10,
            rdna4_assembly_mnemonic="v_dual_max_num_f32",
            xml_instruction_names_by_isa_key=(
                ("rdna3", "V_DUAL_MAX_F32"),
                ("rdna4", "V_DUAL_MAX_NUM_F32"),
            ),
        ),
        _component(
            "min_f32",
            11,
            rdna4_assembly_mnemonic="v_dual_min_num_f32",
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


def _descriptor_set_infos_by_key(
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
) -> dict[str, AmdgpuDescriptorSetInfo]:
    return {info.key: info for info in descriptor_set_infos}


def _descriptor_sets_by_key(
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
) -> dict[str, set[str]]:
    descriptor_sets: dict[str, set[str]] = {}
    for info in descriptor_set_infos:
        try:
            spec = isa_specs[info.isa_xml_key]
        except KeyError as exc:
            raise ValueError(f"AMDGPU VOPD generator is missing ISA XML key '{info.isa_xml_key}' for descriptor set '{info.key}'") from exc
        descriptor_set = build_amdgpu_core_descriptor_set_from_spec(
            info.generator_target,
            spec,
        )
        if descriptor_set.key != info.key:
            raise ValueError(f"AMDGPU descriptor-set builder '{info.generator_target}' produced '{descriptor_set.key}', expected '{info.key}'")
        descriptor_sets[info.key] = {descriptor.key for descriptor in descriptor_set.descriptors}
    return descriptor_sets


def _validate_component_definition(
    component: _VopdComponentDefinition,
    descriptor_set_keys: Sequence[str],
    descriptor_set_infos_by_key: Mapping[str, AmdgpuDescriptorSetInfo],
    descriptor_keys_by_set_key: Mapping[str, set[str]],
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
    descriptor_ref_key_set: set[str],
) -> None:
    owner = f"AMDGPU VOPD component '{component.descriptor_key}'"
    _validate_uint16(owner, "opcode", component.op_value)
    if component.descriptor_key not in descriptor_ref_key_set:
        raise ValueError(f"{owner} requires a descriptor ref")
    if not descriptor_set_keys:
        raise ValueError(f"{owner} has no target descriptor sets")

    for descriptor_set_key in descriptor_set_keys:
        info = descriptor_set_infos_by_key.get(descriptor_set_key)
        if info is None:
            raise ValueError(f"{owner} references unknown descriptor set '{descriptor_set_key}'")
        if not _descriptor_set_supports_vopd(info):
            raise ValueError(f"{owner} references non-VOPD descriptor set '{descriptor_set_key}'")
        descriptor_keys = descriptor_keys_by_set_key[descriptor_set_key]
        if component.descriptor_key not in descriptor_keys:
            raise ValueError(f"{owner} references descriptor set '{descriptor_set_key}' where the scalar component descriptor is absent")
        _validate_xml_instruction(component, info, isa_specs[info.isa_xml_key])


def _materialize_vopd_component_tables(
    descriptor_set_infos: Sequence[AmdgpuDescriptorSetInfo],
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
) -> _VopdComponentTables:
    infos_by_key = _descriptor_set_infos_by_key(descriptor_set_infos)
    descriptor_keys_by_set_key = _descriptor_sets_by_key(descriptor_set_infos, isa_specs)
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
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
            descriptor_keys_by_set_key,
            isa_specs,
            descriptor_ref_key_set,
        )
        rules.append(
            _VopdComponentRule(
                component=component,
                descriptor_ref=_descriptor_ref_constant_name(component.descriptor_key),
                descriptor_set_keys=descriptor_set_keys,
                descriptor_set_mask=_descriptor_set_mask_expr(descriptor_set_keys),
            )
        )

    return _VopdComponentTables(rules=tuple(rules))


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
            generator=("loom.gen.target.arch.amdgpu.planning.amdgpu_vopd_component_tables"),
        ),
        "",
    ]


def _component_rule_initializer(rule: _VopdComponentRule) -> str:
    component = rule.component
    accumulator_index, src0_index, vsrc1_index = component.operand_layout
    return "\n".join(
        [
            "LOOM_AMDGPU_VOPD_COMPONENT_RULE(",
            f"    {rule.descriptor_ref},",
            f"    {rule.descriptor_set_mask},",
            f"    {component.op}, {component.same_op_reason},",
            f"    {_c_string_arg(component.op_name)},",
            f"    {_c_string_arg(component.same_op_reason_name)},",
            f"    {_c_string_arg(component.assembly_mnemonic)},",
            f"    {_c_string_arg(component.rdna4_assembly_mnemonic)},",
            f"    {component.form},",
            f"    {accumulator_index}, {src0_index}, {vsrc1_index},",
            f"    {component.lane_mask}, {component.pairing_mask},",
            f"    {component.source_register_mask})",
        ]
    )


def _emit_component_rules(tables: _VopdComponentTables) -> str:
    return (
        "\n".join(
            [
                *_generated_header(),
                *(_component_rule_initializer(rule) for rule in tables.rules),
            ]
        )
        + "\n"
    )


def _write_output(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU VOPD component planning table fragments.")
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
    args = parser.parse_args(argv)
    if args.component_rules is None:
        parser.error("at least one output path is required")

    tables = _materialize_vopd_component_tables(
        sorted_descriptor_set_infos(),
        _parse_isa_xml_arguments(args.isa_xml),
    )
    _write_output(args.component_rules, _emit_component_rules(tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
