# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validates checked-in target-info SOPP profiles against vendor ISA XML."""

from __future__ import annotations

from pathlib import Path

from python.runfiles import runfiles

from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaInstructionSet,
    parse_amdgpu_isa_xml_instructions_path,
)
from loom.target.arch.amdgpu.target_info import (
    AmdgpuDescriptorSetIsaInfo,
    AmdgpuSoppOpcodeInfo,
    sorted_descriptor_set_infos,
)

_SOPP_INSTRUCTION_NAMES = (
    "S_BRANCH",
    "S_CBRANCH_SCC0",
    "S_CBRANCH_SCC1",
    "S_DELAY_ALU",
    "S_ENDPGM",
    "S_NOP",
)


def _default_sopp_opcode(
    spec: AmdgpuIsaInstructionSet,
    instruction_name: str,
    *,
    optional: bool = False,
) -> int:
    instruction = spec.instruction_map(include_aliases=False).get(instruction_name)
    if instruction is None:
        if optional:
            return 0
        raise ValueError(f"{spec.source_name}: required SOPP instruction {instruction_name} is absent")
    encodings = tuple(encoding for encoding in instruction.encodings if encoding.encoding_name == "ENC_SOPP" and encoding.condition_name == "default")
    if len(encodings) != 1:
        raise ValueError(f"{spec.source_name}: expected one default ENC_SOPP encoding for {instruction_name}, found {len(encodings)}")
    return encodings[0].opcode


def _vendor_sopp_opcodes(
    isa_info: AmdgpuDescriptorSetIsaInfo,
    xml_path: Path,
) -> AmdgpuSoppOpcodeInfo:
    spec = parse_amdgpu_isa_xml_instructions_path(
        xml_path,
        _SOPP_INSTRUCTION_NAMES,
    )
    if spec.architecture_name != isa_info.isa_architecture_name or spec.architecture_id != isa_info.isa_architecture_id:
        raise ValueError(f"{xml_path}: expected {isa_info.isa_architecture_name} architecture id {isa_info.isa_architecture_id}, found {spec.architecture_name} architecture id {spec.architecture_id}")
    return AmdgpuSoppOpcodeInfo(
        nop=_default_sopp_opcode(spec, "S_NOP"),
        delay_alu=_default_sopp_opcode(spec, "S_DELAY_ALU", optional=True),
        endpgm=_default_sopp_opcode(spec, "S_ENDPGM"),
        branch=_default_sopp_opcode(spec, "S_BRANCH"),
        conditional_branch_scc0=_default_sopp_opcode(spec, "S_CBRANCH_SCC0"),
        conditional_branch_scc1=_default_sopp_opcode(spec, "S_CBRANCH_SCC1"),
    )


def main() -> None:
    runfiles_directory = runfiles.Create()
    if runfiles_directory is None:
        raise RuntimeError("Bazel runfiles are required for the ISA oracle test")

    isa_infos_by_key: dict[str, AmdgpuDescriptorSetIsaInfo] = {}
    for descriptor_set_info in sorted_descriptor_set_infos():
        for isa_info in descriptor_set_info.isa_infos:
            previous_info = isa_infos_by_key.setdefault(
                isa_info.isa_xml_key,
                isa_info,
            )
            if previous_info != isa_info:
                raise ValueError(f"AMDGPU descriptor sets disagree about ISA facts for {isa_info.isa_xml_key}")

    for isa_key, isa_info in sorted(isa_infos_by_key.items()):
        runfile_path = runfiles_directory.Rlocation(f"amdgpu_isa_xml/amdgpu_isa_{isa_key}.xml")
        if runfile_path is None:
            raise RuntimeError(f"ISA XML runfile for {isa_key} is unavailable")
        vendor_opcodes = _vendor_sopp_opcodes(isa_info, Path(runfile_path))
        if vendor_opcodes != isa_info.sopp_opcodes:
            raise ValueError(f"AMDGPU {isa_key} SOPP profile {isa_info.sopp_opcodes} does not match vendor ISA facts {vendor_opcodes}")


if __name__ == "__main__":
    main()
