# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# ruff: noqa: F403, F405

"""AMDGPU RDNA 4m descriptor-set base data."""

from __future__ import annotations

from dataclasses import replace

from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaFactSource,
    AmdgpuIsaFunctionalGroup,
    AmdgpuIsaInstruction,
    AmdgpuIsaInstructionEncoding,
    AmdgpuIsaInstructionFlags,
    AmdgpuIsaOperand,
)

from .common import *
from .rdna3 import _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE

_RDNA4M_SUPPLEMENTAL_INSTRUCTION_FLAGS = AmdgpuIsaInstructionFlags(
    is_branch=False,
    is_conditional_branch=False,
    is_indirect_branch=False,
    is_program_terminator=False,
    is_immediately_executed=False,
)


def _rdna4m_supplemental_operand(
    order: int,
    field_name: str,
    data_format_name: str,
    operand_type: str,
    size_bits: int,
    *,
    is_input: bool,
    is_output: bool,
) -> AmdgpuIsaOperand:
    return AmdgpuIsaOperand(
        order=order,
        field_name=field_name,
        data_format_name=data_format_name,
        operand_type=operand_type,
        size_bits=size_bits,
        is_input=is_input,
        is_output=is_output,
        is_implicit=False,
        is_binary_microcode_required=True,
    )


def _rdna4m_supplemental_result(
    data_format_name: str, size_bits: int
) -> AmdgpuIsaOperand:
    return _rdna4m_supplemental_operand(
        1,
        "VDST",
        data_format_name,
        "OPR_VGPR",
        size_bits,
        is_input=False,
        is_output=True,
    )


def _rdna4m_supplemental_source(
    order: int, field_name: str, data_format_name: str, size_bits: int
) -> AmdgpuIsaOperand:
    return _rdna4m_supplemental_operand(
        order,
        field_name,
        data_format_name,
        "OPR_SRC",
        size_bits,
        is_input=True,
        is_output=False,
    )


def _rdna4m_supplemental_instruction(
    name: str,
    encodings: tuple[AmdgpuIsaInstructionEncoding, ...],
) -> AmdgpuIsaInstruction:
    return AmdgpuIsaInstruction(
        name=name,
        aliases=(),
        flags=_RDNA4M_SUPPLEMENTAL_INSTRUCTION_FLAGS,
        encodings=encodings,
        functional_groups=(AmdgpuIsaFunctionalGroup("VALU", ("NOT_ASSIGNED",)),),
    )


_RDNA4M_FP8_DECODE_ROWS = (
    ("V_CVT_F32_FP8", 0x6C, 0x1EC, "FMT_NUM_FP8", 8, "FMT_NUM_F32", 32),
    ("V_CVT_F32_BF8", 0x6D, 0x1ED, "FMT_NUM_BF8", 8, "FMT_NUM_F32", 32),
    (
        "V_CVT_PK_F32_FP8",
        0x6E,
        0x1EE,
        "FMT_NUM_PK2_FP8",
        16,
        "FMT_NUM_PK2_F32",
        64,
    ),
    (
        "V_CVT_PK_F32_BF8",
        0x6F,
        0x1EF,
        "FMT_NUM_PK2_BF8",
        16,
        "FMT_NUM_PK2_F32",
        64,
    ),
)


def _rdna4m_fp8_decode_instruction(
    instruction_name: str,
    vop1_opcode: int,
    vop3_opcode: int,
    source_format_name: str,
    source_size_bits: int,
    result_format_name: str,
    result_size_bits: int,
) -> AmdgpuIsaInstruction:
    operands = (
        _rdna4m_supplemental_result(result_format_name, result_size_bits),
        _rdna4m_supplemental_source(2, "SRC0", source_format_name, source_size_bits),
    )
    return _rdna4m_supplemental_instruction(
        instruction_name,
        (
            AmdgpuIsaInstructionEncoding(
                encoding_name="ENC_VOP1",
                condition_name="default",
                opcode=vop1_opcode,
                operands=operands,
            ),
            AmdgpuIsaInstructionEncoding(
                encoding_name="ENC_VOP3",
                condition_name="default",
                opcode=vop3_opcode,
                operands=operands,
            ),
        ),
    )


_RDNA4M_FP8_ENCODE_ROWS = (
    ("FP8", 0x369, "FMT_NUM_PK2_FP8"),
    ("BF8", 0x36A, "FMT_NUM_PK2_BF8"),
)


def _rdna4m_fp8_encode_instruction(
    target_type: str, opcode: int, result_format_name: str
) -> AmdgpuIsaInstruction:
    return _rdna4m_supplemental_instruction(
        f"V_CVT_PK_{target_type}_F32",
        (
            AmdgpuIsaInstructionEncoding(
                encoding_name="ENC_VOP3",
                condition_name="default",
                opcode=opcode,
                operands=(
                    _rdna4m_supplemental_result(result_format_name, 16),
                    _rdna4m_supplemental_source(2, "SRC0", "FMT_NUM_F32", 32),
                    _rdna4m_supplemental_source(3, "SRC1", "FMT_NUM_F32", 32),
                ),
            ),
        ),
    )


_RDNA4M_FP8_DOT_ROWS = (
    ("FP8", "BF8", 0x24),
    ("BF8", "FP8", 0x25),
    ("FP8", "FP8", 0x26),
    ("BF8", "BF8", 0x27),
)


def _rdna4m_fp8_dot_instruction(
    lhs_type: str, rhs_type: str, opcode: int
) -> AmdgpuIsaInstruction:
    return _rdna4m_supplemental_instruction(
        f"V_DOT4_F32_{lhs_type}_{rhs_type}",
        (
            AmdgpuIsaInstructionEncoding(
                encoding_name="ENC_VOP3P",
                condition_name="default",
                opcode=opcode,
                operands=(
                    _rdna4m_supplemental_result("FMT_NUM_F32", 32),
                    _rdna4m_supplemental_source(
                        2, "SRC0", f"FMT_NUM_PK4_{lhs_type}", 32
                    ),
                    _rdna4m_supplemental_source(
                        3, "SRC1", f"FMT_NUM_PK4_{rhs_type}", 32
                    ),
                    _rdna4m_supplemental_source(4, "SRC2", "FMT_NUM_F32", 32),
                ),
            ),
        ),
    )


_RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS = (
    *(_rdna4m_fp8_decode_instruction(*row) for row in _RDNA4M_FP8_DECODE_ROWS),
    *(_rdna4m_fp8_encode_instruction(*row) for row in _RDNA4M_FP8_ENCODE_ROWS),
    *(_rdna4m_fp8_dot_instruction(*row) for row in _RDNA4M_FP8_DOT_ROWS),
)


def _rdna4m_spec_with_supplemental_instruction_facts(
    spec: AmdgpuIsaFactSource,
) -> AmdgpuIsaFactSource:
    existing_instruction_names = spec.instruction_map(include_aliases=True)
    supplemental_instructions = tuple(
        instruction
        for instruction in _RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS
        if instruction.name not in existing_instruction_names
    )
    if not supplemental_instructions:
        return spec

    # AMD's current ISA archive has no RDNA 4m XML. These narrow facts mirror
    # instructions and encodings proven by LLVM MC for gfx1170-gfx1172.
    return replace(spec, instructions=(*spec.instructions, *supplemental_instructions))


def _without_matrix_schedule_rows(
    descriptor_set: DescriptorSet, key: str
) -> DescriptorSet:
    return _amdgpu_core_descriptor_set(
        key=key,
        reg_classes=descriptor_set.reg_classes,
        register_parts=descriptor_set.register_parts,
        resources=tuple(
            resource
            for resource in descriptor_set.resources
            if resource.kind is not ResourceKind.MATRIX
        ),
        schedule_classes=tuple(
            schedule_class
            for schedule_class in descriptor_set.schedule_classes
            if not any(
                issue_use.resource == _RESOURCE_WMMA
                for issue_use in schedule_class.issue_uses
            )
        ),
    )


_AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE = _without_matrix_schedule_rows(
    _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
    "amdgpu.rdna4m.core",
)

__all__ = (
    "_AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE",
    "_RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS",
    "_rdna4m_spec_with_supplemental_instruction_facts",
)
