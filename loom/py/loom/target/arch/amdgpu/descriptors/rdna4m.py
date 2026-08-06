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


def _rdna4m_supplemental_vgpr_source(
    order: int, field_name: str, data_format_name: str, size_bits: int
) -> AmdgpuIsaOperand:
    return _rdna4m_supplemental_operand(
        order,
        field_name,
        data_format_name,
        "OPR_SRC_VGPR",
        size_bits,
        is_input=True,
        is_output=False,
    )


def _rdna4m_supplemental_inline_accumulator(
    data_format_name: str, size_bits: int
) -> AmdgpuIsaOperand:
    return _rdna4m_supplemental_operand(
        4,
        "SRC2",
        data_format_name,
        "OPR_SRC_VGPR_OR_INLINE",
        size_bits,
        is_input=True,
        is_output=False,
    )


def _rdna4m_supplemental_swmmac_result(
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


_RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS = (
    ("V_MIN_F32", "V_MIN_NUM_F32", 0x00F),
    ("V_MAX_F32", "V_MAX_NUM_F32", 0x010),
    ("V_MIN_F16", "V_MIN_NUM_F16", 0x03A),
    ("V_MAX_F16", "V_MAX_NUM_F16", 0x039),
    ("V_MIN_F64", "V_MIN_NUM_F64", 0x329),
    ("V_MAX_F64", "V_MAX_NUM_F64", 0x32A),
    ("V_PK_MIN_F16", "V_PK_MIN_NUM_F16", 0x011),
    ("V_PK_MAX_F16", "V_PK_MAX_NUM_F16", 0x012),
    ("V_MIN3_F32", "V_MIN3_NUM_F32", 0x219),
    ("V_MAX3_F32", "V_MAX3_NUM_F32", 0x21C),
    ("V_MIN3_F16", "V_MIN3_NUM_F16", 0x249),
    ("V_MAX3_F16", "V_MAX3_NUM_F16", 0x24C),
    ("V_MED3_F32", "V_MED3_NUM_F32", 0x231),
    ("V_MED3_F16", "V_MED3_NUM_F16", 0x232),
    ("V_MINMAX_F32", "V_MINMAX_NUM_F32", 0x25F),
    ("V_MAXMIN_F32", "V_MAXMIN_NUM_F32", 0x25E),
    ("V_MINMAX_F16", "V_MINMAX_NUM_F16", 0x261),
    ("V_MAXMIN_F16", "V_MAXMIN_NUM_F16", 0x260),
)

_RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS = (
    ("V_MIN_F32", "V_MINIMUM_F32", 0x365),
    ("V_MAX_F32", "V_MAXIMUM_F32", 0x366),
    ("V_MIN_F16", "V_MINIMUM_F16", 0x367),
    ("V_MAX_F16", "V_MAXIMUM_F16", 0x368),
    ("V_MIN_F64", "V_MINIMUM_F64", 0x341),
    ("V_MAX_F64", "V_MAXIMUM_F64", 0x342),
    ("V_PK_MIN_F16", "V_PK_MINIMUM_F16", 0x01D),
    ("V_PK_MAX_F16", "V_PK_MAXIMUM_F16", 0x01E),
    ("V_MIN3_F32", "V_MINIMUM3_F32", 0x22D),
    ("V_MAX3_F32", "V_MAXIMUM3_F32", 0x22E),
    ("V_MIN3_F16", "V_MINIMUM3_F16", 0x22F),
    ("V_MAX3_F16", "V_MAXIMUM3_F16", 0x230),
    ("V_MINMAX_F32", "V_MINIMUMMAXIMUM_F32", 0x26C),
    ("V_MAXMIN_F32", "V_MAXIMUMMINIMUM_F32", 0x26D),
    ("V_MINMAX_F16", "V_MINIMUMMAXIMUM_F16", 0x26E),
    ("V_MAXMIN_F16", "V_MAXIMUMMINIMUM_F16", 0x26F),
)


def _rdna4m_retarget_instruction(
    instruction: AmdgpuIsaInstruction,
    instruction_name: str,
    opcode: int,
) -> AmdgpuIsaInstruction:
    return replace(
        instruction,
        name=instruction_name,
        aliases=(),
        encodings=tuple(
            replace(encoding, opcode=opcode) for encoding in instruction.encodings
        ),
    )


def _rdna4m_retarget_vop3_family_instruction(
    instruction: AmdgpuIsaInstruction,
    instruction_name: str,
    opcode: int,
) -> AmdgpuIsaInstruction:
    instruction = replace(
        instruction,
        encodings=tuple(
            encoding
            for encoding in instruction.encodings
            if encoding.encoding_name in ("ENC_VOP3", "ENC_VOP3P")
            or encoding.encoding_name.startswith("VOP3")
        ),
    )
    return _rdna4m_retarget_instruction(instruction, instruction_name, opcode)


def _rdna4m_minmax_instructions(
    spec: AmdgpuIsaFactSource,
) -> tuple[AmdgpuIsaInstruction, ...]:
    instructions = spec.instruction_map()
    return (
        *(
            _rdna4m_retarget_instruction(
                instructions[source_name], instruction_name, opcode
            )
            for source_name, instruction_name, opcode in (
                _RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS
            )
        ),
        *(
            _rdna4m_retarget_vop3_family_instruction(
                instructions[source_name], instruction_name, opcode
            )
            for source_name, instruction_name, opcode in (
                _RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS
            )
        ),
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


_RDNA4M_MATRIX_ELEMENT_BIT_WIDTHS = {
    "F32": 32,
    "I32": 32,
    "F16": 16,
    "BF16": 16,
    "IU8": 8,
    "FP8": 8,
    "BF8": 8,
    "IU4": 4,
}

_RDNA4M_WMMA_ROWS = (
    ("V_WMMA_F32_16X16X16_F16", 0x40, "F32", "F16", "F16", 16),
    ("V_WMMA_F32_16X16X16_BF16", 0x41, "F32", "BF16", "BF16", 16),
    ("V_WMMA_F16_16X16X16_F16", 0x42, "F16", "F16", "F16", 16),
    ("V_WMMA_BF16_16X16X16_BF16", 0x43, "BF16", "BF16", "BF16", 16),
    ("V_WMMA_I32_16X16X16_IU8", 0x44, "I32", "IU8", "IU8", 16),
    ("V_WMMA_I32_16X16X16_IU4", 0x45, "I32", "IU4", "IU4", 16),
    ("V_WMMA_F32_16X16X16_FP8_FP8", 0x46, "F32", "FP8", "FP8", 16),
    ("V_WMMA_F32_16X16X16_FP8_BF8", 0x47, "F32", "FP8", "BF8", 16),
    ("V_WMMA_F32_16X16X16_BF8_FP8", 0x48, "F32", "BF8", "FP8", 16),
    ("V_WMMA_F32_16X16X16_BF8_BF8", 0x49, "F32", "BF8", "BF8", 16),
    ("V_WMMA_I32_16X16X32_IU4", 0x4A, "I32", "IU4", "IU4", 32),
)

_RDNA4M_SWMMAC_ROWS = (
    ("V_SWMMAC_F32_16X16X32_F16", 0x50, "F32", "F16", "F16", 32),
    ("V_SWMMAC_F32_16X16X32_BF16", 0x51, "F32", "BF16", "BF16", 32),
    ("V_SWMMAC_F16_16X16X32_F16", 0x52, "F16", "F16", "F16", 32),
    ("V_SWMMAC_BF16_16X16X32_BF16", 0x53, "BF16", "BF16", "BF16", 32),
    ("V_SWMMAC_I32_16X16X32_IU8", 0x54, "I32", "IU8", "IU8", 32),
    ("V_SWMMAC_I32_16X16X32_IU4", 0x55, "I32", "IU4", "IU4", 32),
    ("V_SWMMAC_I32_16X16X64_IU4", 0x56, "I32", "IU4", "IU4", 64),
    ("V_SWMMAC_F32_16X16X32_FP8_FP8", 0x57, "F32", "FP8", "FP8", 32),
    ("V_SWMMAC_F32_16X16X32_FP8_BF8", 0x58, "F32", "FP8", "BF8", 32),
    ("V_SWMMAC_F32_16X16X32_BF8_FP8", 0x59, "F32", "BF8", "FP8", 32),
    ("V_SWMMAC_F32_16X16X32_BF8_BF8", 0x5A, "F32", "BF8", "BF8", 32),
)


def _rdna4m_matrix_operand_bit_width(inner_dimension: int, element_type: str) -> int:
    return inner_dimension * _RDNA4M_MATRIX_ELEMENT_BIT_WIDTHS[element_type] // 2


def _rdna4m_supplemental_wmma_instruction(
    instruction_name: str,
    opcode: int,
    result_type: str,
    lhs_type: str,
    rhs_type: str,
    inner_dimension: int,
) -> AmdgpuIsaInstruction:
    result_format_name = f"FMT_WMMA_DC_16X16_{result_type}"
    result_size_bits = 8 * _RDNA4M_MATRIX_ELEMENT_BIT_WIDTHS[result_type]
    lhs_size_bits = _rdna4m_matrix_operand_bit_width(inner_dimension, lhs_type)
    rhs_size_bits = _rdna4m_matrix_operand_bit_width(inner_dimension, rhs_type)
    return AmdgpuIsaInstruction(
        name=instruction_name,
        aliases=(),
        flags=_RDNA4M_SUPPLEMENTAL_INSTRUCTION_FLAGS,
        encodings=(
            AmdgpuIsaInstructionEncoding(
                encoding_name="ENC_VOP3P",
                condition_name="default",
                opcode=opcode,
                operands=(
                    _rdna4m_supplemental_result(result_format_name, result_size_bits),
                    _rdna4m_supplemental_vgpr_source(
                        2,
                        "SRC0",
                        f"FMT_WMMA_AB_16X{inner_dimension}_{lhs_type}",
                        lhs_size_bits,
                    ),
                    _rdna4m_supplemental_vgpr_source(
                        3,
                        "SRC1",
                        f"FMT_WMMA_AB_16X{inner_dimension}_{rhs_type}",
                        rhs_size_bits,
                    ),
                    _rdna4m_supplemental_inline_accumulator(
                        result_format_name, result_size_bits
                    ),
                ),
            ),
        ),
        functional_groups=(AmdgpuIsaFunctionalGroup("VALU", ("WMMA",)),),
    )


def _rdna4m_supplemental_swmmac_instruction(
    instruction_name: str,
    opcode: int,
    result_type: str,
    lhs_type: str,
    rhs_type: str,
    inner_dimension: int,
) -> AmdgpuIsaInstruction:
    result_format_name = f"FMT_WMMA_DC_16X16_{result_type}"
    result_size_bits = 8 * _RDNA4M_MATRIX_ELEMENT_BIT_WIDTHS[result_type]
    lhs_size_bits = _rdna4m_matrix_operand_bit_width(inner_dimension // 2, lhs_type)
    rhs_size_bits = _rdna4m_matrix_operand_bit_width(inner_dimension, rhs_type)
    return AmdgpuIsaInstruction(
        name=instruction_name,
        aliases=(),
        flags=_RDNA4M_SUPPLEMENTAL_INSTRUCTION_FLAGS,
        encodings=(
            AmdgpuIsaInstructionEncoding(
                encoding_name="ENC_VOP3P",
                condition_name="default",
                opcode=opcode,
                operands=(
                    _rdna4m_supplemental_swmmac_result(
                        result_format_name, result_size_bits
                    ),
                    _rdna4m_supplemental_vgpr_source(
                        2,
                        "SRC0",
                        f"FMT_WMMA_AB_16X{inner_dimension // 2}_{lhs_type}",
                        lhs_size_bits,
                    ),
                    _rdna4m_supplemental_vgpr_source(
                        3,
                        "SRC1",
                        f"FMT_WMMA_AB_16X{inner_dimension}_{rhs_type}",
                        rhs_size_bits,
                    ),
                    _rdna4m_supplemental_vgpr_source(
                        4, "SRC2", "FMT_WMMA_INDEX_SET", 32
                    ),
                ),
            ),
        ),
        functional_groups=(AmdgpuIsaFunctionalGroup("VALU", ()),),
    )


_RDNA4M_SUPPLEMENTAL_MATRIX_INSTRUCTIONS = (
    *(_rdna4m_supplemental_wmma_instruction(*row) for row in _RDNA4M_WMMA_ROWS),
    *(_rdna4m_supplemental_swmmac_instruction(*row) for row in _RDNA4M_SWMMAC_ROWS),
)

_RDNA4M_SUPPLEMENTAL_INSTRUCTIONS = (
    *_RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS,
    *_RDNA4M_SUPPLEMENTAL_MATRIX_INSTRUCTIONS,
)


def _rdna4m_spec_with_supplemental_instruction_facts(
    spec: AmdgpuIsaFactSource,
) -> AmdgpuIsaFactSource:
    minmax_instructions = _rdna4m_minmax_instructions(spec)
    numeric_minmax_instructions = minmax_instructions[
        : len(_RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS)
    ]
    replacement_instructions = {
        source_name: instruction
        for (source_name, _, _), instruction in zip(
            _RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS,
            numeric_minmax_instructions,
            strict=True,
        )
    }
    supplemental_instructions = {
        instruction.name: instruction
        for instruction in (
            *_RDNA4M_SUPPLEMENTAL_INSTRUCTIONS,
            *minmax_instructions[len(numeric_minmax_instructions) :],
        )
    }
    instructions: list[AmdgpuIsaInstruction] = []
    for instruction in spec.instructions:
        replacement = replacement_instructions.pop(instruction.name, None)
        if replacement is None:
            replacement = supplemental_instructions.pop(instruction.name, instruction)
        instructions.append(replacement)

    if replacement_instructions:
        raise ValueError(
            "RDNA 4m min/max donors are absent from the RDNA 3.5 fact source: "
            + ", ".join(sorted(replacement_instructions))
        )

    # AMD's current ISA archive has no RDNA 4m XML. These narrow facts retarget
    # reused gfx11 physical operand profiles and append the IEEE variants, all
    # proven by LLVM MC for gfx1170-gfx1172.
    return replace(
        spec,
        instructions=(*instructions, *supplemental_instructions.values()),
    )


def _rdna4m_core_descriptor_set(
    descriptor_set: DescriptorSet, key: str
) -> DescriptorSet:
    return _amdgpu_core_descriptor_set(
        key=key,
        reg_classes=descriptor_set.reg_classes,
        register_parts=descriptor_set.register_parts,
        resources=(
            *(
                resource
                for resource in descriptor_set.resources
                if resource.kind is not ResourceKind.MATRIX
            ),
            Resource(
                _RESOURCE_WMMA,
                capacity_per_cycle=1,
                kind=ResourceKind.MATRIX,
                flags=(ResourceFlag.VECTOR_ISSUE,),
            ),
            Resource(
                _RESOURCE_SWMMAC,
                capacity_per_cycle=1,
                kind=ResourceKind.MATRIX,
                flags=(ResourceFlag.VECTOR_ISSUE,),
            ),
        ),
        schedule_classes=(
            *(
                schedule_class
                for schedule_class in descriptor_set.schedule_classes
                if not any(
                    issue_use.resource in (_RESOURCE_WMMA, _RESOURCE_SWMMAC)
                    for issue_use in schedule_class.issue_uses
                )
            ),
            # LLVM assigns gfx1170-gfx1172 the GFX11SpeedModel and leaves the
            # two-address WMMA/SWMMAC pseudos on Write32Bit. That model uses
            # five cycles on the shared VALU/read-control resources. Retain
            # ESTIMATED quality because this is compiler-model evidence, not a
            # hardware-calibrated latency measurement.
            ScheduleClass(
                _SCHEDULE_WMMA,
                latency_kind=LatencyKind.ESTIMATE,
                latency_cycles=5,
                issue_uses=(
                    IssueUse(_RESOURCE_VALU, cycles=1, units=1),
                    IssueUse(_RESOURCE_WMMA, cycles=1, units=1),
                ),
                hazards=_matrix_hazards(_RESOURCE_WMMA),
                model_quality=ModelQuality.ESTIMATED,
            ),
            ScheduleClass(
                _SCHEDULE_SWMMAC,
                latency_kind=LatencyKind.ESTIMATE,
                latency_cycles=5,
                issue_uses=(
                    IssueUse(_RESOURCE_VALU, cycles=1, units=1),
                    IssueUse(_RESOURCE_SWMMAC, cycles=1, units=1),
                ),
                hazards=_matrix_hazards(_RESOURCE_SWMMAC),
                model_quality=ModelQuality.ESTIMATED,
            ),
        ),
    )


_AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE = _rdna4m_core_descriptor_set(
    _AMDGPU_RDNA3_5_CORE_DESCRIPTOR_SET_BASE,
    "amdgpu.rdna4m.core",
)

__all__ = (
    "_AMDGPU_RDNA4M_CORE_DESCRIPTOR_SET_BASE",
    "_RDNA4M_IEEE_MINMAX_INSTRUCTION_ROWS",
    "_RDNA4M_NUMERIC_MINMAX_INSTRUCTION_ROWS",
    "_RDNA4M_SUPPLEMENTAL_FP8_INSTRUCTIONS",
    "_RDNA4M_SUPPLEMENTAL_MATRIX_INSTRUCTIONS",
    "_rdna4m_minmax_instructions",
    "_rdna4m_spec_with_supplemental_instruction_facts",
)
