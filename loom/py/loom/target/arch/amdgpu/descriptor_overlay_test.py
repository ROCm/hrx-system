# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from loom.target.arch.amdgpu.descriptor_overlay import (
    AmdgpuDescriptorOverlay,
    AmdgpuDescriptorOverlayError,
    AmdgpuEncodingFieldAllOnes,
    AmdgpuIgnoredOperandOverlay,
    AmdgpuImplicitOperandOverlay,
    AmdgpuOperandOverlay,
    AmdgpuOperandPredefinedValueRef,
    materialize_amdgpu_descriptor_overlay,
    materialize_amdgpu_descriptor_overlays,
)
from loom.target.arch.amdgpu.encoding import (
    AMDGPU_ENCODING_FORMAT_SOP2,
    AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
    AMDGPU_ENCODING_FORMAT_SOPP,
    AMDGPU_ENCODING_FORMAT_VBUFFER,
    AMDGPU_ENCODING_FORMAT_VOP2_LITERAL,
)
from loom.target.arch.amdgpu.isa_xml import parse_amdgpu_isa_xml_text
from loom.target.arch.amdgpu.isa_xml_test import SAMPLE_XML
from loom.target.low_descriptors import (
    AsmForm,
    AsmImmediate,
    Constraint,
    ConstraintKind,
    DescriptorFlag,
    Immediate,
    ImmediateKind,
    Operand,
    OperandFlag,
    OperandRole,
    RegClassAlt,
)

_REG_SGPR = "amdgpu.sgpr"
_REG_VGPR = "amdgpu.vgpr"

_SGPR_ALT = (RegClassAlt(_REG_SGPR),)
_VGPR_ALT = (RegClassAlt(_REG_VGPR),)

_U32_IMMEDIATE = Immediate(
    "literal",
    ImmediateKind.UNSIGNED,
    bit_width=32,
    unsigned_max=(2**32) - 1,
)

_IGNORE_SCC_OUTPUT = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_SSRC_SPECIAL_SCC",
    data_format_name="FMT_NUM_B1",
    size_bits=1,
    is_input=False,
    is_output=True,
    ignore_reason="value-pseudo-drops-scc",
)

_IGNORE_GLOBAL_READ_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=True,
    is_output=False,
    ignore_reason="modeled-by-global-read-effect",
)

_IGNORE_GLOBAL_WRITE_MEMORY = AmdgpuImplicitOperandOverlay(
    operand_type="OPR_GPUMEM",
    data_format_name="FMT_NUM_B32",
    size_bits=32,
    is_input=False,
    is_output=True,
    ignore_reason="modeled-by-global-write-effect",
)

_BUFFER_ATOMIC_XML = SAMPLE_XML.replace(
    "    </Instructions>",
    """
      <Instruction>
        <InstructionFlags>
          <IsBranch>FALSE</IsBranch>
          <IsConditionalBranch>FALSE</IsConditionalBranch>
          <IsIndirectBranch>FALSE</IsIndirectBranch>
          <IsProgramTerminator>FALSE</IsProgramTerminator>
          <IsImmediatelyExecuted>FALSE</IsImmediatelyExecuted>
        </InstructionFlags>
        <InstructionName>BUFFER_ATOMIC_ADD_U32</InstructionName>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>ENC_VBUFFER</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode Radix="10">53</Opcode>
            <Operands>
              <Operand Input="false" Output="true" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="1">
                <FieldName>VDATA</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_VGPR</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="2">
                <FieldName>VADDR</FieldName>
                <DataFormatName>FMT_ANY</DataFormatName>
                <OperandType>OPR_VGPR</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="3">
                <FieldName>RSRC</FieldName>
                <DataFormatName>FMT_RSRC</DataFormatName>
                <OperandType>OPR_SREG</OperandType>
                <OperandSize>128</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="4">
                <FieldName>SOFFSET</FieldName>
                <DataFormatName>FMT_ANY</DataFormatName>
                <OperandType>OPR_SREG_M0</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="false" Output="true" IsImplicit="true" IsBinaryMicrocodeRequired="true" Order="5">
                <DataFormatName>FMT_NUM_B32</DataFormatName>
                <OperandType>OPR_GPUMEM</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="true" IsBinaryMicrocodeRequired="true" Order="6">
                <DataFormatName>FMT_NUM_B32</DataFormatName>
                <OperandType>OPR_GPUMEM</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
            </Operands>
          </InstructionEncoding>
        </InstructionEncodings>
        <FunctionalGroup>
          <Name>VMEM</Name>
          <FunctionalSubgroups>
            <Subgroup>BUFFER</Subgroup>
            <Subgroup>ATOMIC</Subgroup>
          </FunctionalSubgroups>
        </FunctionalGroup>
      </Instruction>
    </Instructions>""",
)


def _result(field_name: str, reg_alts: tuple[RegClassAlt, ...]) -> Operand:
    return Operand(field_name, OperandRole.RESULT, reg_alts)


def _operand(field_name: str, reg_alts: tuple[RegClassAlt, ...]) -> Operand:
    return Operand(field_name, OperandRole.OPERAND, reg_alts)


def _resource(field_name: str, reg_alts: tuple[RegClassAlt, ...]) -> Operand:
    return Operand(field_name, OperandRole.RESOURCE, reg_alts, unit_count=4)


def _s_add_u32_operands() -> tuple[AmdgpuOperandOverlay, ...]:
    return (
        AmdgpuOperandOverlay("SDST", _result("dst", _SGPR_ALT)),
        AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
        AmdgpuOperandOverlay("SSRC1", _operand("rhs", _SGPR_ALT)),
    )


def _s_add_u32_overlay(
    descriptor_key: str,
    *,
    mnemonic: str | None = None,
    encoding_name: str = "ENC_SOP2",
    operands: tuple[AmdgpuOperandOverlay, ...] | None = None,
    implicit_operands: tuple[AmdgpuImplicitOperandOverlay, ...] = (),
    flags: tuple[DescriptorFlag, ...] = (),
) -> AmdgpuDescriptorOverlay:
    return AmdgpuDescriptorOverlay(
        descriptor_key=descriptor_key,
        instruction_name="S_ADD_U32",
        mnemonic=mnemonic,
        encoding_name=encoding_name,
        semantic_tag="integer.add.u32",
        schedule_class="amdgpu.salu",
        operands=_s_add_u32_operands() if operands is None else operands,
        implicit_operands=implicit_operands,
        flags=flags,
    )


def test_materialize_amdgpu_descriptor_overlays_from_xml_facts() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    descriptors = materialize_amdgpu_descriptor_overlays(
        spec,
        (
            _s_add_u32_overlay(
                "amdgpu.s_add_u32",
                mnemonic="s_add_u32",
                implicit_operands=(_IGNORE_SCC_OUTPUT,),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            AmdgpuDescriptorOverlay(
                descriptor_key="amdgpu.v_add_u32.lit",
                instruction_name="V_ADD_NC_U32",
                mnemonic="v_add_u32",
                encoding_name="VOP2_INST_LITERAL",
                encoding_condition="has_lit",
                semantic_tag="integer.add.u32",
                schedule_class="amdgpu.valu",
                operands=(
                    AmdgpuOperandOverlay("VDST", _result("dst", _VGPR_ALT)),
                    AmdgpuOperandOverlay("VSRC1", _operand("rhs", _VGPR_ALT)),
                ),
                immediate_fields=("LITERAL",),
                immediates=(_U32_IMMEDIATE,),
                flags=(DescriptorFlag.DEAD_REMOVABLE,),
            ),
            AmdgpuDescriptorOverlay(
                descriptor_key="amdgpu.buffer_load_dword",
                instruction_name="BUFFER_LOAD_DWORD",
                mnemonic="buffer_load_dword",
                encoding_name="ENC_VBUFFER",
                semantic_tag="memory.load.u32",
                schedule_class="amdgpu.vmem.load",
                operands=(
                    AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
                    AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
                    AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
                    AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
                ),
                implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
            ),
            AmdgpuDescriptorOverlay(
                descriptor_key="amdgpu.s_wait_idle",
                instruction_name="S_WAIT_IDLE",
                encoding_name="ENC_SOPP",
                semantic_tag="control.waitcnt.idle",
                schedule_class="amdgpu.wait",
                operands=(),
                flags=(DescriptorFlag.SIDE_EFFECTING,),
            ),
        ),
    )

    assert [descriptor.key for descriptor in descriptors] == [
        "amdgpu.s_add_u32",
        "amdgpu.v_add_u32.lit",
        "amdgpu.buffer_load_dword",
        "amdgpu.s_wait_idle",
    ]
    assert descriptors[0].mnemonic == "s_add_u32"
    assert [operand.field_name for operand in descriptors[0].operands] == [
        "dst",
        "lhs",
        "rhs",
    ]
    assert descriptors[0].encoding_format_id == AMDGPU_ENCODING_FORMAT_SOP2
    assert descriptors[1].immediates[0].field_name == _U32_IMMEDIATE.field_name
    assert descriptors[1].immediates[0].encoding_field_id != 0
    assert descriptors[1].encoding_format_id == AMDGPU_ENCODING_FORMAT_VOP2_LITERAL
    assert descriptors[1].encoding_id == 37
    assert descriptors[1].operands[1].field_name == "rhs"
    assert descriptors[2].encoding_format_id == AMDGPU_ENCODING_FORMAT_VBUFFER
    assert descriptors[2].encoding_id == 22
    assert descriptors[2].operands[2].role is OperandRole.RESOURCE
    assert descriptors[3].mnemonic == "s_wait_idle"
    assert descriptors[3].encoding_format_id == AMDGPU_ENCODING_FORMAT_SOPP
    assert descriptors[3].encoding_id == 10


def test_materialize_uses_explicit_asm_forms() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    asm_forms = (
        AsmForm(
            results=("dst",),
            operands=("rhs",),
            immediates=(AsmImmediate("literal"),),
        ),
    )
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.v_add_u32.lit",
        instruction_name="V_ADD_NC_U32",
        mnemonic="v_add_u32",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag="integer.add.u32",
        schedule_class="amdgpu.valu",
        operands=(
            AmdgpuOperandOverlay("VDST", _result("dst", _VGPR_ALT)),
            AmdgpuOperandOverlay("VSRC1", _operand("rhs", _VGPR_ALT)),
        ),
        asm_forms=asm_forms,
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
    )

    descriptor = materialize_amdgpu_descriptor_overlay(spec, overlay)

    assert descriptor.asm_forms == asm_forms
    assert descriptor.asm_forms[0].immediates[0].name is None


def test_fixed_encoding_fields_cover_input_xml_operands() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.buffer_load_dword.off_zero",
            instruction_name="BUFFER_LOAD_DWORD",
            mnemonic="buffer_load_dword",
            encoding_name="ENC_VBUFFER",
            semantic_tag="memory.load.u32",
            schedule_class="amdgpu.vmem.load",
            operands=(
                AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
                AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
            ),
            fixed_encoding_fields=(
                ("VADDR", AmdgpuOperandPredefinedValueRef("v0")),
                ("SOFFSET", AmdgpuOperandPredefinedValueRef("0")),
            ),
            implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        ),
    )

    assert [operand.field_name for operand in descriptor.operands] == [
        "dst",
        "resource",
    ]
    assert [field.value for field in descriptor.encoding_field_values] == [0, 128]
    assert all(
        field.encoding_field_id != 0 for field in descriptor.encoding_field_values
    )


def test_fixed_encoding_fields_resolve_predefined_values() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.buffer_load_dword.off_zero",
            instruction_name="BUFFER_LOAD_DWORD",
            mnemonic="buffer_load_dword",
            encoding_name="ENC_VBUFFER",
            semantic_tag="memory.load.u32",
            schedule_class="amdgpu.vmem.load",
            operands=(
                AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
                AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
            ),
            fixed_encoding_fields=(
                ("VADDR", AmdgpuOperandPredefinedValueRef("v0")),
                ("SOFFSET", AmdgpuOperandPredefinedValueRef("0")),
            ),
            implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        ),
    )

    assert [field.value for field in descriptor.encoding_field_values] == [0, 128]


def test_fixed_encoding_fields_resolve_explicit_predefined_operand_type() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.buffer_load_dword.sbase_zero",
            instruction_name="BUFFER_LOAD_DWORD",
            mnemonic="buffer_load_dword",
            encoding_name="ENC_VBUFFER",
            semantic_tag="memory.load.u32",
            schedule_class="amdgpu.vmem.load",
            operands=(
                AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
                AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
                AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
                AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
            ),
            fixed_encoding_fields=(
                ("SBASE", AmdgpuOperandPredefinedValueRef("v0", "OPR_VGPR")),
            ),
            implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        ),
    )

    assert [field.value for field in descriptor.encoding_field_values] == [0]


def test_fixed_encoding_fields_resolve_all_ones_from_xml_field_width() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.buffer_load_dword.sbase_disabled",
            instruction_name="BUFFER_LOAD_DWORD",
            mnemonic="buffer_load_dword",
            encoding_name="ENC_VBUFFER",
            semantic_tag="memory.load.u32",
            schedule_class="amdgpu.vmem.load",
            operands=(
                AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
                AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
                AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
                AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
            ),
            fixed_encoding_fields=(("SBASE", AmdgpuEncodingFieldAllOnes()),),
            implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        ),
    )

    assert [field.value for field in descriptor.encoding_field_values] == [63]


def test_fixed_encoding_fields_reject_literal_outside_xml_field_width() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_dword.sbase_too_wide",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name="ENC_VBUFFER",
        semantic_tag="memory.load.u32",
        schedule_class="amdgpu.vmem.load",
        operands=(
            AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
            AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
            AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
            AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
        ),
        fixed_encoding_fields=(("SBASE", 64),),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="fixed field 'SBASE' value 64 does not fit in 6 bits",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_fixed_encoding_fields_reject_all_ones_without_xml_field_width() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_load_dword.vaddr_all_ones",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name="ENC_VBUFFER",
        semantic_tag="memory.load.u32",
        schedule_class="amdgpu.vmem.load",
        operands=(
            AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
            AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
            AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
        ),
        fixed_encoding_fields=(("VADDR", AmdgpuEncodingFieldAllOnes()),),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="fixed field 'VADDR' uses all-ones without an XML encoding field",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_missing_instruction() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.missing",
        instruction_name="MISSING",
        encoding_name="ENC_SOP2",
        semantic_tag="missing",
        schedule_class="amdgpu.salu",
        operands=(),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="references unknown AMDGPU instruction 'MISSING'",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_missing_encoding() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay("amdgpu.bad_encoding", encoding_name="ENC_VOP2")

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="could not find encoding 'ENC_VOP2' condition 'default'",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_accepts_exact_duplicate_encoding_rows() -> None:
    encoding_start = SAMPLE_XML.index(
        "          <InstructionEncoding>\n"
        "            <EncodingName>ENC_SOP2</EncodingName>"
    )
    encoding_end = SAMPLE_XML.index(
        "          </InstructionEncoding>", encoding_start
    ) + len("          </InstructionEncoding>\n")
    duplicate_xml = (
        SAMPLE_XML[:encoding_end]
        + SAMPLE_XML[encoding_start:encoding_end]
        + SAMPLE_XML[encoding_end:]
    )
    spec = parse_amdgpu_isa_xml_text(duplicate_xml, source_name="duplicate.xml")

    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        _s_add_u32_overlay(
            "amdgpu.s_add_u32",
            mnemonic="s_add_u32",
            implicit_operands=(_IGNORE_SCC_OUTPUT,),
        ),
    )

    assert descriptor.key == "amdgpu.s_add_u32"
    assert descriptor.encoding_id == 0


def test_materialize_rejects_operand_role_mismatch() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.bad_role",
        operands=(
            AmdgpuOperandOverlay("SDST", _operand("dst", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC1", _operand("rhs", _SGPR_ALT)),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="maps XML field 'SDST' to low operand 'dst' as input",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_operand_width_mismatch() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.bad_width",
        operands=(
            AmdgpuOperandOverlay(
                "SDST",
                Operand("dst", OperandRole.RESULT, _SGPR_ALT, unit_count=2),
            ),
            AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC1", _operand("rhs", _SGPR_ALT)),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="low operand 'dst' with 64-bit low width, but XML operand size is 32",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_accepts_named_operand_width_exception() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        _s_add_u32_overlay(
            "amdgpu.narrowed_width",
            implicit_operands=(_IGNORE_SCC_OUTPUT,),
            operands=(
                AmdgpuOperandOverlay(
                    "SDST",
                    Operand("dst", OperandRole.RESULT, _SGPR_ALT, unit_count=2),
                    size_exception_reason="sample-width-exception",
                ),
                AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
                AmdgpuOperandOverlay("SSRC1", _operand("rhs", _SGPR_ALT)),
            ),
        ),
    )

    assert descriptor.operands[0].unit_count == 2


def test_materialize_rejects_unnecessary_operand_width_exception() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.unnecessary_width_exception",
        operands=(
            AmdgpuOperandOverlay(
                "SDST",
                _result("dst", _SGPR_ALT),
                size_exception_reason="sample-width-exception",
            ),
            AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC1", _operand("rhs", _SGPR_ALT)),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="with an unnecessary size exception",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_accepts_ignored_fixed_binary_operand() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.fixed_vdst",
            instruction_name="BUFFER_LOAD_DWORD",
            mnemonic="buffer_load_dword",
            encoding_name="ENC_VBUFFER",
            semantic_tag="memory.load.fixed_vdst",
            schedule_class="amdgpu.vmem.load",
            operands=(
                AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
                AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
                AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
            ),
            ignored_operands=(
                AmdgpuIgnoredOperandOverlay(
                    "VDATA",
                    ignore_reason="covered-by-target-specific-side-effect",
                    fixed_encoding_value=AmdgpuOperandPredefinedValueRef("v0"),
                ),
            ),
            implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
        ),
    )

    assert [operand.field_name for operand in descriptor.operands] == [
        "vaddr",
        "resource",
        "soffset",
    ]
    assert len(descriptor.encoding_field_values) == 1
    assert descriptor.encoding_field_values[0].value == 0


def test_materialize_rejects_unknown_predefined_value() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.bad_predefined_value",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name="ENC_VBUFFER",
        semantic_tag="memory.load.fixed_vdst",
        schedule_class="amdgpu.vmem.load",
        operands=(
            AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
            AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
            AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
        ),
        fixed_encoding_fields=(("VADDR", AmdgpuOperandPredefinedValueRef("v999")),),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match=r"references AMDGPU predefined value 'OPR_VGPR\.v999'",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_ignored_binary_operand_without_fixed_value() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.ignored_without_fixed_value",
        instruction_name="BUFFER_LOAD_DWORD",
        mnemonic="buffer_load_dword",
        encoding_name="ENC_VBUFFER",
        semantic_tag="memory.load.fixed_vdst",
        schedule_class="amdgpu.vmem.load",
        operands=(
            AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
            AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
            AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
        ),
        ignored_operands=(
            AmdgpuIgnoredOperandOverlay(
                "VDATA",
                ignore_reason="covered-by-target-specific-side-effect",
            ),
        ),
        implicit_operands=(_IGNORE_GLOBAL_READ_MEMORY,),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="ignores binary microcode field 'VDATA' without a fixed encoding value",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_repeated_overlay_xml_field() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.duplicate",
        instruction_name="V_ADD_NC_U32",
        encoding_name="VOP2_INST_LITERAL",
        encoding_condition="has_lit",
        semantic_tag="integer.add.u32",
        schedule_class="amdgpu.valu",
        operands=(
            AmdgpuOperandOverlay("VDST", _result("dst", _VGPR_ALT)),
            AmdgpuOperandOverlay("LITERAL", _operand("lhs", _VGPR_ALT)),
            AmdgpuOperandOverlay("VSRC1", _operand("rhs", _VGPR_ALT)),
        ),
        immediate_fields=("LITERAL",),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="repeats XML field 'LITERAL' across operands and immediates",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_accepts_synthetic_literal_on_literal_encoding_format() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.s_add_u32.rhs_lit",
            instruction_name="S_ADD_U32",
            mnemonic="s_add_u32",
            encoding_name="ENC_SOP2",
            encoding_format_id=AMDGPU_ENCODING_FORMAT_SOP2_LITERAL,
            semantic_tag="integer.add.u32",
            schedule_class="amdgpu.salu",
            operands=(
                AmdgpuOperandOverlay("SDST", _result("dst", _SGPR_ALT)),
                AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
            ),
            immediate_fields=("LITERAL",),
            immediates=(_U32_IMMEDIATE,),
            fixed_encoding_fields=(
                (
                    "SSRC1",
                    AmdgpuOperandPredefinedValueRef("SRC_LITERAL", "OPR_SSRC"),
                ),
            ),
            implicit_operands=(_IGNORE_SCC_OUTPUT,),
            flags=(DescriptorFlag.DEAD_REMOVABLE,),
        ),
    )

    assert descriptor.encoding_format_id == AMDGPU_ENCODING_FORMAT_SOP2_LITERAL
    assert [operand.field_name for operand in descriptor.operands] == ["dst", "lhs"]
    assert [immediate.field_name for immediate in descriptor.immediates] == ["literal"]


def test_materialize_rejects_synthetic_literal_on_plain_encoding_format() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.s_add_u32.rhs_lit",
        instruction_name="S_ADD_U32",
        mnemonic="s_add_u32",
        encoding_name="ENC_SOP2",
        semantic_tag="integer.add.u32",
        schedule_class="amdgpu.salu",
        operands=(
            AmdgpuOperandOverlay("SDST", _result("dst", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
        ),
        immediate_fields=("LITERAL",),
        immediates=(_U32_IMMEDIATE,),
        fixed_encoding_fields=(
            (
                "SSRC1",
                AmdgpuOperandPredefinedValueRef("SRC_LITERAL", "OPR_SSRC"),
            ),
        ),
        flags=(DescriptorFlag.DEAD_REMOVABLE,),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="references missing immediate encoding field 'LITERAL'",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_accepts_tied_buffer_atomic_vdata_overlay() -> None:
    spec = parse_amdgpu_isa_xml_text(_BUFFER_ATOMIC_XML, source_name="sample.xml")
    descriptor = materialize_amdgpu_descriptor_overlay(
        spec,
        AmdgpuDescriptorOverlay(
            descriptor_key="amdgpu.buffer_atomic_add_u32_rtn",
            instruction_name="BUFFER_ATOMIC_ADD_U32",
            mnemonic="buffer_atomic_add_u32",
            encoding_name="ENC_VBUFFER",
            semantic_tag="memory.global.atomic.add.u32.return",
            schedule_class="amdgpu.vmem.atomic.return",
            operands=(
                AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
                AmdgpuOperandOverlay(
                    "VDATA",
                    _operand("value", _VGPR_ALT),
                    role_exception_reason="xml-models-buffer-atomic-vdata-as-output-only",
                ),
                AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
                AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
                AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
            ),
            constraints=(Constraint(ConstraintKind.TIED, 0, 1),),
            implicit_operands=(
                _IGNORE_GLOBAL_WRITE_MEMORY,
                _IGNORE_GLOBAL_READ_MEMORY,
            ),
        ),
    )

    assert [operand.field_name for operand in descriptor.operands] == [
        "dst",
        "value",
        "resource",
        "vaddr",
        "soffset",
    ]
    assert (
        descriptor.operands[0].encoding_field_id
        == descriptor.operands[1].encoding_field_id
    )
    assert descriptor.operands[0].encoding_field_id != 0
    assert descriptor.constraints == (Constraint(ConstraintKind.TIED, 0, 1),)


def test_materialize_rejects_repeated_buffer_atomic_vdata_without_tie() -> None:
    spec = parse_amdgpu_isa_xml_text(_BUFFER_ATOMIC_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_atomic_add_u32_rtn",
        instruction_name="BUFFER_ATOMIC_ADD_U32",
        mnemonic="buffer_atomic_add_u32",
        encoding_name="ENC_VBUFFER",
        semantic_tag="memory.global.atomic.add.u32.return",
        schedule_class="amdgpu.vmem.atomic.return",
        operands=(
            AmdgpuOperandOverlay("VDATA", _result("dst", _VGPR_ALT)),
            AmdgpuOperandOverlay(
                "VDATA",
                _operand("value", _VGPR_ALT),
                role_exception_reason="xml-models-buffer-atomic-vdata-as-output-only",
            ),
            AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
            AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
            AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
        ),
        implicit_operands=(
            _IGNORE_GLOBAL_WRITE_MEMORY,
            _IGNORE_GLOBAL_READ_MEMORY,
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="repeats XML field 'VDATA' without a tied result/input pair",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_output_only_buffer_atomic_input_without_reason() -> None:
    spec = parse_amdgpu_isa_xml_text(_BUFFER_ATOMIC_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.buffer_atomic_add_u32",
        instruction_name="BUFFER_ATOMIC_ADD_U32",
        mnemonic="buffer_atomic_add_u32",
        encoding_name="ENC_VBUFFER",
        semantic_tag="memory.global.atomic.add.u32",
        schedule_class="amdgpu.vmem.atomic.no_return",
        operands=(
            AmdgpuOperandOverlay("VDATA", _operand("value", _VGPR_ALT)),
            AmdgpuOperandOverlay("RSRC", _resource("resource", _SGPR_ALT)),
            AmdgpuOperandOverlay("VADDR", _operand("vaddr", _VGPR_ALT)),
            AmdgpuOperandOverlay("SOFFSET", _operand("soffset", _SGPR_ALT)),
        ),
        implicit_operands=(
            _IGNORE_GLOBAL_WRITE_MEMORY,
            _IGNORE_GLOBAL_READ_MEMORY,
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="maps XML field 'VDATA' to low operand 'value' as input",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_uncovered_explicit_xml_operand() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.incomplete",
        operands=(
            AmdgpuOperandOverlay("SDST", _result("dst", _SGPR_ALT)),
            AmdgpuOperandOverlay("SSRC0", _operand("lhs", _SGPR_ALT)),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="does not cover XML operand field\\(s\\): SSRC1",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_uncovered_implicit_xml_operand() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay("amdgpu.incomplete_implicit")

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="does not cover implicit XML operand\\(s\\): "
        "order=4,type=OPR_SSRC_SPECIAL_SCC",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_implicit_decision_without_xml_operand() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = AmdgpuDescriptorOverlay(
        descriptor_key="amdgpu.extra_implicit",
        instruction_name="S_WAIT_IDLE",
        encoding_name="ENC_SOPP",
        semantic_tag="control.waitcnt.idle",
        schedule_class="amdgpu.wait",
        operands=(),
        implicit_operands=(_IGNORE_SCC_OUTPUT,),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match=r"has implicit operand decision\(s\).*has no implicit operands",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_missing_implicit_xml_operand_reference() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.missing_implicit_reference",
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                operand_type="OPR_MISSING",
                ignore_reason="not-present-in-sample",
            ),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="references missing implicit XML operand 'type=OPR_MISSING'",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_implicit_ignore_without_reason() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.implicit_without_reason",
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                operand_type="OPR_SSRC_SPECIAL_SCC",
                data_format_name="FMT_NUM_B1",
                is_output=True,
            ),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match=r"ignores implicit XML operand .* without a named reason",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_repeated_implicit_xml_operand_decision() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.repeated_implicit",
        implicit_operands=(_IGNORE_SCC_OUTPUT, _IGNORE_SCC_OUTPUT),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="repeats implicit XML operand order=4,type=OPR_SSRC_SPECIAL_SCC",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_maps_implicit_xml_operand_to_low_operand() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    scc_operand = Operand(
        "scc",
        OperandRole.IMPLICIT,
        _SGPR_ALT,
        flags=(OperandFlag.IMPLICIT,),
        unit_count=1,
    )
    overlay = _s_add_u32_overlay(
        "amdgpu.s_add_u32.scc",
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                operand_type="OPR_SSRC_SPECIAL_SCC",
                descriptor_operand=scc_operand,
                data_format_name="FMT_NUM_B1",
                size_bits=1,
                is_input=False,
                is_output=True,
            ),
        ),
    )

    descriptor = materialize_amdgpu_descriptor_overlay(spec, overlay)

    assert descriptor.operands[-1] == scc_operand


def test_materialize_maps_implicit_xml_output_to_result_operand() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    scc_operand = Operand(
        "scc",
        OperandRole.RESULT,
        _SGPR_ALT,
        flags=(OperandFlag.IMPLICIT,),
        unit_count=1,
    )
    overlay = _s_add_u32_overlay(
        "amdgpu.s_add_u32.scc_explicit",
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                operand_type="OPR_SSRC_SPECIAL_SCC",
                descriptor_operand=scc_operand,
                data_format_name="FMT_NUM_B1",
                size_bits=1,
                is_input=False,
                is_output=True,
            ),
        ),
    )

    descriptor = materialize_amdgpu_descriptor_overlay(spec, overlay)

    assert descriptor.operands[1] == scc_operand
    assert descriptor.operands[2].field_name == "lhs"
    assert descriptor.asm_forms[0].operands == ("lhs", "rhs")


def test_materialize_rejects_implicit_xml_output_as_packet_operand() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.bad_output_packet_operand",
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                operand_type="OPR_SSRC_SPECIAL_SCC",
                descriptor_operand=Operand(
                    "scc",
                    OperandRole.RESOURCE,
                    _SGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                data_format_name="FMT_NUM_B1",
            ),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match=r"maps output-only implicit XML operand .* to packet operand",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_implicit_mapping_with_invalid_low_role() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.bad_implicit_mapping_role",
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                operand_type="OPR_SSRC_SPECIAL_SCC",
                descriptor_operand=Operand(
                    "scc",
                    OperandRole.OPERAND_RESULT,
                    _SGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                data_format_name="FMT_NUM_B1",
            ),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="with invalid low role",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_implicit_mapping_with_ignore_reason() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.bad_implicit_mapping_reason",
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                operand_type="OPR_SSRC_SPECIAL_SCC",
                descriptor_operand=Operand(
                    "scc",
                    OperandRole.IMPLICIT,
                    _SGPR_ALT,
                    flags=(OperandFlag.IMPLICIT,),
                ),
                ignore_reason="conflicting-decision",
                data_format_name="FMT_NUM_B1",
            ),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="and also provides an ignore reason",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)


def test_materialize_rejects_implicit_mapping_without_implicit_flag() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")
    overlay = _s_add_u32_overlay(
        "amdgpu.bad_implicit_mapping",
        implicit_operands=(
            AmdgpuImplicitOperandOverlay(
                operand_type="OPR_SSRC_SPECIAL_SCC",
                descriptor_operand=Operand("scc", OperandRole.IMPLICIT, _SGPR_ALT),
                data_format_name="FMT_NUM_B1",
            ),
        ),
    )

    with pytest.raises(
        AmdgpuDescriptorOverlayError,
        match="without implicit flag",
    ):
        materialize_amdgpu_descriptor_overlay(spec, overlay)
