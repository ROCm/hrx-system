# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from loom.target.arch.amdgpu.isa_xml import (
    AmdgpuIsaXmlError,
    parse_amdgpu_isa_xml_path,
    parse_amdgpu_isa_xml_text,
)

SAMPLE_XML = """
<Spec>
  <ISA>
    <Architecture>
      <ArchitectureName>AMD RDNA 4</ArchitectureName>
      <ArchitectureId>10</ArchitectureId>
    </Architecture>
    <Encodings>
      <Encoding Order="0">
        <EncodingName>ENC_SOP2</EncodingName>
        <BitCount>32</BitCount>
        <EncodingIdentifierMask Radix="2">11110000</EncodingIdentifierMask>
        <EncodingIdentifiers>
          <EncodingIdentifier Radix="2">10000000</EncodingIdentifier>
        </EncodingIdentifiers>
        <MicrocodeFormat>
          <BitMap>
            <Field IsConditional="false">
              <FieldName>OP</FieldName>
              <BitLayout RangeCount="1">
                <Range Order="0">
                  <BitCount>7</BitCount>
                  <BitOffset>16</BitOffset>
                </Range>
              </BitLayout>
            </Field>
          </BitMap>
        </MicrocodeFormat>
      </Encoding>
      <Encoding Order="1">
        <EncodingName>ENC_VOP2</EncodingName>
        <BitCount>32</BitCount>
        <EncodingIdentifierMask Radix="2">11111100</EncodingIdentifierMask>
        <EncodingIdentifiers>
          <EncodingIdentifier Radix="2">11000000</EncodingIdentifier>
        </EncodingIdentifiers>
        <MicrocodeFormat>
          <BitMap>
            <Field IsConditional="false">
              <FieldName>OPSEL_HI</FieldName>
              <BitLayout RangeCount="2">
                <Range Order="0">
                  <BitCount>2</BitCount>
                  <BitOffset>59</BitOffset>
                </Range>
                <Range Order="1">
                  <BitCount>1</BitCount>
                  <BitOffset>14</BitOffset>
                </Range>
              </BitLayout>
            </Field>
          </BitMap>
        </MicrocodeFormat>
      </Encoding>
      <Encoding Order="2">
        <EncodingName>ENC_SOPP</EncodingName>
        <BitCount>32</BitCount>
        <EncodingIdentifierMask Radix="2">11111111</EncodingIdentifierMask>
        <EncodingIdentifiers>
          <EncodingIdentifier Radix="2">10101010</EncodingIdentifier>
        </EncodingIdentifiers>
        <MicrocodeFormat>
          <BitMap>
            <Field IsConditional="true">
              <FieldName>SIMM16</FieldName>
              <BitLayout RangeCount="1">
                <Range Order="0">
                  <BitCount>16</BitCount>
                  <BitOffset>0</BitOffset>
                </Range>
              </BitLayout>
            </Field>
          </BitMap>
        </MicrocodeFormat>
      </Encoding>
      <Encoding Order="3">
        <EncodingName>ENC_VBUFFER</EncodingName>
        <BitCount>64</BitCount>
        <EncodingIdentifierMask Radix="2">1111111111111111</EncodingIdentifierMask>
        <EncodingIdentifiers>
          <EncodingIdentifier Radix="2">1111000011110000</EncodingIdentifier>
        </EncodingIdentifiers>
        <MicrocodeFormat>
          <BitMap>
            <Field IsConditional="false">
              <FieldName>SBASE</FieldName>
              <BitLayout RangeCount="1">
                <Range Order="0">
                  <BitCount>6</BitCount>
                  <BitOffset>0</BitOffset>
                  <Padding>
                    <BitCount>1</BitCount>
                    <Value Radix="2">0</Value>
                  </Padding>
                </Range>
              </BitLayout>
            </Field>
          </BitMap>
        </MicrocodeFormat>
      </Encoding>
    </Encodings>
    <Instructions>
      <Instruction>
        <InstructionFlags>
          <IsBranch>FALSE</IsBranch>
          <IsConditionalBranch>FALSE</IsConditionalBranch>
          <IsIndirectBranch>FALSE</IsIndirectBranch>
          <IsProgramTerminator>FALSE</IsProgramTerminator>
          <IsImmediatelyExecuted>FALSE</IsImmediatelyExecuted>
        </InstructionFlags>
        <InstructionName>S_ADD_CO_U32</InstructionName>
        <AliasedInstructionNames>
          <InstructionName>S_ADD_U32</InstructionName>
        </AliasedInstructionNames>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>ENC_SOP2</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode Radix="10">0</Opcode>
            <Operands>
              <Operand Input="false" Output="true" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="1">
                <FieldName>SDST</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_SDST</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="2">
                <FieldName>SSRC0</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_SSRC</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="3">
                <FieldName>SSRC1</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_SSRC</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="false" Output="true" IsImplicit="true" IsBinaryMicrocodeRequired="true" Order="4">
                <DataFormatName>FMT_NUM_B1</DataFormatName>
                <OperandType>OPR_SSRC_SPECIAL_SCC</OperandType>
                <OperandSize>1</OperandSize>
              </Operand>
            </Operands>
          </InstructionEncoding>
        </InstructionEncodings>
        <FunctionalGroup>
          <Name>SALU</Name>
          <FunctionalSubgroups>
            <Subgroup>INTEGER</Subgroup>
          </FunctionalSubgroups>
        </FunctionalGroup>
      </Instruction>
      <Instruction>
        <InstructionFlags>
          <IsBranch>FALSE</IsBranch>
          <IsConditionalBranch>FALSE</IsConditionalBranch>
          <IsIndirectBranch>FALSE</IsIndirectBranch>
          <IsProgramTerminator>FALSE</IsProgramTerminator>
          <IsImmediatelyExecuted>FALSE</IsImmediatelyExecuted>
        </InstructionFlags>
        <InstructionName>V_ADD_NC_U32</InstructionName>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>ENC_VOP2</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode Radix="10">37</Opcode>
            <Operands>
              <Operand Input="false" Output="true" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="1">
                <FieldName>VDST</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_VGPR</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="2">
                <FieldName>SRC0</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_SRC</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="3">
                <FieldName>VSRC1</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_VGPR</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
            </Operands>
          </InstructionEncoding>
          <InstructionEncoding>
            <EncodingName>VOP2_INST_LITERAL</EncodingName>
            <EncodingCondition>has_lit</EncodingCondition>
            <Opcode Radix="10">37</Opcode>
            <Operands>
              <Operand Input="false" Output="true" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="1">
                <FieldName>VDST</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_VGPR</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="2">
                <FieldName>LITERAL</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_SRC</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
              <Operand Input="true" Output="false" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="3">
                <FieldName>VSRC1</FieldName>
                <DataFormatName>FMT_NUM_U32</DataFormatName>
                <OperandType>OPR_VGPR</OperandType>
                <OperandSize>32</OperandSize>
              </Operand>
            </Operands>
          </InstructionEncoding>
        </InstructionEncodings>
        <FunctionalGroup>
          <Name>VALU</Name>
          <FunctionalSubgroups>
            <Subgroup>INTEGER</Subgroup>
          </FunctionalSubgroups>
        </FunctionalGroup>
      </Instruction>
      <Instruction>
        <InstructionFlags>
          <IsBranch>FALSE</IsBranch>
          <IsConditionalBranch>FALSE</IsConditionalBranch>
          <IsIndirectBranch>FALSE</IsIndirectBranch>
          <IsProgramTerminator>FALSE</IsProgramTerminator>
          <IsImmediatelyExecuted>TRUE</IsImmediatelyExecuted>
        </InstructionFlags>
        <InstructionName>S_WAIT_IDLE</InstructionName>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>ENC_SOPP</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode Radix="10">10</Opcode>
            <Operands/>
          </InstructionEncoding>
        </InstructionEncodings>
        <FunctionalGroup>
          <Name>WAVE_CONTROL</Name>
          <FunctionalSubgroups>
            <Subgroup>NOT_ASSIGNED</Subgroup>
          </FunctionalSubgroups>
        </FunctionalGroup>
      </Instruction>
      <Instruction>
        <InstructionFlags>
          <IsBranch>FALSE</IsBranch>
          <IsConditionalBranch>FALSE</IsConditionalBranch>
          <IsIndirectBranch>FALSE</IsIndirectBranch>
          <IsProgramTerminator>FALSE</IsProgramTerminator>
          <IsImmediatelyExecuted>FALSE</IsImmediatelyExecuted>
        </InstructionFlags>
        <InstructionName>BUFFER_LOAD_B32</InstructionName>
        <AliasedInstructionNames>
          <InstructionName>BUFFER_LOAD_DWORD</InstructionName>
        </AliasedInstructionNames>
        <InstructionEncodings>
          <InstructionEncoding>
            <EncodingName>ENC_VBUFFER</EncodingName>
            <EncodingCondition>default</EncodingCondition>
            <Opcode Radix="10">22</Opcode>
            <Operands>
              <Operand Input="false" Output="true" IsImplicit="false" IsBinaryMicrocodeRequired="true" Order="1">
                <FieldName>VDATA</FieldName>
                <DataFormatName>FMT_NUM_B32</DataFormatName>
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
              <Operand Input="true" Output="false" IsImplicit="true" IsBinaryMicrocodeRequired="true" Order="5">
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
            <Subgroup>LOAD</Subgroup>
          </FunctionalSubgroups>
        </FunctionalGroup>
      </Instruction>
    </Instructions>
    <OperandTypes>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_SDST</OperandTypeName>
        <OperandPredefinedValues>
          <PredefinedValue>
            <Name>NULL</Name>
            <Value>124</Value>
          </PredefinedValue>
          <PredefinedValue>
            <Name>M0</Name>
            <Value>125</Value>
          </PredefinedValue>
          <PredefinedValue>
            <Name>EXEC_LO</Name>
            <Value>126</Value>
          </PredefinedValue>
        </OperandPredefinedValues>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_SDST_EXEC</OperandTypeName>
        <OperandPredefinedValues>
          <PredefinedValue>
            <Name>EXEC_LO</Name>
            <Value>126</Value>
          </PredefinedValue>
        </OperandPredefinedValues>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_SDST_M0</OperandTypeName>
        <OperandPredefinedValues>
          <PredefinedValue>
            <Name>M0</Name>
            <Value>125</Value>
          </PredefinedValue>
        </OperandPredefinedValues>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_SSRC</OperandTypeName>
        <OperandPredefinedValues>
          <PredefinedValue>
            <Name>EXEC_LO</Name>
            <Value>126</Value>
          </PredefinedValue>
          <PredefinedValue>
            <Name>SRC_LITERAL</Name>
            <Value>255</Value>
          </PredefinedValue>
        </OperandPredefinedValues>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_SRC</OperandTypeName>
        <OperandPredefinedValues>
          <PredefinedValue>
            <Name>SRC_LITERAL</Name>
            <Value>255</Value>
          </PredefinedValue>
        </OperandPredefinedValues>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_SREG_M0</OperandTypeName>
        <OperandPredefinedValues>
          <PredefinedValue>
            <Name>0</Name>
            <Value>128</Value>
          </PredefinedValue>
          <PredefinedValue>
            <Name>M0</Name>
            <Value>125</Value>
          </PredefinedValue>
        </OperandPredefinedValues>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_SREG</OperandTypeName>
        <OperandPredefinedValues/>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_VGPR</OperandTypeName>
        <OperandPredefinedValues>
          <PredefinedValue>
            <Name>v0</Name>
            <Value>0</Value>
          </PredefinedValue>
        </OperandPredefinedValues>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_SSRC_SPECIAL_SCC</OperandTypeName>
        <OperandPredefinedValues/>
      </OperandType>
      <OperandType IsPartitioned="false">
        <OperandTypeName>OPR_GPUMEM</OperandTypeName>
        <OperandPredefinedValues/>
      </OperandType>
    </OperandTypes>
  </ISA>
</Spec>
"""


def test_parse_amdgpu_isa_xml_text_extracts_instruction_facts() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    assert spec.architecture_name == "AMD RDNA 4"
    assert spec.architecture_id == 10
    assert list(spec.encoding_map()) == [
        "ENC_SOP2",
        "ENC_VOP2",
        "ENC_SOPP",
        "ENC_VBUFFER",
    ]
    assert spec.encoding_map()["ENC_SOP2"].fields[0].name == "OP"
    assert spec.encoding_map()["ENC_VOP2"].fields[0].ranges[1].bit_offset == 14
    assert spec.encoding_map()["ENC_SOPP"].fields[0].is_conditional
    assert spec.encoding_map()["ENC_VBUFFER"].fields[0].ranges[0].padding_bit_count == 1
    assert [instruction.name for instruction in spec.instructions] == [
        "BUFFER_LOAD_B32",
        "S_ADD_CO_U32",
        "S_WAIT_IDLE",
        "V_ADD_NC_U32",
    ]
    assert list(spec.operand_type_map()) == [
        "OPR_GPUMEM",
        "OPR_SDST",
        "OPR_SDST_EXEC",
        "OPR_SDST_M0",
        "OPR_SRC",
        "OPR_SREG",
        "OPR_SREG_M0",
        "OPR_SSRC",
        "OPR_SSRC_SPECIAL_SCC",
        "OPR_VGPR",
    ]
    assert spec.operand_predefined_value("OPR_SDST_M0", "M0") == 125
    assert spec.operand_predefined_value("OPR_SSRC", "SRC_LITERAL") == 255

    add_instruction = spec.instruction_map(include_aliases=True)["S_ADD_U32"]
    assert add_instruction.name == "S_ADD_CO_U32"
    assert add_instruction.aliases == ("S_ADD_U32",)
    assert add_instruction.functional_groups[0].name == "SALU"
    assert add_instruction.encodings[0].opcode == 0
    assert add_instruction.encodings[0].operands[0].field_name == "SDST"
    assert add_instruction.encodings[0].operands[3].field_name is None
    assert add_instruction.encodings[0].operands[3].is_implicit

    wait_instruction = spec.instruction_map()["S_WAIT_IDLE"]
    assert wait_instruction.flags.is_immediately_executed
    assert wait_instruction.encodings[0].operands == ()

    buffer_instruction = spec.instruction_map(include_aliases=True)["BUFFER_LOAD_DWORD"]
    assert buffer_instruction.name == "BUFFER_LOAD_B32"
    assert buffer_instruction.functional_groups[0].subgroups == ("BUFFER", "LOAD")
    assert buffer_instruction.encodings[0].operands[2].size_bits == 128


def test_parse_amdgpu_isa_xml_path_uses_explicit_path(tmp_path: Path) -> None:
    xml_path = tmp_path / "amdgpu_isa.xml"
    xml_path.write_text(SAMPLE_XML)

    spec = parse_amdgpu_isa_xml_path(xml_path)

    assert spec.source_name == str(xml_path)
    assert spec.architecture_name == "AMD RDNA 4"


def test_select_instructions_accepts_aliases_and_fails_loudly() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    selected = spec.select_instructions(["BUFFER_LOAD_DWORD", "V_ADD_NC_U32"])

    assert [instruction.name for instruction in selected] == [
        "BUFFER_LOAD_B32",
        "V_ADD_NC_U32",
    ]
    with pytest.raises(
        AmdgpuIsaXmlError,
        match=r"sample\.xml: unknown AMDGPU ISA instruction\(s\): MISSING",
    ):
        spec.select_instructions(["MISSING"])


def test_instruction_encoding_summaries_are_deterministic() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    summaries = spec.instruction_encoding_summaries(
        ["V_ADD_NC_U32", "BUFFER_LOAD_DWORD"]
    )

    assert [
        (summary.instruction_name, summary.encoding_name, summary.condition_name)
        for summary in summaries
    ] == [
        ("BUFFER_LOAD_B32", "ENC_VBUFFER", "default"),
        ("V_ADD_NC_U32", "ENC_VOP2", "default"),
        ("V_ADD_NC_U32", "VOP2_INST_LITERAL", "has_lit"),
    ]
    assert summaries[0].functional_groups == ("VMEM",)
    assert summaries[0].functional_subgroups == ("BUFFER", "LOAD")
    assert summaries[0].explicit_operand_count == 4
    assert summaries[0].implicit_operand_count == 1
    assert summaries[1].operand_types == ("OPR_SRC", "OPR_VGPR")
    assert summaries[2].data_format_names == ("FMT_NUM_U32",)


def test_encoding_field_summaries_include_ranges_and_padding() -> None:
    spec = parse_amdgpu_isa_xml_text(SAMPLE_XML, source_name="sample.xml")

    summaries = spec.encoding_field_summaries(["ENC_VOP2", "ENC_SOPP", "ENC_VBUFFER"])

    assert [
        (summary.encoding_name, summary.field_name, summary.total_bit_count)
        for summary in summaries
    ] == [
        ("ENC_SOPP", "SIMM16", 16),
        ("ENC_VBUFFER", "SBASE", 6),
        ("ENC_VOP2", "OPSEL_HI", 3),
    ]
    assert summaries[0].is_conditional
    assert summaries[1].ranges[0].padding_bit_count == 1
    assert summaries[1].ranges[0].padding_value == 0
    assert summaries[1].total_padding_bit_count == 1
    assert summaries[2].ranges[0].bit_offset == 59
    assert summaries[2].ranges[1].bit_offset == 14


def test_parse_amdgpu_isa_xml_rejects_missing_required_fields() -> None:
    malformed_xml = SAMPLE_XML.replace("<OperandType>OPR_SDST</OperandType>", "", 1)

    with pytest.raises(
        AmdgpuIsaXmlError,
        match="expected <OperandType> under <Operand>",
    ):
        parse_amdgpu_isa_xml_text(malformed_xml, source_name="broken.xml")


def test_parse_amdgpu_isa_xml_rejects_bit_range_count_mismatch() -> None:
    malformed_xml = SAMPLE_XML.replace('RangeCount="2"', 'RangeCount="3"', 1)

    with pytest.raises(
        AmdgpuIsaXmlError,
        match="Field\\(OPSEL_HI\\) declares RangeCount=3 but has 2 ranges",
    ):
        parse_amdgpu_isa_xml_text(malformed_xml, source_name="broken.xml")


def test_parse_amdgpu_isa_xml_rejects_duplicate_instruction_names() -> None:
    duplicate_xml = SAMPLE_XML.replace(
        "<InstructionName>V_ADD_NC_U32</InstructionName>",
        "<InstructionName>S_ADD_CO_U32</InstructionName>",
        1,
    )

    with pytest.raises(
        AmdgpuIsaXmlError,
        match="duplicate AMDGPU ISA instruction 'S_ADD_CO_U32'",
    ):
        parse_amdgpu_isa_xml_text(duplicate_xml, source_name="broken.xml")


def test_operand_predefined_values_accept_exact_duplicates() -> None:
    duplicate_xml = SAMPLE_XML.replace(
        "          <PredefinedValue>\n"
        "            <Name>M0</Name>\n"
        "            <Value>125</Value>\n"
        "          </PredefinedValue>",
        "          <PredefinedValue>\n"
        "            <Name>M0</Name>\n"
        "            <Value>125</Value>\n"
        "          </PredefinedValue>\n"
        "          <PredefinedValue>\n"
        "            <Name>M0</Name>\n"
        "            <Value>125</Value>\n"
        "          </PredefinedValue>",
        1,
    )

    spec = parse_amdgpu_isa_xml_text(duplicate_xml, source_name="duplicate.xml")

    assert spec.operand_predefined_value("OPR_SDST", "M0") == 125


def test_operand_predefined_values_reject_conflicting_duplicates() -> None:
    duplicate_xml = SAMPLE_XML.replace(
        "          <PredefinedValue>\n"
        "            <Name>M0</Name>\n"
        "            <Value>125</Value>\n"
        "          </PredefinedValue>",
        "          <PredefinedValue>\n"
        "            <Name>M0</Name>\n"
        "            <Value>125</Value>\n"
        "          </PredefinedValue>\n"
        "          <PredefinedValue>\n"
        "            <Name>M0</Name>\n"
        "            <Value>126</Value>\n"
        "          </PredefinedValue>",
        1,
    )

    with pytest.raises(
        AmdgpuIsaXmlError,
        match="OperandType\\(OPR_SDST\\) has conflicting predefined value 'M0'",
    ):
        parse_amdgpu_isa_xml_text(duplicate_xml, source_name="broken.xml")
