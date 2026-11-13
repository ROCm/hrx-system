// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/function_envelope.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/encoding/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"

namespace loom {
namespace {

// Inspect native operands, independently of envelope provenance or descriptor
// selection. This verifies what executes, including the signed SP displacement.
void ExpectOperand(
    const loom_aie2p_function_envelope_instruction_t& instruction,
    iree_string_view_t form_name, iree_string_view_t operand_name,
    iree_string_view_t register_name, int64_t immediate = 0) {
  const auto form = loom_aie2p_machine_find_form(form_name);
  ASSERT_NE(form, LOOM_AIE2P_MACHINE_FORM_ID_INVALID);
  loom_aie2p_encoding_field_value_t fields[4];
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(loom_aie2p_encoding_unpack_instruction(
      form, instruction.slot.encoded_slot.value, IREE_ARRAYSIZE(fields), fields,
      &count));
  const auto field = loom_aie2p_encoding_find_field(operand_name);
  loom_aie2p_machine_form_info_t form_info;
  ASSERT_TRUE(loom_aie2p_machine_query_form(form, &form_info));
  for (uint8_t i = 0; i < form_info.output_count + form_info.input_count; ++i) {
    loom_aie2p_machine_operand_info_t operand;
    ASSERT_TRUE(loom_aie2p_machine_query_form_operand(form, i, &operand));
    if (!iree_string_view_equal(operand.name, operand_name)) {
      continue;
    }
    for (iree_host_size_t j = 0; j < count; ++j) {
      if (fields[j].field_id != field) {
        continue;
      }
      if (operand.kind == LOOM_AIE2P_MACHINE_OPERAND_KIND_IMMEDIATE) {
        int64_t actual = 0;
        IREE_ASSERT_OK(loom_aie2p_machine_decode_immediate(
            operand.type_id, fields[j].value, &actual));
        EXPECT_EQ(actual, immediate);
      } else {
        const auto physical_register =
            loom_aie2p_machine_find_physical_register(register_name);
        ASSERT_NE(physical_register, LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID);
        uint8_t expected = 0;
        ASSERT_TRUE(loom_aie2p_machine_encode_register(
            operand.kind == LOOM_AIE2P_MACHINE_OPERAND_KIND_REGISTER_ADAPTER
                ? operand.type_id
                : LOOM_AIE2P_REGISTER_ADAPTER_ID_DIRECT,
            physical_register, &expected));
        EXPECT_EQ(fields[j].value, expected);
      }
      return;
    }
  }
  FAIL() << "Expected native operand was not encoded";
}

void ExpectLink(const loom_aie2p_function_envelope_instruction_t& instruction,
                bool save) {
  ExpectOperand(instruction, IREE_SV("MOV_alu_mv_mv_mv_scl"), IREE_SV("dst"),
                save ? IREE_SV("r23") : IREE_SV("lr"));
  ExpectOperand(instruction, IREE_SV("MOV_alu_mv_mv_mv_scl"), IREE_SV("src"),
                save ? IREE_SV("lr") : IREE_SV("r23"));
}

void ExpectAdjustment(
    const loom_aie2p_function_envelope_instruction_t& instruction,
    int64_t displacement) {
  ExpectOperand(instruction, IREE_SV("PADDXM_pstm_sp_imm"), IREE_SV("imm"), {},
                displacement);
}

TEST(Aie2pFunctionEnvelopeTest, LeafDoesNotPreserveUnusedReturnState) {
  loom_aie2p_stack_frame_t stack = {};
  loom_aie2p_function_envelope_t envelope;
  loom_aie2p_function_envelope_build(loom_aie2p_core_descriptor_set(), &stack,
                                     {}, &envelope);
  EXPECT_EQ(envelope.entry.count, 0u);
  EXPECT_EQ(envelope.exit.count, 0u);

  stack.byte_length = 128;
  loom_aie2p_function_envelope_build(loom_aie2p_core_descriptor_set(), &stack,
                                     {}, &envelope);
  ASSERT_EQ(envelope.entry.count, 1u);
  ASSERT_EQ(envelope.exit.count, 1u);
  ExpectAdjustment(envelope.entry.instructions[0], 128);
  ExpectAdjustment(envelope.exit.instructions[0], -128);
}

TEST(Aie2pFunctionEnvelopeTest, ReturningRootOwnsItsShadowWithoutSaveTraffic) {
  const loom_aie2p_stack_frame_t stack = {};
  const loom_aie2p_return_address_t return_address = {
      .kind = LOOM_AIE2P_RETURN_ADDRESS_REGISTER,
      .shadow_register =
          loom_aie2p_machine_find_physical_register(IREE_SV("r23")),
  };
  loom_aie2p_function_envelope_t envelope;
  loom_aie2p_function_envelope_build(loom_aie2p_core_descriptor_set(), &stack,
                                     return_address, &envelope);
  ASSERT_EQ(envelope.entry.count, 1u);
  ASSERT_EQ(envelope.exit.count, 1u);
  ExpectLink(envelope.entry.instructions[0], true);
  ExpectLink(envelope.exit.instructions[0], false);
}

TEST(Aie2pFunctionEnvelopeTest,
     NestedHelperRestoresParentShadowBeforeFrameRelease) {
  const loom_aie2p_return_address_t return_address = {
      .kind = LOOM_AIE2P_RETURN_ADDRESS_FRAME,
      .shadow_register =
          loom_aie2p_machine_find_physical_register(IREE_SV("r23")),
  };
  for (uint64_t frame_bytes : {64, 128, 4096}) {
    SCOPED_TRACE(frame_bytes);
    const loom_aie2p_stack_frame_t stack = {.byte_length = frame_bytes};
    loom_aie2p_function_envelope_t envelope;
    loom_aie2p_function_envelope_build(loom_aie2p_core_descriptor_set(), &stack,
                                       return_address, &envelope);
    ASSERT_EQ(envelope.entry.count, 3u);
    ASSERT_EQ(envelope.exit.count, 3u);
    ExpectAdjustment(envelope.entry.instructions[0], frame_bytes);
    ExpectOperand(envelope.entry.instructions[1], IREE_SV("ST_dms_sts_spill"),
                  IREE_SV("src"), IREE_SV("r23"));
    ExpectOperand(envelope.entry.instructions[1], IREE_SV("ST_dms_sts_spill"),
                  IREE_SV("imm"), {}, -4);
    ExpectLink(envelope.entry.instructions[2], true);
    ExpectLink(envelope.exit.instructions[0], false);
    ExpectOperand(envelope.exit.instructions[1], IREE_SV("LDA_dms_lda_spill"),
                  IREE_SV("dst"), IREE_SV("r23"));
    ExpectOperand(envelope.exit.instructions[1], IREE_SV("LDA_dms_lda_spill"),
                  IREE_SV("imm"), {}, -4);
    ExpectAdjustment(envelope.exit.instructions[2], -(int64_t)frame_bytes);
  }
}

}  // namespace
}  // namespace loom
