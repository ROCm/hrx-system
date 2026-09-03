// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verification.h"

#include <cstring>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/vm/bytecode/wire/core.h"

namespace {

struct VerificationContext {
  uint8_t value_register_count;
  uint16_t core_major;
  uint16_t core_minor;
};

bool VerifyLocalRules(const uint8_t* record, uint32_t descriptor,
                      uint8_t rule_count, const VerificationContext& context,
                      uint8_t* out_contextual_rule_count) {
  *out_contextual_rule_count = 0;
  const uint16_t rule_base =
      iree_vm_bytecode_verification_rule_base(descriptor);
  for (uint8_t i = 0; i < rule_count; ++i) {
    const iree_vm_bytecode_verification_rule_t& rule =
        iree_vm_bytecode_verification_rules[rule_base + i];
    const uint8_t* field = record + rule.field_offset;
    switch (rule.kind) {
      case IREE_VM_BYTECODE_VERIFICATION_RULE_ZERO:
        for (uint8_t j = 0; j < rule.field_length; ++j) {
          if (field[j] != 0) return false;
        }
        break;
      case IREE_VM_BYTECODE_VERIFICATION_RULE_REGISTER_VALUE:
        if (field[0] >= context.value_register_count) return false;
        break;
      case IREE_VM_BYTECODE_VERIFICATION_RULE_EXACT_BYTES:
        if (std::memcmp(
                field,
                iree_vm_bytecode_verification_parameters + rule.parameter,
                rule.field_length) != 0) {
          return false;
        }
        break;
      case IREE_VM_BYTECODE_VERIFICATION_RULE_CORE_MAJOR:
        if (iree_unaligned_load_le_u16(field) != context.core_major) {
          return false;
        }
        break;
      case IREE_VM_BYTECODE_VERIFICATION_RULE_CORE_REQUIRED_MINOR:
        if (iree_unaligned_load_le_u16(field) > context.core_minor)
          return false;
        break;
      case IREE_VM_BYTECODE_VERIFICATION_RULE_ALLOWED_RANGE: {
        const uint32_t value = iree_unaligned_load_le_u32(field);
        const uint32_t* range =
            iree_vm_bytecode_verification_parameters + rule.parameter;
        if (value < range[0] || value > range[1]) return false;
        break;
      }
      default:
        ++*out_contextual_rule_count;
        break;
    }
  }
  return true;
}

TEST(VerificationTest, EvaluatesInstructionRulesFromGeneratedPlan) {
  iree_vm_bytecode_control_branch_if_s16_t instruction = {};
  instruction.opcode = IREE_VM_BYTECODE_OPCODE_CONTROL_BRANCH_IF_S16;
  instruction.condition_v8 = 3;
  instruction.target_word_offset_s16 = -2;
  const uint32_t descriptor =
      iree_vm_bytecode_instruction_verification[instruction.opcode];
  EXPECT_EQ(iree_vm_bytecode_verification_byte_length(descriptor),
            sizeof(instruction));
  EXPECT_EQ(iree_vm_bytecode_instruction_verification_control_flow(descriptor),
            IREE_VM_BYTECODE_CONTROL_FLOW_CONDITIONAL_BRANCH);

  uint8_t contextual_rule_count = 0;
  VerificationContext context{/*value_register_count=*/4,
                              /*core_major=*/0, /*core_minor=*/0};
  EXPECT_TRUE(VerifyLocalRules(
      reinterpret_cast<const uint8_t*>(&instruction), descriptor,
      iree_vm_bytecode_instruction_verification_rule_count(descriptor), context,
      &contextual_rule_count));
  EXPECT_EQ(contextual_rule_count, 1);

  instruction.condition_v8 = 4;
  EXPECT_FALSE(VerifyLocalRules(
      reinterpret_cast<const uint8_t*>(&instruction), descriptor,
      iree_vm_bytecode_instruction_verification_rule_count(descriptor), context,
      &contextual_rule_count));
}

TEST(VerificationTest, EvaluatesModuleRulesFromGeneratedPlan) {
  iree_vm_bytecode_v0_image_header_t header = {};
  std::memcpy(header.magic_u8, "IREEVM\0\0", sizeof(header.magic_u8));
  header.core_major_u16 = 0;
  header.core_required_minor_u16 = 0;
  const uint32_t descriptor = iree_vm_bytecode_module_record_verification
      [IREE_VM_BYTECODE_MODULE_RECORD_IMAGE_HEADER];
  EXPECT_EQ(iree_vm_bytecode_verification_byte_length(descriptor),
            sizeof(header));

  uint8_t contextual_rule_count = 0;
  VerificationContext context{/*value_register_count=*/0,
                              /*core_major=*/0, /*core_minor=*/0};
  EXPECT_TRUE(VerifyLocalRules(
      reinterpret_cast<const uint8_t*>(&header), descriptor,
      iree_vm_bytecode_module_verification_rule_count(descriptor), context,
      &contextual_rule_count));
  EXPECT_EQ(contextual_rule_count, 0);

  header.magic_u8[0] = 'X';
  EXPECT_FALSE(VerifyLocalRules(
      reinterpret_cast<const uint8_t*>(&header), descriptor,
      iree_vm_bytecode_module_verification_rule_count(descriptor), context,
      &contextual_rule_count));
}

}  // namespace
