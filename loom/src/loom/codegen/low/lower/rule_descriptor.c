// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_descriptor.h"

#include "loom/target/registers.h"

const loom_low_operand_t* loom_low_lower_rule_descriptor_result_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index) {
  IREE_ASSERT_LT(result_index, descriptor->result_count);
  IREE_ASSERT((uint64_t)descriptor->operand_start + result_index <
              descriptor_set->operand_count);
  return &descriptor_set->operands[descriptor->operand_start + result_index];
}

iree_status_t loom_low_lower_rule_descriptor_result_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    uint16_t result_index, loom_type_t* out_type) {
  *out_type = loom_type_none();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_operand_t* operand =
      loom_low_lower_rule_descriptor_result_operand(descriptor_set, descriptor,
                                                    result_index);
  IREE_ASSERT_EQ(operand->reg_class_alt_count, 1);
  const uint32_t alt_index = operand->reg_class_alt_start;
  IREE_ASSERT_LT(alt_index, descriptor_set->reg_class_alt_count);
  const loom_low_reg_class_alt_t* alt =
      &descriptor_set->reg_class_alts[alt_index];
  IREE_ASSERT_NE(alt->reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  IREE_ASSERT_FALSE(
      iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE));
  IREE_ASSERT_GT(operand->unit_count, 0);
  return loom_low_lower_make_register_type(context, alt->reg_class_id,
                                           operand->unit_count, out_type);
}

static const loom_low_operand_t* loom_low_lower_rule_descriptor_packet_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t operand_index) {
  const uint32_t operand_end =
      descriptor->operand_start + descriptor->operand_count;
  IREE_ASSERT_LE(operand_end, descriptor_set->operand_count);
  for (uint32_t i = descriptor->operand_start; i < operand_end; ++i) {
    const loom_low_operand_t* operand = &descriptor_set->operands[i];
    if (loom_low_operand_role_is_packet_operand(operand->role) &&
        operand->source_value_index == operand_index) {
      return operand;
    }
  }
  IREE_ASSERT_UNREACHABLE("trusted descriptor packet operand must exist");
  IREE_BUILTIN_UNREACHABLE();
}

iree_status_t loom_low_lower_rule_descriptor_copy_operand_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    uint16_t operand_index, loom_type_t source_type, loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_ASSERT(loom_low_type_is_register(source_type));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_operand_t* operand =
      loom_low_lower_rule_descriptor_packet_operand(descriptor_set, descriptor,
                                                    operand_index);
  IREE_ASSERT_EQ(loom_low_register_type_unit_count(source_type),
                 operand->unit_count);

  const uint16_t source_reg_class_id =
      loom_low_register_type_class_id(source_type);
  uint16_t target_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  uint16_t register_alternative_count = 0;
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    IREE_ASSERT_LT(alt_index, descriptor_set->reg_class_alt_count);
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    ++register_alternative_count;
    target_reg_class_id = alt->reg_class_id;
    if (alt->reg_class_id == source_reg_class_id) {
      *out_type = source_type;
      return iree_ok_status();
    }
  }
  IREE_ASSERT_EQ(register_alternative_count, 1);
  IREE_ASSERT_NE(target_reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  if (loom_type_register_has_value_type(source_type)) {
    return loom_low_lower_make_typed_register_type(
        context, target_reg_class_id, operand->unit_count,
        *loom_type_register_value_type(source_type), out_type);
  }
  return loom_low_lower_make_register_type(context, target_reg_class_id,
                                           operand->unit_count, out_type);
}
