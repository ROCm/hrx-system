// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/addressability.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"

namespace loom {
namespace {

// clang-format off
static const uint8_t kAddressabilityStrings[] =
    LOOM_BSTRING_LITERAL(0, "")
    LOOM_BSTRING_LITERAL(9, "test.core")
    LOOM_BSTRING_LITERAL(8, "test.gpr")
    LOOM_BSTRING_LITERAL(12, "test.packet")
    LOOM_BSTRING_LITERAL(3, "dst")
    LOOM_BSTRING_LITERAL(3, "src");
// clang-format on

enum {
  ADDRESSABILITY_STRING_empty = 0,
  ADDRESSABILITY_STRING_set_key = ADDRESSABILITY_STRING_empty + sizeof(""),
  ADDRESSABILITY_STRING_reg_gpr =
      ADDRESSABILITY_STRING_set_key + sizeof("test.core"),
  ADDRESSABILITY_STRING_descriptor_packet =
      ADDRESSABILITY_STRING_reg_gpr + sizeof("test.gpr"),
  ADDRESSABILITY_STRING_field_dst =
      ADDRESSABILITY_STRING_descriptor_packet + sizeof("test.packet"),
  ADDRESSABILITY_STRING_field_src =
      ADDRESSABILITY_STRING_field_dst + sizeof("dst"),
  ADDRESSABILITY_STRING_END = ADDRESSABILITY_STRING_field_src + sizeof("src"),
};

static_assert(
    ADDRESSABILITY_STRING_END == sizeof(kAddressabilityStrings) - 1,
    "addressability test string offsets must cover the table payload");

#define ADDRESSABILITY_STRING_OFFSET(field) \
  static_cast<loom_bstring_table_offset_t>(ADDRESSABILITY_STRING_##field)

struct AddressabilityTestState {
  loom_low_reg_class_t reg_classes[1] = {};
  loom_low_reg_class_alt_t reg_class_alts[1] = {};
  loom_low_operand_t operands[2] = {};
  loom_low_descriptor_t descriptors[1] = {};
  loom_low_descriptor_set_t descriptor_set = {};
  loom_module_t module = {};
  loom_op_t function_op = {};
  loom_op_t packet_op = {};
  loom_block_t block = {};
  loom_low_schedule_node_t nodes[1] = {};
  uint32_t scheduled_node_indices[1] = {};
  loom_low_schedule_table_t schedule = {};
  loom_low_allocation_assignment_t assignments[2] = {};
  loom_value_id_t value_ids[2] = {};
  uint32_t assignment_indices_by_value_ordinal[2] = {};
  loom_low_allocation_table_t allocation = {};
};

struct CapturedDiagnostic {
  uint32_t count = 0;
  loom_error_domain_t domain = LOOM_ERROR_DOMAIN_COUNT_;
  uint16_t code = 0;
  uint32_t packet_index = UINT32_MAX;
};

iree_status_t CaptureDiagnostic(void* user_data,
                                const loom_diagnostic_emission_t* emission) {
  CapturedDiagnostic* captured = static_cast<CapturedDiagnostic*>(user_data);
  ++captured->count;
  captured->domain = emission->error->domain;
  captured->code = emission->error->code;
  if (emission->param_count > 5) {
    captured->packet_index = emission->params[5].u32;
  }
  return iree_ok_status();
}

void InitializeAddressabilityTestState(
    loom_low_operand_address_map_kind_t address_map_kind,
    uint16_t addressable_unit_count, uint32_t assigned_base,
    AddressabilityTestState* state, uint32_t assigned_count = 1) {
  *state = {};
  state->descriptor_set.stable_id = 0x1000u;
  state->descriptor_set.key_string_offset =
      ADDRESSABILITY_STRING_OFFSET(set_key);
  state->descriptor_set.string_table.data = kAddressabilityStrings;
  state->descriptor_set.string_table.data_length =
      sizeof(kAddressabilityStrings) - 1;
  state->descriptor_set.reg_classes = state->reg_classes;
  state->descriptor_set.reg_class_count = IREE_ARRAYSIZE(state->reg_classes);
  state->descriptor_set.reg_class_alts = state->reg_class_alts;
  state->descriptor_set.reg_class_alt_count =
      IREE_ARRAYSIZE(state->reg_class_alts);
  state->descriptor_set.operands = state->operands;
  state->descriptor_set.operand_count = IREE_ARRAYSIZE(state->operands);
  state->descriptor_set.descriptors = state->descriptors;
  state->descriptor_set.descriptor_count = IREE_ARRAYSIZE(state->descriptors);

  state->reg_classes[0].name_string_offset =
      ADDRESSABILITY_STRING_OFFSET(reg_gpr);
  state->reg_class_alts[0] = (loom_low_reg_class_alt_t){
      /*.reg_class_id=*/0,
      /*.flags=*/LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED,
  };
  state->operands[0] = (loom_low_operand_t){
      /*.field_name_string_offset=*/ADDRESSABILITY_STRING_OFFSET(field_dst),
      /*.encoding_field_id=*/{},
      /*.role=*/LOOM_LOW_OPERAND_ROLE_RESULT,
      /*.flags=*/{},
      /*.reg_class_alt_start=*/0,
      /*.reg_class_alt_count=*/1,
      /*.unit_count=*/1,
      /*.address_map_kind=*/{},
      /*.addressable_unit_count=*/{},
      /*.data_format_id=*/{},
      /*.register_part_id=*/LOOM_LOW_REGISTER_PART_NONE,
  };
  state->operands[1] = (loom_low_operand_t){
      /*.field_name_string_offset=*/ADDRESSABILITY_STRING_OFFSET(field_src),
      /*.encoding_field_id=*/{},
      /*.role=*/LOOM_LOW_OPERAND_ROLE_OPERAND,
      /*.flags=*/{},
      /*.reg_class_alt_start=*/0,
      /*.reg_class_alt_count=*/1,
      /*.unit_count=*/static_cast<uint16_t>(assigned_count),
      /*.address_map_kind=*/address_map_kind,
      /*.addressable_unit_count=*/addressable_unit_count,
      /*.data_format_id=*/{},
      /*.register_part_id=*/LOOM_LOW_REGISTER_PART_NONE,
  };
  state->descriptors[0] = (loom_low_descriptor_t){
      /*.key_string_offset=*/ADDRESSABILITY_STRING_OFFSET(descriptor_packet),
      /*.stable_id=*/{},
      /*.mnemonic_string_offset=*/{},
      /*.semantic_tag_string_offset=*/{},
      /*.feature_mask_word_start=*/{},
      /*.feature_mask_word_count=*/{},
      /*.encoding_field_value_start=*/{},
      /*.encoding_field_value_count=*/{},
      /*.encoding_format_id=*/{},
      /*.encoding_id=*/{},
      /*.operand_start=*/0,
      /*.operand_count=*/IREE_ARRAYSIZE(state->operands),
      /*.result_count=*/1,
  };

  state->nodes[0].op = &state->packet_op;
  state->nodes[0].block = &state->block;
  state->nodes[0].kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
  state->nodes[0].descriptor = &state->descriptors[0];
  state->nodes[0].operand_count = 1;
  state->nodes[0].result_count = 1;
  loom_value_ordinal_t* value_ordinals =
      loom_low_schedule_node_value_ordinals(&state->nodes[0]);
  value_ordinals[0] = 1;
  value_ordinals[1] = 0;
  state->scheduled_node_indices[0] = 0;
  state->schedule.module = &state->module;
  state->schedule.function_op = &state->function_op;
  state->schedule.target.descriptor_set = &state->descriptor_set;
  state->schedule.nodes = state->nodes;
  state->schedule.node_count = IREE_ARRAYSIZE(state->nodes);
  state->schedule.scheduled_node_indices = state->scheduled_node_indices;
  state->schedule.scheduled_node_count =
      IREE_ARRAYSIZE(state->scheduled_node_indices);

  state->assignments[0] = (loom_low_allocation_assignment_t){
      /*.value_id=*/0,
      /*.value_class=*/
      {
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/{}, /*.register_descriptor_set_stable_id=*/
          state->descriptor_set.stable_id,
          /*.register_class_id=*/0,
      },
      /*.descriptor_reg_class_id=*/0,
      /*.start_point=*/{},
      /*.end_point=*/{},
      /*.unit_count=*/assigned_count,
      /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      /*.location_base=*/0,
      /*.location_count=*/1,
  };
  state->assignments[1] = (loom_low_allocation_assignment_t){
      /*.value_id=*/1,
      /*.value_class=*/
      {
          /*.type_kind=*/LOOM_TYPE_REGISTER,
          /*.element_type=*/{}, /*.register_descriptor_set_stable_id=*/
          state->descriptor_set.stable_id,
          /*.register_class_id=*/0,
      },
      /*.descriptor_reg_class_id=*/0,
      /*.start_point=*/{},
      /*.end_point=*/{},
      /*.unit_count=*/assigned_count,
      /*.location_kind=*/LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      /*.location_base=*/assigned_base,
      /*.location_count=*/assigned_count,
  };
  state->assignment_indices_by_value_ordinal[0] = 0;
  state->assignment_indices_by_value_ordinal[1] = 1;
  state->value_ids[0] = 0;
  state->value_ids[1] = 1;
  state->allocation.module = &state->module;
  state->allocation.function_op = &state->function_op;
  state->allocation.target.descriptor_set = &state->descriptor_set;
  state->allocation.liveness.value_ids = state->value_ids;
  state->allocation.liveness.value_count = IREE_ARRAYSIZE(state->value_ids);
  state->allocation.assignments = state->assignments;
  state->allocation.assignment_count = IREE_ARRAYSIZE(state->assignments);
  state->allocation.assignment_indices_by_value_ordinal =
      state->assignment_indices_by_value_ordinal;
}

TEST(LowAddressabilityTest, AcceptsDirectAddressMap) {
  AddressabilityTestState state;
  InitializeAddressabilityTestState(LOOM_LOW_OPERAND_ADDRESS_MAP_DIRECT,
                                    /*addressable_unit_count=*/0,
                                    /*assigned_base=*/1024, &state);

  CapturedDiagnostic captured;
  const iree_diagnostic_emitter_t emitter = {
      /*.fn=*/CaptureDiagnostic,
      /*.user_data=*/&captured,
  };
  loom_low_addressability_validation_result_t result = {};
  IREE_ASSERT_OK(loom_low_addressability_validate_allocated_packets(
      &state.schedule, &state.allocation, emitter, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_EQ(captured.count, 0u);
}

TEST(LowAddressabilityTest, AcceptsLowSubsetAddressableAssignment) {
  AddressabilityTestState state;
  InitializeAddressabilityTestState(LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET,
                                    /*addressable_unit_count=*/2,
                                    /*assigned_base=*/1, &state);

  loom_low_addressability_validation_result_t result = {};
  IREE_ASSERT_OK(loom_low_addressability_validate_allocated_packets(
      &state.schedule, &state.allocation, /*emitter=*/{}, &result));

  EXPECT_EQ(result.error_count, 0u);
}

TEST(LowAddressabilityTest, ReportsLowSubsetUnaddressableAssignment) {
  AddressabilityTestState state;
  InitializeAddressabilityTestState(LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET,
                                    /*addressable_unit_count=*/2,
                                    /*assigned_base=*/2, &state);

  CapturedDiagnostic captured;
  const iree_diagnostic_emitter_t emitter = {
      /*.fn=*/CaptureDiagnostic,
      /*.user_data=*/&captured,
  };
  loom_low_addressability_validation_result_t result = {};
  IREE_ASSERT_OK(loom_low_addressability_validate_allocated_packets(
      &state.schedule, &state.allocation, emitter, &result));

  EXPECT_EQ(result.error_count, 1u);
  EXPECT_EQ(captured.count, 1u);
  EXPECT_EQ(captured.domain, LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(captured.code, 20u);
  EXPECT_EQ(captured.packet_index, 0u);
}

TEST(LowAddressabilityTest, AcceptsTargetStateAssignmentInsideWindow) {
  AddressabilityTestState state;
  InitializeAddressabilityTestState(LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE,
                                    /*addressable_unit_count=*/256,
                                    /*assigned_base=*/511, &state);

  loom_low_addressability_validation_result_t result = {};
  IREE_ASSERT_OK(loom_low_addressability_validate_allocated_packets(
      &state.schedule, &state.allocation, /*emitter=*/{}, &result));

  EXPECT_EQ(result.error_count, 0u);
}

TEST(LowAddressabilityTest, ReportsTargetStateAssignmentCrossingWindow) {
  AddressabilityTestState state;
  InitializeAddressabilityTestState(LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE,
                                    /*addressable_unit_count=*/256,
                                    /*assigned_base=*/255, &state,
                                    /*assigned_count=*/2);

  CapturedDiagnostic captured;
  const iree_diagnostic_emitter_t emitter = {
      /*.fn=*/CaptureDiagnostic,
      /*.user_data=*/&captured,
  };
  loom_low_addressability_validation_result_t result = {};
  IREE_ASSERT_OK(loom_low_addressability_validate_allocated_packets(
      &state.schedule, &state.allocation, emitter, &result));

  EXPECT_EQ(result.error_count, 1u);
  EXPECT_EQ(captured.count, 1u);
  EXPECT_EQ(captured.domain, LOOM_ERROR_DOMAIN_BACKEND);
  EXPECT_EQ(captured.code, 20u);
  EXPECT_EQ(captured.packet_index, 0u);
}

TEST(LowAddressabilityTest, CountsErrorsWithoutEmitterCallback) {
  AddressabilityTestState state;
  InitializeAddressabilityTestState(LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET,
                                    /*addressable_unit_count=*/2,
                                    /*assigned_base=*/2, &state);

  loom_low_addressability_validation_result_t result = {};
  IREE_ASSERT_OK(loom_low_addressability_validate_allocated_packets(
      &state.schedule, &state.allocation, /*emitter=*/{}, &result));

  EXPECT_EQ(result.error_count, 1u);
}

}  // namespace
}  // namespace loom
