// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/contract.h"

#include <cstdint>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/ir.h"

namespace {

constexpr uint8_t kTestDialectId = 7;
constexpr uint8_t kLegalOpIndex = 3;

const loom_target_contract_descriptor_rule_t kDescriptorRules[] = {{0}};
const loom_target_contract_fragment_t kContractFragment = {
    .descriptor_rule_count = 1,
    .descriptor_rules = kDescriptorRules,
};
const loom_target_contract_binding_t kBindings[] = {{&kContractFragment, 5, 0}};
const loom_target_contract_case_t kCases[] = {
    {LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE, 0, 0},
};
const loom_target_contract_op_entry_t kOpEntries[] = {
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {0, 1},
};
const loom_target_contract_dialect_table_t kDialects[] = {
    {IREE_ARRAYSIZE(kOpEntries), kOpEntries},
    {0, nullptr},
};
const loom_target_contract_index_t kIndex = {
    kTestDialectId,
    IREE_ARRAYSIZE(kDialects),
    kDialects,
    IREE_ARRAYSIZE(kCases),
    kCases,
    IREE_ARRAYSIZE(kBindings),
    0,
    kBindings,
    nullptr,
    nullptr,
};

TEST(TargetContractQueryEnvironmentTest, MissingAllocatorReturnsNull) {
  loom_target_contract_query_environment_t environment = {};
  int key = 0;
  int stored_data = 0;
  void* data = &stored_data;
  IREE_ASSERT_OK(loom_target_contract_query_get_or_allocate_target_state(
      &environment, &key, sizeof(key), &data));

  EXPECT_EQ(data, nullptr);
}

struct QueryStateAllocatorTestState {
  // Expected callback key.
  const void* key = nullptr;
  // Requested allocation length in bytes.
  iree_host_size_t data_length = 0;
  // Borrowed storage returned by the allocator.
  void* data = nullptr;
  // Number of calls observed by the fixture.
  int call_count = 0;
};

static iree_status_t AllocateQueryStateForTest(void* user_data, const void* key,
                                               iree_host_size_t data_length,
                                               void** out_data) {
  auto* state = reinterpret_cast<QueryStateAllocatorTestState*>(user_data);
  ++state->call_count;
  EXPECT_EQ(key, state->key);
  EXPECT_EQ(data_length, state->data_length);
  *out_data = state->data;
  return iree_ok_status();
}

TEST(TargetContractQueryEnvironmentTest, DelegatesToAllocator) {
  int key = 0;
  int stored_data = 0;
  QueryStateAllocatorTestState state = {
      &key,
      sizeof(stored_data),
      &stored_data,
      0,
  };
  loom_target_contract_query_environment_t environment = {
      .target_state_allocator =
          {
              AllocateQueryStateForTest,
              &state,
          },
  };

  void* data = nullptr;
  IREE_ASSERT_OK(loom_target_contract_query_get_or_allocate_target_state(
      &environment, &key, sizeof(stored_data), &data));

  EXPECT_EQ(data, &stored_data);
  EXPECT_EQ(state.call_count, 1);
}

TEST(TargetContractIndexTest, LookupKindSelectsDescriptorRuleCase) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_index_lookup_kind(
          &kIndex, LOOM_OP_KIND(kTestDialectId, kLegalOpIndex));

  ASSERT_FALSE(loom_target_contract_op_entry_is_empty(entry));
  EXPECT_EQ(entry.case_start, 0);
  EXPECT_EQ(entry.case_count, 1);
  const loom_target_contract_case_t* contract_case =
      &kIndex.cases[entry.case_start];
  EXPECT_EQ(contract_case->system, LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE);
  EXPECT_EQ(contract_case->binding_index, 0);
  ASSERT_NE(contract_case->row_index, LOOM_TARGET_CONTRACT_ROW_NONE);
  const loom_target_contract_binding_t* binding =
      &kIndex.bindings[contract_case->binding_index];
  EXPECT_EQ(binding->rule_set_index, 5);
  const loom_target_contract_descriptor_rule_t* descriptor_rule =
      &binding->fragment->descriptor_rules[contract_case->row_index];
  EXPECT_EQ(descriptor_rule->rule_index, 0);
}

TEST(TargetContractIndexTest, LookupKindIgnoresUncoveredDialectSlot) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_index_lookup_kind(
          &kIndex, LOOM_OP_KIND(kTestDialectId - 1, 0));

  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
  EXPECT_EQ(entry.case_start, LOOM_TARGET_CONTRACT_ROW_NONE);
  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(
      loom_target_contract_index_lookup_kind(
          &kIndex, LOOM_OP_KIND(kTestDialectId + 1, 0))));
  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(
      loom_target_contract_index_lookup_kind(
          &kIndex, LOOM_OP_KIND(kTestDialectId + 2, 0))));
}

TEST(TargetContractIndexTest, LookupKindIgnoresUncoveredOps) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_index_lookup_kind(&kIndex,
                                             LOOM_OP_KIND(kTestDialectId, 1));

  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(
      loom_target_contract_index_lookup_kind(
          &kIndex, LOOM_OP_KIND(kTestDialectId, kLegalOpIndex + 1))));
}

TEST(TargetContractVectorLaneProjectionTest, ProjectsIndependentLaneCounts) {
  const loom_target_contract_vector_lane_projection_t projection = {
      .source_lane_count = 65,
      .projected_lane_count = 32,
      .secondary_source_lane_count = 17,
      .secondary_projected_lane_count = 8,
  };

  EXPECT_EQ(loom_target_contract_query_vector_lane_count(projection, 65), 32u);
  EXPECT_EQ(loom_target_contract_query_vector_lane_count(projection, 17), 8u);
  EXPECT_EQ(loom_target_contract_query_vector_lane_count(projection, 4), 4u);
}

TEST(TargetContractVectorLaneProjectionTest, PrimaryProjectionHasPriority) {
  const loom_target_contract_vector_lane_projection_t projection = {
      .source_lane_count = 17,
      .projected_lane_count = 16,
      .secondary_source_lane_count = 17,
      .secondary_projected_lane_count = 8,
  };

  EXPECT_EQ(loom_target_contract_query_vector_lane_count(projection, 17), 16u);
}

TEST(TargetVectorPacketPolicyTest, RequiresPacketsAtBitOrLaneLimit) {
  const loom_target_vector_packet_lane_limit_t lane_limits[] = {
      {LOOM_SCALAR_TYPE_I16, 32u},
  };
  const loom_target_vector_packet_policy_t policy = {
      .structural_lane_limits = lane_limits,
      .maximum_unpacketized_bit_count = 1024u,
      .structural_lane_limit_count = IREE_ARRAYSIZE(lane_limits),
  };

  EXPECT_FALSE(loom_target_vector_packet_policy_requires_packets(
      &policy,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I16, 32, 0)));
  EXPECT_TRUE(loom_target_vector_packet_policy_requires_packets(
      &policy,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I16, 64, 0)));
  EXPECT_FALSE(loom_target_vector_packet_policy_requires_packets(
      &policy,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, 32, 0)));
  EXPECT_TRUE(loom_target_vector_packet_policy_requires_packets(
      &policy,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, 33, 0)));
}

TEST(TargetVectorPacketPolicyTest, ZeroBudgetPacketsEveryStaticPayload) {
  const loom_target_vector_packet_policy_t policy = {
      .maximum_unpacketized_bit_count = 0u,
  };

  EXPECT_TRUE(loom_target_vector_packet_policy_requires_packets(
      &policy,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, 1, 0)));
  EXPECT_FALSE(loom_target_vector_packet_policy_requires_packets(
      &policy, loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                                   loom_dim_pack_dynamic(1), 0)));
}

TEST(TargetVectorPacketPolicyTest, UsesTargetAddressLaneWidth) {
  const loom_target_vector_packet_policy_t policy = {
      .index_bit_count = 32u,
      .maximum_unpacketized_bit_count = 64u,
  };

  EXPECT_FALSE(loom_target_vector_packet_policy_requires_packets(
      &policy,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_INDEX, 2, 0)));
  EXPECT_TRUE(loom_target_vector_packet_policy_requires_packets(
      &policy,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_INDEX, 3, 0)));
}

}  // namespace
