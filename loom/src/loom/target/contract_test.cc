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

const loom_target_contract_fragment_op_span_t kOpSpans[] = {
    {
        LOOM_OP_KIND(kTestDialectId, kLegalOpIndex),
        0,
        1,
    },
};

const loom_target_contract_op_entry_t kOpEntries[] = {
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {0, 1},
};

const loom_target_contract_dialect_table_t kDialectTables[] = {
    {
        0,
        nullptr,
    },
    {
        IREE_ARRAYSIZE(kOpEntries),
        kOpEntries,
    },
};

const loom_target_contract_case_t kFragmentCases[] = {
    {
        LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE,
        0,
        0,
    },
};

const loom_target_contract_descriptor_rule_t kDescriptorRules[] = {
    {
        0,
    },
};

const loom_target_contract_fragment_t kContractFragment = {
    IREE_ARRAYSIZE(kOpSpans),
    IREE_ARRAYSIZE(kFragmentCases),
    IREE_ARRAYSIZE(kDescriptorRules),
    0,
    0,
    kOpSpans,
    kFragmentCases,
    kDescriptorRules,
    nullptr,
};

const loom_target_contract_binding_t kContractBindings[] = {
    {
        &kContractFragment,
        5,
    },
};

const loom_target_contract_case_t kContractCases[] = {
    {
        LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE,
        0,
        0,
    },
};

const loom_target_contract_index_t kContractIndex = {
    kTestDialectId - 1,
    IREE_ARRAYSIZE(kDialectTables),
    IREE_ARRAYSIZE(kContractCases),
    IREE_ARRAYSIZE(kContractBindings),
    kDialectTables,
    kContractCases,
    kContractBindings,
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
  const void* key = nullptr;
  iree_host_size_t data_length = 0;
  void* data = nullptr;
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
  loom_target_contract_query_environment_t environment = {};
  environment.target_state_allocator = {
      AllocateQueryStateForTest,
      &state,
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
          &kContractIndex, LOOM_OP_KIND(kTestDialectId, kLegalOpIndex));

  ASSERT_FALSE(loom_target_contract_op_entry_is_empty(entry));
  EXPECT_EQ(entry.case_start, 0);
  EXPECT_EQ(entry.case_count, 1);
  const loom_target_contract_case_t* contract_case =
      &kContractIndex.cases[entry.case_start];
  EXPECT_EQ(contract_case->system, LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE);
  EXPECT_EQ(contract_case->binding_index, 0);
  ASSERT_NE(contract_case->row_index, LOOM_TARGET_CONTRACT_ROW_NONE);
  const loom_target_contract_binding_t* binding =
      &kContractIndex.bindings[contract_case->binding_index];
  EXPECT_EQ(binding->rule_set_index, 5);
  const loom_target_contract_descriptor_rule_t* descriptor_rule =
      &binding->fragment->descriptor_rules[contract_case->row_index];
  EXPECT_EQ(descriptor_rule->rule_index, 0);
}

TEST(TargetContractIndexTest, LookupKindIgnoresUncoveredDialectSlot) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_index_lookup_kind(
          &kContractIndex, LOOM_OP_KIND(kTestDialectId - 1, 0));

  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
}

TEST(TargetContractIndexTest, LookupKindIgnoresUncoveredOps) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_index_lookup_kind(&kContractIndex,
                                             LOOM_OP_KIND(kTestDialectId, 1));

  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
}

}  // namespace
