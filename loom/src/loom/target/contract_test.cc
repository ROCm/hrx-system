// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/contract.h"

#include <cstdint>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/ir.h"

namespace {

constexpr uint8_t kTestDialectId = 7;
constexpr uint8_t kLegalOpIndex = 3;

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

const loom_target_contract_fragment_case_t kFragmentCases[] = {
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
    kTestDialectId - 1,
    IREE_ARRAYSIZE(kDialectTables),
    0,
    kDialectTables,
    IREE_ARRAYSIZE(kFragmentCases),
    kFragmentCases,
    IREE_ARRAYSIZE(kDescriptorRules),
    kDescriptorRules,
    0,
    nullptr,
};

class TargetContractIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(TargetContractIndexTest, LookupKindSelectsDescriptorRuleCase) {
  const loom_target_contract_binding_t bindings[] = {
      {
          &kContractFragment,
          5,
      },
  };
  loom_target_contract_index_t index = {};
  IREE_ASSERT_OK(loom_target_contract_index_compose(
      bindings, IREE_ARRAYSIZE(bindings), &index, &arena_));

  loom_target_contract_op_entry_t entry =
      loom_target_contract_index_lookup_kind(
          &index, LOOM_OP_KIND(kTestDialectId, kLegalOpIndex));

  ASSERT_FALSE(loom_target_contract_op_entry_is_empty(entry));
  EXPECT_EQ(entry.case_start, 0);
  EXPECT_EQ(entry.case_count, 1);
  const loom_target_contract_case_t* contract_case =
      &index.cases[entry.case_start];
  EXPECT_EQ(contract_case->system, LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE);
  EXPECT_EQ(contract_case->binding_index, 0);
  ASSERT_NE(contract_case->row_index, LOOM_TARGET_CONTRACT_ROW_NONE);
  const loom_target_contract_binding_t* binding =
      &index.bindings[contract_case->binding_index];
  EXPECT_EQ(binding->rule_set_index, 5);
  const loom_target_contract_descriptor_rule_t* descriptor_rule =
      &binding->fragment->descriptor_rules[contract_case->row_index];
  EXPECT_EQ(descriptor_rule->rule_index, 0);
}

TEST_F(TargetContractIndexTest, LookupKindIgnoresUncoveredDialectSlot) {
  const loom_target_contract_binding_t bindings[] = {
      {
          &kContractFragment,
          5,
      },
  };
  loom_target_contract_index_t index = {};
  IREE_ASSERT_OK(loom_target_contract_index_compose(
      bindings, IREE_ARRAYSIZE(bindings), &index, &arena_));

  loom_target_contract_op_entry_t entry =
      loom_target_contract_index_lookup_kind(
          &index, LOOM_OP_KIND(kTestDialectId - 1, 0));

  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
}

TEST_F(TargetContractIndexTest, LookupKindIgnoresUncoveredOps) {
  const loom_target_contract_binding_t bindings[] = {
      {
          &kContractFragment,
          5,
      },
  };
  loom_target_contract_index_t index = {};
  IREE_ASSERT_OK(loom_target_contract_index_compose(
      bindings, IREE_ARRAYSIZE(bindings), &index, &arena_));

  loom_target_contract_op_entry_t entry =
      loom_target_contract_index_lookup_kind(&index,
                                             LOOM_OP_KIND(kTestDialectId, 1));

  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
}

}  // namespace
