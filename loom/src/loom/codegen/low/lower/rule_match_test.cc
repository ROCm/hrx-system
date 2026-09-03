// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_match.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"

namespace loom {
namespace {

class LowLowerRuleMatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_low_source_workload_register_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("lower_rule_match_test"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_op_t* BuildConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(LowLowerRuleMatchTest, SelectsFirstRuleWhoseGuardsMatch) {
  loom_low_lower_guard_t guards[2] = {};
  guards[0].kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE;
  guards[0].attr_index = 0;
  guards[0].diagnostic_index = 0;
  guards[0].payload.i64_range.minimum = 0;
  guards[0].payload.i64_range.maximum = 3;
  guards[1].kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE;
  guards[1].attr_index = 0;
  guards[1].diagnostic_index = 1;
  guards[1].payload.i64_range.minimum = 4;
  guards[1].payload.i64_range.maximum = 8;
  const loom_low_lower_guard_ref_t guard_refs[] = {0, 1};
  loom_low_lower_rule_t rules[2] = {};
  rules[0].source_op_kind = LOOM_OP_INDEX_CONSTANT;
  rules[0].guard_start = 0;
  rules[0].guard_count = 1;
  rules[1].source_op_kind = LOOM_OP_INDEX_CONSTANT;
  rules[1].guard_start = 1;
  rules[1].guard_count = 1;
  const loom_low_lower_rule_span_t span = {
      /*.source_op_kind=*/LOOM_OP_INDEX_CONSTANT,
      /*.rule_start=*/0,
      /*.rule_count=*/2,
  };
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.spans = &span;
  rule_set.span_count = 1;
  rule_set.rules = rules;
  rule_set.rule_count = IREE_ARRAYSIZE(rules);
  rule_set.guards = guards;
  rule_set.guard_count = IREE_ARRAYSIZE(guards);
  rule_set.guard_refs = guard_refs;
  rule_set.guard_ref_count = IREE_ARRAYSIZE(guard_refs);
  loom_low_lower_rule_match_context_t match_context = {};
  match_context.module = module_;
  const loom_op_t* source_op = BuildConstant(5);

  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, source_op, &selection));

  EXPECT_EQ(selection.rule, &rules[1]);
  EXPECT_EQ(selection.rule_index, 1u);
  EXPECT_TRUE(selection.has_source_op_span);
}

TEST_F(LowLowerRuleMatchTest, ContractQueriesMaySelectContractOnlyRules) {
  loom_low_lower_rule_t rules[2] = {};
  rules[0].source_op_kind = LOOM_OP_INDEX_CONSTANT;
  rules[0].flags = LOOM_LOW_LOWER_RULE_FLAG_CONTRACT_ONLY;
  rules[1].source_op_kind = LOOM_OP_INDEX_CONSTANT;
  const loom_low_lower_rule_span_t span = {
      /*.source_op_kind=*/LOOM_OP_INDEX_CONSTANT,
      /*.rule_start=*/0,
      /*.rule_count=*/2,
  };
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.spans = &span;
  rule_set.span_count = 1;
  rule_set.rules = rules;
  rule_set.rule_count = IREE_ARRAYSIZE(rules);
  loom_low_lower_rule_match_context_t match_context = {};
  match_context.module = module_;
  const loom_op_t* source_op = BuildConstant(5);

  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, source_op, &selection));
  EXPECT_EQ(selection.rule, &rules[1]);

  match_context.flags = LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY;
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, source_op, &selection));
  EXPECT_EQ(selection.rule, &rules[0]);
}

TEST_F(LowLowerRuleMatchTest, SourceRepresentationGuardBindsOnlyFinalLowering) {
  const loom_op_t* source_op = BuildConstant(5);
  constexpr uint64_t kRejectedGroupKey = UINT64_C(0x10);
  constexpr uint64_t kSelectedGroupKey = UINT64_C(0x20);
  loom_low_lower_guard_t guards[2] = {};
  guards[0].kind = LOOM_LOW_LOWER_GUARD_SOURCE_REPRESENTATION_GROUP;
  guards[0].payload.u64 = kRejectedGroupKey;
  guards[1].kind = LOOM_LOW_LOWER_GUARD_SOURCE_REPRESENTATION_GROUP;
  guards[1].payload.u64 = kSelectedGroupKey;
  const loom_low_lower_guard_ref_t guard_refs[] = {0, 1};
  loom_low_lower_rule_t rules[2] = {};
  rules[0].source_op_kind = LOOM_OP_INDEX_CONSTANT;
  rules[0].guard_count = 1;
  rules[1].source_op_kind = LOOM_OP_INDEX_CONSTANT;
  rules[1].guard_start = 1;
  rules[1].guard_count = 1;
  const loom_low_lower_rule_span_t span = {
      /*.source_op_kind=*/LOOM_OP_INDEX_CONSTANT,
      /*.rule_start=*/0,
      /*.rule_count=*/2,
  };
  const loom_low_lower_rule_set_t rule_set = {
      .spans = &span,
      .span_count = 1,
      .rules = rules,
      .rule_count = IREE_ARRAYSIZE(rules),
      .guards = guards,
      .guard_count = IREE_ARRAYSIZE(guards),
      .guard_refs = guard_refs,
      .guard_ref_count = IREE_ARRAYSIZE(guard_refs),
  };

  static const uint8_t kNames[] = LOOM_BSTRING_LITERAL(4, "test");
  const loom_low_source_representation_group_t groups[2] = {
      {.stable_key = kRejectedGroupKey, .name_string_offset = 0},
      {.stable_key = kSelectedGroupKey, .name_string_offset = 0},
  };
  const loom_low_source_representation_candidate_t candidate = {
      .stable_key = UINT64_C(1),
      .name_string_offset = 0,
      .target_data_ordinal =
          LOOM_LOW_SOURCE_REPRESENTATION_TARGET_DATA_ORDINAL_NONE,
  };
  const loom_low_source_representation_provider_t provider = {
      .string_table = {.data = kNames, .data_length = sizeof(kNames) - 1},
      .group_count = IREE_ARRAYSIZE(groups),
      .groups = groups,
      .candidate_count = 1,
      .candidates = &candidate,
  };
  loom_source_program_node_t node = {
      .object = source_op,
      .subtree_limit = 1,
      .kind = LOOM_SOURCE_PROGRAM_NODE_OPERATION,
  };
  const loom_source_program_t program = {
      .nodes = &node,
      .node_count = 1,
  };
  loom_low_source_representation_node_selection_t node_selection = {
      .group_start = 0,
      .group_count = 2,
  };
  loom_low_source_representation_group_selection_t selections[2] = {
      {
          .group_index = 0,
          .candidate_index =
              LOOM_LOW_SOURCE_REPRESENTATION_CANDIDATE_INDEX_NONE,
      },
      {.group_index = 1, .candidate_index = 0},
  };
  uint32_t operation_lookup_slot = 0;
  const loom_low_source_representation_plan_t representation_plan = {
      .provider = &provider,
      .program = &program,
      .node_selections = &node_selection,
      .selected_groups = selections,
      .selected_group_count = IREE_ARRAYSIZE(selections),
      .operation_lookup_slots = &operation_lookup_slot,
      .operation_lookup_slot_count = 1,
  };

  loom_low_lower_rule_match_context_t match_context = {
      .module = module_,
      .source_representation_plan = &representation_plan,
  };
  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, source_op, &selection));
  EXPECT_EQ(selection.rule, &rules[1]);

  match_context.flags = LOOM_LOW_LOWER_RULE_MATCH_FLAG_CONTRACT_ONLY;
  match_context.source_representation_plan = nullptr;
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, source_op, &selection));
  EXPECT_EQ(selection.rule, &rules[0]);
}

}  // namespace
}  // namespace loom
