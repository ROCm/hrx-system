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
#include "loom/ops/scalar/ops.h"

namespace loom {
namespace {

TEST(LowLowerRuleSelectionTest, RanksActionableFailuresBeforeDepth) {
  loom_low_lower_rule_selection_t diagnostic_failure = {};
  diagnostic_failure.has_source_op_span = true;
  diagnostic_failure.diagnostic_index = 1;
  diagnostic_failure.matched_guard_count = 1;
  loom_low_lower_rule_selection_t structural_nonmatch = {};
  structural_nonmatch.has_source_op_span = true;
  structural_nonmatch.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  structural_nonmatch.matched_guard_count = 10;
  structural_nonmatch.source_memory_compatible = true;

  EXPECT_TRUE(loom_low_lower_rule_selection_failure_is_better(
      diagnostic_failure, structural_nonmatch));
  EXPECT_FALSE(loom_low_lower_rule_selection_failure_is_better(
      structural_nonmatch, diagnostic_failure));
}

TEST(LowLowerRuleSelectionTest, UsesCompatibilityThenDepthForEqualFailures) {
  loom_low_lower_rule_selection_t shallow_failure = {};
  shallow_failure.has_source_op_span = true;
  shallow_failure.diagnostic_index = 1;
  shallow_failure.matched_guard_count = 1;
  loom_low_lower_rule_selection_t deep_failure = {};
  deep_failure.has_source_op_span = true;
  deep_failure.diagnostic_index = 2;
  deep_failure.matched_guard_count = 2;
  loom_low_lower_rule_selection_t compatible_failure = shallow_failure;
  compatible_failure.source_memory_compatible = true;

  EXPECT_TRUE(loom_low_lower_rule_selection_failure_is_better(deep_failure,
                                                              shallow_failure));
  EXPECT_TRUE(loom_low_lower_rule_selection_failure_is_better(
      compatible_failure, deep_failure));
}

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

  loom_op_t* BuildScalarConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_scalar_constant_build(
        &builder_, loom_attr_i64(value), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildAdd(loom_value_id_t lhs, loom_value_id_t rhs) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_scalar_addi_build(
        &builder_, /*overflow_flags=*/0, lhs, rhs,
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildMultiply(loom_value_id_t lhs, loom_value_id_t rhs) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_scalar_muli_build(
        &builder_, /*overflow_flags=*/0, lhs, rhs,
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  struct SourceGraphSelection {
    const loom_op_t* source_nodes[2] = {};
    const loom_op_t* diagnostic_source_op = nullptr;
    uint8_t source_node_count = 0;
    bool selected = false;
    bool has_source_op_span = false;
  };

  SourceGraphSelection SelectSourceGraph(
      const loom_op_t* source_op, loom_op_kind_t related_op_kind,
      loom_low_lower_source_node_relation_t relation,
      bool reject_related_guard = false) {
    loom_low_lower_value_ref_t value_refs[2] = {};
    loom_low_lower_source_node_t source_node = {};
    source_node.relation = relation;
    source_node.source_op_kind = related_op_kind;
    source_node.parent_node_index = 0;
    source_node.parent_value_ref_index = 0;
    source_node.node_value_ref_index = 1;
    if (relation == LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_UNIQUE_USER) {
      value_refs[0].kind = LOOM_LOW_LOWER_VALUE_REF_RESULT;
      value_refs[0].source_node_index = 0;
      value_refs[1].kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND;
      value_refs[1].source_node_index = 1;
    } else {
      value_refs[0].kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND;
      value_refs[0].source_node_index = 0;
      value_refs[1].kind = LOOM_LOW_LOWER_VALUE_REF_RESULT;
      value_refs[1].source_node_index = 1;
    }

    loom_low_lower_guard_t guard = {};
    const loom_low_lower_guard_ref_t guard_ref = 0;
    if (reject_related_guard) {
      guard.kind = LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL;
      guard.diagnostic_index = 0;
      guard.payload.u64 = UINT64_MAX;
      source_node.guard_count = 1;
    }
    loom_low_lower_rule_t rule = {};
    rule.source_op_kind = source_op->kind;
    rule.source_node_span = LOOM_LOW_LOWER_SOURCE_NODE_SPAN(0, 1);
    const loom_low_lower_rule_span_t span = {
        /*.source_op_kind=*/source_op->kind,
        /*.rule_start=*/0,
        /*.rule_count=*/1,
    };
    loom_low_lower_rule_set_t rule_set = {};
    rule_set.spans = &span;
    rule_set.span_count = 1;
    rule_set.rules = &rule;
    rule_set.rule_count = 1;
    rule_set.value_refs = value_refs;
    rule_set.value_ref_count = IREE_ARRAYSIZE(value_refs);
    rule_set.source_nodes = &source_node;
    rule_set.source_node_count = 1;
    if (reject_related_guard) {
      rule_set.guards = &guard;
      rule_set.guard_count = 1;
      rule_set.guard_refs = &guard_ref;
      rule_set.guard_ref_count = 1;
    }
    loom_low_lower_rule_match_context_t match_context = {};
    match_context.module = module_;

    loom_low_lower_rule_selection_t selection = {};
    IREE_EXPECT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &rule_set, source_op, &selection));
    SourceGraphSelection result = {
        /*.source_nodes=*/{selection.source_nodes[0],
                           selection.source_nodes[1]},
        /*.diagnostic_source_op=*/selection.diagnostic_source_op,
        /*.source_node_count=*/selection.source_node_count,
        /*.selected=*/selection.rule != nullptr,
        /*.has_source_op_span=*/selection.has_source_op_span,
    };
    return result;
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

TEST_F(LowLowerRuleMatchTest, SelectsAdjacentUniqueUserSourceNode) {
  const loom_value_id_t lhs =
      loom_scalar_constant_result(BuildScalarConstant(2));
  const loom_value_id_t rhs =
      loom_scalar_constant_result(BuildScalarConstant(3));
  const loom_value_id_t accumulator =
      loom_scalar_constant_result(BuildScalarConstant(5));
  const loom_op_t* producer = BuildAdd(lhs, rhs);
  const loom_op_t* consumer =
      BuildMultiply(loom_scalar_addi_result(producer), accumulator);

  const SourceGraphSelection selection =
      SelectSourceGraph(producer, LOOM_OP_SCALAR_MULI,
                        LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_UNIQUE_USER);

  EXPECT_TRUE(selection.selected);
  ASSERT_EQ(selection.source_node_count, 2u);
  EXPECT_EQ(selection.source_nodes[0], producer);
  EXPECT_EQ(selection.source_nodes[1], consumer);
}

TEST_F(LowLowerRuleMatchTest, SelectsAdjacentDefinitionSourceNode) {
  const loom_value_id_t lhs =
      loom_scalar_constant_result(BuildScalarConstant(2));
  const loom_value_id_t rhs =
      loom_scalar_constant_result(BuildScalarConstant(3));
  const loom_value_id_t accumulator =
      loom_scalar_constant_result(BuildScalarConstant(5));
  const loom_op_t* producer = BuildAdd(lhs, rhs);
  const loom_op_t* consumer =
      BuildMultiply(loom_scalar_addi_result(producer), accumulator);

  const SourceGraphSelection selection =
      SelectSourceGraph(consumer, LOOM_OP_SCALAR_ADDI,
                        LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_DEFINITION);

  EXPECT_TRUE(selection.selected);
  ASSERT_EQ(selection.source_node_count, 2u);
  EXPECT_EQ(selection.source_nodes[0], consumer);
  EXPECT_EQ(selection.source_nodes[1], producer);
}

TEST_F(LowLowerRuleMatchTest, RejectsNonAdjacentSourceNode) {
  const loom_value_id_t lhs =
      loom_scalar_constant_result(BuildScalarConstant(2));
  const loom_value_id_t rhs =
      loom_scalar_constant_result(BuildScalarConstant(3));
  const loom_value_id_t accumulator =
      loom_scalar_constant_result(BuildScalarConstant(5));
  const loom_op_t* producer = BuildAdd(lhs, rhs);
  BuildAdd(lhs, rhs);
  BuildMultiply(loom_scalar_addi_result(producer), accumulator);

  const SourceGraphSelection selection =
      SelectSourceGraph(producer, LOOM_OP_SCALAR_MULI,
                        LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_UNIQUE_USER);

  EXPECT_FALSE(selection.selected);
  EXPECT_TRUE(selection.has_source_op_span);
  EXPECT_EQ(selection.source_node_count, 0u);
}

TEST_F(LowLowerRuleMatchTest, RejectsSourceConnectionWithMultipleUsers) {
  const loom_value_id_t lhs =
      loom_scalar_constant_result(BuildScalarConstant(2));
  const loom_value_id_t rhs =
      loom_scalar_constant_result(BuildScalarConstant(3));
  const loom_value_id_t accumulator =
      loom_scalar_constant_result(BuildScalarConstant(5));
  const loom_op_t* producer = BuildAdd(lhs, rhs);
  BuildMultiply(loom_scalar_addi_result(producer), accumulator);
  BuildMultiply(loom_scalar_addi_result(producer), accumulator);

  const SourceGraphSelection selection =
      SelectSourceGraph(producer, LOOM_OP_SCALAR_MULI,
                        LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_UNIQUE_USER);

  EXPECT_FALSE(selection.selected);
  EXPECT_TRUE(selection.has_source_op_span);
}

TEST_F(LowLowerRuleMatchTest, AttributesRelatedGuardDiagnosticToSourceNode) {
  const loom_value_id_t lhs =
      loom_scalar_constant_result(BuildScalarConstant(2));
  const loom_value_id_t rhs =
      loom_scalar_constant_result(BuildScalarConstant(3));
  const loom_value_id_t accumulator =
      loom_scalar_constant_result(BuildScalarConstant(5));
  const loom_op_t* producer = BuildAdd(lhs, rhs);
  const loom_op_t* consumer =
      BuildMultiply(loom_scalar_addi_result(producer), accumulator);

  const SourceGraphSelection selection =
      SelectSourceGraph(producer, LOOM_OP_SCALAR_MULI,
                        LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_UNIQUE_USER,
                        /*reject_related_guard=*/true);

  EXPECT_FALSE(selection.selected);
  EXPECT_EQ(selection.diagnostic_source_op, consumer);
}

}  // namespace
}  // namespace loom
