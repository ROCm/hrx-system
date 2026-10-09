// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_match.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/consumption.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

TEST(LowLowerSourceMemoryMatchTest, SelectsExactRejectionReason) {
  loom_low_lower_source_memory_diagnostics_t diagnostics = {};
  for (uint16_t reason = 0;
       reason < LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_COUNT; ++reason) {
    diagnostics.rejection_diagnostic_indices[reason] =
        static_cast<uint16_t>(100 + reason);
  }
  loom_low_lower_source_memory_t constraint = {};
  constraint.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  constraint.root_kind = LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ANY;
  constraint.address_layout = LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_LAYOUT_ANY;
  constraint.dynamic_term_count = 0;
  constraint.dynamic_view_base_term_count = 0;
  constraint.dynamic_index_source =
      LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE;
  constraint.memory_space_mask = LOOM_LOW_LOWER_MEMORY_SPACE_GLOBAL;
  constraint.element_byte_count = 4;
  constraint.vector_lane_count = 1;
  constraint.minimum_alignment = 4;
  loom_low_lower_source_memory_shape_t shape = {
      .vector_lane_byte_stride = 4,
      .static_byte_offset_minimum = 0,
      .static_byte_offset_maximum = 0};

  loom_low_source_memory_access_plan_t access = {};
  access.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  access.root_value_id = 0;
  access.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
  access.element_byte_count = 4;
  access.vector_lane_count = 1;
  access.vector_lane_byte_stride = 4;
  access.minimum_alignment = 4;

  loom_low_lower_rule_match_context_t context = {};
  uint16_t diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  EXPECT_TRUE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));

  access.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
  EXPECT_FALSE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));
  EXPECT_EQ(diagnostic_index,
            100 + LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_MEMORY_SPACE);
  access.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;

  access.vector_lane_count = 2;
  EXPECT_FALSE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));
  EXPECT_EQ(diagnostic_index,
            100 + LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_VECTOR_LANE_COUNT);

  context.vector_lane_projection = {
      .source_lane_count = 2,
      .projected_lane_count = 1,
  };
  EXPECT_TRUE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));
  access.minimum_alignment = 2;
  EXPECT_FALSE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));
  EXPECT_EQ(diagnostic_index,
            100 + LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_MINIMUM_ALIGNMENT);
  access.minimum_alignment = 4;
  context.vector_lane_projection.source_lane_count = 3;
  EXPECT_FALSE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));
  EXPECT_EQ(diagnostic_index,
            100 + LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_VECTOR_LANE_COUNT);
  context.vector_lane_projection = {};
  access.vector_lane_count = 1;

  access.minimum_alignment = 2;
  access.static_byte_offset = 1;
  EXPECT_FALSE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));
  EXPECT_EQ(diagnostic_index,
            100 + LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_MINIMUM_ALIGNMENT);
  access.minimum_alignment = 4;
  EXPECT_FALSE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));
  EXPECT_EQ(diagnostic_index,
            100 + LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_STATIC_OFFSET);

  EXPECT_FALSE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, nullptr,
      LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT, &diagnostic_index));
  EXPECT_EQ(diagnostic_index,
            100 + LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_LAYOUT);
}

TEST(LowLowerRuleSelectionTest, RanksActionableFailuresBeforeDepth) {
  loom_low_lower_rule_failure_t diagnostic_failure = {};
  diagnostic_failure.has_source_op_span = true;
  diagnostic_failure.diagnostic_index = 1;
  diagnostic_failure.matched_guard_count = 1;
  loom_low_lower_rule_failure_t structural_nonmatch = {};
  structural_nonmatch.has_source_op_span = true;
  structural_nonmatch.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  structural_nonmatch.matched_guard_count = 10;
  structural_nonmatch.source_memory_compatible = true;

  EXPECT_TRUE(loom_low_lower_rule_failure_is_better(diagnostic_failure,
                                                    structural_nonmatch));
  EXPECT_FALSE(loom_low_lower_rule_failure_is_better(structural_nonmatch,
                                                     diagnostic_failure));
}

TEST(LowLowerRuleSelectionTest, UsesCompatibilityThenDepthForEqualFailures) {
  loom_low_lower_rule_failure_t shallow_failure = {};
  shallow_failure.has_source_op_span = true;
  shallow_failure.diagnostic_index = 1;
  shallow_failure.matched_guard_count = 1;
  loom_low_lower_rule_failure_t deep_failure = {};
  deep_failure.has_source_op_span = true;
  deep_failure.diagnostic_index = 2;
  deep_failure.matched_guard_count = 2;
  loom_low_lower_rule_failure_t compatible_failure = shallow_failure;
  compatible_failure.source_memory_compatible = true;

  EXPECT_TRUE(
      loom_low_lower_rule_failure_is_better(deep_failure, shallow_failure));
  EXPECT_TRUE(
      loom_low_lower_rule_failure_is_better(compatible_failure, deep_failure));
}

class LowLowerRuleMatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &query_arena_);
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
    iree_arena_deinitialize(&query_arena_);
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

  loom_op_t* BuildFloatConstant(double value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_scalar_constant_build(
        &builder_, loom_attr_f64(value), loom_type_scalar(LOOM_SCALAR_TYPE_F32),
        LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildFloatTruncate(uint8_t instance_flags, loom_value_id_t input) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_scalar_fptrunc_build(
        &builder_, instance_flags, input,
        loom_type_scalar(LOOM_SCALAR_TYPE_F32),
        loom_type_scalar(LOOM_SCALAR_TYPE_BF16), LOOM_LOCATION_UNKNOWN, &op));
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
    loom_low_lower_guard_payload_t guard_payload = {};
    const loom_low_lower_guard_ref_t guard_ref = 0;
    if (reject_related_guard) {
      guard.kind = LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL;
      guard.diagnostic_index = 0;
      guard.payload_ordinal = 1;
      guard_payload.u64 = UINT64_MAX;
      source_node.guard_count = 1;
    }
    loom_low_lower_rule_t rule = {};
    rule.source_node_span = LOOM_LOW_LOWER_SOURCE_NODE_SPAN(0, 1);
    const loom_low_lower_rule_span_t span = {
        .source_op_kind = source_op->kind,
        .rule_start = 0,
        .rule_count = 1,
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
      rule_set.guard_payloads = &guard_payload;
      rule_set.guard_payload_count = 1;
      rule_set.guards = &guard;
      rule_set.guard_count = 1;
      rule_set.guard_refs = &guard_ref;
      rule_set.guard_ref_count = 1;
    }
    loom_low_lower_rule_match_context_t match_context = {.module = module_};

    loom_low_lower_rule_selection_t selection = {};
    IREE_EXPECT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &rule_set, source_op, &selection));
    SourceGraphSelection result = {
        .source_nodes = {},
        .diagnostic_source_op = selection.failure.diagnostic_source_op,
        .source_node_count = selection.source_node_count,
        .selected = selection.rule != nullptr,
        .has_source_op_span = selection.failure.has_source_op_span,
    };
    for (uint8_t i = 0; i < selection.source_node_count; ++i) {
      result.source_nodes[i] = selection.source_nodes[i];
    }
    return result;
  }

  bool SelectValueNoUsesAfter(
      const loom_op_t* source_op,
      loom_consumption_region_query_t* consumption_query) {
    loom_low_lower_guard_t guard = {};
    guard.kind = LOOM_LOW_LOWER_GUARD_VALUE_NO_USES_AFTER;
    guard.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
    const loom_low_lower_guard_ref_t guard_ref = 0;
    loom_low_lower_value_ref_t value_ref = {
        .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND};
    loom_low_lower_rule_t rule = {.guard_count = 1};
    const loom_low_lower_rule_span_t span = {
        .source_op_kind = source_op->kind,
        .rule_start = 0,
        .rule_count = 1,
    };
    loom_low_lower_rule_set_t rule_set = {};
    rule_set.spans = &span;
    rule_set.span_count = 1;
    rule_set.rules = &rule;
    rule_set.rule_count = 1;
    rule_set.guards = &guard;
    rule_set.guard_count = 1;
    rule_set.guard_refs = &guard_ref;
    rule_set.guard_ref_count = 1;
    rule_set.value_refs = &value_ref;
    rule_set.value_ref_count = 1;
    loom_low_lower_rule_match_context_t match_context = {
        .module = module_, .consumption_query = consumption_query};
    loom_low_lower_rule_selection_t selection = {};
    IREE_EXPECT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &rule_set, source_op, &selection));
    return selection.rule != nullptr;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t query_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(LowLowerRuleMatchTest, MatchesValueWithNoDynamicallyLaterUses) {
  const loom_value_id_t available_lhs =
      loom_scalar_constant_result(BuildScalarConstant(1));
  const loom_value_id_t available_rhs =
      loom_scalar_constant_result(BuildScalarConstant(2));
  BuildMultiply(available_lhs, available_rhs);
  const loom_op_t* available_source = BuildAdd(available_lhs, available_rhs);

  const loom_value_id_t observed_lhs =
      loom_scalar_constant_result(BuildScalarConstant(3));
  const loom_value_id_t observed_rhs =
      loom_scalar_constant_result(BuildScalarConstant(4));
  BuildMultiply(observed_lhs, observed_rhs);
  const loom_op_t* observed_source = BuildAdd(observed_lhs, observed_rhs);
  BuildMultiply(observed_lhs, observed_rhs);

  loom_consumption_region_query_t consumption_query;
  loom_consumption_region_query_initialize(module_, module_->body,
                                           &query_arena_, &consumption_query);
  EXPECT_TRUE(SelectValueNoUsesAfter(available_source, &consumption_query));
  EXPECT_FALSE(SelectValueNoUsesAfter(observed_source, &consumption_query));
  EXPECT_FALSE(SelectValueNoUsesAfter(available_source, nullptr));
}

TEST_F(LowLowerRuleMatchTest, SelectsFirstMatchAndResetsReusedSelection) {
  loom_low_lower_guard_t guards[2] = {};
  loom_low_lower_guard_payload_t guard_payloads[2] = {};
  guards[0].kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE;
  guards[0].selector.attribute.attr_index = 0;
  guards[0].diagnostic_index = 0;
  guards[0].payload_ordinal = 1;
  guard_payloads[0].i64_range.minimum = 0;
  guard_payloads[0].i64_range.maximum = 3;
  guards[1].kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE;
  guards[1].selector.attribute.attr_index = 0;
  guards[1].diagnostic_index = 1;
  guards[1].payload_ordinal = 2;
  guard_payloads[1].i64_range.minimum = 4;
  guard_payloads[1].i64_range.maximum = 8;
  const loom_low_lower_guard_ref_t guard_refs[] = {0, 1};
  loom_low_lower_rule_t rules[3] = {};
  rules[0].guard_start = 0;
  rules[0].guard_count = 1;
  rules[1].guard_start = 0;
  rules[1].guard_count = 1;
  rules[2].guard_start = 1;
  rules[2].guard_count = 1;
  const loom_low_lower_rule_span_t span = {
      .source_op_kind = LOOM_OP_INDEX_CONSTANT,
      .rule_start = 0,
      .rule_count = 3,
  };
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.spans = &span;
  rule_set.span_count = 1;
  rule_set.rules = rules;
  rule_set.rule_count = IREE_ARRAYSIZE(rules);
  rule_set.guard_payloads = guard_payloads;
  rule_set.guard_payload_count = IREE_ARRAYSIZE(guard_payloads);
  rule_set.guards = guards;
  rule_set.guard_count = IREE_ARRAYSIZE(guards);
  rule_set.guard_refs = guard_refs;
  rule_set.guard_ref_count = IREE_ARRAYSIZE(guard_refs);
  loom_low_lower_rule_match_context_t match_context = {.module = module_};
  const loom_op_t* source_op = BuildConstant(5);

  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, source_op, &selection));

  EXPECT_EQ(selection.rule, &rules[2]);
  EXPECT_EQ(selection.rule_index, 2u);
  EXPECT_TRUE(selection.failure.has_source_op_span);
  ASSERT_EQ(selection.source_node_count, 1u);
  EXPECT_EQ(selection.source_nodes[0], source_op);

  const loom_op_t* rejected_op = BuildConstant(9);
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, rejected_op, &selection));
  EXPECT_EQ(selection.rule, nullptr);
  EXPECT_EQ(selection.rule_index, UINT16_MAX);
  EXPECT_EQ(selection.source_node_count, 0u);
  EXPECT_FALSE(selection.uses_source_memory_access);
  EXPECT_TRUE(selection.failure.has_source_op_span);
  EXPECT_EQ(selection.failure.diagnostic_source_op, rejected_op);
  EXPECT_EQ(selection.failure.diagnostic_index, 0u);
  EXPECT_EQ(selection.failure.matched_guard_count, 0u);

  IREE_ASSERT_OK(loom_low_lower_rule_set_select_rule_range_with_match_context(
      &match_context, &rule_set, source_op, 0, 0, &selection));
  EXPECT_EQ(selection.rule, nullptr);
  EXPECT_EQ(selection.rule_index, UINT16_MAX);
  EXPECT_EQ(selection.source_node_count, 0u);
  EXPECT_FALSE(selection.failure.has_source_op_span);
  EXPECT_EQ(selection.failure.diagnostic_index, LOOM_LOW_LOWER_DIAGNOSTIC_NONE);
}

TEST_F(LowLowerRuleMatchTest, MatchesI64AttributeSumsWithoutSignedOverflow) {
  loom_low_lower_guard_t guard = {};
  guard.kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_SUM_EQ;
  guard.selector.attribute.attr_index = 0;
  guard.selector.attribute.other_attr_index = 0;
  guard.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  guard.payload_ordinal = 1;
  loom_low_lower_guard_payload_t guard_payload = {};
  const loom_low_lower_guard_ref_t guard_ref = 0;
  loom_low_lower_rule_t rule = {.guard_count = 1};
  const loom_low_lower_rule_span_t span = {
      .source_op_kind = LOOM_OP_INDEX_CONSTANT,
      .rule_start = 0,
      .rule_count = 1,
  };
  loom_low_lower_rule_set_t rule_set = {.spans = &span,
                                        .span_count = 1,
                                        .rules = &rule,
                                        .rule_count = 1,
                                        .guard_payloads = &guard_payload,
                                        .guard_payload_count = 1,
                                        .guards = &guard,
                                        .guard_count = 1,
                                        .guard_refs = &guard_ref,
                                        .guard_ref_count = 1};
  struct Case {
    int64_t value;
    int64_t expected_sum;
    bool matches;
  };
  const Case cases[] = {
      {16, 32, true},
      {8, 32, false},
      {INT64_MAX / 2, INT64_MAX - 1, true},
      {INT64_MAX, -2, false},
      {INT64_MIN, 0, false},
  };
  for (const Case& test_case : cases) {
    const loom_op_t* op = BuildConstant(test_case.value);
    guard_payload.i64 = test_case.expected_sum;
    loom_low_lower_rule_match_context_t match_context = {.module = module_};
    loom_low_lower_rule_selection_t selection = {};
    IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &rule_set, op, &selection));
    EXPECT_EQ(selection.rule != nullptr, test_case.matches)
        << "value=" << test_case.value
        << " expected_sum=" << test_case.expected_sum;
  }
}

TEST_F(LowLowerRuleMatchTest, MatchesBiasedPowersWithoutSignedOverflow) {
  loom_low_lower_guard_t guard = {};
  loom_low_lower_guard_payload_t guard_payload = {};
  guard.kind = LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_I64;
  guard.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  guard.payload_ordinal = 1;
  const loom_low_lower_guard_ref_t guard_ref = 0;
  loom_low_lower_value_ref_t value_ref = {.kind =
                                              LOOM_LOW_LOWER_VALUE_REF_RESULT};
  loom_low_lower_rule_t rule = {.guard_count = 1};
  const loom_low_lower_rule_span_t span = {
      .source_op_kind = LOOM_OP_INDEX_CONSTANT,
      .rule_start = 0,
      .rule_count = 1,
  };
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.spans = &span;
  rule_set.span_count = 1;
  rule_set.rules = &rule;
  rule_set.rule_count = 1;
  rule_set.guard_payloads = &guard_payload;
  rule_set.guard_payload_count = 1;
  rule_set.guards = &guard;
  rule_set.guard_count = 1;
  rule_set.guard_refs = &guard_ref;
  rule_set.guard_ref_count = 1;
  rule_set.value_refs = &value_ref;
  rule_set.value_ref_count = 1;
  struct Case {
    // Exact source value.
    int64_t value;
    // Bias in the guard row.
    int64_t addend;
    // Whether the biased value is a positive power of two.
    bool matches;
  };
  const Case cases[] = {
      {8, 0, true},
      {0, 0, false},
      {-8, 0, false},
      {3, -1, true},
      {5, -1, true},
      {7, -1, false},
      {9, -1, true},
      {0, 1, true},
      {INT64_MAX, 1, false},
      {INT64_MIN, -1, false},
      {INT64_MAX, INT64_MIN, false},
      {INT64_MIN, INT64_MAX, false},
      {INT64_MAX, 1 - INT64_MAX, true},
  };
  for (const Case& test_case : cases) {
    const loom_op_t* op = BuildConstant(test_case.value);
    loom_value_fact_table_t facts = {};
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &module_->arena,
                                                    module_->values.count));
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &facts, loom_index_constant_result(op),
        loom_value_facts_exact_i64(test_case.value)));
    guard_payload.addend = test_case.addend;
    loom_low_lower_rule_match_context_t match_context = {.module = module_,
                                                         .fact_table = &facts};
    loom_low_lower_rule_selection_t selection = {};
    IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &rule_set, op, &selection));
    EXPECT_EQ(selection.rule != nullptr, test_case.matches)
        << "value=" << test_case.value << " addend=" << test_case.addend;
  }
}

TEST_F(LowLowerRuleMatchTest, MatchesFloatingPowersInExponentRange) {
  loom_low_lower_guard_t guard = {};
  guard.kind = LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_FLOAT;
  guard.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  guard.payload_ordinal = 1;
  loom_low_lower_guard_payload_t guard_payload = {};
  guard_payload.i64_range.minimum = -9;
  guard_payload.i64_range.maximum = -8;
  const loom_low_lower_guard_ref_t guard_ref = 0;
  loom_low_lower_value_ref_t value_ref = {.kind =
                                              LOOM_LOW_LOWER_VALUE_REF_RESULT};
  loom_low_lower_rule_t rule = {.guard_count = 1};
  const loom_low_lower_rule_span_t span = {
      .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
      .rule_start = 0,
      .rule_count = 1,
  };
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.spans = &span;
  rule_set.span_count = 1;
  rule_set.rules = &rule;
  rule_set.rule_count = 1;
  rule_set.guard_payloads = &guard_payload;
  rule_set.guard_payload_count = 1;
  rule_set.guards = &guard;
  rule_set.guard_count = 1;
  rule_set.guard_refs = &guard_ref;
  rule_set.guard_ref_count = 1;
  rule_set.value_refs = &value_ref;
  rule_set.value_ref_count = 1;
  struct Case {
    double value;
    bool matches;
  };
  const Case cases[] = {
      {-0x1p-9, true}, {0x1p-8, true}, {0x1p-7, false},
      {1.5, false},    {-0.0, false},
  };
  for (const Case& test_case : cases) {
    const loom_op_t* op = BuildFloatConstant(test_case.value);
    loom_value_fact_table_t facts = {};
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &module_->arena,
                                                    module_->values.count));
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &facts, loom_scalar_constant_result(op),
        loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, test_case.value)));
    loom_low_lower_rule_match_context_t match_context = {.module = module_,
                                                         .fact_table = &facts};
    loom_low_lower_rule_selection_t selection = {};
    IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &rule_set, op, &selection));
    EXPECT_EQ(selection.rule != nullptr, test_case.matches)
        << "value=" << test_case.value;
  }
}

TEST_F(LowLowerRuleMatchTest, MatchesSubnormalPolicyPermissionOrRetainedFact) {
  loom_low_lower_guard_t guard = {};
  guard.kind =
      LOOM_LOW_LOWER_GUARD_VALUE_NOT_SUBNORMAL_OR_INSTANCE_FLAGS_HAS_ALL;
  guard.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  guard.payload_ordinal = 1;
  loom_low_lower_guard_payload_t guard_payload = {};
  guard_payload.u64 = LOOM_SCALAR_FLOATCONVERSIONFLAGS_DAZ;
  const loom_low_lower_guard_ref_t guard_ref = 0;
  loom_low_lower_value_ref_t value_ref = {};
  value_ref.kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND;
  loom_low_lower_rule_t rule = {};
  rule.guard_count = 1;
  const loom_low_lower_rule_span_t span = {
      .source_op_kind = LOOM_OP_SCALAR_FPTRUNC,
      .rule_start = 0,
      .rule_count = 1,
  };
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.spans = &span;
  rule_set.span_count = 1;
  rule_set.rules = &rule;
  rule_set.rule_count = 1;
  rule_set.value_refs = &value_ref;
  rule_set.value_ref_count = 1;
  rule_set.guard_payloads = &guard_payload;
  rule_set.guard_payload_count = 1;
  rule_set.guards = &guard;
  rule_set.guard_count = 1;
  rule_set.guard_refs = &guard_ref;
  rule_set.guard_ref_count = 1;

  struct Case {
    uint8_t instance_flags;
    double input;
    bool provide_facts;
    bool matches;
  };
  const Case cases[] = {
      {0, 1.0, true, true},
      {0, 0x1p-149, true, false},
      {LOOM_SCALAR_FLOATCONVERSIONFLAGS_DAZ, 0x1p-149, false, true},
  };
  for (const Case& test_case : cases) {
    const loom_op_t* input_op = BuildFloatConstant(test_case.input);
    const loom_op_t* source_op = BuildFloatTruncate(
        test_case.instance_flags, loom_scalar_constant_result(input_op));
    loom_value_fact_table_t facts = {};
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &module_->arena,
                                                    module_->values.count));
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &facts, loom_scalar_constant_result(input_op),
        loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, test_case.input)));
    loom_low_lower_rule_match_context_t match_context = {};
    match_context.module = module_;
    match_context.fact_table = test_case.provide_facts ? &facts : nullptr;
    loom_low_lower_rule_selection_t selection = {};
    IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
        &match_context, &rule_set, source_op, &selection));
    EXPECT_EQ(selection.rule != nullptr, test_case.matches);
  }
}

TEST_F(LowLowerRuleMatchTest, MatchesCompleteStorageOperandSchema) {
  const loom_op_t* source_op = BuildScalarConstant(7);
  const loom_value_fact_encoded_operand_schema_t actual_schema = {
      /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN,
      /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0,
      /*.secondary_scale_format=*/{},
      /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES,
      /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D,
      /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY,
      /*.rounding_policy=*/{},
      /*.codebook_policy=*/{},
      /*.sparsity_policy=*/{},
      /*.flags=*/{},
      /*.sparsity_group=*/{},
      /*.payload_register_count=*/{},
      /*.payload_element_count=*/8,
      /*.scale_group=*/
      {
          .element_count = 8,
          .shape = {8},
      },
      /*.scale_operand_count=*/1,
  };
  loom_value_fact_table_t facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &module_->arena,
                                                  module_->values.count));
  loom_value_facts_t source_facts = {};
  const loom_value_fact_encoding_summary_t summary = {
      .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
      .static_spec_encoding_id = {},
      .address_layout = {},
      .storage_schema =
          {
              .static_spec_encoding_id = {},
              .encoded_operand = actual_schema,
          },
  };
  IREE_ASSERT_OK(loom_value_facts_make_encoding_summary(&facts.context, summary,
                                                        &source_facts));
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &facts, loom_scalar_constant_result(source_op), source_facts));

  loom_low_lower_guard_t guard = {};
  guard.kind = LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_OPERAND_SCHEMA;
  guard.diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  const loom_low_lower_guard_ref_t guard_ref = 0;
  loom_low_lower_value_ref_t value_ref = {.kind =
                                              LOOM_LOW_LOWER_VALUE_REF_RESULT};
  loom_low_lower_rule_t rule = {.guard_count = 1};
  const loom_low_lower_rule_span_t span = {
      .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
      .rule_start = 0,
      .rule_count = 1,
  };
  loom_value_fact_encoded_operand_schema_t expected_schema = actual_schema;
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.spans = &span;
  rule_set.span_count = 1;
  rule_set.rules = &rule;
  rule_set.rule_count = 1;
  rule_set.guards = &guard;
  rule_set.guard_count = 1;
  rule_set.storage_operand_schemas = &expected_schema;
  rule_set.storage_operand_schema_count = 1;
  rule_set.guard_refs = &guard_ref;
  rule_set.guard_ref_count = 1;
  rule_set.value_refs = &value_ref;
  rule_set.value_ref_count = 1;
  loom_low_lower_rule_match_context_t match_context = {.module = module_,
                                                       .fact_table = &facts};

  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, source_op, &selection));
  EXPECT_EQ(selection.rule, &rule);

  expected_schema.scale_group.element_count = 4;
  expected_schema.scale_group.shape[0] = 4;
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &rule_set, source_op, &selection));
  EXPECT_EQ(selection.rule, nullptr);
}

TEST_F(LowLowerRuleMatchTest, ContractQueriesMaySelectContractOnlyRules) {
  loom_low_lower_rule_t rules[2] = {};
  rules[0].flags = LOOM_LOW_LOWER_RULE_FLAG_CONTRACT_ONLY;
  const loom_low_lower_rule_span_t span = {
      .source_op_kind = LOOM_OP_INDEX_CONSTANT,
      .rule_start = 0,
      .rule_count = 2,
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

TEST_F(LowLowerRuleMatchTest, SelectsRootKindRejection) {
  loom_low_lower_source_memory_diagnostics_t diagnostics = {};
  diagnostics.rejection_diagnostic_indices
      [LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_ROOT_KIND] = 7;
  loom_low_lower_source_memory_t constraint = {};
  constraint.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  constraint.root_kind = LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ALLOCA;
  constraint.address_layout = LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_LAYOUT_ANY;
  constraint.dynamic_term_count = 0;
  constraint.dynamic_view_base_term_count = 0;
  constraint.dynamic_index_source =
      LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE;
  constraint.memory_space_mask = LOOM_LOW_LOWER_MEMORY_SPACE_GLOBAL;
  constraint.element_byte_count = 4;
  constraint.vector_lane_count = 1;
  loom_low_lower_source_memory_shape_t shape = {
      .vector_lane_byte_stride = 4,
      .static_byte_offset_minimum = 0,
      .static_byte_offset_maximum = 0};

  loom_low_source_memory_access_plan_t access = {};
  access.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  access.root_value_id = loom_index_constant_result(BuildConstant(0));
  access.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
  access.element_byte_count = 4;
  access.vector_lane_count = 1;
  access.vector_lane_byte_stride = 4;

  loom_low_lower_rule_match_context_t context = {.module = module_};
  uint16_t diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE;
  EXPECT_FALSE(loom_low_lower_rule_source_memory_matches(
      &context, &constraint, &shape, &diagnostics, &access, 0,
      &diagnostic_index));
  EXPECT_EQ(diagnostic_index, 7);
}

}  // namespace
}  // namespace loom
