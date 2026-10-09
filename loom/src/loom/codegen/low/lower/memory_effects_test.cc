// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/memory_effects.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_expr_test_fixture.h"

namespace loom {
namespace {

class LowLowerMemoryEffectsTest : public SymbolicExprTest {
 protected:
  void SetUp() override {
    SymbolicExprTest::SetUp();
    iree_arena_initialize(&block_pool_, &plan_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&plan_arena_);
    SymbolicExprTest::TearDown();
  }

  // Retained proof storage has a lifetime independent of analysis scratch.
  iree_arena_allocator_t plan_arena_;
};

TEST_F(LowLowerMemoryEffectsTest, RetainedOriginSurvivesAnalysisRetirement) {
  const auto uniform = DefineIndexValue();
  const auto varying = DefineIndexValue();
  auto uniform_facts = loom_value_facts_make(0, 1024, 1);
  loom_value_facts_mark_uniform_at_scope(
      &uniform_facts, LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP);
  DefineFacts(uniform, uniform_facts);
  DefineFacts(varying, loom_value_facts_make(0, 3, 1));

  loom_low_source_memory_access_plan_t source = {
      .root_value_id = 7,
      .root_uniform_scope = LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .element_byte_count = 4,
      .vector_lane_count = 2,
      .vector_lane_byte_stride = 4,
      .static_byte_offset = 160,
      .physical_root_byte_offset = 128,
      .dynamic_term_count = 2};
  source.dynamic_terms[0].index = uniform;
  source.dynamic_terms[0].byte_stride = 16;
  source.dynamic_terms[0].byte_facts = loom_value_facts_make(0, 16384, 16);
  source.dynamic_terms[1].index = varying;
  source.dynamic_terms[1].byte_stride = 4;
  source.dynamic_terms[1].byte_facts = loom_value_facts_make(0, 12, 4);

  const loom_low_lower_memory_origin_t* origin = nullptr;
  IREE_ASSERT_OK(loom_low_lower_memory_origin_plan(
      &expression_context_, &source, &plan_arena_, &origin));
  ASSERT_NE(origin, nullptr);
  iree_arena_reset(&analysis_arena_);

  loom_low_memory_relative_interval_t left = {};
  int64_t lane_byte_count = 0;
  ASSERT_TRUE(loom_low_lower_memory_packet_interval(
      origin, &source, loom_value_facts_make(0, 4, 4), &left,
      &lane_byte_count));
  EXPECT_EQ(lane_byte_count, 8);
  ASSERT_EQ(left.origin.term_count, 1);
  EXPECT_EQ(left.origin.terms[0].value_id, uniform);
  EXPECT_EQ(left.origin.terms[0].coefficient, 16);
  EXPECT_EQ(left.lower, 32);
  EXPECT_EQ(left.upper, 56);

  // Two instructions share the uniform translation but cover different bytes.
  // The varying participant must remain an envelope, never a cancelable term.
  source.static_byte_offset += 64;
  loom_low_memory_relative_interval_t right = {};
  ASSERT_TRUE(loom_low_lower_memory_packet_interval(
      origin, &source, loom_value_facts_make(0, 4, 4), &right,
      &lane_byte_count));
  loom_low_memory_access_summary_t left_access = {
      .memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP,
      .relative_interval = &left};
  auto right_access = left_access;
  right_access.relative_interval = &right;
  EXPECT_FALSE(loom_low_memory_access_summaries_may_alias(
      &left_access, &right_access, LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION));
  EXPECT_TRUE(loom_low_memory_access_summaries_may_alias(
      &left_access, &right_access, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT));
}

TEST_F(LowLowerMemoryEffectsTest, StaticGeometryNeedsNoDynamicProof) {
  loom_low_source_memory_access_plan_t source = {};
  source.root_value_id = 7;
  source.root_uniform_scope = LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP;
  source.element_byte_count = 4;
  source.vector_lane_count = 2;
  source.vector_lane_byte_stride = -4;
  source.static_byte_offset = 12;

  const loom_low_lower_memory_origin_t* origin = nullptr;
  IREE_ASSERT_OK(loom_low_lower_memory_origin_plan(
      &expression_context_, &source, &plan_arena_, &origin));
  EXPECT_EQ(origin, nullptr);
  loom_low_memory_relative_interval_t interval = {};
  int64_t lane_byte_count = 0;
  ASSERT_TRUE(loom_low_lower_memory_packet_interval(
      origin, &source, loom_value_facts_exact_i64(0), &interval,
      &lane_byte_count));
  EXPECT_EQ(interval.lower, 8);
  EXPECT_EQ(interval.upper, 16);
  EXPECT_EQ(lane_byte_count, 8);

  source.static_byte_offset = INT64_MAX;
  EXPECT_FALSE(loom_low_lower_memory_packet_interval(
      origin, &source, loom_value_facts_exact_i64(0), &interval,
      &lane_byte_count));
  source.root_uniform_scope = LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP;
  EXPECT_FALSE(loom_low_lower_memory_packet_interval(
      origin, &source, loom_value_facts_exact_i64(0), &interval,
      &lane_byte_count));
}

}  // namespace
}  // namespace loom
