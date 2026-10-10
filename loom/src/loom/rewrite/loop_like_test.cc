// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/loop_like.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_loop.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class LoopLikeReplacementTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    vtables = loom_scf_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    iree_arena_initialize(&block_pool_, &scratch_arena_);

    loom_builder_t module_builder = {};
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("host"), &name));
    uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
        (loom_symbol_ref_t){0, symbol}, /*arg_types=*/nullptr,
        /*arg_types_count=*/0, /*result_types=*/nullptr, /*result_count=*/0,
        /*tied_results=*/nullptr, /*tied_result_count=*/0,
        /*predicates=*/nullptr, /*predicates_count=*/0, LOOM_LOCATION_UNKNOWN,
        &function_));
    function_body_ = loom_test_func_body(function_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(function_body_), &builder_);
    builder_.ip.parent_op = function_;
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_id_t BuildConstant(loom_attribute_t value, loom_type_t type) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_constant_build(&builder_, value, type,
                                           LOOM_LOCATION_UNKNOWN, &op));
    return loom_test_constant_result(op);
  }

  loom_value_id_t BuildIndex(int64_t value) {
    return BuildConstant(loom_attr_i64(value),
                         loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  }

  void BuildTestYield(loom_op_t* owner, loom_region_t* region,
                      const loom_value_id_t* values, uint16_t value_count) {
    const loom_builder_ip_t saved =
        loom_builder_enter_region(&builder_, owner, region);
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, values, value_count,
                                         LOOM_LOCATION_UNKNOWN, &yield));
    loom_builder_restore(&builder_, saved);
  }

  void BuildScfYield(loom_op_t* owner, loom_region_t* region,
                     const loom_value_id_t* values, uint16_t value_count) {
    const loom_builder_ip_t saved =
        loom_builder_enter_region(&builder_, owner, region);
    loom_op_t* yield = nullptr;
    IREE_ASSERT_OK(loom_scf_yield_build(&builder_, values, value_count,
                                        LOOM_LOCATION_UNKNOWN, &yield));
    loom_builder_restore(&builder_, saved);
  }

  void Erase(loom_op_t* op) {
    loom_rewriter_t rewriter = {};
    loom_rewriter_initialize(&rewriter, module_, &scratch_arena_);
    IREE_ASSERT_OK(loom_rewriter_erase(&rewriter, op));
    loom_rewriter_deinitialize(&rewriter);
  }

  void FinishAndVerify() {
    loom_builder_set_block(&builder_, loom_region_entry_block(function_body_));
    loom_op_t* function_yield = nullptr;
    IREE_ASSERT_OK(
        loom_test_yield_build(&builder_, /*values=*/nullptr, /*values_count=*/0,
                              LOOM_LOCATION_UNKNOWN, &function_yield));
    const loom_verify_options_t options = {
        .sink = {loom_diagnostic_stderr_sink, nullptr},
    };
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  void Name(loom_value_id_t value, iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, name, &name_id));
    IREE_ASSERT_OK(loom_module_set_value_name(module_, value, name_id));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* function_ = nullptr;
  loom_region_t* function_body_ = nullptr;
  loom_builder_t builder_ = {};
  iree_arena_allocator_t scratch_arena_;
};

TEST_F(LoopLikeReplacementTest,
       ExpandsSyntheticLoopStateAndPreservesInterfacePayload) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t tile = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  const loom_value_id_t lower = BuildIndex(0);
  const loom_value_id_t upper = BuildIndex(8);
  const loom_value_id_t step = BuildIndex(1);
  const loom_value_id_t first = BuildConstant(loom_attr_f64(1.0), tile);
  const loom_value_id_t second = BuildConstant(loom_attr_f64(2.0), tile);
  const loom_value_id_t counter = BuildConstant(loom_attr_i64(3), index);
  const loom_value_id_t source_initial[] = {first, counter};
  const loom_type_t source_types[] = {tile, index};
  const loom_tied_result_t source_ties[] = {{
      .result_index = 0,
      .operand_index = 3,
      .has_type_change = false,
  }};
  loom_op_t* source = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(
      &builder_, lower, upper, step, source_initial,
      IREE_ARRAYSIZE(source_initial), source_types, source_ties,
      IREE_ARRAYSIZE(source_ties), LOOM_LOCATION_UNKNOWN, &source));
  loom_block_t* source_body =
      loom_region_entry_block(loom_test_loop_body(source));
  const loom_value_id_t source_yielded[] = {loom_block_arg_id(source_body, 1),
                                            loom_block_arg_id(source_body, 2)};
  BuildTestYield(source, loom_test_loop_body(source), source_yielded,
                 IREE_ARRAYSIZE(source_yielded));
  Name(loom_block_arg_id(source_body, 0), IREE_SV("source_iv"));
  Name(loom_block_arg_id(source_body, 2), IREE_SV("source_counter"));
  Name(loom_op_results(source)[1], IREE_SV("source_result"));
  const iree_string_view_t op_comments[] = {IREE_SV("loop comment")};
  const iree_string_view_t block_comments[] = {IREE_SV("entry comment")};
  IREE_ASSERT_OK(loom_module_attach_op_comments(module_, source, op_comments,
                                                IREE_ARRAYSIZE(op_comments)));
  IREE_ASSERT_OK(loom_module_attach_block_comments(
      module_, source_body, block_comments, IREE_ARRAYSIZE(block_comments)));

  loom_builder_set_before(&builder_, source);
  loom_value_id_t target_initial[] = {first, second, counter};
  const loom_type_t target_types[] = {tile, tile, index};
  const uint16_t source_offsets[] = {0, 2, 3};
  const loom_loop_like_replacement_state_t target_state = {
      .initial_values = {target_initial,
                         (uint16_t)IREE_ARRAYSIZE(target_initial)},
      .header_types = nullptr,
      .source_header_offsets = source_offsets,
      .result_types = target_types,
      .result_count = (uint16_t)IREE_ARRAYSIZE(target_types),
      .source_result_offsets = source_offsets,
  };
  loom_loop_like_replacement_t replacement = {};
  IREE_ASSERT_OK(loom_loop_like_build_replacement(
      &builder_, loom_loop_like_cast(module_, source), &target_state,
      &scratch_arena_, &replacement));

  ASSERT_TRUE(loom_test_loop_isa(replacement.loop.op));
  EXPECT_EQ(replacement.loop.vtable,
            loom_loop_like_cast(module_, source).vtable);
  EXPECT_EQ(replacement.loop.op->operand_count, 6u);
  EXPECT_EQ(replacement.results.count, 3u);
  EXPECT_EQ(replacement.body_state.count, 3u);
  EXPECT_EQ(loom_test_loop_lower_bound(replacement.loop.op), lower);
  EXPECT_EQ(loom_test_loop_upper_bound(replacement.loop.op), upper);
  EXPECT_EQ(loom_test_loop_step(replacement.loop.op), step);
  ASSERT_EQ(replacement.loop.op->tied_result_count, 2u);
  EXPECT_EQ(loom_op_tied_results(replacement.loop.op)[0].result_index, 0u);
  EXPECT_EQ(loom_op_tied_results(replacement.loop.op)[0].operand_index, 3u);
  EXPECT_EQ(loom_op_tied_results(replacement.loop.op)[1].result_index, 1u);
  EXPECT_EQ(loom_op_tied_results(replacement.loop.op)[1].operand_index, 4u);
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, loom_loop_like_iv(replacement.loop)),
      IREE_SV("source_iv")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, replacement.body_state.values[2]),
      IREE_SV("source_counter")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, replacement.results.values[2]),
      IREE_SV("source_result")));
  iree_host_size_t comment_count = 0;
  EXPECT_NE(
      loom_module_op_comments(module_, replacement.loop.op, &comment_count),
      nullptr);
  EXPECT_EQ(comment_count, 1u);
  EXPECT_NE(loom_module_block_comments(module_, replacement.body_entry,
                                       &comment_count),
            nullptr);
  EXPECT_EQ(comment_count, 1u);

  BuildTestYield(replacement.loop.op, loom_loop_like_body(replacement.loop),
                 replacement.body_state.values, replacement.body_state.count);
  Erase(source);
  FinishAndVerify();
}

TEST_F(LoopLikeReplacementTest, RebuildsStatelessCountedLoop) {
  const loom_value_id_t lower = BuildIndex(0);
  const loom_value_id_t upper = BuildIndex(8);
  const loom_value_id_t step = BuildIndex(1);
  loom_op_t* source = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, /*build_flags=*/0, lower, upper, step,
      /*iter_args=*/nullptr, /*iter_args_count=*/0, /*result_types=*/nullptr,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED, LOOM_LOCATION_UNKNOWN,
      &source));
  BuildScfYield(source, loom_scf_for_body(source), /*values=*/nullptr,
                /*value_count=*/0);

  loom_builder_set_before(&builder_, source);
  const loom_loop_like_replacement_state_t target_state = {
      .initial_values = {nullptr, 0},
      .header_types = nullptr,
      .source_header_offsets = nullptr,
      .result_types = nullptr,
      .result_count = 0,
      .source_result_offsets = nullptr,
  };
  loom_loop_like_replacement_t replacement = {};
  IREE_ASSERT_OK(loom_loop_like_build_replacement(
      &builder_, loom_loop_like_cast(module_, source), &target_state,
      &scratch_arena_, &replacement));

  EXPECT_EQ(replacement.results.count, 0u);
  EXPECT_EQ(replacement.body_state.count, 0u);
  EXPECT_EQ(replacement.body_entry->arg_count, 1u);
  BuildScfYield(replacement.loop.op, loom_loop_like_body(replacement.loop),
                /*values=*/nullptr, /*value_count=*/0);
  Erase(source);
  FinishAndVerify();
}

TEST_F(LoopLikeReplacementTest, RemovesSyntheticLoopState) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_value_id_t lower = BuildIndex(0);
  const loom_value_id_t upper = BuildIndex(8);
  const loom_value_id_t step = BuildIndex(1);
  const loom_value_id_t initial = BuildConstant(loom_attr_f64(1.0), f32);
  loom_op_t* source = nullptr;
  IREE_ASSERT_OK(
      loom_test_loop_build(&builder_, lower, upper, step, &initial, 1, &f32,
                           /*tied_results=*/nullptr, /*tied_result_count=*/0,
                           LOOM_LOCATION_UNKNOWN, &source));
  loom_block_t* source_body =
      loom_region_entry_block(loom_test_loop_body(source));
  BuildTestYield(source, loom_test_loop_body(source), source_body->arg_ids + 1,
                 1);

  loom_builder_set_before(&builder_, source);
  const uint16_t source_offsets[] = {0, 0};
  const loom_loop_like_replacement_state_t target_state = {
      .initial_values = {nullptr, 0},
      .header_types = nullptr,
      .source_header_offsets = source_offsets,
      .result_types = nullptr,
      .result_count = 0,
      .source_result_offsets = source_offsets,
  };
  loom_loop_like_replacement_t replacement = {};
  IREE_ASSERT_OK(loom_loop_like_build_replacement(
      &builder_, loom_loop_like_cast(module_, source), &target_state,
      &scratch_arena_, &replacement));

  ASSERT_TRUE(loom_test_loop_isa(replacement.loop.op));
  EXPECT_EQ(replacement.loop.op->operand_count, 3u);
  EXPECT_EQ(replacement.results.count, 0u);
  EXPECT_EQ(replacement.body_state.count, 0u);
  EXPECT_EQ(replacement.body_entry->arg_count, 1u);
  EXPECT_EQ(replacement.loop.op->tied_result_count, 0u);

  BuildTestYield(replacement.loop.op, loom_loop_like_body(replacement.loop),
                 /*values=*/nullptr, /*value_count=*/0);
  Erase(source);
  FinishAndVerify();
}

TEST_F(LoopLikeReplacementTest, PreservesSegmentedScfForPolicies) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_value_id_t lower = BuildIndex(0);
  const loom_value_id_t upper = BuildIndex(8);
  const loom_value_id_t step = BuildIndex(1);
  const loom_value_id_t pipeline_depth = BuildIndex(3);
  const loom_value_id_t unroll_factor = BuildIndex(2);
  const loom_value_id_t first = BuildConstant(loom_attr_f64(1.0), f32);
  const loom_value_id_t second = BuildConstant(loom_attr_f64(2.0), f32);
  loom_op_t* source = nullptr;
  const loom_scf_for_build_flags_t flags =
      LOOM_SCF_FOR_BUILD_FLAG_HAS_PIPELINE_DEPTH |
      LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR |
      LOOM_SCF_FOR_BUILD_FLAG_HAS_UNROLL_SCHEDULE;
  IREE_ASSERT_OK(
      loom_scf_for_build(&builder_, flags, lower, upper, step, &first, 1, &f32,
                         /*tied_results=*/nullptr, /*tied_result_count=*/0,
                         pipeline_depth, unroll_factor, /*unroll_policy=*/0,
                         LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED,
                         LOOM_LOCATION_UNKNOWN, &source));
  loom_block_t* source_body =
      loom_region_entry_block(loom_scf_for_body(source));
  BuildScfYield(source, loom_scf_for_body(source), source_body->arg_ids + 1, 1);

  loom_builder_set_before(&builder_, source);
  loom_value_id_t target_initial[] = {first, second};
  const loom_type_t target_types[] = {f32, f32};
  const uint16_t source_offsets[] = {0, 2};
  const loom_loop_like_replacement_state_t target_state = {
      .initial_values = {target_initial,
                         (uint16_t)IREE_ARRAYSIZE(target_initial)},
      .header_types = nullptr,
      .source_header_offsets = source_offsets,
      .result_types = target_types,
      .result_count = (uint16_t)IREE_ARRAYSIZE(target_types),
      .source_result_offsets = source_offsets,
  };
  loom_loop_like_replacement_t replacement = {};
  IREE_ASSERT_OK(loom_loop_like_build_replacement(
      &builder_, loom_loop_like_cast(module_, source), &target_state,
      &scratch_arena_, &replacement));

  ASSERT_TRUE(loom_scf_for_isa(replacement.loop.op));
  const uint16_t expected_segments[] = {1, 1, 1, 2, 1, 1};
  EXPECT_EQ(memcmp(loom_op_const_operand_segment_counts(replacement.loop.op),
                   expected_segments, sizeof(expected_segments)),
            0);
  EXPECT_EQ(loom_scf_for_pipeline_depth(replacement.loop.op), pipeline_depth);
  EXPECT_EQ(loom_scf_for_unroll_factor(replacement.loop.op), unroll_factor);
  EXPECT_FALSE(loom_scf_for_has_unroll_policy(replacement.loop.op));
  EXPECT_EQ(loom_scf_for_unroll_schedule(replacement.loop.op),
            LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED);

  BuildScfYield(replacement.loop.op, loom_loop_like_body(replacement.loop),
                replacement.body_state.values, replacement.body_state.count);
  Erase(source);
  FinishAndVerify();
}

TEST_F(LoopLikeReplacementTest,
       ExpandsConditionLoopAndInstantiatesDependentEndpointTypes) {
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t bank_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(2), 0);
  const loom_value_id_t rows = BuildIndex(8);
  const loom_type_t initial_dependent_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(rows), 0);
  const loom_value_id_t bank = BuildConstant(loom_attr_f64(0.0), bank_type);
  const loom_value_id_t dependent =
      BuildConstant(loom_attr_f64(0.0), initial_dependent_type);
  const loom_value_id_t first = BuildConstant(loom_attr_f64(1.0), f32);
  const loom_value_id_t second = BuildConstant(loom_attr_f64(2.0), f32);
  const loom_value_id_t condition = BuildConstant(
      loom_attr_bool(true), loom_type_scalar(LOOM_SCALAR_TYPE_I1));

  const loom_value_id_t source_initial[] = {bank, dependent, rows};
  loom_value_id_t source_identities[2 * IREE_ARRAYSIZE(source_initial)] = {};
  IREE_ASSERT_OK(loom_builder_reserve_values(
      &builder_, IREE_ARRAYSIZE(source_identities), source_identities));
  const loom_type_t source_header_types[] = {
      bank_type,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_identities[2]), 0),
      index,
  };
  const loom_value_id_t* source_results =
      source_identities + IREE_ARRAYSIZE(source_initial);
  const loom_type_t source_result_types[] = {
      bank_type,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_results[2]), 0),
      index,
  };
  loom_op_t* source = nullptr;
  IREE_ASSERT_OK(loom_scf_while_build(
      &builder_, source_initial, IREE_ARRAYSIZE(source_initial),
      source_header_types, source_result_types,
      IREE_ARRAYSIZE(source_result_types), /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &source));
  loom_block_t* source_condition =
      loom_region_entry_block(loom_scf_while_before(source));
  loom_builder_ip_t saved = loom_builder_enter_region(
      &builder_, source, loom_scf_while_before(source));
  loom_op_t* source_condition_op = nullptr;
  IREE_ASSERT_OK(
      loom_scf_condition_build(&builder_, condition, source_condition->arg_ids,
                               source_condition->arg_count,
                               LOOM_LOCATION_UNKNOWN, &source_condition_op));
  loom_builder_restore(&builder_, saved);
  loom_block_t* source_body =
      loom_region_entry_block(loom_scf_while_after(source));
  BuildScfYield(source, loom_scf_while_after(source), source_body->arg_ids,
                source_body->arg_count);
  Name(loom_block_arg_id(source_condition, 1),
       IREE_SV("source_header_dependent"));
  Name(loom_block_arg_id(source_body, 0), IREE_SV("source_body_bank"));
  Name(loom_op_results(source)[0], IREE_SV("source_result_bank"));

  loom_builder_set_before(&builder_, source);
  loom_value_id_t target_initial[] = {first, second, dependent, rows};
  loom_value_id_t target_identities[2 * IREE_ARRAYSIZE(target_initial)] = {};
  IREE_ASSERT_OK(loom_builder_reserve_values(
      &builder_, IREE_ARRAYSIZE(target_identities), target_identities));
  const loom_type_t target_header_types[] = {
      f32,
      f32,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(target_identities[3]), 0),
      index,
  };
  const loom_value_id_t* target_results =
      target_identities + IREE_ARRAYSIZE(target_initial);
  const loom_type_t target_result_types[] = {
      f32,
      f32,
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(target_results[3]), 0),
      index,
  };
  const uint16_t source_header_offsets[] = {0, 2, 3, 4};
  const uint16_t source_result_offsets[] = {0, 1, 3, 4};
  const loom_loop_like_replacement_state_t target_state = {
      .initial_values = {target_initial,
                         (uint16_t)IREE_ARRAYSIZE(target_initial)},
      .header_types = target_header_types,
      .source_header_offsets = source_header_offsets,
      .result_types = target_result_types,
      .result_count = (uint16_t)IREE_ARRAYSIZE(target_result_types),
      .source_result_offsets = source_result_offsets,
  };
  loom_loop_like_replacement_t replacement = {};
  IREE_ASSERT_OK(loom_loop_like_build_replacement(
      &builder_, loom_loop_like_cast(module_, source), &target_state,
      &scratch_arena_, &replacement));

  ASSERT_TRUE(loom_scf_while_isa(replacement.loop.op));
  ASSERT_NE(replacement.condition_entry, nullptr);
  EXPECT_EQ(replacement.condition_state.count, 4u);
  EXPECT_EQ(replacement.body_state.count, 4u);
  EXPECT_EQ(
      loom_type_dim_value_id_at(
          loom_module_value_type(module_, replacement.results.values[2]), 0),
      replacement.results.values[3]);
  EXPECT_EQ(loom_type_dim_value_id_at(
                loom_module_value_type(module_,
                                       replacement.condition_state.values[2]),
                0),
            replacement.condition_state.values[3]);
  EXPECT_EQ(
      loom_type_dim_value_id_at(
          loom_module_value_type(module_, replacement.body_state.values[2]), 0),
      replacement.body_state.values[3]);
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, replacement.condition_state.values[2]),
      IREE_SV("source_header_dependent")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, replacement.body_state.values[0]),
      IREE_SV("source_body_bank")));
  EXPECT_TRUE(iree_string_view_equal(
      loom_module_value_name(module_, replacement.results.values[0]),
      IREE_SV("source_result_bank")));

  saved = loom_builder_enter_region(
      &builder_, replacement.loop.op,
      loom_loop_like_condition_region(replacement.loop));
  loom_op_t* target_condition_op = nullptr;
  IREE_ASSERT_OK(loom_scf_condition_build(
      &builder_, condition, replacement.condition_state.values,
      replacement.condition_state.count, LOOM_LOCATION_UNKNOWN,
      &target_condition_op));
  loom_builder_restore(&builder_, saved);
  BuildScfYield(replacement.loop.op, loom_loop_like_body(replacement.loop),
                replacement.body_state.values, replacement.body_state.count);
  Erase(source);
  FinishAndVerify();
}

TEST_F(LoopLikeReplacementTest, CountedFactsTrackBoundReplacementAndCloning) {
  const loom_value_id_t lower = BuildIndex(0);
  const loom_value_id_t upper = BuildIndex(8);
  const loom_value_id_t same_range_upper = BuildIndex(8);
  const loom_value_id_t shorter_upper = BuildIndex(4);
  const loom_value_id_t step = BuildIndex(1);
  loom_op_t* loop = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, /*build_flags=*/0, lower, upper, step,
      /*iter_args=*/nullptr, /*iter_args_count=*/0, /*result_types=*/nullptr,
      /*tied_results=*/nullptr, /*tied_result_count=*/0, LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID, /*unroll_policy=*/0,
      LOOM_SCF_FOR_UNROLL_SCHEDULE_INTERLEAVED, LOOM_LOCATION_UNKNOWN, &loop));
  BuildScfYield(loop, loom_scf_for_body(loop), nullptr, 0);
  FinishAndVerify();

  loom_value_fact_table_t facts;
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts, &scratch_arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_compute(
      &facts, module_, loom_func_like_cast(module_, function_)));
  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module_, &scratch_arena_);
  IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter));
  loom_rewriter_attach_value_facts(&rewriter, &facts);
  const loom_value_id_t induction =
      loom_block_arg_id(loom_region_entry_block(loom_scf_for_body(loop)), 0);
  const auto* domain = loom_value_fact_table_lookup_counted_loop_domain(
      &facts, module_, induction);
  ASSERT_NE(domain, nullptr);
  EXPECT_EQ(domain->lower_bound, lower);
  EXPECT_EQ(domain->upper_bound, upper);
  EXPECT_EQ(domain->step, step);
  EXPECT_EQ(
      loom_value_fact_table_lookup_counted_loop_domain(&facts, module_, upper),
      nullptr);

  // A semantic bound change publishes new identities immediately, even when
  // its numeric facts and every loop result remain identical.
  IREE_ASSERT_OK(
      loom_rewriter_set_operand(&rewriter, loop, 1, same_range_upper));
  EXPECT_EQ(domain->upper_bound, same_range_upper);
  EXPECT_EQ(loom_value_fact_table_lookup(&facts, induction).range_hi, 7);
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, loop, 1, shorter_upper));
  EXPECT_EQ(domain->upper_bound, shorter_upper);
  EXPECT_EQ(loom_value_fact_table_lookup(&facts, induction).range_hi, 3);
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, loop, 0, step));
  IREE_ASSERT_OK(loom_rewriter_set_operand(&rewriter, loop, 2, shorter_upper));
  EXPECT_EQ(domain->lower_bound, step);
  EXPECT_EQ(domain->step, shorter_upper);

  loom_ir_remap_options_t options = {
      .allow_unmapped_values = true,
  };
  loom_ir_remap_t remap;
  IREE_ASSERT_OK(loom_ir_remap_initialize(module_, module_, &scratch_arena_,
                                          &options, &remap));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, shorter_upper, upper));
  loom_builder_set_before(&rewriter.builder, loop);
  loom_op_t* cloned = nullptr;
  IREE_ASSERT_OK(loom_ir_clone_op(&rewriter.builder, loop, &remap, &cloned));
  const loom_value_id_t cloned_induction =
      loom_block_arg_id(loom_region_entry_block(loom_scf_for_body(cloned)), 0);
  const auto* cloned_domain = loom_value_fact_table_lookup_counted_loop_domain(
      &facts, module_, cloned_induction);
  ASSERT_NE(cloned_domain, nullptr);
  EXPECT_NE(cloned_domain, domain);
  EXPECT_EQ(cloned_domain->lower_bound, step);
  EXPECT_EQ(cloned_domain->upper_bound, upper);
  EXPECT_EQ(cloned_domain->step, upper);
  EXPECT_EQ(domain->upper_bound, shorter_upper);

  // Reordering the owner keeps its body identities; deleting the source does
  // not disturb the clone's separately published domain.
  IREE_ASSERT_OK(loom_rewriter_move_before(&rewriter, loop, cloned));
  EXPECT_EQ(loom_value_fact_table_lookup_counted_loop_domain(&facts, module_,
                                                             induction),
            domain);
  IREE_ASSERT_OK(loom_rewriter_erase(&rewriter, loop));
  EXPECT_EQ(loom_value_fact_table_lookup_counted_loop_domain(&facts, module_,
                                                             cloned_induction),
            cloned_domain);
  loom_rewriter_deinitialize(&rewriter);

  loom_value_fact_table_clear_scope(&facts);
  EXPECT_FALSE(facts.has_counted_loop_domains);
  EXPECT_EQ(loom_value_fact_table_lookup_counted_loop_domain(&facts, module_,
                                                             cloned_induction),
            nullptr);
  IREE_ASSERT_OK(loom_value_fact_table_compute(
      &facts, module_, loom_func_like_cast(module_, function_)));
  const auto* rebuilt = loom_value_fact_table_lookup_counted_loop_domain(
      &facts, module_, cloned_induction);
  ASSERT_NE(rebuilt, nullptr);
  EXPECT_EQ(rebuilt->lower_bound, step);
  EXPECT_EQ(rebuilt->upper_bound, upper);
  EXPECT_EQ(rebuilt->step, upper);

  // Splicing the body into its parent is the structural part of SCF-to-CFG.
  // Its argument is no longer an intrinsic counted IV; the CFG guard owns
  // any subsequent proof of the moved block's range.
  loom_rewriter_initialize(&rewriter, module_, &scratch_arena_);
  loom_rewriter_attach_value_facts(&rewriter, &facts);
  loom_block_t* moved_entry = nullptr;
  IREE_ASSERT_OK(loom_rewriter_move_region_blocks(
      &rewriter, loom_scf_for_body(cloned), cloned, function_body_, 1,
      function_, &moved_entry));
  EXPECT_EQ(loom_block_arg_id(moved_entry, 0), cloned_induction);
  EXPECT_EQ(loom_value_fact_table_lookup_counted_loop_domain(&facts, module_,
                                                             cloned_induction),
            nullptr);
  loom_rewriter_deinitialize(&rewriter);
}

}  // namespace
}  // namespace loom
