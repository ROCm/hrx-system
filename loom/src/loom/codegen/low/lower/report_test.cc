// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/report.h"

#include <algorithm>
#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/target/test/target_records.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class LowLowerReportTest : public ::testing::Test {
 protected:
  static iree_status_t ObservePlannedRows(void* user_data,
                                          loom_low_lower_context_t* context) {
    auto* test = static_cast<LowLowerReportTest*>(user_data);
    // Selection and source-frequency evidence precede the first instruction.
    EXPECT_NE(test->result_.report, nullptr);
    EXPECT_EQ(test->result_.report->selected_source_op_count, 1u);
    EXPECT_EQ(test->result_.report->emitted_low_op_count, 0u);
    EXPECT_EQ(test->result_.report->rows.count, 1u);
    if (test->result_.report->rows.head) {
      const loom_low_lower_report_row_t& row =
          loom_low_lower_report_row_vec_const_rows(
              test->result_.report->rows.head)[0];
      EXPECT_EQ(row.execution_count_plus_one, 2u);
      EXPECT_EQ(row.emitted_low_op_count, 0u);
      EXPECT_EQ(row.source_op_kind, LOOM_OP_SCALAR_ADDI);
    }
    return iree_ok_status();
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_low_source_workload_register_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("lower_report_test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    target_facts_.fact_type = &loom_test_target_fact_type;
    target_facts_.storage.bundle = *loom_test_target_bundles.values[1];
    BuildFunction();
    IREE_ASSERT_OK(loom_value_fact_table_initialize(
        &fact_table_, &analysis_arena_, module_->values.count));
    fact_table_.context.target_facts = &target_facts_;
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(&fact_table_, module_, function_));

    options_.target_ref = loom_symbol_ref_null();
    options_.target_facts = &target_facts_;
    options_.descriptor_registry = &descriptor_registry_.registry;
    options_.policy = loom_test_low_lower_policy();
    options_.fact_table = &fact_table_;
  }

  void TearDown() override {
    loom_low_lower_result_deinitialize(&result_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void BuildFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&module_builder, IREE_SV("add"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const loom_type_t argument_types[] = {i32_type, i32_type};
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        argument_types, IREE_ARRAYSIZE(argument_types), &i32_type, 1, nullptr,
        0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op));
    function_ = loom_func_like_cast(module_, function_op);

    loom_block_t* entry_block =
        loom_region_entry_block(loom_func_like_body(function_));
    const loom_value_id_t lhs = loom_block_arg_id(entry_block, 0);
    const loom_value_id_t rhs = loom_block_arg_id(entry_block, 1);
    loom_builder_t body_builder;
    loom_builder_initialize(module_, &module_->arena, entry_block,
                            &body_builder);
    body_builder.ip.parent_op = function_op;
    loom_op_t* add_op = nullptr;
    IREE_ASSERT_OK(loom_scalar_addi_build(&body_builder, 0, lhs, rhs, i32_type,
                                          LOOM_LOCATION_UNKNOWN, &add_op));
    const loom_value_id_t result = loom_scalar_addi_result(add_op);
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &result, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_ = {};
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_target_facts_t target_facts_ = {};
  loom_value_fact_table_t fact_table_ = {};
  loom_low_lower_options_t options_ = {};
  loom_low_lower_result_t result_ = {};
};

TEST_F(LowLowerReportTest, CapturesAndReleasesSelectionRows) {
  loom_low_lower_policy_t policy = *options_.policy;
  policy.emit_preamble = {ObservePlannedRows, this};
  options_.policy = &policy;
  options_.report_allocator = iree_allocator_system();
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));

  ASSERT_NE(result_.report, nullptr);
  EXPECT_EQ(result_.report->selected_source_op_count, 1u);
  EXPECT_EQ(result_.report->emitted_low_op_count, 1u);
  ASSERT_EQ(result_.report->rows.count, 1u);
  ASSERT_NE(result_.report->rows.head, nullptr);
  ASSERT_EQ(result_.report->rows.head, result_.report->rows.tail);
  ASSERT_EQ(result_.report->rows.head->count, 1u);
  const loom_low_lower_report_row_t& row =
      loom_low_lower_report_row_vec_const_rows(result_.report->rows.head)[0];
  EXPECT_TRUE(iree_string_view_equal(row.function_name, IREE_SV("add")));
  EXPECT_EQ(row.source_op_kind, LOOM_OP_SCALAR_ADDI);
  EXPECT_EQ(row.selection_kind, LOOM_LOW_LOWER_REPORT_SELECTION_RULE);
  EXPECT_EQ(row.emitted_low_op_count, 1u);
  EXPECT_EQ(row.execution_count_plus_one, 2u);
  EXPECT_TRUE(iree_string_view_equal(row.descriptor_semantic_tag,
                                     IREE_SV("integer.add.i32")));

  loom_low_lower_result_deinitialize(&result_);
  EXPECT_EQ(result_.report, nullptr);
}

TEST_F(LowLowerReportTest, DisabledReportsProduceNoRows) {
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));

  EXPECT_EQ(result_.report, nullptr);
}

TEST_F(LowLowerReportTest, ReleasesRowsWhenAPlannedFunctionIsNotEmitted) {
  options_.report_allocator = iree_allocator_system();
  iree_arena_allocator_t plan_arena;
  iree_arena_initialize(&block_pool_, &plan_arena);
  loom_low_lower_function_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_low_lower_plan_function(module_, function_, &options_,
                                              &plan_arena, &result_, &plan));
  ASSERT_NE(plan, nullptr);
  ASSERT_NE(result_.report, nullptr);
  EXPECT_EQ(result_.report->rows.count, 1u);
  EXPECT_EQ(result_.low_func_op, nullptr);

  // A later function's rejection can discard the transaction before this
  // plan executes. Returned reports have their own cleanup boundary.
  iree_arena_deinitialize(&plan_arena);
  loom_low_lower_result_deinitialize(&result_);
  EXPECT_EQ(result_.report, nullptr);
}

class LowLowerMemoryExpressionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &context_.analysis_arena);
  }

  void TearDown() override {
    iree_arena_deinitialize(&context_.analysis_arena);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static loom_low_source_memory_access_plan_t MakePlan(uint8_t term_count = 1) {
    loom_low_source_memory_access_plan_t plan = {
        .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
        .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
        .element_byte_count = 4,
        .vector_lane_count = 1,
        .vector_lane_byte_stride = 4,
        .dynamic_term_count = term_count,
    };
    for (uint8_t i = 0; i < term_count; ++i) {
      plan.dynamic_terms[i].index = i + 1;
      plan.dynamic_terms[i].byte_stride = 4;
    }
    return plan;
  }

  loom_low_byte_interval_t Intern(
      const loom_low_source_memory_access_plan_t& plan) {
    loom_low_lower_memory_report_row_t row = {};
    IREE_EXPECT_OK(loom_low_lower_memory_report_row_populate_source_interval(
        &context_, &plan, &row));
    return row.source_interval;
  }

  // Reusable storage for independent function planning lifetimes.
  iree_arena_block_pool_t block_pool_;
  // Reporting consumes only the function's arena and private expression domain.
  loom_low_lower_context_t context_ = {};
};

TEST_F(LowLowerMemoryExpressionTest, CanonicalTermsShareFirstSeenIdentity) {
  auto plan = MakePlan(3);
  plan.dynamic_terms[0].index = 9;
  plan.dynamic_terms[1].index = 3;
  plan.dynamic_terms[1].byte_stride = 8;
  plan.dynamic_terms[2].index = 9;
  plan.dynamic_terms[2].byte_stride = 2;
  const auto first = Intern(plan);
  EXPECT_EQ(first.begin_expr_id, 0u);
  EXPECT_EQ(first.end_expr_id, 1u);
  EXPECT_TRUE(iree_all_bits_set(first.precision_flags,
                                LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_EXPR |
                                    LOOM_LOW_BYTE_INTERVAL_PRECISION_END_EXPR));

  auto canonical = MakePlan(2);
  canonical.dynamic_terms[0] = plan.dynamic_terms[1];
  canonical.dynamic_terms[1] = plan.dynamic_terms[0];
  canonical.dynamic_terms[1].byte_stride = 6;
  const auto used = context_.analysis_arena.used_allocation_size;
  const auto duplicate = Intern(canonical);
  EXPECT_EQ(duplicate.begin_expr_id, first.begin_expr_id);
  EXPECT_EQ(duplicate.end_expr_id, first.end_expr_id);
  EXPECT_EQ(context_.analysis_arena.used_allocation_size, used);

  // Adjacent rows must share the same endpoint identity in either role.
  canonical.static_byte_offset = 4;
  const auto adjacent = Intern(canonical);
  EXPECT_EQ(adjacent.begin_expr_id, first.end_expr_id);
  EXPECT_EQ(adjacent.end_expr_id, 2u);
}

TEST_F(LowLowerMemoryExpressionTest, EveryTermCapacityRetainsExactKeys) {
  uint32_t next_id = 0;
  for (uint8_t count = 1; count <= LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY;
       ++count) {
    SCOPED_TRACE(count);
    auto plan = MakePlan(count);
    const auto first = Intern(plan);
    EXPECT_EQ(first.begin_expr_id, next_id++);
    EXPECT_EQ(first.end_expr_id, next_id++);
    std::reverse(plan.dynamic_terms, plan.dynamic_terms + count);
    const auto reordered = Intern(plan);
    EXPECT_EQ(reordered.begin_expr_id, first.begin_expr_id);
    EXPECT_EQ(reordered.end_expr_id, first.end_expr_id);
    // Every term participates in equality, including long common prefixes.
    for (uint8_t term = 0; term < count; ++term) {
      plan.dynamic_terms[term].byte_stride = 8;
      const auto changed = Intern(plan);
      EXPECT_EQ(changed.begin_expr_id, next_id++);
      EXPECT_EQ(changed.end_expr_id, next_id++);
      plan.dynamic_terms[term].byte_stride = 4;
    }
    const auto original = Intern(plan);
    EXPECT_EQ(original.begin_expr_id, first.begin_expr_id);
    EXPECT_EQ(original.end_expr_id, first.end_expr_id);
  }
  EXPECT_EQ(context_.analysis_arena.allocation_head, nullptr);
}

TEST_F(LowLowerMemoryExpressionTest, EveryKeyComponentDistinguishesIdentity) {
  const auto base = MakePlan();
  EXPECT_EQ(Intern(base).begin_expr_id, 0u);
  auto plan = base;
  plan.static_byte_offset = INT64_MIN;
  EXPECT_EQ(Intern(plan).begin_expr_id, 2u);
  plan.static_byte_offset = INT64_MAX - 4;
  EXPECT_EQ(Intern(plan).begin_expr_id, 4u);
  plan = base;
  plan.dynamic_terms[0].index = 2;
  EXPECT_EQ(Intern(plan).begin_expr_id, 6u);
  plan = base;
  plan.dynamic_terms[0].byte_stride = INT64_MIN;
  EXPECT_EQ(Intern(plan).begin_expr_id, 8u);
  plan.dynamic_terms[0].byte_stride = INT64_MAX;
  EXPECT_EQ(Intern(plan).begin_expr_id, 10u);
  EXPECT_EQ(Intern(base).begin_expr_id, 0u);
}

TEST_F(LowLowerMemoryExpressionTest, OrderedAndPermutedGrowthStaysPooled) {
  constexpr uint32_t kRowCount = 2048;
  std::array<loom_low_memory_expr_id_t, kRowCount> first_ids = {};
  for (uint32_t order = 0; order < 3; ++order) {
    SCOPED_TRACE(order);
    auto plan = MakePlan();
    for (uint32_t i = 0; i < kRowCount; ++i) {
      uint32_t key = i;
      if (order == 1) {
        key = kRowCount - 1 - i;
      } else if (order == 2) {
        key = (i * 109) % kRowCount;
      }
      plan.static_byte_offset = 8 * key;
      const auto interval = Intern(plan);
      EXPECT_EQ(interval.begin_expr_id, 2 * i);
      EXPECT_EQ(interval.end_expr_id, 2 * i + 1);
      first_ids[key] = interval.begin_expr_id;
    }
    const auto used = context_.analysis_arena.used_allocation_size;
    EXPECT_EQ(context_.report.memory_expressions.count, 2 * kRowCount);
    EXPECT_LE(used, 2 * kRowCount * 64u);
    EXPECT_EQ(context_.analysis_arena.allocation_head, nullptr);
    for (uint32_t key = 0; key < kRowCount; ++key) {
      plan.static_byte_offset = 8 * key;
      const auto interval = Intern(plan);
      EXPECT_EQ(interval.begin_expr_id, first_ids[key]);
      EXPECT_EQ(interval.end_expr_id, first_ids[key] + 1);
    }
    EXPECT_EQ(context_.analysis_arena.used_allocation_size, used);

    // A new function restarts IDs while reusing the retired function's blocks.
    iree_arena_reset(&context_.analysis_arena);
    context_.report = {};
  }
}

TEST_F(LowLowerMemoryExpressionTest, UnrepresentableKeysDoNotAllocateIdentity) {
  auto plan = MakePlan(2);
  plan.dynamic_terms[1].index = plan.dynamic_terms[0].index;
  plan.dynamic_terms[1].byte_stride = -plan.dynamic_terms[0].byte_stride;
  const auto cancelled = Intern(plan);
  EXPECT_EQ(cancelled.begin_expr_id, LOOM_LOW_MEMORY_EXPR_ID_NONE);
  EXPECT_EQ(cancelled.end_expr_id, LOOM_LOW_MEMORY_EXPR_ID_NONE);

  plan.dynamic_terms[0].byte_stride = INT64_MAX;
  plan.dynamic_terms[1].byte_stride = 1;
  EXPECT_EQ(Intern(plan).begin_expr_id, LOOM_LOW_MEMORY_EXPR_ID_NONE);
  plan = MakePlan();
  plan.static_byte_offset = INT64_MAX;
  EXPECT_EQ(Intern(plan).begin_expr_id, LOOM_LOW_MEMORY_EXPR_ID_NONE);
  plan = MakePlan();
  plan.dynamic_terms[0].stride_value_count = 1;
  plan.dynamic_terms[0].stride_values[0] = 2;
  EXPECT_EQ(Intern(plan).begin_expr_id, LOOM_LOW_MEMORY_EXPR_ID_NONE);
  EXPECT_EQ(context_.report.memory_expressions.count, 0u);
  EXPECT_EQ(context_.analysis_arena.used_allocation_size, 0u);
}

TEST_F(LowLowerMemoryExpressionTest, AllocationFailureDoesNotPublishIdentity) {
  // Exhaust a fixed allocator through the same pool boundary as a real plan.
  iree_allocator_inline_storage_t storage = {};
  iree_arena_block_pool_t limited_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_inline_arena(&storage),
                                   &limited_pool);
  loom_low_lower_context_t limited_context = {};
  iree_arena_initialize(&limited_pool, &limited_context.analysis_arena);
  const auto plan = MakePlan();
  loom_low_lower_memory_report_row_t row = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_low_lower_memory_report_row_populate_source_interval(
          &limited_context, &plan, &row));
  EXPECT_EQ(limited_context.report.memory_expressions.root, nullptr);
  EXPECT_EQ(limited_context.report.memory_expressions.count, 0u);
  EXPECT_EQ(row.source_interval.begin_expr_id, LOOM_LOW_MEMORY_EXPR_ID_NONE);
  EXPECT_EQ(row.source_interval.end_expr_id, LOOM_LOW_MEMORY_EXPR_ID_NONE);
  EXPECT_FALSE(iree_any_bit_set(row.source_interval.precision_flags,
                                LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_EXPR |
                                    LOOM_LOW_BYTE_INTERVAL_PRECISION_END_EXPR));
  iree_arena_deinitialize(&limited_context.analysis_arena);
  iree_arena_block_pool_deinitialize(&limited_pool);
}

}  // namespace
}  // namespace loom
