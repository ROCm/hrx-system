// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/report.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
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
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
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
  options_.report_allocator = iree_allocator_system();
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));

  EXPECT_EQ(result_.selected_source_op_count, 1u);
  EXPECT_EQ(result_.emitted_low_op_count, 1u);
  ASSERT_EQ(result_.report_rows.count, 1u);
  ASSERT_NE(result_.report_rows.head, nullptr);
  ASSERT_EQ(result_.report_rows.head, result_.report_rows.tail);
  ASSERT_EQ(result_.report_rows.head->count, 1u);
  const loom_low_lower_report_row_t& row =
      loom_low_lower_report_row_vec_const_rows(result_.report_rows.head)[0];
  EXPECT_TRUE(iree_string_view_equal(row.function_name, IREE_SV("add")));
  EXPECT_EQ(row.source_op_kind, LOOM_OP_SCALAR_ADDI);
  EXPECT_EQ(row.selection_kind, LOOM_LOW_LOWER_REPORT_SELECTION_RULE);
  EXPECT_EQ(row.emitted_low_op_count, 1u);
  EXPECT_EQ(row.execution_count_plus_one, 2u);
  EXPECT_TRUE(iree_string_view_equal(row.descriptor_semantic_tag,
                                     IREE_SV("integer.add.i32")));

  loom_low_lower_result_deinitialize(&result_);
  EXPECT_EQ(result_.report_rows.count, 0u);
  EXPECT_EQ(result_.report_rows.head, nullptr);
  EXPECT_EQ(result_.report_rows.tail, nullptr);
  EXPECT_TRUE(iree_allocator_is_null(result_.report_allocator));
  EXPECT_TRUE(iree_allocator_is_null(result_.memory_report_row_allocator));
}

TEST_F(LowLowerReportTest, DisabledReportsProduceNoRows) {
  IREE_ASSERT_OK(
      loom_low_lower_function(module_, function_, &options_, &result_));

  EXPECT_EQ(result_.selected_source_op_count, 0u);
  EXPECT_EQ(result_.emitted_low_op_count, 0u);
  EXPECT_EQ(result_.report_rows.count, 0u);
  EXPECT_EQ(result_.report_rows.head, nullptr);
  EXPECT_EQ(result_.memory_report_rows.count, 0u);
  EXPECT_EQ(result_.memory_report_rows.rows, nullptr);
}

}  // namespace
}  // namespace loom
