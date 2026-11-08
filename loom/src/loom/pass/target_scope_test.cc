// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"

namespace loom {
namespace {

class TargetScopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("target_scope"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_pass_value_fact_owner_initialize(&block_pool_, &fact_owner_);
    IREE_ASSERT_OK(BuildTarget(IREE_SV("construction"), 64, &construction_));
    IREE_ASSERT_OK(BuildTarget(IREE_SV("worker"), 32, &worker_));
    loom_symbol_ref_t pipeline = {};
    IREE_ASSERT_OK(AddSymbol(IREE_SV("pipeline"), &pipeline));
    loom_op_t* definition = nullptr;
    IREE_ASSERT_OK(loom_pipeline_def_build(
        &builder_, LOOM_PIPELINE_DEF_BUILD_FLAG_HAS_TARGET, 0, 0, 0,
        construction_, pipeline, nullptr, 0, nullptr, 0, nullptr, 0,
        LOOM_LOCATION_UNKNOWN, &definition));
    function_ = loom_func_like_cast(module_, definition);
    loom_builder_enter_region(&builder_, definition,
                              loom_pipeline_def_body(definition));
    const int64_t origin = 0;
    const int64_t count = 1;
    const int64_t stride = 1;
    IREE_ASSERT_OK(loom_pipeline_strand_build(
        &builder_, LOOM_PIPELINE_STRAND_BUILD_FLAG_HAS_TARGET, 0, worker_,
        nullptr, 0, &origin, 1, nullptr, 0, &count, 1, nullptr, 0, &stride, 1,
        LOOM_LOCATION_UNKNOWN, &strand_));
    loom_builder_enter_region(&builder_, strand_,
                              loom_pipeline_strand_body(strand_));
    IREE_ASSERT_OK(loom_target_subgroup_size_build(
        &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &query_));
    loom_op_t* terminator = nullptr;
    IREE_ASSERT_OK(
        loom_pipeline_end_build(&builder_, LOOM_LOCATION_UNKNOWN, &terminator));
    loom_builder_enter_region(&builder_, definition,
                              loom_pipeline_def_body(definition));
    IREE_ASSERT_OK(loom_pipeline_finish_build(&builder_, LOOM_LOCATION_UNKNOWN,
                                              &terminator));
  }

  void TearDown() override {
    loom_pass_value_fact_owner_deinitialize(&fact_owner_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t AddSymbol(iree_string_view_t name,
                          loom_symbol_ref_t* out_symbol) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(module_, name, &name_id));
    out_symbol->module_id = 0;
    return loom_module_add_symbol(module_, name_id, &out_symbol->symbol_id);
  }

  iree_status_t BuildTarget(iree_string_view_t name, int64_t width,
                            loom_symbol_ref_t* out_symbol) {
    IREE_RETURN_IF_ERROR(AddSymbol(name, out_symbol));
    loom_op_t* target = nullptr;
    return loom_test_target_build(
        &builder_, LOOM_TEST_TARGET_BUILD_FLAG_HAS_SUBGROUP_SIZE,
        LOOM_TEST_TARGET_KIND_LOW_CORE, *out_symbol,
        /*codegen_format=*/0, /*artifact_format=*/0,
        /*default_pointer_bitwidth=*/0, /*index_bitwidth=*/0,
        /*offset_bitwidth=*/0,
        /*max_workgroup_size_x=*/0, /*max_workgroup_size_y=*/0,
        /*max_workgroup_size_z=*/0,
        /*max_flat_workgroup_size=*/0, /*max_workgroup_storage_bytes=*/0,
        /*subgroup_size=*/width, /*max_grid_size_x=*/0, /*max_grid_size_y=*/0,
        /*max_grid_size_z=*/0, /*max_flat_grid_size=*/0,
        /*max_workgroup_count_x=*/0,
        /*max_workgroup_count_y=*/0, /*max_workgroup_count_z=*/0,
        /*memory_space_generic=*/0, /*memory_space_global=*/0,
        /*memory_space_workgroup=*/0,
        /*memory_space_constant=*/0, /*memory_space_private=*/0,
        /*memory_space_host=*/0,
        /*memory_space_descriptor=*/0, /*abi=*/0,
        /*export_symbol=*/0, /*linkage=*/0, /*contract_set_key=*/0,
        /*contract_feature_bits=*/0, LOOM_LOCATION_UNKNOWN, &target);
  }

  // Allocation backing for the module and analysis scopes.
  iree_arena_block_pool_t block_pool_ = {};
  // Registered production and test operation schemas.
  loom_context_t context_ = {};
  // Owned valid pipeline source fixture.
  loom_module_t* module_ = nullptr;
  // Builder used while constructing the fixture.
  loom_builder_t builder_ = {};
  // Owner of target projections and value facts.
  loom_pass_value_fact_owner_t fact_owner_ = {};
  // Target used by pipeline construction.
  loom_symbol_ref_t construction_ = {};
  // Target used by the independently executing strand.
  loom_symbol_ref_t worker_ = {};
  // Function-like handle for the pipeline definition.
  loom_func_like_t function_ = {};
  // Independent region whose target can be rebound.
  loom_op_t* strand_ = nullptr;
  // Target observation used to compare fresh and incremental inference.
  loom_op_t* query_ = nullptr;
};

TEST_F(TargetScopeTest, IndependentRegionCanBeAnalyzedAsRoot) {
  loom_value_fact_table_t* table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &fact_owner_, module_,
      loom_pass_value_fact_scope_region(
          function_, loom_pipeline_strand_body(strand_), strand_),
      &table));
  const auto facts = loom_value_fact_table_lookup(
      table, loom_target_subgroup_size_result(query_));
  ASSERT_TRUE(loom_value_facts_is_exact(facts));
  EXPECT_EQ(facts.range_lo, 32);
}

TEST_F(TargetScopeTest, RebindingRegionRefreshesFactsBeforeQueuedRewrites) {
  loom_value_fact_table_t* table = nullptr;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &fact_owner_, module_, loom_pass_value_fact_scope_function(function_),
      &table));
  loom_rewriter_t rewriter = {};
  loom_rewriter_initialize(&rewriter, module_, &module_->arena);
  loom_rewriter_attach_value_facts(&rewriter, table);
  const auto result = loom_target_subgroup_size_result(query_);
  EXPECT_EQ(loom_value_fact_table_lookup(table, result).range_lo, 32);
  IREE_ASSERT_OK(loom_rewriter_set_attr(&rewriter, strand_, 0,
                                        loom_attr_symbol(construction_)));
  EXPECT_EQ(loom_value_fact_table_lookup(table, result).range_lo, 64);
  IREE_ASSERT_OK(
      loom_rewriter_set_attr(&rewriter, strand_, 0, loom_attr_absent()));
  // The enclosing analysis scope has no supplied target facts.
  EXPECT_FALSE(
      loom_value_facts_is_exact(loom_value_fact_table_lookup(table, result)));
  loom_rewriter_deinitialize(&rewriter);
}

}  // namespace
}  // namespace loom
