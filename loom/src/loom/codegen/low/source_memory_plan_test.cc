// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class SourceMemoryPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    BuildFunction();
  }

  void TearDown() override {
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
    IREE_ASSERT_OK(loom_builder_intern_string(
        &module_builder, IREE_SV("source_memory"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    loom_op_t* func_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, 0, 0, 0, symbol, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    function_ = loom_func_like_cast(module_, func_op);
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_value_id_t DefineBufferArg() {
    loom_value_id_t buffer = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_buffer(), &buffer));
    return buffer;
  }

  loom_value_id_t DefineIndexArg() {
    loom_value_id_t index = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &index));
    return index;
  }

  loom_value_id_t DefineOffsetArg() {
    loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), &offset));
    return offset;
  }

  loom_value_id_t BuildNoalias(loom_value_id_t buffer) {
    loom_op_t* op = nullptr;
    const loom_type_t result_type = loom_type_buffer();
    IREE_CHECK_OK(loom_buffer_assume_noalias_build(
        &builder_, &buffer, 1, &result_type, 1, LOOM_LOCATION_UNKNOWN, &op));
    return loom_buffer_assume_noalias_results(op).values[0];
  }

  loom_value_id_t BuildAligned(loom_value_id_t buffer,
                               int64_t minimum_alignment) {
    loom_op_t* op = nullptr;
    const loom_type_t result_type = loom_type_buffer();
    IREE_CHECK_OK(loom_buffer_assume_alignment_build(
        &builder_, &buffer, 1, minimum_alignment, &result_type, 1,
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_buffer_assume_alignment_results(op).values[0];
  }

  loom_op_t* BuildOffsetConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_value_id_t BuildIndexAssumeRange(loom_value_id_t source,
                                        int64_t minimum_value,
                                        int64_t maximum_value) {
    loom_predicate_t* predicate = nullptr;
    IREE_CHECK_OK(iree_arena_allocate_array(
        &module_->arena, 1, sizeof(*predicate), (void**)&predicate));
    *predicate = (loom_predicate_t){
        /*.kind=*/LOOM_PREDICATE_RANGE,
        /*.arg_count=*/3,
        /*.arg_tags=*/
        {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST},
        /*.reserved=*/{},
        /*.args=*/{source, minimum_value, maximum_value},
    };
    const loom_type_t result_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_assume_build(&builder_, &source, 1, predicate, 1,
                                          &result_type, 1,
                                          LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_assume_results(op).values[0];
  }

  loom_value_id_t BuildDenseLayout() {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_encoding_layout_dense_build(
        &builder_,
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_dense_result(op);
  }

  loom_type_t ViewType1D(int64_t extent, loom_value_id_t layout) {
    loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                                           loom_dim_pack_static(extent), 0);
    type.encoding_id = (uint16_t)layout;
    type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return type;
  }

  loom_type_t ViewType2D(int64_t rows, int64_t columns,
                         loom_value_id_t layout) {
    loom_type_t type = loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                                           loom_dim_pack_static(rows),
                                           loom_dim_pack_static(columns), 0);
    type.encoding_id = (uint16_t)layout;
    type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return type;
  }

  loom_type_t VectorType1D(int64_t extent) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                               loom_dim_pack_static(extent), 0);
  }

  void ComputeFacts(loom_value_fact_table_t* out_facts) {
    IREE_ASSERT_OK(loom_value_fact_table_initialize(out_facts, &analysis_arena_,
                                                    module_->values.count));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(out_facts, module_, function_));
  }

  bool BuildPlan(const loom_value_fact_table_t* facts, const loom_op_t* op,
                 loom_low_source_memory_access_plan_t* out_plan,
                 loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
    return loom_low_source_memory_access_plan_build(module_, facts, op,
                                                    out_plan, out_diagnostic);
  }

  bool BuildViewPlan(
      const loom_value_fact_table_t* facts, loom_value_id_t view,
      loom_low_source_memory_access_plan_t* out_plan,
      loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
    loom_vector_memory_cache_policy_t cache_policy = {0};
    return loom_low_source_memory_access_plan_build_view(
        module_, facts, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, view,
        cache_policy, out_plan, out_diagnostic);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_;
  loom_builder_t builder_;
};

TEST_F(SourceMemoryPlanTest, StaticDenseLoadIncludesViewBase) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(16));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(32, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  int64_t static_indices[] = {3};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.operation_kind, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD);
  EXPECT_EQ(plan.view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.root_value_id, buffer);
  EXPECT_EQ(plan.memory_space, LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN);
  EXPECT_EQ(plan.element_byte_count, 4u);
  EXPECT_EQ(plan.vector_lane_count, 4u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 4);
  EXPECT_EQ(plan.static_byte_offset, 28);
  EXPECT_EQ(plan.static_view_base_byte_offset, 16);
  EXPECT_EQ(plan.root_minimum_alignment, 1u);
  EXPECT_EQ(plan.minimum_alignment, 1u);
  EXPECT_EQ(plan.dynamic_term_count, 0u);
  EXPECT_EQ(plan.dynamic_view_base_term_count, 0u);
}

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadTracksViewBaseBoundary) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t base = DefineOffsetArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t header = loom_index_constant_result(BuildOffsetConstant(16));
  loom_op_t* base_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, base, header,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                      LOOM_LOCATION_UNKNOWN, &base_op));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, buffer, loom_index_add_result(base_op), ViewType1D(32, layout),
      LOOM_LOCATION_UNKNOWN, &view_op));
  int64_t static_indices[] = {3};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 28);
  EXPECT_EQ(plan.static_view_base_byte_offset, 16);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_view_base_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, base);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  EXPECT_EQ(plan.dynamic_terms[0].axis,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 1);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 0u);
}

TEST_F(SourceMemoryPlanTest, StaticOffsetCombinesWithRootAlignment) {
  loom_value_id_t buffer = BuildAligned(DefineBufferArg(), 16);
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(8));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(32, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  int64_t static_indices[] = {1};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.root_minimum_alignment, 16u);
  EXPECT_EQ(plan.static_byte_offset, 12);
  EXPECT_EQ(plan.static_view_base_byte_offset, 8);
  EXPECT_EQ(plan.minimum_alignment, 4u);
}

TEST_F(SourceMemoryPlanTest, ExternalBufferArgHasNoComparableAliasScope) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(16));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(32, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  int64_t static_indices[] = {3};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.root_value_id, buffer);
  EXPECT_EQ(plan.alias_scope_id, LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE);

  loom_low_byte_interval_t interval = {};
  loom_low_memory_access_summary_t summary = {};
  loom_low_source_memory_access_plan_make_summary(&plan, &interval, &summary);
  EXPECT_FALSE(iree_any_bit_set(summary.precision_flags,
                                LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT));
  EXPECT_EQ(summary.alias_root_id, LOOM_LOW_MEMORY_ALIAS_ID_NONE);
}

TEST_F(SourceMemoryPlanTest, NoaliasBufferArgFeedsComparableAliasScope) {
  loom_value_id_t buffer = BuildNoalias(DefineBufferArg());
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(16));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(32, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  int64_t static_indices[] = {3};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_NE(plan.alias_scope_id, LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE);

  loom_low_byte_interval_t interval = {};
  loom_low_memory_access_summary_t summary = {};
  loom_low_source_memory_access_plan_make_summary(&plan, &interval, &summary);
  EXPECT_TRUE(iree_any_bit_set(summary.precision_flags,
                               LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT));
  EXPECT_EQ(summary.alias_root_id, plan.alias_scope_id);
}

TEST_F(SourceMemoryPlanTest, StaticDenseScalarLoadUsesMemoryAccessFacet) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(16));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(32, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  int64_t static_indices[] = {3};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_,
      LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE |
          LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL,
      loom_buffer_view_result(view_op), nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), LOOM_CACHE_SCOPE_DEVICE,
      LOOM_CACHE_TEMPORAL_NON_TEMPORAL, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.operation_kind, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD);
  EXPECT_EQ(plan.view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.root_value_id, buffer);
  EXPECT_EQ(plan.element_byte_count, 4u);
  EXPECT_EQ(plan.vector_lane_count, 1u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 4);
  EXPECT_EQ(plan.static_byte_offset, 28);
  EXPECT_EQ(plan.static_view_base_byte_offset, 16);
  EXPECT_EQ(plan.cache_policy.build_flags,
            LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE |
                LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL);
  EXPECT_EQ(plan.cache_policy.cache_scope, LOOM_CACHE_SCOPE_DEVICE);
  EXPECT_EQ(plan.cache_policy.cache_temporal, LOOM_CACHE_TEMPORAL_NON_TEMPORAL);
}

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadClassifiesWorkitemIndex) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(8));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(32, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* workitem_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &workitem_op));
  const loom_value_id_t dynamic_indices[] = {
      loom_kernel_workitem_id_result(workitem_op),
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 8);
  EXPECT_EQ(plan.static_view_base_byte_offset, 8);
  EXPECT_EQ(plan.dynamic_view_base_term_count, 0u);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index,
            loom_kernel_workitem_id_result(workitem_op));
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 2u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 4);

  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, 112);
  EXPECT_TRUE(loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(
      &plan.dynamic_terms[0], 32));
  EXPECT_TRUE(loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      &plan, /*static_byte_offset=*/plan.static_byte_offset, 32));
}

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadKeepsAssumedWorkitemSource) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(8));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(32, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* workitem_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &workitem_op));
  const loom_value_id_t assumed_index =
      BuildIndexAssumeRange(loom_kernel_workitem_id_result(workitem_op), 0, 7);
  const loom_value_id_t dynamic_indices[] = {
      assumed_index,
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 8);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, assumed_index);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 2u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, 28);
}

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadFactorsScaledWorkitemIndex) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(8));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(128, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* workitem_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &workitem_op));
  loom_value_id_t two = loom_index_constant_result(BuildIndexConstant(2));
  loom_op_t* scaled_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(
      &builder_, loom_kernel_workitem_id_result(workitem_op), two,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN,
      &scaled_op));
  const loom_value_id_t dynamic_indices[] = {
      loom_index_mul_result(scaled_op),
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 8);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index,
            loom_kernel_workitem_id_result(workitem_op));
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 8);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 3u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, 504);
}

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadFactorsScaledOffsetWorkitemIndex) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(8));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(128, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* workitem_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &workitem_op));
  loom_value_id_t two = loom_index_constant_result(BuildIndexConstant(2));
  loom_op_t* scaled_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(
      &builder_, loom_kernel_workitem_id_result(workitem_op), two,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN,
      &scaled_op));
  loom_value_id_t one = loom_index_constant_result(BuildIndexConstant(1));
  loom_op_t* offset_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_,
                                      loom_index_mul_result(scaled_op), one,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &offset_op));
  const loom_value_id_t dynamic_indices[] = {
      loom_index_add_result(offset_op),
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 12);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index,
            loom_kernel_workitem_id_result(workitem_op));
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 8);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 3u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, 504);
}

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadFactorsMaddWorkitemIndex) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(8));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(128, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* workitem_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &workitem_op));
  loom_value_id_t two = loom_index_constant_result(BuildIndexConstant(2));
  loom_value_id_t one = loom_index_constant_result(BuildIndexConstant(1));
  loom_op_t* offset_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(
      &builder_, loom_kernel_workitem_id_result(workitem_op), two, one,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN,
      &offset_op));
  const loom_value_id_t dynamic_indices[] = {
      loom_index_madd_result(offset_op),
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 12);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index,
            loom_kernel_workitem_id_result(workitem_op));
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 8);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 3u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, 504);
}

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadClassifiesMultipleIndices) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t first_index = DefineIndexArg();
  loom_value_id_t second_index = DefineIndexArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(8, 8, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  const loom_value_id_t dynamic_indices[] = {
      first_index,
      second_index,
  };
  int64_t static_indices[] = {INT64_MIN, INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 0);
  ASSERT_EQ(plan.dynamic_term_count, 2u);
  EXPECT_EQ(plan.dynamic_terms[0].index, first_index);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 32);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, 224);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 5u);
  EXPECT_EQ(plan.dynamic_terms[1].index, second_index);
  EXPECT_EQ(plan.dynamic_terms[1].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE);
  EXPECT_EQ(plan.dynamic_terms[1].dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  EXPECT_EQ(plan.dynamic_terms[1].axis, 1u);
  EXPECT_EQ(plan.dynamic_terms[1].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[1].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[1].byte_facts.range_hi, 16);
  EXPECT_EQ(plan.dynamic_terms[1].byte_shift, 2u);
  EXPECT_TRUE(loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      &plan, /*static_byte_offset=*/plan.static_byte_offset, 32));
}

TEST_F(SourceMemoryPlanTest, ScalarViewAccessClassifiesCoordinateAxes) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(16, 64, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* row_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workgroup_id_build(&builder_, LOOM_KERNEL_DIMENSION_Y,
                                     loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                     LOOM_LOCATION_UNKNOWN, &row_op));
  const loom_value_id_t row =
      BuildIndexAssumeRange(loom_kernel_workgroup_id_result(row_op), 0, 15);
  loom_op_t* column_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &column_op));
  const loom_value_id_t column =
      BuildIndexAssumeRange(loom_kernel_workitem_id_result(column_op), 0, 63);
  const loom_value_id_t dynamic_indices[] = {
      row,
      column,
  };
  int64_t static_indices[] = {INT64_MIN, INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &load_op));
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_view_store_build(
      &builder_, 0, loom_view_load_result(load_op),
      loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, LOOM_LOCATION_UNKNOWN, &store_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  for (const loom_op_t* op : {load_op, store_op}) {
    loom_low_source_memory_access_plan_t plan = {};
    loom_low_source_memory_access_diagnostic_t diagnostic = {0};
    ASSERT_TRUE(BuildPlan(&facts, op, &plan, &diagnostic));
    EXPECT_EQ(plan.static_byte_offset, 0);
    ASSERT_EQ(plan.dynamic_term_count, 2u);
    EXPECT_EQ(plan.dynamic_terms[0].index, row);
    EXPECT_EQ(plan.dynamic_terms[0].source,
              LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID);
    EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_Y);
    EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
    EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 256);
    EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 8u);
    EXPECT_EQ(plan.dynamic_terms[1].index, column);
    EXPECT_EQ(plan.dynamic_terms[1].source,
              LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
    EXPECT_EQ(plan.dynamic_terms[1].dimension, LOOM_KERNEL_DIMENSION_X);
    EXPECT_EQ(plan.dynamic_terms[1].axis, 1u);
    EXPECT_EQ(plan.dynamic_terms[1].byte_stride, 4);
    EXPECT_EQ(plan.dynamic_terms[1].byte_shift, 2u);
    EXPECT_EQ(plan.vector_lane_count, 1u);
    EXPECT_EQ(plan.vector_lane_byte_stride, 4);
  }
}

TEST_F(SourceMemoryPlanTest, LinearizedScalarViewLoadRecoversCoordinateTerms) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(512, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* block_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workgroup_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                     loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                     LOOM_LOCATION_UNKNOWN, &block_op));
  const loom_value_id_t block =
      BuildIndexAssumeRange(loom_kernel_workgroup_id_result(block_op), 0, 1);
  loom_op_t* row_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workgroup_id_build(&builder_, LOOM_KERNEL_DIMENSION_Y,
                                     loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                     LOOM_LOCATION_UNKNOWN, &row_op));
  const loom_value_id_t row =
      BuildIndexAssumeRange(loom_kernel_workgroup_id_result(row_op), 0, 3);
  loom_op_t* lane_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &lane_op));
  const loom_value_id_t lane =
      BuildIndexAssumeRange(loom_kernel_workitem_id_result(lane_op), 0, 63);
  const loom_value_id_t row_scale =
      loom_index_constant_result(BuildIndexConstant(64));
  loom_op_t* row_linear_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, row, row_scale, lane,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &row_linear_op));
  const loom_value_id_t block_scale =
      loom_index_constant_result(BuildIndexConstant(256));
  loom_op_t* linear_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, block, block_scale,
                                       loom_index_madd_result(row_linear_op),
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &linear_op));

  const loom_value_id_t dynamic_indices[] = {loom_index_madd_result(linear_op)};
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 0);
  ASSERT_EQ(plan.dynamic_term_count, 3u);
  EXPECT_EQ(plan.dynamic_terms[0].index, block);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[0].axis,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 1024);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 10u);
  EXPECT_EQ(plan.dynamic_terms[1].index, row);
  EXPECT_EQ(plan.dynamic_terms[1].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID);
  EXPECT_EQ(plan.dynamic_terms[1].dimension, LOOM_KERNEL_DIMENSION_Y);
  EXPECT_EQ(plan.dynamic_terms[1].axis,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE);
  EXPECT_EQ(plan.dynamic_terms[1].byte_stride, 256);
  EXPECT_EQ(plan.dynamic_terms[1].byte_shift, 8u);
  EXPECT_EQ(plan.dynamic_terms[2].index, lane);
  EXPECT_EQ(plan.dynamic_terms[2].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[2].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[2].axis,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE);
  EXPECT_EQ(plan.dynamic_terms[2].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[2].byte_shift, 2u);
  EXPECT_EQ(plan.vector_lane_count, 1u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 4);
}

TEST_F(SourceMemoryPlanTest, LinearizedWorkitemViewLoadKeepsSingleTerm) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(16, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* row_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_Y,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &row_op));
  const loom_value_id_t row =
      BuildIndexAssumeRange(loom_kernel_workitem_id_result(row_op), 0, 3);
  loom_op_t* column_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &column_op));
  const loom_value_id_t column =
      BuildIndexAssumeRange(loom_kernel_workitem_id_result(column_op), 0, 3);
  const loom_value_id_t row_scale =
      loom_index_constant_result(BuildIndexConstant(4));
  loom_op_t* linear_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, row, row_scale, column,
                                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                       LOOM_LOCATION_UNKNOWN, &linear_op));

  const loom_value_id_t dynamic_indices[] = {loom_index_madd_result(linear_op)};
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 0);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, loom_index_madd_result(linear_op));
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 2u);
  EXPECT_EQ(plan.vector_lane_count, 1u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 4);
}

TEST_F(SourceMemoryPlanTest, ExactDynamicIndexFoldsIntoStaticOffset) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t first_index = DefineIndexArg();
  loom_value_id_t second_index =
      loom_index_constant_result(BuildIndexConstant(3));
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(8));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(8, 8, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  const loom_value_id_t dynamic_indices[] = {
      first_index,
      second_index,
  };
  int64_t static_indices[] = {INT64_MIN, INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 20);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, first_index);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 32);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 5u);
}

TEST_F(SourceMemoryPlanTest, WholeRank1ViewPlanIncludesBase) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(16));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(4, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildViewPlan(&facts, loom_buffer_view_result(view_op), &plan,
                            &diagnostic));
  EXPECT_EQ(plan.operation_kind, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD);
  EXPECT_EQ(plan.view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.root_value_id, buffer);
  EXPECT_EQ(plan.memory_space, LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN);
  EXPECT_EQ(plan.element_byte_count, 4u);
  EXPECT_EQ(plan.vector_lane_count, 4u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 4);
  EXPECT_EQ(plan.static_byte_offset, 16);
  EXPECT_EQ(plan.dynamic_term_count, 0u);
}

TEST_F(SourceMemoryPlanTest, SubviewPlanClassifiesWorkitemRow) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(8));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(64, 4, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* workitem_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &workitem_op));
  const loom_value_id_t dynamic_offsets[] = {
      loom_kernel_workitem_id_result(workitem_op),
  };
  int64_t static_offsets[] = {INT64_MIN, 0};
  loom_op_t* subview_op = nullptr;
  IREE_ASSERT_OK(loom_view_subview_build(
      &builder_, loom_buffer_view_result(view_op), dynamic_offsets,
      IREE_ARRAYSIZE(dynamic_offsets), static_offsets,
      IREE_ARRAYSIZE(static_offsets), ViewType2D(1, 4, layout),
      LOOM_LOCATION_UNKNOWN, &subview_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildViewPlan(&facts, loom_view_subview_result(subview_op), &plan,
                            &diagnostic));
  EXPECT_EQ(plan.view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.root_value_id, buffer);
  EXPECT_EQ(plan.static_byte_offset, 8);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index,
            loom_kernel_workitem_id_result(workitem_op));
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 16);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, 1008);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 4u);
  EXPECT_EQ(plan.vector_lane_count, 4u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 4);
}

TEST_F(SourceMemoryPlanTest, WholeRank2ViewPlanFlattensStaticFootprint) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(2, 2, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildViewPlan(&facts, loom_buffer_view_result(view_op), &plan,
                            &diagnostic));
  EXPECT_EQ(plan.element_byte_count, 4u);
  EXPECT_EQ(plan.vector_lane_count, 4u);
  EXPECT_EQ(plan.static_byte_offset, 0);
  EXPECT_EQ(plan.dynamic_term_count, 0u);
}

}  // namespace
}  // namespace loom
