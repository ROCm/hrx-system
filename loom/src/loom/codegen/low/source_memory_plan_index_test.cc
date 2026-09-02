// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan_test_fixture.h"

namespace loom {
namespace {

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
      /*instance_flags=*/0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), LOOM_CACHE_SCOPE_DEVICE,
      LOOM_CACHE_TEMPORAL_NON_TEMPORAL, loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.operation_kind, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD);
  EXPECT_EQ(plan.view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.base_view_value_id, loom_buffer_view_result(view_op));
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
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
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

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadRetainsSubgroupUniformIndex) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(32, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_op_t* subgroup_op = nullptr;
  IREE_ASSERT_OK(loom_kernel_subgroup_id_build(
      &builder_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &subgroup_op));
  const loom_value_id_t subgroup_id =
      loom_kernel_subgroup_id_result(subgroup_op);
  const loom_value_id_t dynamic_indices[] = {subgroup_id};
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  ASSERT_TRUE(loom_value_facts_is_subgroup_uniform(
      loom_value_fact_table_lookup(&facts, subgroup_id)));
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, subgroup_id);
  EXPECT_TRUE(
      loom_value_facts_is_subgroup_uniform(plan.dynamic_terms[0].byte_facts));
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
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 8);
  EXPECT_FALSE(plan.source_index_static_offset_extracted);
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

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadClassifiesForwardedWorkitemIndex) {
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
  const loom_value_id_t forwarded =
      BuildForwardedIndexBlockArg(loom_kernel_workitem_id_result(workitem_op));
  const loom_value_id_t dynamic_indices[] = {
      forwarded,
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, forwarded);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 4);
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
      LOOM_LOCATION_UNKNOWN, &scaled_op));
  const loom_value_id_t dynamic_indices[] = {
      loom_index_mul_result(scaled_op),
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 8);
  EXPECT_FALSE(plan.source_index_static_offset_extracted);
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
      LOOM_LOCATION_UNKNOWN, &scaled_op));
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
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 12);
  EXPECT_TRUE(plan.source_index_static_offset_extracted);
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
      LOOM_LOCATION_UNKNOWN, &offset_op));
  const loom_value_id_t dynamic_indices[] = {
      loom_index_madd_result(offset_op),
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 12);
  EXPECT_TRUE(plan.source_index_static_offset_extracted);
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

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadFactorsDeepAffineIndex) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t source_index = DefineIndexArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(256, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  loom_value_id_t one = loom_index_constant_result(BuildIndexConstant(1));
  loom_value_id_t deep_index = source_index;
  for (int i = 0; i < 40; ++i) {
    loom_op_t* add_op = nullptr;
    IREE_ASSERT_OK(loom_index_add_build(
        &builder_, deep_index, one, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &add_op));
    deep_index = loom_index_add_result(add_op);
  }
  const loom_value_id_t dynamic_indices[] = {deep_index};
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 160);
  EXPECT_TRUE(plan.source_index_static_offset_extracted);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, source_index);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 2u);
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
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
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

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadRetainsOffsetAcrossDynamicStride) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t row = DefineIndexArg();
  loom_value_id_t row_component = DefineIndexArg();
  loom_value_id_t column_count = DefineIndexArg();
  loom_value_id_t column = DefineIndexArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_type_t view_type = loom_type_shaped_2d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F16, loom_dim_pack_static(16),
      loom_dim_pack_dynamic(column_count), /*encoding_id=*/0);
  view_type.encoding_id = (uint16_t)layout;
  view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        view_type, LOOM_LOCATION_UNKNOWN,
                                        &view_op));

  loom_value_id_t one = loom_index_constant_result(BuildIndexConstant(1));
  loom_op_t* row_sum_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, row, row_component,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &row_sum_op));
  loom_op_t* offset_row_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_,
                                      loom_index_add_result(row_sum_op), one,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &offset_row_op));
  const loom_value_id_t dynamic_indices[] = {
      loom_index_add_result(offset_row_op),
      column,
  };
  const int64_t static_indices[] = {INT64_MIN, INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      VectorType1D(LOOM_SCALAR_TYPE_F16, 1), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_id_t two = loom_index_constant_result(BuildIndexConstant(2));
  loom_op_t* scaled_row_op = nullptr;
  IREE_ASSERT_OK(loom_index_mul_build(&builder_, row, two,
                                      LOOM_LOCATION_UNKNOWN, &scaled_row_op));
  const loom_value_id_t scaled_dynamic_indices[] = {
      loom_index_mul_result(scaled_row_op),
      column,
  };
  loom_op_t* scaled_load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      scaled_dynamic_indices, IREE_ARRAYSIZE(scaled_dynamic_indices),
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0,
      VectorType1D(LOOM_SCALAR_TYPE_F16, 1), LOOM_LOCATION_UNKNOWN,
      &scaled_load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 0);
  EXPECT_FALSE(plan.source_index_static_offset_extracted);
  ASSERT_EQ(plan.dynamic_term_count, 2u);
  EXPECT_EQ(plan.dynamic_terms[0].index, loom_index_add_result(offset_row_op));
  EXPECT_EQ(plan.dynamic_terms[0].axis, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 2);
  ASSERT_EQ(plan.dynamic_terms[0].stride_value_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].stride_values[0], column_count);
  EXPECT_EQ(plan.dynamic_terms[1].index, column);
  EXPECT_EQ(plan.dynamic_terms[1].axis, 1u);
  EXPECT_EQ(plan.dynamic_terms[1].byte_stride, 2);
  EXPECT_EQ(plan.dynamic_terms[1].stride_value_count, 0u);

  loom_low_source_memory_access_plan_t scaled_plan = {};
  ASSERT_TRUE(BuildPlan(&facts, scaled_load_op, &scaled_plan, &diagnostic));
  EXPECT_EQ(scaled_plan.static_byte_offset, 0);
  EXPECT_FALSE(scaled_plan.source_index_static_offset_extracted);
  ASSERT_EQ(scaled_plan.dynamic_term_count, 2u);
  EXPECT_EQ(scaled_plan.dynamic_terms[0].index, row);
  EXPECT_EQ(scaled_plan.dynamic_terms[0].byte_stride, 4);
  ASSERT_EQ(scaled_plan.dynamic_terms[0].stride_value_count, 1u);
  EXPECT_EQ(scaled_plan.dynamic_terms[0].stride_values[0], column_count);
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
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &load_op));
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_view_store_build(
      &builder_, 0, /*instance_flags=*/0, loom_view_load_result(load_op),
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
                                       LOOM_LOCATION_UNKNOWN, &row_linear_op));
  const loom_value_id_t block_scale =
      loom_index_constant_result(BuildIndexConstant(256));
  loom_op_t* linear_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, block, block_scale,
                                       loom_index_madd_result(row_linear_op),
                                       LOOM_LOCATION_UNKNOWN, &linear_op));

  const loom_value_id_t dynamic_indices[] = {loom_index_madd_result(linear_op)};
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 0);
  EXPECT_FALSE(plan.source_index_static_offset_extracted);
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

TEST_F(SourceMemoryPlanTest,
       LinearizedUniformAndVaryingViewLoadCanonicalizesPrefixTerms) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(4096, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  const loom_value_id_t stage = DefineIndexArg();
  loom_op_t* lane_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &lane_op));
  const loom_value_id_t lane =
      BuildIndexAssumeRange(loom_kernel_workitem_id_result(lane_op), 0, 31);
  const loom_value_id_t stage_scale =
      loom_index_constant_result(BuildIndexConstant(64));
  loom_op_t* staged_lane_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, stage, stage_scale, lane,
                                       LOOM_LOCATION_UNKNOWN, &staged_lane_op));
  const loom_value_id_t row_offset =
      loom_index_constant_result(BuildIndexConstant(256));
  loom_op_t* linear_op = nullptr;
  IREE_ASSERT_OK(
      loom_index_add_build(&builder_, loom_index_madd_result(staged_lane_op),
                           row_offset, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                           LOOM_LOCATION_UNKNOWN, &linear_op));

  const loom_value_id_t dynamic_indices[] = {
      loom_index_add_result(linear_op),
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_value_facts_t stage_facts = loom_value_facts_make(0, 1, 1);
  loom_value_facts_mark_workgroup_uniform(&stage_facts);
  IREE_ASSERT_OK(loom_value_fact_table_define(&facts, stage, stage_facts));
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 1024);
  EXPECT_TRUE(plan.source_index_static_offset_extracted);
  ASSERT_EQ(plan.dynamic_term_count, 2u);
  EXPECT_EQ(plan.dynamic_terms[0].index, stage);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 256);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 8u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, 256);
  EXPECT_EQ(plan.dynamic_terms[1].index, lane);
  EXPECT_EQ(plan.dynamic_terms[1].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[1].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[1].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[1].byte_shift, 2u);
  EXPECT_EQ(plan.dynamic_terms[1].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[1].byte_facts.range_hi, 124);
  ASSERT_EQ(plan.dynamic_realization_count, 1u);
  EXPECT_EQ(plan.dynamic_realizations[0].term.index,
            loom_index_madd_result(staged_lane_op));
  EXPECT_EQ(plan.dynamic_realizations[0].term.byte_stride, 4);
  EXPECT_EQ(plan.dynamic_realizations[0].term.byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_realizations[0].term.byte_facts.range_hi, 380);
  EXPECT_EQ(plan.dynamic_realizations[0].first_term, 0u);
  EXPECT_EQ(plan.dynamic_realizations[0].term_count, 2u);
  EXPECT_TRUE(loom_value_facts_is_lane_varying(
      loom_value_fact_table_lookup(&facts, plan.dynamic_terms[1].index)));
}

TEST_F(SourceMemoryPlanTest,
       LinearizedConstrainedAndVaryingViewLoadPreservesAggregateFacts) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(4096, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  const loom_value_id_t unknown = DefineIndexArg();
  loom_op_t* lane_op = nullptr;
  IREE_ASSERT_OK(
      loom_kernel_workitem_id_build(&builder_, LOOM_KERNEL_DIMENSION_X,
                                    loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                    LOOM_LOCATION_UNKNOWN, &lane_op));
  const loom_value_id_t scale =
      loom_index_constant_result(BuildIndexConstant(64));
  loom_op_t* linear_op = nullptr;
  IREE_ASSERT_OK(loom_index_madd_build(&builder_, unknown, scale,
                                       loom_kernel_workitem_id_result(lane_op),
                                       LOOM_LOCATION_UNKNOWN, &linear_op));
  const loom_value_id_t constrained_linear =
      BuildIndexAssumeRange(loom_index_madd_result(linear_op), 0, 1023);

  const loom_value_id_t dynamic_indices[] = {
      constrained_linear,
  };
  int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 0);
  ASSERT_EQ(plan.dynamic_term_count, 2u);
  EXPECT_EQ(plan.dynamic_terms[0].index, unknown);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 256);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 8u);
  EXPECT_EQ(plan.dynamic_terms[1].index,
            loom_kernel_workitem_id_result(lane_op));
  EXPECT_EQ(plan.dynamic_terms[1].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(plan.dynamic_terms[1].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(plan.dynamic_terms[1].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[1].byte_shift, 2u);
  ASSERT_EQ(plan.dynamic_realization_count, 1u);
  EXPECT_EQ(plan.dynamic_realizations[0].term.index, constrained_linear);
  EXPECT_EQ(plan.dynamic_realizations[0].term.byte_stride, 4);
  EXPECT_EQ(plan.dynamic_realizations[0].term.byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_realizations[0].term.byte_facts.range_hi, 4092);
  EXPECT_EQ(plan.dynamic_realizations[0].first_term, 0u);
  EXPECT_EQ(plan.dynamic_realizations[0].term_count, 2u);
  const loom_value_facts_t offset_facts =
      loom_low_source_memory_dynamic_offset_facts(&plan,
                                                  plan.static_byte_offset);
  EXPECT_EQ(offset_facts.range_lo, 0);
  EXPECT_EQ(offset_facts.range_hi, 4092);
  EXPECT_TRUE(loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      &plan, plan.static_byte_offset, 12));
}

TEST_F(SourceMemoryPlanTest,
       RankedAndLinearizedWorkitemIndicesShareCanonicalAddressTerms) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* ranked_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, buffer, base_offset, ViewType2D(4, 4, layout),
      LOOM_LOCATION_UNKNOWN, &ranked_view_op));
  loom_op_t* linear_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, buffer, base_offset, ViewType1D(16, layout),
      LOOM_LOCATION_UNKNOWN, &linear_view_op));
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
                                       LOOM_LOCATION_UNKNOWN, &linear_op));

  const loom_value_id_t ranked_dynamic_indices[] = {row, column};
  int64_t ranked_static_indices[] = {INT64_MIN, INT64_MIN};
  loom_op_t* ranked_load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, /*instance_flags=*/0,
      loom_buffer_view_result(ranked_view_op), ranked_dynamic_indices,
      IREE_ARRAYSIZE(ranked_dynamic_indices), ranked_static_indices,
      IREE_ARRAYSIZE(ranked_static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN,
      &ranked_load_op));
  const loom_value_id_t linear_dynamic_indices[] = {
      loom_index_madd_result(linear_op)};
  int64_t linear_static_indices[] = {INT64_MIN};
  loom_op_t* linear_load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, /*instance_flags=*/0,
      loom_buffer_view_result(linear_view_op), linear_dynamic_indices,
      IREE_ARRAYSIZE(linear_dynamic_indices), linear_static_indices,
      IREE_ARRAYSIZE(linear_static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN,
      &linear_load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t ranked_plan = {};
  loom_low_source_memory_access_diagnostic_t ranked_diagnostic = {0};
  ASSERT_TRUE(
      BuildPlan(&facts, ranked_load_op, &ranked_plan, &ranked_diagnostic));
  loom_low_source_memory_access_plan_t linear_plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, linear_load_op, &linear_plan, &diagnostic));

  EXPECT_EQ(linear_plan.static_byte_offset, ranked_plan.static_byte_offset);
  ASSERT_EQ(linear_plan.dynamic_term_count, ranked_plan.dynamic_term_count);
  for (uint8_t i = 0; i < linear_plan.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t& ranked_term =
        ranked_plan.dynamic_terms[i];
    const loom_low_source_memory_dynamic_term_t& linear_term =
        linear_plan.dynamic_terms[i];
    EXPECT_EQ(linear_term.index, ranked_term.index);
    EXPECT_EQ(linear_term.source, ranked_term.source);
    EXPECT_EQ(linear_term.dimension, ranked_term.dimension);
    EXPECT_EQ(linear_term.byte_stride, ranked_term.byte_stride);
    EXPECT_EQ(linear_term.byte_shift, ranked_term.byte_shift);
    EXPECT_EQ(linear_term.byte_facts.range_lo, ranked_term.byte_facts.range_lo);
    EXPECT_EQ(linear_term.byte_facts.range_hi, ranked_term.byte_facts.range_hi);
    EXPECT_EQ(linear_term.byte_facts.known_divisor,
              ranked_term.byte_facts.known_divisor);
  }

  ASSERT_EQ(linear_plan.dynamic_term_count, 2u);
  EXPECT_EQ(linear_plan.dynamic_terms[0].index, row);
  EXPECT_EQ(linear_plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(linear_plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_Y);
  EXPECT_EQ(linear_plan.dynamic_terms[0].byte_stride, 16);
  EXPECT_EQ(linear_plan.dynamic_terms[0].byte_shift, 4u);
  EXPECT_EQ(linear_plan.dynamic_terms[1].index, column);
  EXPECT_EQ(linear_plan.dynamic_terms[1].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID);
  EXPECT_EQ(linear_plan.dynamic_terms[1].dimension, LOOM_KERNEL_DIMENSION_X);
  EXPECT_EQ(linear_plan.dynamic_terms[1].byte_stride, 4);
  EXPECT_EQ(linear_plan.dynamic_terms[1].byte_shift, 2u);
  ASSERT_EQ(linear_plan.dynamic_realization_count, 1u);
  EXPECT_EQ(linear_plan.dynamic_realizations[0].term.index,
            loom_index_madd_result(linear_op));
  EXPECT_EQ(linear_plan.dynamic_realizations[0].term.byte_stride, 4);
  EXPECT_EQ(linear_plan.dynamic_realizations[0].first_term, 0u);
  EXPECT_EQ(linear_plan.dynamic_realizations[0].term_count, 2u);
  EXPECT_EQ(linear_plan.vector_lane_count, ranked_plan.vector_lane_count);
  EXPECT_EQ(linear_plan.vector_lane_byte_stride,
            ranked_plan.vector_lane_byte_stride);
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
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
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
  EXPECT_EQ(plan.base_view_value_id, loom_buffer_view_result(view_op));
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
  EXPECT_EQ(plan.view_value_id, loom_view_subview_result(subview_op));
  EXPECT_EQ(plan.base_view_value_id, loom_buffer_view_result(view_op));
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
