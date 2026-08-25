// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan_test_fixture.h"

namespace loom {
namespace {

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
  EXPECT_EQ(plan.base_view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.root_value_id, buffer);
  EXPECT_EQ(plan.memory_space, LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN);
  EXPECT_EQ(plan.address_layout,
            LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR);
  EXPECT_EQ(plan.element_byte_count, 4u);
  EXPECT_EQ(plan.vector_lane_count, 4u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 4);
  EXPECT_EQ(plan.vector_offset_kind, LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_NONE);
  EXPECT_EQ(plan.static_byte_offset, 28);
  EXPECT_EQ(plan.static_view_base_byte_offset, 16);
  EXPECT_EQ(plan.dynamic_view_base_value_id, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(plan.root_minimum_alignment, 1u);
  EXPECT_EQ(plan.minimum_alignment, 1u);
  EXPECT_EQ(plan.dynamic_term_count, 0u);
  EXPECT_EQ(plan.dynamic_view_base_term_count, 0u);
}

TEST_F(SourceMemoryPlanTest, PhysicalByteLoadUsesBufferReferenceAndOffset) {
  const loom_value_id_t root_buffer = DefineBufferArg();
  const loom_value_id_t buffer = BuildAligned(root_buffer, 16);
  const loom_value_id_t byte_offset =
      loom_index_constant_result(BuildOffsetConstant(12));
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_load_i8_u_build(&builder_, buffer, byte_offset,
                                             LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.operation_kind, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD);
  EXPECT_EQ(plan.view_value_id, buffer);
  EXPECT_EQ(plan.base_view_value_id, buffer);
  EXPECT_EQ(plan.root_value_id, root_buffer);
  EXPECT_EQ(plan.address_layout,
            LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR);
  EXPECT_EQ(plan.element_byte_count, 1u);
  EXPECT_EQ(plan.vector_lane_count, 1u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 1);
  EXPECT_EQ(plan.static_byte_offset, 12);
  EXPECT_EQ(plan.root_minimum_alignment, 16u);
  EXPECT_EQ(plan.minimum_alignment, 4u);
  EXPECT_EQ(plan.dynamic_term_count, 0u);
}

TEST_F(SourceMemoryPlanTest, PhysicalByteStoreRetainsDynamicOffset) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t byte_offset = DefineOffsetArg();
  const loom_value_id_t value =
      DefineArg(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_store_i8_build(
      &builder_, value, buffer, byte_offset, LOOM_LOCATION_UNKNOWN, &store_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, store_op, &plan, &diagnostic));
  EXPECT_EQ(plan.operation_kind, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE);
  EXPECT_EQ(plan.root_value_id, buffer);
  EXPECT_EQ(plan.static_byte_offset, 0);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, byte_offset);
  EXPECT_EQ(plan.dynamic_terms[0].source,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE);
  EXPECT_EQ(plan.dynamic_terms[0].dimension, LOOM_KERNEL_DIMENSION_COUNT_);
  EXPECT_EQ(plan.dynamic_terms[0].axis,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 1);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 0u);
}

TEST_F(SourceMemoryPlanTest, StaticStridedLayoutClassifiesCompactness) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));
  const int64_t row_strides[] = {16, 32};
  const loom_low_source_memory_address_layout_t expected_layouts[] = {
      LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR,
      LOOM_LOW_SOURCE_MEMORY_ADDRESS_LAYOUT_UNPROVEN,
  };
  loom_op_t* load_ops[IREE_ARRAYSIZE(row_strides)] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(row_strides); ++i) {
    loom_value_id_t layout = BuildStridedLayout(row_strides[i], 1);
    loom_op_t* view_op = nullptr;
    IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                          ViewType2D(8, 16, layout),
                                          LOOM_LOCATION_UNKNOWN, &view_op));
    const int64_t static_indices[] = {0, 0};
    IREE_ASSERT_OK(loom_vector_load_build(
        &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
        static_indices, IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
        LOOM_LOCATION_UNKNOWN, &load_ops[i]));
  }

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(row_strides); ++i) {
    loom_low_source_memory_access_plan_t plan = {};
    loom_low_source_memory_access_diagnostic_t diagnostic = {0};
    ASSERT_TRUE(BuildPlan(&facts, load_ops[i], &plan, &diagnostic));
    EXPECT_EQ(plan.address_layout, expected_layouts[i]);
  }
}

TEST_F(SourceMemoryPlanTest, DynamicStridedLayoutScalesDynamicOrigin) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t row_stride = DefineIndexArg();
  const loom_value_id_t row = DefineIndexArg();
  const loom_value_id_t layout =
      BuildDynamicStridedLayout(row_stride, /*column_stride=*/1);
  const loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));
  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(8, 16, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  const int64_t static_indices[] = {INT64_MIN, 0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), &row, 1, static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, row);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 4);
  ASSERT_EQ(plan.dynamic_terms[0].stride_value_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].stride_values[0], row_stride);
}

TEST_F(SourceMemoryPlanTest, DynamicStridedLayoutScalesStaticOrigin) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t row_stride = DefineIndexArg();
  const loom_value_id_t layout =
      BuildDynamicStridedLayout(row_stride, /*column_stride=*/1);
  const loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));
  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(8, 16, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  const int64_t static_indices[] = {3, 0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 0);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, row_stride);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 12);
  EXPECT_EQ(plan.dynamic_terms[0].stride_value_count, 0u);
}

TEST_F(SourceMemoryPlanTest, ExactDynamicStrideFoldsIntoStaticOffset) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t row_stride =
      loom_index_constant_result(BuildIndexConstant(64));
  const loom_value_id_t layout =
      BuildDynamicStridedLayout(row_stride, /*column_stride=*/1);
  const loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));
  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(8, 16, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  const int64_t static_indices[] = {3, 0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 768);
  EXPECT_EQ(plan.dynamic_term_count, 0u);
}

TEST_F(SourceMemoryPlanTest, DenseAxisStrideRetainsMultipleRuntimeFactors) {
  const loom_value_id_t rows = DefineIndexArg();
  const loom_value_id_t exact_rows =
      loom_index_constant_result(BuildIndexConstant(8));
  const loom_value_id_t columns = DefineIndexArg();
  const loom_value_id_t layout = BuildDenseLayout();
  const loom_type_t view_type =
      ViewType3D(loom_dim_pack_static(4), loom_dim_pack_dynamic(rows),
                 loom_dim_pack_dynamic(columns), layout);
  const loom_type_t exact_row_view_type =
      ViewType3D(loom_dim_pack_static(4), loom_dim_pack_dynamic(exact_rows),
                 loom_dim_pack_dynamic(columns), layout);
  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_vector_memory_access_t access = {};
  ASSERT_TRUE(loom_vector_memory_access_describe(
      &facts.context, module_, view_type, VectorType1D(1), &access));

  loom_low_source_memory_axis_byte_stride_t block_stride = {};
  loom_low_source_memory_query_axis_byte_stride(&facts, &access, 0,
                                                &block_stride);
  EXPECT_EQ(block_stride.kind, LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC);
  EXPECT_EQ(block_stride.static_byte_coefficient, 4);
  ASSERT_EQ(block_stride.dynamic_factor_count, 2u);
  EXPECT_EQ(block_stride.dynamic_factors[0], rows);
  EXPECT_EQ(block_stride.dynamic_factors[1], columns);

  loom_low_source_memory_axis_byte_stride_t row_stride = {};
  loom_low_source_memory_query_axis_byte_stride(&facts, &access, 1,
                                                &row_stride);
  EXPECT_EQ(row_stride.kind, LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC);
  EXPECT_EQ(row_stride.static_byte_coefficient, 4);
  ASSERT_EQ(row_stride.dynamic_factor_count, 1u);
  EXPECT_EQ(row_stride.dynamic_factors[0], columns);

  loom_vector_memory_access_t exact_row_access = {};
  ASSERT_TRUE(loom_vector_memory_access_describe(
      &facts.context, module_, exact_row_view_type, VectorType1D(1),
      &exact_row_access));
  loom_low_source_memory_axis_byte_stride_t folded_block_stride = {};
  loom_low_source_memory_query_axis_byte_stride(&facts, &exact_row_access, 0,
                                                &folded_block_stride);
  EXPECT_EQ(folded_block_stride.static_byte_coefficient, 32);
  ASSERT_EQ(folded_block_stride.dynamic_factor_count, 1u);
  EXPECT_EQ(folded_block_stride.dynamic_factors[0], columns);
}

TEST_F(SourceMemoryPlanTest, FactOnlyRuntimeStrideIsNotMaterialized) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t unknown_layout = DefineArg(
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT));
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_encoding_layout_assume_strided_build(
      &builder_, unknown_layout, 2,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
      LOOM_LOCATION_UNKNOWN, &assume_op));
  const loom_value_id_t layout =
      loom_encoding_layout_assume_strided_result(assume_op);
  const loom_value_id_t row = DefineIndexArg();
  const loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));
  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(8, 16, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));
  const int64_t static_indices[] = {INT64_MIN, 0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), &row, 1, static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  EXPECT_FALSE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_NE(diagnostic.rejection_bits &
                LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE,
            0u);
}

TEST_F(SourceMemoryPlanTest, ViewMemoryOperationKindUsesInterfaceShape) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t value = DefineArg(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  loom_value_id_t replacement =
      DefineArg(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(
      loom_buffer_view_build(&builder_, buffer, base_offset,
                             ViewType1D(LOOM_SCALAR_TYPE_I32, 16, layout),
                             LOOM_LOCATION_UNKNOWN, &view_op));
  int64_t static_indices[] = {0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &load_op));
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_view_store_build(
      &builder_, 0, value, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0,
      LOOM_LOCATION_UNKNOWN, &store_op));
  loom_op_t* atomic_reduce_op = nullptr;
  IREE_ASSERT_OK(loom_view_atomic_reduce_build(
      &builder_, 0, LOOM_ATOMIC_KIND_ADDI, value,
      loom_buffer_view_result(view_op), nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), LOOM_ATOMIC_ORDERING_RELAXED,
      LOOM_ATOMIC_SCOPE_WORKGROUP, 0, 0, LOOM_LOCATION_UNKNOWN,
      &atomic_reduce_op));
  loom_op_t* atomic_rmw_op = nullptr;
  IREE_ASSERT_OK(loom_view_atomic_rmw_build(
      &builder_, 0, LOOM_ATOMIC_KIND_ADDI, value,
      loom_buffer_view_result(view_op), nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), LOOM_ATOMIC_ORDERING_RELAXED,
      LOOM_ATOMIC_SCOPE_WORKGROUP, 0, 0, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      LOOM_LOCATION_UNKNOWN, &atomic_rmw_op));
  loom_op_t* atomic_cmpxchg_op = nullptr;
  IREE_ASSERT_OK(loom_view_atomic_cmpxchg_build(
      &builder_, 0, value, replacement, loom_buffer_view_result(view_op),
      nullptr, 0, static_indices, IREE_ARRAYSIZE(static_indices),
      LOOM_ATOMIC_ORDERING_ACQ_REL, LOOM_ATOMIC_ORDERING_ACQUIRE,
      LOOM_ATOMIC_SCOPE_WORKGROUP, 0, 0, loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      LOOM_LOCATION_UNKNOWN, &atomic_cmpxchg_op));
  loom_op_t* prefetch_op = nullptr;
  IREE_ASSERT_OK(loom_view_prefetch_build(
      &builder_, loom_buffer_view_result(view_op), nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), LOOM_VIEW_PREFETCH_INTENT_READ,
      LOOM_VIEW_PREFETCH_LOCALITY_L2, LOOM_LOCATION_UNKNOWN, &prefetch_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  struct Case {
    const loom_op_t* op;
    loom_low_source_memory_operation_kind_t expected_kind;
  };
  const Case cases[] = {
      {load_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD},
      {store_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE},
      {atomic_reduce_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE},
      {atomic_rmw_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW},
      {atomic_cmpxchg_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG},
      {prefetch_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH},
  };
  for (const Case& test_case : cases) {
    loom_low_source_memory_access_plan_t plan = {};
    loom_low_source_memory_access_diagnostic_t diagnostic = {0};
    ASSERT_TRUE(BuildPlan(&facts, test_case.op, &plan, &diagnostic));
    EXPECT_EQ(plan.operation_kind, test_case.expected_kind);
    EXPECT_EQ(plan.view_value_id, loom_buffer_view_result(view_op));
    EXPECT_EQ(plan.vector_lane_count, 1u);
  }
}

TEST_F(SourceMemoryPlanTest, VectorAtomicReduceTracksIdentityIotaOffsets) {
  loom_value_id_t root_buffer = DefineBufferArg();
  loom_value_id_t buffer = BuildAligned(root_buffer, 4);
  loom_value_id_t value = DefineArg(VectorType1D(LOOM_SCALAR_TYPE_F16, 2));
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(
      loom_buffer_view_build(&builder_, buffer, base_offset,
                             ViewType1D(LOOM_SCALAR_TYPE_F16, 16, layout),
                             LOOM_LOCATION_UNKNOWN, &view_op));
  loom_value_id_t offsets =
      BuildIndexIotaOffsets(0, 1, VectorType1D(LOOM_SCALAR_TYPE_INDEX, 2));
  int64_t static_indices[] = {0};
  loom_op_t* atomic_op = nullptr;
  IREE_ASSERT_OK(loom_vector_atomic_reduce_build(
      &builder_, 0, LOOM_ATOMIC_KIND_ADDF, value,
      loom_buffer_view_result(view_op), nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), offsets, LOOM_ATOMIC_ORDERING_RELAXED,
      LOOM_ATOMIC_SCOPE_WORKGROUP, 0, 0, LOOM_LOCATION_UNKNOWN, &atomic_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, atomic_op, &plan, &diagnostic));
  EXPECT_EQ(plan.operation_kind,
            LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE);
  EXPECT_EQ(plan.view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.base_view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.root_value_id, root_buffer);
  EXPECT_EQ(plan.element_byte_count, 2u);
  EXPECT_EQ(plan.vector_lane_count, 2u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 2);
  EXPECT_EQ(plan.vector_offset_kind,
            LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_IDENTITY_IOTA);
  EXPECT_EQ(plan.static_byte_offset, 0);
  EXPECT_EQ(plan.root_minimum_alignment, 4u);
  EXPECT_EQ(plan.minimum_alignment, 4u);
}

TEST_F(SourceMemoryPlanTest,
       VectorAtomicReduceTracksFromElementsIdentityOffsets) {
  loom_value_id_t root_buffer = DefineBufferArg();
  loom_value_id_t buffer = BuildAligned(root_buffer, 4);
  loom_value_id_t value = DefineArg(VectorType1D(LOOM_SCALAR_TYPE_BF16, 2));
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(
      loom_buffer_view_build(&builder_, buffer, base_offset,
                             ViewType1D(LOOM_SCALAR_TYPE_BF16, 16, layout),
                             LOOM_LOCATION_UNKNOWN, &view_op));
  loom_value_id_t offsets = BuildIndexFromElementOffsets(
      0, 1, VectorType1D(LOOM_SCALAR_TYPE_INDEX, 2));
  int64_t static_indices[] = {0};
  loom_op_t* atomic_op = nullptr;
  IREE_ASSERT_OK(loom_vector_atomic_reduce_build(
      &builder_, 0, LOOM_ATOMIC_KIND_ADDF, value,
      loom_buffer_view_result(view_op), nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), offsets, LOOM_ATOMIC_ORDERING_RELAXED,
      LOOM_ATOMIC_SCOPE_WORKGROUP, 0, 0, LOOM_LOCATION_UNKNOWN, &atomic_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, atomic_op, &plan, &diagnostic));
  EXPECT_EQ(plan.operation_kind,
            LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE);
  EXPECT_EQ(plan.view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.base_view_value_id, loom_buffer_view_result(view_op));
  EXPECT_EQ(plan.root_value_id, root_buffer);
  EXPECT_EQ(plan.element_byte_count, 2u);
  EXPECT_EQ(plan.vector_lane_count, 2u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 2);
  EXPECT_EQ(plan.vector_offset_kind,
            LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_IDENTITY_IOTA);
  EXPECT_EQ(plan.static_byte_offset, 0);
  EXPECT_EQ(plan.root_minimum_alignment, 4u);
  EXPECT_EQ(plan.minimum_alignment, 4u);
}

TEST_F(SourceMemoryPlanTest, VectorAtomicRmwClassifiesNonIdentityOffsets) {
  loom_value_id_t buffer = BuildAligned(DefineBufferArg(), 4);
  loom_value_id_t value = DefineArg(VectorType1D(LOOM_SCALAR_TYPE_F16, 2));
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t base_offset =
      loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(
      loom_buffer_view_build(&builder_, buffer, base_offset,
                             ViewType1D(LOOM_SCALAR_TYPE_F16, 16, layout),
                             LOOM_LOCATION_UNKNOWN, &view_op));
  loom_value_id_t offsets =
      BuildIndexIotaOffsets(0, 2, VectorType1D(LOOM_SCALAR_TYPE_INDEX, 2));
  int64_t static_indices[] = {0};
  loom_op_t* atomic_op = nullptr;
  IREE_ASSERT_OK(loom_vector_atomic_rmw_build(
      &builder_, 0, LOOM_ATOMIC_KIND_ADDF, value,
      loom_buffer_view_result(view_op), nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), offsets, LOOM_ATOMIC_ORDERING_RELAXED,
      LOOM_ATOMIC_SCOPE_WORKGROUP, 0, 0, VectorType1D(LOOM_SCALAR_TYPE_F16, 2),
      LOOM_LOCATION_UNKNOWN, &atomic_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, atomic_op, &plan, &diagnostic));
  EXPECT_EQ(plan.operation_kind, LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW);
  EXPECT_EQ(plan.vector_offset_kind,
            LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_OTHER);
  EXPECT_EQ(plan.element_byte_count, 2u);
  EXPECT_EQ(plan.vector_lane_count, 2u);
  EXPECT_EQ(plan.vector_lane_byte_stride, 2);
  EXPECT_EQ(plan.minimum_alignment, 4u);
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
  EXPECT_EQ(plan.dynamic_view_base_value_id, loom_index_add_result(base_op));
  EXPECT_EQ(plan.dynamic_view_base_value_static_byte_offset, 16);
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

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadFactorsScaledViewBase) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t element = DefineIndexArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t byte_stride =
      loom_index_constant_result(BuildOffsetConstant(4));
  loom_op_t* byte_offset_op = nullptr;
  IREE_ASSERT_OK(
      loom_index_scale_build(&builder_, element, byte_stride,
                             loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                             LOOM_LOCATION_UNKNOWN, &byte_offset_op));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, buffer, loom_index_scale_result(byte_offset_op),
      ViewType1D(LOOM_SCALAR_TYPE_I32, 1, layout), LOOM_LOCATION_UNKNOWN,
      &view_op));
  int64_t static_indices[] = {0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.dynamic_view_base_value_id,
            loom_index_scale_result(byte_offset_op));
  EXPECT_EQ(plan.dynamic_view_base_value_static_byte_offset, 0);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_view_base_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, element);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 2u);
}

TEST_F(SourceMemoryPlanTest,
       DynamicDenseLoadPreservesConstrainedAffineViewBaseWithStaticSuffix) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t base = DefineIndexArg();
  loom_value_id_t local = DefineIndexArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_op_t* sum_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, base, local,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                                      LOOM_LOCATION_UNKNOWN, &sum_op));
  const loom_value_id_t bounded_sum =
      BuildIndexAssumeRange(loom_index_add_result(sum_op), 0, 1023);
  const loom_value_id_t byte_stride =
      loom_index_constant_result(BuildOffsetConstant(4));
  loom_op_t* byte_offset_op = nullptr;
  IREE_ASSERT_OK(
      loom_index_scale_build(&builder_, bounded_sum, byte_stride,
                             loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                             LOOM_LOCATION_UNKNOWN, &byte_offset_op));
  const loom_value_id_t byte_offset = loom_index_scale_result(byte_offset_op);
  const loom_value_id_t header =
      loom_index_constant_result(BuildOffsetConstant(16));
  loom_op_t* payload_offset_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(
      &builder_, byte_offset, header, loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
      LOOM_LOCATION_UNKNOWN, &payload_offset_op));
  const loom_value_id_t payload_offset =
      loom_index_add_result(payload_offset_op);

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(
      loom_buffer_view_build(&builder_, buffer, payload_offset,
                             ViewType1D(LOOM_SCALAR_TYPE_I32, 1, layout),
                             LOOM_LOCATION_UNKNOWN, &view_op));
  int64_t static_indices[] = {0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 16);
  EXPECT_EQ(plan.dynamic_view_base_value_id, payload_offset);
  EXPECT_EQ(plan.dynamic_view_base_value_static_byte_offset, 16);
  ASSERT_EQ(plan.dynamic_term_count, 2u);
  EXPECT_EQ(plan.dynamic_view_base_term_count, 2u);
  EXPECT_EQ(plan.dynamic_terms[0].index, base);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 4);
  EXPECT_EQ(plan.dynamic_terms[1].index, local);
  EXPECT_EQ(plan.dynamic_terms[1].byte_stride, 4);
  EXPECT_FALSE(loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(
      &plan.dynamic_terms[0], 13));
  EXPECT_FALSE(loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(
      &plan.dynamic_terms[1], 13));
  EXPECT_EQ(plan.dynamic_realization_count, 0u);
  EXPECT_EQ(plan.dynamic_view_base_byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_view_base_byte_facts.range_hi, 4092);
  const loom_value_facts_t offset_facts =
      loom_low_source_memory_dynamic_offset_facts(&plan,
                                                  plan.static_byte_offset);
  EXPECT_EQ(offset_facts.range_lo, 16);
  EXPECT_EQ(offset_facts.range_hi, 4108);
  EXPECT_TRUE(loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
      &plan, plan.static_byte_offset, 13));
}

TEST_F(SourceMemoryPlanTest, DynamicDenseLoadTracksMaterializedI32ViewBase) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* header_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, buffer, zero, ViewType1D(LOOM_SCALAR_TYPE_I32, 8, layout),
      LOOM_LOCATION_UNKNOWN, &header_view_op));
  int64_t header_indices[] = {7};
  loom_op_t* base_load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(header_view_op), nullptr, 0,
      header_indices, IREE_ARRAYSIZE(header_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN,
      &base_load_op));
  loom_op_t* base_cast_op = nullptr;
  IREE_ASSERT_OK(
      loom_index_cast_build(&builder_, loom_view_load_result(base_load_op),
                            loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                            loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                            LOOM_LOCATION_UNKNOWN, &base_cast_op));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, buffer, loom_index_cast_result(base_cast_op),
      ViewType1D(LOOM_SCALAR_TYPE_I32, 4, layout), LOOM_LOCATION_UNKNOWN,
      &view_op));
  int64_t static_indices[] = {2};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_view_load_build(
      &builder_, 0, loom_buffer_view_result(view_op), nullptr, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &load_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_low_source_memory_access_plan_t plan = {};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  ASSERT_TRUE(BuildPlan(&facts, load_op, &plan, &diagnostic));
  EXPECT_EQ(plan.static_byte_offset, 8);
  EXPECT_EQ(plan.static_view_base_byte_offset, 0);
  EXPECT_EQ(plan.dynamic_view_base_value_id,
            loom_index_cast_result(base_cast_op));
  EXPECT_EQ(plan.dynamic_view_base_value_static_byte_offset, 0);
  ASSERT_EQ(plan.dynamic_term_count, 1u);
  EXPECT_EQ(plan.dynamic_view_base_term_count, 1u);
  EXPECT_EQ(plan.dynamic_terms[0].index, loom_view_load_result(base_load_op));
  EXPECT_EQ(plan.dynamic_terms[0].axis,
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE);
  EXPECT_EQ(plan.dynamic_terms[0].byte_stride, 1);
  EXPECT_EQ(plan.dynamic_terms[0].byte_shift, 0u);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_lo, 0);
  EXPECT_EQ(plan.dynamic_terms[0].byte_facts.range_hi, UINT32_MAX);
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
  EXPECT_EQ(plan.dynamic_view_base_value_id, LOOM_VALUE_ID_INVALID);
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
  EXPECT_FALSE(
      iree_any_bit_set(summary.precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL));
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

TEST_F(SourceMemoryPlanTest, SummaryCapturesStridedPacketSlot) {
  loom_low_source_memory_access_plan_t plan = {};
  plan.operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
  plan.memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
  plan.alias_scope_id = 7;
  plan.element_byte_count = 2;
  plan.vector_lane_count = 8;
  plan.vector_lane_byte_stride = 2;
  plan.static_byte_offset = 16;
  plan.dynamic_term_count = 1;
  plan.dynamic_terms[0].byte_stride = 64;
  plan.dynamic_terms[0].byte_facts = loom_value_facts_exact_i64(0);

  loom_low_byte_interval_t interval = {};
  loom_low_memory_access_summary_t summary = {};
  loom_low_source_memory_access_plan_make_summary(&plan, &interval, &summary);
  EXPECT_TRUE(
      iree_any_bit_set(summary.precision_flags,
                       LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL));
  EXPECT_EQ(summary.strided_interval.stride_bytes, 64u);
  EXPECT_EQ(summary.strided_interval.begin_bytes, 16u);
  EXPECT_EQ(summary.strided_interval.end_bytes, 32u);

  plan.static_byte_offset = 0;
  loom_low_byte_interval_t preceding_interval = {};
  loom_low_memory_access_summary_t preceding_summary = {};
  loom_low_source_memory_access_plan_make_summary(&plan, &preceding_interval,
                                                  &preceding_summary);
  EXPECT_FALSE(
      loom_low_memory_access_summaries_may_alias(&preceding_summary, &summary));
}

}  // namespace
}  // namespace loom
