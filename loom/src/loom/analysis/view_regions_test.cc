// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/view_regions.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class ViewRegionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);

    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    BuildFunction();
  }

  void TearDown() override {
    loom_local_value_domain_release(&value_domain_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn vtables_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtables_fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  void BuildFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    loom_op_t* func_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    function_ = loom_func_like_cast(module_, func_op);
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder_);
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

  loom_value_id_t DefineLayoutArg() {
    loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        &layout));
    return layout;
  }

  loom_op_t* BuildOffsetConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_value_id_t BuildDenseLayout() {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_encoding_layout_dense_build(
        &builder_,
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_dense_result(op);
  }

  loom_value_id_t BuildDynamicStridedLayout(loom_value_id_t row_stride) {
    loom_op_t* op = nullptr;
    int64_t* static_strides = nullptr;
    IREE_CHECK_OK(
        iree_arena_allocate_array(&module_->arena, 2, sizeof(*static_strides),
                                  reinterpret_cast<void**>(&static_strides)));
    static_strides[0] = INT64_MIN;
    static_strides[1] = 1;
    IREE_CHECK_OK(loom_encoding_layout_strided_build(
        &builder_, &row_stride, 1, static_strides, 2,
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_strided_result(op);
  }

  loom_value_id_t BuildStaticStridedLayout(const int64_t* strides,
                                           iree_host_size_t stride_count) {
    loom_op_t* op = nullptr;
    int64_t* stored_strides = nullptr;
    IREE_CHECK_OK(iree_arena_allocate_array(
        &module_->arena, stride_count, sizeof(*stored_strides),
        reinterpret_cast<void**>(&stored_strides)));
    for (iree_host_size_t i = 0; i < stride_count; ++i) {
      stored_strides[i] = strides[i];
    }
    IREE_CHECK_OK(loom_encoding_layout_strided_build(
        &builder_, NULL, 0, stored_strides, stride_count,
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_strided_result(op);
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

  void DefineStaticStridedLayoutFacts(loom_value_fact_table_t* facts,
                                      loom_value_id_t layout_value_id,
                                      int64_t row_stride,
                                      int64_t column_stride) {
    loom_value_facts_t* strides = nullptr;
    IREE_CHECK_OK(
        iree_arena_allocate_array(&analysis_arena_, 2, sizeof(*strides),
                                  reinterpret_cast<void**>(&strides)));
    strides[0] = loom_value_facts_exact_i64(row_stride);
    strides[1] = loom_value_facts_exact_i64(column_stride);
    loom_value_fact_encoding_summary_t summary = {
        /*.role=*/LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        /*.static_spec_encoding_id=*/0,
        /*.address_layout=*/
        {
            /*.kind=*/LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
            /*.rank=*/2,
            /*.strides=*/strides,
        },
    };
    loom_value_facts_t layout_facts = loom_value_facts_unknown();
    IREE_CHECK_OK(loom_value_facts_make_encoding_summary(
        &facts->context, summary, &layout_facts));
    IREE_CHECK_OK(
        loom_value_fact_table_define(facts, layout_value_id, layout_facts));
  }

  void Analyze(loom_value_fact_table_t* facts,
               loom_view_region_table_t* out_table) {
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, loom_func_like_body(function_), &analysis_arena_,
        &value_domain_));
    IREE_ASSERT_OK(loom_view_region_table_initialize(
        facts, &value_domain_, &analysis_arena_, out_table));
    IREE_ASSERT_OK(loom_view_region_table_analyze(out_table));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_;
  loom_local_value_domain_t value_domain_ = {};
  loom_builder_t builder_;
};

TEST_F(ViewRegionsTest, ProvesDisjointReadAndWriteViewsInOneSlab) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));
  loom_value_id_t one_hundred_twenty_eight =
      loom_index_constant_result(BuildOffsetConstant(128));

  loom_op_t* read_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, zero,
                                        ViewType1D(16, layout),
                                        LOOM_LOCATION_UNKNOWN, &read_view_op));
  loom_op_t* write_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, buffer, one_hundred_twenty_eight, ViewType1D(16, layout),
      LOOM_LOCATION_UNKNOWN, &write_view_op));

  int64_t static_indices[] = {0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, loom_buffer_view_result(read_view_op), NULL, 0,
      static_indices, IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(4),
      LOOM_LOCATION_UNKNOWN, &load_op));
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_vector_store_build(
      &builder_, 0, loom_vector_load_result(load_op),
      loom_buffer_view_result(write_view_op), NULL, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, LOOM_LOCATION_UNKNOWN, &store_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {0};
  Analyze(&facts, &table);

  const loom_view_region_t* read_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(read_view_op), &read_region));
  const loom_view_region_t* write_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(write_view_op), &write_region));

  ASSERT_NE(read_region, nullptr);
  ASSERT_NE(write_region, nullptr);
  EXPECT_EQ(read_region->root_value_id, buffer);
  EXPECT_EQ(write_region->root_value_id, buffer);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&read_region->begin_byte_offset));
  EXPECT_EQ(read_region->begin_byte_offset.constant, 0);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&read_region->byte_length));
  EXPECT_EQ(read_region->byte_length.constant, 64);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&read_region->end_byte_offset));
  EXPECT_EQ(read_region->end_byte_offset.constant, 64);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&write_region->begin_byte_offset));
  EXPECT_EQ(write_region->begin_byte_offset.constant, 128);

  bool no_overlap = false;
  IREE_ASSERT_OK(loom_view_regions_prove_no_overlap(&table, read_region,
                                                    write_region, &no_overlap));
  EXPECT_TRUE(no_overlap);
  EXPECT_EQ(read_region->access_flags, LOOM_VIEW_ACCESS_READ);
  EXPECT_EQ(write_region->access_flags, LOOM_VIEW_ACCESS_WRITE);
  EXPECT_EQ(loom_view_region_table_root_access_flags(&table, buffer),
            LOOM_VIEW_ACCESS_READ | LOOM_VIEW_ACCESS_WRITE);
}

TEST_F(ViewRegionsTest, ProvesSymbolicOffsetCancellation) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t base_offset = DefineOffsetArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t sixty_four =
      loom_index_constant_result(BuildOffsetConstant(64));
  loom_op_t* second_offset_op = nullptr;
  IREE_ASSERT_OK(loom_index_add_build(&builder_, base_offset, sixty_four,
                                      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET),
                                      LOOM_LOCATION_UNKNOWN,
                                      &second_offset_op));

  loom_op_t* first_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType1D(16, layout),
                                        LOOM_LOCATION_UNKNOWN, &first_view_op));
  loom_op_t* second_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, buffer, loom_index_add_result(second_offset_op),
      ViewType1D(16, layout), LOOM_LOCATION_UNKNOWN, &second_view_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {0};
  Analyze(&facts, &table);

  const loom_view_region_t* first_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(first_view_op), &first_region));
  const loom_view_region_t* second_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(second_view_op), &second_region));

  ASSERT_NE(first_region, nullptr);
  ASSERT_NE(second_region, nullptr);
  EXPECT_TRUE(loom_symbolic_expr_is_linear(&first_region->begin_byte_offset));
  ASSERT_EQ(first_region->begin_byte_offset.term_count, 1);
  EXPECT_EQ(first_region->begin_byte_offset.terms[0].value_id, base_offset);
  EXPECT_EQ(first_region->begin_byte_offset.constant, 0);
  EXPECT_TRUE(loom_symbolic_expr_is_linear(&second_region->begin_byte_offset));
  ASSERT_EQ(second_region->begin_byte_offset.term_count, 1);
  EXPECT_EQ(second_region->begin_byte_offset.terms[0].value_id, base_offset);
  EXPECT_EQ(second_region->begin_byte_offset.constant, 64);

  bool no_overlap = false;
  IREE_ASSERT_OK(loom_view_regions_prove_no_overlap(
      &table, first_region, second_region, &no_overlap));
  EXPECT_TRUE(no_overlap);
}

TEST_F(ViewRegionsTest, KeepsOverlappingAndDifferentRootViewsConservative) {
  loom_value_id_t first_buffer = DefineBufferArg();
  loom_value_id_t second_buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));
  loom_value_id_t thirty_two =
      loom_index_constant_result(BuildOffsetConstant(32));

  loom_op_t* first_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, first_buffer, zero,
                                        ViewType1D(16, layout),
                                        LOOM_LOCATION_UNKNOWN, &first_view_op));
  loom_op_t* overlapping_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, first_buffer, thirty_two, ViewType1D(16, layout),
      LOOM_LOCATION_UNKNOWN, &overlapping_view_op));
  loom_op_t* other_root_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, second_buffer, zero, ViewType1D(16, layout),
      LOOM_LOCATION_UNKNOWN, &other_root_view_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {0};
  Analyze(&facts, &table);

  const loom_view_region_t* first_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(first_view_op), &first_region));
  const loom_view_region_t* overlapping_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(overlapping_view_op),
      &overlapping_region));
  const loom_view_region_t* other_root_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(other_root_view_op), &other_root_region));

  bool no_overlap = true;
  IREE_ASSERT_OK(loom_view_regions_prove_no_overlap(
      &table, first_region, overlapping_region, &no_overlap));
  EXPECT_FALSE(no_overlap);

  no_overlap = true;
  IREE_ASSERT_OK(loom_view_regions_prove_no_overlap(
      &table, first_region, other_root_region, &no_overlap));
  EXPECT_FALSE(no_overlap);
}

TEST_F(ViewRegionsTest, SubviewPreservesRootAndAddsLogicalOffset) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t base_offset = DefineOffsetArg();
  loom_value_id_t layout = BuildDenseLayout();

  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, base_offset,
                                        ViewType2D(8, 16, layout),
                                        LOOM_LOCATION_UNKNOWN, &source_op));

  int64_t static_offsets[] = {2, 0};
  loom_op_t* subview_op = nullptr;
  IREE_ASSERT_OK(loom_view_subview_build(
      &builder_, loom_buffer_view_result(source_op), NULL, 0, static_offsets,
      IREE_ARRAYSIZE(static_offsets), ViewType2D(1, 16, layout),
      LOOM_LOCATION_UNKNOWN, &subview_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {0};
  Analyze(&facts, &table);

  const loom_view_region_t* region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_view_subview_result(subview_op), &region));

  ASSERT_NE(region, nullptr);
  EXPECT_EQ(region->root_value_id, buffer);
  EXPECT_TRUE(loom_symbolic_expr_is_linear(&region->begin_byte_offset));
  ASSERT_EQ(region->begin_byte_offset.term_count, 1);
  EXPECT_EQ(region->begin_byte_offset.terms[0].value_id, base_offset);
  EXPECT_EQ(region->begin_byte_offset.constant, 128);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&region->byte_length));
  EXPECT_EQ(region->byte_length.constant, 64);
}

TEST_F(ViewRegionsTest, StridedFootprintIncludesPaddingGaps) {
  loom_value_id_t buffer = DefineBufferArg();
  int64_t strides[] = {8, 1};
  loom_value_id_t layout =
      BuildStaticStridedLayout(strides, IREE_ARRAYSIZE(strides));
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, zero,
                                        ViewType2D(4, 4, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {0};
  Analyze(&facts, &table);

  const loom_view_region_t* region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(view_op), &region));

  ASSERT_NE(region, nullptr);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&region->byte_length));
  EXPECT_EQ(region->byte_length.constant, 112);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&region->end_byte_offset));
  EXPECT_EQ(region->end_byte_offset.constant, 112);
}

TEST_F(ViewRegionsTest, DynamicStridedFootprintKeepsStrideExpression) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t row_stride = DefineIndexArg();
  loom_value_id_t layout = BuildDynamicStridedLayout(row_stride);
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, zero,
                                        ViewType2D(4, 4, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {0};
  Analyze(&facts, &table);

  const loom_view_region_t* region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(view_op), &region));

  ASSERT_NE(region, nullptr);
  EXPECT_TRUE(loom_symbolic_expr_is_linear(&region->byte_length));
  EXPECT_EQ(region->byte_length.constant, 16);
  ASSERT_EQ(region->byte_length.term_count, 1);
  EXPECT_EQ(region->byte_length.terms[0].coefficient, 12);
  EXPECT_EQ(region->byte_length.terms[0].value_id, row_stride);
}

TEST_F(ViewRegionsTest, SeededLayoutFactsForEncodingArgDriveFootprint) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = DefineLayoutArg();
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, buffer, zero,
                                        ViewType2D(4, 4, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  DefineStaticStridedLayoutFacts(&facts, layout, /*row_stride=*/8,
                                 /*column_stride=*/1);
  loom_view_region_table_t table = {0};
  Analyze(&facts, &table);

  const loom_view_region_t* region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(view_op), &region));

  ASSERT_NE(region, nullptr);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&region->byte_length));
  EXPECT_EQ(region->byte_length.constant, 112);
}

}  // namespace
}  // namespace loom
