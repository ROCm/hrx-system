// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/view_regions.h"

#include <cstdint>
#include <tuple>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/type_registry.h"
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
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
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

  loom_op_t* BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
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

  loom_type_t ViewType1D(
      int64_t extent, loom_value_id_t layout,
      loom_scalar_type_t element_type = LOOM_SCALAR_TYPE_F32) {
    loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VIEW, element_type,
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

  loom_value_id_t BuildNoAliasBuffer(
      loom_value_id_t buffer, loom_value_fact_memory_space_t memory_space =
                                  LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) {
    loom_op_t* space = nullptr;
    IREE_CHECK_OK(loom_buffer_assume_memory_space_build(
        &builder_, memory_space, buffer, loom_type_buffer(),
        LOOM_LOCATION_UNKNOWN, &space));
    const loom_value_id_t buffer_with_space =
        loom_buffer_assume_memory_space_result(space);
    const loom_type_t buffer_type = loom_type_buffer();
    loom_op_t* noalias = nullptr;
    IREE_CHECK_OK(loom_buffer_assume_noalias_build(
        &builder_, &buffer_with_space, 1, &buffer_type, 1,
        LOOM_LOCATION_UNKNOWN, &noalias));
    return loom_op_results(noalias)[0];
  }

  loom_value_id_t BuildView(
      loom_value_id_t buffer,
      loom_scalar_type_t element_type = LOOM_SCALAR_TYPE_F32) {
    const loom_value_id_t layout = BuildDenseLayout();
    const loom_value_id_t base =
        loom_index_constant_result(BuildOffsetConstant(0));
    loom_op_t* view = nullptr;
    IREE_CHECK_OK(loom_buffer_view_build(&builder_, 0, buffer, base, 0,
                                         ViewType1D(4, layout, element_type),
                                         LOOM_LOCATION_UNKNOWN, &view));
    return loom_buffer_view_result(view);
  }

  loom_value_id_t BuildReadOnlyView(loom_value_id_t buffer) {
    const loom_value_id_t view = BuildView(buffer);
    const int64_t indices[] = {0};
    loom_op_t* load = nullptr;
    IREE_CHECK_OK(loom_vector_load_build(&builder_, 0, 0, view, nullptr, 0,
                                         indices, 1, 0, 0, VectorType1D(4),
                                         LOOM_LOCATION_UNKNOWN, &load));
    return view;
  }

  loom_value_id_t DefineScalarArg(loom_scalar_type_t element_type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_scalar(element_type), &value));
    return value;
  }

  bool RootIsStable(loom_view_region_table_t* table, loom_value_id_t view) {
    const loom_view_region_t* region = nullptr;
    IREE_CHECK_OK(loom_view_region_table_get(table, view, &region));
    return loom_view_region_table_root_is_stable(table, region->root_value_id,
                                                 region->alias_scope_id,
                                                 region->memory_space);
  }

  loom_type_t VectorType1D(int64_t extent) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                               loom_dim_pack_static(extent), 0);
  }

  void ComputeFacts(loom_value_fact_table_t* out_facts) {
    IREE_ASSERT_OK(loom_value_fact_table_initialize(out_facts, &analysis_arena_,
                                                    module_->values.count));
    loom_type_registry_configure_fact_context(&out_facts->context);
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
    loom_symbolic_expr_context_initialize(
        module_, &value_domain_, facts, &analysis_arena_, &expression_context_);
    IREE_ASSERT_OK(loom_view_region_table_initialize(
        &value_domain_, &expression_context_, out_table));
    IREE_ASSERT_OK(loom_view_region_table_analyze(out_table));
  }

  // Pool backing module and analysis arenas.
  iree_arena_block_pool_t block_pool_;
  // Arena retaining the fact and region analysis results.
  iree_arena_allocator_t analysis_arena_;
  // Context with the source dialects used by these programs.
  loom_context_t context_;
  // Owned module containing the analyzed function.
  loom_module_t* module_ = nullptr;
  // Function containing the source memory operations.
  loom_func_like_t function_;
  // Local value numbering retained through region analysis.
  loom_local_value_domain_t value_domain_ = {};
  // Symbolic expressions shared by analyzed view regions.
  loom_symbolic_expr_context_t expression_context_ = {};
  // Builder positioned in the function body.
  loom_builder_t builder_;
};

TEST_F(ViewRegionsTest, SelectedViewRetainsUnknownRootRelativeOffset) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(loom_func_like_body(function_)),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), &condition));
  loom_value_id_t layout = BuildDenseLayout();
  loom_type_t view_type = ViewType1D(16, layout);
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));
  loom_value_id_t second_offset =
      loom_index_constant_result(BuildOffsetConstant(128));
  loom_op_t* first_view = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, LOOM_BUFFER_VIEW_BUILD_FLAG_HAS_ADDRESS_BITWIDTH, buffer, zero,
      32, view_type, LOOM_LOCATION_UNKNOWN, &first_view));
  loom_op_t* second_view = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, LOOM_BUFFER_VIEW_BUILD_FLAG_HAS_ADDRESS_BITWIDTH, buffer,
      second_offset, 32, view_type, LOOM_LOCATION_UNKNOWN, &second_view));
  loom_op_t* select = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(
      &builder_, condition, loom_buffer_view_result(first_view),
      loom_buffer_view_result(second_view), view_type, LOOM_LOCATION_UNKNOWN,
      &select));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  const loom_view_region_t* selected_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_scf_select_result(select), &selected_region));
  const loom_view_region_t* second_region = nullptr;
  IREE_ASSERT_OK(loom_view_region_table_get(
      &table, loom_buffer_view_result(second_view), &second_region));

  ASSERT_NE(selected_region, nullptr);
  EXPECT_EQ(selected_region->root_value_id, buffer);
  EXPECT_EQ(selected_region->address_bitwidth, 32);
  EXPECT_FALSE(
      loom_symbolic_expr_is_constant(&selected_region->begin_byte_offset));
  EXPECT_EQ(selected_region->begin_byte_offset.facts.range_lo, 0);
  EXPECT_EQ(selected_region->begin_byte_offset.facts.range_hi, 128);
  EXPECT_TRUE(loom_symbolic_expr_is_constant(&selected_region->byte_length));
  EXPECT_EQ(selected_region->byte_length.constant, 64);
  bool no_overlap = false;
  IREE_ASSERT_OK(loom_view_regions_prove_no_overlap(
      &table, selected_region, second_region, &no_overlap));
  EXPECT_FALSE(no_overlap);
}

TEST_F(ViewRegionsTest, ViewAddressDomainRequiresJoinAgreement) {
  const loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(loom_func_like_body(function_)),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), &condition));
  const loom_value_id_t layout = BuildDenseLayout();
  const loom_type_t view_type = ViewType1D(8, layout);
  const loom_value_id_t base =
      loom_index_constant_result(BuildOffsetConstant(0));

  auto build_view = [&](uint8_t address_bitwidth) {
    loom_op_t* view = nullptr;
    const loom_buffer_view_build_flags_t flags =
        address_bitwidth != 0 ? LOOM_BUFFER_VIEW_BUILD_FLAG_HAS_ADDRESS_BITWIDTH
                              : 0;
    IREE_EXPECT_OK(loom_buffer_view_build(&builder_, flags, buffer, base,
                                          address_bitwidth, view_type,
                                          LOOM_LOCATION_UNKNOWN, &view));
    return loom_buffer_view_result(view);
  };
  const loom_value_id_t first32 = build_view(32);
  const loom_value_id_t second32 = build_view(32);
  const loom_value_id_t view64 = build_view(64);
  const loom_value_id_t ordinary = build_view(0);

  auto build_select = [&](loom_value_id_t lhs, loom_value_id_t rhs) {
    loom_op_t* select = nullptr;
    IREE_EXPECT_OK(loom_scf_select_build(&builder_, condition, lhs, rhs,
                                         view_type, LOOM_LOCATION_UNKNOWN,
                                         &select));
    return loom_scf_select_result(select);
  };
  const loom_value_id_t same_width = build_select(first32, second32);
  const loom_value_id_t different_width = build_select(first32, view64);
  const loom_value_id_t missing_width = build_select(first32, ordinary);

  const int64_t subview_offset = 1;
  loom_op_t* subview = nullptr;
  const loom_type_t subview_type = ViewType1D(4, layout);
  IREE_ASSERT_OK(loom_view_subview_build(&builder_, first32, nullptr, 0,
                                         &subview_offset, 1, subview_type,
                                         LOOM_LOCATION_UNKNOWN, &subview));
  loom_op_t* refined = nullptr;
  IREE_ASSERT_OK(loom_view_refine_build(
      &builder_, loom_view_subview_result(subview), ViewType1D(2, layout),
      LOOM_LOCATION_UNKNOWN, &refined));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  auto address_bitwidth = [&](loom_value_id_t value) {
    loom_value_fact_view_reference_t reference = {};
    EXPECT_TRUE(loom_value_facts_query_view_reference(
        &facts.context, loom_value_fact_table_lookup(&facts, value),
        &reference));
    return reference.address_bitwidth;
  };
  EXPECT_EQ(address_bitwidth(first32), 32);
  EXPECT_EQ(address_bitwidth(loom_view_subview_result(subview)), 32);
  EXPECT_EQ(address_bitwidth(loom_view_refine_result(refined)), 32);
  EXPECT_EQ(address_bitwidth(same_width), 32);
  EXPECT_EQ(address_bitwidth(different_width), 0);
  EXPECT_EQ(address_bitwidth(missing_width), 0);
}

TEST_F(ViewRegionsTest, ProvesDisjointReadAndWriteViewsInOneSlab) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));
  loom_value_id_t one_hundred_twenty_eight =
      loom_index_constant_result(BuildOffsetConstant(128));

  loom_op_t* read_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, 0, buffer, zero, 0,
                                        ViewType1D(16, layout),
                                        LOOM_LOCATION_UNKNOWN, &read_view_op));
  loom_op_t* write_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, 0, buffer, one_hundred_twenty_eight, 0, ViewType1D(16, layout),
      LOOM_LOCATION_UNKNOWN, &write_view_op));

  int64_t static_indices[] = {0};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(read_view_op),
      NULL, 0, static_indices, IREE_ARRAYSIZE(static_indices), 0, 0,
      VectorType1D(4), LOOM_LOCATION_UNKNOWN, &load_op));
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_vector_store_build(
      &builder_, 0, /*instance_flags=*/0, loom_vector_load_result(load_op),
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
  EXPECT_FALSE(RootIsStable(&table, loom_buffer_view_result(read_view_op)));
}

TEST_F(ViewRegionsTest, UnknownEffectsPreventStorageStability) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t view = BuildReadOnlyView(BuildNoAliasBuffer(buffer));
  loom_op_t* opaque = nullptr;
  IREE_ASSERT_OK(loom_test_clause_copy_build(&builder_, buffer, buffer,
                                             LOOM_LOCATION_UNKNOWN, &opaque));
  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(loom_view_region_table_root_access_flags(&table, buffer),
            LOOM_VIEW_ACCESS_READ);
  EXPECT_FALSE(RootIsStable(&table, view));
}

TEST_F(ViewRegionsTest, UnmodeledFencePreventsStorageStability) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t view = BuildReadOnlyView(BuildNoAliasBuffer(buffer));
  loom_op_t* fence = nullptr;
  IREE_ASSERT_OK(loom_test_memory_fence_build(
      &builder_, buffer, loom_type_buffer(), LOOM_LOCATION_UNKNOWN, &fence));
  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(loom_view_region_table_root_access_flags(&table, buffer),
            LOOM_VIEW_ACCESS_READ);
  EXPECT_FALSE(RootIsStable(&table, view));
}

TEST_F(ViewRegionsTest, RawByteWriteSharesTypedViewRoot) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t unique = BuildNoAliasBuffer(buffer);
  const loom_value_id_t view = BuildReadOnlyView(unique);
  const loom_value_id_t base =
      loom_index_constant_result(BuildOffsetConstant(0));
  const loom_value_id_t byte = DefineScalarArg(LOOM_SCALAR_TYPE_I32);
  loom_op_t* store = nullptr;
  IREE_ASSERT_OK(loom_buffer_store_i8_build(&builder_, byte, unique, base,
                                            LOOM_LOCATION_UNKNOWN, &store));
  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(loom_view_region_table_root_access_flags(&table, buffer),
            LOOM_VIEW_ACCESS_READ | LOOM_VIEW_ACCESS_WRITE);
  EXPECT_FALSE(RootIsStable(&table, view));
}

TEST_F(ViewRegionsTest, ReadOnlyStorageRequiresAliasOrConstantProof) {
  const loom_value_id_t unique =
      BuildReadOnlyView(BuildNoAliasBuffer(DefineBufferArg()));
  const loom_value_id_t unmarked = BuildReadOnlyView(DefineBufferArg());
  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(loom_buffer_assume_memory_space_build(
      &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT, DefineBufferArg(),
      loom_type_buffer(), LOOM_LOCATION_UNKNOWN, &constant));
  const loom_value_id_t immutable =
      BuildReadOnlyView(loom_buffer_assume_memory_space_result(constant));
  const loom_value_id_t unread =
      BuildView(BuildNoAliasBuffer(DefineBufferArg()));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_TRUE(RootIsStable(&table, unique));
  EXPECT_FALSE(RootIsStable(&table, unmarked));
  EXPECT_TRUE(RootIsStable(&table, immutable));
  EXPECT_FALSE(RootIsStable(&table, unread));
}

class AtomicStorageStabilityTest
    : public ViewRegionsTest,
      public ::testing::WithParamInterface<std::tuple<
          loom_atomic_ordering_t, loom_atomic_scope_t, bool, loom_op_kind_t>> {
};

TEST_P(AtomicStorageStabilityTest, AcquisitionInterferesWithSharedStorage) {
  const auto [ordering, scope, shared_stable, kind] = GetParam();
  const loom_value_id_t global =
      BuildReadOnlyView(BuildNoAliasBuffer(DefineBufferArg()));
  const loom_value_id_t workgroup = BuildReadOnlyView(BuildNoAliasBuffer(
      DefineBufferArg(), LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP));
  const loom_value_id_t private_view = BuildReadOnlyView(BuildNoAliasBuffer(
      DefineBufferArg(), LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE));
  const loom_value_id_t constant = BuildReadOnlyView(BuildNoAliasBuffer(
      DefineBufferArg(), LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT));
  // The token's memory space does not bound the acquired payload's space.
  const loom_value_id_t token =
      BuildView(BuildNoAliasBuffer(DefineBufferArg(),
                                   LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP),
                LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t value = DefineScalarArg(LOOM_SCALAR_TYPE_I32);
  const int64_t indices[] = {0};
  loom_op_t* atomic = nullptr;
  if (kind == LOOM_OP_VIEW_ATOMIC_LOAD) {
    IREE_ASSERT_OK(loom_view_atomic_load_build(
        &builder_, 0, token, nullptr, 0, indices, 1, ordering, scope, 0, 0,
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN,
        &atomic));
  } else if (kind == LOOM_OP_VIEW_ATOMIC_STORE) {
    IREE_ASSERT_OK(loom_view_atomic_store_build(
        &builder_, 0, value, token, nullptr, 0, indices, 1, ordering, scope, 0,
        0, LOOM_LOCATION_UNKNOWN, &atomic));
  } else {
    IREE_ASSERT_OK(loom_view_atomic_rmw_build(
        &builder_, 0, LOOM_ATOMIC_KIND_ADDI, /*instance_flags=*/0, value, token,
        nullptr, 0, indices, 1, ordering, scope, 0, 0,
        loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN,
        &atomic));
  }

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(RootIsStable(&table, global), shared_stable);
  EXPECT_EQ(RootIsStable(&table, workgroup), shared_stable);
  EXPECT_TRUE(RootIsStable(&table, private_view));
  EXPECT_TRUE(RootIsStable(&table, constant));
  EXPECT_EQ(RootIsStable(&table, token),
            kind == LOOM_OP_VIEW_ATOMIC_LOAD && shared_stable);
}

INSTANTIATE_TEST_SUITE_P(
    ViewRegions, AtomicStorageStabilityTest,
    ::testing::Values(
        std::make_tuple(LOOM_ATOMIC_ORDERING_RELAXED, LOOM_ATOMIC_SCOPE_DEVICE,
                        true, LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_RELEASE, LOOM_ATOMIC_SCOPE_DEVICE,
                        true, LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE, LOOM_ATOMIC_SCOPE_DEVICE,
                        false, LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_ACQ_REL, LOOM_ATOMIC_SCOPE_DEVICE,
                        false, LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_SEQ_CST, LOOM_ATOMIC_SCOPE_DEVICE,
                        false, LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE, LOOM_ATOMIC_SCOPE_THREAD,
                        true, LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE,
                        LOOM_ATOMIC_SCOPE_SUBGROUP, false,
                        LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE,
                        LOOM_ATOMIC_SCOPE_WORKGROUP, false,
                        LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE, LOOM_ATOMIC_SCOPE_SYSTEM,
                        false, LOOM_OP_VIEW_ATOMIC_RMW),
        std::make_tuple(LOOM_ATOMIC_ORDERING_RELAXED, LOOM_ATOMIC_SCOPE_DEVICE,
                        true, LOOM_OP_VIEW_ATOMIC_LOAD),
        std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE, LOOM_ATOMIC_SCOPE_DEVICE,
                        false, LOOM_OP_VIEW_ATOMIC_LOAD),
        std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE, LOOM_ATOMIC_SCOPE_THREAD,
                        true, LOOM_OP_VIEW_ATOMIC_LOAD),
        std::make_tuple(LOOM_ATOMIC_ORDERING_RELEASE, LOOM_ATOMIC_SCOPE_DEVICE,
                        true, LOOM_OP_VIEW_ATOMIC_STORE)));

class CompareExchangeStorageStabilityTest
    : public ViewRegionsTest,
      public ::testing::WithParamInterface<
          std::tuple<loom_atomic_ordering_t, loom_atomic_ordering_t, bool>> {};

TEST_P(CompareExchangeStorageStabilityTest, AcquiringOrderingsInterfere) {
  const auto [success_ordering, failure_ordering, stable] = GetParam();
  const loom_value_id_t payload =
      BuildReadOnlyView(BuildNoAliasBuffer(DefineBufferArg()));
  const loom_value_id_t token =
      BuildView(BuildNoAliasBuffer(DefineBufferArg()), LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t expected = DefineScalarArg(LOOM_SCALAR_TYPE_I32);
  const loom_value_id_t replacement = DefineScalarArg(LOOM_SCALAR_TYPE_I32);
  const int64_t indices[] = {0};
  loom_op_t* atomic = nullptr;
  IREE_ASSERT_OK(loom_view_atomic_cmpxchg_build(
      &builder_, 0, expected, replacement, token, nullptr, 0, indices, 1,
      success_ordering, failure_ordering, LOOM_ATOMIC_SCOPE_DEVICE, 0, 0,
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &atomic));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(RootIsStable(&table, payload), stable);
}

INSTANTIATE_TEST_SUITE_P(
    ViewRegions, CompareExchangeStorageStabilityTest,
    ::testing::Values(std::make_tuple(LOOM_ATOMIC_ORDERING_RELEASE,
                                      LOOM_ATOMIC_ORDERING_RELAXED, true),
                      std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE,
                                      LOOM_ATOMIC_ORDERING_RELAXED, false),
                      std::make_tuple(LOOM_ATOMIC_ORDERING_ACQ_REL,
                                      LOOM_ATOMIC_ORDERING_ACQUIRE, false)));

class FenceStorageStabilityTest
    : public ViewRegionsTest,
      public ::testing::WithParamInterface<std::tuple<
          loom_value_fact_memory_space_t, loom_atomic_ordering_t, bool, bool>> {
};

class ThreadFenceStorageStabilityTest
    : public ViewRegionsTest,
      public ::testing::WithParamInterface<
          std::tuple<loom_atomic_ordering_t, loom_atomic_scope_t, bool>> {};

TEST_P(ThreadFenceStorageStabilityTest, OnlyAcquisitionImportsSharedWrites) {
  const auto [ordering, scope, shared_stable] = GetParam();
  const loom_value_id_t global =
      BuildReadOnlyView(BuildNoAliasBuffer(DefineBufferArg()));
  const loom_value_id_t private_view = BuildReadOnlyView(BuildNoAliasBuffer(
      DefineBufferArg(), LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE));
  loom_op_t* fence = nullptr;
  IREE_ASSERT_OK(loom_buffer_fence_build(&builder_, scope, ordering,
                                         LOOM_LOCATION_UNKNOWN, &fence));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(RootIsStable(&table, global), shared_stable);
  EXPECT_TRUE(RootIsStable(&table, private_view));
}

INSTANTIATE_TEST_SUITE_P(
    ViewRegions, ThreadFenceStorageStabilityTest,
    ::testing::Values(std::make_tuple(LOOM_ATOMIC_ORDERING_RELEASE,
                                      LOOM_ATOMIC_SCOPE_SYSTEM, true),
                      std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE,
                                      LOOM_ATOMIC_SCOPE_SYSTEM, false),
                      std::make_tuple(LOOM_ATOMIC_ORDERING_ACQUIRE,
                                      LOOM_ATOMIC_SCOPE_THREAD, true)));

TEST_P(FenceStorageStabilityTest, InterferenceRespectsOrderingAndMemorySpace) {
  const auto [memory_space, ordering, global_stable, workgroup_stable] =
      GetParam();
  const loom_value_id_t global =
      BuildReadOnlyView(BuildNoAliasBuffer(DefineBufferArg()));
  const loom_value_id_t workgroup = BuildReadOnlyView(BuildNoAliasBuffer(
      DefineBufferArg(), LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP));
  loom_op_t* fence = nullptr;
  IREE_ASSERT_OK(loom_kernel_barrier_build(
      &builder_, memory_space, LOOM_ATOMIC_SCOPE_WORKGROUP, ordering,
      LOOM_LOCATION_UNKNOWN, &fence));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(RootIsStable(&table, global), global_stable);
  EXPECT_EQ(RootIsStable(&table, workgroup), workgroup_stable);
}

INSTANTIATE_TEST_SUITE_P(
    ViewRegions, FenceStorageStabilityTest,
    ::testing::Values(
        std::make_tuple(LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
                        LOOM_ATOMIC_ORDERING_ACQ_REL, true, false),
        std::make_tuple(LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
                        LOOM_ATOMIC_ORDERING_ACQUIRE, false, true),
        std::make_tuple(LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL,
                        LOOM_ATOMIC_ORDERING_RELEASE, true, true)));

TEST_F(ViewRegionsTest, SubviewWriteSharesStorageRoot) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t view = BuildReadOnlyView(BuildNoAliasBuffer(buffer));
  const int64_t indices[] = {0};
  loom_op_t* alias = nullptr;
  IREE_ASSERT_OK(loom_view_subview_build(
      &builder_, view, nullptr, 0, indices, 1,
      loom_module_value_type(module_, view), LOOM_LOCATION_UNKNOWN, &alias));
  loom_op_t* store = nullptr;
  IREE_ASSERT_OK(loom_view_store_build(
      &builder_, 0, 0, DefineScalarArg(LOOM_SCALAR_TYPE_F32),
      loom_view_subview_result(alias), nullptr, 0, indices, 1, 0, 0,
      LOOM_LOCATION_UNKNOWN, &store));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(loom_view_region_table_root_access_flags(&table, buffer),
            LOOM_VIEW_ACCESS_READ | LOOM_VIEW_ACCESS_WRITE);
  EXPECT_FALSE(RootIsStable(&table, view));
}

TEST_F(ViewRegionsTest, WriteWithoutComparableAliasPreventsStorageStability) {
  const loom_value_id_t buffer = DefineBufferArg();
  const loom_value_id_t view = BuildReadOnlyView(BuildNoAliasBuffer(buffer));
  loom_op_t* store = nullptr;
  IREE_ASSERT_OK(loom_buffer_store_i8_build(
      &builder_, DefineScalarArg(LOOM_SCALAR_TYPE_I32), DefineBufferArg(),
      loom_index_constant_result(BuildOffsetConstant(0)), LOOM_LOCATION_UNKNOWN,
      &store));

  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  EXPECT_EQ(loom_view_region_table_root_access_flags(&table, buffer),
            LOOM_VIEW_ACCESS_READ);
  EXPECT_FALSE(RootIsStable(&table, view));
}

TEST_F(ViewRegionsTest, PrecomputesReusedMemoryIndexExpression) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t source_index = DefineIndexArg();
  loom_value_id_t unrelated_index = DefineIndexArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));
  loom_value_id_t one = loom_index_constant_result(BuildIndexConstant(1));

  loom_op_t* view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, 0, buffer, zero, 0,
                                        ViewType1D(4096, layout),
                                        LOOM_LOCATION_UNKNOWN, &view_op));

  loom_value_id_t deep_index = source_index;
  for (int i = 0; i < 40; ++i) {
    loom_op_t* add_op = nullptr;
    IREE_ASSERT_OK(loom_index_add_build(
        &builder_, deep_index, one, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &add_op));
    deep_index = loom_index_add_result(add_op);
  }

  const loom_value_id_t dynamic_indices[] = {deep_index};
  const int64_t static_indices[] = {INT64_MIN};
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, /*instance_flags=*/0, loom_buffer_view_result(view_op),
      dynamic_indices, IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, VectorType1D(1),
      LOOM_LOCATION_UNKNOWN, &load_op));
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_vector_store_build(
      &builder_, 0, /*instance_flags=*/0, loom_vector_load_result(load_op),
      loom_buffer_view_result(view_op), dynamic_indices,
      IREE_ARRAYSIZE(dynamic_indices), static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, LOOM_LOCATION_UNKNOWN, &store_op));

  loom_value_fact_table_t facts = {0};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {0};
  Analyze(&facts, &table);

  loom_symbolic_expr_summary_t summary = {};
  ASSERT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      table.expression_context, deep_index, &summary));
  ASSERT_TRUE(loom_symbolic_expr_is_linear(&summary.expression));
  EXPECT_EQ(summary.expression.constant, 40);
  ASSERT_EQ(summary.expression.term_count, 1u);
  EXPECT_EQ(summary.expression.terms[0].value_id, source_index);
  loom_symbolic_expr_summary_t repeated_summary = {};
  EXPECT_TRUE(loom_symbolic_expr_context_try_lookup_summary(
      table.expression_context, deep_index, &repeated_summary));
  EXPECT_EQ(repeated_summary.expression.terms, summary.expression.terms);
  loom_symbolic_expr_summary_t unrelated_summary = {};
  EXPECT_FALSE(loom_symbolic_expr_context_try_lookup_summary(
      table.expression_context, unrelated_index, &unrelated_summary));
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
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, 0, buffer, base_offset, 0,
                                        ViewType1D(16, layout),
                                        LOOM_LOCATION_UNKNOWN, &first_view_op));
  loom_op_t* second_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, 0, buffer, loom_index_add_result(second_offset_op), 0,
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
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, 0, first_buffer, zero, 0,
                                        ViewType1D(16, layout),
                                        LOOM_LOCATION_UNKNOWN, &first_view_op));
  loom_op_t* overlapping_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, 0, first_buffer, thirty_two, 0, ViewType1D(16, layout),
      LOOM_LOCATION_UNKNOWN, &overlapping_view_op));
  loom_op_t* other_root_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, 0, second_buffer, zero, 0, ViewType1D(16, layout),
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

TEST_F(ViewRegionsTest, AllocationFreshnessRelationships) {
  loom_value_id_t incoming = DefineBufferArg();
  loom_value_id_t other_incoming = DefineBufferArg();
  loom_value_id_t bytes = loom_index_constant_result(BuildOffsetConstant(256));
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));
  loom_value_id_t separate_offset =
      loom_index_constant_result(BuildOffsetConstant(128));
  loom_value_id_t layout = BuildDenseLayout();
  loom_op_t* first_allocation = nullptr;
  loom_op_t* second_allocation = nullptr;
  IREE_ASSERT_OK(loom_buffer_alloca_build(
      &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 64, bytes,
      loom_type_buffer(), LOOM_LOCATION_UNKNOWN, &first_allocation));
  IREE_ASSERT_OK(loom_buffer_alloca_build(
      &builder_, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, 64, bytes,
      loom_type_buffer(), LOOM_LOCATION_UNKNOWN, &second_allocation));
  const loom_value_id_t roots[] = {
      loom_buffer_alloca_result(first_allocation),
      loom_buffer_alloca_result(second_allocation),
      loom_buffer_alloca_result(first_allocation),
      incoming,
      other_incoming,
      incoming,
  };
  loom_op_t* views[IREE_ARRAYSIZE(roots)] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(roots); ++i) {
    IREE_ASSERT_OK(loom_buffer_view_build(
        &builder_, 0, roots[i], i == 5 ? separate_offset : zero, 0,
        ViewType1D(16, layout), LOOM_LOCATION_UNKNOWN, &views[i]));
  }
  loom_value_fact_table_t facts = {};
  ComputeFacts(&facts);
  loom_view_region_table_t table = {};
  Analyze(&facts, &table);
  const loom_view_region_t* regions[IREE_ARRAYSIZE(roots)] = {};
  for (size_t i = 0; i < IREE_ARRAYSIZE(roots); ++i) {
    IREE_ASSERT_OK(loom_view_region_table_get(
        &table, loom_buffer_view_result(views[i]), &regions[i]));
    ASSERT_NE(regions[i], nullptr);
  }
  struct Comparison {
    // Relationship exercised by this pair.
    const char* name;
    // First region index.
    size_t left;
    // Second region index.
    size_t right;
    // Whether the storage/interval guarantees prove disjointness.
    bool expected_disjoint;
  };
  const Comparison comparisons[] = {
      {"two_fresh_allocations", 0, 1, true},
      {"two_overlapping_views_of_one_allocation", 0, 2, false},
      {"fresh_allocation_and_incoming_buffer", 0, 3, true},
      {"two_unmarked_incoming_buffers", 3, 4, false},
      {"disjoint_ranges_of_one_unmarked_buffer", 3, 5, true},
  };
  for (const auto& comparison : comparisons) {
    bool disjoint = false;
    IREE_ASSERT_OK(loom_view_regions_prove_no_overlap(
        &table, regions[comparison.left], regions[comparison.right],
        &disjoint));
    EXPECT_EQ(disjoint, comparison.expected_disjoint) << comparison.name;
  }
}

TEST_F(ViewRegionsTest, ProvesDistinctComparableRootsDisjoint) {
  loom_value_id_t first_buffer = DefineBufferArg();
  loom_value_id_t second_buffer = DefineBufferArg();
  loom_value_id_t buffers[] = {first_buffer, second_buffer};
  loom_type_t buffer_types[] = {loom_type_buffer(), loom_type_buffer()};
  loom_op_t* noalias_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_assume_noalias_build(
      &builder_, buffers, IREE_ARRAYSIZE(buffers), buffer_types,
      IREE_ARRAYSIZE(buffer_types), LOOM_LOCATION_UNKNOWN, &noalias_op));
  loom_value_slice_t noalias_buffers =
      loom_buffer_assume_noalias_results(noalias_op);
  ASSERT_EQ(noalias_buffers.count, 2);

  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t zero = loom_index_constant_result(BuildOffsetConstant(0));
  loom_op_t* first_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, 0, noalias_buffers.values[0],
                                        zero, 0, ViewType1D(16, layout),
                                        LOOM_LOCATION_UNKNOWN, &first_view_op));
  loom_op_t* second_view_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, 0, noalias_buffers.values[1], zero, 0, ViewType1D(16, layout),
      LOOM_LOCATION_UNKNOWN, &second_view_op));

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
  EXPECT_EQ(first_region->root_value_id, first_buffer);
  EXPECT_EQ(first_region->alias_scope_id, first_buffer);
  EXPECT_EQ(second_region->root_value_id, second_buffer);
  EXPECT_EQ(second_region->alias_scope_id, second_buffer);

  bool no_overlap = false;
  IREE_ASSERT_OK(loom_view_regions_prove_no_overlap(
      &table, first_region, second_region, &no_overlap));
  EXPECT_TRUE(no_overlap);
}

TEST_F(ViewRegionsTest, SubviewPreservesRootAndAddsLogicalOffset) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t base_offset = DefineOffsetArg();
  loom_value_id_t layout = BuildDenseLayout();

  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_buffer_view_build(
      &builder_, LOOM_BUFFER_VIEW_BUILD_FLAG_HAS_ADDRESS_BITWIDTH, buffer,
      base_offset, 32, ViewType2D(8, 16, layout), LOOM_LOCATION_UNKNOWN,
      &source_op));

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
  EXPECT_EQ(region->address_bitwidth, 32);
  EXPECT_EQ(region->alias_scope_id, LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE);
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
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, 0, buffer, zero, 0,
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
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, 0, buffer, zero, 0,
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
  IREE_ASSERT_OK(loom_buffer_view_build(&builder_, 0, buffer, zero, 0,
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
