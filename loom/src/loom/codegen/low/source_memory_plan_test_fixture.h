// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_TEST_FIXTURE_H_
#define LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_TEST_FIXTURE_H_

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_table.h"

namespace loom {

class SourceMemoryPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
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
    IREE_EXPECT_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_buffer(), &buffer));
    return buffer;
  }

  loom_value_id_t DefineIndexArg() {
    loom_value_id_t index = LOOM_VALUE_ID_INVALID;
    IREE_EXPECT_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &index));
    return index;
  }

  loom_value_id_t DefineOffsetArg() {
    loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
    IREE_EXPECT_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), &offset));
    return offset;
  }

  loom_value_id_t DefineArg(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_EXPECT_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        type, &value));
    return value;
  }

  loom_value_id_t BuildForwardedIndexBlockArg(loom_value_id_t source) {
    loom_region_t* body = loom_func_like_body(function_);
    loom_block_t* successor = nullptr;
    IREE_EXPECT_OK(loom_region_append_block(module_, body, &successor));
    loom_value_id_t forwarded = LOOM_VALUE_ID_INVALID;
    IREE_EXPECT_OK(loom_builder_define_block_arg(
        &builder_, successor, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        &forwarded));
    loom_op_t* branch_op = nullptr;
    IREE_EXPECT_OK(loom_cfg_br_build(&builder_, successor, &source, 1,
                                     LOOM_LOCATION_UNKNOWN, &branch_op));
    loom_builder_initialize(module_, &module_->arena, successor, &builder_);
    return forwarded;
  }

  loom_value_id_t BuildNoalias(loom_value_id_t buffer) {
    loom_op_t* op = nullptr;
    const loom_type_t result_type = loom_type_buffer();
    IREE_EXPECT_OK(loom_buffer_assume_noalias_build(
        &builder_, &buffer, 1, &result_type, 1, LOOM_LOCATION_UNKNOWN, &op));
    return loom_buffer_assume_noalias_results(op).values[0];
  }

  loom_value_id_t BuildAligned(loom_value_id_t buffer,
                               int64_t minimum_alignment) {
    loom_op_t* op = nullptr;
    const loom_type_t result_type = loom_type_buffer();
    IREE_EXPECT_OK(loom_buffer_assume_alignment_build(
        &builder_, &buffer, 1, minimum_alignment, &result_type, 1,
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_buffer_assume_alignment_results(op).values[0];
  }

  loom_op_t* BuildOffsetConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildIndexConstant(int64_t value) {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_index_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_value_id_t BuildIndexAssumeRange(loom_value_id_t source,
                                        int64_t minimum_value,
                                        int64_t maximum_value) {
    loom_predicate_t* predicate = nullptr;
    IREE_EXPECT_OK(iree_arena_allocate_array(
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
    IREE_EXPECT_OK(loom_index_assume_build(&builder_, &source, 1, predicate, 1,
                                           &result_type, 1,
                                           LOOM_LOCATION_UNKNOWN, &op));
    return loom_index_assume_results(op).values[0];
  }

  loom_value_id_t BuildDenseLayout() {
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_encoding_layout_dense_build(
        &builder_,
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_dense_result(op);
  }

  loom_value_id_t BuildStridedLayout(int64_t row_stride,
                                     int64_t column_stride) {
    const int64_t static_strides[] = {row_stride, column_stride};
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_encoding_layout_strided_build(
        &builder_, /*strides=*/nullptr, /*strides_count=*/0, static_strides,
        IREE_ARRAYSIZE(static_strides),
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_strided_result(op);
  }

  loom_value_id_t BuildDynamicStridedLayout(loom_value_id_t row_stride,
                                            int64_t column_stride) {
    const int64_t static_strides[] = {INT64_MIN, column_stride};
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_encoding_layout_strided_build(
        &builder_, &row_stride, 1, static_strides,
        IREE_ARRAYSIZE(static_strides),
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_strided_result(op);
  }

  loom_type_t ViewType1D(loom_scalar_type_t element_type, int64_t extent,
                         loom_value_id_t layout) {
    loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VIEW, element_type,
                                           loom_dim_pack_static(extent),
                                           /*encoding_id=*/0);
    type.encoding_id = (uint16_t)layout;
    type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return type;
  }

  loom_type_t ViewType1D(int64_t extent, loom_value_id_t layout) {
    return ViewType1D(LOOM_SCALAR_TYPE_F32, extent, layout);
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

  loom_type_t ViewType3D(uint64_t blocks, uint64_t rows, uint64_t columns,
                         loom_value_id_t layout) {
    loom_overflow_dim_t* dimensions = nullptr;
    IREE_EXPECT_OK(iree_arena_allocate_array(
        &module_->arena, 3, sizeof(*dimensions), (void**)&dimensions));
    dimensions[0] = blocks;
    dimensions[1] = rows;
    dimensions[2] = columns;
    loom_type_t type = {};
    type.header =
        loom_type_make_header(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, 3, 0);
    type.encoding_id = (uint16_t)layout;
    type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    type.dims[0] = (uint64_t)(uintptr_t)dimensions;
    return type;
  }

  loom_type_t VectorType1D(loom_scalar_type_t element_type, int64_t extent) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                               loom_dim_pack_static(extent), 0);
  }

  loom_type_t VectorType1D(int64_t extent) {
    return VectorType1D(LOOM_SCALAR_TYPE_F32, extent);
  }

  loom_value_id_t BuildIndexIotaOffsets(int64_t base, int64_t step,
                                        loom_type_t vector_type) {
    loom_op_t* base_op = BuildIndexConstant(base);
    loom_op_t* step_op = BuildIndexConstant(step);
    loom_op_t* iota_op = nullptr;
    IREE_EXPECT_OK(
        loom_vector_iota_build(&builder_, loom_index_constant_result(base_op),
                               loom_index_constant_result(step_op), vector_type,
                               LOOM_LOCATION_UNKNOWN, &iota_op));
    return loom_vector_iota_result(iota_op);
  }

  loom_value_id_t BuildIndexFromElementOffsets(int64_t first, int64_t second,
                                               loom_type_t vector_type) {
    loom_op_t* first_op = BuildIndexConstant(first);
    loom_op_t* second_op = BuildIndexConstant(second);
    loom_value_id_t elements[] = {
        loom_index_constant_result(first_op),
        loom_index_constant_result(second_op),
    };
    loom_op_t* offsets_op = nullptr;
    IREE_EXPECT_OK(loom_vector_from_elements_build(
        &builder_, elements, IREE_ARRAYSIZE(elements), vector_type,
        LOOM_LOCATION_UNKNOWN, &offsets_op));
    return loom_vector_from_elements_result(offsets_op);
  }

  void ComputeFacts(loom_value_fact_table_t* out_facts) {
    IREE_ASSERT_OK(loom_value_fact_table_initialize(out_facts, &analysis_arena_,
                                                    module_->values.count));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(out_facts, module_, function_));
  }

  bool BuildPlan(loom_value_fact_table_t* facts, const loom_op_t* op,
                 loom_low_source_memory_access_plan_t* out_plan,
                 loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
    loom_local_value_domain_t value_domain = {};
    IREE_EXPECT_OK(loom_local_value_domain_acquire_for_region(
        module_, loom_func_like_body(function_), &analysis_arena_,
        &value_domain));
    loom_view_region_table_t view_regions = {};
    loom_symbolic_expr_context_t expression_context = {};
    loom_symbolic_expr_context_initialize(
        module_, &value_domain, facts, &analysis_arena_, &expression_context);
    IREE_EXPECT_OK(loom_view_region_table_initialize(
        &value_domain, &expression_context, &view_regions));
    IREE_EXPECT_OK(loom_view_region_table_analyze(&view_regions));
    const bool built = loom_low_source_memory_access_plan_build(
        &view_regions, op, out_plan, out_diagnostic);
    loom_local_value_domain_release(&value_domain);
    return built;
  }

  bool BuildViewPlan(
      loom_value_fact_table_t* facts, loom_value_id_t view,
      loom_low_source_memory_access_plan_t* out_plan,
      loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
    loom_vector_memory_cache_policy_t cache_policy = {0};
    loom_local_value_domain_t value_domain = {};
    IREE_EXPECT_OK(loom_local_value_domain_acquire_for_region(
        module_, loom_func_like_body(function_), &analysis_arena_,
        &value_domain));
    loom_view_region_table_t view_regions = {};
    loom_symbolic_expr_context_t expression_context = {};
    loom_symbolic_expr_context_initialize(
        module_, &value_domain, facts, &analysis_arena_, &expression_context);
    IREE_EXPECT_OK(loom_view_region_table_initialize(
        &value_domain, &expression_context, &view_regions));
    IREE_EXPECT_OK(loom_view_region_table_analyze(&view_regions));
    const bool built = loom_low_source_memory_access_plan_build_view(
        &view_regions, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, view,
        cache_policy, out_plan, out_diagnostic);
    loom_local_value_domain_release(&value_domain);
    return built;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_;
  loom_builder_t builder_;
};

}  // namespace loom

#endif  // LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_TEST_FIXTURE_H_
