// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/movement.h"

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
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class MovementTest : public ::testing::Test {
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
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
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
                                              IREE_SV("movement"), &name_id));
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

  loom_value_id_t DefineArg(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_builder_define_block_arg(
        &builder_, loom_region_entry_block(loom_func_like_body(function_)),
        type, &value));
    return value;
  }

  loom_value_id_t DefineBufferArg() { return DefineArg(loom_type_buffer()); }

  loom_value_id_t BuildAllocaBuffer(int64_t byte_length,
                                    int64_t base_alignment) {
    loom_value_id_t byte_length_value =
        loom_index_constant_result(BuildOffsetConstant(byte_length));
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_buffer_alloca_build(
        &builder_, byte_length_value, base_alignment,
        LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP, loom_type_buffer(),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_buffer_alloca_result(op);
  }

  loom_value_id_t DefineVectorArg(loom_type_t type) { return DefineArg(type); }

  loom_value_id_t DefinePredicateArg() {
    return DefineArg(loom_type_scalar(LOOM_SCALAR_TYPE_I1));
  }

  loom_value_id_t DefineI32Arg() {
    return DefineArg(loom_type_scalar(LOOM_SCALAR_TYPE_I32));
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

  loom_value_id_t BuildStaticStridedLayout(int64_t row_stride,
                                           int64_t column_stride) {
    int64_t* strides = nullptr;
    IREE_CHECK_OK(
        iree_arena_allocate_array(&module_->arena, 2, sizeof(*strides),
                                  reinterpret_cast<void**>(&strides)));
    strides[0] = row_stride;
    strides[1] = column_stride;
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_encoding_layout_strided_build(
        &builder_, nullptr, 0, strides, 2,
        loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
        LOOM_LOCATION_UNKNOWN, &op));
    return loom_encoding_layout_strided_result(op);
  }

  loom_type_t ViewType1D(loom_scalar_type_t element_type, int64_t extent,
                         loom_value_id_t layout) {
    loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VIEW, element_type,
                                           loom_dim_pack_static(extent), 0);
    type.encoding_id = (uint16_t)layout;
    type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return type;
  }

  loom_type_t ViewType2D(loom_scalar_type_t element_type, int64_t rows,
                         int64_t columns, loom_value_id_t layout) {
    loom_type_t type = loom_type_shaped_2d(LOOM_TYPE_VIEW, element_type,
                                           loom_dim_pack_static(rows),
                                           loom_dim_pack_static(columns), 0);
    type.encoding_id = (uint16_t)layout;
    type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return type;
  }

  loom_type_t VectorType1D(loom_scalar_type_t element_type, int64_t extent) {
    return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                               loom_dim_pack_static(extent), 0);
  }

  loom_type_t KernelAsyncTokenType() {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(
        &builder_, IREE_SV("kernel.async.token"), &name_id));
    return loom_type_dialect_opaque(name_id);
  }

  loom_type_t KernelTensorLdsDescriptorType() {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_builder_intern_string(
        &builder_, IREE_SV("kernel.tensor.lds.descriptor"), &name_id));
    return loom_type_dialect_opaque(name_id);
  }

  loom_value_id_t BuildBufferView(loom_value_id_t buffer, int64_t byte_offset,
                                  loom_type_t view_type) {
    loom_value_id_t offset =
        loom_index_constant_result(BuildOffsetConstant(byte_offset));
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_buffer_view_build(&builder_, buffer, offset, view_type,
                                         LOOM_LOCATION_UNKNOWN, &op));
    return loom_buffer_view_result(op);
  }

  void ComputeFacts(loom_value_fact_table_t* out_facts) {
    IREE_ASSERT_OK(loom_value_fact_table_initialize(out_facts, &analysis_arena_,
                                                    module_->values.count));
    IREE_ASSERT_OK(
        loom_value_fact_table_compute(out_facts, module_, function_));
  }

  void InitializeAnalysis(loom_movement_analysis_t* out_analysis) {
    ComputeFacts(&facts_);
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, loom_func_like_body(function_), &analysis_arena_,
        &value_domain_));
    IREE_ASSERT_OK(loom_movement_analysis_initialize(
        &facts_, &value_domain_, &analysis_arena_, out_analysis));
    IREE_ASSERT_OK(loom_movement_analysis_analyze(out_analysis));
  }

  bool Describe(loom_movement_analysis_t* analysis, const loom_op_t* op,
                loom_movement_request_t* out_request,
                loom_movement_diagnostic_t* out_diagnostic) {
    bool described = false;
    IREE_EXPECT_OK(loom_movement_request_describe_op(
        analysis, op, out_request, out_diagnostic, &described));
    return described;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_;
  loom_builder_t builder_;
  loom_value_fact_table_t facts_ = {};
  loom_local_value_domain_t value_domain_ = {};
};

TEST_F(MovementTest, ClassifiesStaticDenseVectorLoadFootprint) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t view =
      BuildBufferView(buffer, 16, ViewType1D(LOOM_SCALAR_TYPE_F32, 32, layout));
  int64_t static_indices[] = {3};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(
      &builder_, 0, view, nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0,
      VectorType1D(LOOM_SCALAR_TYPE_F32, 4), LOOM_LOCATION_UNKNOWN, &op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_VECTOR_LOAD);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_DENSE);
  EXPECT_EQ(request.schema_kind, LOOM_MOVEMENT_SCHEMA_TYPED_ELEMENT);
  EXPECT_TRUE(
      iree_all_bits_set(request.flags, LOOM_MOVEMENT_REQUEST_STATIC_TRANSFER));
  EXPECT_EQ(request.transferred_byte_count, 16);
  EXPECT_EQ(request.source.kind, LOOM_MOVEMENT_ENDPOINT_VIEW);
  EXPECT_EQ(request.source.root_value_id, buffer);
  EXPECT_EQ(request.source.static_begin_byte_offset, 28);
  EXPECT_EQ(request.source.static_byte_length, 16);
  EXPECT_EQ(request.dest.kind, LOOM_MOVEMENT_ENDPOINT_REGISTER);
  EXPECT_EQ(diagnostic.rejection_bits, 0u);
}

TEST_F(MovementTest, CarriesViewAlignmentFacts) {
  loom_value_id_t buffer = BuildAllocaBuffer(256, 64);
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t aligned_view =
      BuildBufferView(buffer, 0, ViewType1D(LOOM_SCALAR_TYPE_F32, 32, layout));
  loom_value_id_t misaligned_view =
      BuildBufferView(buffer, 28, ViewType1D(LOOM_SCALAR_TYPE_F32, 32, layout));
  int64_t static_indices[] = {0};

  loom_op_t* aligned_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(&builder_, 0, aligned_view, nullptr, 0,
                                        static_indices,
                                        IREE_ARRAYSIZE(static_indices), 0, 0,
                                        VectorType1D(LOOM_SCALAR_TYPE_F32, 4),
                                        LOOM_LOCATION_UNKNOWN, &aligned_op));
  loom_op_t* misaligned_op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_build(&builder_, 0, misaligned_view, nullptr,
                                        0, static_indices,
                                        IREE_ARRAYSIZE(static_indices), 0, 0,
                                        VectorType1D(LOOM_SCALAR_TYPE_F32, 4),
                                        LOOM_LOCATION_UNKNOWN, &misaligned_op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, aligned_op, &request, &diagnostic));
  EXPECT_EQ(request.source.root_minimum_alignment, 64u);
  EXPECT_EQ(loom_movement_endpoint_minimum_byte_alignment(&request.source),
            64u);

  ASSERT_TRUE(Describe(&analysis, misaligned_op, &request, &diagnostic));
  EXPECT_EQ(request.source.root_minimum_alignment, 64u);
  EXPECT_EQ(request.source.minimum_alignment, 4u);
  EXPECT_EQ(loom_movement_endpoint_minimum_byte_alignment(&request.source), 4u);
}

TEST_F(MovementTest, ClassifiesStaticStridedVectorStore) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t value =
      DefineVectorArg(VectorType1D(LOOM_SCALAR_TYPE_F32, 4));
  loom_value_id_t layout = BuildStaticStridedLayout(8, 1);
  loom_value_id_t view = BuildBufferView(
      buffer, 0, ViewType2D(LOOM_SCALAR_TYPE_F32, 8, 8, layout));
  int64_t static_indices[] = {1, 0};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_vector_store_build(
      &builder_, 0, value, view, nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), 0, 0, LOOM_LOCATION_UNKNOWN, &op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_VECTOR_STORE);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_STATIC_STRIDED);
  EXPECT_EQ(request.source.kind, LOOM_MOVEMENT_ENDPOINT_REGISTER);
  EXPECT_EQ(request.dest.kind, LOOM_MOVEMENT_ENDPOINT_VIEW);
  EXPECT_EQ(request.dest.root_value_id, buffer);
  EXPECT_EQ(request.dest.static_begin_byte_offset, 32);
  EXPECT_EQ(request.dest.static_byte_length, 16);
  int64_t lane_axis_stride = 0;
  ASSERT_TRUE(loom_vector_memory_access_static_axis_stride(
      &request.vector_access, request.vector_access.first_vector_axis,
      &lane_axis_stride));
  EXPECT_EQ(lane_axis_stride, 1);
}

TEST_F(MovementTest, ClassifiesMaskedLoadPolicyAndMask) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t mask = DefineVectorArg(VectorType1D(LOOM_SCALAR_TYPE_I1, 4));
  loom_value_id_t passthrough =
      DefineVectorArg(VectorType1D(LOOM_SCALAR_TYPE_F32, 4));
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t view =
      BuildBufferView(buffer, 0, ViewType1D(LOOM_SCALAR_TYPE_F32, 32, layout));
  int64_t static_indices[] = {0};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_vector_load_mask_build(
      &builder_, LOOM_VECTOR_LOAD_MASK_BUILD_FLAG_HAS_CACHE_SCOPE, view,
      nullptr, 0, static_indices, IREE_ARRAYSIZE(static_indices), mask,
      passthrough, LOOM_CACHE_SCOPE_CU, 0,
      VectorType1D(LOOM_SCALAR_TYPE_F32, 4), LOOM_LOCATION_UNKNOWN, &op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_VECTOR_LOAD_MASK);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_DENSE);
  EXPECT_TRUE(iree_any_bit_set(request.flags, LOOM_MOVEMENT_REQUEST_MASKED));
  EXPECT_EQ(request.mask_value_id, mask);
  EXPECT_TRUE(
      iree_any_bit_set(request.cache_policy.build_flags,
                       LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE));
  EXPECT_EQ(request.cache_policy.cache_scope, LOOM_CACHE_SCOPE_CU);
}

TEST_F(MovementTest, ClassifiesVectorGatherOffsets) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t offsets =
      DefineVectorArg(VectorType1D(LOOM_SCALAR_TYPE_INDEX, 4));
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t view =
      BuildBufferView(buffer, 0, ViewType1D(LOOM_SCALAR_TYPE_F32, 32, layout));
  int64_t static_indices[] = {0};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_vector_gather_build(
      &builder_, 0, view, nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), offsets, 0, 0,
      VectorType1D(LOOM_SCALAR_TYPE_F32, 4), LOOM_LOCATION_UNKNOWN, &op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_VECTOR_GATHER);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_GATHER_SCATTER);
  EXPECT_TRUE(
      iree_any_bit_set(request.flags, LOOM_MOVEMENT_REQUEST_IRREGULAR_OFFSETS));
  EXPECT_EQ(request.offsets_value_id, offsets);
  EXPECT_EQ(request.transferred_byte_count, 16);
}

TEST_F(MovementTest, ClassifiesMaskedScatterOffsets) {
  loom_value_id_t buffer = DefineBufferArg();
  loom_value_id_t value =
      DefineVectorArg(VectorType1D(LOOM_SCALAR_TYPE_F32, 4));
  loom_value_id_t offsets =
      DefineVectorArg(VectorType1D(LOOM_SCALAR_TYPE_INDEX, 4));
  loom_value_id_t mask = DefineVectorArg(VectorType1D(LOOM_SCALAR_TYPE_I1, 4));
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t view =
      BuildBufferView(buffer, 0, ViewType1D(LOOM_SCALAR_TYPE_F32, 32, layout));
  int64_t static_indices[] = {0};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_vector_scatter_mask_build(
      &builder_, 0, value, view, nullptr, 0, static_indices,
      IREE_ARRAYSIZE(static_indices), offsets, mask, 0, 0,
      LOOM_LOCATION_UNKNOWN, &op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_VECTOR_SCATTER_MASK);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_GATHER_SCATTER);
  EXPECT_TRUE(iree_all_bits_set(
      request.flags,
      LOOM_MOVEMENT_REQUEST_IRREGULAR_OFFSETS | LOOM_MOVEMENT_REQUEST_MASKED));
  EXPECT_EQ(request.offsets_value_id, offsets);
  EXPECT_EQ(request.mask_value_id, mask);
  EXPECT_EQ(request.dest.kind, LOOM_MOVEMENT_ENDPOINT_VIEW);
}

TEST_F(MovementTest, ClassifiesAsyncCopyAsBytePreserving) {
  loom_value_id_t source_buffer = DefineBufferArg();
  loom_value_id_t dest_buffer = DefineBufferArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t source = BuildBufferView(
      source_buffer, 0, ViewType1D(LOOM_SCALAR_TYPE_F32, 4, layout));
  loom_value_id_t dest = BuildBufferView(
      dest_buffer, 64, ViewType1D(LOOM_SCALAR_TYPE_I8, 16, layout));
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_kernel_async_copy_build(
      &builder_, source, dest, LOOM_CACHE_SCOPE_CU, LOOM_CACHE_TEMPORAL_REGULAR,
      LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP, KernelAsyncTokenType(),
      LOOM_LOCATION_UNKNOWN, &op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_BYTE_RANGE);
  EXPECT_EQ(request.schema_kind, LOOM_MOVEMENT_SCHEMA_BYTE_PRESERVING);
  EXPECT_TRUE(iree_all_bits_set(request.flags,
                                LOOM_MOVEMENT_REQUEST_ASYNC |
                                    LOOM_MOVEMENT_REQUEST_ASYNC_ELIGIBLE |
                                    LOOM_MOVEMENT_REQUEST_STATIC_TRANSFER));
  EXPECT_EQ(request.source.root_value_id, source_buffer);
  EXPECT_EQ(request.dest.root_value_id, dest_buffer);
  EXPECT_EQ(request.dest.static_begin_byte_offset, 64);
  EXPECT_EQ(request.transferred_byte_count, 16);
}

TEST_F(MovementTest, ClassifiesAsyncGatherAsSubgroupGather) {
  loom_value_id_t source_buffer = DefineBufferArg();
  loom_value_id_t dest_buffer = DefineBufferArg();
  loom_value_id_t predicate = DefinePredicateArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t source = BuildBufferView(
      source_buffer, 0, ViewType1D(LOOM_SCALAR_TYPE_I8, 4, layout));
  loom_value_id_t dest = BuildBufferView(
      dest_buffer, 128, ViewType2D(LOOM_SCALAR_TYPE_I8, 64, 4, layout));
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_kernel_async_gather_mask_build(
      &builder_, source, dest, predicate, LOOM_CACHE_SCOPE_CU,
      LOOM_CACHE_TEMPORAL_REGULAR, KernelAsyncTokenType(),
      LOOM_LOCATION_UNKNOWN, &op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_KERNEL_ASYNC_GATHER_MASK);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_SUBGROUP_GATHER);
  EXPECT_EQ(request.schema_kind, LOOM_MOVEMENT_SCHEMA_BYTE_PRESERVING);
  EXPECT_TRUE(iree_any_bit_set(request.flags, LOOM_MOVEMENT_REQUEST_ASYNC));
  EXPECT_TRUE(iree_any_bit_set(request.flags, LOOM_MOVEMENT_REQUEST_MASKED));
  EXPECT_EQ(request.mask_value_id, predicate);
  EXPECT_EQ(request.source.static_byte_length, 4);
  EXPECT_EQ(request.dest.static_begin_byte_offset, 128);
  EXPECT_EQ(request.transferred_byte_count, 4);
}

TEST_F(MovementTest, ClassifiesAsyncClusterGatherControlOperands) {
  loom_value_id_t source_buffer = DefineBufferArg();
  loom_value_id_t dest_buffer = DefineBufferArg();
  loom_value_id_t cluster_mask = DefineI32Arg();
  loom_value_id_t predicate = DefinePredicateArg();
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t source = BuildBufferView(
      source_buffer, 0, ViewType1D(LOOM_SCALAR_TYPE_I8, 16, layout));
  loom_value_id_t dest = BuildBufferView(
      dest_buffer, 256, ViewType1D(LOOM_SCALAR_TYPE_I8, 16, layout));
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_kernel_async_cluster_gather_mask_build(
      &builder_, source, dest, cluster_mask, predicate, LOOM_CACHE_SCOPE_SE,
      LOOM_CACHE_TEMPORAL_HIGH_TEMPORAL, KernelAsyncTokenType(),
      LOOM_LOCATION_UNKNOWN, &op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_KERNEL_ASYNC_CLUSTER_GATHER_MASK);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_CLUSTER_GATHER);
  EXPECT_TRUE(iree_all_bits_set(request.flags,
                                LOOM_MOVEMENT_REQUEST_ASYNC |
                                    LOOM_MOVEMENT_REQUEST_MASKED |
                                    LOOM_MOVEMENT_REQUEST_ASYNC_ELIGIBLE |
                                    LOOM_MOVEMENT_REQUEST_STATIC_TRANSFER));
  EXPECT_EQ(request.cluster_mask_value_id, cluster_mask);
  EXPECT_EQ(request.mask_value_id, predicate);
  EXPECT_EQ(request.transferred_byte_count, 16);
  EXPECT_EQ(request.dest.static_begin_byte_offset, 256);
}

TEST_F(MovementTest, ClassifiesAsyncTensorDescriptorsAndDirections) {
  loom_value_id_t source_buffer = DefineBufferArg();
  loom_value_id_t lds_buffer = DefineBufferArg();
  loom_value_id_t dest_buffer = DefineBufferArg();
  loom_value_id_t descriptor = DefineArg(KernelTensorLdsDescriptorType());
  loom_value_id_t layout = BuildDenseLayout();
  loom_value_id_t source = BuildBufferView(
      source_buffer, 0, ViewType2D(LOOM_SCALAR_TYPE_F32, 2, 2, layout));
  loom_value_id_t lds = BuildBufferView(
      lds_buffer, 64, ViewType2D(LOOM_SCALAR_TYPE_F32, 2, 2, layout));
  loom_value_id_t dest = BuildBufferView(
      dest_buffer, 128, ViewType2D(LOOM_SCALAR_TYPE_F32, 2, 2, layout));
  loom_op_t* load_op = nullptr;
  IREE_ASSERT_OK(loom_kernel_async_tensor_load_to_lds_build(
      &builder_, source, lds, descriptor, LOOM_CACHE_SCOPE_CU,
      LOOM_CACHE_TEMPORAL_REGULAR, KernelAsyncTokenType(),
      LOOM_LOCATION_UNKNOWN, &load_op));
  loom_op_t* store_op = nullptr;
  IREE_ASSERT_OK(loom_kernel_async_tensor_store_from_lds_build(
      &builder_, lds, dest, descriptor, LOOM_CACHE_SCOPE_DEVICE,
      LOOM_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK, KernelAsyncTokenType(),
      LOOM_LOCATION_UNKNOWN, &store_op));

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  ASSERT_TRUE(Describe(&analysis, load_op, &request, &diagnostic));
  EXPECT_EQ(request.kind, LOOM_MOVEMENT_KIND_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_TENSOR_TILE);
  EXPECT_EQ(request.descriptor_value_id, descriptor);
  EXPECT_TRUE(
      iree_any_bit_set(request.flags, LOOM_MOVEMENT_REQUEST_HAS_DIRECTION));
  EXPECT_EQ(request.direction, LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP);
  EXPECT_EQ(request.transferred_byte_count, 16);

  ASSERT_TRUE(Describe(&analysis, store_op, &request, &diagnostic));
  EXPECT_EQ(request.kind,
            LOOM_MOVEMENT_KIND_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS);
  EXPECT_EQ(request.layout_kind, LOOM_MOVEMENT_LAYOUT_TENSOR_TILE);
  EXPECT_EQ(request.descriptor_value_id, descriptor);
  EXPECT_TRUE(
      iree_any_bit_set(request.flags, LOOM_MOVEMENT_REQUEST_HAS_DIRECTION));
  EXPECT_EQ(request.direction, LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL);
  EXPECT_EQ(request.transferred_byte_count, 16);
}

TEST_F(MovementTest, ReportsUnsupportedOpsWithoutFailingAnalysis) {
  loom_op_t* op = BuildOffsetConstant(42);

  loom_movement_analysis_t analysis = {};
  InitializeAnalysis(&analysis);
  loom_movement_request_t request = {};
  loom_movement_diagnostic_t diagnostic = {};
  EXPECT_FALSE(Describe(&analysis, op, &request, &diagnostic));
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_bits,
                               LOOM_MOVEMENT_REJECTION_UNSUPPORTED_OP));
}

}  // namespace
}  // namespace loom
