// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/type_propagation.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/value_facts.h"

namespace loom {
namespace {

class TypePropagationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VIEW, loom_view_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCF, loom_scf_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    CreateFunction();
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVTableFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                             out_count);

  void RegisterDialect(uint8_t dialect_id, DialectVTableFn dialect_vtables) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  void CreateFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t symbol = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, symbol, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    function_ = loom_func_like_cast(module_, func_op);
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &builder_);
  }

  iree_status_t Propagate(loom_op_t* op, bool* out_changed) {
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_rewriter_t rewriter;
    bool rewriter_initialized = false;
    iree_status_t status =
        loom_rewriter_initialize(&rewriter, module_, &pass_arena);
    if (iree_status_is_ok(status)) {
      rewriter_initialized = true;
    }
    loom_type_propagator_t* propagator = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_type_propagator_allocate(module_, &pass_arena, &propagator);
    }
    if (iree_status_is_ok(status)) {
      status = loom_type_propagator_prepare_function(propagator, function_);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_type_propagator_apply_op(propagator, &rewriter, op, out_changed);
    }
    loom_type_propagator_deinitialize(propagator);
    if (rewriter_initialized) loom_rewriter_deinitialize(&rewriter);
    iree_arena_deinitialize(&pass_arena);
    return status;
  }

  iree_status_t MayApply(loom_op_t* op, loom_value_fact_table_t* facts,
                         bool* out_may_apply) {
    *out_may_apply = false;
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_rewriter_t rewriter;
    bool rewriter_initialized = false;
    iree_status_t status =
        loom_rewriter_initialize(&rewriter, module_, &pass_arena);
    if (iree_status_is_ok(status)) {
      rewriter_initialized = true;
    }
    if (iree_status_is_ok(status) && facts) {
      status = loom_rewriter_enable_analysis(&rewriter, function_, facts);
    }
    loom_type_propagator_t* propagator = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_type_propagator_allocate(module_, &pass_arena, &propagator);
    }
    if (iree_status_is_ok(status)) {
      status = loom_type_propagator_prepare_function(propagator, function_);
    }
    if (iree_status_is_ok(status)) {
      const loom_op_vtable_t* vtable = loom_op_vtable(module_, op);
      *out_may_apply =
          loom_type_propagator_may_apply_op(propagator, &rewriter, op, vtable);
    }
    loom_type_propagator_deinitialize(propagator);
    if (rewriter_initialized) loom_rewriter_deinitialize(&rewriter);
    iree_arena_deinitialize(&pass_arena);
    return status;
  }

  iree_status_t BuildConstant(loom_attribute_t value, loom_type_t type,
                              loom_op_t** out_op) {
    return loom_test_constant_build(&builder_, value, type,
                                    LOOM_LOCATION_UNKNOWN, out_op);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_;
  loom_builder_t builder_;
};

TEST_F(TypePropagationTest, MayApplySkipsScalarOnlyConstraints) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* lhs_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(1), i32_type, &lhs_op));
  loom_op_t* rhs_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(2), i32_type, &rhs_op));

  loom_op_t* add_op = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, loom_test_constant_result(lhs_op),
                           loom_test_constant_result(rhs_op), i32_type,
                           LOOM_LOCATION_UNKNOWN, &add_op));

  bool may_apply = true;
  IREE_EXPECT_OK(MayApply(add_op, NULL, &may_apply));

  EXPECT_FALSE(may_apply);
}

TEST_F(TypePropagationTest, MayApplySkipsRefinableOpWithConcreteTypes) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(1), i32_type, &source_op));

  loom_named_attr_slice_t empty_attrs = {0};
  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, loom_test_constant_result(source_op), empty_attrs, i32_type,
      LOOM_LOCATION_UNKNOWN, &attrs_op));

  bool may_apply = true;
  IREE_EXPECT_OK(MayApply(attrs_op, NULL, &may_apply));

  EXPECT_FALSE(may_apply);
}

TEST_F(TypePropagationTest, MayApplyAcceptsRefinableOpWithDynamicTypes) {
  loom_type_t static_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(16), 0);
  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(1), 0);
  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(1), static_vector, &source_op));

  loom_named_attr_slice_t empty_attrs = {0};
  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, loom_test_constant_result(source_op), empty_attrs,
      dynamic_vector, LOOM_LOCATION_UNKNOWN, &attrs_op));

  bool may_apply = false;
  IREE_EXPECT_OK(MayApply(attrs_op, NULL, &may_apply));

  EXPECT_TRUE(may_apply);
}

TEST_F(TypePropagationTest, MayApplyAcceptsDynamicTypesWhenFactsAreEnabled) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(16), index_type, &dim_op));

  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
      loom_dim_pack_dynamic(loom_test_constant_result(dim_op)), 0);
  loom_op_t* vector_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), dynamic_vector, &vector_op));

  loom_pass_value_fact_owner_t value_fact_owner = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_fact_owner);
  loom_value_fact_table_t* facts = NULL;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_prepare(
      &value_fact_owner, module_,
      loom_pass_value_fact_scope_function(function_), &facts));

  bool may_apply = false;
  IREE_EXPECT_OK(MayApply(vector_op, facts, &may_apply));

  EXPECT_TRUE(may_apply);

  loom_pass_value_fact_owner_deinitialize(&value_fact_owner);
}

TEST_F(TypePropagationTest, SameTypeNarrowsResult) {
  loom_type_t static_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(16), 0);
  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(1), 0);

  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), static_vector, &source_op));
  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(
      loom_test_attrs_build(&builder_, loom_test_constant_result(source_op),
                            (loom_named_attr_slice_t){0}, dynamic_vector,
                            LOOM_LOCATION_UNKNOWN, &attrs_op));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(attrs_op, &changed));

  EXPECT_TRUE(changed);
  loom_type_t result_type =
      loom_module_value_type(module_, loom_test_attrs_result(attrs_op));
  EXPECT_TRUE(loom_type_equal(result_type, static_vector));
}

TEST_F(TypePropagationTest, SameTypeClosureVisitsOtherUsers) {
  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(1), 0);
  loom_type_t static_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(16), 0);

  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), dynamic_vector, &source_op));
  loom_value_id_t source = loom_test_constant_result(source_op);

  loom_op_t* first_user = NULL;
  IREE_ASSERT_OK(
      loom_test_attrs_build(&builder_, source, (loom_named_attr_slice_t){0},
                            static_vector, LOOM_LOCATION_UNKNOWN, &first_user));
  loom_op_t* second_user = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, source, (loom_named_attr_slice_t){0}, dynamic_vector,
      LOOM_LOCATION_UNKNOWN, &second_user));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(first_user, &changed));

  EXPECT_TRUE(changed);
  EXPECT_TRUE(
      loom_type_equal(loom_module_value_type(module_, source), static_vector));
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, loom_test_attrs_result(second_user)),
      static_vector));
}

TEST_F(TypePropagationTest, SameTypeConflictRejectsTransaction) {
  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(1), 0);
  loom_type_t static_vector_16 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(16), 0);
  loom_type_t static_vector_32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(32), 0);

  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), dynamic_vector, &source_op));
  loom_value_id_t source = loom_test_constant_result(source_op);

  loom_op_t* first_user = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, source, (loom_named_attr_slice_t){0}, static_vector_16,
      LOOM_LOCATION_UNKNOWN, &first_user));
  loom_op_t* second_user = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, source, (loom_named_attr_slice_t){0}, static_vector_32,
      LOOM_LOCATION_UNKNOWN, &second_user));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(first_user, &changed));

  EXPECT_FALSE(changed);
  EXPECT_TRUE(
      loom_type_equal(loom_module_value_type(module_, source), dynamic_vector));
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(module_, loom_test_attrs_result(second_user)),
      static_vector_32));
}

TEST_F(TypePropagationTest, SameShapeNarrowsVariadicInputs) {
  loom_type_t static_tile = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t dynamic_tile = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(1), 0);

  loom_op_t* first_input_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), static_tile, &first_input_op));
  loom_op_t* second_input_op = NULL;
  IREE_ASSERT_OK(
      BuildConstant(loom_attr_i64(0), dynamic_tile, &second_input_op));
  loom_value_id_t inputs[] = {
      loom_test_constant_result(first_input_op),
      loom_test_constant_result(second_input_op),
  };
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, IREE_ARRAYSIZE(inputs),
                                     static_tile, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(map_op, &changed));

  EXPECT_TRUE(changed);
  EXPECT_TRUE(
      loom_type_equal(loom_module_value_type(module_, inputs[1]), static_tile));
}

TEST_F(TypePropagationTest, SameEncodingNarrowsSsaEncodingAttachment) {
  loom_op_t* layout_op = NULL;
  IREE_ASSERT_OK(loom_encoding_layout_dense_build(
      &builder_,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
      LOOM_LOCATION_UNKNOWN, &layout_op));
  loom_value_id_t layout = loom_encoding_layout_dense_result(layout_op);

  loom_type_t source_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  source_type.encoding_id = (uint16_t)layout;
  source_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 7);
  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), source_type, &source_op));
  loom_value_id_t source = loom_test_constant_result(source_op);

  loom_op_t* subview_op = NULL;
  IREE_ASSERT_OK(loom_view_subview_build(&builder_, source, NULL, 0, NULL, 0,
                                         result_type, LOOM_LOCATION_UNKNOWN,
                                         &subview_op));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(subview_op, &changed));

  EXPECT_TRUE(changed);
  loom_type_t refined_source_type = loom_module_value_type(module_, source);
  EXPECT_TRUE(loom_type_has_static_encoding(refined_source_type));
  EXPECT_EQ(refined_source_type.encoding_id, 7);
}

TEST_F(TypePropagationTest, ValueFactsNarrowDynamicDimensions) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(16), index_type, &dim_op));
  loom_value_id_t dim = loom_test_constant_result(dim_op);

  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(dim), 0);
  loom_op_t* vector_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), dynamic_vector, &vector_op));
  loom_value_id_t vector = loom_test_constant_result(vector_op);

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_value_fact_owner_t value_fact_owner = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &value_fact_owner);
  loom_value_fact_table_t* facts = NULL;
  IREE_ASSERT_OK(loom_pass_value_fact_owner_prepare(
      &value_fact_owner, module_,
      loom_pass_value_fact_scope_function(function_), &facts));
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &pass_arena));
  IREE_ASSERT_OK(loom_rewriter_enable_analysis(&rewriter, function_, facts));
  loom_type_propagator_t* propagator = NULL;
  IREE_ASSERT_OK(
      loom_type_propagator_allocate(module_, &pass_arena, &propagator));
  IREE_ASSERT_OK(loom_type_propagator_prepare_function(propagator, function_));

  bool changed = false;
  IREE_EXPECT_OK(loom_type_propagator_apply_op(propagator, &rewriter, vector_op,
                                               &changed));

  EXPECT_TRUE(changed);
  loom_type_t refined_vector = loom_module_value_type(module_, vector);
  EXPECT_FALSE(loom_type_dim_is_dynamic_at(refined_vector, 0));
  EXPECT_EQ(loom_type_dim_static_size_at(refined_vector, 0), 16);

  loom_type_propagator_deinitialize(propagator);
  loom_rewriter_deinitialize(&rewriter);
  loom_pass_value_fact_owner_deinitialize(&value_fact_owner);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(TypePropagationTest, TypeUseUsersAreQueuedOnCommit) {
  loom_op_t* layout_op = NULL;
  IREE_ASSERT_OK(loom_encoding_layout_dense_build(
      &builder_, loom_type_encoding(), LOOM_LOCATION_UNKNOWN, &layout_op));
  loom_value_id_t layout = loom_encoding_layout_dense_result(layout_op);

  loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  view_type.encoding_id = (uint16_t)layout;
  view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  loom_op_t* view_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), view_type, &view_op));
  loom_value_id_t view = loom_test_constant_result(view_op);

  loom_op_t* use_op = NULL;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder_, &view, 1, LOOM_LOCATION_UNKNOWN, &use_op));

  loom_op_t* assume_op = NULL;
  IREE_ASSERT_OK(loom_encoding_layout_assume_dense_build(
      &builder_, layout,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
      LOOM_LOCATION_UNKNOWN, &assume_op));

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &pass_arena));
  loom_type_propagator_t* propagator = NULL;
  IREE_ASSERT_OK(
      loom_type_propagator_allocate(module_, &pass_arena, &propagator));
  IREE_ASSERT_OK(loom_type_propagator_prepare_function(propagator, function_));

  bool changed = false;
  IREE_EXPECT_OK(loom_type_propagator_apply_op(propagator, &rewriter, assume_op,
                                               &changed));

  EXPECT_TRUE(changed);
  EXPECT_EQ(loom_type_encoding_role(loom_module_value_type(module_, layout)),
            LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);

  bool saw_view_provider = false;
  bool saw_view_user = false;
  loom_op_t* queued_op = NULL;
  while ((queued_op = loom_rewriter_pop(&rewriter)) != NULL) {
    saw_view_provider |= queued_op == view_op;
    saw_view_user |= queued_op == use_op;
  }
  EXPECT_TRUE(saw_view_provider);
  EXPECT_TRUE(saw_view_user);

  loom_type_propagator_deinitialize(propagator);
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(TypePropagationTest, VectorTransposeNarrowsPermutedResultShape) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* dim0_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), index_type, &dim0_op));
  loom_op_t* dim1_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), index_type, &dim1_op));

  loom_type_t source_type =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);
  loom_type_t result_type = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
      loom_dim_pack_dynamic(loom_test_constant_result(dim0_op)),
      loom_dim_pack_dynamic(loom_test_constant_result(dim1_op)), 0);

  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), source_type, &source_op));
  int64_t permutation[] = {1, 0};
  loom_op_t* transpose_op = NULL;
  IREE_ASSERT_OK(loom_vector_transpose_build(
      &builder_, permutation, IREE_ARRAYSIZE(permutation),
      loom_test_constant_result(source_op), result_type, LOOM_LOCATION_UNKNOWN,
      &transpose_op));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(transpose_op, &changed));

  EXPECT_TRUE(changed);
  loom_type_t refined_type = loom_module_value_type(
      module_, loom_vector_transpose_result(transpose_op));
  EXPECT_FALSE(loom_type_dim_is_dynamic_at(refined_type, 0));
  EXPECT_FALSE(loom_type_dim_is_dynamic_at(refined_type, 1));
  EXPECT_EQ(loom_type_dim_static_size_at(refined_type, 0), 8);
  EXPECT_EQ(loom_type_dim_static_size_at(refined_type, 1), 4);
}

TEST_F(TypePropagationTest, ViewRefineNarrowsSourceStaticShapeAndEncoding) {
  loom_op_t* layout_op = NULL;
  IREE_ASSERT_OK(loom_encoding_layout_dense_build(
      &builder_, loom_type_encoding(), LOOM_LOCATION_UNKNOWN, &layout_op));
  loom_value_id_t layout = loom_encoding_layout_dense_result(layout_op);

  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), index_type, &dim_op));

  loom_type_t source_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
      loom_dim_pack_dynamic(loom_test_constant_result(dim_op)), 0);
  source_type.encoding_id = (uint16_t)layout;
  source_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 7);

  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), source_type, &source_op));
  loom_op_t* refine_op = NULL;
  IREE_ASSERT_OK(
      loom_view_refine_build(&builder_, loom_test_constant_result(source_op),
                             result_type, LOOM_LOCATION_UNKNOWN, &refine_op));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(refine_op, &changed));

  EXPECT_TRUE(changed);
  loom_type_t refined_source =
      loom_module_value_type(module_, loom_test_constant_result(source_op));
  EXPECT_FALSE(loom_type_dim_is_dynamic_at(refined_source, 0));
  EXPECT_EQ(loom_type_dim_static_size_at(refined_source, 0), 16);
  EXPECT_TRUE(loom_type_has_static_encoding(refined_source));
  EXPECT_EQ(refined_source.encoding_id, 7);
}

TEST_F(TypePropagationTest, RegionBranchNarrowsResultFromYieldedValues) {
  loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_op_t* condition_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_bool(true), i1_type, &condition_op));

  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), index_type, &dim_op));

  loom_type_t static_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
      loom_dim_pack_dynamic(loom_test_constant_result(dim_op)), 0);

  loom_op_t* then_value_op = NULL;
  IREE_ASSERT_OK(
      BuildConstant(loom_attr_i64(0), static_vector, &then_value_op));
  loom_op_t* else_value_op = NULL;
  IREE_ASSERT_OK(
      BuildConstant(loom_attr_i64(0), static_vector, &else_value_op));

  loom_op_t* if_op = NULL;
  IREE_ASSERT_OK(loom_scf_if_build(
      &builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      loom_test_constant_result(condition_op), &dynamic_vector, 1, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &if_op));

  loom_builder_t then_builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_scf_if_then_region(if_op)), &then_builder);
  loom_value_id_t then_value = loom_test_constant_result(then_value_op);
  loom_op_t* then_yield = NULL;
  IREE_ASSERT_OK(loom_scf_yield_build(&then_builder, &then_value, 1,
                                      LOOM_LOCATION_UNKNOWN, &then_yield));

  loom_builder_t else_builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_scf_if_else_region(if_op)), &else_builder);
  loom_value_id_t else_value = loom_test_constant_result(else_value_op);
  loom_op_t* else_yield = NULL;
  IREE_ASSERT_OK(loom_scf_yield_build(&else_builder, &else_value, 1,
                                      LOOM_LOCATION_UNKNOWN, &else_yield));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(if_op, &changed));

  EXPECT_TRUE(changed);
  loom_type_t refined_result = loom_module_value_type(
      module_, loom_value_slice_get(loom_scf_if_results(if_op), 0));
  EXPECT_FALSE(loom_type_dim_is_dynamic_at(refined_result, 0));
  EXPECT_EQ(loom_type_dim_static_size_at(refined_result, 0), 16);
}

TEST_F(TypePropagationTest, RegionBranchNarrowsSwitchResultFromYieldedValues) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* selector_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), index_type, &selector_op));
  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), index_type, &dim_op));

  loom_type_t static_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
      loom_dim_pack_dynamic(loom_test_constant_result(dim_op)), 0);

  loom_op_t* case_value_op = NULL;
  IREE_ASSERT_OK(
      BuildConstant(loom_attr_i64(0), static_vector, &case_value_op));
  loom_op_t* default_value_op = NULL;
  IREE_ASSERT_OK(
      BuildConstant(loom_attr_i64(0), static_vector, &default_value_op));

  int64_t case_keys[] = {0};
  loom_op_t* switch_op = NULL;
  IREE_ASSERT_OK(loom_scf_switch_build(
      &builder_, loom_test_constant_result(selector_op), &dynamic_vector, 1,
      NULL, 0, case_keys, IREE_ARRAYSIZE(case_keys), LOOM_LOCATION_UNKNOWN,
      &switch_op));

  loom_builder_t default_builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_scf_switch_default_region(switch_op)),
      &default_builder);
  loom_value_id_t default_value = loom_test_constant_result(default_value_op);
  loom_op_t* default_yield = NULL;
  IREE_ASSERT_OK(loom_scf_yield_build(&default_builder, &default_value, 1,
                                      LOOM_LOCATION_UNKNOWN, &default_yield));

  loom_region_slice_t case_regions = loom_scf_switch_case_regions(switch_op);
  ASSERT_EQ(case_regions.count, 1);
  loom_builder_t case_builder;
  loom_builder_initialize(module_, &module_->arena,
                          loom_region_entry_block(case_regions.regions[0]),
                          &case_builder);
  loom_value_id_t case_value = loom_test_constant_result(case_value_op);
  loom_op_t* case_yield = NULL;
  IREE_ASSERT_OK(loom_scf_yield_build(&case_builder, &case_value, 1,
                                      LOOM_LOCATION_UNKNOWN, &case_yield));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(switch_op, &changed));

  EXPECT_TRUE(changed);
  loom_type_t refined_result = loom_module_value_type(
      module_, loom_value_slice_get(loom_scf_switch_results(switch_op), 0));
  EXPECT_FALSE(loom_type_dim_is_dynamic_at(refined_result, 0));
  EXPECT_EQ(loom_type_dim_static_size_at(refined_result, 0), 16);
}

TEST_F(TypePropagationTest, RegionBranchRejectsConflictingYieldedShapes) {
  loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_op_t* condition_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_bool(true), i1_type, &condition_op));

  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* dim_op = NULL;
  IREE_ASSERT_OK(BuildConstant(loom_attr_i64(0), index_type, &dim_op));

  loom_type_t static_vector_16 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_type_t static_vector_32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(32), 0);
  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
      loom_dim_pack_dynamic(loom_test_constant_result(dim_op)), 0);

  loom_op_t* then_value_op = NULL;
  IREE_ASSERT_OK(
      BuildConstant(loom_attr_i64(0), static_vector_16, &then_value_op));
  loom_op_t* else_value_op = NULL;
  IREE_ASSERT_OK(
      BuildConstant(loom_attr_i64(0), static_vector_32, &else_value_op));

  loom_op_t* if_op = NULL;
  IREE_ASSERT_OK(loom_scf_if_build(
      &builder_, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
      loom_test_constant_result(condition_op), &dynamic_vector, 1, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &if_op));

  loom_builder_t then_builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_scf_if_then_region(if_op)), &then_builder);
  loom_value_id_t then_value = loom_test_constant_result(then_value_op);
  loom_op_t* then_yield = NULL;
  IREE_ASSERT_OK(loom_scf_yield_build(&then_builder, &then_value, 1,
                                      LOOM_LOCATION_UNKNOWN, &then_yield));

  loom_builder_t else_builder;
  loom_builder_initialize(
      module_, &module_->arena,
      loom_region_entry_block(loom_scf_if_else_region(if_op)), &else_builder);
  loom_value_id_t else_value = loom_test_constant_result(else_value_op);
  loom_op_t* else_yield = NULL;
  IREE_ASSERT_OK(loom_scf_yield_build(&else_builder, &else_value, 1,
                                      LOOM_LOCATION_UNKNOWN, &else_yield));

  bool changed = false;
  IREE_EXPECT_OK(Propagate(if_op, &changed));

  EXPECT_FALSE(changed);
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(
          module_, loom_value_slice_get(loom_scf_if_results(if_op), 0)),
      dynamic_vector));
}

}  // namespace
}  // namespace loom
