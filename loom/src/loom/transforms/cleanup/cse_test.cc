// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/cse.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/rewrite/rewriter.h"

namespace loom {
namespace {

static iree_status_t InitializePassStatistics(loom_pass_t* pass,
                                              iree_arena_allocator_t* arena) {
  const loom_pass_statistic_layout_t* layout = pass->info->statistic_layout;
  if (!layout) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, layout->storage_size,
                                           (void**)&pass->statistic_storage));
  memset(pass->statistic_storage, 0, layout->storage_size);
  return iree_ok_status();
}

class CSETest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    // Build a test.func op on the module body block to provide a real
    // func-like interface for passes under test.
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    func_like_ = loom_func_like_cast(module_, func_op);
    body_ = loom_func_like_body(func_like_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t run_cse() {
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    loom_pass_t pass;
    memset(&pass, 0, sizeof(pass));
    pass.info = loom_cse_pass_info();
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    iree_status_t status = InitializePassStatistics(&pass, &pass_arena);
    if (iree_status_is_ok(status)) {
      status = loom_cse_run(&pass, module_, func_like_);
    }
    iree_arena_deinitialize(&pass_arena);
    return status;
  }

  int count_live_ops() {
    int count = 0;
    loom_block_t* block = NULL;
    loom_region_for_each_block(body_, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) { ++count; }
    }
    return count;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t func_like_;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(CSETest, EliminatesDuplicate) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // Build two identical constants.
  loom_op_t* const1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const1));
  loom_value_id_t a = loom_test_constant_result(const1);

  loom_op_t* const2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const2));
  loom_value_id_t b = loom_test_constant_result(const2);

  // Use both so they survive DCE.
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, b, i32, LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 2);  // One constant eliminated.

  // Both operands of addi should now point to the same value.
  EXPECT_EQ(loom_op_operands(addi)[0], loom_op_operands(addi)[1]);
}

TEST_F(CSETest, DistinguishesDifferentOps) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t a = loom_test_constant_result(const_op);

  // neg %a and addi %a, %a are different ops — no CSE.
  loom_op_t* neg = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg));

  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, a, i32, LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 3);  // No change.
}

TEST_F(CSETest, DistinguishesDifferentOperands) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  loom_value_id_t a = loom_test_constant_result(c1);

  loom_op_t* c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &c2));
  loom_value_id_t b = loom_test_constant_result(c2);

  // neg %a and neg %b — same op kind but different operands.
  loom_op_t* neg_a = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, a, i32, LOOM_LOCATION_UNKNOWN, &neg_a));
  loom_op_t* neg_b = NULL;
  IREE_ASSERT_OK(
      loom_test_neg_build(&builder_, b, i32, LOOM_LOCATION_UNKNOWN, &neg_b));

  EXPECT_EQ(count_live_ops(), 4);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 4);  // No change.
}

TEST_F(CSETest, DistinguishesSegmentedOperandBoundaries) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_block_t* entry_block = loom_region_entry_block(body_);
  loom_value_id_t root = LOOM_VALUE_ID_INVALID;
  loom_value_id_t first = LOOM_VALUE_ID_INVALID;
  loom_value_id_t second = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_block_arg(&builder_, entry_block, i32, &root));
  IREE_ASSERT_OK(
      loom_builder_define_block_arg(&builder_, entry_block, i32, &first));
  IREE_ASSERT_OK(
      loom_builder_define_block_arg(&builder_, entry_block, i32, &second));

  loom_value_id_t one_lhs[] = {first};
  loom_value_id_t one_rhs[] = {second};
  loom_op_t* lhs_rhs = NULL;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &builder_, 0, root, LOOM_VALUE_ID_INVALID, one_lhs,
      IREE_ARRAYSIZE(one_lhs), one_rhs, IREE_ARRAYSIZE(one_rhs), i32,
      LOOM_LOCATION_UNKNOWN, &lhs_rhs));

  loom_value_id_t two_lhs[] = {first, second};
  loom_op_t* lhs_only = NULL;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &builder_, 0, root, LOOM_VALUE_ID_INVALID, two_lhs,
      IREE_ARRAYSIZE(two_lhs), NULL, 0, i32, LOOM_LOCATION_UNKNOWN, &lhs_only));

  ASSERT_EQ(lhs_rhs->operand_count, lhs_only->operand_count);
  for (uint16_t i = 0; i < lhs_rhs->operand_count; ++i) {
    ASSERT_EQ(loom_op_const_operands(lhs_rhs)[i],
              loom_op_const_operands(lhs_only)[i]);
  }

  loom_value_id_t observed[] = {
      loom_test_segmented_result(lhs_rhs),
      loom_test_segmented_result(lhs_only),
  };
  loom_op_t* use_op = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, observed,
                                     IREE_ARRAYSIZE(observed),
                                     LOOM_LOCATION_UNKNOWN, &use_op));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 3);
  EXPECT_NE(loom_op_const_operands(use_op)[0],
            loom_op_const_operands(use_op)[1]);
}

TEST_F(CSETest, DistinguishesDifferentAttributes) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // Two constants with different values — different attributes.
  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  loom_op_t* c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(99), i32,
                                          LOOM_LOCATION_UNKNOWN, &c2));

  // Use both.
  loom_value_id_t a = loom_test_constant_result(c1);
  loom_value_id_t b = loom_test_constant_result(c2);
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, b, i32, LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 3);  // No change.
}

TEST_F(CSETest, CanonicalAttrDictOrderDoesNotBlockCSE) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), f32, &input));

  loom_string_id_t axis_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t label_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t foo_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("axis"), &axis_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("label"), &label_id));
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("foo"), &foo_id));

  loom_named_attr_t label_first_entries[2] = {
      {/*.name_id=*/label_id, /*.reserved=*/{},
       /*.value=*/loom_attr_string(foo_id)},
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
  };
  loom_op_t* attrs0 = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, input,
      loom_make_named_attr_slice(label_first_entries,
                                 IREE_ARRAYSIZE(label_first_entries)),
      f32, LOOM_LOCATION_UNKNOWN, &attrs0));

  loom_named_attr_t axis_first_entries[2] = {
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
      {/*.name_id=*/label_id, /*.reserved=*/{},
       /*.value=*/loom_attr_string(foo_id)},
  };
  loom_op_t* attrs1 = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(
      &builder_, input,
      loom_make_named_attr_slice(axis_first_entries,
                                 IREE_ARRAYSIZE(axis_first_entries)),
      f32, LOOM_LOCATION_UNKNOWN, &attrs1));

  loom_value_id_t values[] = {
      loom_test_attrs_result(attrs0),
      loom_test_attrs_result(attrs1),
  };
  loom_op_t* reduce = NULL;
  IREE_ASSERT_OK(loom_test_reduce_build(&builder_, values, 2, f32,
                                        LOOM_LOCATION_UNKNOWN, &reduce));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 2);
  EXPECT_EQ(loom_op_const_operands(reduce)[0],
            loom_op_const_operands(reduce)[1]);
}

TEST_F(CSETest, RewriterReplaceAttrDictBuildsFreshCanonicalDict) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), f32, &input));

  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &attrs_op));

  loom_string_id_t zeta_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t alpha_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("zeta"), &zeta_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("alpha"), &alpha_id));

  loom_named_attr_update_t updates[2] = {
      loom_named_attr_replace(zeta_id, loom_attr_i64(2)),
      loom_named_attr_replace(alpha_id, loom_attr_i64(1)),
  };

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &pass_arena));
  IREE_ASSERT_OK(loom_rewriter_replace_attr_dict(
      &rewriter, attrs_op, loom_test_attrs_dict_ATTR_INDEX,
      loom_make_named_attr_update_slice(updates, IREE_ARRAYSIZE(updates))));
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&pass_arena);

  loom_named_attr_slice_t dict = loom_test_attrs_dict(attrs_op);
  ASSERT_EQ(dict.count, 2u);
  ASSERT_NE(dict.entries, nullptr);
  EXPECT_EQ(dict.entries[0].name_id, alpha_id);
  EXPECT_EQ(loom_attr_as_i64(dict.entries[0].value), 1);
  EXPECT_EQ(dict.entries[1].name_id, zeta_id);
  EXPECT_EQ(loom_attr_as_i64(dict.entries[1].value), 2);
}

TEST_F(CSETest, RewriterReplaceAttrDictRecordsTypeValueRefs) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_op_t* old_dim_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), index,
                                          LOOM_LOCATION_UNKNOWN, &old_dim_op));
  loom_value_id_t old_dim = loom_test_constant_result(old_dim_op);

  loom_op_t* new_dim_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), index,
                                          LOOM_LOCATION_UNKNOWN, &new_dim_op));
  loom_value_id_t new_dim = loom_test_constant_result(new_dim_op);

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), f32, &input));

  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &attrs_op));

  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(old_dim), 0);
  loom_type_id_t vector_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module_, vector_type, &vector_type_id));

  loom_string_id_t shape_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("shape"), &shape_id));
  loom_named_attr_update_t updates[1] = {
      loom_named_attr_replace(shape_id, loom_attr_type(vector_type_id)),
  };

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &pass_arena));
  IREE_ASSERT_OK(loom_rewriter_replace_attr_dict(
      &rewriter, attrs_op, loom_test_attrs_dict_ATTR_INDEX,
      loom_make_named_attr_update_slice(updates, IREE_ARRAYSIZE(updates))));
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&pass_arena);

  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(module_, old_dim)));
  EXPECT_FALSE(
      loom_value_has_attribute_uses(loom_module_value(module_, new_dim)));

  IREE_ASSERT_OK(loom_value_replace_all_uses_with(module_, old_dim, new_dim));

  loom_named_attr_slice_t dict = loom_test_attrs_dict(attrs_op);
  ASSERT_EQ(dict.count, 1u);
  ASSERT_EQ(dict.entries[0].value.kind, LOOM_ATTR_TYPE);
  loom_type_t replaced_type =
      module_->types.entries[dict.entries[0].value.type_id];
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(replaced_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(replaced_type, 0), new_dim);
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(module_, new_dim)));
}

TEST_F(CSETest, RewriterSetAttrRejectsMalformedDictAttr) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), f32, &input));

  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &attrs_op));

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &pass_arena));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_rewriter_set_attr(
          &rewriter, attrs_op, loom_test_attrs_dict_ATTR_INDEX,
          loom_make_canonical_attr_dict(/*entries=*/NULL, /*count=*/1)));
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(CSETest, RewriterSetAttrRejectsNonCanonicalDictAttrOrder) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), f32, &input));

  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &attrs_op));

  loom_string_id_t axis_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t label_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t foo_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("axis"), &axis_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("label"), &label_id));
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("foo"), &foo_id));

  loom_named_attr_t unsorted_entries[2] = {
      {/*.name_id=*/label_id, /*.reserved=*/{},
       /*.value=*/loom_attr_string(foo_id)},
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
  };

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &pass_arena));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_rewriter_set_attr(
          &rewriter, attrs_op, loom_test_attrs_dict_ATTR_INDEX,
          loom_make_canonical_attr_dict(unsorted_entries,
                                        IREE_ARRAYSIZE(unsorted_entries))));
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(CSETest, RewriterSetAttrRejectsDuplicateDictAttrKeys) {
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), f32, &input));

  loom_op_t* attrs_op = NULL;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_make_named_attr_slice(NULL, 0), f32,
                                       LOOM_LOCATION_UNKNOWN, &attrs_op));

  loom_string_id_t axis_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("axis"), &axis_id));

  loom_named_attr_t duplicate_entries[2] = {
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(0)},
      {/*.name_id=*/axis_id, /*.reserved=*/{}, /*.value=*/loom_attr_i64(1)},
  };

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_rewriter_t rewriter;
  IREE_ASSERT_OK(loom_rewriter_initialize(&rewriter, module_, &pass_arena));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_rewriter_set_attr(
          &rewriter, attrs_op, loom_test_attrs_dict_ATTR_INDEX,
          loom_make_canonical_attr_dict(duplicate_entries,
                                        IREE_ARRAYSIZE(duplicate_entries))));
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(CSETest, EliminatesIdenticalBinaryOps) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  loom_value_id_t a = loom_test_constant_result(c1);
  loom_op_t* c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &c2));
  loom_value_id_t b = loom_test_constant_result(c2);

  // Two identical addi ops: addi(%a, %b) twice.
  loom_op_t* add1 = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, b, i32, LOOM_LOCATION_UNKNOWN, &add1));
  loom_op_t* add2 = NULL;
  IREE_ASSERT_OK(
      loom_test_addi_build(&builder_, a, b, i32, LOOM_LOCATION_UNKNOWN, &add2));

  // Use both addi results to keep them alive.
  loom_value_id_t r1 = loom_test_addi_result(add1);
  loom_value_id_t r2 = loom_test_addi_result(add2);
  loom_op_t* final_add = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                      LOOM_LOCATION_UNKNOWN, &final_add));

  // 2 constants + 2 addi + 1 final = 5 ops.
  EXPECT_EQ(count_live_ops(), 5);
  IREE_ASSERT_OK(run_cse());
  // One addi eliminated: 2 constants + 1 addi + 1 final = 4 ops.
  EXPECT_EQ(count_live_ops(), 4);
}

TEST_F(CSETest, ReadOnlyOpsCSEWithoutInterveningWrite) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // Two identical reads from the same pool, no writes between them.
  loom_op_t* read1 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool_id, i32,
                                               LOOM_LOCATION_UNKNOWN, &read1));
  loom_value_id_t r1 = loom_test_read_resource_result(read1);

  loom_op_t* read2 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool_id, i32,
                                               LOOM_LOCATION_UNKNOWN, &read2));
  loom_value_id_t r2 = loom_test_read_resource_result(read2);

  // Use both results to keep them alive.
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 2);  // Second read eliminated.
}

TEST_F(CSETest, ReadOnlyOpsBlockedByInterveningWrite) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // First read.
  loom_op_t* read1 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool_id, i32,
                                               LOOM_LOCATION_UNKNOWN, &read1));
  loom_value_id_t r1 = loom_test_read_resource_result(read1);

  // Write between the two reads — invalidates the CSE table entry.
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(99), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t data = loom_test_constant_result(const_op);
  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, data, LOOM_LOCATION_UNKNOWN, &write_op));

  // Second identical read — should NOT be CSE'd.
  loom_op_t* read2 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool_id, i32,
                                               LOOM_LOCATION_UNKNOWN, &read2));
  loom_value_id_t r2 = loom_test_read_resource_result(read2);

  // Use both read results.
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 5);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 5);  // No change — write blocks CSE.
}

TEST_F(CSETest, PureOpsSurviveWriteBarrier) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // First constant (pure).
  loom_op_t* const1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const1));
  loom_value_id_t c1 = loom_test_constant_result(const1);

  // Write to the pool — triggers write barrier, evicts read-only entries.
  loom_op_t* write_data = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &write_data));
  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, loom_test_constant_result(write_data),
      LOOM_LOCATION_UNKNOWN, &write_op));

  // Second identical constant — should STILL be CSE'd because PURE entries
  // survive the write barrier.
  loom_op_t* const2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const2));
  loom_value_id_t c2 = loom_test_constant_result(const2);

  // Use both constants.
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, c1, c2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  // const1 + write_data + write + const2 + addi = 5 live ops.
  EXPECT_EQ(count_live_ops(), 5);
  IREE_ASSERT_OK(run_cse());
  // const2 eliminated (pure survives write barrier): 4 ops remain.
  EXPECT_EQ(count_live_ops(), 4);
}

TEST_F(CSETest, MutateResourceNotCSEd) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  loom_value_id_t data = loom_test_constant_result(const_op);

  // Two identical mutate ops — readwrite, never CSE-able.
  loom_op_t* mut1 = NULL;
  IREE_ASSERT_OK(loom_test_mutate_resource_build(&builder_, pool_id, data, i32,
                                                 LOOM_LOCATION_UNKNOWN, &mut1));
  loom_op_t* mut2 = NULL;
  IREE_ASSERT_OK(loom_test_mutate_resource_build(&builder_, pool_id, data, i32,
                                                 LOOM_LOCATION_UNKNOWN, &mut2));

  // Use both results.
  loom_value_id_t m1 = loom_test_mutate_resource_old_value(mut1);
  loom_value_id_t m2 = loom_test_mutate_resource_old_value(mut2);
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, m1, m2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 4);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 4);  // No change — writes are never CSE'd.
}

//===----------------------------------------------------------------------===//
// Nested region tests
//===----------------------------------------------------------------------===//

// Helper: count live ops recursively across all blocks and nested regions.
static int count_all_live_ops(loom_region_t* region) {
  int count = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      ++count;
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t r = 0; r < op->region_count; ++r) {
        count += count_all_live_ops(regions[r]);
      }
    }
  }
  return count;
}

TEST_F(CSETest, DuplicatesInsideNestedRegion) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // Build a block arg to feed as input to map.
  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  // Build test.map with one input.
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  // Switch into the map body and build duplicate constants there.
  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, map_op, body);

  loom_op_t* inner_c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(77), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_c1));
  loom_op_t* inner_c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(77), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_c2));
  loom_value_id_t r1 = loom_test_constant_result(inner_c1);
  loom_value_id_t r2 = loom_test_constant_result(inner_c2);
  loom_op_t* inner_add = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                      LOOM_LOCATION_UNKNOWN, &inner_add));

  loom_builder_restore(&builder_, saved);

  // map(1) + inner: c1, c2, addi = 3 inner ops + 1 outer = 4 total.
  EXPECT_EQ(count_all_live_ops(body_), 4);
  IREE_ASSERT_OK(run_cse());
  // One inner constant eliminated: 3 total.
  EXPECT_EQ(count_all_live_ops(body_), 3);
}

TEST_F(CSETest, CrossScopeCSEIntoNestedRegion) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  // Outer constant.
  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));

  // Build test.map.
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  // Inside the map body: duplicate of the outer constant.
  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, map_op, body);

  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));

  loom_builder_restore(&builder_, saved);

  // outer_const + map + inner_const = 3 total.
  EXPECT_EQ(count_all_live_ops(body_), 3);
  IREE_ASSERT_OK(run_cse());
  // Inner constant CSE'd against outer: 2 total.
  EXPECT_EQ(count_all_live_ops(body_), 2);
}

TEST_F(CSETest, NestedCFGRegionUsesEntryBlockDominance) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  loom_region_t* body = loom_test_map_body(map_op);
  body->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
  loom_block_t* exit_block = NULL;
  IREE_ASSERT_OK(loom_region_append_block(module_, body, &exit_block));

  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, map_op, body);
  loom_op_t* entry_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &entry_const));
  loom_op_t* branch = NULL;
  IREE_ASSERT_OK(loom_test_br_build(&builder_, exit_block,
                                    LOOM_LOCATION_UNKNOWN, &branch));

  loom_builder_set_block(&builder_, exit_block);
  loom_op_t* exit_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &exit_const));
  loom_value_id_t exit_value = loom_test_constant_result(exit_const);
  loom_op_t* use = NULL;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &exit_value, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));
  loom_builder_restore(&builder_, saved);

  int before = count_all_live_ops(body_);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_all_live_ops(body_), before - 1);
  EXPECT_EQ(loom_test_use_values(use).values[0],
            loom_test_constant_result(entry_const));
}

TEST_F(CSETest, IsolatedRegionBlocksCrossScope) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  // Outer constant.
  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));

  // Build test.isolated_region (ISOLATED_FROM_ABOVE).
  loom_op_t* iso_op = NULL;
  IREE_ASSERT_OK(loom_test_isolated_region_build(
      &builder_, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &iso_op));

  // Inside the isolated body: same constant value.
  loom_region_t* body = loom_test_isolated_region_body(iso_op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, iso_op, body);

  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));

  loom_builder_restore(&builder_, saved);

  // outer_const + isolated_region + inner_const = 3 total.
  EXPECT_EQ(count_all_live_ops(body_), 3);
  IREE_ASSERT_OK(run_cse());
  // No change — isolated region cannot see outer scope.
  EXPECT_EQ(count_all_live_ops(body_), 3);
}

TEST_F(CSETest, WriteInsideNestedRegionInvalidatesOuterReads) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // Outer read.
  loom_op_t* read1 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool_id, i32,
                                               LOOM_LOCATION_UNKNOWN, &read1));

  // Map with a write inside.
  loom_value_id_t map_input = loom_test_read_resource_result(read1);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &map_input, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  {
    loom_region_t* body = loom_test_map_body(map_op);
    loom_builder_ip_t saved =
        loom_builder_enter_region(&builder_, map_op, body);

    // Write inside nested region — should propagate write barrier up.
    loom_op_t* inner_const = NULL;
    IREE_ASSERT_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(0), i32, LOOM_LOCATION_UNKNOWN, &inner_const));
    loom_op_t* write_op = NULL;
    IREE_ASSERT_OK(loom_test_write_resource_build(
        &builder_, pool_id, loom_test_constant_result(inner_const),
        LOOM_LOCATION_UNKNOWN, &write_op));

    loom_builder_restore(&builder_, saved);
  }

  // Second identical read after the map (which contained a write).
  loom_op_t* read2 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool_id, i32,
                                               LOOM_LOCATION_UNKNOWN, &read2));

  // Use both reads.
  loom_value_id_t r1 = loom_test_read_resource_result(read1);
  loom_value_id_t r2 = loom_test_read_resource_result(read2);
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  int before = count_all_live_ops(body_);
  IREE_ASSERT_OK(run_cse());
  // No change — the write inside the nested region should have
  // invalidated the outer read, preventing CSE of read2.
  EXPECT_EQ(count_all_live_ops(body_), before);
}

TEST_F(CSETest, PureOpSurvivesWriteInsideNestedRegion) {
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // Outer PURE constant.
  loom_op_t* const1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const1));

  // Map with a write inside.
  loom_value_id_t map_input = loom_test_constant_result(const1);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &map_input, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  {
    loom_region_t* body = loom_test_map_body(map_op);
    loom_builder_ip_t saved =
        loom_builder_enter_region(&builder_, map_op, body);

    loom_op_t* inner_wdata = NULL;
    IREE_ASSERT_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(0), i32, LOOM_LOCATION_UNKNOWN, &inner_wdata));
    loom_op_t* write_op = NULL;
    IREE_ASSERT_OK(loom_test_write_resource_build(
        &builder_, pool_id, loom_test_constant_result(inner_wdata),
        LOOM_LOCATION_UNKNOWN, &write_op));

    loom_builder_restore(&builder_, saved);
  }

  // Duplicate PURE constant after the map (which had a write inside).
  // Should still CSE — PURE ops survive write barriers.
  loom_op_t* const2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &const2));

  // Use both.
  loom_value_id_t c1 = loom_test_constant_result(const1);
  loom_value_id_t c2 = loom_test_constant_result(const2);
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, c1, c2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  int before = count_all_live_ops(body_);
  IREE_ASSERT_OK(run_cse());
  // const2 eliminated (PURE survives write barrier): one fewer op.
  EXPECT_EQ(count_all_live_ops(body_), before - 1);
}

TEST_F(CSETest, DeepNestingDoesNotStackOverflow) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  // Build 200 levels of nested test.map ops, each with a duplicate
  // constant pair. This exercises the iterative DFS without blowing
  // the stack.
  static const int kDepth = 200;
  for (int depth = 0; depth < kDepth; ++depth) {
    loom_op_t* map_op = NULL;
    IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &map_op));
    loom_region_t* body = loom_test_map_body(map_op);
    loom_builder_enter_region(&builder_, map_op, body);

    // Duplicate constants at this nesting level.
    loom_op_t* c1 = NULL;
    IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(depth),
                                            i32, LOOM_LOCATION_UNKNOWN, &c1));
    loom_op_t* c2 = NULL;
    IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(depth),
                                            i32, LOOM_LOCATION_UNKNOWN, &c2));
    loom_value_id_t r1 = loom_test_constant_result(c1);
    loom_value_id_t r2 = loom_test_constant_result(c2);
    loom_op_t* add = NULL;
    IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                        LOOM_LOCATION_UNKNOWN, &add));
  }

  int before = count_all_live_ops(body_);
  IREE_ASSERT_OK(run_cse());
  int after = count_all_live_ops(body_);

  // Each nesting level had 1 duplicate constant eliminated.
  EXPECT_EQ(before - after, kDepth);
}

//===----------------------------------------------------------------------===//
// Result type discrimination
//===----------------------------------------------------------------------===//

TEST_F(CSETest, DifferentResultTypesNotCSEd) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i64 = loom_type_scalar(LOOM_SCALAR_TYPE_I64);

  // Block arg as the shared input.
  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &input));

  // Two casts of the same input to different result types.
  // Same kind, same operands, zero attributes — only the result
  // type differs. These must NOT be CSE'd.
  loom_op_t* cast_to_i64 = NULL;
  IREE_ASSERT_OK(loom_test_cast_build(&builder_, input, i32, i64,
                                      LOOM_LOCATION_UNKNOWN, &cast_to_i64));
  loom_op_t* cast_to_f32 = NULL;
  IREE_ASSERT_OK(loom_test_cast_build(&builder_, input, i32, f32,
                                      LOOM_LOCATION_UNKNOWN, &cast_to_f32));

  // Use both results.
  loom_value_id_t r1 = loom_test_cast_result(cast_to_i64);
  loom_value_id_t r2 = loom_test_cast_result(cast_to_f32);
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 3);  // No change.
}

TEST_F(CSETest, SameResultTypesCSEd) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t i64 = loom_type_scalar(LOOM_SCALAR_TYPE_I64);

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &input));

  // Two identical casts — same operand AND same result type.
  loom_op_t* cast1 = NULL;
  IREE_ASSERT_OK(loom_test_cast_build(&builder_, input, i32, i64,
                                      LOOM_LOCATION_UNKNOWN, &cast1));
  loom_op_t* cast2 = NULL;
  IREE_ASSERT_OK(loom_test_cast_build(&builder_, input, i32, i64,
                                      LOOM_LOCATION_UNKNOWN, &cast2));

  loom_value_id_t r1 = loom_test_cast_result(cast1);
  loom_value_id_t r2 = loom_test_cast_result(cast2);
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  EXPECT_EQ(count_live_ops(), 3);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 2);  // One cast eliminated.
}

//===----------------------------------------------------------------------===//
// Tombstone probe chain integrity
//===----------------------------------------------------------------------===//

TEST_F(CSETest, PureOpFoundThroughTombstone) {
  // Regression test for the original probe chain corruption bug.
  // Scenario: a non-PURE read and a PURE constant are inserted into
  // the same table. A write fires, tombstoning the read. A duplicate
  // of the constant must still be found and CSE'd — the tombstone
  // must not terminate the probe chain.
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));

  // Insert a read (non-PURE) into the hash table.
  loom_op_t* read_op = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(
      &builder_, pool_id, i32, LOOM_LOCATION_UNKNOWN, &read_op));

  // Insert a constant (PURE) that may share probe chain slots with
  // the read due to hash collision.
  loom_op_t* const1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(77), i32,
                                          LOOM_LOCATION_UNKNOWN, &const1));

  // Write — tombstones the read, must not break the constant's probe chain.
  loom_op_t* write_data = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &write_data));
  loom_op_t* write_op = NULL;
  IREE_ASSERT_OK(loom_test_write_resource_build(
      &builder_, pool_id, loom_test_constant_result(write_data),
      LOOM_LOCATION_UNKNOWN, &write_op));

  // Duplicate of the PURE constant — should be CSE'd even after
  // the write barrier tombstoned the read entry.
  loom_op_t* const2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(77), i32,
                                          LOOM_LOCATION_UNKNOWN, &const2));

  // Use everything to keep it alive.
  loom_value_id_t c1 = loom_test_constant_result(const1);
  loom_value_id_t c2 = loom_test_constant_result(const2);
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, c1, c2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  int before = count_live_ops();
  IREE_ASSERT_OK(run_cse());
  int after = count_live_ops();
  // const2 should be eliminated: PURE ops survive write barriers,
  // and tombstones must not break probe chain lookups.
  EXPECT_EQ(before - after, 1);
}

//===----------------------------------------------------------------------===//
// Cross-scope CSE through multiple nesting levels
//===----------------------------------------------------------------------===//

TEST_F(CSETest, CSEAcrossMultipleNestingLevels) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), i32, &arg));

  // Outer constant at depth 0.
  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(99), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));

  // Nest 3 levels deep via test.map.
  loom_op_t* map1 = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map1));
  loom_builder_enter_region(&builder_, map1, loom_test_map_body(map1));

  loom_op_t* map2 = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map2));
  loom_builder_enter_region(&builder_, map2, loom_test_map_body(map2));

  loom_op_t* map3 = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, i32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map3));
  loom_builder_enter_region(&builder_, map3, loom_test_map_body(map3));

  // Duplicate of outer constant at depth 3 — should CSE through
  // the entire non-isolated scope chain.
  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(99), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));

  // outer_const + map1 + map2 + map3 + inner_const = 5 total.
  EXPECT_EQ(count_all_live_ops(body_), 5);
  IREE_ASSERT_OK(run_cse());
  // inner_const eliminated via scope chain walk.
  EXPECT_EQ(count_all_live_ops(body_), 4);
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(CSETest, EmptyFunctionBody) {
  // Function with a body but zero ops — should not crash.
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 0);
}

TEST_F(CSETest, NullFunctionBody) {
  // A zero-initialized func_like is the invalid/null sentinel — CSE must
  // handle it gracefully and return OK.
  loom_func_like_t empty_func = {};

  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(&block_pool_, &pass_arena);
  loom_pass_t pass;
  memset(&pass, 0, sizeof(pass));
  pass.info = loom_cse_pass_info();
  pass.instance_arena = &pass_arena;
  pass.arena = &pass_arena;
  IREE_ASSERT_OK(InitializePassStatistics(&pass, &pass_arena));
  IREE_EXPECT_OK(loom_cse_run(&pass, module_, empty_func));
  iree_arena_deinitialize(&pass_arena);
}

TEST_F(CSETest, SingleOpNoCSEOpportunity) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* const_op = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &const_op));
  EXPECT_EQ(count_live_ops(), 1);
  IREE_ASSERT_OK(run_cse());
  EXPECT_EQ(count_live_ops(), 1);
}

TEST_F(CSETest, UniqueIdentityNotCSEd) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));

  // Create an index block arg for the allocation size.
  loom_value_id_t size_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), index_type, &size_id));

  // Two identical alloc ops with the same size operand and result type.
  loom_op_t* alloc1 = NULL;
  IREE_ASSERT_OK(loom_test_alloc_build(&builder_, size_id, pool_type,
                                       LOOM_LOCATION_UNKNOWN, &alloc1));
  loom_value_id_t pool1 = loom_test_alloc_result(alloc1);

  loom_op_t* alloc2 = NULL;
  IREE_ASSERT_OK(loom_test_alloc_build(&builder_, size_id, pool_type,
                                       LOOM_LOCATION_UNKNOWN, &alloc2));
  loom_value_id_t pool2 = loom_test_alloc_result(alloc2);

  // Use both pools via read_resource to keep them alive.
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* read1 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool1, i32,
                                               LOOM_LOCATION_UNKNOWN, &read1));
  loom_op_t* read2 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool2, i32,
                                               LOOM_LOCATION_UNKNOWN, &read2));

  // 2 allocs + 2 reads = 4 ops.
  EXPECT_EQ(count_live_ops(), 4);
  IREE_ASSERT_OK(run_cse());
  // Both allocs must survive — UNIQUE_IDENTITY prevents CSE.
  EXPECT_EQ(count_live_ops(), 4);
}

TEST_F(CSETest, UniqueIdentityDoesNotTriggerWriteBarrier) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t pool_type = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_value_id_t pool_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), pool_type, &pool_id));
  loom_value_id_t size_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body_), index_type, &size_id));

  // First read from the pool.
  loom_op_t* read1 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool_id, i32,
                                               LOOM_LOCATION_UNKNOWN, &read1));
  loom_value_id_t r1 = loom_test_read_resource_result(read1);

  // An alloc between the reads — should NOT act as a write barrier.
  loom_op_t* alloc_op = NULL;
  IREE_ASSERT_OK(loom_test_alloc_build(&builder_, size_id, pool_type,
                                       LOOM_LOCATION_UNKNOWN, &alloc_op));

  // Second identical read from the same pool.
  loom_op_t* read2 = NULL;
  IREE_ASSERT_OK(loom_test_read_resource_build(&builder_, pool_id, i32,
                                               LOOM_LOCATION_UNKNOWN, &read2));
  loom_value_id_t r2 = loom_test_read_resource_result(read2);

  // Use both read results to keep them alive.
  loom_op_t* addi = NULL;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, r1, r2, i32,
                                      LOOM_LOCATION_UNKNOWN, &addi));

  // 2 reads + 1 alloc + 1 addi = 4 ops.
  EXPECT_EQ(count_live_ops(), 4);
  IREE_ASSERT_OK(run_cse());
  // Second read should be CSE'd (alloc is not a write barrier).
  // Alloc becomes unused and will be cleaned up by DCE later, but
  // CSE doesn't remove it — it just skips it as a non-candidate.
  EXPECT_EQ(count_live_ops(), 3);
}

}  // namespace
}  // namespace loom
