// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/motion.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class MotionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t test_vtable_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 test_vtables,
                                                 (uint16_t)test_vtable_count));

    iree_host_size_t scf_vtable_count = 0;
    const loom_op_vtable_t* const* scf_vtables =
        loom_scf_dialect_vtables(&scf_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, scf_vtables, (uint16_t)scf_vtable_count));

    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    IREE_ASSERT_OK(loom_test_func_build(
        &module_builder, 0, 0, 0, callee, nullptr, 0, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op_));
    loom_func_like_t function = loom_func_like_cast(module_, function_op_);
    loom_region_t* body = loom_func_like_body(function);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body), &builder_);
    builder_.ip.parent_op = function_op_;

    iree_arena_initialize(&block_pool_, &motion_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&motion_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void EnterRegion(loom_op_t* parent_op, loom_region_t* region) {
    saved_insertions_.push_back(
        loom_builder_enter_region(&builder_, parent_op, region));
  }

  void LeaveRegion() {
    ASSERT_FALSE(saved_insertions_.empty());
    loom_builder_restore(&builder_, saved_insertions_.back());
    saved_insertions_.pop_back();
  }

  void PrepareMotionAnalysis() {
    IREE_ASSERT_OK(loom_module_compute_uses(module_));
    IREE_ASSERT_OK(
        loom_motion_analysis_initialize(module_, &motion_arena_, &motion_));
  }

  iree_status_t QueryRelocateBefore(loom_op_t* candidate_op,
                                    loom_op_t* before_op,
                                    bool* out_can_relocate) {
    return loom_motion_subtree_can_relocate_before(&motion_, candidate_op,
                                                   before_op, out_can_relocate);
  }

  iree_status_t QuerySpeculateBefore(loom_op_t* candidate_op,
                                     loom_op_t* before_op,
                                     bool* out_can_speculate) {
    return loom_motion_subtree_can_speculate_before(
        &motion_, candidate_op, before_op, out_can_speculate);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* function_op_ = nullptr;
  loom_builder_t builder_;
  iree_arena_allocator_t motion_arena_;
  loom_motion_analysis_t motion_;
  std::vector<loom_builder_ip_t> saved_insertions_;
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_F(MotionTest, LocalClassificationSeparatesEraseRelocateAndSpeculate) {
  loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  loom_type_t tensor_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);

  loom_op_t* lhs_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &lhs_op));
  loom_value_id_t lhs = loom_test_constant_result(lhs_op);

  loom_op_t* rhs_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &rhs_op));
  loom_value_id_t rhs = loom_test_constant_result(rhs_op);

  loom_op_t* dead_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(99), i32_type, LOOM_LOCATION_UNKNOWN, &dead_op));

  loom_op_t* add_op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, lhs, rhs, i32_type,
                                      LOOM_LOCATION_UNKNOWN, &add_op));
  loom_value_id_t add_result = loom_test_addi_result(add_op);

  loom_op_t* use_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &add_result, 1,
                                     LOOM_LOCATION_UNKNOWN, &use_op));

  loom_op_t* convergent_op = nullptr;
  IREE_ASSERT_OK(loom_test_convergent_build(
      &builder_, lhs, i32_type, LOOM_LOCATION_UNKNOWN, &convergent_op));

  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &source_op));
  loom_value_id_t source = loom_test_constant_result(source_op);

  loom_op_t* target_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2),
                                          tensor_type, LOOM_LOCATION_UNKNOWN,
                                          &target_op));
  loom_value_id_t target = loom_test_constant_result(target_op);

  int64_t static_offsets[] = {0};
  loom_op_t* update_op = nullptr;
  IREE_ASSERT_OK(
      loom_test_update_build(&builder_, source, target, nullptr, 0,
                             static_offsets, IREE_ARRAYSIZE(static_offsets),
                             tensor_type, LOOM_LOCATION_UNKNOWN, &update_op));

  loom_op_t* condition_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_bool(true),
                                          i1_type, LOOM_LOCATION_UNKNOWN,
                                          &condition_op));
  loom_value_id_t condition = loom_test_constant_result(condition_op);
  loom_op_t* select_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(&builder_, condition, lhs, rhs, i32_type,
                                       LOOM_LOCATION_UNKNOWN, &select_op));

  PrepareMotionAnalysis();

  EXPECT_TRUE(loom_motion_op_can_erase(module_, dead_op));
  EXPECT_FALSE(loom_motion_op_can_erase(module_, add_op));
  EXPECT_FALSE(loom_motion_op_can_erase(module_, use_op));
  EXPECT_FALSE(loom_motion_op_can_erase(module_, convergent_op));
  EXPECT_TRUE(loom_motion_op_can_relocate_effect_free(module_, add_op));
  EXPECT_FALSE(loom_motion_op_can_relocate_effect_free(module_, use_op));
  EXPECT_FALSE(loom_motion_op_can_relocate_effect_free(module_, convergent_op));
  EXPECT_FALSE(loom_motion_op_can_relocate_effect_free(module_, update_op));
  EXPECT_FALSE(loom_motion_op_can_speculate(module_, add_op));
  EXPECT_FALSE(loom_motion_op_can_speculate(module_, convergent_op));
  EXPECT_TRUE(loom_motion_op_can_speculate(module_, select_op));
}

TEST_F(MotionTest, SubtreeAllowsOperandsAvailableBeforeInsertion) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* lhs_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &lhs_op));
  loom_value_id_t lhs = loom_test_constant_result(lhs_op);

  loom_op_t* rhs_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &rhs_op));
  loom_value_id_t rhs = loom_test_constant_result(rhs_op);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* add_op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, lhs, rhs, i32_type,
                                      LOOM_LOCATION_UNKNOWN, &add_op));

  PrepareMotionAnalysis();

  bool can_relocate = false;
  IREE_ASSERT_OK(QueryRelocateBefore(add_op, anchor_op, &can_relocate));
  EXPECT_TRUE(can_relocate);
}

TEST_F(MotionTest, SubtreeRejectsOperandDefinedAfterInsertion) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* late_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &late_op));
  loom_value_id_t late = loom_test_constant_result(late_op);

  loom_op_t* add_op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, late, late, i32_type,
                                      LOOM_LOCATION_UNKNOWN, &add_op));

  PrepareMotionAnalysis();

  bool can_relocate = true;
  IREE_ASSERT_OK(QueryRelocateBefore(add_op, anchor_op, &can_relocate));
  EXPECT_FALSE(can_relocate);
}

TEST_F(MotionTest, SubtreeAllowsValuesDefinedInsideMovedRegion) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &input, 1, tile_type, nullptr,
                                     0, LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* map_body = loom_test_map_body(map_op);
  loom_block_t* map_entry = loom_region_entry_block(map_body);
  loom_value_id_t element = loom_block_arg_id(map_entry, 0);

  EnterRegion(map_op, map_body);
  loom_op_t* doubled_op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, element, element, i32_type,
                                      LOOM_LOCATION_UNKNOWN, &doubled_op));
  loom_value_id_t doubled = loom_test_addi_result(doubled_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &doubled, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  LeaveRegion();

  PrepareMotionAnalysis();

  bool can_relocate = false;
  IREE_ASSERT_OK(QueryRelocateBefore(map_op, anchor_op, &can_relocate));
  EXPECT_TRUE(can_relocate);
}

TEST_F(MotionTest, SubtreeAllowsTypeReferenceToBlockArgInEmptyMovedRegion) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &input, 1, tile_type, nullptr,
                                     0, LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* map_body = loom_test_map_body(map_op);
  loom_block_t* map_entry = loom_region_entry_block(map_body);
  loom_value_id_t element = loom_block_arg_id(map_entry, 0);
  loom_type_t block_arg_sized_result = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(element), 0);
  IREE_ASSERT_OK(loom_module_set_value_type(
      module_, loom_test_map_result(map_op), block_arg_sized_result));

  PrepareMotionAnalysis();

  bool can_relocate = false;
  IREE_ASSERT_OK(QueryRelocateBefore(map_op, anchor_op, &can_relocate));
  EXPECT_TRUE(can_relocate);
}

TEST_F(MotionTest, SubtreeRejectsNestedEffects) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &input, 1, tile_type, nullptr,
                                     0, LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* map_body = loom_test_map_body(map_op);
  loom_block_t* map_entry = loom_region_entry_block(map_body);
  loom_value_id_t element = loom_block_arg_id(map_entry, 0);

  EnterRegion(map_op, map_body);
  loom_op_t* use_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&builder_, &element, 1,
                                     LOOM_LOCATION_UNKNOWN, &use_op));
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &element, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  LeaveRegion();

  PrepareMotionAnalysis();

  bool can_relocate = true;
  IREE_ASSERT_OK(QueryRelocateBefore(map_op, anchor_op, &can_relocate));
  EXPECT_FALSE(can_relocate);
}

TEST_F(MotionTest, SubtreeRejectsUnavailableTypeReference) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* late_dim_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(4),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &late_dim_op));
  loom_value_id_t late_dim = loom_test_constant_result(late_dim_op);

  loom_type_t dynamic_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_dynamic(late_dim), 0);
  loom_op_t* vector_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1),
                                          dynamic_vector_type,
                                          LOOM_LOCATION_UNKNOWN, &vector_op));

  PrepareMotionAnalysis();

  bool can_relocate = true;
  IREE_ASSERT_OK(QueryRelocateBefore(vector_op, anchor_op, &can_relocate));
  EXPECT_FALSE(can_relocate);
}

TEST_F(MotionTest, SubtreeRejectsUnavailablePredicateAttrReference) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* late_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &late_op));
  loom_value_id_t late = loom_test_constant_result(late_op);

  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_EQ,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{(int64_t)late, 2},
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_test_assume_build(&builder_, &input, 1, &predicate, 1,
                                        &i32_type, 1, LOOM_LOCATION_UNKNOWN,
                                        &assume_op));

  PrepareMotionAnalysis();

  bool can_relocate = true;
  IREE_ASSERT_OK(QueryRelocateBefore(assume_op, anchor_op, &can_relocate));
  EXPECT_FALSE(can_relocate);
}

TEST_F(MotionTest, SpeculationRequiresSafeToSpeculate) {
  loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* lhs_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &lhs_op));
  loom_value_id_t lhs = loom_test_constant_result(lhs_op);

  loom_op_t* rhs_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &rhs_op));
  loom_value_id_t rhs = loom_test_constant_result(rhs_op);

  loom_op_t* condition_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_bool(true),
                                          i1_type, LOOM_LOCATION_UNKNOWN,
                                          &condition_op));
  loom_value_id_t condition = loom_test_constant_result(condition_op);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* add_op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, lhs, rhs, i32_type,
                                      LOOM_LOCATION_UNKNOWN, &add_op));

  loom_op_t* select_op = nullptr;
  IREE_ASSERT_OK(loom_scf_select_build(&builder_, condition, lhs, rhs, i32_type,
                                       LOOM_LOCATION_UNKNOWN, &select_op));

  PrepareMotionAnalysis();

  bool can_relocate = false;
  IREE_ASSERT_OK(QueryRelocateBefore(add_op, anchor_op, &can_relocate));
  EXPECT_TRUE(can_relocate);

  bool can_speculate = true;
  IREE_ASSERT_OK(QuerySpeculateBefore(add_op, anchor_op, &can_speculate));
  EXPECT_FALSE(can_speculate);

  can_speculate = false;
  IREE_ASSERT_OK(QuerySpeculateBefore(select_op, anchor_op, &can_speculate));
  EXPECT_TRUE(can_speculate);
}

TEST_F(MotionTest, RejectsMovingSubtreeBeforeDescendant) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &input, 1, tile_type, nullptr,
                                     0, LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* map_body = loom_test_map_body(map_op);
  loom_block_t* map_entry = loom_region_entry_block(map_body);
  loom_value_id_t element = loom_block_arg_id(map_entry, 0);

  EnterRegion(map_op, map_body);
  loom_op_t* doubled_op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, element, element, i32_type,
                                      LOOM_LOCATION_UNKNOWN, &doubled_op));
  loom_value_id_t doubled = loom_test_addi_result(doubled_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &doubled, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  LeaveRegion();

  PrepareMotionAnalysis();

  bool can_relocate = true;
  IREE_ASSERT_OK(QueryRelocateBefore(map_op, doubled_op, &can_relocate));
  EXPECT_FALSE(can_relocate);
}

}  // namespace
}  // namespace loom
