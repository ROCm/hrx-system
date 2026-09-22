// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/materialize.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/rewrite/rewriter.h"

namespace loom {
namespace {

class MaterializeTest : public ::testing::Test {
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
    vtables = loom_scf_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SCF, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &source_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("target"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &target_));
    loom_builder_initialize(source_, &source_->arena,
                            loom_module_block(source_), &source_builder_);
    loom_builder_initialize(target_, &target_->arena,
                            loom_module_block(target_), &target_builder_);
    iree_arena_initialize(&block_pool_, &remap_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&remap_arena_);
    loom_module_free(target_);
    loom_module_free(source_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_ir_remap_t InitializeRemap(
      bool allow_unmapped_values = false,
      const loom_ir_remap_options_t* options = nullptr) {
    loom_ir_remap_options_t local_options = {};
    if (options) {
      local_options = *options;
    }
    local_options.allow_unmapped_values = allow_unmapped_values;
    loom_ir_remap_t remap = {};
    IREE_CHECK_OK(loom_ir_remap_initialize(source_, target_, &remap_arena_,
                                           &local_options, &remap));
    return remap;
  }

  loom_ir_remap_t InitializeSameModuleRemap(bool allow_unmapped_values) {
    loom_ir_remap_options_t options = {
        /*.allow_unmapped_values=*/allow_unmapped_values,
    };
    loom_ir_remap_t remap = {};
    IREE_CHECK_OK(loom_ir_remap_initialize(source_, source_, &remap_arena_,
                                           &options, &remap));
    return remap;
  }

  loom_availability_analysis_t InitializeAvailability() {
    loom_availability_analysis_t availability = {};
    IREE_CHECK_OK(loom_availability_analysis_initialize(source_, &remap_arena_,
                                                        &availability));
    return availability;
  }

  loom_op_t* BuildRegionOwner(loom_builder_t* builder) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_block_args_build(builder, nullptr, 0,
                                             LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildHint(loom_builder_t* builder) {
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(
        loom_scf_schedule_fence_build(builder, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* source_ = nullptr;
  loom_module_t* target_ = nullptr;
  loom_builder_t source_builder_ = {};
  loom_builder_t target_builder_ = {};
  iree_arena_allocator_t remap_arena_;
};

TEST_F(MaterializeTest, ClonesOwnedDeclarationArguments) {
  loom_string_id_t source_name = LOOM_STRING_ID_INVALID;
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("source"), &source_name));
  IREE_ASSERT_OK(
      loom_module_intern_string(target_, IREE_SV("target"), &target_name));
  uint16_t source_symbol = LOOM_SYMBOL_ID_INVALID;
  uint16_t target_symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(source_, source_name, &source_symbol));
  IREE_ASSERT_OK(loom_module_add_symbol(target_, target_name, &target_symbol));
  const loom_type_t argument_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_pool(loom_dim_pack_static(4)),
  };
  loom_op_t* source_declaration = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(
      &source_builder_, /*build_flags=*/0, /*visibility=*/0, /*cc=*/0,
      {0, source_symbol}, argument_types, IREE_ARRAYSIZE(argument_types),
      /*result_types=*/nullptr, /*result_count=*/0, /*tied_results=*/nullptr,
      /*tied_result_count=*/0, LOOM_LOCATION_UNKNOWN, &source_declaration));
  const loom_value_slice_t source_arguments =
      loom_test_decl_args(source_declaration);
  IREE_ASSERT_OK(loom_module_set_value_type(
      source_, source_arguments.values[1],
      loom_type_pool(loom_dim_pack_dynamic(source_arguments.values[0]))));

  loom_op_t* target_constant = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &target_builder_, loom_attr_i64(4), argument_types[0],
      LOOM_LOCATION_UNKNOWN, &target_constant));
  loom_symbol_ref_t target_callee = {0, target_symbol};
  loom_ir_remap_options_t options = {};
  options.remap_symbol = loom_ir_remap_symbol_callback_make(
      [](void* user_data, const loom_module_t*, loom_module_t*,
         loom_symbol_ref_t, loom_symbol_ref_t* out_ref) {
        *out_ref = *static_cast<const loom_symbol_ref_t*>(user_data);
        return iree_ok_status();
      },
      &target_callee);
  loom_ir_remap_t remap =
      InitializeRemap(/*allow_unmapped_values=*/false, &options);
  loom_op_t* target_declaration = nullptr;
  IREE_ASSERT_OK(loom_ir_clone_op(&target_builder_, source_declaration, &remap,
                                  &target_declaration));
  const loom_value_slice_t target_arguments =
      loom_test_decl_args(target_declaration);
  ASSERT_EQ(target_arguments.count, source_arguments.count);
  for (uint16_t i = 0; i < target_arguments.count; ++i) {
    EXPECT_NE(target_arguments.values[i], source_arguments.values[i]);
    const loom_value_t* argument =
        loom_module_value(target_, target_arguments.values[i]);
    EXPECT_EQ(argument->use_count, 1u);
    EXPECT_EQ(loom_value_def_op(argument), nullptr);
    EXPECT_EQ(loom_value_owner_op(argument), target_declaration);
  }
  EXPECT_TRUE(loom_type_equal(
      loom_module_value_type(target_, target_arguments.values[1]),
      loom_type_pool(loom_dim_pack_dynamic(target_arguments.values[0]))));
  IREE_ASSERT_OK(loom_op_erase(target_, target_declaration));
  EXPECT_FALSE(loom_module_has_active_type_uses(target_));
  EXPECT_TRUE(loom_module_has_active_type_uses(source_));
}

TEST_F(MaterializeTest, ClonesCoResultDynamicTypeReferences) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t input_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(0),
                                          input_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t source_input = loom_test_constant_result(input_op);

  loom_value_id_t reserved_results[2] = {};
  IREE_ASSERT_OK(loom_builder_reserve_results(
      &source_builder_, IREE_ARRAYSIZE(reserved_results), reserved_results));
  loom_type_t result_types[] = {
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(reserved_results[1]), 0),
      index_type,
  };
  loom_op_t* deflate_op = nullptr;
  IREE_ASSERT_OK(
      loom_test_deflate_build(&source_builder_, source_input, result_types,
                              IREE_ARRAYSIZE(result_types), nullptr, 0,
                              LOOM_LOCATION_UNKNOWN, &deflate_op));

  loom_op_t* target_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(0),
                                          input_type, LOOM_LOCATION_UNKNOWN,
                                          &target_input_op));
  loom_value_id_t target_input = loom_test_constant_result(target_input_op);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_input, target_input));
  loom_op_t* cloned_op = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, deflate_op, &remap, &cloned_op));

  ASSERT_TRUE(loom_test_deflate_isa(cloned_op));
  loom_value_slice_t cloned_results = loom_test_deflate_results(cloned_op);
  ASSERT_EQ(cloned_results.count, 2u);
  loom_type_t cloned_output_type =
      loom_module_value_type(target_, cloned_results.values[0]);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(cloned_output_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(cloned_output_type, 0),
            cloned_results.values[1]);
}

TEST_F(MaterializeTest, ClonesSegmentedOperandCounts) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t source_root = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_guard = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_lhs0 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_lhs1 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_rhs = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_value(&source_builder_, i32, &source_root));
  IREE_ASSERT_OK(
      loom_builder_define_value(&source_builder_, i32, &source_guard));
  IREE_ASSERT_OK(
      loom_builder_define_value(&source_builder_, i32, &source_lhs0));
  IREE_ASSERT_OK(
      loom_builder_define_value(&source_builder_, i32, &source_lhs1));
  IREE_ASSERT_OK(loom_builder_define_value(&source_builder_, i32, &source_rhs));

  loom_value_id_t lhs[] = {source_lhs0, source_lhs1};
  loom_value_id_t rhs[] = {source_rhs};
  loom_op_t* source_op = NULL;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &source_builder_, LOOM_TEST_SEGMENTED_BUILD_FLAG_HAS_GUARD, source_root,
      source_guard, lhs, IREE_ARRAYSIZE(lhs), rhs, IREE_ARRAYSIZE(rhs), i32,
      LOOM_LOCATION_UNKNOWN, &source_op));

  loom_value_id_t target_root = LOOM_VALUE_ID_INVALID;
  loom_value_id_t target_guard = LOOM_VALUE_ID_INVALID;
  loom_value_id_t target_lhs0 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t target_lhs1 = LOOM_VALUE_ID_INVALID;
  loom_value_id_t target_rhs = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_value(&target_builder_, i32, &target_root));
  IREE_ASSERT_OK(
      loom_builder_define_value(&target_builder_, i32, &target_guard));
  IREE_ASSERT_OK(
      loom_builder_define_value(&target_builder_, i32, &target_lhs0));
  IREE_ASSERT_OK(
      loom_builder_define_value(&target_builder_, i32, &target_lhs1));
  IREE_ASSERT_OK(loom_builder_define_value(&target_builder_, i32, &target_rhs));

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_root, target_root));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_guard, target_guard));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_lhs0, target_lhs0));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_lhs1, target_lhs1));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_rhs, target_rhs));

  loom_op_t* cloned_op = NULL;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, source_op, &remap, &cloned_op));

  ASSERT_TRUE(loom_test_segmented_isa(cloned_op));
  const uint16_t* counts = loom_op_const_operand_segment_counts(cloned_op);
  EXPECT_EQ(counts[0], 1u);
  EXPECT_EQ(counts[1], 1u);
  EXPECT_EQ(counts[2], 2u);
  EXPECT_EQ(counts[3], 1u);
  EXPECT_EQ(loom_test_segmented_root(cloned_op), target_root);
  EXPECT_EQ(loom_test_segmented_guard(cloned_op), target_guard);
  loom_value_slice_t cloned_lhs = loom_test_segmented_lhs(cloned_op);
  ASSERT_EQ(cloned_lhs.count, 2u);
  EXPECT_EQ(cloned_lhs.values[0], target_lhs0);
  EXPECT_EQ(cloned_lhs.values[1], target_lhs1);
  loom_value_slice_t cloned_rhs = loom_test_segmented_rhs(cloned_op);
  ASSERT_EQ(cloned_rhs.count, 1u);
  EXPECT_EQ(cloned_rhs.values[0], target_rhs);
}

TEST_F(MaterializeTest, ClonesNestedRegionsAndBlockArguments) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(0),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t source_input = loom_test_constant_result(input_op);

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&source_builder_, &source_input, 1,
                                     tile_type, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* source_body = loom_test_map_body(map_op);
  loom_builder_ip_t saved_source_ip =
      loom_builder_enter_region(&source_builder_, map_op, source_body);
  loom_value_id_t element = loom_region_entry_arg_id(source_body, 0);
  loom_op_t* neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&source_builder_, element, f32_type,
                                     LOOM_LOCATION_UNKNOWN, &neg_op));
  loom_value_id_t negated = loom_test_neg_result(neg_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&source_builder_, &negated, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&source_builder_, saved_source_ip);

  loom_op_t* target_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(0),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &target_input_op));
  loom_value_id_t target_input = loom_test_constant_result(target_input_op);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_input, target_input));
  loom_op_t* cloned_map_op = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, map_op, &remap, &cloned_map_op));

  ASSERT_TRUE(loom_test_map_isa(cloned_map_op));
  EXPECT_EQ(loom_test_map_inputs(cloned_map_op).values[0], target_input);
  loom_region_t* cloned_body = loom_test_map_body(cloned_map_op);
  ASSERT_NE(cloned_body, nullptr);
  ASSERT_EQ(loom_region_entry_arg_count(cloned_body), 1u);
  loom_block_t* cloned_block = loom_region_entry_block(cloned_body);
  ASSERT_EQ(cloned_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(cloned_block, 0)));
  ASSERT_TRUE(loom_test_yield_isa(loom_block_op(cloned_block, 1)));
  EXPECT_EQ(loom_block_op(cloned_block, 0)->parent_op, cloned_map_op);
  EXPECT_EQ(loom_block_op(cloned_block, 1)->parent_op, cloned_map_op);
  EXPECT_EQ(loom_test_neg_input(loom_block_op(cloned_block, 0)),
            loom_region_entry_arg_id(cloned_body, 0));
}

TEST_F(MaterializeTest, ClonesOpResultTypesReferencingRegionArguments) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_INDEX, loom_dim_pack_static(4), 0);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(0),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t source_input = loom_test_constant_result(input_op);

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&source_builder_, &source_input, 1,
                                     tile_type, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  loom_region_t* source_body = loom_test_map_body(map_op);
  loom_value_id_t source_element = loom_region_entry_arg_id(source_body, 0);
  loom_type_t dynamic_result_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_INDEX,
                          loom_dim_pack_dynamic(source_element), 0);
  IREE_ASSERT_OK(loom_module_set_value_type(
      source_, loom_test_map_result(map_op), dynamic_result_type));

  loom_op_t* target_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(0),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &target_input_op));
  loom_value_id_t target_input = loom_test_constant_result(target_input_op);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_input, target_input));
  loom_op_t* cloned_map_op = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, map_op, &remap, &cloned_map_op));

  loom_region_t* cloned_body = loom_test_map_body(cloned_map_op);
  ASSERT_NE(cloned_body, nullptr);
  loom_type_t cloned_result_type =
      loom_module_value_type(target_, loom_test_map_result(cloned_map_op));
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(cloned_result_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(cloned_result_type, 0),
            loom_region_entry_arg_id(cloned_body, 0));
}

TEST_F(MaterializeTest, ClonesRegionSuccessorsToClonedBlocks) {
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 2, &source_region));
  loom_block_t* source_entry = loom_region_block(source_region, 0);
  loom_block_t* source_dest = loom_region_block(source_region, 1);
  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena, source_entry,
                          &source_region_builder);
  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(loom_test_br_build(&source_region_builder, source_dest,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));

  loom_ir_remap_t remap = InitializeRemap();
  loom_region_t* cloned_region = nullptr;
  IREE_ASSERT_OK(loom_ir_clone_region(&target_builder_, source_region, &remap,
                                      &cloned_region));

  ASSERT_NE(cloned_region, nullptr);
  ASSERT_EQ(cloned_region->block_count, 2u);
  loom_block_t* cloned_entry = loom_region_block(cloned_region, 0);
  loom_block_t* cloned_dest = loom_region_block(cloned_region, 1);
  ASSERT_EQ(cloned_entry->op_count, 1u);
  const loom_op_t* cloned_branch = loom_block_const_op(cloned_entry, 0);
  ASSERT_TRUE(loom_test_br_isa(cloned_branch));
  EXPECT_EQ(loom_test_br_dest(cloned_branch), cloned_dest);
  EXPECT_NE(loom_test_br_dest(cloned_branch), source_dest);
  EXPECT_EQ(remap.block_map_count, 0u);
}

TEST_F(MaterializeTest, ClonesReferencesToLaterListedBlockDefinitions) {
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 3, &source_region));
  loom_block_t* source_entry = loom_region_block(source_region, 0);
  loom_block_t* source_use = loom_region_block(source_region, 1);
  loom_block_t* source_definition = loom_region_block(source_region, 2);
  loom_builder_t builder = {};
  loom_builder_initialize(source_, &source_->arena, source_entry, &builder);
  loom_op_t* branch = nullptr;
  IREE_ASSERT_OK(loom_test_br_build(&builder, source_definition,
                                    LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_initialize(source_, &source_->arena, source_definition,
                          &builder);
  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder, loom_attr_i64(16), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      LOOM_LOCATION_UNKNOWN, &constant));
  const loom_value_id_t width = loom_test_constant_result(constant);
  loom_type_t tensor_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(width), 0);
  loom_op_t* tensor = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder, loom_attr_i64(0), tensor_type, LOOM_LOCATION_UNKNOWN, &tensor));
  const loom_value_id_t tensor_value = loom_test_constant_result(tensor);
  IREE_ASSERT_OK(
      loom_test_br_build(&builder, source_use, LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_initialize(source_, &source_->arena, source_use, &builder);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(
      loom_test_use_build(&builder, &width, 1, LOOM_LOCATION_UNKNOWN, &use));
  loom_op_t* nested = nullptr;
  IREE_ASSERT_OK(loom_test_block_args_build(&builder, &tensor_value, 1,
                                            LOOM_LOCATION_UNKNOWN, &nested));

  loom_ir_remap_t remap = InitializeRemap();
  loom_region_t* cloned_region = nullptr;
  IREE_ASSERT_OK(loom_ir_clone_region(&target_builder_, source_region, &remap,
                                      &cloned_region));
  ASSERT_EQ(cloned_region->block_count, 3u);
  const loom_block_t* cloned_use = loom_region_const_block(cloned_region, 1);
  const loom_block_t* cloned_definition =
      loom_region_const_block(cloned_region, 2);
  const loom_value_id_t cloned_width =
      loom_test_constant_result(cloned_definition->first_op);
  EXPECT_EQ(loom_test_use_values(cloned_use->first_op).values[0], cloned_width);
  const loom_region_t* cloned_nested =
      loom_test_block_args_body(cloned_use->last_op);
  const loom_type_t cloned_type = loom_module_value_type(
      target_, loom_region_entry_arg_id(cloned_nested, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(cloned_type, 0), cloned_width);
  EXPECT_EQ(loom_test_br_dest(loom_region_block(cloned_region, 0)->last_op),
            cloned_definition);
  EXPECT_EQ(loom_test_br_dest(cloned_definition->last_op), cloned_use);
}

TEST_F(MaterializeTest, SplicesRegionBlocksAtTargetOffset) {
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 3, &source_region));
  loom_block_t* source_entry = loom_region_block(source_region, 0);
  loom_block_t* source_dest = loom_region_block(source_region, 1);
  loom_block_t* source_definition = loom_region_block(source_region, 2);
  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena, source_entry,
                          &source_region_builder);
  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(loom_test_br_build(&source_region_builder, source_definition,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));
  loom_builder_initialize(source_, &source_->arena, source_definition,
                          &source_region_builder);
  loom_op_t* constant = nullptr;
  IREE_ASSERT_OK(
      loom_test_constant_build(&source_region_builder, loom_attr_i64(7),
                               loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                               LOOM_LOCATION_UNKNOWN, &constant));
  const loom_value_id_t value = loom_test_constant_result(constant);
  IREE_ASSERT_OK(loom_test_br_build(&source_region_builder, source_dest,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));
  loom_builder_initialize(source_, &source_->arena, source_dest,
                          &source_region_builder);
  loom_op_t* use = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&source_region_builder, &value, 1,
                                     LOOM_LOCATION_UNKNOWN, &use));

  loom_region_t* target_region = target_->body;
  loom_block_t* target_entry = loom_region_entry_block(target_region);
  loom_block_t* target_tail0 = nullptr;
  IREE_ASSERT_OK(
      loom_region_append_block(target_, target_region, &target_tail0));
  loom_block_t* target_tail1 = nullptr;
  IREE_ASSERT_OK(
      loom_region_append_block(target_, target_region, &target_tail1));

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_clone_region_blocks(&target_builder_, source_region,
                                             target_region, 1, &remap));

  ASSERT_EQ(target_region->block_count, 6u);
  EXPECT_EQ(loom_region_block(target_region, 0), target_entry);
  loom_block_t* cloned_entry = loom_region_block(target_region, 1);
  loom_block_t* cloned_dest = loom_region_block(target_region, 2);
  loom_block_t* cloned_definition = loom_region_block(target_region, 3);
  EXPECT_EQ(loom_region_block(target_region, 4), target_tail0);
  EXPECT_EQ(loom_region_block(target_region, 5), target_tail1);
  ASSERT_TRUE(loom_test_br_isa(cloned_entry->first_op));
  EXPECT_EQ(loom_test_br_dest(cloned_entry->first_op), cloned_definition);
  EXPECT_EQ(loom_test_br_dest(cloned_definition->last_op), cloned_dest);
  EXPECT_EQ(loom_test_use_values(cloned_dest->first_op).values[0],
            loom_test_constant_result(cloned_definition->first_op));
  for (uint16_t i = 0; i < target_region->block_count; ++i) {
    EXPECT_EQ(loom_block_region_index(loom_region_block(target_region, i)), i);
  }
  EXPECT_EQ(remap.block_map_count, 0u);
}

TEST_F(MaterializeTest, ClonesRegionSourcePresentation) {
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 1, &source_region));
  source_region->source_flags |= LOOM_REGION_SOURCE_FLAG_EXPLICIT_LOW_ASM;

  loom_ir_remap_t remap = InitializeRemap();
  loom_region_t* cloned_region = nullptr;
  IREE_ASSERT_OK(loom_ir_clone_region(&target_builder_, source_region, &remap,
                                      &cloned_region));

  ASSERT_NE(cloned_region, nullptr);
  EXPECT_EQ(cloned_region->source_flags, source_region->source_flags);
}

TEST_F(MaterializeTest, ClonesOperationAndBlockSourcePresentation) {
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 1, &source_region));
  loom_block_t* source_block = loom_region_entry_block(source_region);
  source_block->flags |= LOOM_BLOCK_FLAG_LEADING_BLANK_LINE;
  const iree_string_view_t block_comments[] = {
      IREE_SV("block heading"),
  };
  IREE_ASSERT_OK(loom_module_attach_block_comments(
      source_, source_block, block_comments, IREE_ARRAYSIZE(block_comments)));

  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena, source_block,
                          &source_region_builder);
  loom_op_t* source_op = nullptr;
  IREE_ASSERT_OK(
      loom_test_constant_build(&source_region_builder, loom_attr_i64(42),
                               loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                               LOOM_LOCATION_UNKNOWN, &source_op));
  source_op->flags |= LOOM_OP_FLAG_LEADING_BLANK_LINE;
  const iree_string_view_t op_comments[] = {
      IREE_SV("operation heading"),
      IREE_SV("operation detail"),
  };
  IREE_ASSERT_OK(loom_module_attach_op_comments(source_, source_op, op_comments,
                                                IREE_ARRAYSIZE(op_comments)));

  loom_ir_remap_t remap = InitializeRemap();
  loom_region_t* cloned_region = nullptr;
  IREE_ASSERT_OK(loom_ir_clone_region(&target_builder_, source_region, &remap,
                                      &cloned_region));

  const loom_block_t* cloned_block =
      loom_region_const_entry_block(cloned_region);
  EXPECT_TRUE(iree_any_bit_set(cloned_block->flags,
                               LOOM_BLOCK_FLAG_LEADING_BLANK_LINE));
  iree_host_size_t cloned_block_comment_count = 0;
  const iree_string_view_t* cloned_block_comments = loom_module_block_comments(
      target_, cloned_block, &cloned_block_comment_count);
  ASSERT_EQ(cloned_block_comment_count, IREE_ARRAYSIZE(block_comments));
  EXPECT_TRUE(
      iree_string_view_equal(cloned_block_comments[0], block_comments[0]));

  const loom_op_t* cloned_op = loom_block_const_op(cloned_block, 0);
  EXPECT_TRUE(
      iree_any_bit_set(cloned_op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE));
  iree_host_size_t cloned_op_comment_count = 0;
  const iree_string_view_t* cloned_op_comments =
      loom_module_op_comments(target_, cloned_op, &cloned_op_comment_count);
  ASSERT_EQ(cloned_op_comment_count, IREE_ARRAYSIZE(op_comments));
  EXPECT_TRUE(iree_string_view_equal(cloned_op_comments[0], op_comments[0]));
  EXPECT_TRUE(iree_string_view_equal(cloned_op_comments[1], op_comments[1]));
}

TEST_F(MaterializeTest, RejectsCrossModuleCloneWithUnmappedSuccessor) {
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 2, &source_region));
  loom_block_t* source_entry = loom_region_block(source_region, 0);
  loom_block_t* source_dest = loom_region_block(source_region, 1);
  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena, source_entry,
                          &source_region_builder);
  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(loom_test_br_build(&source_region_builder, source_dest,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));

  loom_ir_remap_t remap = InitializeRemap();
  loom_op_t* cloned_op = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_ir_clone_op(&target_builder_, branch_op, &remap, &cloned_op));
  EXPECT_EQ(cloned_op, nullptr);
  EXPECT_EQ(loom_module_block(target_)->op_count, 0u);
}

TEST_F(MaterializeTest, RejectsCloningRegionIntoItselfBeforeMutation) {
  loom_ir_remap_t remap =
      InitializeSameModuleRemap(/*allow_unmapped_values=*/true);
  const uint16_t original_block_count = source_->body->block_count;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_ir_clone_region_blocks(&source_builder_, source_->body,
                                  source_->body, original_block_count, &remap));

  EXPECT_EQ(source_->body->block_count, original_block_count);
}

TEST_F(MaterializeTest, RejectsSplicingBeforeExistingEntryBeforeMutation) {
  loom_ir_remap_t remap = InitializeRemap();
  loom_region_t* target_region = target_->body;
  loom_block_t* original_entry = loom_region_entry_block(target_region);
  const uint16_t original_block_count = target_region->block_count;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_ir_clone_region_blocks(&target_builder_, source_->body,
                                  target_region, 0, &remap));

  EXPECT_EQ(target_region->block_count, original_block_count);
  EXPECT_EQ(loom_region_entry_block(target_region), original_entry);
}

TEST_F(MaterializeTest, ClonesStandaloneSuccessorThroughExplicitBlockMap) {
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 2, &source_region));
  loom_block_t* source_entry = loom_region_block(source_region, 0);
  loom_block_t* source_dest = loom_region_block(source_region, 1);
  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena, source_entry,
                          &source_region_builder);
  loom_op_t* branch_op = nullptr;
  IREE_ASSERT_OK(loom_test_br_build(&source_region_builder, source_dest,
                                    LOOM_LOCATION_UNKNOWN, &branch_op));

  loom_region_t* target_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(target_, 2, &target_region));
  loom_block_t* target_entry = loom_region_block(target_region, 0);
  loom_block_t* target_dest = loom_region_block(target_region, 1);
  loom_builder_t target_region_builder = {};
  loom_builder_initialize(target_, &target_->arena, target_entry,
                          &target_region_builder);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_block(&remap, source_dest, target_dest));
  loom_op_t* cloned_op = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_region_builder, branch_op, &remap, &cloned_op));

  ASSERT_TRUE(loom_test_br_isa(cloned_op));
  EXPECT_EQ(loom_test_br_dest(cloned_op), target_dest);
}

TEST_F(MaterializeTest, ClonesOperandDictOps) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* source_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(0),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &source_input_op));
  loom_value_id_t source_input = loom_test_constant_result(source_input_op);
  loom_op_t* source_alpha_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(1),
                                          i32_type, LOOM_LOCATION_UNKNOWN,
                                          &source_alpha_op));
  loom_value_id_t source_alpha = loom_test_constant_result(source_alpha_op);
  loom_op_t* source_beta_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(2),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &source_beta_op));
  loom_value_id_t source_beta = loom_test_constant_result(source_beta_op);

  loom_string_id_t source_beta_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("beta"), &source_beta_name));
  loom_string_id_t source_alpha_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("alpha"), &source_alpha_name));
  loom_named_value_t source_params[] = {
      {
          /*.name_id=*/source_beta_name,
          /*.reserved=*/0,
          /*.value_id=*/source_beta,
      },
      {
          /*.name_id=*/source_alpha_name,
          /*.reserved=*/0,
          /*.value_id=*/source_alpha,
      },
  };
  loom_op_t* source_dict_op = nullptr;
  IREE_ASSERT_OK(loom_test_operand_dict_build(
      &source_builder_, source_input, source_params,
      IREE_ARRAYSIZE(source_params), f32_type, LOOM_LOCATION_UNKNOWN,
      &source_dict_op));

  loom_op_t* target_input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(0),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &target_input_op));
  loom_value_id_t target_input = loom_test_constant_result(target_input_op);
  loom_op_t* target_alpha_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(1),
                                          i32_type, LOOM_LOCATION_UNKNOWN,
                                          &target_alpha_op));
  loom_value_id_t target_alpha = loom_test_constant_result(target_alpha_op);
  loom_op_t* target_beta_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(2),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &target_beta_op));
  loom_value_id_t target_beta = loom_test_constant_result(target_beta_op);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_input, target_input));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_alpha, target_alpha));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_beta, target_beta));
  loom_op_t* cloned_op = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, source_dict_op, &remap, &cloned_op));

  ASSERT_TRUE(loom_test_operand_dict_isa(cloned_op));
  EXPECT_EQ(loom_test_operand_dict_input(cloned_op), target_input);
  loom_value_slice_t cloned_params = loom_test_operand_dict_params(cloned_op);
  ASSERT_EQ(cloned_params.count, 2u);
  EXPECT_EQ(cloned_params.values[0], target_alpha);
  EXPECT_EQ(cloned_params.values[1], target_beta);

  loom_named_attr_slice_t names = loom_test_operand_dict_param_names(cloned_op);
  ASSERT_EQ(names.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&target_->strings, names.entries[0].name_id),
      IREE_SV("alpha")));
  EXPECT_EQ(loom_attr_as_i64(names.entries[0].value), 0);
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&target_->strings, names.entries[1].name_id),
      IREE_SV("beta")));
  EXPECT_EQ(loom_attr_as_i64(names.entries[1].value), 1);
}

TEST_F(MaterializeTest, ClonesBlockOpsCanOmitTerminators) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 1, &source_region));
  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena,
                          loom_region_entry_block(source_region),
                          &source_region_builder);
  loom_op_t* constant_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_region_builder,
                                          loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &constant_op));
  loom_value_id_t value = loom_test_constant_result(constant_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&source_region_builder, &value, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));

  loom_ir_remap_t remap = InitializeRemap();
  loom_ir_clone_block_options_t options = {
      /*.omit_terminators=*/true,
  };
  IREE_ASSERT_OK(loom_ir_clone_block_ops(&target_builder_,
                                         loom_region_entry_block(source_region),
                                         &remap, &options));

  loom_block_t* target_block = loom_module_block(target_);
  ASSERT_EQ(target_block->op_count, 1u);
  ASSERT_TRUE(loom_test_constant_isa(loom_block_op(target_block, 0)));
}

TEST_F(MaterializeTest, MovesBlockOpsAndRemapsCapturedBlockArgs) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);

  loom_op_t* replacement_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_f64(1.0),
                                          f32_type, LOOM_LOCATION_UNKNOWN,
                                          &replacement_op));
  loom_value_id_t replacement = loom_test_constant_result(replacement_op);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(0),
                                          tile_type, LOOM_LOCATION_UNKNOWN,
                                          &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&source_builder_, &input, 1, tile_type,
                                     nullptr, 0, LOOM_LOCATION_UNKNOWN,
                                     &map_op));
  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&source_builder_, map_op, body);
  loom_value_id_t element = loom_region_entry_arg_id(body, 0);
  loom_op_t* neg_op = nullptr;
  IREE_ASSERT_OK(loom_test_neg_build(&source_builder_, element, f32_type,
                                     LOOM_LOCATION_UNKNOWN, &neg_op));
  loom_value_id_t negated = loom_test_neg_result(neg_op);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&source_builder_, &negated, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&source_builder_, saved_ip);

  loom_op_t* sentinel_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&source_builder_, &replacement, 1,
                                     LOOM_LOCATION_UNKNOWN, &sentinel_op));

  loom_rewriter_t rewriter = {};
  loom_rewriter_initialize(&rewriter, source_, &remap_arena_);
  loom_ir_remap_t remap =
      InitializeSameModuleRemap(/*allow_unmapped_values=*/true);
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, element, replacement));
  loom_ir_move_block_options_t options = {
      /*.omit_terminators=*/true,
  };
  loom_availability_analysis_t availability = InitializeAvailability();
  IREE_ASSERT_OK(loom_ir_move_block_ops_before(&rewriter, &availability,
                                               loom_region_entry_block(body),
                                               sentinel_op, &remap, &options));

  EXPECT_EQ(neg_op->parent_block, loom_module_block(source_));
  EXPECT_EQ(neg_op->parent_op, nullptr);
  EXPECT_EQ(loom_test_neg_input(neg_op), replacement);
  EXPECT_EQ(neg_op->next_op, sentinel_op);
  ASSERT_EQ(loom_region_entry_block(body)->op_count, 1u);
  EXPECT_EQ(loom_region_entry_block(body)->first_op, yield_op);

  IREE_ASSERT_OK(loom_rewriter_erase(&rewriter, map_op));
  EXPECT_EQ(loom_module_value(source_, negated)->use_count, 0u);
  EXPECT_TRUE(iree_any_bit_set(map_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(neg_op->flags, LOOM_OP_FLAG_DEAD));
  loom_rewriter_deinitialize(&rewriter);
}

TEST_F(MaterializeTest, MovesBlockOpsAndRemapsDynamicResultTypes) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t input_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 1, &source_region));
  loom_block_t* source_block = loom_region_entry_block(source_region);
  loom_value_id_t source_dim = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(source_, index_type, &source_dim));
  IREE_ASSERT_OK(loom_block_add_arg(source_, source_block, source_dim));

  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena, source_block,
                          &source_region_builder);
  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_region_builder,
                                          loom_attr_i64(0), input_type,
                                          LOOM_LOCATION_UNKNOWN, &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);
  loom_type_t result_types[] = {
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_dim), 0),
      index_type,
  };
  loom_op_t* deflate_op = nullptr;
  IREE_ASSERT_OK(loom_test_deflate_build(
      &source_region_builder, input, result_types, IREE_ARRAYSIZE(result_types),
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &deflate_op));

  loom_op_t* target_dim_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(8),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &target_dim_op));
  loom_value_id_t target_dim = loom_test_constant_result(target_dim_op);
  loom_op_t* sentinel_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&source_builder_, &target_dim, 1,
                                     LOOM_LOCATION_UNKNOWN, &sentinel_op));

  loom_rewriter_t rewriter = {};
  loom_rewriter_initialize(&rewriter, source_, &remap_arena_);
  loom_ir_remap_t remap =
      InitializeSameModuleRemap(/*allow_unmapped_values=*/true);
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_dim, target_dim));
  loom_availability_analysis_t availability = InitializeAvailability();
  IREE_ASSERT_OK(loom_ir_move_block_ops_before(
      &rewriter, &availability, source_block, sentinel_op, &remap,
      /*options=*/nullptr));

  EXPECT_EQ(source_block->op_count, 0u);
  EXPECT_EQ(input_op->parent_block, loom_module_block(source_));
  EXPECT_EQ(deflate_op->parent_block, loom_module_block(source_));
  EXPECT_EQ(deflate_op->next_op, sentinel_op);
  loom_value_slice_t moved_results = loom_test_deflate_results(deflate_op);
  ASSERT_EQ(moved_results.count, 2u);
  loom_type_t moved_output_type =
      loom_module_value_type(source_, moved_results.values[0]);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(moved_output_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(moved_output_type, 0), target_dim);
  loom_rewriter_deinitialize(&rewriter);
}

TEST_F(MaterializeTest, MovesBlockOpsAndRemapsPredicateAttrs) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 1, &source_region));
  loom_block_t* source_block = loom_region_entry_block(source_region);
  loom_value_id_t source_dim = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(source_, index_type, &source_dim));
  IREE_ASSERT_OK(loom_block_add_arg(source_, source_block, source_dim));

  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena, source_block,
                          &source_region_builder);
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_MUL,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{(int64_t)source_dim, 16},
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_test_assume_build(&source_region_builder, &source_dim, 1,
                                        &predicate, 1, &index_type, 1,
                                        LOOM_LOCATION_UNKNOWN, &assume_op));

  loom_op_t* target_dim_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(8),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &target_dim_op));
  loom_value_id_t target_dim = loom_test_constant_result(target_dim_op);
  loom_op_t* sentinel_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&source_builder_, &target_dim, 1,
                                     LOOM_LOCATION_UNKNOWN, &sentinel_op));

  loom_rewriter_t rewriter = {};
  loom_rewriter_initialize(&rewriter, source_, &remap_arena_);
  loom_ir_remap_t remap =
      InitializeSameModuleRemap(/*allow_unmapped_values=*/true);
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_dim, target_dim));
  loom_availability_analysis_t availability = InitializeAvailability();
  IREE_ASSERT_OK(loom_ir_move_block_ops_before(
      &rewriter, &availability, source_block, sentinel_op, &remap,
      /*options=*/nullptr));

  EXPECT_EQ(source_block->op_count, 0u);
  EXPECT_EQ(assume_op->parent_block, loom_module_block(source_));
  EXPECT_EQ(loom_test_assume_values(assume_op).values[0], target_dim);
  loom_attribute_t predicates = loom_op_attrs(assume_op)[0];
  ASSERT_EQ(predicates.kind, LOOM_ATTR_PREDICATE_LIST);
  ASSERT_EQ(predicates.count, 1u);
  EXPECT_EQ(predicates.predicate_list[0].args[0], (int64_t)target_dim);
  EXPECT_FALSE(
      loom_value_has_attribute_uses(loom_module_value(source_, source_dim)));
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(source_, target_dim)));
  EXPECT_EQ(assume_op->next_op, sentinel_op);
  loom_rewriter_deinitialize(&rewriter);
}

TEST_F(MaterializeTest, ClonedPredicateOwnersRemainModuleLocal) {
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* source_constant = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_builder_, loom_attr_i64(1),
                                          index, LOOM_LOCATION_UNKNOWN,
                                          &source_constant));
  const loom_value_id_t source_value =
      loom_test_constant_result(source_constant);
  loom_predicate_t predicate = {
      LOOM_PREDICATE_EQ, 2, {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST}, {},
      {source_value, 1},
  };
  loom_op_t* source_owner = nullptr;
  IREE_ASSERT_OK(loom_test_assume_build(&source_builder_, &source_value, 1,
                                        &predicate, 1, &index, 1,
                                        LOOM_LOCATION_UNKNOWN, &source_owner));
  loom_op_t* target_constant = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(1),
                                          index, LOOM_LOCATION_UNKNOWN,
                                          &target_constant));
  const loom_value_id_t target_value =
      loom_test_constant_result(target_constant);
  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_value, target_value));
  loom_op_t* target_owner = nullptr;
  IREE_ASSERT_OK(
      loom_ir_clone_op(&target_builder_, source_owner, &remap, &target_owner));
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(source_, source_value)));
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(target_, target_value)));
  loom_op_t* replacement_constant = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&target_builder_, loom_attr_i64(1),
                                          index, LOOM_LOCATION_UNKNOWN,
                                          &replacement_constant));
  const loom_value_id_t replacement =
      loom_test_constant_result(replacement_constant);
  IREE_ASSERT_OK(
      loom_value_replace_all_uses_with(target_, target_value, replacement));
  EXPECT_EQ(loom_op_const_attrs(source_owner)[0].predicate_list[0].args[0],
            source_value);
  EXPECT_EQ(loom_op_const_attrs(target_owner)[0].predicate_list[0].args[0],
            replacement);
  EXPECT_FALSE(
      loom_value_has_attribute_uses(loom_module_value(target_, target_value)));
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(target_, replacement)));
  IREE_ASSERT_OK(loom_op_erase(target_, target_owner));
  EXPECT_FALSE(
      loom_value_has_attribute_uses(loom_module_value(target_, replacement)));
  EXPECT_TRUE(
      loom_value_has_attribute_uses(loom_module_value(source_, source_value)));
}

TEST_F(MaterializeTest, RejectsMoveWithUnavailableRemappedCaptures) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t input_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_region_t* source_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 1, &source_region));
  loom_block_t* source_block = loom_region_entry_block(source_region);
  loom_value_id_t source_dim = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(source_, index_type, &source_dim));
  IREE_ASSERT_OK(loom_block_add_arg(source_, source_block, source_dim));

  loom_builder_t source_region_builder = {};
  loom_builder_initialize(source_, &source_->arena, source_block,
                          &source_region_builder);
  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&source_region_builder,
                                          loom_attr_i64(0), input_type,
                                          LOOM_LOCATION_UNKNOWN, &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);
  loom_type_t result_types[] = {
      loom_type_shaped_1d(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_dim), 0),
      index_type,
  };
  loom_op_t* deflate_op = nullptr;
  IREE_ASSERT_OK(loom_test_deflate_build(
      &source_region_builder, input, result_types, IREE_ARRAYSIZE(result_types),
      nullptr, 0, LOOM_LOCATION_UNKNOWN, &deflate_op));

  loom_op_t* sentinel_op = nullptr;
  IREE_ASSERT_OK(loom_test_use_build(&source_builder_, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &sentinel_op));

  loom_rewriter_t rewriter = {};
  loom_rewriter_initialize(&rewriter, source_, &remap_arena_);
  loom_ir_remap_t remap =
      InitializeSameModuleRemap(/*allow_unmapped_values=*/true);
  loom_availability_analysis_t availability = InitializeAvailability();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_ir_move_block_ops_before(&rewriter, &availability, source_block,
                                    sentinel_op, &remap, /*options=*/nullptr));

  EXPECT_EQ(source_block->op_count, 2u);
  EXPECT_EQ(input_op->parent_block, source_block);
  EXPECT_EQ(deflate_op->parent_block, source_block);
  loom_rewriter_deinitialize(&rewriter);
}

TEST_F(MaterializeTest, HintPresenceTracksLastSourceAcrossRebuildAndErase) {
  loom_op_t* outer = BuildRegionOwner(&source_builder_);
  loom_region_t* outer_region = loom_test_block_args_body(outer);
  loom_builder_enter_region(&source_builder_, outer, outer_region);
  loom_op_t* inner = BuildRegionOwner(&source_builder_);
  loom_region_t* inner_region = loom_test_block_args_body(inner);
  const auto outer_ip =
      loom_builder_enter_region(&source_builder_, inner, inner_region);
  EXPECT_FALSE(loom_op_regions_have_hints(outer));
  loom_op_t* first_hint = BuildHint(&source_builder_);
  loom_op_t* last_hint = BuildHint(&source_builder_);
  loom_module_record_op_summaries(source_, first_hint);
  loom_module_update_op_direct_summaries(
      source_, first_hint, first_hint->traits, first_hint->traits);
  EXPECT_EQ(inner_region->hint_source_count, 2u);
  EXPECT_EQ(outer_region->hint_source_count, 1u);
  EXPECT_EQ(source_->body->hint_source_count, 1u);
  EXPECT_TRUE(loom_op_regions_have_hints(outer));
  EXPECT_FALSE(loom_op_regions_have_hints(first_hint));

  loom_builder_restore(&source_builder_, outer_ip);
  loom_op_t* outer_hint = BuildHint(&source_builder_);
  EXPECT_EQ(outer_region->hint_source_count, 2u);
  EXPECT_EQ(source_->body->hint_source_count, 1u);
  IREE_ASSERT_OK(loom_op_erase(source_, first_hint));
  EXPECT_EQ(inner_region->hint_source_count, 1u);
  EXPECT_EQ(outer_region->hint_source_count, 2u);

  // Rebuilding is idempotent and preserves one contribution per child region,
  // not one contribution per transitive hint.
  for (int iteration = 0; iteration < 2; ++iteration) {
    IREE_ASSERT_OK(loom_module_compute_uses(source_));
    EXPECT_EQ(inner_region->hint_source_count, 1u);
    EXPECT_EQ(outer_region->hint_source_count, 2u);
    EXPECT_EQ(source_->body->hint_source_count, 1u);
  }
  IREE_ASSERT_OK(loom_op_erase(source_, last_hint));
  EXPECT_FALSE(loom_op_regions_have_hints(inner));
  EXPECT_TRUE(loom_op_regions_have_hints(outer));
  EXPECT_EQ(outer_region->hint_source_count, 1u);
  IREE_ASSERT_OK(loom_op_erase(source_, outer_hint));
  EXPECT_FALSE(loom_op_regions_have_hints(outer));
  EXPECT_FALSE(loom_region_has_hints(source_->body));
}

TEST_F(MaterializeTest, ClonesNestedHintPresenceBeforeOwnerFinalization) {
  loom_op_t* outer = BuildRegionOwner(&source_builder_);
  loom_builder_enter_region(&source_builder_, outer,
                            loom_test_block_args_body(outer));
  loom_op_t* inner = BuildRegionOwner(&source_builder_);
  loom_builder_enter_region(&source_builder_, inner,
                            loom_test_block_args_body(inner));
  BuildHint(&source_builder_);
  BuildHint(&source_builder_);

  loom_ir_remap_t remap = InitializeRemap();
  loom_op_t* clone = nullptr;
  IREE_ASSERT_OK(loom_ir_clone_op(&target_builder_, outer, &remap, &clone));
  loom_region_t* cloned_region = loom_test_block_args_body(clone);
  loom_op_t* cloned_inner = loom_region_entry_block(cloned_region)->first_op;
  EXPECT_EQ(cloned_region->hint_source_count, 1u);
  EXPECT_EQ(loom_test_block_args_body(cloned_inner)->hint_source_count, 2u);
  EXPECT_EQ(target_->body->hint_source_count, 1u);
  IREE_ASSERT_OK(loom_op_erase(source_, outer));
  EXPECT_FALSE(loom_region_has_hints(source_->body));
  EXPECT_TRUE(loom_op_regions_have_hints(clone));
  IREE_ASSERT_OK(loom_module_compute_uses(target_));
  EXPECT_EQ(cloned_region->hint_source_count, 1u);
  EXPECT_EQ(loom_test_block_args_body(cloned_inner)->hint_source_count, 2u);
  IREE_ASSERT_OK(loom_op_erase(target_, clone));
  EXPECT_FALSE(loom_region_has_hints(target_->body));
}

TEST_F(MaterializeTest, MovesNestedAndWholeRegionHintSources) {
  loom_op_t* source_owner = BuildRegionOwner(&source_builder_);
  loom_op_t* target_owner = BuildRegionOwner(&source_builder_);
  loom_region_t* source_region = loom_test_block_args_body(source_owner);
  loom_region_t* target_region = loom_test_block_args_body(target_owner);
  loom_builder_enter_region(&source_builder_, source_owner, source_region);
  loom_op_t* direct_hint = BuildHint(&source_builder_);
  loom_op_t* nested = BuildRegionOwner(&source_builder_);
  loom_builder_enter_region(&source_builder_, nested,
                            loom_test_block_args_body(nested));
  BuildHint(&source_builder_);
  BuildHint(&source_builder_);
  EXPECT_EQ(source_region->hint_source_count, 2u);
  EXPECT_FALSE(loom_region_has_hints(target_region));

  loom_rewriter_t rewriter = {};
  loom_rewriter_initialize(&rewriter, source_, &remap_arena_);
  IREE_ASSERT_OK(loom_rewriter_move_to_block_end(
      &rewriter, nested, loom_region_entry_block(target_region), target_owner));
  EXPECT_EQ(source_region->hint_source_count, 1u);
  EXPECT_EQ(target_region->hint_source_count, 1u);
  EXPECT_EQ(source_->body->hint_source_count, 2u);
  EXPECT_EQ(loom_test_block_args_body(nested)->hint_source_count, 2u);

  loom_block_t* moved_entry = nullptr;
  IREE_ASSERT_OK(loom_rewriter_move_region_blocks(
      &rewriter, source_region, source_owner, target_region, 1, target_owner,
      &moved_entry));
  EXPECT_FALSE(loom_region_has_hints(source_region));
  EXPECT_EQ(target_region->hint_source_count, 2u);
  EXPECT_EQ(source_->body->hint_source_count, 1u);
  EXPECT_EQ(direct_hint->parent_block, moved_entry);
  IREE_ASSERT_OK(loom_op_erase(source_, nested));
  EXPECT_EQ(target_region->hint_source_count, 1u);
  IREE_ASSERT_OK(loom_op_erase(source_, direct_hint));
  EXPECT_FALSE(loom_region_has_hints(target_region));
  EXPECT_FALSE(loom_region_has_hints(source_->body));
  loom_rewriter_deinitialize(&rewriter);
}

TEST_F(MaterializeTest, MovesHintSubtreeFromParentlessDetachedRegion) {
  loom_op_t* insertion_target = BuildRegionOwner(&source_builder_);
  loom_region_t* detached_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(source_, 1, &detached_region));
  loom_builder_t detached_builder = {};
  loom_builder_initialize(source_, &source_->arena,
                          loom_region_entry_block(detached_region),
                          &detached_builder);
  loom_op_t* nested = BuildRegionOwner(&detached_builder);
  loom_builder_enter_region(&detached_builder, nested,
                            loom_test_block_args_body(nested));
  BuildHint(&detached_builder);
  EXPECT_TRUE(loom_region_has_hints(detached_region));
  EXPECT_FALSE(loom_region_has_hints(source_->body));

  loom_rewriter_t rewriter = {};
  loom_rewriter_initialize(&rewriter, source_, &remap_arena_);
  IREE_ASSERT_OK(
      loom_rewriter_move_before(&rewriter, nested, insertion_target));
  EXPECT_FALSE(loom_region_has_hints(detached_region));
  EXPECT_TRUE(loom_region_has_hints(source_->body));
  EXPECT_EQ(loom_test_block_args_body(nested)->hint_source_count, 1u);
  IREE_ASSERT_OK(loom_op_erase(source_, nested));
  EXPECT_FALSE(loom_region_has_hints(source_->body));
  loom_rewriter_deinitialize(&rewriter);
}

}  // namespace
}  // namespace loom
