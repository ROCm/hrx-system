// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/availability.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class AvailabilityTest : public ::testing::Test {
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
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
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

  void PrepareAvailabilityAnalysis() {
    IREE_ASSERT_OK(loom_module_compute_uses(module_));
    IREE_ASSERT_OK(loom_availability_analysis_initialize(
        module_, &analysis_arena_, &availability_));
  }

  void InternTypeAndFindId(loom_type_t type, loom_type_id_t* out_type_id) {
    *out_type_id = LOOM_TYPE_ID_INVALID;
    loom_type_t interned_type = {0};
    IREE_ASSERT_OK(loom_module_intern_type(module_, type, &interned_type));
    for (iree_host_size_t i = 0; i < module_->types.count; ++i) {
      if (loom_type_equal(module_->types.entries[i], interned_type)) {
        *out_type_id = (loom_type_id_t)i;
        return;
      }
    }
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  iree_arena_allocator_t analysis_arena_;
  loom_availability_analysis_t availability_;
  std::vector<loom_builder_ip_t> saved_insertions_;
};

TEST_F(AvailabilityTest, PredicateAttrRejectsValueDefinedAfterInsertion) {
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

  PrepareAvailabilityAnalysis();

  bool available = true;
  IREE_ASSERT_OK(loom_availability_op_captures_are_available_before_op(
      &availability_, /*moving_root_op=*/NULL, anchor_op, assume_op,
      &available));
  EXPECT_FALSE(available);
}

TEST_F(AvailabilityTest, PredicateAttrAllowsValueDefinedInsideMovingSubtree) {
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
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_EQ,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_VALUE},
      /*.reserved=*/{},
      /*.args=*/{(int64_t)element, (int64_t)element},
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_test_assume_build(&builder_, &element, 1, &predicate, 1,
                                        &i32_type, 1, LOOM_LOCATION_UNKNOWN,
                                        &assume_op));
  LeaveRegion();

  PrepareAvailabilityAnalysis();

  bool available = false;
  IREE_ASSERT_OK(loom_availability_op_captures_are_available_before_op(
      &availability_, map_op, anchor_op, assume_op, &available));
  EXPECT_TRUE(available);
}

TEST_F(AvailabilityTest, TypeAttrRejectsUnavailableDynamicDimension) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* input_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &input_op));
  loom_value_id_t input = loom_test_constant_result(input_op);

  loom_op_t* anchor_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &anchor_op));

  loom_op_t* late_dim_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(4),
                                          index_type, LOOM_LOCATION_UNKNOWN,
                                          &late_dim_op));
  loom_value_id_t late_dim = loom_test_constant_result(late_dim_op);

  loom_type_t captured_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_dynamic(late_dim), 0);
  loom_type_id_t captured_type_id = LOOM_TYPE_ID_INVALID;
  InternTypeAndFindId(captured_type, &captured_type_id);
  ASSERT_NE(captured_type_id, LOOM_TYPE_ID_INVALID);

  loom_string_id_t type_key = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("type"), &type_key));
  loom_named_attr_t attrs[1] = {{
      /*.name_id=*/type_key,
      /*.reserved=*/{},
      /*.value=*/loom_attr_type(captured_type_id),
  }};
  loom_attribute_t dict = {0};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_, loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
      &dict));

  loom_op_t* attrs_op = nullptr;
  IREE_ASSERT_OK(loom_test_attrs_build(&builder_, input,
                                       loom_attr_as_dict(dict), i32_type,
                                       LOOM_LOCATION_UNKNOWN, &attrs_op));

  PrepareAvailabilityAnalysis();

  bool available = true;
  IREE_ASSERT_OK(loom_availability_op_captures_are_available_before_op(
      &availability_, /*moving_root_op=*/NULL, anchor_op, attrs_op,
      &available));
  EXPECT_FALSE(available);
}

}  // namespace
}  // namespace loom
