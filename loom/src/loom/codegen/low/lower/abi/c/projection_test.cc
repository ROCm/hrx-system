// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/abi/c/projection.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/registers.h"

namespace loom {
namespace {

class CAbiProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static loom_low_c_abi_projection_target_t MakeTarget(loom_type_t bits32,
                                                       loom_type_t bits64) {
    return {
        .ops =
            {
                .register_descriptor_set_stable_id =
                    loom_low_register_type_descriptor_set_stable_id(bits32),
                .bits32_register_class =
                    loom_low_register_type_class_id(bits32),
                .bits64_register_class =
                    loom_low_register_type_class_id(bits64),
                .address_register_class =
                    loom_low_register_type_class_id(bits64),
                .address_byte_length = 8,
                .byte_order = LOOM_LOW_C_ABI_BYTE_ORDER_LITTLE,
                .load_u8 = 1,
                .load_u16 = 2,
                .load_u32 = 3,
                .load_u64 = 4,
                .store_u8 = 5,
                .store_u16 = 6,
                .store_u32 = 7,
                .store_u64 = 8,
                .add_address = 9,
                .mask_u32 = 10,
                .widen_u32 = 11,
                .truncate_u64 = 12,
                .shift_left_u64 = 13,
                .shift_right_u64 = 14,
                .or_u64 = 15,
                .constant_u64 = 16,
                .memory_offset_name = "offset",
                .mask_name = "mask",
                .shift_name = "shift",
                .constant_name = "constant",
            },
    };
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(CAbiProjectionTest, FinalizesDirectScalarAndPointerObjects) {
  const loom_type_t bits32 = loom_low_register_type(1, 1, 1);
  const loom_type_t bits64 = loom_low_register_type(1, 2, 1);
  const loom_c_abi_object_t objects[] = {
      {
          .byte_length = 4,
          .byte_alignment = 4,
          .logical_offset = 0,
          .logical_count = 1,
          .leaf_offset = 0,
          .leaf_count = 1,
      },
      {
          .byte_length = 8,
          .byte_alignment = 8,
          .logical_offset = 1,
          .logical_count = 2,
          .leaf_offset = 1,
          .leaf_count = 1,
      },
  };
  const loom_c_abi_leaf_t leaves[] = {
      {.byte_length = 4, .logical_offset = 0, .logical_count = 1},
      {
          .byte_length = 8,
          .logical_offset = 1,
          .logical_count = 2,
          .flags = LOOM_C_ABI_LEAF_FLAG_POINTER,
      },
  };
  const int64_t roots[] = {0, 1};
  const loom_c_abi_signature_layout_t layout = {
      .signature = {.arguments = loom_make_i64_array(roots, 2)},
      .arguments = objects,
      .argument_leaves = leaves,
      .argument_leaf_count = 2,
      .argument_logical_count = 3,
  };
  const loom_type_t logical_types[] = {bits32, bits64, bits64};
  loom_low_c_abi_projection_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_low_c_abi_projection_plan_allocate(
      &layout, logical_types, 3, /*logical_result_types=*/nullptr,
      /*logical_result_count=*/0, /*argument_carrier_capacity=*/2,
      /*result_carrier_capacity=*/0, &arena_, &plan));
  EXPECT_EQ(loom_low_c_abi_projection_append_argument_carrier(
                plan, /*object_index=*/0, /*byte_offset=*/0,
                /*storage_byte_length=*/8),
            0);
  EXPECT_EQ(loom_low_c_abi_projection_append_argument_carrier(
                plan, /*object_index=*/1, /*byte_offset=*/0,
                /*storage_byte_length=*/8),
            1);

  const loom_type_t physical_types[] = {bits64, bits64};
  const loom_low_c_abi_projection_target_t target = MakeTarget(bits32, bits64);
  iree_string_view_t reason;
  ASSERT_TRUE(loom_low_c_abi_projection_plan_finalize(
      plan, &target, physical_types, 2, /*physical_result_types=*/nullptr,
      /*physical_result_count=*/0, &reason));

  EXPECT_TRUE(plan->arguments[0].has_direct_projection);
  EXPECT_TRUE(plan->arguments[1].has_direct_projection);
  EXPECT_EQ(plan->argument_leaves[0].physical_index, 0);
  EXPECT_EQ(plan->argument_leaves[1].physical_index, 1);
  EXPECT_EQ(plan->argument_leaves[0].load_descriptor, 3u);
  EXPECT_EQ(plan->argument_leaves[1].load_descriptor, 4u);
  EXPECT_EQ(plan->argument_carriers[0].leaf_count, 1);
  EXPECT_EQ(plan->argument_carriers[1].leaf_count, 1);
}

TEST_F(CAbiProjectionTest, RetainsAddressTransportsAndIndirectResult) {
  const loom_type_t bits32 = loom_low_register_type(1, 1, 1);
  const loom_type_t bits64 = loom_low_register_type(1, 2, 1);
  const loom_c_abi_object_t argument = {
      .byte_length = 4,
      .byte_alignment = 4,
      .logical_count = 1,
      .leaf_count = 1,
  };
  const loom_c_abi_object_t result = {
      .byte_length = 4,
      .byte_alignment = 4,
      .logical_count = 1,
      .leaf_count = 1,
  };
  const loom_c_abi_leaf_t argument_leaf = {
      .byte_length = 4,
      .logical_count = 1,
  };
  const loom_c_abi_leaf_t result_leaf = {
      .byte_length = 4,
      .logical_count = 1,
  };
  const int64_t argument_root = 0;
  const int64_t result_root = 0;
  const loom_c_abi_signature_layout_t layout = {
      .signature =
          {
              .arguments = loom_make_i64_array(&argument_root, 1),
              .results = loom_make_i64_array(&result_root, 1),
          },
      .arguments = &argument,
      .results = &result,
      .argument_leaves = &argument_leaf,
      .result_leaves = &result_leaf,
      .argument_leaf_count = 1,
      .result_leaf_count = 1,
      .argument_logical_count = 1,
      .result_logical_count = 1,
  };
  const loom_type_t logical_type = bits32;
  loom_low_c_abi_projection_plan_t* plan = nullptr;
  IREE_ASSERT_OK(loom_low_c_abi_projection_plan_allocate(
      &layout, &logical_type, 1, &logical_type, 1,
      /*argument_carrier_capacity=*/2, /*result_carrier_capacity=*/1, &arena_,
      &plan));
  EXPECT_EQ(loom_low_c_abi_projection_append_argument_carrier(
                plan, /*object_index=*/0, /*byte_offset=*/0,
                /*storage_byte_length=*/8),
            0);
  loom_low_c_abi_projection_set_argument_address(plan, 0);
  EXPECT_EQ(loom_low_c_abi_projection_append_argument_carrier(
                plan, LOOM_LOW_C_ABI_INDEX_NONE, /*byte_offset=*/0,
                /*storage_byte_length=*/8),
            1);
  EXPECT_EQ(loom_low_c_abi_projection_append_result_carrier(
                plan, LOOM_LOW_C_ABI_INDEX_NONE, /*byte_offset=*/0,
                /*storage_byte_length=*/8),
            0);
  loom_low_c_abi_projection_set_indirect_result(
      plan, /*object_index=*/0, /*address_argument=*/1, /*echo_result=*/0);

  const loom_type_t physical_arguments[] = {bits64, bits64};
  const loom_type_t physical_result = bits64;
  const loom_low_c_abi_projection_target_t target = MakeTarget(bits32, bits64);
  iree_string_view_t reason;
  ASSERT_TRUE(loom_low_c_abi_projection_plan_finalize(
      plan, &target, physical_arguments, 2, &physical_result, 1, &reason));

  EXPECT_EQ(plan->arguments[0].transport,
            LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS);
  EXPECT_FALSE(plan->arguments[0].has_direct_projection);
  EXPECT_EQ(plan->indirect_result_object, 0);
  EXPECT_EQ(plan->indirect_result_argument, 1);
  EXPECT_EQ(plan->indirect_result_echo, 0);
  EXPECT_EQ(plan->results[0].transport,
            LOOM_LOW_C_ABI_OBJECT_TRANSPORT_ADDRESS);
}

}  // namespace
}  // namespace loom
