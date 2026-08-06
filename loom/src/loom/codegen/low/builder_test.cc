// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/builder.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class LowBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, /*archive=*/nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Block pool backing the test module arena.
  iree_arena_block_pool_t block_pool_;
  // Finalized context shared by the test module.
  loom_context_t context_;
  // Module owning recursively structural register types.
  loom_module_t* module_ = nullptr;
};

TEST_F(LowBuilderTest, BuildsTypedRegisterFromDescriptorClass) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_type_t value_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);

  loom_type_t register_type = loom_type_none();
  IREE_ASSERT_OK(loom_low_build_typed_register_type(
      module_, descriptor_set, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32,
      /*unit_count=*/4, value_type, &register_type));

  EXPECT_EQ(loom_low_register_type_descriptor_set_stable_id(register_type),
            descriptor_set->stable_id);
  EXPECT_EQ(loom_low_register_type_class_id(register_type),
            TEST_LOW_CORE_REG_CLASS_ID_TEST_I32);
  EXPECT_EQ(loom_low_register_type_unit_count(register_type), 4u);
  const loom_type_t* register_value_type =
      loom_type_register_value_type(register_type);
  ASSERT_NE(register_value_type, nullptr);
  EXPECT_TRUE(loom_type_equal(*register_value_type, value_type));
}

TEST_F(LowBuilderTest, RejectsInvalidTypedRegisterCarrier) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_type_t value_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t register_type = value_type;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_build_typed_register_type(
          module_, descriptor_set, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32,
          /*unit_count=*/0, value_type, &register_type));
  EXPECT_TRUE(loom_type_equal(register_type, loom_type_none()));

  register_type = value_type;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_low_build_typed_register_type(
          module_, descriptor_set, descriptor_set->reg_class_count,
          /*unit_count=*/1, value_type, &register_type));
  EXPECT_TRUE(loom_type_equal(register_type, loom_type_none()));
}

}  // namespace
}  // namespace loom
