// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/value_types.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/arch/spirv/ops/types.h"

namespace {

class SpirvValueTypesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

TEST_F(SpirvValueTypesTest, DecodesCooperativeMatrixPayload) {
  loom_type_t type = loom_type_none();
  IREE_ASSERT_OK(loom_spirv_cooperative_matrix_type_make(
      module_, 16, 32, LOOM_SPIRV_SCALAR_TYPE_U8, LOOM_SPIRV_SCOPE_SUBGROUP,
      LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR, &type));

  loom_spirv_value_type_t value_type = {};
  ASSERT_TRUE(loom_spirv_value_type_from_loom_type(type, &value_type));
  EXPECT_EQ(value_type.value_class, LOOM_SPIRV_VALUE_CLASS_COOPERATIVE_MATRIX);
  EXPECT_EQ(value_type.scalar_type, LOOM_SPIRV_SCALAR_TYPE_U8);
  EXPECT_EQ(value_type.cooperative_matrix.rows, 16);
  EXPECT_EQ(value_type.cooperative_matrix.columns, 32);
  EXPECT_EQ(value_type.cooperative_matrix.scope, LOOM_SPIRV_SCOPE_SUBGROUP);
  EXPECT_EQ(value_type.cooperative_matrix.use,
            LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_AKHR);
}

TEST_F(SpirvValueTypesTest, RejectsInvalidCooperativeMatrixDimensions) {
  static constexpr int64_t kInvalidDimensions[] = {
      0,
      -1,
      (int64_t)UINT16_MAX + 1,
  };
  for (int64_t dimension : kInvalidDimensions) {
    loom_type_t type = loom_type_none();
    IREE_ASSERT_OK(loom_spirv_cooperative_matrix_type_make(
        module_, dimension, 16, LOOM_SPIRV_SCALAR_TYPE_F16,
        LOOM_SPIRV_SCOPE_SUBGROUP,
        LOOM_SPIRV_COOPERATIVE_MATRIX_USE_MATRIX_BKHR, &type));
    loom_spirv_value_type_t value_type = {};
    EXPECT_FALSE(loom_spirv_value_type_from_loom_type(type, &value_type));
  }
}

}  // namespace
