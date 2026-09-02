// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/module_state.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowLowerModuleStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_low_lower_module_state_create(&arena_, &module_state_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_low_lower_module_state_t* module_state_ = nullptr;
};

TEST_F(LowLowerModuleStateTest, InternsZeroInitializedStateByStaticKey) {
  static const uint8_t kFirstKey = 0;
  static const uint8_t kSecondKey = 0;

  uint32_t* first = nullptr;
  IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
      module_state_, &kFirstKey, sizeof(*first), (void**)&first));
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(*first, 0u);
  *first = 42;

  uint32_t* first_again = nullptr;
  IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
      module_state_, &kFirstKey, sizeof(*first_again), (void**)&first_again));
  EXPECT_EQ(first_again, first);
  EXPECT_EQ(*first_again, 42u);

  uint32_t* second = nullptr;
  IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
      module_state_, &kSecondKey, sizeof(*second), (void**)&second));
  ASSERT_NE(second, nullptr);
  EXPECT_NE(second, first);
  EXPECT_EQ(*second, 0u);
}

TEST_F(LowLowerModuleStateTest, PreservesStateAcrossRecordGrowth) {
  static const uint8_t kKeys[32] = {0};
  uint32_t* values[IREE_ARRAYSIZE(kKeys)] = {nullptr};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKeys); ++i) {
    IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
        module_state_, &kKeys[i], sizeof(*values[i]), (void**)&values[i]));
    *values[i] = (uint32_t)(i + 1);
  }

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKeys); ++i) {
    uint32_t* value = nullptr;
    IREE_ASSERT_OK(loom_low_lower_module_state_get_or_allocate(
        module_state_, &kKeys[i], sizeof(*value), (void**)&value));
    EXPECT_EQ(value, values[i]);
    EXPECT_EQ(*value, (uint32_t)(i + 1));
  }
}

TEST_F(LowLowerModuleStateTest, AllocatesPassLocalStorage) {
  void* bytes = nullptr;
  IREE_ASSERT_OK(
      loom_low_lower_module_state_allocate(module_state_, 17, &bytes));
  EXPECT_NE(bytes, nullptr);

  uint32_t* values = nullptr;
  IREE_ASSERT_OK(loom_low_lower_module_state_allocate_array(
      module_state_, 8, sizeof(*values), (void**)&values));
  EXPECT_NE(values, nullptr);
}

TEST(LowLowerModuleStateStandaloneTest, RequiresModuleStateForNonemptyStorage) {
  static const uint8_t kKey = 0;
  void* data = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_lower_module_state_get_or_allocate(nullptr, &kKey, 1, &data));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_lower_module_state_allocate(nullptr, 1, &data));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_lower_module_state_allocate_array(nullptr, 1, 1, &data));
}

}  // namespace
}  // namespace loom
