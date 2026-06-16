// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer.h"

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class DirectCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_amdxdna_device_options_initialize(&device_params_);
    device_ = new iree_hal_amdxdna_device{};
    device_->host_allocator = iree_allocator_system();
    iree_arena_block_pool_initialize(/*total_block_size=*/8 * 1024,
                                     iree_allocator_system(),
                                     &device_->block_pool);
    iree_arena_block_pool_initialize(/*total_block_size=*/8 * 1024,
                                     iree_allocator_system(), &block_pool_);
  }

  void TearDown() override {
    iree_arena_block_pool_deinitialize(&block_pool_);
    iree_arena_block_pool_deinitialize(&device_->block_pool);
    delete device_;
  }

  iree_hal_amdxdna_device_params device_params_;
  iree_hal_amdxdna_device* device_ = nullptr;
  iree_arena_block_pool_t block_pool_;
};

TEST_F(DirectCommandBufferTest, CreateRejectsUnretainedMode) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  iree_status_t status = iree_hal_amdxdna_direct_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED |
          IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0, &block_pool_,
      iree_allocator_system(), &command_buffer);

  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
  EXPECT_EQ(command_buffer, nullptr);
}

TEST_F(DirectCommandBufferTest, CreateAllowsRetainedMode) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_direct_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0, &block_pool_,
      iree_allocator_system(), &command_buffer));

  ASSERT_NE(command_buffer, nullptr);
  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(DirectCommandBufferTest, VtableHasOptionalOperationStubs) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_direct_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      /*binding_capacity=*/0, &block_pool_, iree_allocator_system(),
      &command_buffer));

  ASSERT_NE(command_buffer, nullptr);
  const auto* vtable =
      reinterpret_cast<const iree_hal_command_buffer_vtable_t*>(
          command_buffer->resource.vtable);
  EXPECT_NE(vtable->begin_debug_group, nullptr);
  EXPECT_NE(vtable->end_debug_group, nullptr);
  EXPECT_NE(vtable->advise_buffer, nullptr);
  EXPECT_NE(vtable->collective, nullptr);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(DirectCommandBufferTest, DebugGroupsAreNoOp) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_direct_command_buffer_create(
      device_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*binding_capacity=*/0, &block_pool_,
      iree_allocator_system(), &command_buffer));

  ASSERT_NE(command_buffer, nullptr);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin_debug_group(
      command_buffer, IREE_SV("trace-region"),
      iree_hal_label_color_unspecified(), /*location=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end_debug_group(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

}  // namespace
