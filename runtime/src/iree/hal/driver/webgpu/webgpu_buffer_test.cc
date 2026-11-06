// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/webgpu/webgpu_buffer.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(WebGPUBufferTest, PublishesNativeBindingThroughSubspans) {
  constexpr iree_hal_webgpu_handle_t kBufferHandle = 42;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_webgpu_buffer_create(
      iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_STORAGE,
      /*allocation_size=*/1024, kBufferHandle,
      /*mapped_at_creation=*/false, iree_allocator_system(), &buffer));

  const iree_hal_buffer_binding_layout_t* layout =
      iree_hal_webgpu_buffer_binding_layout();
  ASSERT_EQ(layout->binding_count, 1u);
  EXPECT_EQ(layout->types[0], IREE_HAL_BUFFER_INTERFACE_WEBGPU_BUFFER);

  const iree_hal_buffer_native_binding_slot_t slot = {
      .index = IREE_HAL_WEBGPU_BUFFER_BINDING_RESOURCE,
      .type = IREE_HAL_BUFFER_INTERFACE_WEBGPU_BUFFER,
  };
  const iree_hal_buffer_native_binding_t root_binding =
      iree_hal_buffer_native_binding(buffer, slot);
  EXPECT_EQ(root_binding.webgpu.buffer, kBufferHandle);
  EXPECT_EQ(root_binding.webgpu.offset, 0u);

  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(buffer, /*byte_offset=*/256,
                                         /*byte_length=*/128,
                                         iree_allocator_system(), &subspan));
  const iree_hal_buffer_native_binding_t subspan_binding =
      iree_hal_buffer_native_binding(subspan, slot);
  EXPECT_EQ(subspan_binding.webgpu.buffer, kBufferHandle);
  EXPECT_EQ(subspan_binding.webgpu.offset, 256u);
  EXPECT_EQ(iree_hal_webgpu_buffer_handle(subspan), kBufferHandle);

  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(buffer);
}

}  // namespace
