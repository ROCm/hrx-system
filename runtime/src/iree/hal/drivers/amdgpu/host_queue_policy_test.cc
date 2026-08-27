// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unit tests for the host-queue buffer fence-scope policy. No GPU required.

#include "iree/hal/drivers/amdgpu/host_queue_policy.h"

#include <cstdint>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

// Wraps |storage| in a buffer reporting |memory_type|.
static iree_status_t WrapBufferWithMemoryType(
    iree_hal_memory_type_t memory_type, iree_byte_span_t storage,
    iree_hal_buffer_t** out_buffer) {
  return iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), memory_type,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER,
      storage.data_length, storage, iree_hal_buffer_release_callback_null(),
      iree_allocator_system(), out_buffer);
}

TEST(HostQueuePolicyTest, HostVisibleTargetRequiresSystemRelease) {
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) uint64_t storage = 0;
  const iree_byte_span_t span = iree_make_byte_span(&storage, sizeof(storage));

  iree_hal_buffer_t* host_local = NULL;
  IREE_ASSERT_OK(WrapBufferWithMemoryType(
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      span, &host_local));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_buffer_release_scope(host_local),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_hal_buffer_release(host_local);

  // The large-BAR / APU shape: device-local storage the host can still map.
  iree_hal_buffer_t* device_fine = NULL;
  IREE_ASSERT_OK(WrapBufferWithMemoryType(
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      span, &device_fine));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_buffer_release_scope(device_fine),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_hal_buffer_release(device_fine);
}

TEST(HostQueuePolicyTest, HostVisibleSourceRequiresSystemAcquire) {
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) uint64_t storage = 0;
  const iree_byte_span_t span = iree_make_byte_span(&storage, sizeof(storage));

  iree_hal_buffer_t* host_local = NULL;
  IREE_ASSERT_OK(WrapBufferWithMemoryType(
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
          IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      span, &host_local));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_buffer_acquire_scope(host_local),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_hal_buffer_release(host_local);
}

// A target the host cannot map adds nothing: the signal list already covers
// any other-agent consumer, and SYSTEM here would write back the L2 on every
// device-local transfer.
TEST(HostQueuePolicyTest, DeviceOnlyTargetAddsNoReleaseScope) {
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) uint64_t storage = 0;
  const iree_byte_span_t span = iree_make_byte_span(&storage, sizeof(storage));

  iree_hal_buffer_t* device_coarse = NULL;
  IREE_ASSERT_OK(WrapBufferWithMemoryType(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                                          span, &device_coarse));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_buffer_release_scope(device_coarse),
            IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_buffer_acquire_scope(device_coarse),
            IREE_HSA_FENCE_SCOPE_NONE);
  iree_hal_buffer_release(device_coarse);
}

// The buffer-derived scope composes with a caller-supplied minimum rather than
// replacing it.
TEST(HostQueuePolicyTest, BufferReleaseScopeComposesWithCallerMinimum) {
  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) uint64_t storage = 0;
  const iree_byte_span_t span = iree_make_byte_span(&storage, sizeof(storage));

  iree_hal_buffer_t* device_coarse = NULL;
  IREE_ASSERT_OK(WrapBufferWithMemoryType(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                                          span, &device_coarse));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_max_fence_scope(
                IREE_HSA_FENCE_SCOPE_SYSTEM,
                iree_hal_amdgpu_host_queue_buffer_release_scope(device_coarse)),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_hal_buffer_release(device_coarse);
}

}  // namespace
}  // namespace iree::hal::amdgpu
