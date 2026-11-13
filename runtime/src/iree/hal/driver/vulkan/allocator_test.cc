// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/allocator.h"

#include "iree/testing/gtest.h"

namespace iree::hal::vulkan {
namespace {

// Memory types are already compatible with the requested allocation. Ranking
// must preserve the access contract independently of their enumeration order.
TEST(VulkanAllocatorTest, RandomAccessPrefersCachedHostMemory) {
  for (const auto mapping : {IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
                             IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT}) {
    const iree_hal_buffer_usage_t access_patterns[] = {
        IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM,
        IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
            IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE,
    };
    for (const auto access : access_patterns) {
      const iree_hal_vulkan_buffer_spec_t spec = {
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                  IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | mapping | access,
      };
      for (const iree_hal_memory_type_t cached_coherence :
           {0u, uint32_t{IREE_HAL_MEMORY_TYPE_HOST_COHERENT}}) {
        for (const iree_hal_memory_type_t uncached_coherence :
             {0u, uint32_t{IREE_HAL_MEMORY_TYPE_HOST_COHERENT}}) {
          const iree_hal_memory_type_t cached =
              IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_CACHED | cached_coherence;
          const iree_hal_memory_type_t uncached =
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | uncached_coherence;
          SCOPED_TRACE(mapping);
          SCOPED_TRACE(access);
          SCOPED_TRACE(cached_coherence);
          SCOPED_TRACE(uncached_coherence);
          EXPECT_GT(
              iree_hal_vulkan_allocator_score_memory_type(&spec, cached),
              iree_hal_vulkan_allocator_score_memory_type(&spec, uncached));
        }
      }
    }
  }
}

TEST(VulkanAllocatorTest, SequentialWritesCanPreferUncachedDeviceMemory) {
  const iree_hal_vulkan_buffer_spec_t spec = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE,
  };
  const iree_hal_memory_type_t coherent = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                          IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                                          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  EXPECT_GE(iree_hal_vulkan_allocator_score_memory_type(
                &spec, coherent | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL),
            iree_hal_vulkan_allocator_score_memory_type(
                &spec, coherent | IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                           IREE_HAL_MEMORY_TYPE_HOST_CACHED));
}

TEST(VulkanAllocatorTest, DispatchPrefersDeviceLocalMemory) {
  const iree_hal_vulkan_buffer_spec_t spec = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH,
  };
  EXPECT_GT(iree_hal_vulkan_allocator_score_memory_type(
                &spec, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL),
            iree_hal_vulkan_allocator_score_memory_type(
                &spec, IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                           IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                           IREE_HAL_MEMORY_TYPE_HOST_CACHED |
                           IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST(VulkanAllocatorTest, EquallyCachedMappingsPreferUnifiedDeviceMemory) {
  const iree_hal_vulkan_buffer_spec_t spec = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
  };
  const iree_hal_memory_type_t cached = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                                        IREE_HAL_MEMORY_TYPE_HOST_CACHED;
  EXPECT_GT(iree_hal_vulkan_allocator_score_memory_type(
                &spec, cached | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL),
            iree_hal_vulkan_allocator_score_memory_type(&spec, cached));
}

}  // namespace
}  // namespace iree::hal::vulkan
