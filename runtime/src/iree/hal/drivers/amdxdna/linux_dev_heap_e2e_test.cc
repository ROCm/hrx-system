// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/hal/drivers/amdxdna/api.h"
#include "iree/hal/drivers/amdxdna/native.h"
#include "iree/testing/gtest.h"

namespace {

constexpr iree_host_size_t kLinuxDevHeapBytes = 64u * 1024u * 1024u;
constexpr iree_host_size_t kLinuxMissReserveBytes = 4u * 1024u * 1024u;

iree_hal_amdxdna_native_device_t* TryOpenNativeDevice() {
  iree_hal_amdxdna_device_params params;
  iree_hal_amdxdna_device_options_initialize(&params);
  iree_hal_amdxdna_native_device_t* device = nullptr;
  iree_status_t status = iree_hal_amdxdna_native_device_c_create(
      &params, iree_allocator_system(), &device);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return nullptr;
  }
  return device;
}

TEST(AmdxdnaDevHeapE2E, QueryCapsAdvertisesLinuxHeapOnlyOnKmq) {
  iree_hal_amdxdna_native_device_t* device = TryOpenNativeDevice();
  if (!device) {
    GTEST_SKIP() << "no amdxdna device";
  }
  iree_hal_amdxdna_native_c_device_caps_t caps = {};
  iree_status_t status =
      iree_hal_amdxdna_native_device_c_query_caps(device, &caps);
  EXPECT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);
#if defined(__linux__)
  EXPECT_EQ(caps.max_shared_code_memory_bytes, kLinuxDevHeapBytes);
  EXPECT_EQ(caps.shared_code_memory_miss_reserve_bytes, kLinuxMissReserveBytes);
#else
  EXPECT_EQ(caps.max_shared_code_memory_bytes, 0u);
  EXPECT_EQ(caps.shared_code_memory_miss_reserve_bytes, 0u);
#endif
  iree_hal_amdxdna_native_device_c_destroy(device);
}

#if defined(__linux__)

void DestroyBuffers(std::vector<iree_hal_amdxdna_native_buffer_t*>* buffers) {
  for (iree_hal_amdxdna_native_buffer_t* buffer : *buffers) {
    iree_hal_amdxdna_native_buffer_c_destroy(buffer);
  }
  buffers->clear();
}

iree_status_t AllocDevHeap(iree_hal_amdxdna_native_device_t* device,
                           iree_device_size_t size,
                           iree_hal_amdxdna_native_buffer_c_type_t type,
                           iree_hal_amdxdna_native_buffer_t** out_buffer) {
  *out_buffer = nullptr;
  return iree_hal_amdxdna_native_device_c_alloc_buffer(device, size, type,
                                                       out_buffer);
}

// Fast on-device analog of filling unique CreateContexts until 0xc01e0009:
// carve instruction BOs from the process 64MiB DEV heap until the driver
// reports exhaustion (EAGAIN or ENOSPC), then show that freeing one
// cached-size BO lets the ~33KB miss allocate.
TEST(LinuxDevHeapE2E, InstructionAllocationsExhaustThe64MiBHeap) {
  iree_hal_amdxdna_native_device_t* device = TryOpenNativeDevice();
  if (!device) {
    GTEST_SKIP() << "no amdxdna device";
  }

  const iree_device_size_t chunk = 1u << 20;
  const iree_device_size_t miss = 33u * 1024u;
  std::vector<iree_hal_amdxdna_native_buffer_t*> buffers;
  iree_device_size_t allocated = 0;
  for (;;) {
    iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
    iree_status_t status =
        AllocDevHeap(device, chunk,
                     IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION, &buffer);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      break;
    }
    if (!iree_status_is_ok(status)) {
      DestroyBuffers(&buffers);
      iree_hal_amdxdna_native_device_c_destroy(device);
      FAIL() << "instruction BO alloc failed before the heap was exhausted";
    }
    buffers.push_back(buffer);
    allocated += chunk;
    ASSERT_LT(allocated, 2 * kLinuxDevHeapBytes)
        << "DEV heap did not bound instruction allocations";
  }

  EXPECT_GE(allocated, 48u << 20);
  EXPECT_LE(allocated, kLinuxDevHeapBytes);

  iree_hal_amdxdna_native_buffer_t* miss_buffer = nullptr;
  iree_status_t status = AllocDevHeap(
      device, miss, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION,
      &miss_buffer);
  EXPECT_TRUE(iree_status_is_unavailable(status));
  iree_status_ignore(status);
  EXPECT_EQ(miss_buffer, nullptr);

  iree_hal_amdxdna_native_buffer_t* host_buffer = nullptr;
  status =
      AllocDevHeap(device, chunk, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_HOST_ONLY,
                   &host_buffer);
  EXPECT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);
  iree_hal_amdxdna_native_buffer_c_destroy(host_buffer);

  ASSERT_FALSE(buffers.empty());
  iree_hal_amdxdna_native_buffer_c_destroy(buffers.back());
  buffers.pop_back();

  status = AllocDevHeap(device, miss,
                        IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION,
                        &miss_buffer);
  EXPECT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);
  iree_hal_amdxdna_native_buffer_c_destroy(miss_buffer);

  DestroyBuffers(&buffers);
  iree_hal_amdxdna_native_device_c_destroy(device);
}

// FLM sweep shape: ~32MiB of PDI-like CACHEABLE BOs leave only ~32MiB for
// instruction BOs, so the next 1MiB instruction alloc fails.
TEST(LinuxDevHeapE2E, CacheableAndInstructionShareTheSameHeap) {
  iree_hal_amdxdna_native_device_t* device = TryOpenNativeDevice();
  if (!device) {
    GTEST_SKIP() << "no amdxdna device";
  }

  const iree_device_size_t chunk = 1u << 20;
  std::vector<iree_hal_amdxdna_native_buffer_t*> pdi_buffers;
  for (int i = 0; i < 32; ++i) {
    iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
    iree_status_t status = AllocDevHeap(
        device, chunk, IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_CACHEABLE, &buffer);
    if (!iree_status_is_ok(status)) {
      DestroyBuffers(&pdi_buffers);
      iree_hal_amdxdna_native_device_c_destroy(device);
      GTEST_SKIP() << "could not allocate 32MiB of CACHEABLE BOs";
    }
    pdi_buffers.push_back(buffer);
  }

  std::vector<iree_hal_amdxdna_native_buffer_t*> instruction_buffers;
  iree_device_size_t instruction_bytes = 0;
  for (;;) {
    iree_hal_amdxdna_native_buffer_t* buffer = nullptr;
    iree_status_t status =
        AllocDevHeap(device, chunk,
                     IREE_HAL_AMDXDNA_NATIVE_BUFFER_TYPE_INSTRUCTION, &buffer);
    if (iree_status_is_unavailable(status)) {
      iree_status_ignore(status);
      break;
    }
    if (!iree_status_is_ok(status)) {
      DestroyBuffers(&instruction_buffers);
      DestroyBuffers(&pdi_buffers);
      iree_hal_amdxdna_native_device_c_destroy(device);
      FAIL() << "instruction BO alloc failed before the remaining heap was "
                "exhausted";
    }
    instruction_buffers.push_back(buffer);
    instruction_bytes += chunk;
    ASSERT_LT(instruction_bytes, kLinuxDevHeapBytes);
  }

  EXPECT_LE(instruction_bytes, 32u << 20);
  EXPECT_GE(instruction_bytes, 16u << 20);

  DestroyBuffers(&instruction_buffers);
  DestroyBuffers(&pdi_buffers);
  iree_hal_amdxdna_native_device_c_destroy(device);
}

#endif  // defined(__linux__)

}  // namespace
