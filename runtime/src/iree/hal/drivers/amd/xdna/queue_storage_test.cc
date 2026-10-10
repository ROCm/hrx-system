// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_storage.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct AllocationCounts {
  uint64_t allocations = 0;
  uint64_t frees = 0;
};

iree_status_t CountingAllocatorControl(void* self,
                                       iree_allocator_command_t command,
                                       const void* params,
                                       void** inout_pointer) {
  auto* counts = static_cast<AllocationCounts*>(self);
  if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
      command == IREE_ALLOCATOR_COMMAND_CALLOC) {
    ++counts->allocations;
  } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
    ++counts->frees;
  }
  iree_allocator_t system_allocator = iree_allocator_system();
  return system_allocator.ctl(system_allocator.self, command, params,
                              inout_pointer);
}

iree_allocator_t CountingAllocator(AllocationCounts* counts) {
  return {counts, CountingAllocatorControl};
}

TEST(XdnaQueueStorageTest, ReusesMetadataAndChunkedPayloadBlocks) {
  AllocationCounts counts;
  iree_hal_amd_xdna_queue_storage_t storage;
  iree_hal_amd_xdna_queue_storage_initialize(CountingAllocator(&counts),
                                             &storage);
  std::vector<uint8_t> source(128 * 1024);
  for (iree_host_size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i);
  }

  auto capture_and_check = [&]() {
    iree_hal_amd_xdna_queue_capture_t capture;
    iree_hal_amd_xdna_queue_capture_initialize(&storage, &capture);
    void* metadata = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &capture, 2048, &metadata));
    ASSERT_NE(metadata, nullptr);
    iree_hal_amd_xdna_queue_payload_t payload;
    IREE_ASSERT_OK(iree_hal_amd_xdna_queue_capture_payload(
        &capture, iree_make_const_byte_span(source.data(), source.size()),
        &payload));
    EXPECT_EQ(payload.data_length, source.size());
    std::vector<uint8_t> actual(source.size());
    iree_hal_amd_xdna_queue_payload_copy(&payload, actual.data());
    EXPECT_EQ(actual, source);
    iree_hal_amd_xdna_queue_capture_deinitialize(&capture);
  };

  capture_and_check();
  const AllocationCounts warm_counts = counts;
  capture_and_check();
  EXPECT_EQ(counts.allocations, warm_counts.allocations);
  EXPECT_EQ(counts.frees, warm_counts.frees);

  iree_hal_amd_xdna_queue_storage_deinitialize(&storage);
  EXPECT_EQ(counts.allocations, counts.frees);
}

TEST(XdnaQueueStorageTest, RejectsOversizedMetadataWithoutAllocation) {
  AllocationCounts counts;
  iree_hal_amd_xdna_queue_storage_t storage;
  iree_hal_amd_xdna_queue_storage_initialize(CountingAllocator(&counts),
                                             &storage);
  iree_hal_amd_xdna_queue_capture_t capture;
  iree_hal_amd_xdna_queue_capture_initialize(&storage, &capture);

  void* metadata = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_amd_xdna_queue_capture_allocate_metadata(
                            &capture, 4096, &metadata));
  EXPECT_EQ(metadata, nullptr);
  EXPECT_EQ(counts.allocations, 0u);
  EXPECT_EQ(counts.frees, 0u);

  iree_hal_amd_xdna_queue_capture_deinitialize(&capture);
  iree_hal_amd_xdna_queue_storage_deinitialize(&storage);
}

}  // namespace
