// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

#include "common/internal.h"
#include "common/rocm_hostcall_packet.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(RocmHostcallTest, BoundsCyclicPacketChainsWithHostAllocationSize) {
  iree_hal_streaming_hostcall_packet_header_t packet_headers[4] = {};
  packet_headers[1].next = UINT64_C(0x101);

  iree_hal_streaming_hostcall_packet_iterator_t iterator;
  iree_hal_streaming_hostcall_packet_iterator_initialize(
      packet_headers, IREE_ARRAYSIZE(packet_headers), UINT64_C(0x101),
      &iterator);
  uint32_t packet_index = 0;
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(packet_headers); ++i) {
    ASSERT_TRUE(iree_hal_streaming_hostcall_packet_iterator_advance(
        &iterator, &packet_index));
    EXPECT_EQ(1u, packet_index);
  }
  EXPECT_FALSE(iree_hal_streaming_hostcall_packet_iterator_advance(
      &iterator, &packet_index));
}

TEST(RocmHostcallTest, MasksTaggedPacketIndexesWithHostAllocationSize) {
  iree_hal_streaming_hostcall_packet_header_t packet_headers[4] = {};
  iree_hal_streaming_hostcall_packet_iterator_t iterator;
  iree_hal_streaming_hostcall_packet_iterator_initialize(
      packet_headers, IREE_ARRAYSIZE(packet_headers), UINT64_MAX, &iterator);

  uint32_t packet_index = 0;
  ASSERT_TRUE(iree_hal_streaming_hostcall_packet_iterator_advance(
      &iterator, &packet_index));
  EXPECT_EQ(3u, packet_index);
}

TEST(RocmHostcallTest, RoundsResidentWavesToPowerOfTwo) {
  iree_hal_streaming_device_t device = {};
  device.raw_compute_unit_count = 304;
  device.maximum_resident_subgroup_count = 32;

  uint32_t packet_count = 0;
  IREE_ASSERT_OK(iree_hal_streaming_rocm_hostcall_calculate_packet_count(
      &device, &packet_count));
  EXPECT_EQ(16384u, packet_count);
}

TEST(RocmHostcallTest, UsesUnadjustedPhysicalProperties) {
  iree_hal_streaming_device_t device = {};
  device.raw_compute_unit_count = 60;
  device.maximum_resident_subgroup_count = 64;
  device.multiprocessor_count = 30;
  device.warp_size = 32;

  uint32_t packet_count = 0;
  IREE_ASSERT_OK(iree_hal_streaming_rocm_hostcall_calculate_packet_count(
      &device, &packet_count));
  EXPECT_EQ(4096u, packet_count);
}

TEST(RocmHostcallTest, RejectsIncompletePhysicalProperties) {
  iree_hal_streaming_device_t device = {};
  uint32_t packet_count = 123;

  iree_status_t status =
      iree_hal_streaming_rocm_hostcall_calculate_packet_count(&device,
                                                              &packet_count);
  EXPECT_EQ(IREE_STATUS_FAILED_PRECONDITION, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_EQ(0u, packet_count);
}

TEST(RocmHostcallTest, RejectsUnsupportedHostAtomicsBeforeServiceCreation) {
  iree_hal_streaming_device_t device = {};
  iree_hal_streaming_context_t context = {};
  context.device_entry = &device;

  uint64_t buffer_device_ptr = UINT64_C(1);
  iree_status_t status = iree_hal_streaming_context_rocm_hostcall_buffer(
      &context, &buffer_device_ptr);
  EXPECT_EQ(IREE_STATUS_UNIMPLEMENTED, iree_status_code(status));
  iree_status_ignore(status);
  EXPECT_EQ(0u, buffer_device_ptr);
}

TEST(RocmHostcallTest, ReturnsPublishedServiceBufferWithoutContextLock) {
  iree_hal_streaming_device_t device = {};
  device.host_native_atomic_supported = true;
  iree_hal_streaming_context_t context = {};
  context.device_entry = &device;
  iree_atomic_store(&context.rocm_device_runtime.hostcall_buffer_device_ptr,
                    UINT64_C(0x123456789ABCDEF0), iree_memory_order_release);

  uint64_t buffer_device_ptr = 0;
  IREE_ASSERT_OK(iree_hal_streaming_context_rocm_hostcall_buffer(
      &context, &buffer_device_ptr));
  EXPECT_EQ(UINT64_C(0x123456789ABCDEF0), buffer_device_ptr);
}

}  // namespace
