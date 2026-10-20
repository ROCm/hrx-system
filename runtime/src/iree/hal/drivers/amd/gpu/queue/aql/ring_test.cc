// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/ring.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"

namespace {

class RingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (auto& packet : packets_) {
      for (uint32_t& word : packet.words) {
        word = 0xa5a5a5a5;
      }
      packet.words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
    }
    iree_hal_amd_gpu_aql_ring_initialize(packets_.data(), 0x123400001000,
                                         packets_.size(), &read_index_,
                                         &write_index_, &doorbell_, &ring_);
  }

  // Dependency-owned native storage; this fixture models consumption only.
  std::array<iree_hal_amd_gpu_aql_packet_t, 8> packets_;
  // Firmware-owned consumption index.
  iree_atomic_int64_t read_index_ = IREE_ATOMIC_VAR_INIT(0);
  // Native reservation index, published before a packet header.
  iree_atomic_int64_t write_index_ = IREE_ATOMIC_VAR_INIT(0);
  // Host-memory stand-in for the write-only native notification aperture.
  uint64_t doorbell_ = UINT64_MAX;
  // Actual publication implementation under test.
  iree_hal_amd_gpu_aql_ring_t ring_;
};

TEST_F(RingTest, DeviceProducedHeaderRemainsInvalidAtHostNotification) {
  uint64_t position = UINT64_MAX;
  ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 2, &position));
  EXPECT_EQ(position, 0u);
  EXPECT_EQ(iree_atomic_load(&write_index_, iree_memory_order_relaxed), 2);
  const auto prepare = iree_hal_amd_gpu_aql_packet_barrier(
      IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND, 0, nullptr, 0x123456780040);
  auto target = prepare;
  target.words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH | (1u << 16);
  iree_hal_amd_gpu_aql_ring_write_body(&ring_, position, &prepare);
  iree_hal_amd_gpu_aql_ring_write_body(&ring_, position + 1, &target);
  EXPECT_EQ(packets_[0].words[0], IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
  EXPECT_EQ(packets_[1].words[0], IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
  iree_hal_amd_gpu_aql_ring_publish_header(&ring_, position, prepare.words[0]);
  iree_hal_amd_gpu_aql_ring_notify(&ring_);
  EXPECT_EQ(doorbell_, 1u);
  EXPECT_EQ(std::memcmp(&packets_[0], &prepare, sizeof(prepare)), 0);
  EXPECT_EQ(packets_[1].words[0], IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
  EXPECT_EQ(std::memcmp(packets_[1].words + 1, target.words + 1, 60), 0);
  // Native GPU fixup owns the second header; the host never rewrites it.
}

TEST_F(RingTest, CapacityPressureChangesNoReservationOrNativeBytes) {
  uint64_t position;
  ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 8, &position));
  const auto saved_packets = packets_;
  position = 900;
  EXPECT_FALSE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 1, &position));
  EXPECT_EQ(position, 900u);
  EXPECT_EQ(ring_.reserved, 8u);
  EXPECT_EQ(iree_atomic_load(&write_index_, iree_memory_order_relaxed), 8);
  EXPECT_EQ(doorbell_, UINT64_MAX);
  EXPECT_EQ(
      std::memcmp(packets_.data(), saved_packets.data(), sizeof(packets_)), 0);
  iree_atomic_store(&read_index_, 1, iree_memory_order_release);
  ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 1, &position));
  EXPECT_EQ(position, 8u);
  EXPECT_EQ(ring_.consumed, 1u);
  EXPECT_EQ(iree_hal_amd_gpu_aql_ring_packet_address(&ring_, position),
            0x123400001000u);
}

TEST_F(RingTest, CachedCreditAvoidsConsumptionReadsUntilNeeded) {
  uint64_t position;
  ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 3, &position));
  iree_atomic_store(&read_index_, 3, iree_memory_order_release);
  ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 5, &position));
  EXPECT_EQ(position, 3u);
  EXPECT_EQ(ring_.consumed, 0u);
  ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 2, &position));
  EXPECT_EQ(position, 8u);
  EXPECT_EQ(ring_.consumed, 3u);
  ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 1, &position));
  EXPECT_EQ(position, 10u);
  EXPECT_FALSE(iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, 1, &position));
}

TEST_F(RingTest, WrapPublishesWholeHeadersFromImmutableSource) {
  uint64_t expected_position = 0;
  for (uint32_t round = 0; round < 131; ++round) {
    const uint32_t count = 1 + round % 7;
    uint64_t position;
    ASSERT_TRUE(
        iree_hal_amd_gpu_aql_ring_try_reserve(&ring_, count, &position));
    ASSERT_EQ(position, expected_position);
    for (uint32_t i = 0; i < count; ++i) {
      const auto packet = iree_hal_amd_gpu_aql_packet_barrier(
          IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED,
          0, nullptr, (position + i + 1) * 64);
      iree_hal_amd_gpu_aql_ring_write_body(&ring_, position + i, &packet);
      iree_hal_amd_gpu_aql_ring_publish_header(&ring_, position + i,
                                               packet.words[0]);
      EXPECT_EQ(std::memcmp(&packets_[(position + i) % packets_.size()],
                            &packet, sizeof(packet)),
                0);
    }
    iree_hal_amd_gpu_aql_ring_notify(&ring_);
    expected_position += count;
    EXPECT_EQ(doorbell_, expected_position - 1);
    // Firmware invalidates slots before it returns their consumption credit.
    for (uint32_t i = 0; i < count; ++i) {
      packets_[(position + i) % packets_.size()].words[0] =
          IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
    }
    iree_atomic_store(&read_index_, expected_position,
                      iree_memory_order_release);
  }
}

}  // namespace
