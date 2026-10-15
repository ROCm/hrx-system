// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/transfer.h"

#include <array>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

class TransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    words_.fill(kGuard);
    staging_.fill(0xCC);
  }

  iree_hal_amd_gpu_pm4_transfer_packet_t Emit(
      const iree_hal_amd_gpu_pm4_transfer_span_t& span, uint64_t offset,
      size_t capacity) {
    return iree_hal_amd_gpu_pm4_transfer_emit_packet(
        &span, offset, iree_make_byte_span(staging_.data(), capacity),
        0x987600003000, words_.data() + 1);
  }

  void Expect(std::initializer_list<uint32_t> packet) {
    EXPECT_EQ(std::vector<uint32_t>(words_.begin() + 1, words_.end() - 1),
              std::vector<uint32_t>(packet));
    EXPECT_EQ(words_.front(), kGuard);
    EXPECT_EQ(words_.back(), kGuard);
  }

  // Guard words detect writes outside the one native packet promised by Emit.
  static constexpr uint32_t kGuard = 0xD8C37F91;
  // One seven-DWORD native packet with a guard on both sides.
  std::array<uint32_t, 9> words_;
  // Caller-owned staging whose unused tail must remain unchanged.
  std::array<uint8_t, 8> staging_;
};

TEST_F(TransferTest, CopySplitsAtFirmwareLimitWithoutStaging) {
  const iree_hal_amd_gpu_pm4_transfer_span_t span = {
      .source_address = 0x1234FFFFFFF0,
      .target_address = 0x567800000003,
      .length = 32739,
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
  };
  auto packet = Emit(span, 0, 0);
  EXPECT_EQ(packet.length, 32736u);
  EXPECT_EQ(packet.staging_length, 0u);
  Expect({0xC0055000, 0x60300000, 0xFFFFFFF0, 0x1234, 3, 0x5678, 32736});
  packet = Emit(span, 32736, 0);
  EXPECT_EQ(packet.length, 3u);
  EXPECT_EQ(packet.staging_length, 0u);
  Expect({0xC0055000, 0x60300000, 0x7FD0, 0x1235, 0x7FE3, 0x5678, 3});
}

TEST_F(TransferTest, FillRotatesPatternAcrossStagedEdgesAndImmediateBody) {
  const iree_hal_amd_gpu_pm4_transfer_span_t span = {
      .target_address = 0x567800001001,
      .length = 12,
      .pattern = 0xA1B2C3D4,
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
  };
  auto packet = Emit(span, 0, 8);
  EXPECT_EQ(packet.length, 3u);
  EXPECT_EQ(packet.staging_length, 3u);
  Expect({0xC0055000, 0x60300000, 0x3000, 0x9876, 0x1001, 0x5678, 3});
  EXPECT_EQ(staging_, (std::array<uint8_t, 8>{0xD4, 0xC3, 0xB2, 0xCC, 0xCC,
                                              0xCC, 0xCC, 0xCC}));
  packet = Emit(span, 3, 0);
  EXPECT_EQ(packet.length, 8u);
  EXPECT_EQ(packet.staging_length, 0u);
  Expect({0xC0055000, 0x40300000, 0xB2C3D4A1, 0, 0x1004, 0x5678, 8});
  packet = Emit(span, 11, 1);
  EXPECT_EQ(packet.length, 1u);
  EXPECT_EQ(packet.staging_length, 1u);
  Expect({0xC0055000, 0x60300000, 0x3000, 0x9876, 0x100C, 0x5678, 1});
  EXPECT_EQ(staging_[0], 0xA1);
}

TEST_F(TransferTest, HostSourcesRespectCursorAndAvailableStaging) {
  const uint8_t source[] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (auto type : {IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD}) {
    SCOPED_TRACE(type);
    const iree_hal_amd_gpu_pm4_transfer_span_t span = {
        .target_address = 0x567800001000,
        .source_bytes = source,
        .length = sizeof(source),
        .type = type,
    };
    staging_.fill(0xCC);
    auto packet = Emit(span, 3, 3);
    EXPECT_EQ(packet.length, 3u);
    EXPECT_EQ(packet.staging_length, 3u);
    Expect({0xC0055000, 0x60300000, 0x3000, 0x9876, 0x1003, 0x5678, 3});
    EXPECT_EQ(staging_,
              (std::array<uint8_t, 8>{4, 5, 6, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC}));
    packet = Emit(span, 6, 3);
    EXPECT_EQ(packet.length, 2u);
    EXPECT_EQ(packet.staging_length, 2u);
    Expect({0xC0055000, 0x60300000, 0x3000, 0x9876, 0x1006, 0x5678, 2});
    EXPECT_EQ(staging_[0], 7);
    EXPECT_EQ(staging_[1], 8);
    EXPECT_EQ(staging_[2], 6);
  }
}

TEST_F(TransferTest, DownloadTargetsOwnedStagingWithoutReadingDeviceBytes) {
  const iree_hal_amd_gpu_pm4_transfer_span_t span = {
      .source_address = 0x123400001000,
      .length = 100,
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
  };
  const auto packet = Emit(span, 5, 7);
  EXPECT_EQ(packet.length, 7u);
  EXPECT_EQ(packet.staging_length, 7u);
  Expect({0xC0055000, 0x60300000, 0x1005, 0x1234, 0x3000, 0x9876, 7});
  for (uint8_t byte : staging_) {
    EXPECT_EQ(byte, 0xCC);
  }
}

TEST_F(TransferTest, EmptyStagingEmitsNothingForStagedOperations) {
  for (auto type : {IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD}) {
    SCOPED_TRACE(type);
    const iree_hal_amd_gpu_pm4_transfer_span_t span = {
        .target_address = 1,
        .length = 1,
        .type = type,
    };
    const auto packet = Emit(span, 0, 0);
    EXPECT_EQ(packet.length, 0u);
    EXPECT_EQ(packet.staging_length, 0u);
    for (uint32_t word : words_) {
      EXPECT_EQ(word, kGuard);
    }
  }
}

}  // namespace
