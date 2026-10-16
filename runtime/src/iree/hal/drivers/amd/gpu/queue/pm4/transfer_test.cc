// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/transfer.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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

TEST_F(TransferTest, CopySplitsAtPacketLimitWithoutStaging) {
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

TEST(TransferBatchTest, HostTransfersFillOneWindowAcrossPacketBoundaries) {
  constexpr size_t kLength = 1024 * 1024;
  constexpr uint64_t kDeviceAddress = UINT64_C(0x123400001003);
  constexpr uint64_t kStagingAddress = UINT64_C(0x567800002005);
  std::vector<uint8_t> host(kLength + 17), staging(kLength + 2, 0xCC);
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<uint8_t>(i ^ (i >> 9));
  }
  const iree_hal_buffer_native_binding_t binding = {
      .device_address = kDeviceAddress,
  };
  iree_hal_buffer_t buffer = {};
  buffer.memory.bindings = &binding;
  for (auto type : {IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD}) {
    SCOPED_TRACE(type);
    std::fill(staging.begin(), staging.end(), 0xCC);
    const bool download = type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    iree_hal_amd_gpu_pm4_transfer_operation_t operation = {
        .type = type,
        .length = host.size(),
    };
    auto& device = download ? operation.source.device : operation.target.device;
    device.buffer = &buffer;
    device.slot = {
        .index = 0,
        .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    (download ? operation.target.host : operation.source.host) = host.data();
    const iree_hal_amd_gpu_pm4_transfer_t transfer = {
        .count = 1,
        .operations = &operation,
    };
    iree_hal_amd_gpu_pm4_transfer_chunk_t chunk;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_transfer_prepare_chunk(
        &transfer, IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_WORDS,
        iree_make_byte_span(staging.data() + 1, kLength), kStagingAddress,
        &chunk));
    ASSERT_EQ(chunk.word_count, 33u * 7);
    EXPECT_EQ(chunk.staging_length, kLength);
    EXPECT_EQ(chunk.next.index, 0u);
    EXPECT_EQ(chunk.next.offset, kLength);
    // All packets share one retirement extent. Check their complete native
    // encoding, including the small final packet and high address words.
    for (size_t i = 0; i < 33; ++i) {
      const uint64_t offset = i * 32736;
      const uint64_t source =
          (download ? kDeviceAddress : kStagingAddress) + offset;
      const uint64_t target =
          (download ? kStagingAddress : kDeviceAddress) + offset;
      const std::array<uint32_t, 7> expected = {
          0xC0055000,
          0x60300000,
          static_cast<uint32_t>(source),
          static_cast<uint32_t>(source >> 32),
          static_cast<uint32_t>(target),
          static_cast<uint32_t>(target >> 32),
          i == 32 ? 1024u : 32736u};
      EXPECT_EQ(
          std::vector<uint32_t>(chunk.words + i * 7, chunk.words + (i + 1) * 7),
          std::vector<uint32_t>(expected.begin(), expected.end()));
    }
    EXPECT_EQ(staging.front(), 0xCC);
    EXPECT_EQ(staging.back(), 0xCC);
    if (download) {
      EXPECT_EQ(chunk.download.source, staging.data() + 1);
      EXPECT_EQ(chunk.download.target, host.data());
      EXPECT_EQ(chunk.download.length, kLength);
      EXPECT_TRUE(std::all_of(staging.begin(), staging.end(),
                              [](uint8_t byte) { return byte == 0xCC; }));
    } else {
      EXPECT_EQ(chunk.download.length, 0u);
      EXPECT_TRUE(
          std::equal(staging.begin() + 1, staging.end() - 1, host.begin()));
    }
  }
}

TEST(TransferBatchTest, DownloadRetirementStopsAtTheHostDestinationBoundary) {
  std::vector<uint8_t> host(65536), staging(1024 * 1024);
  const iree_hal_buffer_native_binding_t binding = {.device_address = 0x10000};
  iree_hal_buffer_t buffer = {};
  buffer.memory.bindings = &binding;
  std::array<iree_hal_amd_gpu_pm4_transfer_operation_t, 2> operations = {};
  for (auto& operation : operations) {
    operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    operation.source.device.buffer = &buffer;
    operation.source.device.slot = {
        .index = 0,
        .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    operation.target.host = host.data();
    operation.length = host.size();
  }
  const iree_hal_amd_gpu_pm4_transfer_t transfer = {
      .count = operations.size(),
      .operations = operations.data(),
  };
  iree_hal_amd_gpu_pm4_transfer_chunk_t chunk;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_transfer_prepare_chunk(
      &transfer, IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_WORDS,
      iree_make_byte_span(staging.data(), staging.size()), 0x20000, &chunk));
  EXPECT_EQ(chunk.word_count, 3u * 7);
  EXPECT_EQ(chunk.next.index, 1u);
  EXPECT_EQ(chunk.next.offset, 0u);
  EXPECT_EQ(chunk.staging_length, host.size());
  EXPECT_EQ(chunk.download.length, host.size());
  EXPECT_EQ(chunk.download.source, staging.data());
  EXPECT_EQ(chunk.download.target, host.data());
}

}  // namespace
