// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/transfer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(AqlTransferPacketTest, BoundedGridTraversesTheCompleteByteRange) {
  struct Case {
    // Complete transfer extent, independent of argument/staging capacity.
    uint64_t length;
    // Native grid width required by the fixed-stride blit kernel.
    uint32_t grid;
  };
  const Case cases[] = {
      {0, 0},
      {1, 64},
      {1024, 64},
      {1025, 128},
      {262144, 16384},
      {262145, 16384},
      {UINT64_C(8) << 30, 16384},
  };
  for (auto [length, grid] : cases) {
    SCOPED_TRACE(length);
    const auto packet = iree_hal_amd_gpu_aql_transfer_packet(
        0x123456789000, 0xABCDEF010020, length,
        IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
            IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM);
    const std::array<uint32_t, 16> expected = {
        0x10502,    0x10040, 1,          grid,   1, 1, 0, 0,
        0x56789000, 0x1234,  0xEF010020, 0xABCD, 0, 0, 0, 0};
    EXPECT_EQ(std::vector<uint32_t>(packet.words, packet.words + 16),
              std::vector<uint32_t>(expected.begin(), expected.end()));
  }
}

class AqlTransferTest : public ::testing::Test {
 protected:
  void SetUp() override { buffer_.memory.bindings = &binding_; }

  iree_hal_amd_gpu_transfer_address_t Device(uint64_t offset) {
    return {
        .device =
            {
                .buffer = &buffer_,
                .offset = offset,
                .slot =
                    {
                        .index = 0,
                        .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                    },
            },
    };
  }

  iree_hal_amd_gpu_blit_arguments_t Arguments(const uint8_t* bytes) {
    iree_hal_amd_gpu_blit_arguments_t arguments;
    std::memcpy(&arguments, bytes, sizeof(arguments));
    return arguments;
  }

  // Prepared native-address dependency; the encoder never reads device bytes.
  iree_hal_buffer_native_binding_t binding_ = {
      .device_address = 0x1234FFFFFFF0,
  };
  // Borrowed logical buffer whose immutable slot was admitted at capture.
  iree_hal_buffer_t buffer_ = {};
};

TEST_F(AqlTransferTest, MultiGigabyteDeviceRangesNeedOnlyOnePacketEach) {
  const uint64_t length = (UINT64_C(8) << 30) + 17;
  iree_hal_amd_gpu_transfer_operation_t operations[] = {
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .source = Device(16),
          .target = Device(32),
          .length = length,
      },
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .pattern = 0xA1B2C3D4,
          .target = Device(48),
          .length = length,
      },
  };
  const iree_hal_amd_gpu_transfer_t transfer = {
      .count = IREE_ARRAYSIZE(operations),
      .operations = operations,
  };
  std::array<uint8_t, 66> bytes;
  bytes.fill(0xCC);
  iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
      &transfer, 0x1000, 2, iree_make_byte_span(bytes.data() + 1, 64), 0x2000,
      &chunk));
  EXPECT_EQ(chunk.packet_count, 2u);
  EXPECT_EQ(chunk.argument_length, 64u);
  EXPECT_EQ(chunk.next.index, 2u);
  EXPECT_EQ(chunk.next.offset, 0u);
  EXPECT_EQ(chunk.download.length, 0u);
  auto copy = Arguments(bytes.data() + 1);
  EXPECT_EQ(copy.source, 0x123500000000u);
  EXPECT_EQ(copy.target, 0x123500000010u);
  EXPECT_EQ(copy.length, length);
  EXPECT_EQ(copy.fill, 0u);
  auto fill = Arguments(bytes.data() + 33);
  EXPECT_EQ(fill.source, 0u);
  EXPECT_EQ(fill.target, 0x123500000020u);
  EXPECT_EQ(fill.length, length);
  EXPECT_EQ(fill.pattern, 0xA1B2C3D4u);
  EXPECT_EQ(fill.fill, 1u);
  EXPECT_EQ(chunk.packets[0].words[10], 0x2000u);
  EXPECT_EQ(chunk.packets[1].words[10], 0x2020u);
  EXPECT_EQ(bytes.front(), 0xCC);
  EXPECT_EQ(bytes.back(), 0xCC);
}

TEST_F(AqlTransferTest, HostRangesRespectCursorAndStagingCapacity) {
  const std::array<uint8_t, 8> source = {1, 2, 3, 4, 5, 6, 7, 8};
  for (auto type : {IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD}) {
    SCOPED_TRACE(type);
    std::array<uint8_t, 8> target = {};
    const bool download = type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    iree_hal_amd_gpu_transfer_operation_t operation = {
        .type = type,
        .length = source.size(),
    };
    if (download) {
      operation.source = Device(32);
      operation.target.host = target.data();
    } else {
      operation.source.host = const_cast<uint8_t*>(source.data());
      operation.target = Device(32);
    }
    iree_hal_amd_gpu_transfer_t transfer = {
        .count = 1,
        .operations = &operation,
        .cursor = {.offset = 3},
    };
    std::array<uint8_t, 37> bytes;
    bytes.fill(0xCC);
    iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
        &transfer, 0x1000, 1, iree_make_byte_span(bytes.data() + 1, 35), 0x2000,
        &chunk));
    EXPECT_EQ(chunk.packet_count, 1u);
    EXPECT_EQ(chunk.argument_length, 35u);
    EXPECT_EQ(chunk.next.index, 0u);
    EXPECT_EQ(chunk.next.offset, 6u);
    auto arguments = Arguments(bytes.data() + 1);
    EXPECT_EQ(arguments.length, 3u);
    EXPECT_EQ(arguments.source, download ? 0x123500000013u : 0x2020u);
    EXPECT_EQ(arguments.target, download ? 0x2020u : 0x123500000013u);
    if (download) {
      EXPECT_EQ(chunk.download.source, bytes.data() + 33);
      EXPECT_EQ(chunk.download.target, target.data() + 3);
      EXPECT_EQ(chunk.download.length, 3u);
      EXPECT_TRUE(std::all_of(bytes.begin() + 33, bytes.end(),
                              [](uint8_t byte) { return byte == 0xCC; }));
      EXPECT_EQ(target, (std::array<uint8_t, 8>{}));
    } else {
      EXPECT_EQ(chunk.download.length, 0u);
      EXPECT_EQ(std::memcmp(bytes.data() + 33, source.data() + 3, 3), 0);
    }
    transfer.cursor = chunk.next;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
        &transfer, 0x1000, 1, iree_make_byte_span(bytes.data() + 1, 35), 0x2000,
        &chunk));
    EXPECT_EQ(chunk.packet_count, 1u);
    EXPECT_EQ(chunk.argument_length, 34u);
    EXPECT_EQ(chunk.next.index, 1u);
    EXPECT_EQ(chunk.next.offset, 0u);
    EXPECT_EQ(Arguments(bytes.data() + 1).length, 2u);
    EXPECT_EQ(bytes.front(), 0xCC);
    EXPECT_EQ(bytes.back(), 0xCC);
  }
}

TEST_F(AqlTransferTest, DownloadEndsThePieceBeforeItsSibling) {
  std::array<uint8_t, 8> target = {};
  iree_hal_amd_gpu_transfer_operation_t operations[] = {
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .source = Device(0),
          .target = {.host = target.data()},
          .length = target.size(),
      },
      {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .target = Device(16),
          .length = 16,
      },
  };
  const iree_hal_amd_gpu_transfer_t transfer = {
      .count = 2,
      .operations = operations,
  };
  std::array<uint8_t, 72> bytes;
  bytes.fill(0xCC);
  iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
      &transfer, 0x1000, 2, iree_make_byte_span(bytes.data(), bytes.size()),
      0x2000, &chunk));
  EXPECT_EQ(chunk.packet_count, 1u);
  EXPECT_EQ(chunk.next.index, 1u);
  EXPECT_EQ(chunk.next.offset, 0u);
  EXPECT_EQ(chunk.argument_length, 72u);
  EXPECT_EQ(chunk.download.source, bytes.data() + 64);
  EXPECT_EQ(chunk.download.target, target.data());
  EXPECT_EQ(chunk.download.length, 8u);
  EXPECT_EQ(Arguments(bytes.data()).target, 0x2040u);
  EXPECT_TRUE(std::all_of(bytes.begin() + 32, bytes.end(),
                          [](uint8_t byte) { return byte == 0xCC; }));
}

}  // namespace
