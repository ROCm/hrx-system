// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/transfer.h"

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

TEST(AqlTransferPacketTest, DownloadBatchUsesExistingWindowWithoutGrowth) {
  struct Case {
    // Captured maximum host operation size, capped by the normal bulk limit.
    uint32_t staging;
    // Remaining captured descriptors.
    size_t count;
    // Already allocated transport-ring bytes; zero queries minimum growth.
    uint64_t capacity;
    // Required transport span.
    uint64_t expected;
  };
  const Case cases[] = {
      {17, 37, 0, 17},
      {17, 37, 8, 17},
      {17, 37, 1024, 272},
      {17, 37, 2048, 272},
      {4096, 256, 65536, 32768},
      {4096, 2, 65536, 8192},
      {4096, 1, 65536, 4096},
      {1048576, 1, 65536, 1048576},
      {1048576, 1, 1048576, 524288},
      {1048576, 32, UINT64_C(1) << 32, 1048576},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message() << test.staging << 'x' << test.count
                                      << " in " << test.capacity);
    const iree_hal_amd_gpu_transfer_t transfer = {
        .count = test.count + 3,
        .cursor = {.index = 3},
        .staging_capacity = test.staging,
        .host_completion_required = true,
    };
    EXPECT_EQ(
        iree_hal_amd_gpu_aql_transfer_staging_length(&transfer, test.capacity),
        test.expected);
    iree_hal_amd_gpu_transfer_operation_t operation = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .length = test.staging,
    };
    auto upload = transfer;
    upload.operations = &operation;
    upload.count = 1;
    upload.cursor.index = 0;
    upload.host_completion_required = false;
    EXPECT_EQ(
        iree_hal_amd_gpu_aql_transfer_staging_length(&upload, test.capacity),
        test.staging);
  }
}

TEST(AqlTransferPacketTest, LargeUploadsOverlapWithoutSplittingFittingSources) {
  struct Case {
    // Complete source payload; values above the staging bound require pieces.
    uint64_t length;
    // Existing power-of-two transport allocation or zero for the growth query.
    uint64_t capacity;
    // One publication's source-byte reservation.
    uint64_t expected;
  };
  constexpr uint64_t kBound = 16 * 1024 * 1024;
  constexpr uint32_t kStaging = IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_UPLOAD_LENGTH;
  const Case cases[] = {
      {1, 1024, 1},
      {1048699, 2097152, 1048699},
      {kStaging, kBound, kBound},
      {kStaging + 1, 0, kBound},
      {kStaging + 1, kBound, kBound / 2},
      {4 * kBound, kBound, kBound / 2},
      {4 * kBound, 2 * kBound, kBound},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << test.length << " in " << test.capacity);
    iree_hal_amd_gpu_transfer_operation_t operation = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
        .length = test.length,
    };
    const iree_hal_amd_gpu_transfer_t transfer = {
        .count = 1,
        .operations = &operation,
        .staging_capacity = static_cast<uint32_t>(
            std::min(test.length, static_cast<uint64_t>(kStaging))),
        .host_wait_required = true,
    };
    EXPECT_EQ(
        iree_hal_amd_gpu_aql_transfer_staging_length(&transfer, test.capacity),
        test.expected);
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
      iree_byte_span_empty(), 0, &chunk));
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
    std::array<uint8_t, 34> bytes;
    bytes.fill(0xCC);
    std::array<uint8_t, 5> staging;
    staging.fill(0xDD);
    iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
        &transfer, 0x1000, 1, iree_make_byte_span(bytes.data() + 1, 32), 0x2000,
        iree_make_byte_span(staging.data() + 1, 3), 0xABC000004000, &chunk));
    EXPECT_EQ(chunk.packet_count, 1u);
    EXPECT_EQ(chunk.argument_length, 32u);
    EXPECT_EQ(chunk.staging_length, 3u);
    EXPECT_EQ(chunk.next.index, 0u);
    EXPECT_EQ(chunk.next.offset, 6u);
    auto arguments = Arguments(bytes.data() + 1);
    EXPECT_EQ(arguments.length, 3u);
    EXPECT_EQ(arguments.source, download ? 0x123500000013u : 0xABC000004000u);
    EXPECT_EQ(arguments.target, download ? 0xABC000004000u : 0x123500000013u);
    if (download) {
      EXPECT_EQ(chunk.download.source, staging.data() + 1);
      EXPECT_EQ(chunk.download.operation, &operation);
      EXPECT_EQ(chunk.download.offset, 3u);
      EXPECT_EQ(chunk.download.length, 3u);
      EXPECT_TRUE(std::all_of(staging.begin(), staging.end(),
                              [](uint8_t byte) { return byte == 0xDD; }));
      EXPECT_EQ(target, (std::array<uint8_t, 8>{}));
    } else {
      EXPECT_EQ(chunk.download.length, 0u);
      EXPECT_EQ(std::memcmp(staging.data() + 1, source.data() + 3, 3), 0);
    }
    transfer.cursor = chunk.next;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
        &transfer, 0x1000, 1, iree_make_byte_span(bytes.data() + 1, 32), 0x2000,
        iree_make_byte_span(staging.data() + 1, 3), 0xABC000004000, &chunk));
    EXPECT_EQ(chunk.packet_count, 1u);
    EXPECT_EQ(chunk.argument_length, 32u);
    EXPECT_EQ(chunk.staging_length, 2u);
    EXPECT_EQ(chunk.next.index, 1u);
    EXPECT_EQ(chunk.next.offset, 0u);
    EXPECT_EQ(Arguments(bytes.data() + 1).length, 2u);
    EXPECT_EQ(bytes.front(), 0xCC);
    EXPECT_EQ(bytes.back(), 0xCC);
    EXPECT_EQ(staging.front(), 0xDD);
    EXPECT_EQ(staging.back(), 0xDD);
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
  std::array<uint8_t, 64> bytes;
  bytes.fill(0xCC);
  std::array<uint8_t, 8> staging;
  staging.fill(0xDD);
  iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
      &transfer, 0x1000, 2, iree_make_byte_span(bytes.data(), bytes.size()),
      0x2000, iree_make_byte_span(staging.data(), staging.size()), 0x9000,
      &chunk));
  EXPECT_EQ(chunk.packet_count, 1u);
  EXPECT_EQ(chunk.next.index, 1u);
  EXPECT_EQ(chunk.next.offset, 0u);
  EXPECT_EQ(chunk.argument_length, 32u);
  EXPECT_EQ(chunk.staging_length, 8u);
  EXPECT_EQ(chunk.download.source, staging.data());
  EXPECT_EQ(chunk.download.operation, operations);
  EXPECT_EQ(chunk.download.offset, 0u);
  EXPECT_EQ(chunk.download.length, 8u);
  EXPECT_EQ(Arguments(bytes.data()).target, 0x9000u);
  EXPECT_TRUE(std::all_of(bytes.begin() + 32, bytes.end(),
                          [](uint8_t byte) { return byte == 0xCC; }));
}

TEST_F(AqlTransferTest, ConsecutiveDownloadsShareOneRetirement) {
  std::array<std::array<uint8_t, 10>, 3> targets = {};
  std::array<iree_hal_amd_gpu_transfer_operation_t, 3> operations = {};
  for (size_t i = 0; i < operations.size(); ++i) {
    operations[i] = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .source = Device(i * 16),
        .target = {.host = targets[i].data() + 1},
        .length = 8,
    };
  }
  const iree_hal_amd_gpu_transfer_t transfer = {
      .count = operations.size(),
      .operations = operations.data(),
      .cursor = {.offset = 3},
  };
  for (size_t capacity : {0u, 1u, 5u, 6u, 13u, 14u, 21u, 22u}) {
    SCOPED_TRACE(capacity);
    std::array<uint8_t, 98> bytes;
    bytes.fill(0xCC);
    std::array<uint8_t, 24> staging;
    staging.fill(0xDD);
    iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
        &transfer, 0x1000, 3, iree_make_byte_span(bytes.data() + 1, 96), 0x2000,
        iree_make_byte_span(staging.data() + 1, capacity), 0x9000, &chunk));
    const size_t length = std::min(capacity, size_t{21});
    const size_t packets = !length ? 0 : 1 + (length + 2) / 8;
    EXPECT_EQ(chunk.packet_count, packets);
    EXPECT_EQ(chunk.argument_length, packets * 32);
    EXPECT_EQ(chunk.staging_length, length);
    EXPECT_EQ(chunk.download.length, length);
    EXPECT_EQ(chunk.next.index, (length + 3) / 8);
    EXPECT_EQ(chunk.next.offset, (length + 3) % 8);
    if (length) {
      EXPECT_EQ(chunk.download.source, staging.data() + 1);
      EXPECT_EQ(chunk.download.operation, operations.data());
      EXPECT_EQ(chunk.download.offset, 3u);
    }
    if (length > 5) {
      const auto arguments = Arguments(bytes.data() + 33);
      EXPECT_EQ(arguments.source, binding_.device_address + 16);
      EXPECT_EQ(arguments.target, 0x9005u);
      EXPECT_EQ(arguments.length, std::min(length - 5, size_t{8}));
    }
    EXPECT_EQ(bytes.front(), 0xCC);
    EXPECT_EQ(bytes.back(), 0xCC);
    EXPECT_TRUE(std::all_of(staging.begin(), staging.end(),
                            [](uint8_t byte) { return byte == 0xDD; }));
  }
}

}  // namespace
