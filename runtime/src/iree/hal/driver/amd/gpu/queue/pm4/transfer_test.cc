// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/transfer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "iree/hal/driver/amd/gpu/queue/blit.h"
#include "iree/hal/driver/amd/gpu/queue/dma_transfer.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/launch.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static_assert(sizeof(void*) != 8 || sizeof(iree_hal_amd_gpu_transfer_t) == 40);
static_assert(sizeof(void*) != 8 ||
              sizeof(iree_hal_amd_gpu_transfer_operation_t) == 64);

class TransferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    words_.fill(kGuard);
    staging_.fill(0xCC);
  }

  iree_hal_amd_gpu_dma_transfer_packet_t Emit(
      const iree_hal_amd_gpu_dma_transfer_span_t& span, uint64_t offset,
      size_t capacity) {
    return iree_hal_amd_gpu_dma_transfer_emit_packet(
        &iree_hal_amd_gpu_pm4_transfer_encoding, &span, offset,
        iree_make_byte_span(staging_.data(), capacity), 0x987600003000,
        words_.data() + 1);
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
  const iree_hal_amd_gpu_dma_transfer_span_t span = {
      .source_address = 0x1234FFFFFFF0,
      .target_address = 0x567800000003,
      .length = 32 * 1024 * 1024 + 3,
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
  };
  auto packet = Emit(span, 0, 0);
  EXPECT_EQ(packet.length, 32u * 1024 * 1024);
  EXPECT_EQ(packet.staging_length, 0u);
  Expect({0xC0055000, 0x60300000, 0xFFFFFFF0, 0x1234, 3, 0x5678, 0x2000000});
  packet = Emit(span, 32 * 1024 * 1024, 0);
  EXPECT_EQ(packet.length, 3u);
  EXPECT_EQ(packet.staging_length, 0u);
  Expect({0xC0055000, 0x60300000, 0x1FFFFF0, 0x1235, 0x2000003, 0x5678, 3});
}

TEST_F(TransferTest, DmaRectangleSplitsAtRowsWithIndependentEndpointStrides) {
  const iree_hal_transfer_region_t region = {
      .row_length = 5,
      .row_count = 3,
      .slice_count = 2,
      .source_strides = {.row_stride = 8, .slice_stride = 32},
      .target_strides = {.row_stride = 11, .slice_stride = 40},
  };
  const iree_hal_amd_gpu_dma_transfer_span_t span = {
      .source_address = 0x123400001000,
      .target_address = 0x567800002000,
      .length = 30,
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .region = &region,
  };
  for (uint32_t offset = 0; offset < 30; ++offset) {
    SCOPED_TRACE(offset);
    const auto packet = Emit(span, offset, 0);
    const uint32_t length = 5 - offset % 5;
    EXPECT_EQ(packet.length, length);
    EXPECT_EQ(packet.staging_length, 0u);
    Expect({0xC0055000, 0x60300000,
            0x1000 + (offset / 15) * 32 + (offset % 15 / 5) * 8 + offset % 5,
            0x1234,
            0x2000 + (offset / 15) * 40 + (offset % 15 / 5) * 11 + offset % 5,
            0x5678, length});
  }
}

TEST_F(TransferTest, FillRotatesPatternAcrossStagedEdgesAndImmediateBody) {
  const iree_hal_amd_gpu_dma_transfer_span_t span = {
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
    const iree_hal_amd_gpu_dma_transfer_span_t span = {
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
  const iree_hal_amd_gpu_dma_transfer_span_t span = {
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
    const iree_hal_amd_gpu_dma_transfer_span_t span = {
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

TEST(TransferBatchTest, HostTransfersUseAvailableStagingInOnePacket) {
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
    iree_hal_amd_gpu_transfer_operation_t operation = {
        .type = static_cast<uint32_t>(type),
        .length = host.size(),
    };
    auto& device = download ? operation.source.device : operation.target.device;
    device.buffer = &buffer;
    device.slot = {
        .index = 0,
        .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    (download ? operation.target.host : operation.source.host) = host.data();
    const iree_hal_amd_gpu_transfer_t transfer = {
        .count = 1,
        .operations = &operation,
    };
    iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
    IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
        &iree_hal_amd_gpu_pm4_transfer_encoding, &transfer,
        IREE_HAL_AMD_GPU_DMA_TRANSFER_MAX_WORDS,
        iree_make_byte_span(staging.data() + 1, kLength), kStagingAddress,
        &chunk));
    ASSERT_EQ(chunk.word_count, 7u);
    EXPECT_EQ(chunk.staging_length, kLength);
    EXPECT_EQ(chunk.next.index, 0u);
    EXPECT_EQ(chunk.next.offset, kLength);
    const uint64_t source = download ? kDeviceAddress : kStagingAddress;
    const uint64_t target = download ? kStagingAddress : kDeviceAddress;
    const std::array<uint32_t, 7> expected = {
        0xC0055000,
        0x60300000,
        static_cast<uint32_t>(source),
        static_cast<uint32_t>(source >> 32),
        static_cast<uint32_t>(target),
        static_cast<uint32_t>(target >> 32),
        kLength};
    EXPECT_EQ(
        std::vector<uint32_t>(chunk.words, chunk.words + chunk.word_count),
        std::vector<uint32_t>(expected.begin(), expected.end()));
    EXPECT_EQ(staging.front(), 0xCC);
    EXPECT_EQ(staging.back(), 0xCC);
    if (download) {
      EXPECT_EQ(chunk.download.source, staging.data() + 1);
      EXPECT_EQ(chunk.download.operation, &operation);
      EXPECT_EQ(chunk.download.offset, 0u);
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

TEST(TransferBatchTest, DownloadRunsShareStagingAndRetirement) {
  const iree_hal_buffer_native_binding_t binding = {.device_address = 0x10000};
  iree_hal_buffer_t buffer = {};
  buffer.memory.bindings = &binding;
  std::array<std::array<uint8_t, 10>, 3> targets = {};
  std::array<iree_hal_amd_gpu_transfer_operation_t, 4> operations = {};
  for (size_t i = 0; i < targets.size(); ++i) {
    auto& operation = operations[i];
    operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    operation.source.device.buffer = &buffer;
    operation.source.device.offset = 16 * i;
    operation.source.device.slot = {
        .index = 0,
        .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
    };
    operation.target.host = targets[i].data() + 1;
    operation.length = 8;
  }
  operations.back().type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  const iree_hal_amd_gpu_transfer_t transfer = {
      .count = operations.size(),
      .operations = operations.data(),
      .cursor = {.offset = 3},
  };
  for (uint32_t capacity : {0u, 1u, 5u, 6u, 13u, 14u, 21u, 22u}) {
    SCOPED_TRACE(capacity);
    std::array<uint8_t, 24> staging;
    staging.fill(0xCC);
    iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
    IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
        &iree_hal_amd_gpu_pm4_transfer_encoding, &transfer,
        IREE_HAL_AMD_GPU_DMA_TRANSFER_MAX_WORDS,
        iree_make_byte_span(staging.data() + 1, capacity), 0x20000, &chunk));
    const uint32_t length = std::min(capacity, 21u);
    const uint32_t packets = !length ? 0 : 1 + (length + 2) / 8;
    EXPECT_EQ(chunk.word_count, packets * 7);
    EXPECT_EQ(chunk.staging_length, length);
    EXPECT_EQ(chunk.download.length, length);
    EXPECT_EQ(chunk.next.index, (length + 3) / 8);
    EXPECT_EQ(chunk.next.offset, (length + 3) % 8);
    if (length) {
      EXPECT_EQ(chunk.download.source, staging.data() + 1);
      EXPECT_EQ(chunk.download.operation, operations.data());
      EXPECT_EQ(chunk.download.offset, 3u);
    }
    // CPU preparation never writes download staging. Each native destination
    // follows its predecessor tightly even when host targets are disjoint.
    EXPECT_TRUE(std::all_of(staging.begin(), staging.end(),
                            [](uint8_t byte) { return byte == 0xCC; }));
    if (length > 5) {
      EXPECT_EQ(chunk.words[7 + 2], 0x10010u);
      EXPECT_EQ(chunk.words[7 + 4], 0x20005u);
    }
  }
  operations[0].length = (UINT64_C(8) << 30) + 8;
  auto large_transfer = transfer;
  large_transfer.cursor.offset = (UINT64_C(8) << 30) + 3;
  std::array<uint8_t, 5> staging;
  iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
  IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
      &iree_hal_amd_gpu_pm4_transfer_encoding, &large_transfer, 7,
      iree_make_byte_span(staging.data(), staging.size()), 0x20000, &chunk));
  EXPECT_EQ(chunk.download.offset, large_transfer.cursor.offset);
  EXPECT_EQ(chunk.next.index, 1u);
  EXPECT_EQ(chunk.next.offset, 0u);
}

TEST(TransferBatchTest, DeviceCopiesUseCommandCapacityWithoutStaging) {
  constexpr uint64_t kLength = 64 * 1024 * 1024 + 1;
  const iree_hal_buffer_native_binding_t source_binding = {
      .device_address = UINT64_C(0x1234FFFFFFF0),
  };
  const iree_hal_buffer_native_binding_t target_binding = {
      .device_address = UINT64_C(0x567800000003),
  };
  iree_hal_buffer_t source = {}, target = {};
  source.memory.bindings = &source_binding;
  target.memory.bindings = &target_binding;
  iree_hal_amd_gpu_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .length = kLength,
  };
  operation.source.device = {
      .buffer = &source,
      .slot = {.index = 0, .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS},
  };
  operation.target.device = {
      .buffer = &target,
      .slot = {.index = 0, .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS},
  };
  iree_hal_amd_gpu_transfer_t transfer = {.count = 1, .operations = &operation};
  iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
  IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
      &iree_hal_amd_gpu_pm4_transfer_encoding, &transfer, 14,
      iree_byte_span_empty(), 0, &chunk));
  const std::vector<uint32_t> expected = {
      0xC0055000, 0x60300000, 0xFFFFFFF0, 0x1234, 3,         0x5678, 0x2000000,
      0xC0055000, 0x60300000, 0x1FFFFF0,  0x1235, 0x2000003, 0x5678, 0x2000000};
  ASSERT_EQ(chunk.word_count, expected.size());
  EXPECT_EQ(std::vector<uint32_t>(chunk.words, chunk.words + chunk.word_count),
            expected);
  EXPECT_EQ(chunk.next.index, 0u);
  EXPECT_EQ(chunk.next.offset, kLength - 1);
  EXPECT_EQ(chunk.staging_length, 0u);
  EXPECT_EQ(chunk.download.length, 0u);
  transfer.cursor = chunk.next;
  IREE_ASSERT_OK(iree_hal_amd_gpu_dma_transfer_prepare_chunk(
      &iree_hal_amd_gpu_pm4_transfer_encoding, &transfer, 7,
      iree_byte_span_empty(), 0, &chunk));
  ASSERT_EQ(chunk.word_count, 7u);
  EXPECT_EQ(std::vector<uint32_t>(chunk.words, chunk.words + chunk.word_count),
            (std::vector<uint32_t>{0xC0055000, 0x60300000, 0x3FFFFF0, 0x1235,
                                   0x4000003, 0x5678, 1}));
  EXPECT_EQ(chunk.next.index, 1u);
  EXPECT_EQ(chunk.next.offset, 0u);
  EXPECT_EQ(chunk.staging_length, 0u);
}

TEST_F(TransferTest, BulkFillPreservesPatternPhaseAcrossPacketBoundary) {
  const iree_hal_amd_gpu_dma_transfer_span_t span = {
      .target_address = 0x567800001001,
      .length = 32 * 1024 * 1024 + 7,
      .pattern = 0xA1B2C3D4,
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
  };
  auto packet = Emit(span, 0, 3);
  ASSERT_EQ(packet.length, 3u);
  EXPECT_EQ(packet.staging_length, 3u);
  EXPECT_EQ(staging_[0], 0xD4);
  EXPECT_EQ(staging_[1], 0xC3);
  EXPECT_EQ(staging_[2], 0xB2);
  packet = Emit(span, 3, 0);
  EXPECT_EQ(packet.length, 32u * 1024 * 1024);
  EXPECT_EQ(packet.staging_length, 0u);
  Expect({0xC0055000, 0x40300000, 0xB2C3D4A1, 0, 0x1004, 0x5678, 0x2000000});
  packet = Emit(span, 32 * 1024 * 1024 + 3, 0);
  EXPECT_EQ(packet.length, 4u);
  EXPECT_EQ(packet.staging_length, 0u);
  Expect({0xC0055000, 0x40300000, 0xB2C3D4A1, 0, 0x2001004, 0x5678, 4});
}

TEST(TransferBatchTest, ProgramCopiesRespectPacketArgumentsAndOperationBounds) {
  const iree_hal_amd_gpu_image_kernel_t kernel = {
      .entry_offset = 0x200,
      .compute_pgm_rsrc1 = 0xe0af0000,
      .compute_pgm_rsrc2 = 0x84,
      .compute_pgm_rsrc3 = 0x20,
      .code_properties = 0x408,
  };
  const iree_hal_buffer_native_binding_t bindings[] = {
      {.device_address = UINT64_C(0x1234FFFFFFF0)},
      {.device_address = UINT64_C(0x567800000000)},
  };
  iree_hal_buffer_t source = {}, target_buffer = {};
  source.memory.bindings = &bindings[0];
  target_buffer.memory.bindings = &bindings[1];
  std::array<iree_hal_amd_gpu_transfer_operation_t, 3> operations = {};
  for (size_t i = 0; i < 2; ++i) {
    operations[i] = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
        .source =
            {
                .device =
                    {
                        .buffer = &source,
                        .offset = 16,
                        .slot =
                            {
                                .index = 0,
                                .type =
                                    IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                            },
                    },
            },
        .target =
            {
                .device =
                    {
                        .buffer = &target_buffer,
                        .offset = (static_cast<uint64_t>(i) << 34) + 32,
                        .slot =
                            {
                                .index = 0,
                                .type =
                                    IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
                            },
                    },
            },
        .length = i ? 1025 : (UINT64_C(8) << 30) + 17,
    };
  }
  operations[2].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  const iree_hal_amd_gpu_transfer_t transfer = {
      .count = operations.size(),
      .operations = operations.data(),
      .cursor = {.index = 0, .offset = 3},
  };
  for (auto target : {iree_hal_amd_gpu_gfxip_version_t{11, 0, 0},
                      {11, 5, 1},
                      {12, 0, 1},
                      {12, 5, 0}}) {
    iree_hal_amd_gpu_pm4_launch_t launch;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(
        target, 0, &kernel, UINT64_C(0x123456780000), &launch));
    for (uint32_t word_capacity : {0u, 30u, 31u, 61u, 62u, 224u}) {
      for (size_t byte_capacity : {0u, 79u, 80u, 159u, 160u, 240u}) {
        SCOPED_TRACE(::testing::Message() << target.major << '.' << target.minor
                                          << " words=" << word_capacity
                                          << " bytes=" << byte_capacity);
        std::array<uint8_t, 242> bytes;
        bytes.fill(0xCC);
        iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
        std::memset(&chunk, 0xA5, sizeof(chunk));
        uint32_t argument_length = UINT32_MAX;
        IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_transfer_prepare_program_chunk(
            &transfer, &launch, word_capacity,
            iree_make_byte_span(bytes.data() + 1, byte_capacity),
            UINT64_C(0xABCDEF010020), &chunk, &argument_length));
        const size_t count = std::min(
            {size_t{2}, size_t{word_capacity / 31},
             byte_capacity / sizeof(iree_hal_amd_gpu_blit_arguments_t)});
        EXPECT_EQ(chunk.word_count, count * 31);
        EXPECT_EQ(argument_length,
                  count * sizeof(iree_hal_amd_gpu_blit_arguments_t));
        EXPECT_EQ(chunk.staging_length, 0u);
        EXPECT_EQ(chunk.next.index, count);
        EXPECT_EQ(chunk.next.offset, count ? 0u : 3u);
        EXPECT_EQ(chunk.download.length, 0u);
        EXPECT_EQ(bytes.front(), 0xCC);
        for (size_t i = 0; i < count; ++i) {
          const std::array<uint32_t, 31> expected = {
              0xC0027602,
              0x20C,
              0x34567802,
              0x12,
              0xC0027602,
              0x212,
              0xE0AF0000,
              0x84,
              0xC0017602,
              target.major == 12 && target.minor == 5 ? 0x223u : 0x228u,
              0x20,
              0xC0017602,
              0x215,
              0,
              0xC0067602,
              0x204,
              0,
              0,
              0,
              64,
              1,
              1,
              0xC0027602,
              0x240,
              0xEF010020u + static_cast<uint32_t>(
                                i * sizeof(iree_hal_amd_gpu_blit_arguments_t)),
              0xABCD,
              0xC0031502,
              i ? 2u : 256u,
              1,
              1,
              0x8005,
          };
          EXPECT_EQ(std::vector<uint32_t>(chunk.words + i * 31,
                                          chunk.words + (i + 1) * 31),
                    std::vector<uint32_t>(expected.begin(), expected.end()));
          iree_hal_amd_gpu_blit_arguments_t arguments;
          std::memcpy(
              &arguments,
              bytes.data() + 1 + i * sizeof(iree_hal_amd_gpu_blit_arguments_t),
              sizeof(arguments));
          const uint64_t offset = i ? 0 : 3;
          EXPECT_EQ(arguments.source, UINT64_C(0x123500000000) + offset);
          EXPECT_EQ(arguments.target, UINT64_C(0x567800000020) +
                                          (static_cast<uint64_t>(i) << 34) +
                                          offset);
          EXPECT_EQ(arguments.length, operations[i].length - offset);
          EXPECT_EQ(arguments.pattern, 0u);
          EXPECT_EQ(arguments.fill, 0u);
          EXPECT_EQ(arguments.row_count, 1u);
          EXPECT_EQ(arguments.slice_count, 1u);
          EXPECT_EQ(arguments.groups_y, 1u);
          EXPECT_EQ(arguments.groups_z, 1u);
        }
        EXPECT_TRUE(std::all_of(
            bytes.begin() + 1 +
                count * sizeof(iree_hal_amd_gpu_blit_arguments_t),
            bytes.end(), [](uint8_t byte) { return byte == 0xCC; }));
        EXPECT_TRUE(
            std::all_of(chunk.words + count * 31,
                        chunk.words + IREE_HAL_AMD_GPU_DMA_TRANSFER_MAX_WORDS,
                        [](uint32_t word) { return word == 0xA5A5A5A5; }));
      }
    }
  }
}

TEST(TransferBatchTest, ProgramRectanglesKeepOneLaunchForCopyAndFill) {
  iree_hal_amd_gpu_image_kernel_t kernel;
  iree_const_byte_span_t code;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_blit_program_query({11, 0, 0}, &code, &kernel));
  iree_hal_amd_gpu_pm4_launch_t launch;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize({11, 0, 0}, 0, &kernel,
                                                        0x100000, &launch));
  const iree_hal_buffer_native_binding_t binding = {.device_address = 0x1000};
  iree_hal_buffer_t buffer = {};
  buffer.memory.bindings = &binding;
  const iree_hal_transfer_region_t region = {
      .row_length = 37,
      .row_count = 128,
      .slice_count = 4,
      .source_strides = {.row_stride = 48, .slice_stride = 8192},
      .target_strides = {.row_stride = 64, .slice_stride = 12288},
  };
  for (auto type : {IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_FILL}) {
    SCOPED_TRACE(type);
    iree_hal_amd_gpu_transfer_operation_t operation = {
        .type = static_cast<uint32_t>(type),
        .rectangular = true,
        .pattern = 0xA5A5A5A5,
        .region = &region,
    };
    operation.source.device = {
        .buffer = &buffer,
        .offset = 16,
        .slot = {.index = 0, .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS},
    };
    operation.target.device = operation.source.device;
    operation.target.device.offset = 32;
    const iree_hal_amd_gpu_transfer_t transfer = {
        .count = 1,
        .operations = &operation,
    };
    iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
    iree_hal_amd_gpu_blit_arguments_t arguments;
    uint32_t argument_length = 0;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_transfer_prepare_program_chunk(
        &transfer, &launch, 31,
        iree_make_byte_span(&arguments, sizeof(arguments)), 0x2000, &chunk,
        &argument_length));
    EXPECT_EQ(chunk.word_count, 31u);
    EXPECT_EQ(chunk.next.index, 1u);
    EXPECT_EQ(argument_length, sizeof(arguments));
    EXPECT_EQ(arguments.target, 0x1020u);
    EXPECT_EQ(arguments.length, 37u);
    EXPECT_EQ(arguments.row_count, 128u);
    EXPECT_EQ(arguments.slice_count, 4u);
    EXPECT_EQ(arguments.target_row_stride, 64u);
    EXPECT_EQ(arguments.target_slice_stride, 12288u);
    EXPECT_EQ(arguments.fill, type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL);
    const std::array<uint32_t, 4> dispatch = {0xC0031502, 1, 128, 2};
    EXPECT_EQ(std::vector<uint32_t>(chunk.words + 26, chunk.words + 30),
              std::vector<uint32_t>(dispatch.begin(), dispatch.end()));
  }
}

TEST(TransferBatchTest, StagedRectanglesPreserveEveryPartialWindow) {
  iree_hal_amd_gpu_image_kernel_t kernel;
  iree_const_byte_span_t code;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_blit_program_query({11, 0, 0}, &code, &kernel));
  iree_hal_amd_gpu_pm4_launch_t launch;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize({11, 0, 0}, 0, &kernel,
                                                        0x100000, &launch));
  const iree_hal_buffer_native_binding_t binding = {.device_address = 0x1000};
  iree_hal_buffer_t buffer = {};
  buffer.memory.bindings = &binding;
  std::array<uint8_t, 200> input, output;
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = (uint8_t)(i * 13);
  }
  for (auto type : {IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
                    IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD}) {
    SCOPED_TRACE(type);
    const bool download = type == IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    const iree_hal_transfer_region_t region = {
        .row_length = 13,
        .row_count = 3,
        .slice_count = 2,
        .source_strides = type == IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE
                              ? iree_hal_transfer_strides_t{13, 39}
                              : iree_hal_transfer_strides_t{19, 67},
        .target_strides = {23, 83},
    };
    iree_hal_amd_gpu_transfer_operation_t operation = {
        .type = static_cast<uint32_t>(type),
        .rectangular = true,
        .region = &region,
    };
    auto& device = download ? operation.source.device : operation.target.device;
    device = {
        .buffer = &buffer,
        .offset = 16,
        .slot = {.index = 0, .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS},
    };
    if (download) {
      operation.target.host = output.data();
    } else {
      operation.source.host = input.data();
    }
    for (uint64_t offset : {0u, 5u, 13u, 38u, 39u, 77u}) {
      SCOPED_TRACE(offset);
      for (size_t capacity : {0u, 1u, 7u, 13u, 14u, 25u, 39u, 78u}) {
        SCOPED_TRACE(capacity);
        std::array<uint8_t, 256> storage;
        storage.fill(0xCD);
        output.fill(0xCD);
        auto expected = output;
        const iree_hal_amd_gpu_transfer_t transfer = {
            .count = 1,
            .operations = &operation,
            .cursor = {.index = 0, .offset = offset},
        };
        iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
        uint32_t argument_length = 0;
        IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_transfer_prepare_program_chunk(
            &transfer, &launch, 31,
            iree_make_byte_span(
                storage.data(),
                sizeof(iree_hal_amd_gpu_blit_arguments_t) + capacity),
            0x2000, &chunk, &argument_length));
        if (!capacity) {
          EXPECT_EQ(chunk.word_count, 0u);
          EXPECT_EQ(chunk.next.offset, offset);
          EXPECT_EQ(argument_length, 0u);
          EXPECT_TRUE(std::all_of(storage.begin(), storage.end(),
                                  [](uint8_t byte) { return byte == 0xCD; }));
          continue;
        }
        iree_hal_amd_gpu_blit_arguments_t arguments;
        std::memcpy(&arguments, storage.data(), sizeof(arguments));
        EXPECT_EQ(chunk.word_count, 31u);
        EXPECT_EQ(argument_length, sizeof(arguments));
        ASSERT_GT(chunk.staging_length, 0u);
        ASSERT_LE(chunk.staging_length,
                  std::min(capacity, size_t(78 - offset)));
        EXPECT_EQ(chunk.staging_length, arguments.length * arguments.row_count *
                                            arguments.slice_count);
        EXPECT_EQ(download ? arguments.target : arguments.source,
                  0x2000u + sizeof(arguments));
        size_t consumed = 0;
        for (uint32_t z = 0; z < arguments.slice_count; ++z) {
          for (uint32_t y = 0; y < arguments.row_count; ++y) {
            for (uint64_t x = 0; x < arguments.length; ++x, ++consumed) {
              const uint64_t logical = offset + consumed;
              const uint64_t source =
                  (logical / 39) * region.source_strides.slice_stride +
                  ((logical % 39) / 13) * region.source_strides.row_stride +
                  logical % 13;
              const uint64_t target = (logical / 39) * 83 +
                                      ((logical % 39) / 13) * 23 + logical % 13;
              EXPECT_EQ(
                  download
                      ? arguments.source + z * arguments.source_slice_stride +
                            y * arguments.source_row_stride + x
                      : arguments.target + z * arguments.target_slice_stride +
                            y * arguments.target_row_stride + x,
                  0x1010u + (download ? source : target));
              if (download) {
                storage[sizeof(arguments) + consumed] = input[source];
                expected[target] = input[source];
              } else {
                EXPECT_EQ(storage[sizeof(arguments) + consumed], input[source]);
              }
            }
          }
        }
        EXPECT_EQ(chunk.next.index ? 78u : chunk.next.offset,
                  offset + chunk.staging_length);
        EXPECT_TRUE(std::all_of(
            storage.begin() + argument_length + chunk.staging_length,
            storage.end(), [](uint8_t byte) { return byte == 0xCD; }));
        if (download) {
          iree_hal_amd_gpu_download_complete(&chunk.download);
          EXPECT_EQ(output, expected);
        }
      }
    }
  }
}

TEST(TransferBatchTest, StagedBatchSeparatesArgumentsFromDownloadBytes) {
  iree_hal_amd_gpu_image_kernel_t kernel;
  iree_const_byte_span_t code;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_blit_program_query({11, 0, 0}, &code, &kernel));
  iree_hal_amd_gpu_pm4_launch_t launch;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize({11, 0, 0}, 0, &kernel,
                                                        0x100000, &launch));
  const iree_hal_buffer_native_binding_t binding = {.device_address = 0x1000};
  iree_hal_buffer_t buffer = {};
  buffer.memory.bindings = &binding;
  const iree_hal_transfer_region_t region = {
      .row_length = 3,
      .row_count = 2,
      .slice_count = 1,
      .source_strides = {5, 10},
      .target_strides = {7, 14},
  };
  std::array<uint8_t, 16> input = {1, 2, 3, 0, 0, 4,  5,  6,
                                   7, 8, 9, 0, 0, 10, 11, 12};
  std::array<uint8_t, 14> output;
  output.fill(0xCD);
  std::array<iree_hal_amd_gpu_transfer_operation_t, 4> operations = {};
  for (size_t i = 0; i < operations.size(); ++i) {
    auto& operation = operations[i];
    operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    operation.rectangular = true;
    operation.region = &region;
    operation.source.host = input.data() + (i % 2) * 8;
    operation.target.device = {
        .buffer = &buffer,
        .offset = i * 32,
        .slot = {.index = 0, .type = IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS},
    };
  }
  operations[2].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  operations[2].source.device = operations[2].target.device;
  operations[2].target.host = output.data();
  const iree_hal_amd_gpu_transfer_t transfer = {
      .count = operations.size(),
      .operations = operations.data(),
  };
  std::array<uint8_t, 512> storage;
  storage.fill(0xCD);
  iree_hal_amd_gpu_dma_transfer_chunk_t chunk;
  uint32_t argument_length = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_transfer_prepare_program_chunk(
      &transfer, &launch, 124,
      iree_make_byte_span(storage.data(), storage.size()), 0x2000, &chunk,
      &argument_length));
  EXPECT_EQ(chunk.word_count, 93u);
  EXPECT_EQ(chunk.next.index, 3u);
  EXPECT_EQ(chunk.next.offset, 0u);
  EXPECT_EQ(chunk.staging_length, 18u);
  EXPECT_EQ(argument_length, 260u);
  for (size_t i = 0; i < 3; ++i) {
    iree_hal_amd_gpu_blit_arguments_t arguments;
    std::memcpy(&arguments, storage.data() + i * 96, sizeof(arguments));
    EXPECT_EQ(i == 2 ? arguments.target : arguments.source,
              0x2000u + i * 96 + 80);
    EXPECT_EQ(i == 2 ? arguments.source : arguments.target, 0x1000u + i * 32);
    if (i < 2) {
      for (size_t j = 0; j < 6; ++j) {
        EXPECT_EQ(storage[i * 96 + 80 + j], i * 6 + j + 1);
      }
      EXPECT_TRUE(std::all_of(storage.begin() + i * 96 + 86,
                              storage.begin() + (i + 1) * 96,
                              [](uint8_t byte) { return byte == 0xCD; }));
    }
  }
  ASSERT_EQ(chunk.download.source, storage.data() + 272);
  EXPECT_EQ(chunk.download.operation, &operations[2]);
  ASSERT_EQ(chunk.download.length, 6u);
  std::memcpy(storage.data() + 272, "abcdef", 6);
  iree_hal_amd_gpu_download_complete(&chunk.download);
  const std::array<uint8_t, 14> expected = {'a',  'b',  'c',  0xCD, 0xCD,
                                            0xCD, 0xCD, 'd',  'e',  'f',
                                            0xCD, 0xCD, 0xCD, 0xCD};
  EXPECT_EQ(output, expected);
  EXPECT_TRUE(std::all_of(storage.begin() + 278, storage.end(),
                          [](uint8_t byte) { return byte == 0xCD; }));
}

}  // namespace
