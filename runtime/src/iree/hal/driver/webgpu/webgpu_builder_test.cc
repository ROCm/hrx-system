// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/webgpu/webgpu_builder.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class WebGPUBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(512, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(iree_hal_webgpu_builder_initialize(
        &block_pool_, 2, iree_allocator_system(), &builder_));
  }
  void TearDown() override {
    iree_hal_webgpu_builder_deinitialize(&builder_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static uint64_t ReadU64(const uint32_t* words) {
    return uint64_t(words[0]) | (uint64_t(words[1]) << 32);
  }

  // The same block walk used at the wasm boundary: instruction headers, rather
  // than command-specific lengths, must make every block traversable.
  template <typename Visitor>
  void Walk(Visitor visitor) {
    for (uint32_t block = 0; block < builder_.block_count; ++block) {
      const uint32_t count = block + 1 == builder_.block_count
                                 ? builder_.cursor
                                 : builder_.block_word_capacity;
      const uint32_t* words = builder_.block_ptrs[block];
      uint32_t position = 0;
      while (position < count) {
        const uint32_t size =
            iree_hal_webgpu_isa_header_size_words(words[position]);
        ASSERT_GT(size, 0u);
        ASSERT_LE(size, count - position);
        visitor(words + position);
        position += size;
      }
    }
  }

  // Fixed recording storage, deliberately small enough to exercise block tails.
  iree_arena_block_pool_t block_pool_ = {};
  // Production instruction encoder under test.
  iree_hal_webgpu_builder_t builder_ = {};
};

TEST_F(WebGPUBuilderTest, UpdateCapturesUsefulBytesAcrossBlockBoundaries) {
  const uint8_t pattern = 0xCD;
  const iree_hal_fill_region_t fill_region = {
      .row_length = 16384,
      .row_count = 1,
      .slice_count = 1,
  };
  IREE_ASSERT_OK(iree_hal_webgpu_builder_fill_buffer(
      &builder_, iree_hal_make_indirect_buffer_ref(0, 0, 16384), &pattern,
      sizeof(pattern), &fill_region));
  const iree_hal_transfer_region_t region = {
      .row_length = 37,
      .row_count = 128,
      .slice_count = 2,
      .source_strides = {.row_stride = 48, .slice_stride = 6144},
      .target_strides = {.row_stride = 64, .slice_stride = 8192},
  };
  std::vector<uint8_t> source(12288);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = uint8_t(i * 19 + 5);
  }
  std::vector<uint8_t> expected(16384, 0xCD);
  for (size_t slice = 0; slice < 2; ++slice) {
    for (size_t row = 0; row < 128; ++row) {
      memcpy(expected.data() + slice * 8192 + row * 64,
             source.data() + slice * 6144 + row * 48, 37);
    }
  }
  IREE_ASSERT_OK(iree_hal_webgpu_builder_update_buffer(
      &builder_, source.data(), 0,
      iree_hal_make_indirect_buffer_ref(0, 0, expected.size()), &region));
  std::fill(source.begin(), source.end(), 0xEE);
  IREE_ASSERT_OK(iree_hal_webgpu_builder_finalize(&builder_));
  ASSERT_GT(builder_.block_count, 1u);

  std::vector<uint8_t> actual(16384, 0xCD);
  size_t captured_bytes = 0;
  Walk([&](const uint32_t* words) {
    if (iree_hal_webgpu_isa_header_opcode(words[0]) !=
        IREE_HAL_WEBGPU_ISA_OP_UPDATE_BUFFER) {
      return;
    }
    ASSERT_EQ(iree_hal_webgpu_isa_header_flags(words[0]),
              IREE_HAL_WEBGPU_ISA_TRANSFER_FLAG_RECTANGULAR);
    iree_hal_transfer_region_t chunk;
    memcpy(&chunk, words + 5, sizeof(chunk));
    const uint8_t* data = reinterpret_cast<const uint8_t*>(words + 19);
    size_t cursor = 0;
    for (uint64_t slice = 0; slice < chunk.slice_count; ++slice) {
      for (uint64_t row = 0; row < chunk.row_count; ++row) {
        const uint64_t offset = ReadU64(words + 2) +
                                slice * chunk.target_strides.slice_stride +
                                row * chunk.target_strides.row_stride;
        ASSERT_LE(offset + chunk.row_length, actual.size());
        memcpy(actual.data() + offset, data + cursor, chunk.row_length);
        cursor += chunk.row_length;
      }
    }
    EXPECT_EQ(cursor, words[4]);
    captured_bytes += cursor;
  });
  EXPECT_EQ(captured_bytes, 37u * 128 * 2);
  EXPECT_EQ(actual, expected);
}

TEST_F(WebGPUBuilderTest, PreservesWideCopyAndDispatchWindows) {
  const iree_hal_transfer_region_t region = {
      .row_length = 37,
      .row_count = 2,
      .slice_count = 2,
      .source_strides =
          {
              .row_stride = UINT64_C(1) << 33,
              .slice_stride = UINT64_C(1) << 40,
          },
      .target_strides =
          {
              .row_stride = UINT64_C(1) << 34,
              .slice_stride = UINT64_C(1) << 41,
          },
  };
  const uint64_t source_offset = (UINT64_C(1) << 35) + 3;
  const uint64_t target_offset = (UINT64_C(1) << 36) + 5;
  IREE_ASSERT_OK(iree_hal_webgpu_builder_copy_buffer(
      &builder_,
      iree_hal_make_indirect_buffer_ref(0, source_offset, UINT64_C(1) << 42),
      iree_hal_make_indirect_buffer_ref(1, target_offset, UINT64_C(1) << 43),
      &region));
  // Leave too little room for the following dispatch, including its bindings.
  while (builder_.cursor + 10 < builder_.block_word_capacity) {
    IREE_ASSERT_OK(iree_hal_webgpu_builder_execution_barrier(&builder_));
  }
  const iree_hal_buffer_ref_t bindings[] = {
      iree_hal_make_indirect_buffer_ref(0, source_offset, UINT64_C(1) << 42),
      iree_hal_make_indirect_buffer_ref(1, target_offset, UINT64_C(1) << 43),
  };
  const uint32_t groups[] = {2, 3, 4};
  IREE_ASSERT_OK(
      iree_hal_webgpu_builder_dispatch(&builder_, 7, 8, groups, {2, bindings}));
  IREE_ASSERT_OK(iree_hal_webgpu_builder_finalize(&builder_));
  size_t copies = 0;
  size_t dispatches = 0;
  Walk([&](const uint32_t* words) {
    switch (iree_hal_webgpu_isa_header_opcode(words[0])) {
      case IREE_HAL_WEBGPU_ISA_OP_COPY_BUFFER:
        EXPECT_EQ(ReadU64(words + 2), source_offset);
        EXPECT_EQ(ReadU64(words + 5), target_offset);
        EXPECT_EQ(0, memcmp(words + 9, &region, sizeof(region)));
        ++copies;
        break;
      case IREE_HAL_WEBGPU_ISA_OP_DISPATCH:
        EXPECT_EQ(iree_hal_webgpu_isa_header_size_words(words[0]), 16u);
        EXPECT_EQ(ReadU64(words + 7), source_offset);
        EXPECT_EQ(ReadU64(words + 9), UINT64_C(1) << 42);
        EXPECT_EQ(ReadU64(words + 12), target_offset);
        EXPECT_EQ(ReadU64(words + 14), UINT64_C(1) << 43);
        ++dispatches;
        break;
    }
  });
  EXPECT_EQ(copies, 1u);
  EXPECT_EQ(dispatches, 1u);
}

}  // namespace
