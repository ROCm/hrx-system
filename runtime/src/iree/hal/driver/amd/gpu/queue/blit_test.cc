// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/blit.h"

#include <cstring>
#include <vector>

#include "iree/base/byte_sequence.h"
#include "iree/hal/driver/amd/gpu/image/hsaco.h"
#include "iree/hal/driver/amd/gpu/queue/blit_programs.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(BlitGeometryTest, ChunkedGridCoversUsefulBytesWithoutPadding) {
  for (uint64_t width : {1u, 15u, 37u}) {
    for (uint64_t rows : {1u, 3u}) {
      for (uint64_t slices : {1u, 4u}) {
        const iree_hal_transfer_region_t region = {
            .row_length = width,
            .row_count = rows,
            .slice_count = slices,
            .source_strides = {width + 7, (width + 7) * rows + 11},
            .target_strides = {width + 13, (width + 13) * rows + 23},
        };
        const uint64_t total = width * rows * slices;
        for (uint64_t capacity : {1u, 13u, 31u, 64u, 4096u}) {
          SCOPED_TRACE(::testing::Message()
                       << width << 'x' << rows << 'x' << slices
                       << " capacity=" << capacity);
          uint64_t cursor = 0;
          while (cursor < total) {
            iree_hal_amd_gpu_blit_arguments_t arguments;
            const uint64_t length = iree_hal_amd_gpu_blit_prepare_region(
                &region, cursor, capacity, &arguments);
            ASSERT_GT(length, 0u);
            ASSERT_LE(length, capacity);
            ASSERT_LE(length, total - cursor);
            EXPECT_LE(iree_hal_amd_gpu_blit_workgroup_count(arguments.length) *
                          arguments.groups_y * arguments.groups_z,
                      256u);
            uint64_t emitted = 0;
            for (uint32_t z = 0; z < arguments.slice_count; ++z) {
              for (uint32_t y = 0; y < arguments.row_count; ++y) {
                for (uint64_t x = 0; x < arguments.length; ++x) {
                  const uint64_t logical = cursor + emitted++;
                  const uint64_t row = logical / width % rows;
                  const uint64_t slice = logical / (width * rows);
                  const uint64_t column = logical % width;
                  EXPECT_EQ(
                      arguments.source + z * arguments.source_slice_stride +
                          y * arguments.source_row_stride + x,
                      slice * region.source_strides.slice_stride +
                          row * region.source_strides.row_stride + column);
                  EXPECT_EQ(
                      arguments.target + z * arguments.target_slice_stride +
                          y * arguments.target_row_stride + x,
                      slice * region.target_strides.slice_stride +
                          row * region.target_strides.row_stride + column);
                }
              }
            }
            EXPECT_EQ(emitted, length);
            cursor += length;
          }
        }
      }
    }
  }
}

TEST(BlitGeometryTest, PublicDimensionsExceedNativeSignedIndices) {
  constexpr uint64_t kCount = UINT64_C(1) << 32;
  iree_hal_transfer_region_t region = {
      .row_length = 1,
      .row_count = kCount + 7,
      .slice_count = 2,
      .source_strides = {2, (kCount + 7) * 2},
      .target_strides = {3, (kCount + 7) * 3},
  };
  uint64_t cursor = 0;
  unsigned chunks = 0;
  while (cursor < region.row_count * region.slice_count) {
    iree_hal_amd_gpu_blit_arguments_t arguments;
    const uint64_t length = iree_hal_amd_gpu_blit_prepare_region(
        &region, cursor, UINT64_MAX, &arguments);
    ASSERT_GT(length, 0u);
    EXPECT_LE(arguments.row_count - 1 + arguments.groups_y,
              uint32_t{INT32_MAX});
    EXPECT_EQ(arguments.source, cursor * 2);
    EXPECT_EQ(arguments.target, cursor * 3);
    cursor += length;
    ++chunks;
  }
  EXPECT_EQ(chunks, 6u);
  region.row_count = 1;
  region.slice_count = kCount + 7;
  region.source_strides.slice_stride = 5;
  region.target_strides.slice_stride = 7;
  cursor = 0;
  chunks = 0;
  while (cursor < region.slice_count) {
    iree_hal_amd_gpu_blit_arguments_t arguments;
    const uint64_t length = iree_hal_amd_gpu_blit_prepare_region(
        &region, cursor, UINT64_MAX, &arguments);
    ASSERT_GT(length, 0u);
    EXPECT_LE(arguments.slice_count - 1 + arguments.groups_z,
              uint32_t{INT32_MAX});
    EXPECT_EQ(arguments.source, cursor * region.source_strides.slice_stride);
    cursor += length;
    ++chunks;
  }
  EXPECT_EQ(chunks, 3u);
}

class BlitTest : public ::testing::TestWithParam<size_t> {
 protected:
  void TearDown() override { iree_hal_amd_gpu_image_deinitialize(&image_); }

  // Normalized source program, retained through every assertion.
  iree_hal_amd_gpu_image_t image_ = {};
};

TEST_P(BlitTest, EmbeddedInstructionsAndDescriptorMatchSource) {
  const auto& file = iree_hal_amd_gpu_blit_programs_create()[GetParam()];
  SCOPED_TRACE(file.name);
  iree_byte_span_t source = {nullptr, file.size};
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(), file.size,
                                       reinterpret_cast<void**>(&source.data)));
  memcpy(source.data, file.data, file.size);
  iree_byte_sequence_t* sequence = nullptr;
  iree_status_t status = iree_byte_sequence_create_from_span_move(
      &source, iree_allocator_system(), &sequence);
  iree_allocator_free(iree_allocator_system(), source.data);
  IREE_ASSERT_OK(status);
  status = iree_hal_amd_gpu_hsaco_image_initialize(
      sequence, iree_allocator_system(), &image_);
  iree_byte_sequence_release(sequence);
  IREE_ASSERT_OK(status);

  ASSERT_EQ(image_.metadata.kernel_count, 1u);
  ASSERT_EQ(image_.relocation_count, 0u);
  ASSERT_EQ(image_.import_count, 0u);
  const auto& metadata = image_.metadata.kernels[0];
  EXPECT_EQ(metadata.kernarg_segment_size,
            sizeof(iree_hal_amd_gpu_blit_arguments_t));
  EXPECT_EQ(metadata.private_segment_fixed_size, 0u);
  EXPECT_EQ(metadata.group_segment_fixed_size, 0u);
  ASSERT_EQ(metadata.argument_count, 13u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(metadata.arguments[i].offset, i * 8);
    EXPECT_EQ(metadata.arguments[i].size, 8u);
  }
  EXPECT_EQ(metadata.arguments[3].offset, 24u);
  EXPECT_EQ(metadata.arguments[3].size, 4u);
  EXPECT_EQ(metadata.arguments[4].offset, 28u);
  EXPECT_EQ(metadata.arguments[4].size, 4u);
  for (size_t i = 5; i < 9; ++i) {
    EXPECT_EQ(metadata.arguments[i].offset, 32 + (i - 5) * 8);
    EXPECT_EQ(metadata.arguments[i].size, 8u);
  }
  for (size_t i = 9; i < 13; ++i) {
    EXPECT_EQ(metadata.arguments[i].offset, 64 + (i - 9) * 4);
    EXPECT_EQ(metadata.arguments[i].size, 4u);
  }
  iree_const_byte_span_t code;
  iree_hal_amd_gpu_image_kernel_t descriptor;
  IREE_ASSERT_OK(iree_hal_amd_gpu_blit_program_query(image_.target.version,
                                                     &code, &descriptor));
  const auto& kernel = image_.kernels[0];
  EXPECT_EQ(kernel.compute_pgm_rsrc1, descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(kernel.compute_pgm_rsrc2, descriptor.compute_pgm_rsrc2);
  EXPECT_EQ(kernel.compute_pgm_rsrc3, descriptor.compute_pgm_rsrc3);
  EXPECT_EQ(kernel.code_properties, descriptor.code_properties);
  EXPECT_EQ(kernel.kernarg_preload, descriptor.kernarg_preload);
  const auto* symbol =
      iree_hal_amd_gpu_image_find_symbol(&image_, IREE_SV("blit"));
  ASSERT_NE(symbol, nullptr);
  ASSERT_EQ(symbol->length, code.data_length);
  EXPECT_EQ(symbol->address.offset, kernel.entry_offset);
  std::vector<uint8_t> storage(image_.byte_length);
  std::vector<iree_byte_span_t> mappings;
  for (size_t i = 0; i < image_.segment_count; ++i) {
    const auto& segment = image_.segments[i];
    mappings.push_back(
        iree_make_byte_span(storage.data() + segment.offset, segment.length));
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_image_load(&image_, 0x100000000, 0, nullptr,
                                             mappings.size(), mappings.data()));
  EXPECT_EQ(
      memcmp(storage.data() + kernel.entry_offset, code.data, code.data_length),
      0);
}

INSTANTIATE_TEST_SUITE_P(
    Targets, BlitTest,
    ::testing::Range(size_t{0}, iree_hal_amd_gpu_blit_programs_size()));

}  // namespace
