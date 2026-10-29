// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/gpu/dispatch.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/queue/grid_sync_programs.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class GridSyncProgramTest : public ::testing::TestWithParam<size_t> {
 protected:
  void TearDown() override { iree_hal_amd_gpu_image_deinitialize(&image_); }

  // Source-compiled descriptor and instructions, owned through each assertion.
  iree_hal_amd_gpu_image_t image_ = {};
};

TEST_P(GridSyncProgramTest, EmbeddedInstructionsAndAbiMatchConfiguredSource) {
  const auto& file = iree_hal_amd_gpu_grid_sync_programs_create()[GetParam()];
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
  if (image_.target.version.major != 9 && image_.target.version.major != 10) {
    EXPECT_EQ(image_.metadata.kernel_count, 0u);
    return;
  }
  ASSERT_EQ(image_.metadata.kernel_count, 1u);
  ASSERT_EQ(image_.relocation_count, 0u);
  ASSERT_EQ(image_.import_count, 0u);
  const auto& metadata = image_.metadata.kernels[0];
  EXPECT_EQ(metadata.kernarg_segment_size,
            sizeof(iree_hal_amd_gpu_grid_sync_t));
  EXPECT_EQ(metadata.kernarg_segment_alignment, 8u);
  EXPECT_EQ(metadata.private_segment_fixed_size, 0u);
  EXPECT_EQ(metadata.group_segment_fixed_size, 0u);
  ASSERT_EQ(metadata.argument_count, 1u);
  EXPECT_EQ(metadata.arguments[0].offset, 0u);
  EXPECT_EQ(metadata.arguments[0].size, sizeof(iree_hal_amd_gpu_grid_sync_t));
  EXPECT_EQ(offsetof(iree_hal_amd_gpu_grid_sync_t, workgroup_count), 40u);

  iree_const_byte_span_t code;
  iree_hal_amd_gpu_image_kernel_t expected;
  IREE_ASSERT_OK(iree_hal_amd_gpu_grid_sync_program_query(image_.target.version,
                                                          &code, &expected));
  const auto& kernel = image_.kernels[0];
  EXPECT_EQ(kernel.compute_pgm_rsrc1, expected.compute_pgm_rsrc1);
  EXPECT_EQ(kernel.compute_pgm_rsrc2, expected.compute_pgm_rsrc2);
  EXPECT_EQ(kernel.compute_pgm_rsrc3, expected.compute_pgm_rsrc3);
  EXPECT_EQ(kernel.code_properties, expected.code_properties);
  EXPECT_EQ(kernel.kernarg_preload, expected.kernarg_preload);
  const auto* symbol = iree_hal_amd_gpu_image_find_symbol(
      &image_, IREE_SV("initialize_grid_sync"));
  ASSERT_NE(symbol, nullptr);
  ASSERT_EQ(symbol->length, code.data_length);
  ASSERT_EQ(symbol->address.offset, kernel.entry_offset);
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
    Targets, GridSyncProgramTest,
    ::testing::Range(size_t{0}, iree_hal_amd_gpu_grid_sync_programs_size()));

}  // namespace
