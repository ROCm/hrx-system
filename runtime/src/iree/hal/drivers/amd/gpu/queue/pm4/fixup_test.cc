// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/fixup.h"

#include <cstring>
#include <vector>

#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/fixup_programs.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class FixupTest : public ::testing::TestWithParam<size_t> {
 protected:
  void TearDown() override { iree_hal_amd_gpu_image_deinitialize(&image_); }

  // Normalized source program, retained through every assertion.
  iree_hal_amd_gpu_image_t image_ = {};
};

TEST_P(FixupTest, EmbeddedInstructionsAndDescriptorMatchSource) {
  const auto& file = iree_hal_amd_gpu_pm4_fixup_programs_create()[GetParam()];
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
  EXPECT_EQ(metadata.kernarg_segment_size, 32u);
  EXPECT_EQ(metadata.private_segment_fixed_size, 0u);
  EXPECT_EQ(metadata.group_segment_fixed_size, 0u);
  ASSERT_EQ(metadata.argument_count, 4u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(metadata.arguments[i].offset, i * 8);
    EXPECT_EQ(metadata.arguments[i].size, 8u);
  }
  EXPECT_EQ(metadata.arguments[3].offset, 24u);
  EXPECT_EQ(metadata.arguments[3].size, 4u);
  iree_const_byte_span_t code;
  iree_hal_amd_gpu_image_kernel_t descriptor;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_fixup_program_query(image_.target.version,
                                                          &code, &descriptor));
  const auto& kernel = image_.kernels[0];
  EXPECT_EQ(kernel.compute_pgm_rsrc1, descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(kernel.compute_pgm_rsrc2, descriptor.compute_pgm_rsrc2);
  EXPECT_EQ(kernel.compute_pgm_rsrc3, descriptor.compute_pgm_rsrc3);
  EXPECT_EQ(kernel.code_properties, descriptor.code_properties);
  EXPECT_EQ(kernel.kernarg_preload, descriptor.kernarg_preload);
  const auto* symbol =
      iree_hal_amd_gpu_image_find_symbol(&image_, IREE_SV("apply_fixups"));
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
    Targets, FixupTest,
    ::testing::Range(size_t{0}, iree_hal_amd_gpu_pm4_fixup_programs_size()));

}  // namespace
