// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/control_program.h"

#include <cstring>
#include <vector>

#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/queue/control_programs.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ControlProgramTest : public ::testing::TestWithParam<size_t> {
 protected:
  void TearDown() override { iree_hal_amd_gpu_image_deinitialize(&image_); }

  // Normalized source program, retained through every assertion.
  iree_hal_amd_gpu_image_t image_ = {};
};

TEST_P(ControlProgramTest, EmbeddedInstructionsAndDescriptorMatchSource) {
  const auto& file = iree_hal_amd_gpu_control_programs_create()[GetParam()];
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

  ASSERT_EQ(image_.metadata.kernel_count, 4u);
  ASSERT_EQ(image_.relocation_count, 0u);
  ASSERT_EQ(image_.import_count, 0u);
  std::vector<uint8_t> storage(image_.byte_length);
  std::vector<iree_byte_span_t> mappings;
  for (size_t i = 0; i < image_.segment_count; ++i) {
    const auto& segment = image_.segments[i];
    mappings.push_back(
        iree_make_byte_span(storage.data() + segment.offset, segment.length));
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_image_load(&image_, 0x100000000, 0, nullptr,
                                             mappings.size(), mappings.data()));
  constexpr iree_string_view_t names[] = {
      IREE_SVL("control_device_32"),
      IREE_SVL("control_device_64"),
      IREE_SVL("control_system_32"),
      IREE_SVL("control_system_64"),
  };
  for (uint32_t i = 0; i < IREE_HAL_AMD_GPU_CONTROL_PROGRAM_COUNT; ++i) {
    SCOPED_TRACE(i);
    iree_const_byte_span_t code;
    iree_hal_amd_gpu_image_kernel_t descriptor;
    IREE_ASSERT_OK(iree_hal_amd_gpu_control_program_query(
        image_.target.version, i, &code, &descriptor));
    const auto* symbol = iree_hal_amd_gpu_image_find_symbol(&image_, names[i]);
    ASSERT_NE(symbol, nullptr);
    ASSERT_EQ(symbol->length, code.data_length);
    const iree_hal_amd_gpu_image_kernel_t* kernel = nullptr;
    size_t ordinal = 0;
    for (; ordinal < image_.metadata.kernel_count; ++ordinal) {
      if (image_.kernels[ordinal].entry_offset == symbol->address.offset) {
        kernel = &image_.kernels[ordinal];
        break;
      }
    }
    ASSERT_NE(kernel, nullptr);
    const auto& metadata = image_.metadata.kernels[ordinal];
    EXPECT_EQ(metadata.kernarg_segment_size,
              sizeof(iree_hal_amd_gpu_control_program_arguments_t));
    EXPECT_EQ(metadata.private_segment_fixed_size, 0u);
    EXPECT_EQ(metadata.group_segment_fixed_size, 0u);
    ASSERT_EQ(metadata.argument_count, 4u);
    for (size_t argument = 0; argument < 4; ++argument) {
      EXPECT_EQ(metadata.arguments[argument].offset, argument * 8);
      EXPECT_EQ(metadata.arguments[argument].size, argument == 3 ? 4u : 8u);
    }
    EXPECT_EQ(kernel->compute_pgm_rsrc1, descriptor.compute_pgm_rsrc1);
    EXPECT_EQ(kernel->compute_pgm_rsrc2, descriptor.compute_pgm_rsrc2);
    EXPECT_EQ(kernel->compute_pgm_rsrc3, descriptor.compute_pgm_rsrc3);
    EXPECT_EQ(kernel->code_properties, descriptor.code_properties);
    EXPECT_EQ(kernel->kernarg_preload, descriptor.kernarg_preload);
    EXPECT_EQ(memcmp(storage.data() + kernel->entry_offset, code.data,
                     code.data_length),
              0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Targets, ControlProgramTest,
    ::testing::Range(size_t{0}, iree_hal_amd_gpu_control_programs_size()));

}  // namespace
