// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/materialize.h"

#include <cstring>
#include <vector>

#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/materialize_programs.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class MaterializeProgramTest : public ::testing::TestWithParam<size_t> {
 protected:
  void TearDown() override { iree_hal_amd_gpu_image_deinitialize(&image_); }

  // Normalized source program, retained through every assertion.
  iree_hal_amd_gpu_image_t image_ = {};
};

TEST_P(MaterializeProgramTest, EmbeddedInstructionsAndDescriptorMatchSource) {
  const auto& file =
      iree_hal_amd_gpu_aql_materialize_programs_create()[GetParam() / 2];
  const bool profiling = GetParam() % 2;
  SCOPED_TRACE(file.name);
  SCOPED_TRACE(profiling);
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

  ASSERT_EQ(image_.metadata.kernel_count, 2u);
  ASSERT_EQ(image_.relocation_count, 0u);
  ASSERT_EQ(image_.import_count, 0u);
  const auto name =
      profiling ? IREE_SV("materialize_profile") : IREE_SV("materialize");
  size_t ordinal = 0;
  while (ordinal < image_.metadata.kernel_count &&
         !iree_string_view_equal(image_.metadata.kernels[ordinal].name, name)) {
    ++ordinal;
  }
  ASSERT_LT(ordinal, image_.metadata.kernel_count);
  const auto& metadata = image_.metadata.kernels[ordinal];
  EXPECT_EQ(metadata.kernarg_segment_size,
            profiling
                ? sizeof(iree_hal_amd_gpu_aql_materialize_profile_arguments_t)
                : sizeof(iree_hal_amd_gpu_aql_materialize_arguments_t));
  EXPECT_EQ(metadata.private_segment_fixed_size, 0u);
  EXPECT_EQ(metadata.group_segment_fixed_size, 0u);
  ASSERT_TRUE(metadata.has_required_workgroup_size);
  const auto packet = iree_hal_amd_gpu_aql_materialize_packet(0, 0, 0);
  const uint32_t workgroup_size[] = {packet.words[1] & 0xFFFFu,
                                     packet.words[1] >> 16, packet.words[2]};
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(workgroup_size[i], metadata.required_workgroup_size[i]);
    EXPECT_EQ(packet.words[3 + i], workgroup_size[i]);
  }
  ASSERT_EQ(metadata.argument_count, profiling ? 9u : 7u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(metadata.arguments[i].offset, i * 8);
    EXPECT_EQ(metadata.arguments[i].size, 8u);
  }
  for (size_t i = 4; i < 7; ++i) {
    EXPECT_EQ(metadata.arguments[i].offset, 32 + (i - 4) * 4);
    EXPECT_EQ(metadata.arguments[i].size, 4u);
  }
  if (profiling) {
    for (size_t i = 7; i < 9; ++i) {
      EXPECT_EQ(metadata.arguments[i].offset, 48 + (i - 7) * 8);
      EXPECT_EQ(metadata.arguments[i].size, 8u);
    }
  }
  iree_const_byte_span_t code;
  iree_hal_amd_gpu_image_kernel_t descriptor;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_materialize_program_query(
      image_.target.version, profiling, &code, &descriptor));
  const auto& kernel = image_.kernels[ordinal];
  EXPECT_EQ(kernel.compute_pgm_rsrc1, descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(kernel.compute_pgm_rsrc2, descriptor.compute_pgm_rsrc2);
  EXPECT_EQ(kernel.compute_pgm_rsrc3, descriptor.compute_pgm_rsrc3);
  EXPECT_EQ(kernel.code_properties, descriptor.code_properties);
  EXPECT_EQ(kernel.kernarg_preload, descriptor.kernarg_preload);
  const auto* symbol = iree_hal_amd_gpu_image_find_symbol(&image_, name);
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
    Targets, MaterializeProgramTest,
    ::testing::Range(size_t{0},
                     iree_hal_amd_gpu_aql_materialize_programs_size() * 2));

}  // namespace
