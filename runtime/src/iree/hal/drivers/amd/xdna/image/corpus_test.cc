// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/image.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/add_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/sparse_copy_i32.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// A file-backed source may split every field across arbitrary segments.
struct SegmentedSource {
  // Ref-counted sequence interface borrowing the static embedded fixture.
  iree_byte_sequence_t base;
  // Complete static compiler output.
  iree_const_byte_span_t bytes;
  // Maximum size delivered per callback.
  size_t segment_length;
  // Observed final release of the borrowed source.
  bool destroyed = false;
};

void DestroySource(iree_byte_sequence_t* base) {
  reinterpret_cast<SegmentedSource*>(base)->destroyed = true;
}

iree_status_t EnumerateSource(const iree_byte_sequence_t* base,
                              iree_byte_sequence_segment_callback_t callback) {
  const auto& source = *reinterpret_cast<const SegmentedSource*>(base);
  for (size_t offset = 0; offset < source.bytes.data_length;
       offset += source.segment_length) {
    IREE_RETURN_IF_ERROR(callback.fn(
        callback.user_data, iree_make_const_byte_span(
                                source.bytes.data + offset,
                                iree_min(source.segment_length,
                                         source.bytes.data_length - offset))));
  }
  return iree_ok_status();
}

const iree_byte_sequence_vtable_t kSourceVtable = {DestroySource,
                                                   EnumerateSource, nullptr};

class XdnaImageCorpusTest : public ::testing::TestWithParam<const char*> {};

TEST_P(XdnaImageCorpusTest,
       RetainsSegmentedCompilerImagesAcrossSupportedProfiles) {
  const char* targets[] = {"amd.xdna.strix_halo.17f0_11",
                           "amd.xdna.strix.17f0_10",
                           "amd.xdna.krackan.17f0_20"};
  for (size_t device = 0; device < std::size(targets); ++device) {
    const bool multiply = std::strcmp(GetParam(), "mul_i32") == 0;
    const iree_file_toc_t* file = nullptr;
    if (multiply) {
      file = device == 0 ? iree_hal_amd_xdna_test_mul_i32_create()
                         : iree_hal_amd_xdna_test_mul_i32_npu4_create();
    } else {
      file = device == 0 ? iree_hal_amd_xdna_test_add_i32_create()
                         : iree_hal_amd_xdna_test_add_i32_npu4_create();
    }
    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        iree_make_cstring_view(targets[device]), 1, &target));
    for (size_t segment_length : {1, 7, 31, 257}) {
      SCOPED_TRACE(targets[device]);
      SCOPED_TRACE(segment_length);
      SegmentedSource source = {};
      source.bytes = iree_make_const_byte_span(file->data, file->size);
      source.segment_length = segment_length;
      iree_byte_sequence_initialize(&kSourceVtable, file->size, &source.base);
      iree_hal_amd_xdna_image_t* image = nullptr;
      IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
          &source.base, &target, iree_allocator_system(), &image));
      iree_byte_sequence_release(&source.base);
      EXPECT_FALSE(source.destroyed);
      uint32_t ordinal = UINT32_MAX;
      IREE_EXPECT_OK(iree_hal_amd_xdna_image_find_entry(
          image, iree_make_cstring_view(GetParam()), &ordinal));
      const auto* tables = iree_hal_amd_xdna_image_tables(image);
      const auto entry = iree_hal_amd_xdna_image_tables_entry(tables, ordinal);
      EXPECT_EQ(entry.binding_count, 3u);
      const auto* directory = iree_hal_amd_xdna_image_directory(image);
      for (uint32_t i = 0; i < entry.allocation_use_count; ++i) {
        const auto allocation = iree_hal_amd_xdna_image_tables_allocation(
            tables, iree_hal_amd_xdna_image_tables_allocation_use(
                        tables, entry.first_allocation_use + i));
        for (uint32_t j = 0; j < allocation.load_count; ++j) {
          const auto* load = iree_hal_amd_xdna_image_directory_program_header(
              directory, allocation.first_load + j);
          std::vector<uint8_t> bytes(load->file_range.length);
          IREE_EXPECT_OK(iree_hal_amd_xdna_image_directory_read_source_range(
              directory, load->file_range,
              iree_make_byte_span(bytes.data(), bytes.size())));
          EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(),
                                 source.bytes.data + load->file_range.offset));
        }
      }
      iree_hal_amd_xdna_image_destroy(image);
      EXPECT_TRUE(source.destroyed);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ArithmeticPrograms, XdnaImageCorpusTest,
                         ::testing::Values("mul_i32", "add_i32"));

TEST(XdnaImageSparseBindingTest, PreservesUnusedMiddleLaunchBinding) {
  const iree_file_toc_t* file = iree_hal_amd_xdna_test_sparse_copy_i32_create();
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      iree_make_cstring_view("amd.xdna.strix_halo.17f0_11"), 1, &target));
  SegmentedSource source = {};
  source.bytes = iree_make_const_byte_span(file->data, file->size);
  source.segment_length = 7;
  iree_byte_sequence_initialize(&kSourceVtable, file->size, &source.base);
  iree_hal_amd_xdna_image_t* image = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
      &source.base, &target, iree_allocator_system(), &image));
  iree_byte_sequence_release(&source.base);
  EXPECT_FALSE(source.destroyed);

  uint32_t ordinal = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_find_entry(
      image, iree_make_cstring_view("sparse_copy_i32"), &ordinal));
  const auto* tables = iree_hal_amd_xdna_image_tables(image);
  const auto entry = iree_hal_amd_xdna_image_tables_entry(tables, ordinal);
  ASSERT_EQ(entry.binding_count, 3u);
  const uint16_t expected_kinds[] = {
      IREE_XDNA_ELF_BINDING_KIND_BUFFER,
      IREE_XDNA_ELF_BINDING_KIND_NONE,
      IREE_XDNA_ELF_BINDING_KIND_BUFFER,
  };
  for (uint32_t i = 0; i < entry.binding_count; ++i) {
    const auto binding =
        iree_hal_amd_xdna_image_tables_binding(tables, entry.first_binding + i);
    EXPECT_EQ(binding.kind, expected_kinds[i]);
  }

  bool referenced_bindings[3] = {};
  for (uint32_t i = 0; i < entry.dynamic_relocation_count; ++i) {
    const auto relocation = iree_hal_amd_xdna_image_tables_relocation(
        tables, entry.first_dynamic_relocation + i);
    EXPECT_LT(relocation.source_ordinal, entry.binding_count);
    if (relocation.source_ordinal < entry.binding_count) {
      referenced_bindings[relocation.source_ordinal] = true;
    }
  }
  EXPECT_TRUE(referenced_bindings[0]);
  EXPECT_FALSE(referenced_bindings[1]);
  EXPECT_TRUE(referenced_bindings[2]);

  iree_hal_amd_xdna_image_destroy(image);
  EXPECT_TRUE(source.destroyed);
}

}  // namespace
