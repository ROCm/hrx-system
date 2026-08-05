// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/irpa/irpa_builder.h"

#include <limits>
#include <vector>

#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

class IrpaBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_io_parameter_archive_builder_initialize(
        iree_allocator_system(), &builder_));
  }

  void TearDown() override {
    iree_io_parameter_archive_builder_deinitialize(&builder_);
  }

  iree_io_physical_size_t HeaderSize() {
    iree_io_physical_size_t header_size = 0;
    IREE_EXPECT_OK(
        iree_io_parameter_archive_builder_header_size(&builder_, &header_size));
    return header_size;
  }

  iree_io_physical_size_t TotalSize() {
    iree_io_physical_size_t total_size = 0;
    IREE_EXPECT_OK(
        iree_io_parameter_archive_builder_total_size(&builder_, &total_size));
    return total_size;
  }

  iree_io_parameter_archive_builder_t builder_;
};

TEST_F(IrpaBuilderTest, EmptyLayout) {
  EXPECT_TRUE(iree_io_parameter_archive_builder_is_empty(&builder_));
  EXPECT_EQ(96u, HeaderSize());
  EXPECT_EQ(4096u, TotalSize());
}

TEST_F(IrpaBuilderTest, WritesExpectedV0Layout) {
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("key0"), iree_const_byte_span_empty(), 64, 16));
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("key1"), iree_const_byte_span_empty(), 64, 5));

  EXPECT_EQ(320u, HeaderSize());
  const iree_io_physical_size_t total_size = TotalSize();
  EXPECT_EQ(4096u, total_size);

  std::vector<uint8_t> file_contents(total_size, 0);
  iree_io_file_handle_t* file_handle = NULL;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(file_contents.data(), file_contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_io_stream_t* stream = NULL;
  IREE_ASSERT_OK(iree_io_stream_open(IREE_IO_STREAM_MODE_WRITABLE, file_handle,
                                     0, iree_allocator_system(), &stream));
  iree_io_parameter_index_t* built_index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &built_index));
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_write(
      &builder_, file_handle, 0, stream, built_index));

  const auto* header =
      reinterpret_cast<const iree_io_parameter_archive_header_v0_t*>(
          file_contents.data());
  EXPECT_EQ(96u, header->entry_segment.offset);
  EXPECT_EQ(156u, header->entry_segment.length);
  EXPECT_EQ(252u, header->metadata_segment.offset);
  EXPECT_EQ(8u, header->metadata_segment.length);
  EXPECT_EQ(320u, header->storage_segment.offset);
  EXPECT_EQ(69u, header->storage_segment.length);

  const iree_io_parameter_index_entry_t* entry = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(built_index, IREE_SV("key0"), &entry));
  EXPECT_EQ(320u, entry->storage.file.offset);
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(built_index, IREE_SV("key1"), &entry));
  EXPECT_EQ(384u, entry->storage.file.offset);

  iree_io_parameter_index_t* parsed_index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &parsed_index));
  IREE_EXPECT_OK(iree_io_parse_irpa_index(file_handle, parsed_index,
                                          iree_allocator_system()));
  EXPECT_EQ(2u, iree_io_parameter_index_count(parsed_index));

  iree_io_parameter_index_release(parsed_index);
  iree_io_parameter_index_release(built_index);
  iree_io_stream_release(stream);
  iree_io_file_handle_release(file_handle);
}

TEST_F(IrpaBuilderTest, UsesStrongestStorageAlignment) {
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("a"), iree_const_byte_span_empty(), 16, 1));
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("b"), iree_const_byte_span_empty(), 256, 1));

  EXPECT_EQ(256u, HeaderSize());
  EXPECT_EQ(4096u, TotalSize());
  const iree_io_parameter_index_entry_t* entry = NULL;
  IREE_ASSERT_OK(iree_io_parameter_index_get(builder_.index, 0, &entry));
  EXPECT_EQ(0u, entry->storage.file.offset);
  EXPECT_EQ(0u, (HeaderSize() + entry->storage.file.offset) % 16);
  IREE_ASSERT_OK(iree_io_parameter_index_get(builder_.index, 1, &entry));
  EXPECT_EQ(256u, entry->storage.file.offset);
  EXPECT_EQ(0u, (HeaderSize() + entry->storage.file.offset) % 256);
}

TEST_F(IrpaBuilderTest, TreatsZeroAlignmentAsUnspecified) {
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("a"), iree_const_byte_span_empty(), 0, 1));
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("b"), iree_const_byte_span_empty(), 0, 1));

  const iree_io_parameter_index_entry_t* entry = NULL;
  IREE_ASSERT_OK(iree_io_parameter_index_get(builder_.index, 0, &entry));
  EXPECT_EQ(0u, entry->storage.file.offset);
  IREE_ASSERT_OK(iree_io_parameter_index_get(builder_.index, 1, &entry));
  EXPECT_EQ(1u, entry->storage.file.offset);
}

TEST_F(IrpaBuilderTest, RejectsInvalidAlignmentWithoutMutation) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_io_parameter_archive_builder_add_data_entry(
          &builder_, IREE_SV("invalid"), iree_const_byte_span_empty(), 3, 1));
  EXPECT_TRUE(iree_io_parameter_archive_builder_is_empty(&builder_));
  EXPECT_EQ(4096u, TotalSize());
}

TEST_F(IrpaBuilderTest, RejectsOverflowWithoutMutation) {
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("valid"), iree_const_byte_span_empty(), 64, 64));
  const iree_io_physical_size_t total_size = TotalSize();

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_io_parameter_archive_builder_add_data_entry(
          &builder_, IREE_SV("overflow"), iree_const_byte_span_empty(), 64,
          std::numeric_limits<iree_io_physical_size_t>::max() - 64));
  EXPECT_EQ(1u, iree_io_parameter_index_count(builder_.index));
  EXPECT_EQ(total_size, TotalSize());
}

TEST_F(IrpaBuilderTest, RejectsInvalidFileAlignment) {
  builder_.file_alignment = 3;
  iree_io_physical_size_t total_size = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_io_parameter_archive_builder_total_size(&builder_, &total_size));
}

TEST_F(IrpaBuilderTest, RejectsNonIntegralSplatPattern) {
  const uint32_t pattern = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_io_parameter_archive_builder_add_splat_entry(
          &builder_, IREE_SV("invalid"), iree_const_byte_span_empty(), &pattern,
          sizeof(pattern), 6));
  EXPECT_TRUE(iree_io_parameter_archive_builder_is_empty(&builder_));
}

}  // namespace
}  // namespace iree
