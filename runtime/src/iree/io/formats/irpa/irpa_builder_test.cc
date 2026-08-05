// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/irpa/irpa_builder.h"

#include <array>
#include <limits>
#include <vector>

#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

struct BuildArchiveTarget {
  // Full target file contents allocated by the callback.
  std::vector<uint8_t> contents;
  // Resolved archive offset reported to the callback.
  iree_io_physical_offset_t archive_offset = 0;
  // Archive byte length reported to the callback.
  iree_io_physical_size_t archive_length = 0;
  // Number of times the callback has been invoked.
  iree_host_size_t open_count = 0;
  // Whether to link a valid empty archive at offset zero to the built archive.
  bool link_from_file_start = false;
};

static iree_status_t OpenBuildArchiveTarget(
    void* user_data, iree_io_physical_offset_t archive_offset,
    iree_io_physical_size_t archive_length,
    iree_io_file_handle_t** out_file_handle) {
  auto* target = static_cast<BuildArchiveTarget*>(user_data);
  ++target->open_count;
  target->archive_offset = archive_offset;
  target->archive_length = archive_length;

  uint64_t file_length = 0;
  if (!iree_checked_add_u64(archive_offset, archive_length, &file_length) ||
      file_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "test archive allocation range overflow");
  }
  target->contents.assign(static_cast<size_t>(file_length), 0xCD);
  if (target->link_from_file_start) {
    if (archive_offset < sizeof(iree_io_parameter_archive_header_v0_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "embedded archive overlaps prefix header");
    }
    iree_io_parameter_archive_header_v0_t prefix_header = {};
    prefix_header.prefix.magic = IREE_IO_PARAMETER_ARCHIVE_MAGIC;
    prefix_header.prefix.version_major = 0;
    prefix_header.prefix.version_minor = 0;
    prefix_header.prefix.header_size = sizeof(prefix_header);
    prefix_header.prefix.next_header_offset = archive_offset;
    memcpy(target->contents.data(), &prefix_header, sizeof(prefix_header));
  }

  return iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(target->contents.data(), target->contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      out_file_handle);
}

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

  std::vector<uint8_t> file_contents(total_size, 0xCD);
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

  iree_io_physical_offset_t entry_offset = header->entry_segment.offset;
  for (iree_host_size_t i = 0; i < header->entry_count; ++i) {
    const auto* archive_entry =
        reinterpret_cast<const iree_io_parameter_archive_entry_header_t*>(
            file_contents.data() + entry_offset);
    const iree_io_physical_offset_t unaligned_next_entry_offset =
        entry_offset + archive_entry->entry_size;
    const iree_io_physical_offset_t next_entry_offset = iree_align_uint64(
        unaligned_next_entry_offset, IREE_IO_PARAMETER_ARCHIVE_ENTRY_ALIGNMENT);
    if (i + 1 < header->entry_count) {
      for (iree_io_physical_offset_t padding_offset =
               unaligned_next_entry_offset;
           padding_offset < next_entry_offset; ++padding_offset) {
        EXPECT_EQ(0u, file_contents[padding_offset]);
      }
    }
    entry_offset = next_entry_offset;
  }

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

TEST_F(IrpaBuilderTest, PreservesPerEntryStorageAlignment) {
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("a"), iree_const_byte_span_empty(), 0, 1));
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("b"), iree_const_byte_span_empty(), 16, 1));
  IREE_ASSERT_OK(iree_io_parameter_archive_builder_add_data_entry(
      &builder_, IREE_SV("c"), iree_const_byte_span_empty(), 256, 1));

  EXPECT_EQ(512u, HeaderSize());
  EXPECT_EQ(4096u, TotalSize());
  const std::array<iree_string_view_t, 3> keys = {IREE_SV("a"), IREE_SV("b"),
                                                  IREE_SV("c")};
  const std::array<iree_io_physical_offset_t, 3> relative_offsets = {0, 16,
                                                                     256};
  const std::array<iree_io_physical_size_t, 3> minimum_alignments = {0, 16,
                                                                     256};
  const iree_io_parameter_index_entry_t* entry = NULL;
  for (iree_host_size_t i = 0; i < keys.size(); ++i) {
    IREE_ASSERT_OK(iree_io_parameter_index_get(builder_.index, i, &entry));
    EXPECT_EQ(relative_offsets[i], entry->storage.file.offset);
    EXPECT_EQ(minimum_alignments[i], entry->storage.file.minimum_alignment);
  }

  std::vector<uint8_t> file_contents(TotalSize(), 0);
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

  iree_io_parameter_index_t* parsed_index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &parsed_index));
  IREE_ASSERT_OK(iree_io_parse_irpa_index(file_handle, parsed_index,
                                          iree_allocator_system()));
  const std::array<iree_io_parameter_index_t*, 2> output_indices = {
      built_index, parsed_index};
  for (iree_io_parameter_index_t* output_index : output_indices) {
    for (iree_host_size_t i = 0; i < keys.size(); ++i) {
      IREE_ASSERT_OK(
          iree_io_parameter_index_lookup(output_index, keys[i], &entry));
      EXPECT_EQ(HeaderSize() + relative_offsets[i], entry->storage.file.offset);
      EXPECT_EQ(minimum_alignments[i], entry->storage.file.minimum_alignment);
      if (minimum_alignments[i] != 0) {
        EXPECT_EQ(0u, entry->storage.file.offset % minimum_alignments[i]);
      }
    }
  }

  iree_io_parameter_index_release(parsed_index);
  iree_io_parameter_index_release(built_index);
  iree_io_stream_release(stream);
  iree_io_file_handle_release(file_handle);
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

TEST(IrpaBuilderIntegrationTest, BuildsEmbeddedArchiveAtReportedOffset) {
  const std::array<uint8_t, 5> source_contents = {1, 3, 5, 7, 9};
  iree_io_file_handle_t* source_file_handle = NULL;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(const_cast<uint8_t*>(source_contents.data()),
                          source_contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &source_file_handle));
  iree_io_parameter_index_t* source_index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &source_index));
  iree_io_parameter_index_entry_t source_entry = {};
  source_entry.key = IREE_SV("embedded");
  source_entry.metadata = iree_const_byte_span_empty();
  source_entry.length = source_contents.size();
  source_entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
  source_entry.storage.file.handle = source_file_handle;
  source_entry.storage.file.offset = 0;
  source_entry.storage.file.minimum_alignment = 256;
  IREE_ASSERT_OK(iree_io_parameter_index_add(source_index, &source_entry));
  iree_io_file_handle_release(source_file_handle);

  iree_io_parameter_index_t* target_index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &target_index));
  BuildArchiveTarget target;
  target.link_from_file_start = true;
  const iree_io_parameter_archive_file_open_callback_t open_callback = {
      /*.fn=*/OpenBuildArchiveTarget,
      /*.user_data=*/&target,
  };
  const iree_io_physical_offset_t requested_offset =
      sizeof(iree_io_parameter_archive_header_v0_t) + 1;
  IREE_ASSERT_OK(iree_io_build_parameter_archive(
      source_index, target_index, open_callback, requested_offset,
      iree_allocator_system()));

  ASSERT_EQ(1u, target.open_count);
  const iree_io_physical_size_t archive_alignment =
      iree_max(IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT,
               source_entry.storage.file.minimum_alignment);
  ASSERT_EQ(iree_align_uint64(requested_offset, archive_alignment),
            target.archive_offset);
  EXPECT_EQ(target.archive_offset + target.archive_length,
            target.contents.size());
  EXPECT_EQ(0xCD, target.contents[requested_offset]);
  const auto* archive_header =
      reinterpret_cast<const iree_io_parameter_archive_header_v0_t*>(
          target.contents.data() + target.archive_offset);
  EXPECT_EQ(IREE_IO_PARAMETER_ARCHIVE_MAGIC, archive_header->prefix.magic);

  const iree_io_parameter_index_entry_t* target_entry = NULL;
  IREE_ASSERT_OK(iree_io_parameter_index_lookup(
      target_index, IREE_SV("embedded"), &target_entry));
  const iree_io_physical_offset_t expected_data_offset =
      target.archive_offset + archive_header->storage_segment.offset;
  EXPECT_EQ(expected_data_offset, target_entry->storage.file.offset);
  EXPECT_EQ(0u, target_entry->storage.file.offset % archive_alignment);
  EXPECT_EQ(source_entry.storage.file.minimum_alignment,
            target_entry->storage.file.minimum_alignment);
  EXPECT_EQ(0, memcmp(target.contents.data() + expected_data_offset,
                      source_contents.data(), source_contents.size()));

  iree_io_file_handle_t* target_file_handle = NULL;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(target.contents.data(), target.contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &target_file_handle));
  iree_io_parameter_index_t* parsed_index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &parsed_index));
  IREE_ASSERT_OK(iree_io_parse_irpa_index(target_file_handle, parsed_index,
                                          iree_allocator_system()));
  const iree_io_parameter_index_entry_t* parsed_entry = NULL;
  IREE_ASSERT_OK(iree_io_parameter_index_lookup(
      parsed_index, IREE_SV("embedded"), &parsed_entry));
  EXPECT_EQ(expected_data_offset, parsed_entry->storage.file.offset);
  EXPECT_EQ(source_entry.storage.file.minimum_alignment,
            parsed_entry->storage.file.minimum_alignment);

  iree_io_parameter_index_release(parsed_index);
  iree_io_file_handle_release(target_file_handle);
  iree_io_parameter_index_release(target_index);
  iree_io_parameter_index_release(source_index);
}

TEST(IrpaBuilderIntegrationTest, RejectsEmbeddedRangeBeforeOpen) {
  iree_io_parameter_index_t* source_index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &source_index));
  iree_io_parameter_index_t* target_index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &target_index));
  BuildArchiveTarget target;
  const iree_io_parameter_archive_file_open_callback_t open_callback = {
      /*.fn=*/OpenBuildArchiveTarget,
      /*.user_data=*/&target,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_io_build_parameter_archive(
          source_index, target_index, open_callback,
          std::numeric_limits<iree_io_physical_offset_t>::max() - 1,
          iree_allocator_system()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_io_build_parameter_archive(
          source_index, target_index, open_callback,
          std::numeric_limits<iree_io_physical_offset_t>::max() -
              (IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT - 1),
          iree_allocator_system()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_io_build_parameter_archive(
          source_index, target_index, open_callback,
          static_cast<iree_io_physical_offset_t>(INT64_MAX) -
              (IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT - 1),
          iree_allocator_system()));
  EXPECT_EQ(0u, target.open_count);
  EXPECT_EQ(0u, iree_io_parameter_index_count(target_index));

  iree_io_parameter_index_release(target_index);
  iree_io_parameter_index_release(source_index);
}

}  // namespace
}  // namespace iree
