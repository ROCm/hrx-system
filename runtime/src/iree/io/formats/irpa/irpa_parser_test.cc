// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/irpa/irpa_parser.h"

#include <array>
#include <vector>

#include "iree/io/formats/irpa/testdata/irpa_files.h"
#include "iree/schemas/parameter_archive.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

static iree_status_t ParseTestArchive(std::vector<uint8_t>& archive_contents,
                                      iree_io_parameter_index_t* index) {
  iree_io_file_handle_t* file_handle = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(archive_contents.data(), archive_contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_status_t status =
      iree_io_parse_irpa_index(file_handle, index, iree_allocator_system());
  iree_io_file_handle_release(file_handle);
  return status;
}

static iree_host_size_t ArchiveHeaderStride() {
  return (iree_host_size_t)iree_align_uint64(
      sizeof(iree_io_parameter_archive_header_v0_t),
      IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT);
}

static iree_io_parameter_archive_header_v0_t MakeArchiveHeader(
    iree_io_physical_offset_t next_header_offset) {
  iree_io_parameter_archive_header_v0_t header = {};
  header.prefix.magic = IREE_IO_PARAMETER_ARCHIVE_MAGIC;
  header.prefix.version_major = 0;
  header.prefix.version_minor = 0;
  header.prefix.header_size = sizeof(header);
  header.prefix.next_header_offset = next_header_offset;
  return header;
}

template <typename T>
static void WriteArchiveStruct(std::vector<uint8_t>& archive_contents,
                               iree_host_size_t offset, const T& value) {
  const iree_host_size_t required_size = offset + sizeof(value);
  if (archive_contents.size() < required_size) {
    archive_contents.resize(required_size);
  }
  memcpy(archive_contents.data() + offset, &value, sizeof(value));
}

static std::vector<uint8_t> BuildEmptyArchiveChain(
    iree_host_size_t header_count) {
  const iree_host_size_t header_stride = ArchiveHeaderStride();
  std::vector<uint8_t> archive_contents(header_count * header_stride, 0);
  for (iree_host_size_t i = 0; i < header_count; ++i) {
    const iree_io_parameter_archive_header_v0_t header =
        MakeArchiveHeader(i + 1 < header_count ? header_stride : 0);
    WriteArchiveStruct(archive_contents, i * header_stride, header);
  }
  return archive_contents;
}

static std::vector<uint8_t> BuildLinkedSplatArchive(
    iree_io_physical_offset_t name_offset, uint8_t pattern_length = 1,
    iree_io_physical_size_t data_length = 4) {
  const iree_host_size_t header_stride = ArchiveHeaderStride();
  const iree_host_size_t entry_offset = header_stride;
  const iree_host_size_t metadata_offset =
      entry_offset + sizeof(iree_io_parameter_archive_splat_entry_t);
  const iree_string_view_t key = IREE_SV("linked");

  std::vector<uint8_t> archive_contents(
      header_stride + metadata_offset + key.size, 0);
  const iree_io_parameter_archive_header_v0_t first_header =
      MakeArchiveHeader(header_stride);
  WriteArchiveStruct(archive_contents, 0, first_header);

  iree_io_parameter_archive_header_v0_t linked_header = MakeArchiveHeader(0);
  linked_header.entry_count = 1;
  linked_header.entry_segment.offset = entry_offset;
  linked_header.entry_segment.length =
      sizeof(iree_io_parameter_archive_splat_entry_t);
  linked_header.metadata_segment.offset = metadata_offset;
  linked_header.metadata_segment.length = key.size;
  WriteArchiveStruct(archive_contents, header_stride, linked_header);

  iree_io_parameter_archive_splat_entry_t entry = {};
  entry.header.entry_size = sizeof(entry);
  entry.header.type = IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_SPLAT;
  entry.header.name.offset = name_offset;
  entry.header.name.length = key.size;
  entry.length = data_length;
  entry.pattern[0] = 0x5A;
  entry.pattern_length = pattern_length;
  WriteArchiveStruct(archive_contents, header_stride + entry_offset, entry);
  memcpy(archive_contents.data() + header_stride + metadata_offset, key.data,
         key.size);
  return archive_contents;
}

static std::vector<uint8_t> BuildDataArchive(
    iree_io_physical_size_t minimum_alignment,
    iree_io_physical_offset_t storage_relative_offset) {
  const iree_host_size_t entry_offset = ArchiveHeaderStride();
  const iree_host_size_t metadata_offset =
      entry_offset + sizeof(iree_io_parameter_archive_data_entry_t);
  const iree_host_size_t storage_segment_offset =
      (iree_host_size_t)iree_align_uint64(
          metadata_offset, IREE_IO_PARAMETER_ARCHIVE_DEFAULT_DATA_ALIGNMENT);
  const iree_host_size_t storage_segment_length =
      (iree_host_size_t)storage_relative_offset + 1;

  std::vector<uint8_t> archive_contents(
      storage_segment_offset + storage_segment_length, 0);
  iree_io_parameter_archive_header_v0_t header = MakeArchiveHeader(0);
  header.entry_count = 1;
  header.entry_segment.offset = entry_offset;
  header.entry_segment.length = sizeof(iree_io_parameter_archive_data_entry_t);
  header.metadata_segment.offset = metadata_offset;
  header.storage_segment.offset = storage_segment_offset;
  header.storage_segment.length = storage_segment_length;
  WriteArchiveStruct(archive_contents, 0, header);

  iree_io_parameter_archive_data_entry_t entry = {};
  entry.header.entry_size = sizeof(entry);
  entry.header.type = IREE_IO_PARAMETER_ARCHIVE_ENTRY_TYPE_DATA;
  entry.header.minimum_alignment = minimum_alignment;
  entry.storage.offset = storage_relative_offset;
  entry.storage.length = 1;
  WriteArchiveStruct(archive_contents, entry_offset, entry);
  return archive_contents;
}

static iree_io_file_handle_t* OpenTestFile(const char* name) {
  const struct iree_file_toc_t* file_toc = iree_io_irpa_files_create();
  for (size_t i = 0; i < iree_io_irpa_files_size(); ++i) {
    if (strcmp(file_toc[i].name, name) == 0) {
      iree_io_file_handle_t* file_handle = NULL;
      IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ,
          iree_make_byte_span((void*)file_toc[i].data, file_toc[i].size),
          iree_io_file_handle_release_callback_null(), iree_allocator_system(),
          &file_handle));
      return file_handle;
    }
  }
  IREE_CHECK_OK(iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "test file `%s` not found embedded into test binary", name));
  return NULL;
}

TEST(IrpaFormatTest, Empty) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("empty.irpa");
  IREE_ASSERT_OK(
      iree_io_parse_irpa_index(file_handle, index, iree_allocator_system()));
  EXPECT_EQ(0, iree_io_parameter_index_count(index));
  iree_io_file_handle_release(file_handle);

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, SingleParameters) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("single.irpa");
  IREE_ASSERT_OK(
      iree_io_parse_irpa_index(file_handle, index, iree_allocator_system()));
  EXPECT_EQ(1, iree_io_parameter_index_count(index));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry0->storage.file.offset, 192);
  EXPECT_EQ(entry0->length, 16);

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, MultipleParameters) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("multiple.irpa");
  IREE_ASSERT_OK(
      iree_io_parse_irpa_index(file_handle, index, iree_allocator_system()));
  EXPECT_EQ(2, iree_io_parameter_index_count(index));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry0->storage.file.offset, 320);
  EXPECT_EQ(entry0->length, 16);

  const iree_io_parameter_index_entry_t* entry1 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key1"), &entry1));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key1"), entry1->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry1->metadata));
  EXPECT_EQ(entry1->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry1->storage.file.offset, 384);
  EXPECT_EQ(entry1->length, 5);

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, MixedDataAndSplats) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("mixed.irpa");
  IREE_ASSERT_OK(
      iree_io_parse_irpa_index(file_handle, index, iree_allocator_system()));
  EXPECT_EQ(4, iree_io_parameter_index_count(index));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry0->storage.file.offset, 512);
  EXPECT_EQ(entry0->length, 16);

  const iree_io_parameter_index_entry_t* entry1 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key1"), &entry1));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key1"), entry1->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry1->metadata));
  EXPECT_EQ(entry1->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  EXPECT_EQ(entry1->storage.file.offset, 576);
  EXPECT_EQ(entry1->length, 5);

  const iree_io_parameter_index_entry_t* entry2 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key2"), &entry2));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key2"), entry2->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry2->metadata));
  EXPECT_EQ(entry2->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT);
  EXPECT_EQ(entry2->storage.splat.pattern_length, 1);
  const int8_t entry2_pattern = 102;
  EXPECT_EQ(0, memcmp(&entry2_pattern, entry2->storage.splat.pattern,
                      sizeof(entry2_pattern)));
  EXPECT_EQ(entry2->length, 1);

  const iree_io_parameter_index_entry_t* entry3 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("key3"), &entry3));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("key3"), entry3->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry3->metadata));
  EXPECT_EQ(entry3->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT);
  EXPECT_EQ(entry3->storage.splat.pattern_length, 8);
  const int64_t entry3_pattern = 9223372036854775807ll;
  EXPECT_EQ(0, memcmp(&entry3_pattern, entry3->storage.splat.pattern,
                      sizeof(entry3_pattern)));
  EXPECT_EQ(entry3->length, 33554432);

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, LinkedHeadersUseSelectedArchive) {
  std::vector<uint8_t> archive_contents = BuildLinkedSplatArchive(0);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_ASSERT_OK(ParseTestArchive(archive_contents, index));
  ASSERT_EQ(1, iree_io_parameter_index_count(index));
  const iree_io_parameter_index_entry_t* entry = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("linked"), &entry));
  EXPECT_EQ(IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT, entry->type);
  EXPECT_EQ(4, entry->length);
  EXPECT_EQ(1, entry->storage.splat.pattern_length);
  EXPECT_EQ(0x5A, entry->storage.splat.pattern[0]);

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, LongLinkedHeaderChain) {
  std::vector<uint8_t> archive_contents = BuildEmptyArchiveChain(32 * 1024);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_OK(ParseTestArchive(archive_contents, index));
  EXPECT_EQ(0, iree_io_parameter_index_count(index));

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, RejectsTruncatedLinkedHeader) {
  std::vector<uint8_t> archive_contents = BuildEmptyArchiveChain(2);
  archive_contents.resize(ArchiveHeaderStride() +
                          sizeof(iree_io_parameter_archive_header_prefix_t));
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        ParseTestArchive(archive_contents, index));

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, RejectsUnalignedLinkedHeaderOffset) {
  std::vector<uint8_t> archive_contents = BuildEmptyArchiveChain(1);
  const iree_io_parameter_archive_header_v0_t header = MakeArchiveHeader(1);
  WriteArchiveStruct(archive_contents, 0, header);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ParseTestArchive(archive_contents, index));

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, RejectsOverflowingLinkedHeaderOffset) {
  std::vector<uint8_t> archive_contents = BuildEmptyArchiveChain(2);
  iree_io_parameter_archive_header_v0_t linked_header = MakeArchiveHeader(
      UINT64_MAX - (IREE_IO_PARAMETER_ARCHIVE_HEADER_ALIGNMENT - 1));
  WriteArchiveStruct(archive_contents, ArchiveHeaderStride(), linked_header);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        ParseTestArchive(archive_contents, index));

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, RejectsOverflowingSegmentRange) {
  std::vector<uint8_t> archive_contents = BuildEmptyArchiveChain(2);
  iree_io_parameter_archive_header_v0_t linked_header = MakeArchiveHeader(0);
  linked_header.entry_segment.offset = UINT64_MAX;
  linked_header.entry_segment.length = 1;
  WriteArchiveStruct(archive_contents, ArchiveHeaderStride(), linked_header);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        ParseTestArchive(archive_contents, index));

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, RejectsOverflowingMetadataReference) {
  std::vector<uint8_t> archive_contents = BuildLinkedSplatArchive(UINT64_MAX);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        ParseTestArchive(archive_contents, index));

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, RejectsInvalidSplatPatternLengths) {
  const std::array<uint8_t, 3> invalid_pattern_lengths = {0, 3, 17};
  for (uint8_t pattern_length : invalid_pattern_lengths) {
    SCOPED_TRACE(static_cast<unsigned int>(pattern_length));
    std::vector<uint8_t> archive_contents =
        BuildLinkedSplatArchive(0, pattern_length, 4);
    iree_io_parameter_index_t* index = NULL;
    IREE_ASSERT_OK(
        iree_io_parameter_index_create(iree_allocator_system(), &index));

    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          ParseTestArchive(archive_contents, index));

    iree_io_parameter_index_release(index);
  }
}

TEST(IrpaFormatTest, RejectsNonIntegralSplatPattern) {
  std::vector<uint8_t> archive_contents = BuildLinkedSplatArchive(0, 4, 6);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ParseTestArchive(archive_contents, index));

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, RejectsNonPowerOfTwoDataAlignment) {
  std::vector<uint8_t> archive_contents = BuildDataArchive(3, 0);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ParseTestArchive(archive_contents, index));

  iree_io_parameter_index_release(index);
}

TEST(IrpaFormatTest, RejectsMisalignedDataStorage) {
  std::vector<uint8_t> archive_contents = BuildDataArchive(64, 1);
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ParseTestArchive(archive_contents, index));

  iree_io_parameter_index_release(index);
}

}  // namespace
}  // namespace iree
