// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/directory.h"

#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::ImageBuilder;
using iree::hal::amd::xdna::testing::ImageBuilderOptions;
using iree::hal::amd::xdna::testing::MakeProgramHeader;
using testing::HasSubstr;

struct ByteSequenceDeleter {
  void operator()(iree_byte_sequence_t* sequence) const {
    iree_byte_sequence_release(sequence);
  }
};

using ByteSequencePtr =
    std::unique_ptr<iree_byte_sequence_t, ByteSequenceDeleter>;

struct DirectoryDeleter {
  void operator()(iree_hal_amd_xdna_image_directory_t* directory) const {
    iree_hal_amd_xdna_image_directory_destroy(directory);
  }
};

using DirectoryPtr =
    std::unique_ptr<iree_hal_amd_xdna_image_directory_t, DirectoryDeleter>;

static std::vector<uint8_t> make_image(
    const std::vector<iree_hal_amd_xdna_image_program_header_t>&
        program_headers,
    ImageBuilderOptions options = {}) {
  ImageBuilder builder(options);
  for (const auto& program_header : program_headers) {
    builder.AddProgram(program_header);
  }
  return builder.Build();
}

static ByteSequencePtr make_owned_sequence(const std::vector<uint8_t>& bytes) {
  iree_byte_span_t storage = iree_byte_span_empty();
  storage.data_length = bytes.size();
  IREE_CHECK_OK(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), storage.data_length, (void**)&storage.data));
  memcpy(storage.data, bytes.data(), bytes.size());
  iree_byte_sequence_t* sequence = nullptr;
  IREE_CHECK_OK(iree_byte_sequence_create_from_span_move(
      &storage, iree_allocator_system(), &sequence));
  return ByteSequencePtr(sequence);
}

static DirectoryPtr open_directory(const std::vector<uint8_t>& bytes) {
  ByteSequencePtr sequence = make_owned_sequence(bytes);
  iree_hal_amd_xdna_image_directory_t* directory = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_image_directory_create(
      sequence.get(), iree_allocator_system(), &directory));
  return DirectoryPtr(directory);
}

static Status open_directory_status(const std::vector<uint8_t>& bytes) {
  ByteSequencePtr sequence = make_owned_sequence(bytes);
  iree_hal_amd_xdna_image_directory_t* directory = nullptr;
  Status status(iree_hal_amd_xdna_image_directory_create(
      sequence.get(), iree_allocator_system(), &directory));
  iree_hal_amd_xdna_image_directory_destroy(directory);
  return status;
}

static void expect_status(const Status& status, StatusCode expected_code,
                          const char* expected_message_substring) {
  EXPECT_EQ(status.code(), expected_code);
  EXPECT_THAT(status.ToString(), HasSubstr(expected_message_substring));
}

static iree_status_t append_source_segment(void* user_data,
                                           iree_const_byte_span_t segment) {
  auto* bytes = static_cast<std::vector<uint8_t>*>(user_data);
  bytes->insert(bytes->end(), segment.data, segment.data + segment.data_length);
  return iree_ok_status();
}

static iree_status_t reject_source_segment(void* user_data,
                                           iree_const_byte_span_t segment) {
  (void)user_data;
  (void)segment;
  return iree_make_status(IREE_STATUS_ABORTED, "caller rejected segment");
}

TEST(ImageDirectoryTest, DecodesDirectoryAndReadsSourceRanges) {
  auto first =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE, 256, 8);
  auto second = MakeProgramHeader(0x6C58FFFF, 272, 12);
  second.virtual_address = 16;
  second.alignment = 16;
  const std::vector<uint8_t> bytes = make_image({first, second});
  DirectoryPtr directory = open_directory(bytes);

  EXPECT_EQ(iree_hal_amd_xdna_image_directory_source_length(directory.get()),
            bytes.size());
  EXPECT_EQ(iree_hal_amd_xdna_image_directory_target_flags(directory.get()),
            IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS);
  ASSERT_EQ(
      iree_hal_amd_xdna_image_directory_program_header_count(directory.get()),
      2);
  const auto* decoded_first =
      iree_hal_amd_xdna_image_directory_program_header(directory.get(), 0);
  ASSERT_NE(decoded_first, nullptr);
  EXPECT_EQ(decoded_first->type, first.type);
  EXPECT_EQ(decoded_first->file_range.offset, first.file_range.offset);
  EXPECT_EQ(decoded_first->file_range.length, first.file_range.length);
  EXPECT_EQ(decoded_first->flags, first.flags);
  const auto* decoded_second =
      iree_hal_amd_xdna_image_directory_program_header(directory.get(), 1);
  ASSERT_NE(decoded_second, nullptr);
  EXPECT_EQ(decoded_second->type, second.type);
  EXPECT_EQ(decoded_second->virtual_address, second.virtual_address);
  EXPECT_EQ(decoded_second->alignment, second.alignment);
  EXPECT_EQ(
      iree_hal_amd_xdna_image_directory_program_header(directory.get(), 2),
      nullptr);

  std::vector<uint8_t> copied(first.file_range.length);
  IREE_EXPECT_OK(iree_hal_amd_xdna_image_directory_read_source_range(
      directory.get(), first.file_range,
      iree_make_byte_span(copied.data(), copied.size())));
  EXPECT_EQ(copied, std::vector<uint8_t>(first.file_range.length, 0xA0));

  std::vector<uint8_t> enumerated;
  IREE_EXPECT_OK(iree_hal_amd_xdna_image_directory_enumerate_source_range(
      directory.get(), second.file_range,
      {
          append_source_segment,
          &enumerated,
      }));
  EXPECT_EQ(enumerated, std::vector<uint8_t>(second.file_range.length, 0xA1));
}

TEST(ImageDirectoryTest, PreservesTargetFlagsForQualification) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  const std::vector<uint8_t> bytes =
      make_image({program_header}, {
                                       /*.target_flags=*/0x12345678,
                                       /*.section_header_offset=*/0,
                                       /*.section_header_count=*/0,
                                       /*.minimum_source_length=*/512,
                                   });
  DirectoryPtr directory = open_directory(bytes);
  EXPECT_EQ(iree_hal_amd_xdna_image_directory_target_flags(directory.get()),
            0x12345678);
}

TEST(ImageDirectoryTest, AcceptsBoundedDiagnosticSectionDirectory) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 16);
  const std::vector<uint8_t> bytes = make_image(
      {program_header}, {
                            /*.target_flags=*/IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS,
                            /*.section_header_offset=*/320,
                            /*.section_header_count=*/2,
                            /*.minimum_source_length=*/416,
                        });
  DirectoryPtr directory = open_directory(bytes);
  EXPECT_EQ(iree_hal_amd_xdna_image_directory_source_length(directory.get()),
            416);
}

TEST(ImageDirectoryTest, PreservesExactPayloadAliasesAsDistinctEntries) {
  auto first =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE, 256, 16);
  first.physical_address = 0x00010200;
  auto second = first;
  second.physical_address = 0x00010400;
  const std::vector<uint8_t> bytes = make_image({first, second});
  DirectoryPtr directory = open_directory(bytes);

  ASSERT_EQ(
      iree_hal_amd_xdna_image_directory_program_header_count(directory.get()),
      2);
  const auto* decoded_first =
      iree_hal_amd_xdna_image_directory_program_header(directory.get(), 0);
  const auto* decoded_second =
      iree_hal_amd_xdna_image_directory_program_header(directory.get(), 1);
  ASSERT_NE(decoded_first, nullptr);
  ASSERT_NE(decoded_second, nullptr);
  EXPECT_EQ(decoded_first->file_range.offset,
            decoded_second->file_range.offset);
  EXPECT_EQ(decoded_first->file_range.length,
            decoded_second->file_range.length);
  EXPECT_NE(decoded_first->physical_address, decoded_second->physical_address);
}

typedef struct test_segmented_sequence_t {
  // Byte sequence interface exposed to the directory.
  iree_byte_sequence_t base;
  // Ordered source segments returned during enumeration.
  const iree_const_byte_span_t* segments;
  // Number of entries in |segments|.
  iree_host_size_t segment_count;
  // Counter incremented when the final reference is released.
  int* destroy_count;
} test_segmented_sequence_t;

static void test_segmented_sequence_destroy(
    iree_byte_sequence_t* base_sequence) {
  auto* sequence = reinterpret_cast<test_segmented_sequence_t*>(base_sequence);
  ++*sequence->destroy_count;
}

static iree_status_t test_segmented_sequence_enumerate(
    const iree_byte_sequence_t* base_sequence,
    iree_byte_sequence_segment_callback_t callback) {
  const auto* sequence =
      reinterpret_cast<const test_segmented_sequence_t*>(base_sequence);
  for (iree_host_size_t i = 0; i < sequence->segment_count; ++i) {
    IREE_RETURN_IF_ERROR(
        callback.fn(callback.user_data, sequence->segments[i]));
  }
  return iree_ok_status();
}

static const iree_byte_sequence_vtable_t test_segmented_sequence_vtable = {
    /*.destroy=*/test_segmented_sequence_destroy,
    /*.enumerate=*/test_segmented_sequence_enumerate,
    /*.try_get_contiguous_span=*/nullptr,
};

TEST(ImageDirectoryTest, ReadsFieldsAndPayloadAcrossArbitrarySegments) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 16);
  const std::vector<uint8_t> bytes = make_image({program_header});
  const std::vector<iree_host_size_t> boundaries = {1,  7,  51,  53,
                                                    83, 85, 259, 263};
  std::vector<iree_const_byte_span_t> segments;
  iree_host_size_t previous_boundary = 0;
  for (iree_host_size_t boundary : boundaries) {
    segments.push_back(iree_make_const_byte_span(
        bytes.data() + previous_boundary, boundary - previous_boundary));
    previous_boundary = boundary;
  }
  segments.push_back(iree_make_const_byte_span(
      bytes.data() + previous_boundary, bytes.size() - previous_boundary));
  int destroy_count = 0;
  test_segmented_sequence_t sequence = {
      /*.base=*/{},
      /*.segments=*/segments.data(),
      /*.segment_count=*/segments.size(),
      /*.destroy_count=*/&destroy_count,
  };
  iree_byte_sequence_initialize(&test_segmented_sequence_vtable, bytes.size(),
                                &sequence.base);

  iree_hal_amd_xdna_image_directory_t* raw_directory = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_directory_create(
      &sequence.base, iree_allocator_system(), &raw_directory));
  DirectoryPtr directory(raw_directory);
  iree_byte_sequence_release(&sequence.base);
  EXPECT_EQ(destroy_count, 0);

  const auto* decoded =
      iree_hal_amd_xdna_image_directory_program_header(directory.get(), 0);
  ASSERT_NE(decoded, nullptr);
  EXPECT_EQ(decoded->file_range.offset, 256);
  EXPECT_EQ(decoded->file_range.length, 16);
  std::vector<uint8_t> payload;
  IREE_EXPECT_OK(iree_hal_amd_xdna_image_directory_enumerate_source_range(
      directory.get(), decoded->file_range,
      {
          append_source_segment,
          &payload,
      }));
  EXPECT_EQ(payload, std::vector<uint8_t>(16, 0xA0));

  directory.reset();
  EXPECT_EQ(destroy_count, 1);
}

TEST(ImageDirectoryTest, PropagatesRangeCallbackFailure) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  DirectoryPtr directory = open_directory(make_image({program_header}));
  expect_status(Status(iree_hal_amd_xdna_image_directory_enumerate_source_range(
                    directory.get(), program_header.file_range,
                    {
                        reject_source_segment,
                        nullptr,
                    })),
                StatusCode::kAborted, "caller rejected segment");
}

TEST(ImageDirectoryTest, RejectsInvalidRangeAndDestinationSize) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  DirectoryPtr directory = open_directory(make_image({program_header}));
  uint8_t storage[7];
  expect_status(Status(iree_hal_amd_xdna_image_directory_read_source_range(
                    directory.get(), program_header.file_range,
                    iree_make_byte_span(storage, sizeof(storage)))),
                StatusCode::kInvalidArgument,
                "exactly 8 bytes of destination storage");
  expect_status(Status(iree_hal_amd_xdna_image_directory_enumerate_source_range(
                    directory.get(),
                    {
                        /*.offset=*/510,
                        /*.length=*/8,
                    },
                    {
                        append_source_segment,
                        nullptr,
                    })),
                StatusCode::kOutOfRange, "exceeds the source length");
}

TEST(ImageDirectoryTest, RejectsInvalidElfIdentity) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  std::vector<uint8_t> bytes = make_image({program_header});
  bytes[4] = 2;
  expect_status(open_directory_status(bytes), StatusCode::kInvalidArgument,
                "not canonical ELF32LE");
}

TEST(ImageDirectoryTest, RejectsTruncatedProgramHeaderDirectory) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  std::vector<uint8_t> bytes = make_image({program_header});
  iree_unaligned_store_le_u16(bytes.data() + 44, 32);
  expect_status(open_directory_status(bytes), StatusCode::kOutOfRange,
                "exceeds the source length");
}

TEST(ImageDirectoryTest, RejectsIncompleteSectionHeaderDirectory) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  std::vector<uint8_t> bytes = make_image({program_header});
  iree_unaligned_store_le_u32(bytes.data() + 32, 320);
  expect_status(open_directory_status(bytes), StatusCode::kInvalidArgument,
                "section-header directory is invalid");
}

TEST(ImageDirectoryTest, RejectsPayloadOutsideSource) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  std::vector<uint8_t> bytes = make_image({program_header});
  uint8_t* encoded_program_header =
      bytes.data() + IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE;
  iree_unaligned_store_le_u32(encoded_program_header + 4, 508);
  iree_unaligned_store_le_u32(encoded_program_header + 16, 8);
  iree_unaligned_store_le_u32(encoded_program_header + 20, 8);
  expect_status(open_directory_status(bytes), StatusCode::kOutOfRange,
                "exceeds the source length");
}

TEST(ImageDirectoryTest, RejectsPayloadOverlappingElfDirectory) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 4);
  std::vector<uint8_t> bytes = make_image({program_header});
  uint8_t* encoded_program_header =
      bytes.data() + IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE;
  iree_unaligned_store_le_u32(encoded_program_header + 4,
                              IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE);
  expect_status(open_directory_status(bytes), StatusCode::kInvalidArgument,
                "overlaps an ELF directory");
}

TEST(ImageDirectoryTest, RejectsPartiallyOverlappingPayloads) {
  const auto first =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE, 256, 16);
  const auto second =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE, 264, 16);
  const std::vector<uint8_t> bytes = make_image({first, second});
  expect_status(open_directory_status(bytes), StatusCode::kInvalidArgument,
                "partially overlap");
}

TEST(ImageDirectoryTest, RejectsInvalidProgramAlignment) {
  auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  program_header.alignment = 3;
  const std::vector<uint8_t> bytes = make_image({program_header});
  expect_status(open_directory_status(bytes), StatusCode::kInvalidArgument,
                "invalid size or alignment");
}

TEST(ImageDirectoryTest, RejectsUnknownProgramPermissionBits) {
  auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 256, 8);
  program_header.flags |= 0x8;
  const std::vector<uint8_t> bytes = make_image({program_header});
  expect_status(open_directory_status(bytes), StatusCode::kInvalidArgument,
                "unknown permission flags");
}

TEST(ImageDirectoryTest, RejectsSectionDirectoryOverlappingPayload) {
  const auto program_header =
      MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 320, 80);
  const std::vector<uint8_t> bytes = make_image(
      {program_header}, {
                            /*.target_flags=*/IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS,
                            /*.section_header_offset=*/384,
                            /*.section_header_count=*/1,
                            /*.minimum_source_length=*/512,
                        });
  expect_status(open_directory_status(bytes), StatusCode::kInvalidArgument,
                "overlaps an ELF directory");
}

}  // namespace
