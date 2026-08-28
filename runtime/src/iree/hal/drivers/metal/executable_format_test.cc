// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/executable_format.h"

#include <array>
#include <initializer_list>
#include <string_view>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum : iree_host_size_t {
  kHeaderMagicOffset = 0,
  kHeaderVersionOffset = 4,
  kHeaderLibraryCountOffset = 8,
  kHeaderPipelineCountOffset = 12,
  kLibrarySourceOffsetOffset = 0,
  kLibrarySourceLengthOffset = 4,
  kLibrarySourceVersionOffset = 8,
  kLibraryMetallibOffsetOffset = 12,
  kLibraryMetallibLengthOffset = 16,
  kPipelineLibraryOrdinalOffset = 0,
  kPipelineEntryPointOffsetOffset = 4,
  kPipelineEntryPointLengthOffset = 8,
  kPipelineMaxThreadsOffset = 12,
  kPipelineThreadgroupSizeXOffset = 16,
  kPipelineThreadgroupSizeYOffset = 20,
  kPipelineThreadgroupSizeZOffset = 24,
  kPipelineFlagsOffset = 28,
  kPipelineConstantCountOffset = 32,
  kPipelineBindingCountOffset = 36,
  kPipelineBindingReadOnlyBitsOffset = 40,
};

struct TestLibrary {
  // Optional MSL source stored in the bundle payload.
  std::string_view source;
  // MSL language version or the default-version sentinel.
  uint32_t source_version;
  // Optional metallib image stored in the bundle payload.
  std::string_view metallib;
};

struct TestPipeline {
  // Ordinal of the library containing the entry point.
  uint32_t library_ordinal;
  // Entry-point name stored in the bundle payload.
  std::string_view entry_point;
  // Optional maximum total threads per threadgroup.
  uint32_t max_threads_per_threadgroup;
  // Static threadgroup size encoded in the pipeline record.
  std::array<uint32_t, 3> threadgroup_size;
  // Pipeline behavior flags.
  iree_hal_metal_executable_pipeline_flags_t flags;
  // Number of dispatch constants encoded in the pipeline record.
  uint32_t constant_count;
  // Number of buffer bindings encoded in the pipeline record.
  uint32_t binding_count;
  // One bit per binding indicating that the binding is read-only.
  uint64_t binding_read_only_bits;
};

static void StoreU32(std::vector<uint8_t>* data, iree_host_size_t offset,
                     uint32_t value) {
  iree_unaligned_store_le_u32(data->data() + offset, value);
}

static void StoreU64(std::vector<uint8_t>* data, iree_host_size_t offset,
                     uint64_t value) {
  iree_unaligned_store_le_u64(data->data() + offset, value);
}

static uint32_t AppendSpan(std::vector<uint8_t>* data, std::string_view value) {
  if (value.empty()) return 0;
  const uint32_t offset = static_cast<uint32_t>(data->size());
  data->insert(data->end(), value.begin(), value.end());
  return offset;
}

static std::vector<uint8_t> BuildBundle(
    std::initializer_list<TestLibrary> libraries,
    std::initializer_list<TestPipeline> pipelines) {
  const iree_host_size_t library_table_length =
      libraries.size() * IREE_HAL_METAL_EXECUTABLE_FORMAT_LIBRARY_SIZE;
  const iree_host_size_t table_length =
      IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE + library_table_length +
      pipelines.size() * IREE_HAL_METAL_EXECUTABLE_FORMAT_PIPELINE_SIZE;
  std::vector<uint8_t> data(table_length, 0);
  StoreU32(&data, kHeaderMagicOffset, IREE_HAL_METAL_EXECUTABLE_FORMAT_MAGIC);
  StoreU32(&data, kHeaderVersionOffset,
           IREE_HAL_METAL_EXECUTABLE_FORMAT_VERSION);
  StoreU32(&data, kHeaderLibraryCountOffset,
           static_cast<uint32_t>(libraries.size()));
  StoreU32(&data, kHeaderPipelineCountOffset,
           static_cast<uint32_t>(pipelines.size()));

  iree_host_size_t ordinal = 0;
  for (const TestLibrary& library : libraries) {
    const uint32_t source_offset = AppendSpan(&data, library.source);
    const uint32_t metallib_offset = AppendSpan(&data, library.metallib);
    const iree_host_size_t record_offset =
        IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE +
        ordinal * IREE_HAL_METAL_EXECUTABLE_FORMAT_LIBRARY_SIZE;
    StoreU32(&data, record_offset + kLibrarySourceOffsetOffset, source_offset);
    StoreU32(&data, record_offset + kLibrarySourceLengthOffset,
             static_cast<uint32_t>(library.source.size()));
    StoreU32(&data, record_offset + kLibrarySourceVersionOffset,
             library.source_version);
    StoreU32(&data, record_offset + kLibraryMetallibOffsetOffset,
             metallib_offset);
    StoreU32(&data, record_offset + kLibraryMetallibLengthOffset,
             static_cast<uint32_t>(library.metallib.size()));
    ++ordinal;
  }

  ordinal = 0;
  for (const TestPipeline& pipeline : pipelines) {
    const uint32_t entry_point_offset = AppendSpan(&data, pipeline.entry_point);
    const iree_host_size_t record_offset =
        IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE + library_table_length +
        ordinal * IREE_HAL_METAL_EXECUTABLE_FORMAT_PIPELINE_SIZE;
    StoreU32(&data, record_offset + kPipelineLibraryOrdinalOffset,
             pipeline.library_ordinal);
    StoreU32(&data, record_offset + kPipelineEntryPointOffsetOffset,
             entry_point_offset);
    StoreU32(&data, record_offset + kPipelineEntryPointLengthOffset,
             static_cast<uint32_t>(pipeline.entry_point.size()));
    StoreU32(&data, record_offset + kPipelineMaxThreadsOffset,
             pipeline.max_threads_per_threadgroup);
    StoreU32(&data, record_offset + kPipelineThreadgroupSizeXOffset,
             pipeline.threadgroup_size[0]);
    StoreU32(&data, record_offset + kPipelineThreadgroupSizeYOffset,
             pipeline.threadgroup_size[1]);
    StoreU32(&data, record_offset + kPipelineThreadgroupSizeZOffset,
             pipeline.threadgroup_size[2]);
    StoreU32(&data, record_offset + kPipelineFlagsOffset, pipeline.flags);
    StoreU32(&data, record_offset + kPipelineConstantCountOffset,
             pipeline.constant_count);
    StoreU32(&data, record_offset + kPipelineBindingCountOffset,
             pipeline.binding_count);
    StoreU64(&data, record_offset + kPipelineBindingReadOnlyBitsOffset,
             pipeline.binding_read_only_bits);
    ++ordinal;
  }
  return data;
}

static std::vector<uint8_t> BuildValidBundle() {
  return BuildBundle(
      {
          {"kernel void first() {}", 0x00030000, std::string_view("MTLB\0", 5)},
          {{},
           IREE_HAL_METAL_EXECUTABLE_SOURCE_VERSION_DEFAULT,
           std::string_view("MTLB", 4)},
      },
      {
          {0,
           "first",
           128,
           {32, 2, 1},
           IREE_HAL_METAL_EXECUTABLE_PIPELINE_FLAG_THREADGROUP_SIZE_ALIGNED,
           3,
           4,
           0b0101},
          {1, "second", 0, {4, 2, 1}, 0, 7, 8, 0b10101010},
      });
}

static iree_const_byte_span_t MakeSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

static iree_host_size_t LibraryRecordOffset(iree_host_size_t ordinal) {
  return IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE +
         ordinal * IREE_HAL_METAL_EXECUTABLE_FORMAT_LIBRARY_SIZE;
}

static iree_host_size_t PipelineRecordOffset(const std::vector<uint8_t>& data,
                                             iree_host_size_t ordinal) {
  const uint32_t library_count =
      iree_unaligned_load_le_u32(data.data() + kHeaderLibraryCountOffset);
  return IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE +
         (iree_host_size_t)library_count *
             IREE_HAL_METAL_EXECUTABLE_FORMAT_LIBRARY_SIZE +
         ordinal * IREE_HAL_METAL_EXECUTABLE_FORMAT_PIPELINE_SIZE;
}

TEST(MetalExecutableFormatTest, ParsesVersionedBundleFromUnalignedStorage) {
  std::vector<uint8_t> data = BuildValidBundle();
  data.insert(data.begin(), 0xCD);
  const iree_const_byte_span_t unaligned_data =
      iree_make_const_byte_span(data.data() + 1, data.size() - 1);

  iree_hal_metal_executable_format_t format;
  IREE_ASSERT_OK(
      iree_hal_metal_executable_format_parse(unaligned_data, &format));
  EXPECT_EQ(format.library_count, 2u);
  EXPECT_EQ(format.pipeline_count, 2u);

  iree_hal_metal_executable_library_t library;
  IREE_ASSERT_OK(
      iree_hal_metal_executable_format_read_library(&format, 0, &library));
  EXPECT_TRUE(iree_string_view_equal(library.source,
                                     IREE_SV("kernel void first() {}")));
  EXPECT_EQ(library.source_version, 0x00030000u);
  ASSERT_EQ(library.metallib.data_length, 5u);
  EXPECT_EQ(memcmp(library.metallib.data, "MTLB\0", 5), 0);

  IREE_ASSERT_OK(
      iree_hal_metal_executable_format_read_library(&format, 1, &library));
  EXPECT_TRUE(iree_string_view_is_empty(library.source));
  EXPECT_EQ(library.source_version,
            IREE_HAL_METAL_EXECUTABLE_SOURCE_VERSION_DEFAULT);
  ASSERT_EQ(library.metallib.data_length, 4u);
  EXPECT_EQ(memcmp(library.metallib.data, "MTLB", 4), 0);

  iree_hal_metal_executable_pipeline_t pipeline;
  IREE_ASSERT_OK(
      iree_hal_metal_executable_format_read_pipeline(&format, 0, &pipeline));
  EXPECT_EQ(pipeline.library_ordinal, 0u);
  EXPECT_TRUE(iree_string_view_equal(pipeline.entry_point, IREE_SV("first")));
  EXPECT_EQ(pipeline.max_threads_per_threadgroup, 128u);
  EXPECT_EQ(pipeline.threadgroup_size[0], 32u);
  EXPECT_EQ(pipeline.threadgroup_size[1], 2u);
  EXPECT_EQ(pipeline.threadgroup_size[2], 1u);
  EXPECT_EQ(pipeline.flags,
            IREE_HAL_METAL_EXECUTABLE_PIPELINE_FLAG_THREADGROUP_SIZE_ALIGNED);
  EXPECT_EQ(pipeline.constant_count, 3u);
  EXPECT_EQ(pipeline.binding_count, 4u);
  EXPECT_EQ(pipeline.binding_read_only_bits, 0b0101u);
}

TEST(MetalExecutableFormatTest, RejectsEveryTruncatedPrefix) {
  const std::vector<uint8_t> data = BuildValidBundle();
  for (iree_host_size_t length = 0; length < data.size(); ++length) {
    SCOPED_TRACE(length);
    iree_hal_metal_executable_format_t format;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_metal_executable_format_parse(
            iree_make_const_byte_span(data.data(), length), &format));
  }
}

TEST(MetalExecutableFormatTest, RejectsInvalidHeader) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, kHeaderMagicOffset, 0);
  iree_hal_metal_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, kHeaderVersionOffset,
           IREE_HAL_METAL_EXECUTABLE_FORMAT_VERSION + 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));
}

TEST(MetalExecutableFormatTest, RejectsEmptyTables) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, kHeaderLibraryCountOffset, 0);
  iree_hal_metal_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, kHeaderPipelineCountOffset, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));
}

TEST(MetalExecutableFormatTest, RejectsPayloadRangesOutsidePayload) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, LibraryRecordOffset(0) + kLibrarySourceOffsetOffset,
           IREE_HAL_METAL_EXECUTABLE_FORMAT_HEADER_SIZE);
  iree_hal_metal_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data,
           PipelineRecordOffset(data, 0) + kPipelineEntryPointOffsetOffset,
           UINT32_MAX);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));
}

TEST(MetalExecutableFormatTest, RejectsInvalidLibraryRecords) {
  std::vector<uint8_t> data = BuildValidBundle();
  const iree_host_size_t record_offset = LibraryRecordOffset(0);
  StoreU32(&data, record_offset + kLibrarySourceOffsetOffset, 0);
  StoreU32(&data, record_offset + kLibrarySourceLengthOffset, 0);
  StoreU32(&data, record_offset + kLibrarySourceVersionOffset,
           IREE_HAL_METAL_EXECUTABLE_SOURCE_VERSION_DEFAULT);
  StoreU32(&data, record_offset + kLibraryMetallibOffsetOffset, 0);
  StoreU32(&data, record_offset + kLibraryMetallibLengthOffset, 0);
  iree_hal_metal_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, LibraryRecordOffset(1) + kLibrarySourceVersionOffset, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));
}

TEST(MetalExecutableFormatTest, RejectsEmbeddedNulText) {
  std::vector<uint8_t> data = BuildValidBundle();
  const iree_host_size_t source_record_offset = LibraryRecordOffset(0);
  const uint32_t source_offset = iree_unaligned_load_le_u32(
      data.data() + source_record_offset + kLibrarySourceOffsetOffset);
  data[source_offset + 1] = 0;
  iree_hal_metal_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  const iree_host_size_t pipeline_record_offset = PipelineRecordOffset(data, 0);
  const uint32_t entry_point_offset = iree_unaligned_load_le_u32(
      data.data() + pipeline_record_offset + kPipelineEntryPointOffsetOffset);
  data[entry_point_offset + 1] = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));
}

TEST(MetalExecutableFormatTest, RejectsInvalidPipelineRecords) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, PipelineRecordOffset(data, 0) + kPipelineLibraryOrdinalOffset,
           2);
  iree_hal_metal_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data,
           PipelineRecordOffset(data, 0) + kPipelineThreadgroupSizeYOffset, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, PipelineRecordOffset(data, 0) + kPipelineMaxThreadsOffset,
           63);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));
}

TEST(MetalExecutableFormatTest, RejectsUnsupportedPipelineMetadata) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, PipelineRecordOffset(data, 0) + kPipelineFlagsOffset,
           UINT32_C(1) << 31);
  iree_hal_metal_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, PipelineRecordOffset(data, 0) + kPipelineConstantCountOffset,
           IREE_HAL_METAL_MAX_PUSH_CONSTANT_COUNT + 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, PipelineRecordOffset(data, 0) + kPipelineBindingCountOffset,
           IREE_HAL_METAL_MAX_DESCRIPTOR_SET_BINDING_COUNT + 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU64(&data,
           PipelineRecordOffset(data, 0) + kPipelineBindingReadOnlyBitsOffset,
           UINT64_C(1) << 4);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));
}

TEST(MetalExecutableFormatTest, RejectsRecordOrdinalsOutsideTables) {
  const std::vector<uint8_t> data = BuildValidBundle();
  iree_hal_metal_executable_format_t format;
  IREE_ASSERT_OK(
      iree_hal_metal_executable_format_parse(MakeSpan(data), &format));
  iree_hal_metal_executable_library_t library;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_metal_executable_format_read_library(&format, 2, &library));
  iree_hal_metal_executable_pipeline_t pipeline;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_metal_executable_format_read_pipeline(&format, 2, &pipeline));
}

}  // namespace
