// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_executable_format.h"

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
  kHeaderExportCountOffset = 8,
  kExportWgslSourceOffsetOffset = 0,
  kExportWgslSourceLengthOffset = 4,
  kExportEntryPointOffsetOffset = 8,
  kExportEntryPointLengthOffset = 12,
  kExportWorkgroupSizeXOffset = 16,
  kExportWorkgroupSizeYOffset = 20,
  kExportWorkgroupSizeZOffset = 24,
  kExportBindingCountOffset = 28,
};

struct TestExport {
  // WGSL source stored in the bundle payload.
  std::string_view wgsl_source;
  // Entry-point name stored in the bundle payload.
  std::string_view entry_point;
  // Static workgroup size encoded in the export record.
  std::array<uint32_t, 3> workgroup_size;
  // Number of resource bindings encoded in the export record.
  uint32_t binding_count;
};

static void StoreU32(std::vector<uint8_t>* data, iree_host_size_t offset,
                     uint32_t value) {
  iree_unaligned_store_le_u32(data->data() + offset, value);
}

static uint32_t AppendString(std::vector<uint8_t>* data,
                             std::string_view value) {
  const uint32_t offset = static_cast<uint32_t>(data->size());
  data->insert(data->end(), value.begin(), value.end());
  return offset;
}

static std::vector<uint8_t> BuildBundle(
    std::initializer_list<TestExport> exports) {
  const iree_host_size_t table_length =
      IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
      exports.size() * IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_EXPORT_SIZE;
  std::vector<uint8_t> data(table_length, 0);
  StoreU32(&data, kHeaderMagicOffset, IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_MAGIC);
  StoreU32(&data, kHeaderVersionOffset,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_VERSION);
  StoreU32(&data, kHeaderExportCountOffset,
           static_cast<uint32_t>(exports.size()));

  iree_host_size_t ordinal = 0;
  for (const TestExport& export_def : exports) {
    const uint32_t wgsl_source_offset =
        AppendString(&data, export_def.wgsl_source);
    const uint32_t entry_point_offset =
        AppendString(&data, export_def.entry_point);
    const iree_host_size_t record_offset =
        IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
        ordinal * IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_EXPORT_SIZE;
    StoreU32(&data, record_offset + kExportWgslSourceOffsetOffset,
             wgsl_source_offset);
    StoreU32(&data, record_offset + kExportWgslSourceLengthOffset,
             static_cast<uint32_t>(export_def.wgsl_source.size()));
    StoreU32(&data, record_offset + kExportEntryPointOffsetOffset,
             entry_point_offset);
    StoreU32(&data, record_offset + kExportEntryPointLengthOffset,
             static_cast<uint32_t>(export_def.entry_point.size()));
    StoreU32(&data, record_offset + kExportWorkgroupSizeXOffset,
             export_def.workgroup_size[0]);
    StoreU32(&data, record_offset + kExportWorkgroupSizeYOffset,
             export_def.workgroup_size[1]);
    StoreU32(&data, record_offset + kExportWorkgroupSizeZOffset,
             export_def.workgroup_size[2]);
    StoreU32(&data, record_offset + kExportBindingCountOffset,
             export_def.binding_count);
    ++ordinal;
  }
  return data;
}

static iree_const_byte_span_t MakeSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

TEST(WebGPUExecutableFormatTest, ParsesVersionedBundleFromUnalignedStorage) {
  std::vector<uint8_t> data = BuildBundle({
      {
          "@compute @workgroup_size(64) fn first() {}",
          "first",
          {64, 1, 1},
          3,
      },
      {
          "@compute @workgroup_size(4, 2, 1) fn second() {}",
          "second",
          {4, 2, 1},
          7,
      },
  });
  data.insert(data.begin(), 0xCD);
  const iree_const_byte_span_t unaligned_data =
      iree_make_const_byte_span(data.data() + 1, data.size() - 1);

  iree_hal_webgpu_executable_format_t format;
  IREE_ASSERT_OK(
      iree_hal_webgpu_executable_format_parse(unaligned_data, &format));
  EXPECT_EQ(format.export_count, 2u);

  iree_hal_webgpu_executable_export_t export_def;
  IREE_ASSERT_OK(
      iree_hal_webgpu_executable_format_read_export(&format, 0, &export_def));
  EXPECT_TRUE(iree_string_view_equal(
      export_def.wgsl_source,
      IREE_SV("@compute @workgroup_size(64) fn first() {}")));
  EXPECT_TRUE(iree_string_view_equal(export_def.entry_point, IREE_SV("first")));
  EXPECT_EQ(export_def.workgroup_size[0], 64u);
  EXPECT_EQ(export_def.workgroup_size[1], 1u);
  EXPECT_EQ(export_def.workgroup_size[2], 1u);
  EXPECT_EQ(export_def.binding_count, 3u);

  IREE_ASSERT_OK(
      iree_hal_webgpu_executable_format_read_export(&format, 1, &export_def));
  EXPECT_TRUE(iree_string_view_equal(
      export_def.wgsl_source,
      IREE_SV("@compute @workgroup_size(4, 2, 1) fn second() {}")));
  EXPECT_TRUE(
      iree_string_view_equal(export_def.entry_point, IREE_SV("second")));
  EXPECT_EQ(export_def.workgroup_size[0], 4u);
  EXPECT_EQ(export_def.workgroup_size[1], 2u);
  EXPECT_EQ(export_def.workgroup_size[2], 1u);
  EXPECT_EQ(export_def.binding_count, 7u);
}

TEST(WebGPUExecutableFormatTest, RejectsEveryTruncatedPrefix) {
  const std::vector<uint8_t> data = BuildBundle({
      {
          "@compute @workgroup_size(1) fn main() {}",
          "main",
          {1, 1, 1},
          0,
      },
  });
  for (iree_host_size_t length = 0; length < data.size(); ++length) {
    SCOPED_TRACE(length);
    iree_hal_webgpu_executable_format_t format;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_webgpu_executable_format_parse(
            iree_make_const_byte_span(data.data(), length), &format));
  }
}

TEST(WebGPUExecutableFormatTest, RejectsInvalidMagic) {
  std::vector<uint8_t> data = BuildBundle({
      {"source", "main", {1, 1, 1}, 0},
  });
  StoreU32(&data, kHeaderMagicOffset, 0);
  iree_hal_webgpu_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
}

TEST(WebGPUExecutableFormatTest, RejectsUnsupportedVersion) {
  std::vector<uint8_t> data = BuildBundle({
      {"source", "main", {1, 1, 1}, 0},
  });
  StoreU32(&data, kHeaderVersionOffset,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_VERSION + 1);
  iree_hal_webgpu_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
}

TEST(WebGPUExecutableFormatTest, RejectsEmptyExportTable) {
  std::vector<uint8_t> data(IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE, 0);
  StoreU32(&data, kHeaderMagicOffset, IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_MAGIC);
  StoreU32(&data, kHeaderVersionOffset,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_VERSION);
  iree_hal_webgpu_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
}

TEST(WebGPUExecutableFormatTest, RejectsPayloadRangeInsideExportTable) {
  std::vector<uint8_t> data = BuildBundle({
      {"source", "main", {1, 1, 1}, 0},
  });
  StoreU32(&data,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
               kExportWgslSourceOffsetOffset,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE);
  iree_hal_webgpu_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
}

TEST(WebGPUExecutableFormatTest, RejectsPayloadRangeOutsideBundle) {
  std::vector<uint8_t> data = BuildBundle({
      {"source", "main", {1, 1, 1}, 0},
  });
  StoreU32(&data,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
               kExportEntryPointOffsetOffset,
           UINT32_MAX);
  iree_hal_webgpu_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
}

TEST(WebGPUExecutableFormatTest, RejectsEmptyPayloads) {
  std::vector<uint8_t> data = BuildBundle({
      {"source", "main", {1, 1, 1}, 0},
  });
  StoreU32(&data,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
               kExportWgslSourceLengthOffset,
           0);
  iree_hal_webgpu_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));

  data = BuildBundle({
      {"source", "main", {1, 1, 1}, 0},
  });
  StoreU32(&data,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
               kExportEntryPointLengthOffset,
           0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
}

TEST(WebGPUExecutableFormatTest, RejectsZeroWorkgroupDimension) {
  std::vector<uint8_t> data = BuildBundle({
      {"source", "main", {1, 1, 1}, 0},
  });
  StoreU32(&data,
           IREE_HAL_WEBGPU_EXECUTABLE_FORMAT_HEADER_SIZE +
               kExportWorkgroupSizeYOffset,
           0);
  iree_hal_webgpu_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
}

TEST(WebGPUExecutableFormatTest, RejectsBindingCountBeyondHalLimit) {
  std::vector<uint8_t> data = BuildBundle({
      {"source", "main", {1, 1, 1}, UINT16_MAX + 1u},
  });
  iree_hal_webgpu_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
}

TEST(WebGPUExecutableFormatTest, RejectsExportOrdinalOutsideTable) {
  const std::vector<uint8_t> data = BuildBundle({
      {"source", "main", {1, 1, 1}, 0},
  });
  iree_hal_webgpu_executable_format_t format;
  IREE_ASSERT_OK(
      iree_hal_webgpu_executable_format_parse(MakeSpan(data), &format));
  iree_hal_webgpu_executable_export_t export_def;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_webgpu_executable_format_read_export(&format, 1, &export_def));
}

}  // namespace
