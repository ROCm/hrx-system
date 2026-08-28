// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/executable/format.h"

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
  kHeaderModuleCountOffset = 8,
  kHeaderExportCountOffset = 12,
  kModuleHsacoImageOffsetOffset = 0,
  kModuleHsacoImageLengthOffset = 4,
  kExportModuleOrdinalOffset = 0,
  kExportKernelNameOffsetOffset = 4,
  kExportKernelNameLengthOffset = 8,
  kExportBlockSizeXOffset = 12,
  kExportBlockSizeYOffset = 16,
  kExportBlockSizeZOffset = 20,
  kExportConstantCountOffset = 24,
  kExportBindingCountOffset = 28,
  kElfProgramHeaderOffset = 64,
  kElfSectionHeaderOffset = 120,
  kElfLength = 184,
};

struct TestExport {
  // Ordinal of the HSACO module containing the kernel.
  uint32_t module_ordinal;
  // HIP kernel symbol stored in the bundle payload.
  std::string_view kernel_name;
  // Static HIP block size encoded in the export record.
  std::array<uint32_t, 3> block_size;
  // Number of dispatch constants encoded in the export record.
  uint32_t constant_count;
  // Number of buffer bindings encoded in the export record.
  uint32_t binding_count;
};

static void StoreU16(std::vector<uint8_t>* data, iree_host_size_t offset,
                     uint16_t value) {
  iree_unaligned_store_le_u16(data->data() + offset, value);
}

static void StoreU32(std::vector<uint8_t>* data, iree_host_size_t offset,
                     uint32_t value) {
  iree_unaligned_store_le_u32(data->data() + offset, value);
}

static void StoreU64(std::vector<uint8_t>* data, iree_host_size_t offset,
                     uint64_t value) {
  iree_unaligned_store_le_u64(data->data() + offset, value);
}

static std::vector<uint8_t> BuildMinimalHsaco() {
  std::vector<uint8_t> data(kElfLength, 0);
  data[0] = 0x7F;
  data[1] = 'E';
  data[2] = 'L';
  data[3] = 'F';
  data[4] = 2;               // ELFCLASS64
  data[5] = 1;               // ELFDATA2LSB
  data[6] = 1;               // EV_CURRENT
  StoreU16(&data, 16, 3);    // ET_DYN
  StoreU16(&data, 18, 224);  // EM_AMDGPU
  StoreU32(&data, 20, 1);    // EV_CURRENT
  StoreU64(&data, 32, kElfProgramHeaderOffset);
  StoreU64(&data, 40, kElfSectionHeaderOffset);
  StoreU16(&data, 52, 64);
  StoreU16(&data, 54, 56);
  StoreU16(&data, 56, 1);
  StoreU16(&data, 58, 64);
  StoreU16(&data, 60, 1);
  StoreU16(&data, 62, 0);
  StoreU32(&data, kElfProgramHeaderOffset, 1);  // PT_LOAD
  StoreU64(&data, kElfProgramHeaderOffset + 8, kElfLength);
  StoreU64(&data, kElfProgramHeaderOffset + 32, 0);
  return data;
}

static uint32_t AppendBytes(std::vector<uint8_t>* data,
                            const std::vector<uint8_t>& value) {
  const uint32_t offset = static_cast<uint32_t>(data->size());
  data->insert(data->end(), value.begin(), value.end());
  return offset;
}

static uint32_t AppendCString(std::vector<uint8_t>* data,
                              std::string_view value) {
  const uint32_t offset = static_cast<uint32_t>(data->size());
  data->insert(data->end(), value.begin(), value.end());
  data->push_back(0);
  return offset;
}

static std::vector<uint8_t> BuildBundle(
    std::initializer_list<std::vector<uint8_t>> modules,
    std::initializer_list<TestExport> exports) {
  const iree_host_size_t module_table_length =
      modules.size() * IREE_HAL_HIP_EXECUTABLE_FORMAT_MODULE_SIZE;
  const iree_host_size_t table_length =
      IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE + module_table_length +
      exports.size() * IREE_HAL_HIP_EXECUTABLE_FORMAT_EXPORT_SIZE;
  std::vector<uint8_t> data(table_length, 0);
  StoreU32(&data, kHeaderMagicOffset, IREE_HAL_HIP_EXECUTABLE_FORMAT_MAGIC);
  StoreU32(&data, kHeaderVersionOffset, IREE_HAL_HIP_EXECUTABLE_FORMAT_VERSION);
  StoreU32(&data, kHeaderModuleCountOffset,
           static_cast<uint32_t>(modules.size()));
  StoreU32(&data, kHeaderExportCountOffset,
           static_cast<uint32_t>(exports.size()));

  iree_host_size_t ordinal = 0;
  for (const std::vector<uint8_t>& hsaco_image : modules) {
    const uint32_t hsaco_image_offset = AppendBytes(&data, hsaco_image);
    const iree_host_size_t record_offset =
        IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE +
        ordinal * IREE_HAL_HIP_EXECUTABLE_FORMAT_MODULE_SIZE;
    StoreU32(&data, record_offset + kModuleHsacoImageOffsetOffset,
             hsaco_image_offset);
    StoreU32(&data, record_offset + kModuleHsacoImageLengthOffset,
             static_cast<uint32_t>(hsaco_image.size()));
    ++ordinal;
  }

  ordinal = 0;
  for (const TestExport& export_def : exports) {
    const uint32_t kernel_name_offset =
        AppendCString(&data, export_def.kernel_name);
    const iree_host_size_t record_offset =
        IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE + module_table_length +
        ordinal * IREE_HAL_HIP_EXECUTABLE_FORMAT_EXPORT_SIZE;
    StoreU32(&data, record_offset + kExportModuleOrdinalOffset,
             export_def.module_ordinal);
    StoreU32(&data, record_offset + kExportKernelNameOffsetOffset,
             kernel_name_offset);
    StoreU32(&data, record_offset + kExportKernelNameLengthOffset,
             static_cast<uint32_t>(export_def.kernel_name.size()));
    StoreU32(&data, record_offset + kExportBlockSizeXOffset,
             export_def.block_size[0]);
    StoreU32(&data, record_offset + kExportBlockSizeYOffset,
             export_def.block_size[1]);
    StoreU32(&data, record_offset + kExportBlockSizeZOffset,
             export_def.block_size[2]);
    StoreU32(&data, record_offset + kExportConstantCountOffset,
             export_def.constant_count);
    StoreU32(&data, record_offset + kExportBindingCountOffset,
             export_def.binding_count);
    ++ordinal;
  }
  return data;
}

static std::vector<uint8_t> BuildValidBundle() {
  return BuildBundle({BuildMinimalHsaco(), BuildMinimalHsaco()},
                     {
                         {0, "first", {64, 1, 1}, 3, 4},
                         {0, "second", {4, 2, 1}, 7, 8},
                         {1, "third", {32, 1, 1}, 0, 2},
                     });
}

static iree_const_byte_span_t MakeSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

static iree_host_size_t ModuleRecordOffset(iree_host_size_t ordinal) {
  return IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE +
         ordinal * IREE_HAL_HIP_EXECUTABLE_FORMAT_MODULE_SIZE;
}

static iree_host_size_t ModulePayloadOffset(const std::vector<uint8_t>& data,
                                            iree_host_size_t ordinal) {
  return iree_unaligned_load_le_u32(data.data() + ModuleRecordOffset(ordinal) +
                                    kModuleHsacoImageOffsetOffset);
}

static iree_host_size_t ExportRecordOffset(const std::vector<uint8_t>& data,
                                           iree_host_size_t ordinal) {
  const uint32_t module_count =
      iree_unaligned_load_le_u32(data.data() + kHeaderModuleCountOffset);
  return IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE +
         (iree_host_size_t)module_count *
             IREE_HAL_HIP_EXECUTABLE_FORMAT_MODULE_SIZE +
         ordinal * IREE_HAL_HIP_EXECUTABLE_FORMAT_EXPORT_SIZE;
}

TEST(HIPExecutableFormatTest, ParsesVersionedBundleFromUnalignedStorage) {
  std::vector<uint8_t> data = BuildValidBundle();
  data.insert(data.begin(), 0xCD);
  const iree_const_byte_span_t unaligned_data =
      iree_make_const_byte_span(data.data() + 1, data.size() - 1);

  iree_hal_hip_executable_format_t format;
  IREE_ASSERT_OK(iree_hal_hip_executable_format_parse(unaligned_data, &format));
  EXPECT_EQ(format.module_count, 2u);
  EXPECT_EQ(format.export_count, 3u);

  iree_hal_hip_executable_module_t module;
  IREE_ASSERT_OK(
      iree_hal_hip_executable_format_read_module(&format, 1, &module));
  EXPECT_EQ(module.hsaco_image.data_length, kElfLength);
  EXPECT_EQ(module.hsaco_image.data[0], 0x7F);

  iree_hal_hip_executable_export_t export_def;
  IREE_ASSERT_OK(
      iree_hal_hip_executable_format_read_export(&format, 1, &export_def));
  EXPECT_EQ(export_def.module_ordinal, 0u);
  EXPECT_TRUE(
      iree_string_view_equal(export_def.kernel_name, IREE_SV("second")));
  EXPECT_EQ(export_def.kernel_name.data[export_def.kernel_name.size], '\0');
  EXPECT_EQ(export_def.block_size[0], 4u);
  EXPECT_EQ(export_def.block_size[1], 2u);
  EXPECT_EQ(export_def.block_size[2], 1u);
  EXPECT_EQ(export_def.constant_count, 7u);
  EXPECT_EQ(export_def.binding_count, 8u);
}

TEST(HIPExecutableFormatTest, RejectsEveryTruncatedPrefix) {
  const std::vector<uint8_t> data = BuildValidBundle();
  for (iree_host_size_t length = 0; length < data.size(); ++length) {
    SCOPED_TRACE(length);
    iree_hal_hip_executable_format_t format;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_hip_executable_format_parse(
            iree_make_const_byte_span(data.data(), length), &format));
  }
}

TEST(HIPExecutableFormatTest, RejectsInvalidMagicAndVersion) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, kHeaderMagicOffset, 0);
  iree_hal_hip_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, kHeaderVersionOffset,
           IREE_HAL_HIP_EXECUTABLE_FORMAT_VERSION + 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));
}

TEST(HIPExecutableFormatTest, RejectsEmptyTables) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, kHeaderModuleCountOffset, 0);
  iree_hal_hip_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, kHeaderExportCountOffset, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));
}

TEST(HIPExecutableFormatTest, RejectsPayloadRangesOutsidePayload) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, ModuleRecordOffset(0) + kModuleHsacoImageOffsetOffset,
           IREE_HAL_HIP_EXECUTABLE_FORMAT_HEADER_SIZE);
  iree_hal_hip_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportKernelNameLengthOffset,
           UINT32_MAX);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));
}

TEST(HIPExecutableFormatTest, RejectsEmptyPayloads) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, ModuleRecordOffset(0) + kModuleHsacoImageLengthOffset, 0);
  iree_hal_hip_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportKernelNameLengthOffset,
           0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));
}

TEST(HIPExecutableFormatTest, RejectsMalformedHsacoIdentity) {
  std::vector<uint8_t> data = BuildValidBundle();
  data[ModulePayloadOffset(data, 0)] = 0;
  iree_hal_hip_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  data[ModulePayloadOffset(data, 0) + 4] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU16(&data, ModulePayloadOffset(data, 0) + 18, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));
}

TEST(HIPExecutableFormatTest, RejectsHsacoTablesAndExtentsOutsideImage) {
  std::vector<uint8_t> data = BuildValidBundle();
  const iree_host_size_t hsaco_offset = ModulePayloadOffset(data, 0);
  StoreU64(&data, hsaco_offset + 32, kElfLength - 1);
  iree_hal_hip_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU64(&data, ModulePayloadOffset(data, 0) + kElfProgramHeaderOffset + 32,
           1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  const iree_host_size_t section_header_offset =
      ModulePayloadOffset(data, 0) + kElfSectionHeaderOffset;
  StoreU32(&data, section_header_offset + 4, 1);
  StoreU64(&data, section_header_offset + 24, kElfLength);
  StoreU64(&data, section_header_offset + 32, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));
}

TEST(HIPExecutableFormatTest, RejectsInvalidExportMetadata) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportModuleOrdinalOffset, 2);
  iree_hal_hip_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportBlockSizeYOffset, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  const iree_host_size_t export_record_offset = ExportRecordOffset(data, 0);
  const uint32_t kernel_name_offset = iree_unaligned_load_le_u32(
      data.data() + export_record_offset + kExportKernelNameOffsetOffset);
  const uint32_t kernel_name_length = iree_unaligned_load_le_u32(
      data.data() + export_record_offset + kExportKernelNameLengthOffset);
  data[kernel_name_offset + kernel_name_length] = 'x';
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  const iree_host_size_t embedded_nul_export_record_offset =
      ExportRecordOffset(data, 0);
  const uint32_t embedded_nul_kernel_name_offset = iree_unaligned_load_le_u32(
      data.data() + embedded_nul_export_record_offset +
      kExportKernelNameOffsetOffset);
  data[embedded_nul_kernel_name_offset + 1] = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_hip_executable_format_parse(MakeSpan(data), &format));
}

TEST(HIPExecutableFormatTest, RejectsRecordOrdinalsOutsideTables) {
  const std::vector<uint8_t> data = BuildValidBundle();
  iree_hal_hip_executable_format_t format;
  IREE_ASSERT_OK(iree_hal_hip_executable_format_parse(MakeSpan(data), &format));
  iree_hal_hip_executable_module_t module;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_hip_executable_format_read_module(&format, 2, &module));
  iree_hal_hip_executable_export_t export_def;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_hip_executable_format_read_export(&format, 3, &export_def));
}

}  // namespace
