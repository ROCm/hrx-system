// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/cuda_executable_format.h"

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
  kModulePtxImageOffsetOffset = 0,
  kModulePtxImageLengthOffset = 4,
  kExportModuleOrdinalOffset = 0,
  kExportKernelNameOffsetOffset = 4,
  kExportKernelNameLengthOffset = 8,
  kExportBlockSizeXOffset = 12,
  kExportBlockSizeYOffset = 16,
  kExportBlockSizeZOffset = 20,
  kExportBlockSharedMemorySizeOffset = 24,
  kExportConstantCountOffset = 28,
  kExportBindingCountOffset = 32,
};

struct TestExport {
  // Ordinal of the PTX module containing the kernel.
  uint32_t module_ordinal;
  // CUDA kernel symbol stored in the bundle payload.
  std::string_view kernel_name;
  // Static CUDA block size encoded in the export record.
  std::array<uint32_t, 3> block_size;
  // Dynamic shared memory size encoded in the export record.
  uint32_t block_shared_memory_size;
  // Number of dispatch constants encoded in the export record.
  uint32_t constant_count;
  // Number of buffer bindings encoded in the export record.
  uint32_t binding_count;
};

static void StoreU32(std::vector<uint8_t>* data, iree_host_size_t offset,
                     uint32_t value) {
  iree_unaligned_store_le_u32(data->data() + offset, value);
}

static uint32_t AppendCString(std::vector<uint8_t>* data,
                              std::string_view value) {
  const uint32_t offset = static_cast<uint32_t>(data->size());
  data->insert(data->end(), value.begin(), value.end());
  data->push_back(0);
  return offset;
}

static std::vector<uint8_t> BuildBundle(
    std::initializer_list<std::string_view> modules,
    std::initializer_list<TestExport> exports) {
  const iree_host_size_t module_table_length =
      modules.size() * IREE_HAL_CUDA_EXECUTABLE_FORMAT_MODULE_SIZE;
  const iree_host_size_t table_length =
      IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE + module_table_length +
      exports.size() * IREE_HAL_CUDA_EXECUTABLE_FORMAT_EXPORT_SIZE;
  std::vector<uint8_t> data(table_length, 0);
  StoreU32(&data, kHeaderMagicOffset, IREE_HAL_CUDA_EXECUTABLE_FORMAT_MAGIC);
  StoreU32(&data, kHeaderVersionOffset,
           IREE_HAL_CUDA_EXECUTABLE_FORMAT_VERSION);
  StoreU32(&data, kHeaderModuleCountOffset,
           static_cast<uint32_t>(modules.size()));
  StoreU32(&data, kHeaderExportCountOffset,
           static_cast<uint32_t>(exports.size()));

  iree_host_size_t ordinal = 0;
  for (std::string_view ptx_image : modules) {
    const uint32_t ptx_image_offset = AppendCString(&data, ptx_image);
    const iree_host_size_t record_offset =
        IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE +
        ordinal * IREE_HAL_CUDA_EXECUTABLE_FORMAT_MODULE_SIZE;
    StoreU32(&data, record_offset + kModulePtxImageOffsetOffset,
             ptx_image_offset);
    StoreU32(&data, record_offset + kModulePtxImageLengthOffset,
             static_cast<uint32_t>(ptx_image.size()));
    ++ordinal;
  }

  ordinal = 0;
  for (const TestExport& export_def : exports) {
    const uint32_t kernel_name_offset =
        AppendCString(&data, export_def.kernel_name);
    const iree_host_size_t record_offset =
        IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE + module_table_length +
        ordinal * IREE_HAL_CUDA_EXECUTABLE_FORMAT_EXPORT_SIZE;
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
    StoreU32(&data, record_offset + kExportBlockSharedMemorySizeOffset,
             export_def.block_shared_memory_size);
    StoreU32(&data, record_offset + kExportConstantCountOffset,
             export_def.constant_count);
    StoreU32(&data, record_offset + kExportBindingCountOffset,
             export_def.binding_count);
    ++ordinal;
  }
  return data;
}

static std::vector<uint8_t> BuildValidBundle() {
  return BuildBundle(
      {".version 8.0\n.target sm_80\n", ".version 8.0\n.target sm_90\n"},
      {
          {0, "first", {64, 1, 1}, 128, 3, 4},
          {0, "second", {4, 2, 1}, 0, 7, 8},
          {1, "third", {32, 1, 1}, 256, 0, 2},
      });
}

static iree_const_byte_span_t MakeSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

static iree_host_size_t ModuleRecordOffset(iree_host_size_t ordinal) {
  return IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE +
         ordinal * IREE_HAL_CUDA_EXECUTABLE_FORMAT_MODULE_SIZE;
}

static iree_host_size_t ExportRecordOffset(const std::vector<uint8_t>& data,
                                           iree_host_size_t ordinal) {
  const uint32_t module_count =
      iree_unaligned_load_le_u32(data.data() + kHeaderModuleCountOffset);
  return IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE +
         (iree_host_size_t)module_count *
             IREE_HAL_CUDA_EXECUTABLE_FORMAT_MODULE_SIZE +
         ordinal * IREE_HAL_CUDA_EXECUTABLE_FORMAT_EXPORT_SIZE;
}

TEST(CUDAExecutableFormatTest, ParsesVersionedBundleFromUnalignedStorage) {
  std::vector<uint8_t> data = BuildValidBundle();
  data.insert(data.begin(), 0xCD);
  const iree_const_byte_span_t unaligned_data =
      iree_make_const_byte_span(data.data() + 1, data.size() - 1);

  iree_hal_cuda_executable_format_t format;
  IREE_ASSERT_OK(
      iree_hal_cuda_executable_format_parse(unaligned_data, &format));
  EXPECT_EQ(format.module_count, 2u);
  EXPECT_EQ(format.export_count, 3u);

  iree_hal_cuda_executable_module_t module;
  IREE_ASSERT_OK(
      iree_hal_cuda_executable_format_read_module(&format, 1, &module));
  EXPECT_TRUE(iree_string_view_equal(module.ptx_image,
                                     IREE_SV(".version 8.0\n.target sm_90\n")));
  EXPECT_EQ(module.ptx_image.data[module.ptx_image.size], '\0');

  iree_hal_cuda_executable_export_t export_def;
  IREE_ASSERT_OK(
      iree_hal_cuda_executable_format_read_export(&format, 1, &export_def));
  EXPECT_EQ(export_def.module_ordinal, 0u);
  EXPECT_TRUE(
      iree_string_view_equal(export_def.kernel_name, IREE_SV("second")));
  EXPECT_EQ(export_def.kernel_name.data[export_def.kernel_name.size], '\0');
  EXPECT_EQ(export_def.block_size[0], 4u);
  EXPECT_EQ(export_def.block_size[1], 2u);
  EXPECT_EQ(export_def.block_size[2], 1u);
  EXPECT_EQ(export_def.block_shared_memory_size, 0u);
  EXPECT_EQ(export_def.constant_count, 7u);
  EXPECT_EQ(export_def.binding_count, 8u);
}

TEST(CUDAExecutableFormatTest, RejectsEveryTruncatedPrefix) {
  const std::vector<uint8_t> data = BuildValidBundle();
  for (iree_host_size_t length = 0; length < data.size(); ++length) {
    SCOPED_TRACE(length);
    iree_hal_cuda_executable_format_t format;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_cuda_executable_format_parse(
            iree_make_const_byte_span(data.data(), length), &format));
  }
}

TEST(CUDAExecutableFormatTest, RejectsInvalidMagic) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, kHeaderMagicOffset, 0);
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsUnsupportedVersion) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, kHeaderVersionOffset,
           IREE_HAL_CUDA_EXECUTABLE_FORMAT_VERSION + 1);
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsEmptyTables) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, kHeaderModuleCountOffset, 0);
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, kHeaderExportCountOffset, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsPayloadRangesInsideTables) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, ModuleRecordOffset(0) + kModulePtxImageOffsetOffset,
           IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE);
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportKernelNameOffsetOffset,
           IREE_HAL_CUDA_EXECUTABLE_FORMAT_HEADER_SIZE);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsPayloadRangesOutsideBundle) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, ModuleRecordOffset(0) + kModulePtxImageOffsetOffset,
           UINT32_MAX);
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportKernelNameLengthOffset,
           UINT32_MAX);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsEmptyStrings) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, ModuleRecordOffset(0) + kModulePtxImageLengthOffset, 0);
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportKernelNameLengthOffset,
           0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsUnterminatedAndEmbeddedNulStrings) {
  std::vector<uint8_t> data = BuildValidBundle();
  const iree_host_size_t module_record_offset = ModuleRecordOffset(0);
  const uint32_t ptx_image_offset = iree_unaligned_load_le_u32(
      data.data() + module_record_offset + kModulePtxImageOffsetOffset);
  const uint32_t ptx_image_length = iree_unaligned_load_le_u32(
      data.data() + module_record_offset + kModulePtxImageLengthOffset);
  data[ptx_image_offset + ptx_image_length] = 'x';
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));

  data = BuildValidBundle();
  const iree_host_size_t export_record_offset = ExportRecordOffset(data, 0);
  const uint32_t kernel_name_offset = iree_unaligned_load_le_u32(
      data.data() + export_record_offset + kExportKernelNameOffsetOffset);
  data[kernel_name_offset + 1] = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsModuleOrdinalOutsideTable) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportModuleOrdinalOffset, 2);
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsZeroBlockDimension) {
  std::vector<uint8_t> data = BuildValidBundle();
  StoreU32(&data, ExportRecordOffset(data, 0) + kExportBlockSizeYOffset, 0);
  iree_hal_cuda_executable_format_t format;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
}

TEST(CUDAExecutableFormatTest, RejectsRecordOrdinalsOutsideTables) {
  const std::vector<uint8_t> data = BuildValidBundle();
  iree_hal_cuda_executable_format_t format;
  IREE_ASSERT_OK(
      iree_hal_cuda_executable_format_parse(MakeSpan(data), &format));
  iree_hal_cuda_executable_module_t module;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_cuda_executable_format_read_module(&format, 2, &module));
  iree_hal_cuda_executable_export_t export_def;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_cuda_executable_format_read_export(&format, 3, &export_def));
}

}  // namespace
