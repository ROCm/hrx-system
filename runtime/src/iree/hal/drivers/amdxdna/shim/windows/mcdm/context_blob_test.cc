// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/shim/windows/mcdm/context_blob.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"

namespace iree::hal::amdxdna::mcdm {
namespace {

void WriteU16(std::vector<uint8_t>* data, size_t offset, uint16_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void WriteU32(std::vector<uint8_t>* data, size_t offset, uint32_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void WriteU64(std::vector<uint8_t>* data, size_t offset, uint64_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

uint32_t ReadU32(const uint8_t* data, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

uint64_t ReadU64(const uint8_t* data, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

void WriteString(std::vector<uint8_t>* data, size_t offset, size_t size,
                 const char* value) {
  std::memset(data->data() + offset, 0, size);
  std::memcpy(data->data() + offset, value, std::strlen(value));
}

std::vector<uint8_t> BuildSyntheticMultiPdiXclbin() {
  constexpr size_t kSectionTableOffset = 0x1C8;
  constexpr size_t kSectionCount = 2;
  constexpr size_t kIpLayoutOffset = 0x220;
  constexpr size_t kIpLayoutSize = 8 + 2 * 80;
  constexpr size_t kAiePartitionOffset = 0x2C8;
  constexpr size_t kAiePartitionSize = 0x300;
  std::vector<uint8_t> xclbin(kAiePartitionOffset + kAiePartitionSize, 0);

  WriteString(&xclbin, 0, 8, "xclbin2");
  for (size_t i = 0; i < 16; ++i) {
    xclbin[0x1A0 + i] = static_cast<uint8_t>(0xA0 + i);
  }
  WriteU32(&xclbin, 0x1C0, static_cast<uint32_t>(kSectionCount));

  // IP_LAYOUT section.
  WriteU32(&xclbin, kSectionTableOffset, 8);
  WriteString(&xclbin, kSectionTableOffset + 4, 16, "ip");
  WriteU64(&xclbin, kSectionTableOffset + 24, kIpLayoutOffset);
  WriteU64(&xclbin, kSectionTableOffset + 32, kIpLayoutSize);
  WriteU32(&xclbin, kIpLayoutOffset, 2);
  size_t ip0 = kIpLayoutOffset + 8;
  WriteU32(&xclbin, ip0 + 0, 1);
  WriteU64(&xclbin, ip0 + 8, 0x80000);
  WriteString(&xclbin, ip0 + 16, 64, "dispatch_a:IREE");
  size_t ip1 = ip0 + 80;
  WriteU32(&xclbin, ip1 + 0, 7);
  WriteU64(&xclbin, ip1 + 8, UINT64_MAX);
  WriteString(&xclbin, ip1 + 16, 64, "dispatch_b:IREE");

  // AIE_PARTITION section.
  size_t aie_record = kSectionTableOffset + 40;
  WriteU32(&xclbin, aie_record, 32);
  WriteString(&xclbin, aie_record + 4, 16, "aie");
  WriteU64(&xclbin, aie_record + 24, kAiePartitionOffset);
  WriteU64(&xclbin, aie_record + 32, kAiePartitionSize);

  size_t aie = kAiePartitionOffset;
  WriteU16(&xclbin, aie + 32, 8);
  WriteU32(&xclbin, aie + 40, 1);
  WriteU32(&xclbin, aie + 44, 0x80);
  WriteU16(&xclbin, aie + 0x80, 2);
  WriteU32(&xclbin, aie + 120, 2);
  WriteU32(&xclbin, aie + 124, 0xC8);

  size_t pdi0 = aie + 0xC8;
  size_t pdi1 = pdi0 + 0x60;
  size_t cdo0 = 0x188;
  size_t cdo1 = 0x1F8;
  WriteU32(&xclbin, pdi0 + 24, 1);
  WriteU32(&xclbin, pdi0 + 28, static_cast<uint32_t>(cdo0));
  WriteU32(&xclbin, pdi1 + 24, 1);
  WriteU32(&xclbin, pdi1 + 28, static_cast<uint32_t>(cdo1));

  WriteU32(&xclbin, aie + cdo0 + 0, 0x268);
  WriteU32(&xclbin, aie + cdo0 + 16, 1);
  WriteU32(&xclbin, aie + cdo0 + 20, 0x2E8);
  WriteU32(&xclbin, aie + cdo1 + 0, 0x2A8);
  WriteU32(&xclbin, aie + cdo1 + 16, 1);
  WriteU32(&xclbin, aie + cdo1 + 20, 0x2F0);
  WriteString(&xclbin, aie + 0x268, 64, "DPU_PDI_0");
  WriteString(&xclbin, aie + 0x2A8, 64, "DPU_PDI_1");
  WriteU64(&xclbin, aie + 0x2E8, 0x100);
  WriteU64(&xclbin, aie + 0x2F0, 0x101);

  return xclbin;
}

TEST(ContextBlobTest, ParsesMultiPdiIpLayoutAndAiePartition) {
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();

  iree_byte_span_t private_data = iree_byte_span_empty();
  ContextBlobInfo info;
  Error error;
  ASSERT_TRUE(BuildContextPrivateDataFromXclbin(xclbin.data(), xclbin.size(),
                                                /*process_id=*/1234,
                                                iree_allocator_system(),
                                                &private_data, &info, &error))
      << ErrorMessage(&error);

  EXPECT_STREQ(info.kernel_name, "dispatch_a");
  ASSERT_EQ(info.kernel_name_count, 2u);
  EXPECT_STREQ(ContextBlobInfoKernelName(&info, 0), "dispatch_a");
  EXPECT_STREQ(ContextBlobInfoKernelName(&info, 1), "dispatch_b");
  EXPECT_EQ(info.pdi_count, 2u);
  ASSERT_EQ(info.pdi_name_count, 2u);
  EXPECT_STREQ(ContextBlobInfoPdiName(&info, 0), "DPU_PDI_0");
  EXPECT_STREQ(ContextBlobInfoPdiName(&info, 1), "DPU_PDI_1");
  ASSERT_NE(info.dpu_kernel_ids, nullptr);
  EXPECT_EQ(info.dpu_kernel_ids[0], 0x100u);
  EXPECT_EQ(info.dpu_kernel_ids[1], 0x101u);
  EXPECT_GT(private_data.data_length, xclbin.size());
  ContextBlobInfoDeinitialize(&info);
  iree_allocator_free(iree_allocator_system(), private_data.data);
}

TEST(ContextBlobTest, BuildsCompactContextPrivateData) {
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();
  Buffer context_buffer;
  context_buffer.kind = BufferKind::context_private;
  context_buffer.size = 0x1000;
  context_buffer.allocation = 0x40000280;
  context_buffer.cpu_ptr = reinterpret_cast<void*>(uintptr_t{0x252EBB80000});

  iree_byte_span_t private_data = iree_byte_span_empty();
  ContextBlobInfo info;
  Error error;
  ASSERT_TRUE(BuildCompactContextPrivateDataFromXclbin(
      xclbin.data(), xclbin.size(), /*process_id=*/0x7CA4, context_buffer,
      iree_allocator_system(), &private_data, &info, &error))
      << ErrorMessage(&error);

  ASSERT_EQ(private_data.data_length, 0xA0u);
  EXPECT_EQ(std::memcmp(private_data.data, xclbin.data() + 0x1A0, 16), 0);
  EXPECT_EQ(ReadU64(private_data.data, 0x48), 0x04000000u);
  EXPECT_EQ(ReadU32(private_data.data, 0x50), 0x7CA4u);
  EXPECT_EQ(ReadU32(private_data.data, 0x54), 8u);
  EXPECT_EQ(ReadU32(private_data.data, 0x5C), 0x800u);
  EXPECT_EQ(ReadU32(private_data.data, 0x68), 0x40000280u);
  EXPECT_EQ(ReadU32(private_data.data, 0x74), 0x1000u);
  EXPECT_EQ(ReadU64(private_data.data, 0x78), 0x252EBB80000u);
  EXPECT_EQ(info.column_width, 8u);
  EXPECT_EQ(info.start_column, 2u);

  ContextBlobInfoDeinitialize(&info);
  iree_allocator_free(iree_allocator_system(), private_data.data);
}

TEST(ContextBlobTest, DeviceHelperBuildsNegotiatedLegacyContext) {
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();
  KmtApi api = {};
  Device device = {};
  device.mcdm_abi = McdmAbi::legacy;
  Buffer context_buffer;
  iree_byte_span_t private_data = iree_byte_span_empty();
  ContextBlobInfo info;
  Error error;

  ASSERT_TRUE(BuildContextPrivateDataForDevice(
      api, device, xclbin.data(), xclbin.size(), /*process_id=*/1234,
      iree_allocator_system(), &private_data, &info, &context_buffer, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(context_buffer.allocation, 0u);
  EXPECT_EQ(private_data.data_length, 0xE8 + xclbin.size() + 0x370);
  EXPECT_EQ(std::memcmp(private_data.data, xclbin.data() + 0x1A0, 16), 0);

  ContextBlobInfoDeinitialize(&info);
  iree_allocator_free(iree_allocator_system(), private_data.data);
}

TEST(ContextBlobTest, DeviceHelperBuildsLegacyV0ContextWithXrt240Offsets) {
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();
  KmtApi api = {};
  Device device = {};
  device.mcdm_abi = McdmAbi::legacy_v0;
  Buffer context_buffer;
  iree_byte_span_t private_data = iree_byte_span_empty();
  ContextBlobInfo info;
  Error error;

  ASSERT_TRUE(BuildContextPrivateDataForDevice(
      api, device, xclbin.data(), xclbin.size(), /*process_id=*/0x7654,
      iree_allocator_system(), &private_data, &info, &context_buffer, &error))
      << ErrorMessage(&error);

  const size_t tail = 0xC0 + xclbin.size();
  ASSERT_EQ(private_data.data_length, tail + 0x530);
  EXPECT_EQ(context_buffer.allocation, 0u);
  EXPECT_EQ(std::memcmp(private_data.data, xclbin.data() + 0x1A0, 16), 0);
  EXPECT_EQ(ReadU64(private_data.data, 0x38), 0x04000000u);
  EXPECT_EQ(ReadU64(private_data.data, 0x40), 0x48u);
  EXPECT_EQ(ReadU64(private_data.data, 0x48), private_data.data_length - 0x58);
  EXPECT_EQ(ReadU64(private_data.data, 0x50), 0x7654u);
  EXPECT_EQ(ReadU64(private_data.data, 0x58), 1u);
  EXPECT_EQ(ReadU64(private_data.data, 0xA0), 0x1000u);
  EXPECT_EQ(ReadU64(private_data.data, 0xA8), xclbin.size());
  EXPECT_EQ(ReadU64(private_data.data, 0xB0), private_data.data_length - 0x110);
  EXPECT_EQ(ReadU64(private_data.data, 0xB8), private_data.data_length - 0xC0);
  EXPECT_EQ(std::memcmp(private_data.data + 0xC0, xclbin.data(),
                        xclbin.size()),
            0);
  EXPECT_EQ(ReadU64(private_data.data, tail + 0x40), 0x1000u);
  EXPECT_EQ(ReadU64(private_data.data, tail + 0x48), info.column_width);
  EXPECT_EQ(ReadU64(private_data.data, tail + 0x200), 0x10000u);
  EXPECT_EQ(ReadU64(private_data.data, tail + 0x208), 8u);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x218), 0x100u);

  ContextBlobInfoDeinitialize(&info);
  iree_allocator_free(iree_allocator_system(), private_data.data);
}

TEST(ContextBlobTest, DeviceHelperBuildsLegacyV2ContextWithXrt280Offsets) {
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();
  KmtApi api = {};
  Device device = {};
  device.mcdm_abi = McdmAbi::legacy_v2;
  Buffer context_buffer;
  iree_byte_span_t private_data = iree_byte_span_empty();
  ContextBlobInfo info;
  Error error;

  ASSERT_TRUE(BuildContextPrivateDataForDevice(
      api, device, xclbin.data(), xclbin.size(), /*process_id=*/0x1B24,
      iree_allocator_system(), &private_data, &info, &context_buffer, &error))
      << ErrorMessage(&error);

  const size_t tail = 0xE0 + xclbin.size();
  ASSERT_EQ(private_data.data_length, tail + 0x3C8);
  EXPECT_EQ(context_buffer.allocation, 0u);
  EXPECT_EQ(std::memcmp(private_data.data, xclbin.data() + 0x1A0, 16), 0);
  EXPECT_EQ(ReadU64(private_data.data, 0x38), 0u);
  EXPECT_EQ(ReadU64(private_data.data, 0x40), 0x04000000u);
  EXPECT_EQ(ReadU64(private_data.data, 0x48), 0x48u);
  EXPECT_EQ(ReadU64(private_data.data, 0x50), private_data.data_length - 0x78);
  EXPECT_EQ(ReadU64(private_data.data, 0x58), 0x1B24u);
  EXPECT_EQ(ReadU64(private_data.data, 0x78), 1u);
  EXPECT_EQ(ReadU64(private_data.data, 0xC0), 0x1000u);
  EXPECT_EQ(ReadU64(private_data.data, 0xC8), xclbin.size());
  EXPECT_EQ(ReadU64(private_data.data, 0xD0), private_data.data_length - 0x130);
  EXPECT_EQ(ReadU64(private_data.data, 0xD8), private_data.data_length - 0xE0);
  EXPECT_EQ(std::memcmp(private_data.data + 0xE0, xclbin.data(),
                        xclbin.size()),
            0);
  EXPECT_EQ(ReadU64(private_data.data, tail + 0x40), 0x10000u);
  EXPECT_EQ(ReadU64(private_data.data, tail + 0x48), 9u);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x3B8), 0x800u);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x3BC), 1u);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x3C0), info.column_width);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x3C4), info.start_column);

  ContextBlobInfoDeinitialize(&info);
  iree_allocator_free(iree_allocator_system(), private_data.data);
}

TEST(ContextBlobTest, DeviceHelperBuildsLegacyV2MlirAieContextWithXrt280Tail) {
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();
  constexpr size_t kAiePartitionOffset = 0x2C8;
  WriteString(&xclbin, kAiePartitionOffset + 0x268, 64, "MLIR_AIE");
  KmtApi api = {};
  Device device = {};
  device.mcdm_abi = McdmAbi::legacy_v2;
  Buffer context_buffer;
  iree_byte_span_t private_data = iree_byte_span_empty();
  ContextBlobInfo info;
  Error error;

  ASSERT_TRUE(BuildContextPrivateDataForDevice(
      api, device, xclbin.data(), xclbin.size(), /*process_id=*/0x38A0,
      iree_allocator_system(), &private_data, &info, &context_buffer, &error))
      << ErrorMessage(&error);

  const size_t tail = 0xE0 + xclbin.size();
  ASSERT_EQ(private_data.data_length, tail + 0x370);
  EXPECT_EQ(context_buffer.allocation, 0u);
  EXPECT_EQ(ReadU64(private_data.data, 0x50), private_data.data_length - 0x78);
  EXPECT_EQ(ReadU64(private_data.data, 0x58), 0x38A0u);
  EXPECT_EQ(ReadU64(private_data.data, 0xD0), private_data.data_length - 0x130);
  EXPECT_EQ(ReadU64(private_data.data, 0xD8), private_data.data_length - 0xE0);
  EXPECT_EQ(ReadU64(private_data.data, tail + 0x40), 0x10000u);
  EXPECT_EQ(ReadU64(private_data.data, tail + 0x48), 8u);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x58), 0x901u);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x360), 0x800u);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x364), 1u);
  EXPECT_EQ(ReadU32(private_data.data, tail + 0x368), info.column_width);

  ContextBlobInfoDeinitialize(&info);
  iree_allocator_free(iree_allocator_system(), private_data.data);
}

TEST(ContextBlobTest, RejectsExcessiveAxlfSectionCount) {
  constexpr uint32_t kMaxAxlfSections = 4096;
  constexpr size_t kSectionTableOffset = 0x1C8;
  std::vector<uint8_t> xclbin(kSectionTableOffset + (kMaxAxlfSections + 1) * 40,
                              0);
  WriteString(&xclbin, 0, 8, "xclbin2");
  WriteU32(&xclbin, 0x1C0, kMaxAxlfSections + 1);

  iree_byte_span_t private_data = iree_byte_span_empty();
  Error error;
  EXPECT_FALSE(BuildContextPrivateDataFromXclbin(
      xclbin.data(), xclbin.size(), /*process_id=*/1234,
      iree_allocator_system(), &private_data, /*out_info=*/nullptr, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "section count"), nullptr)
      << ErrorMessage(&error);
}

TEST(ContextBlobTest, RejectsExcessiveAiePartitionPdiCount) {
  constexpr uint32_t kMaxAiePartitionPdis = 4096;
  constexpr size_t kAiePartitionOffset = 0x2C8;
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();
  WriteU32(&xclbin, kAiePartitionOffset + 120, kMaxAiePartitionPdis + 1);

  iree_byte_span_t private_data = iree_byte_span_empty();
  Error error;
  EXPECT_FALSE(BuildContextPrivateDataFromXclbin(
      xclbin.data(), xclbin.size(), /*process_id=*/1234,
      iree_allocator_system(), &private_data, /*out_info=*/nullptr, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "PDI count"), nullptr)
      << ErrorMessage(&error);
}

TEST(ContextBlobTest, RejectsNullOutputBlob) {
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();

  ContextBlobInfo info;
  Error error;
  EXPECT_FALSE(BuildContextPrivateDataFromXclbin(
      xclbin.data(), xclbin.size(), /*process_id=*/1234,
      iree_allocator_system(), /*out_blob=*/nullptr, &info, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "invalid"), nullptr)
      << ErrorMessage(&error);
}

TEST(ContextBlobTest, RejectsOversizedContextBlobBeforeAllocation) {
  std::vector<uint8_t> xclbin = BuildSyntheticMultiPdiXclbin();

  iree_byte_span_t private_data = iree_byte_span_empty();
  Error error;
  EXPECT_FALSE(BuildContextPrivateDataFromXclbin(
      xclbin.data(), 512ull * 1024ull * 1024ull, /*process_id=*/1234,
      iree_allocator_system(), &private_data, /*out_info=*/nullptr, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "size exceeds"), nullptr)
      << ErrorMessage(&error);
  EXPECT_EQ(private_data.data, nullptr);
}

}  // namespace
}  // namespace iree::hal::amdxdna::mcdm
