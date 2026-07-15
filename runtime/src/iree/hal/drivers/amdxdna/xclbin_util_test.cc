// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/xclbin_util.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// AXLF / AIE_PARTITION layout constants mirrored from xclbin_util.cc.
constexpr uint32_t kAiePartitionSection = 32;
constexpr size_t kAxlfSectionCountOffset = 0x1C0;
constexpr size_t kAxlfSectionTableOffset = 0x1C8;
constexpr size_t kAxlfSectionRecordSize = 40;
constexpr size_t kAiePartitionHeaderSize = 0xC8;
constexpr size_t kAiePartitionPdiArraySizeOffset = 120;
constexpr size_t kAiePartitionPdiArrayOffsetOffset = 124;
constexpr size_t kAiePdiRecordSize = 0x60;
constexpr size_t kAiePdiImageSizeOffset = 16;
constexpr size_t kAiePdiImageOffsetOffset = 20;

void WriteU32(std::vector<uint8_t>& v, size_t off, uint32_t val) {
  std::memcpy(v.data() + off, &val, sizeof(val));
}
void WriteU64(std::vector<uint8_t>& v, size_t off, uint64_t val) {
  std::memcpy(v.data() + off, &val, sizeof(val));
}

// Builds an AIE_PARTITION blob: a 0xC8-byte header (pdi count + table offset),
// a contiguous PDI record table, then the PDI image bytes. PDI record offsets
// are relative to the start of the partition.
std::vector<uint8_t> BuildAiePartition(
    const std::vector<std::vector<uint8_t>>& pdis) {
  const size_t table_off = kAiePartitionHeaderSize;
  const size_t images_off = table_off + pdis.size() * kAiePdiRecordSize;
  std::vector<uint8_t> part(images_off, 0);
  WriteU32(part, kAiePartitionPdiArraySizeOffset,
           static_cast<uint32_t>(pdis.size()));
  WriteU32(part, kAiePartitionPdiArrayOffsetOffset,
           static_cast<uint32_t>(table_off));
  size_t cur = images_off;
  for (size_t i = 0; i < pdis.size(); ++i) {
    const size_t rec = table_off + i * kAiePdiRecordSize;
    WriteU32(part, rec + kAiePdiImageSizeOffset,
             static_cast<uint32_t>(pdis[i].size()));
    WriteU32(part, rec + kAiePdiImageOffsetOffset, static_cast<uint32_t>(cur));
    part.insert(part.end(), pdis[i].begin(), pdis[i].end());
    cur += pdis[i].size();
  }
  return part;
}

// Wraps an AIE_PARTITION blob in a single-section AXLF/xclbin2 container.
std::vector<uint8_t> BuildXclbin(const std::vector<std::vector<uint8_t>>& pdis,
                                 uint32_t section_kind = kAiePartitionSection) {
  std::vector<uint8_t> part = BuildAiePartition(pdis);
  const size_t part_off = kAxlfSectionTableOffset + kAxlfSectionRecordSize;
  std::vector<uint8_t> axlf(part_off, 0);
  std::memcpy(axlf.data(), "xclbin2\0", 8);
  WriteU32(axlf, kAxlfSectionCountOffset, 1u);
  WriteU32(axlf, kAxlfSectionTableOffset + 0, section_kind);
  WriteU64(axlf, kAxlfSectionTableOffset + 24, part_off);
  WriteU64(axlf, kAxlfSectionTableOffset + 32, part.size());
  axlf.insert(axlf.end(), part.begin(), part.end());
  return axlf;
}

iree_const_byte_span_t Span(const std::vector<uint8_t>& v) {
  return iree_make_const_byte_span(v.data(), v.size());
}

iree_status_t ExtractPdi(const std::vector<uint8_t>& xclbin, uint32_t pdi_index,
                         std::vector<uint8_t>* out_pdi) {
  iree_byte_span_t pdi_span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_xclbin_extract_pdi(
      Span(xclbin), pdi_index, iree_allocator_system(), &pdi_span));
  out_pdi->assign(pdi_span.data, pdi_span.data + pdi_span.data_length);
  iree_allocator_free(iree_allocator_system(), pdi_span.data);
  return iree_ok_status();
}

TEST(XclbinUtilTest, ExtractSinglePdi) {
  std::vector<uint8_t> pdi(64, 0xAB);
  auto xclbin = BuildXclbin({pdi});
  std::vector<uint8_t> out;
  IREE_ASSERT_OK(ExtractPdi(xclbin, 0, &out));
  EXPECT_EQ(out, pdi);
}

TEST(XclbinUtilTest, ExtractMultiplePdisByIndex) {
  std::vector<uint8_t> pdi0(16, 0x10);
  std::vector<uint8_t> pdi1(48, 0x21);
  std::vector<uint8_t> pdi2(100, 0x32);
  auto xclbin = BuildXclbin({pdi0, pdi1, pdi2});
  std::vector<uint8_t> out;
  IREE_ASSERT_OK(ExtractPdi(xclbin, 2, &out));
  EXPECT_EQ(out, pdi2);
  IREE_ASSERT_OK(ExtractPdi(xclbin, 0, &out));
  EXPECT_EQ(out, pdi0);
  IREE_ASSERT_OK(ExtractPdi(xclbin, 1, &out));
  EXPECT_EQ(out, pdi1);
}

TEST(XclbinUtilTest, PdiIndexOutOfRange) {
  auto xclbin = BuildXclbin({std::vector<uint8_t>(8, 0x01)});
  std::vector<uint8_t> out;
  iree_status_t status = ExtractPdi(xclbin, 5, &out);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_ignore(status);
}

TEST(XclbinUtilTest, RejectsNonXclbinMagic) {
  std::vector<uint8_t> not_xclbin(0x200, 0x00);
  std::memcpy(not_xclbin.data(), "NOTXCLBN", 8);
  std::vector<uint8_t> out;
  iree_status_t status = ExtractPdi(not_xclbin, 0, &out);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(XclbinUtilTest, RejectsTooSmallInput) {
  std::vector<uint8_t> tiny(8, 0x00);
  std::memcpy(tiny.data(), "xclbin2\0", 8);
  std::vector<uint8_t> out;
  iree_status_t status = ExtractPdi(tiny, 0, &out);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(XclbinUtilTest, RejectsMissingAiePartitionSection) {
  // Valid AXLF with a single section of a non-AIE_PARTITION kind.
  auto xclbin = BuildXclbin({std::vector<uint8_t>(8, 0x01)}, /*kind=*/7u);
  std::vector<uint8_t> out;
  iree_status_t status = ExtractPdi(xclbin, 0, &out);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(XclbinUtilTest, RejectsEmptyPdiImage) {
  auto xclbin = BuildXclbin({std::vector<uint8_t>()});  // zero-length PDI
  std::vector<uint8_t> out;
  iree_status_t status = ExtractPdi(xclbin, 0, &out);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(XclbinUtilTest, RejectsOutOfBoundsPdiOffset) {
  std::vector<uint8_t> pdi(32, 0x55);
  auto xclbin = BuildXclbin({pdi});
  // Corrupt the PDI image offset (in the partition record) to point past the
  // end of the partition. Partition starts at section table + one record.
  const size_t part_off = kAxlfSectionTableOffset + kAxlfSectionRecordSize;
  const size_t rec = part_off + kAiePartitionHeaderSize;  // first PDI record
  WriteU32(xclbin, rec + kAiePdiImageOffsetOffset, 0x7FFFFFFFu);
  std::vector<uint8_t> out;
  iree_status_t status = ExtractPdi(xclbin, 0, &out);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

}  // namespace
