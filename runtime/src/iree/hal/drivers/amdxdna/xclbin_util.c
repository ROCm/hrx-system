// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/xclbin_util.h"

#include <string.h>

static const uint32_t kAiePartitionSection = 32;
static const size_t kAxlfSectionCountOffset = 0x1C0;
static const size_t kAxlfSectionTableOffset = 0x1C8;
static const size_t kAxlfSectionRecordSize = 40;
static const size_t kAxlfSectionKindOffset = 0;
static const size_t kAxlfSectionOffsetOffset = 24;
static const size_t kAxlfSectionSizeOffset = 32;

static const size_t kAiePartitionMinSize = 0xC8;
static const size_t kAiePartitionPdiArraySizeOffset = 120;
static const size_t kAiePartitionPdiArrayOffsetOffset = 124;
static const size_t kAiePdiRecordSize = 0x60;
static const size_t kAiePdiImageSizeOffset = 16;
static const size_t kAiePdiImageOffsetOffset = 20;

static uint32_t ReadU32(const uint8_t* data, size_t offset) {
  uint32_t value = 0;
  memcpy(&value, data + offset, sizeof(value));
  return value;
}

static uint64_t ReadU64(const uint8_t* data, size_t offset) {
  uint64_t value = 0;
  memcpy(&value, data + offset, sizeof(value));
  return value;
}

static bool IsRangeInBounds(size_t data_size, uint64_t offset, uint64_t size) {
  return offset <= data_size && size <= data_size - offset;
}

static iree_status_t CheckRange(size_t data_size, uint64_t offset,
                                uint64_t size, const char* what) {
  if (IREE_UNLIKELY(!IsRangeInBounds(data_size, offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "xclbin %s is out of bounds", what);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_xclbin_extract_pdi(
    iree_const_byte_span_t xclbin, uint32_t pdi_index,
    iree_allocator_t host_allocator, iree_byte_span_t* out_pdi) {
  IREE_ASSERT_ARGUMENT(out_pdi);
  *out_pdi = iree_byte_span_empty();

  if (IREE_UNLIKELY(!xclbin.data || xclbin.data_length < 0x1C8 ||
                    memcmp(xclbin.data, "xclbin2\0", 8) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "input is not an AXLF/xclbin2 file");
  }

  const uint8_t* data = xclbin.data;
  const size_t data_size = xclbin.data_length;
  const uint32_t section_count = ReadU32(data, kAxlfSectionCountOffset);
  IREE_RETURN_IF_ERROR(CheckRange(
      data_size, kAxlfSectionTableOffset,
      (uint64_t)kAxlfSectionRecordSize * section_count, "section table"));

  const uint8_t* aie_partition = NULL;
  size_t aie_partition_size = 0;
  for (uint32_t i = 0; i < section_count; ++i) {
    const size_t record =
        kAxlfSectionTableOffset + (size_t)i * kAxlfSectionRecordSize;
    const uint32_t kind = ReadU32(data, record + kAxlfSectionKindOffset);
    const uint64_t offset = ReadU64(data, record + kAxlfSectionOffsetOffset);
    const uint64_t size = ReadU64(data, record + kAxlfSectionSizeOffset);
    IREE_RETURN_IF_ERROR(CheckRange(data_size, offset, size, "section"));
    if (kind == kAiePartitionSection) {
      aie_partition = data + offset;
      aie_partition_size = (size_t)size;
      break;
    }
  }
  if (IREE_UNLIKELY(!aie_partition)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "xclbin has no AIE_PARTITION section");
  }
  if (IREE_UNLIKELY(aie_partition_size < kAiePartitionMinSize)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "xclbin AIE_PARTITION section is too small");
  }

  const uint32_t pdi_count =
      ReadU32(aie_partition, kAiePartitionPdiArraySizeOffset);
  const uint32_t pdi_table_offset =
      ReadU32(aie_partition, kAiePartitionPdiArrayOffsetOffset);
  if (IREE_UNLIKELY(pdi_index >= pdi_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "xclbin AIE_PARTITION PDI index %u out of range; contains %u PDIs",
        pdi_index, pdi_count);
  }
  IREE_RETURN_IF_ERROR(CheckRange(aie_partition_size, pdi_table_offset,
                                  (uint64_t)kAiePdiRecordSize * pdi_count,
                                  "AIE_PARTITION PDI table"));

  const size_t pdi_record =
      pdi_table_offset + (size_t)pdi_index * kAiePdiRecordSize;
  const uint32_t pdi_size =
      ReadU32(aie_partition, pdi_record + kAiePdiImageSizeOffset);
  const uint32_t pdi_offset =
      ReadU32(aie_partition, pdi_record + kAiePdiImageOffsetOffset);
  if (IREE_UNLIKELY(pdi_size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "xclbin AIE_PARTITION PDI image is empty");
  }
  IREE_RETURN_IF_ERROR(CheckRange(aie_partition_size, pdi_offset, pdi_size,
                                  "AIE_PARTITION PDI image"));

  uint8_t* pdi_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, pdi_size, (void**)&pdi_data));
  memcpy(pdi_data, aie_partition + pdi_offset, pdi_size);
  *out_pdi = iree_make_byte_span(pdi_data, pdi_size);
  return iree_ok_status();
}
