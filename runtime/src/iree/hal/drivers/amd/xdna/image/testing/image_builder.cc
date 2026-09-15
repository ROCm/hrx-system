// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/testing/image_builder.h"

#include <algorithm>
#include <utility>

namespace iree::hal::amd::xdna::testing {

iree_hal_amd_xdna_image_program_header_t MakeProgramHeader(uint32_t type,
                                                           uint32_t file_offset,
                                                           uint32_t file_size) {
  return {
      /*.type=*/type,
      /*.file_range=*/
      {
          /*.offset=*/file_offset,
          /*.length=*/file_size,
      },
      /*.virtual_address=*/0,
      /*.physical_address=*/0,
      /*.memory_size=*/file_size,
      /*.flags=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ,
      /*.alignment=*/4,
  };
}

ImageBuilder::ImageBuilder(ImageBuilderOptions options) : options_(options) {}

ImageBuilder& ImageBuilder::AddProgram(
    iree_hal_amd_xdna_image_program_header_t program_header) {
  programs_.push_back({
      /*.header=*/program_header,
      /*.payload=*/{},
      /*.has_payload=*/false,
  });
  return *this;
}

ImageBuilder& ImageBuilder::AddProgram(
    iree_hal_amd_xdna_image_program_header_t program_header,
    std::vector<uint8_t> payload) {
  IREE_ASSERT(payload.size() == program_header.file_range.length);
  programs_.push_back({
      /*.header=*/program_header,
      /*.payload=*/std::move(payload),
      /*.has_payload=*/true,
  });
  return *this;
}

static void StoreProgramHeader(
    const iree_hal_amd_xdna_image_program_header_t& program_header,
    uint8_t* storage) {
  iree_unaligned_store_le_u32(storage + 0, program_header.type);
  iree_unaligned_store_le_u32(storage + 4,
                              (uint32_t)program_header.file_range.offset);
  iree_unaligned_store_le_u32(storage + 8, program_header.virtual_address);
  iree_unaligned_store_le_u32(storage + 12, program_header.physical_address);
  iree_unaligned_store_le_u32(storage + 16,
                              (uint32_t)program_header.file_range.length);
  iree_unaligned_store_le_u32(storage + 20, program_header.memory_size);
  iree_unaligned_store_le_u32(storage + 24, program_header.flags);
  iree_unaligned_store_le_u32(storage + 28, program_header.alignment);
}

std::vector<uint8_t> ImageBuilder::Build() const {
  IREE_ASSERT(!programs_.empty());
  IREE_ASSERT(programs_.size() <= UINT16_MAX);
  iree_host_size_t source_length = std::max(
      options_.minimum_source_length,
      static_cast<iree_host_size_t>(
          IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE +
          programs_.size() * IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE));
  for (const Program& program : programs_) {
    source_length = std::max(
        source_length,
        static_cast<iree_host_size_t>(program.header.file_range.offset +
                                      program.header.file_range.length));
  }
  if (options_.section_header_count != 0) {
    source_length = std::max(
        source_length, static_cast<iree_host_size_t>(
                           options_.section_header_offset +
                           options_.section_header_count *
                               IREE_HAL_AMD_XDNA_ELF_SECTION_HEADER_SIZE));
  }
  std::vector<uint8_t> bytes(source_length, 0);

  bytes[0] = 0x7F;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = IREE_HAL_AMD_XDNA_ELF_CLASS_32;
  bytes[5] = IREE_HAL_AMD_XDNA_ELF_DATA_LITTLE_ENDIAN;
  bytes[6] = IREE_HAL_AMD_XDNA_ELF_VERSION_CURRENT;
  bytes[7] = IREE_HAL_AMD_XDNA_ELF_OS_ABI_NONE;
  bytes[8] = IREE_HAL_AMD_XDNA_ELF_ABI_VERSION_NONE;
  iree_unaligned_store_le_u16(bytes.data() + 16,
                              IREE_HAL_AMD_XDNA_ELF_FILE_TYPE_EXEC);
  iree_unaligned_store_le_u16(bytes.data() + 18,
                              IREE_HAL_AMD_XDNA_ELF_MACHINE_AIE);
  iree_unaligned_store_le_u32(bytes.data() + 20,
                              IREE_HAL_AMD_XDNA_ELF_VERSION_CURRENT);
  iree_unaligned_store_le_u32(bytes.data() + 24, 0);
  iree_unaligned_store_le_u32(bytes.data() + 28,
                              IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE);
  iree_unaligned_store_le_u32(bytes.data() + 32,
                              options_.section_header_offset);
  iree_unaligned_store_le_u32(bytes.data() + 36, options_.target_flags);
  iree_unaligned_store_le_u16(bytes.data() + 40,
                              IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE);
  iree_unaligned_store_le_u16(bytes.data() + 42,
                              IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE);
  iree_unaligned_store_le_u16(bytes.data() + 44,
                              static_cast<uint16_t>(programs_.size()));
  iree_unaligned_store_le_u16(bytes.data() + 46,
                              options_.section_header_count == 0
                                  ? 0
                                  : IREE_HAL_AMD_XDNA_ELF_SECTION_HEADER_SIZE);
  iree_unaligned_store_le_u16(bytes.data() + 48, options_.section_header_count);
  iree_unaligned_store_le_u16(bytes.data() + 50, 0);

  for (iree_host_size_t i = 0; i < programs_.size(); ++i) {
    const Program& program = programs_[i];
    StoreProgramHeader(program.header,
                       bytes.data() + IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE +
                           i * IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE);
    auto payload_begin = bytes.begin() + program.header.file_range.offset;
    if (program.has_payload) {
      std::copy(program.payload.begin(), program.payload.end(), payload_begin);
    } else {
      std::fill(payload_begin, payload_begin + program.header.file_range.length,
                static_cast<uint8_t>(0xA0 + i));
    }
  }
  return bytes;
}

}  // namespace iree::hal::amd::xdna::testing
