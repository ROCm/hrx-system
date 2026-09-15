// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"

#include <inttypes.h>

enum {
  IREE_HAL_AMD_XDNA_AIE2P_NPU2_COLUMN_COUNT = 8,
  IREE_HAL_AMD_XDNA_AIE2P_NPU2_ROW_COUNT = 6,
  IREE_HAL_AMD_XDNA_AIE2P_NPU2_SHIM_ROW = 0,
  IREE_HAL_AMD_XDNA_AIE2P_NPU2_MEMORY_ROW = 1,
  IREE_HAL_AMD_XDNA_AIE2P_NPU2_FIRST_COMPUTE_ROW = 2,
};

typedef uint8_t iree_hal_amd_xdna_aie2p_npu2_tile_kinds_t;
enum iree_hal_amd_xdna_aie2p_npu2_tile_kind_bits_e {
  IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM = 1u << 0,
  IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY = 1u << 1,
  IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE = 1u << 2,
};

// One regular family of writable NPU2 configuration registers.
typedef struct iree_hal_amd_xdna_aie2p_npu2_register_pattern_t {
  // Tile-relative address of the first register word.
  uint32_t base_offset;
  // Register bits represented by the public NPU2 register corpus.
  uint32_t writable_mask;
  // Number of elements in the first address dimension.
  uint16_t first_count;
  // Number of elements in the second address dimension.
  uint16_t second_count;
  // Byte stride of the first address dimension.
  uint16_t first_stride;
  // Byte stride of the second address dimension.
  uint16_t second_stride;
  // Tile kinds implementing this register pattern.
  iree_hal_amd_xdna_aie2p_npu2_tile_kinds_t tile_kinds;
} iree_hal_amd_xdna_aie2p_npu2_register_pattern_t;

// One tile-relative data-memory window and its physical owner.
typedef struct iree_hal_amd_xdna_aie2p_npu2_address_window_t {
  // Tile-relative base address of the window.
  uint32_t base;
  // Addressable byte capacity of the window.
  uint32_t capacity;
  // Signed column displacement from the referencing tile to the owner.
  int8_t owner_column_delta;
  // Signed row displacement from the referencing tile to the owner.
  int8_t owner_row_delta;
  // Required tile kind of the resolved owner.
  iree_hal_amd_xdna_aie2p_npu2_tile_kinds_t owner_kind;
} iree_hal_amd_xdna_aie2p_npu2_address_window_t;

#define IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(kind, base, mask, count, stride) \
  {base, mask, count, 1, stride, 0, kind}

#define IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN_2D(                              \
    kind, base, mask, first_count, first_stride, second_count, second_stride) \
  {base, mask, first_count, second_count, first_stride, second_stride, kind}

// Generated from the NPU2 register database identity
// AM025-2024-11-13-1.1 and independently cross-checked against AIE-RT commit
// 8849e208bdcc533b20a0ed3f95c1ce961dee9c3a. Patterns describe addressable
// words and their writable bits; read-only status registers are absent.
static const iree_hal_amd_xdna_aie2p_npu2_register_pattern_t
    iree_hal_amd_xdna_aie2p_npu2_register_patterns[] = {
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1D000, 0x0FFFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1D004, 0xFFFF0000,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1D008, 0x03FFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1D00C, 0x1FFFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1D010, 0x01FFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1D014, 0xFFFDFFEF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0000, 0xFFFFFFFF,
            48, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0004, 0xFFFFFFFF,
            48, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0008, 0x87FFFFFF,
            48, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA000C, 0xFFFFFFFF,
            48, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0010, 0xFFFFFFFF,
            48, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0014, 0xFFFFFFFF,
            48, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0018, 0x1FFFFFFF,
            48, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA001C, 0xFFFFFFFF,
            48, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D000, 0xFFFFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D004, 0xFFFFFFFC,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D008, 0x7FFFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D00C, 0x7FFFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D010, 0xFFFFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D014, 0xFFFFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D018, 0xFFFFFFFF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D01C, 0xFFFDFFEF,
            16, 0x20),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1DE00, 0x0003FF1A,
            2, 0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1DE04, 0x80FF000F,
            2, 0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1DE10, 0x0000FF12,
            2, 0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1DE14, 0x80FF000F,
            2, 0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0600, 0x0003FF1A,
            6, 0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0604, 0x80FF003F,
            6, 0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0630, 0x0000FF12,
            6, 0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xA0634, 0x80FF003F,
            6, 0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D200, 0x0003FF0E, 2,
            0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D204, 0x80FF000F, 2,
            0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D210, 0x0000FF06, 2,
            0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1D214, 0x80FF000F, 2,
            0x08),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x3F000, 0xC00000FF,
            23, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x3F100, 0xC0000000,
            25, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN_2D(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x3F200, 0x1F1F0137,
            25, 0x10, 4, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xB0000, 0xC00000FF,
            17, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xB0100, 0xC0000000,
            18, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN_2D(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xB0200, 0x1F1F0137,
            18, 0x10, 4, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x3F000, 0xC00000FF,
            22, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x3F100, 0xC0000000,
            23, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN_2D(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x3F200, 0x1F1F0137,
            23, 0x10, 4, 0x04),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x1F000, 0x0000003F,
            16, 0x10),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY, 0xC0000, 0x0000003F,
            64, 0x10),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x14000, 0x0000003F,
            16, 0x10),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE, 0x32000, 0x00000003,
            1, 0),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1F000, 0x0000FF00, 1,
            0),
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN(
            IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM, 0x1F004, 0x00000FF0, 1,
            0),
};

#undef IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN_2D
#undef IREE_HAL_AMD_XDNA_AIE2P_NPU2_PATTERN

static iree_hal_amd_xdna_aie2p_npu2_tile_kinds_t
iree_hal_amd_xdna_aie2p_npu2_tile_kind(uint32_t row) {
  if (row == IREE_HAL_AMD_XDNA_AIE2P_NPU2_SHIM_ROW) {
    return IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_SHIM;
  } else if (row == IREE_HAL_AMD_XDNA_AIE2P_NPU2_MEMORY_ROW) {
    return IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY;
  }
  return IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE;
}

static bool iree_hal_amd_xdna_aie2p_npu2_query_register_mask(
    uint32_t row, uint32_t register_offset, uint32_t* out_writable_mask) {
  const iree_hal_amd_xdna_aie2p_npu2_tile_kinds_t tile_kind =
      iree_hal_amd_xdna_aie2p_npu2_tile_kind(row);
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amd_xdna_aie2p_npu2_register_patterns);
       ++i) {
    const iree_hal_amd_xdna_aie2p_npu2_register_pattern_t* pattern =
        &iree_hal_amd_xdna_aie2p_npu2_register_patterns[i];
    if ((pattern->tile_kinds & tile_kind) == 0) continue;
    for (uint16_t first = 0; first < pattern->first_count; ++first) {
      for (uint16_t second = 0; second < pattern->second_count; ++second) {
        const uint32_t pattern_offset = pattern->base_offset +
                                        first * pattern->first_stride +
                                        second * pattern->second_stride;
        if (register_offset == pattern_offset) {
          *out_writable_mask = pattern->writable_mask;
          return true;
        }
      }
    }
  }
  *out_writable_mask = 0;
  return false;
}

static iree_status_t iree_hal_amd_xdna_aie2p_npu2_query_register_word(
    uint32_t address, uint32_t* out_writable_mask) {
  const uint32_t row = (address >> 20) & 0x1Fu;
  const uint32_t register_offset = address & 0xFFFFFu;
  if (!iree_hal_amd_xdna_aie2p_npu2_query_register_mask(row, register_offset,
                                                        out_writable_mask)) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "AIE2P register 0x%08" PRIx32
                            " is not writable on an NPU2 tile",
                            address);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_npu2_validate_register_value(
    uint32_t address, uint32_t value) {
  uint32_t writable_mask = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_npu2_query_register_word(
      address, &writable_mask));
  if ((value & ~writable_mask) != 0) {
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "AIE2P register 0x%08" PRIx32 " writes reserved NPU2 bits", address);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_npu2_validate_register_mask(
    uint32_t address, uint32_t mask) {
  uint32_t writable_mask = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_npu2_query_register_word(
      address, &writable_mask));
  if ((mask & ~writable_mask) != 0) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "AIE2P masked write to register 0x%08" PRIx32
                            " selects reserved NPU2 bits",
                            address);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_npu2_validate_register(
    const void* user_data, iree_hal_amd_xdna_elf_program_type_t program_type,
    const iree_hal_amd_xdna_aie2p_program_record_t* record) {
  (void)user_data;
  (void)program_type;
  switch (record->type) {
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      return iree_hal_amd_xdna_aie2p_npu2_validate_register_value(
          record->value.register_write32.address,
          record->value.register_write32.value);
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      return iree_hal_amd_xdna_aie2p_npu2_validate_register_mask(
          record->value.register_mask_write32.address,
          record->value.register_mask_write32.mask);
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32:
      for (uint32_t i = 0; i < record->value.register_block_write32.word_count;
           ++i) {
        uint32_t word = 0;
        IREE_RETURN_IF_ERROR(
            iree_hal_amd_xdna_aie2p_program_record_read_block_word(record, i,
                                                                   &word));
        IREE_RETURN_IF_ERROR(
            iree_hal_amd_xdna_aie2p_npu2_validate_register_value(
                record->value.register_block_write32.address +
                    i * sizeof(uint32_t),
                word));
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "record is not an AIE2P register operation");
  }
}

static bool iree_hal_amd_xdna_aie2p_npu2_range_contains(uint32_t base,
                                                        uint32_t capacity,
                                                        uint32_t address,
                                                        uint32_t byte_length,
                                                        uint32_t* out_offset) {
  const uint64_t end = (uint64_t)address + byte_length;
  const uint64_t range_end = (uint64_t)base + capacity;
  if (address < base || end > range_end) return false;
  *out_offset = address - base;
  return true;
}

static iree_status_t iree_hal_amd_xdna_aie2p_npu2_resolve_tile_memory(
    const void* user_data,
    const iree_hal_amd_xdna_elf_tile_destination_t* destination,
    uint32_t virtual_address, uint32_t byte_length,
    iree_hal_amd_xdna_image_tile_placement_t* out_placement) {
  (void)user_data;
  *out_placement = (iree_hal_amd_xdna_image_tile_placement_t){0};
  if (destination->column >= IREE_HAL_AMD_XDNA_AIE2P_NPU2_COLUMN_COUNT ||
      destination->row >= IREE_HAL_AMD_XDNA_AIE2P_NPU2_ROW_COUNT ||
      byte_length == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P TILE placement exceeds the NPU2 array");
  }

  if (destination->memory_space ==
      IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM) {
    const uint32_t program_capacity = 16 * 1024;
    if (destination->row < IREE_HAL_AMD_XDNA_AIE2P_NPU2_FIRST_COMPUTE_ROW ||
        virtual_address != 0 || byte_length > program_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AIE2P TILE program does not fit NPU2 program memory");
    }
    *out_placement = (iree_hal_amd_xdna_image_tile_placement_t){
        .owner_column = destination->column,
        .owner_row = destination->row,
        .memory_space = destination->memory_space,
        .owner_offset = 0,
        .byte_length = byte_length,
        .available_capacity = program_capacity,
    };
    return iree_ok_status();
  }
  if (destination->memory_space !=
      IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_DATA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown AIE2P TILE memory space");
  }

  iree_hal_amd_xdna_aie2p_npu2_address_window_t windows[4] = {0};
  iree_host_size_t window_count = 0;
  if (destination->row == IREE_HAL_AMD_XDNA_AIE2P_NPU2_MEMORY_ROW) {
    windows[0] = (iree_hal_amd_xdna_aie2p_npu2_address_window_t){
        0x00000, 512 * 1024, -1, 0,
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY};
    windows[1] = (iree_hal_amd_xdna_aie2p_npu2_address_window_t){
        0x80000, 512 * 1024, 0, 0,
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY};
    windows[2] = (iree_hal_amd_xdna_aie2p_npu2_address_window_t){
        0x100000, 512 * 1024, 1, 0,
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_MEMORY};
    window_count = 3;
  } else if (destination->row >=
             IREE_HAL_AMD_XDNA_AIE2P_NPU2_FIRST_COMPUTE_ROW) {
    windows[0] = (iree_hal_amd_xdna_aie2p_npu2_address_window_t){
        0x40000, 64 * 1024, 0, -1,
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE};
    windows[1] = (iree_hal_amd_xdna_aie2p_npu2_address_window_t){
        0x50000, 64 * 1024, -1, 0,
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE};
    windows[2] = (iree_hal_amd_xdna_aie2p_npu2_address_window_t){
        0x60000, 64 * 1024, 0, 1,
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE};
    windows[3] = (iree_hal_amd_xdna_aie2p_npu2_address_window_t){
        0x70000, 64 * 1024, 0, 0,
        IREE_HAL_AMD_XDNA_AIE2P_NPU2_TILE_KIND_COMPUTE};
    window_count = 4;
  }
  for (iree_host_size_t i = 0; i < window_count; ++i) {
    uint32_t owner_offset = 0;
    if (!iree_hal_amd_xdna_aie2p_npu2_range_contains(
            windows[i].base, windows[i].capacity, virtual_address, byte_length,
            &owner_offset)) {
      continue;
    }
    const int32_t owner_column =
        (int32_t)destination->column + windows[i].owner_column_delta;
    const int32_t owner_row =
        (int32_t)destination->row + windows[i].owner_row_delta;
    if (owner_column < 0 ||
        owner_column >= IREE_HAL_AMD_XDNA_AIE2P_NPU2_COLUMN_COUNT ||
        owner_row < 0 || owner_row >= IREE_HAL_AMD_XDNA_AIE2P_NPU2_ROW_COUNT ||
        iree_hal_amd_xdna_aie2p_npu2_tile_kind((uint32_t)owner_row) !=
            windows[i].owner_kind) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AIE2P TILE data window has no compatible NPU2 owner");
    }
    *out_placement = (iree_hal_amd_xdna_image_tile_placement_t){
        .owner_column = (uint16_t)owner_column,
        .owner_row = (uint16_t)owner_row,
        .memory_space = destination->memory_space,
        .owner_offset = owner_offset,
        .byte_length = byte_length,
        .available_capacity = windows[i].capacity - owner_offset,
    };
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "AIE2P TILE data does not fit an NPU2 load window");
}

static iree_status_t iree_hal_amd_xdna_aie2p_npu2_validate_dma_task_wait(
    const void* user_data,
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* wait) {
  (void)user_data;
  for (uint16_t row = wait->row; row < (uint16_t)wait->row + wait->row_count;
       ++row) {
    const uint8_t channel_count =
        row == IREE_HAL_AMD_XDNA_AIE2P_NPU2_MEMORY_ROW ? 6 : 2;
    if (wait->dma_channel >= channel_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AIE2P DMA task wait channel is unavailable on an NPU2 tile");
    }
  }
  return iree_ok_status();
}

// Canonical compiler deployment identities sharing the NPU2 native contract.
static const struct {
  // Exact passive endpoint key, independent of native platform transport.
  iree_string_view_t target_id;
  // Compiler-owned identity serialized in the image ABI note.
  uint64_t device_profile_id;
} iree_hal_amd_xdna_aie2p_npu2_profiles[] = {
    {IREE_SVL("amd.xdna.strix.17f0_10"), UINT64_C(0x5354524958000001)},
    {IREE_SVL("amd.xdna.strix_halo.17f0_11"), UINT64_C(0x535848414C4F0001)},
};

iree_status_t iree_hal_amd_xdna_aie2p_npu2_target_initialize(
    iree_string_view_t target_id, uint16_t context_column_count,
    iree_hal_amd_xdna_aie2p_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  uint64_t device_profile_id = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amd_xdna_aie2p_npu2_profiles); ++i) {
    if (iree_string_view_equal(
            target_id, iree_hal_amd_xdna_aie2p_npu2_profiles[i].target_id)) {
      device_profile_id =
          iree_hal_amd_xdna_aie2p_npu2_profiles[i].device_profile_id;
      break;
    }
  }
  if (device_profile_id == 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no NPU2 image target for '%.*s'",
                            (int)target_id.size, target_id.data);
  }
  if (context_column_count == 0 ||
      context_column_count > IREE_HAL_AMD_XDNA_AIE2P_NPU2_COLUMN_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "NPU2 XDNA context column count must be in [1, 8]");
  }
  *out_target = (iree_hal_amd_xdna_aie2p_target_t){
      .identity =
          {
              .device_profile_revision = 1,
              .device_profile_id = device_profile_id,
              .firmware_abi_id = UINT64_C(0x4E5055320006000C),
              .policy_id = UINT64_C(0x413250504C414E01),
          },
      .supported_capabilities = IREE_HAL_AMD_XDNA_ELF_KNOWN_CAPABILITIES,
      .context =
          {
              .column_count = context_column_count,
              .row_count = IREE_HAL_AMD_XDNA_AIE2P_NPU2_ROW_COUNT,
          },
      .native =
          {
              .transaction =
                  {
                      .device_generation = 4,
                      .memory_tile_row_count = 1,
                  },
              .register_address =
                  {
                      .column_shift = 25,
                      .row_shift = 20,
                  },
              .program_memory =
                  {
                      .host_offset = UINT32_C(0x00020000),
                  },
          },
      .tile_memory_resolver =
          {
              .fn = iree_hal_amd_xdna_aie2p_npu2_resolve_tile_memory,
          },
      .configuration_register_validator =
          {
              .fn = iree_hal_amd_xdna_aie2p_npu2_validate_register,
          },
      .dma_task_wait_validator =
          {
              .fn = iree_hal_amd_xdna_aie2p_npu2_validate_dma_task_wait,
          },
  };
  return iree_ok_status();
}
