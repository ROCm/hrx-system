// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/program_format.h"

#include <inttypes.h>
#include <string.h>

static iree_status_t iree_hal_amd_xdna_aie2p_require_mutable_storage(
    iree_byte_span_t storage, iree_host_size_t required_size) {
  if (storage.data == NULL || storage.data_length != required_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P program record storage must contain exactly %" PRIhsz " bytes",
        required_size);
  }
  memset(storage.data, 0, storage.data_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_validate_register_range(
    uint32_t address, iree_host_size_t word_count) {
  if (word_count == 0 || (address % sizeof(uint32_t)) != 0 ||
      word_count - 1 > (UINT32_MAX - address) / sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P register word range is invalid");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_prepare_program_record(
    iree_hal_amd_xdna_aie2p_program_record_type_t type,
    iree_byte_span_t storage) {
  if (storage.data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P program record length overflows");
  }
  const iree_hal_amd_xdna_elf_program_record_header_t header = {
      .type = (uint16_t)type,
      .flags = 0,
      .byte_length = (uint32_t)storage.data_length,
  };
  return iree_hal_amd_xdna_elf_encode_program_record_header(
      &header,
      iree_make_byte_span(storage.data,
                          IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE));
}

static iree_status_t iree_hal_amd_xdna_aie2p_decode_program_record_header(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_elf_program_record_header_t* out_header) {
  *out_header = (iree_hal_amd_xdna_elf_program_record_header_t){0};
  if (storage.data == NULL ||
      storage.data_length < IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P program record is truncated");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_elf_decode_program_record_header(
      iree_make_const_byte_span(
          storage.data, IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE),
      out_header));
  if (out_header->flags != 0 ||
      (iree_host_size_t)out_header->byte_length != storage.data_length) {
    *out_header = (iree_hal_amd_xdna_elf_program_record_header_t){0};
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P program record framing is invalid");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_measure_register_block_write32(
    iree_host_size_t word_count, iree_host_size_t* out_byte_length) {
  IREE_ASSERT_ARGUMENT(out_byte_length);
  *out_byte_length = 0;
  if (word_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P block-write record must contain words");
  }
  if (!iree_host_size_checked_mul_add(
          IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE,
          word_count, sizeof(uint32_t), out_byte_length) ||
      *out_byte_length > IREE_HAL_AMD_XDNA_ELF_MAX_PROGRAM_RECORD_SIZE) {
    *out_byte_length = 0;
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P block-write record size is invalid");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_encode_register_write32(
    const iree_hal_amd_xdna_aie2p_register_write32_t* value,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(value);
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_aie2p_validate_register_range(value->address, 1));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_prepare_program_record(
      IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32, storage));
  iree_unaligned_store_le_u32(storage.data + 8, value->address);
  iree_unaligned_store_le_u32(storage.data + 12, value->value);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_encode_register_mask_write32(
    const iree_hal_amd_xdna_aie2p_register_mask_write32_t* value,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(value);
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_aie2p_validate_register_range(value->address, 1));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_AIE2P_REGISTER_MASK_WRITE32_RECORD_SIZE));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_prepare_program_record(
      IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32, storage));
  iree_unaligned_store_le_u32(storage.data + 8, value->address);
  iree_unaligned_store_le_u32(storage.data + 12, value->mask);
  iree_unaligned_store_le_u32(storage.data + 16, value->value);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_encode_register_block_write32(
    uint32_t address, iree_host_size_t word_count, const uint32_t* words,
    iree_byte_span_t storage) {
  iree_host_size_t required_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_measure_register_block_write32(
      word_count, &required_size));
  if (words == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P block-write words are missing");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_aie2p_validate_register_range(address, word_count));
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_aie2p_require_mutable_storage(storage, required_size));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_prepare_program_record(
      IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32, storage));
  iree_unaligned_store_le_u32(storage.data + 8, address);
  iree_unaligned_store_le_u32(storage.data + 12, (uint32_t)word_count);
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    iree_unaligned_store_le_u32(storage.data + 16 + i * sizeof(uint32_t),
                                words[i]);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_encode_tile_program_load(
    const iree_hal_amd_xdna_aie2p_tile_program_load_t* value,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(value);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_AIE2P_TILE_PROGRAM_LOAD_RECORD_SIZE));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_prepare_program_record(
      IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD, storage));
  iree_unaligned_store_le_u32(storage.data + 8, value->program_header_ordinal);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_aie2p_validate_dma_task_wait(
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* value) {
  if (value->direction !=
          IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_STREAM_TO_MEMORY &&
      value->direction !=
          IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_MEMORY_TO_STREAM) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P DMA task wait direction is invalid");
  }
  if (value->column_count == 0 || value->row_count == 0 ||
      (uint16_t)value->column + value->column_count > UINT8_MAX + 1u ||
      (uint16_t)value->row + value->row_count > UINT8_MAX + 1u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P DMA task wait is invalid");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_encode_dma_task_wait(
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* value,
    iree_byte_span_t storage) {
  IREE_ASSERT_ARGUMENT(value);
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_validate_dma_task_wait(value));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_require_mutable_storage(
      storage, IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_prepare_program_record(
      IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT, storage));
  storage.data[8] = value->column;
  storage.data[9] = value->row;
  storage.data[10] = (uint8_t)value->direction;
  storage.data[11] = value->dma_channel;
  storage.data[12] = value->column_count;
  storage.data[13] = value->row_count;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_decode_program_record(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_aie2p_program_record_t* out_record) {
  IREE_ASSERT_ARGUMENT(out_record);
  *out_record = (iree_hal_amd_xdna_aie2p_program_record_t){0};
  iree_hal_amd_xdna_elf_program_record_header_t header;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_aie2p_decode_program_record_header(storage, &header));
  iree_hal_amd_xdna_aie2p_program_record_t record = {
      .type = (iree_hal_amd_xdna_aie2p_program_record_type_t)header.type,
  };
  switch (record.type) {
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32:
      if (storage.data_length !=
          IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P register-write record size is invalid");
      }
      record.value.register_write32 =
          (iree_hal_amd_xdna_aie2p_register_write32_t){
              .address = iree_unaligned_load_le_u32(storage.data + 8),
              .value = iree_unaligned_load_le_u32(storage.data + 12),
          };
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_validate_register_range(
          record.value.register_write32.address, 1));
      break;
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32:
      if (storage.data_length !=
          IREE_HAL_AMD_XDNA_AIE2P_REGISTER_MASK_WRITE32_RECORD_SIZE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P masked register-write record size is invalid");
      }
      record.value.register_mask_write32 =
          (iree_hal_amd_xdna_aie2p_register_mask_write32_t){
              .address = iree_unaligned_load_le_u32(storage.data + 8),
              .mask = iree_unaligned_load_le_u32(storage.data + 12),
              .value = iree_unaligned_load_le_u32(storage.data + 16),
          };
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_validate_register_range(
          record.value.register_mask_write32.address, 1));
      break;
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32: {
      if (storage.data_length <=
              IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE ||
          (storage.data_length -
           IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE) %
                  sizeof(uint32_t) !=
              0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P block-write record size is invalid");
      }
      const uint32_t word_count = iree_unaligned_load_le_u32(storage.data + 12);
      const iree_host_size_t word_data_length =
          storage.data_length -
          IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE;
      if ((iree_host_size_t)word_count != word_data_length / sizeof(uint32_t)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P block-write word count does not match its record");
      }
      record.value.register_block_write32 =
          (iree_hal_amd_xdna_aie2p_register_block_write32_t){
              .address = iree_unaligned_load_le_u32(storage.data + 8),
              .word_count = word_count,
              .word_data = iree_make_const_byte_span(
                  storage.data +
                      IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE,
                  word_data_length),
          };
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_validate_register_range(
          record.value.register_block_write32.address, word_count));
      break;
    }
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD:
      if (storage.data_length !=
          IREE_HAL_AMD_XDNA_AIE2P_TILE_PROGRAM_LOAD_RECORD_SIZE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AIE2P tile-program-load record size is invalid");
      }
      record.value.tile_program_load =
          (iree_hal_amd_xdna_aie2p_tile_program_load_t){
              .program_header_ordinal =
                  iree_unaligned_load_le_u32(storage.data + 8),
          };
      break;
    case IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT:
      if (storage.data_length !=
              IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE ||
          storage.data[14] != 0 || storage.data[15] != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AIE2P DMA task wait framing is invalid");
      }
      record.value.dma_task_wait = (iree_hal_amd_xdna_aie2p_dma_task_wait_t){
          .column = storage.data[8],
          .row = storage.data[9],
          .direction =
              (iree_hal_amd_xdna_aie2p_dma_direction_t)storage.data[10],
          .dma_channel = storage.data[11],
          .column_count = storage.data[12],
          .row_count = storage.data[13],
      };
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_validate_dma_task_wait(
          &record.value.dma_task_wait));
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unknown AIE2P program record type %u",
                              (unsigned)header.type);
  }
  *out_record = record;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_program_record_read_block_word(
    const iree_hal_amd_xdna_aie2p_program_record_t* record,
    iree_host_size_t word_ordinal, uint32_t* out_word) {
  IREE_ASSERT_ARGUMENT(record);
  IREE_ASSERT_ARGUMENT(out_word);
  *out_word = 0;
  if (record->type !=
      IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P program record is not a block write");
  }
  if (record->value.register_block_write32.word_data.data == NULL ||
      record->value.register_block_write32.word_data.data_length !=
          (iree_host_size_t)record->value.register_block_write32.word_count *
              sizeof(uint32_t) ||
      word_ordinal >= record->value.register_block_write32.word_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P block-write word ordinal is invalid");
  }
  *out_word = iree_unaligned_load_le_u32(
      record->value.register_block_write32.word_data.data +
      word_ordinal * sizeof(uint32_t));
  return iree_ok_status();
}
