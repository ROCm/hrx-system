// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/aie2p/transaction.h"

#include <string.h>

typedef enum iree_hal_amd_xdna_aie2p_transaction_opcode_e {
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_OPCODE_WRITE32 = 0,
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_OPCODE_BLOCK_WRITE32 = 1,
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_OPCODE_MASK_WRITE32 = 3,
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_OPCODE_DMA_TASK_WAIT = 128,
} iree_hal_amd_xdna_aie2p_transaction_opcode_t;

static iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_reserve(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer,
    iree_host_size_t operation_size, uint8_t** out_operation) {
  *out_operation = NULL;
  if (writer == NULL || writer->storage.data == NULL ||
      writer->byte_offset > writer->storage.data_length) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P transaction writer state is invalid");
  }
  if (writer->operation_count == UINT32_MAX ||
      operation_size > writer->storage.data_length - writer->byte_offset) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P transaction storage is exhausted");
  }
  uint8_t* operation = writer->storage.data + writer->byte_offset;
  memset(operation, 0, operation_size);
  *out_operation = operation;
  return iree_ok_status();
}

static void iree_hal_amd_xdna_aie2p_transaction_writer_commit(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer,
    iree_host_size_t operation_size) {
  writer->byte_offset += operation_size;
  ++writer->operation_count;
}

iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_initialize(
    const iree_hal_amd_xdna_aie2p_transaction_target_t* target,
    iree_byte_span_t storage,
    iree_hal_amd_xdna_aie2p_transaction_writer_t* out_writer) {
  IREE_ASSERT_ARGUMENT(out_writer);
  *out_writer = (iree_hal_amd_xdna_aie2p_transaction_writer_t){0};
  if (target == NULL || target->device_generation == 0 ||
      target->row_count == 0 || target->column_count == 0 ||
      storage.data == NULL ||
      storage.data_length < IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE ||
      storage.data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P transaction writer parameters are invalid");
  }
  memset(storage.data, 0, IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE);
  *out_writer = (iree_hal_amd_xdna_aie2p_transaction_writer_t){
      .storage = storage,
      .byte_offset = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE,
      .operation_count = 0,
      .target = *target,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_append_write32(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer, uint32_t address,
    uint32_t value) {
  uint8_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_transaction_writer_reserve(
      writer, IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_WRITE32_SIZE,
      &operation));
  operation[0] = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_OPCODE_WRITE32;
  iree_unaligned_store_le_u32(operation + 8, address);
  iree_unaligned_store_le_u32(operation + 16, value);
  iree_unaligned_store_le_u32(
      operation + 20,
      IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_WRITE32_SIZE);
  iree_hal_amd_xdna_aie2p_transaction_writer_commit(
      writer, IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_WRITE32_SIZE);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_append_mask_write32(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer, uint32_t address,
    uint32_t value, uint32_t mask) {
  uint8_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_transaction_writer_reserve(
      writer, IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_MASK_WRITE32_SIZE,
      &operation));
  operation[0] = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_OPCODE_MASK_WRITE32;
  iree_unaligned_store_le_u32(operation + 8, address);
  iree_unaligned_store_le_u32(operation + 16, value);
  iree_unaligned_store_le_u32(operation + 20, mask);
  iree_unaligned_store_le_u32(
      operation + 24,
      IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_MASK_WRITE32_SIZE);
  iree_hal_amd_xdna_aie2p_transaction_writer_commit(
      writer, IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_MASK_WRITE32_SIZE);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_append_block_write32(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer, uint32_t address,
    uint8_t column, uint8_t row, iree_const_byte_span_t word_data) {
  if (word_data.data == NULL || word_data.data_length == 0 ||
      word_data.data_length % sizeof(uint32_t) != 0 ||
      word_data.data_length >
          UINT32_MAX -
              IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_BLOCK_WRITE32_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P transaction block write is invalid");
  }
  const iree_host_size_t operation_size =
      IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_BLOCK_WRITE32_HEADER_SIZE +
      word_data.data_length;
  uint8_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_transaction_writer_reserve(
      writer, operation_size, &operation));
  operation[0] = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_OPCODE_BLOCK_WRITE32;
  operation[4] = column;
  operation[5] = row;
  iree_unaligned_store_le_u32(operation + 8, address);
  iree_unaligned_store_le_u32(operation + 12, (uint32_t)operation_size);
  memcpy(
      operation +
          IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_BLOCK_WRITE32_HEADER_SIZE,
      word_data.data, word_data.data_length);
  iree_hal_amd_xdna_aie2p_transaction_writer_commit(writer, operation_size);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_append_dma_task_wait(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer,
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* wait) {
  if (wait == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P transaction DMA task wait is invalid");
  }
  uint8_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_transaction_writer_reserve(
      writer, IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_DMA_TASK_WAIT_SIZE,
      &operation));
  operation[0] = IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_OPCODE_DMA_TASK_WAIT;
  iree_unaligned_store_le_u32(
      operation + 4, IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_DMA_TASK_WAIT_SIZE);
  iree_unaligned_store_le_u32(
      operation + 8, ((uint32_t)wait->direction) | ((uint32_t)wait->row << 8) |
                         ((uint32_t)wait->column << 16));
  iree_unaligned_store_le_u32(operation + 12,
                              ((uint32_t)wait->row_count << 8) |
                                  ((uint32_t)wait->column_count << 16) |
                                  ((uint32_t)wait->dma_channel << 24));
  iree_hal_amd_xdna_aie2p_transaction_writer_commit(
      writer, IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_DMA_TASK_WAIT_SIZE);
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_finalize(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer,
    iree_const_byte_span_t* out_transaction) {
  IREE_ASSERT_ARGUMENT(out_transaction);
  *out_transaction = iree_const_byte_span_empty();
  if (writer == NULL || writer->storage.data == NULL ||
      writer->byte_offset < IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE ||
      writer->byte_offset > writer->storage.data_length ||
      writer->byte_offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P transaction writer state is invalid");
  }
  uint8_t* header = writer->storage.data;
  header[0] = 0;
  header[1] = 1;
  header[2] = writer->target.device_generation;
  header[3] = writer->target.row_count;
  header[4] = writer->target.column_count;
  header[5] = writer->target.memory_tile_row_count;
  header[6] = 0;
  header[7] = 0;
  iree_unaligned_store_le_u32(header + 8, writer->operation_count);
  iree_unaligned_store_le_u32(header + 12, (uint32_t)writer->byte_offset);
  *out_transaction =
      iree_make_const_byte_span(writer->storage.data, writer->byte_offset);
  return iree_ok_status();
}
