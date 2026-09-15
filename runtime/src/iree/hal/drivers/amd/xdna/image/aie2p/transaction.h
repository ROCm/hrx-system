// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P transaction 0.1 encoding.
//
// This writer owns only the target-native firmware wire format. Image record
// interpretation, target qualification, and provider object construction live
// in higher layers.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_TRANSACTION_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_TRANSACTION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/program_format.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  // Serialized byte length of a transaction 0.1 header.
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_HEADER_SIZE = 16,
  // Serialized byte length of a register-write operation.
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_WRITE32_SIZE = 24,
  // Serialized byte length of a masked register-write operation.
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_MASK_WRITE32_SIZE = 28,
  // Serialized header byte length preceding block-write words.
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_REGISTER_BLOCK_WRITE32_HEADER_SIZE = 16,
  // Serialized byte length of a DMA task-wait operation.
  IREE_HAL_AMD_XDNA_AIE2P_TRANSACTION_DMA_TASK_WAIT_SIZE = 16,
};

// Complete context-specific fields encoded in a transaction 0.1 header.
typedef struct iree_hal_amd_xdna_aie2p_transaction_target_t {
  // AIE-RT device-generation value.
  uint8_t device_generation;
  // Number of rows in the logical context.
  uint8_t row_count;
  // Number of columns in the logical context.
  uint8_t column_count;
  // Number of memory-tile rows in the target generation.
  uint8_t memory_tile_row_count;
} iree_hal_amd_xdna_aie2p_transaction_target_t;

// In-place writer for one caller-owned transaction buffer.
//
// Append calls are transactional: insufficient capacity or invalid input
// leaves the writer offset and operation count unchanged. The storage contents
// are complete only after finalization.
typedef struct iree_hal_amd_xdna_aie2p_transaction_writer_t {
  // Complete caller-owned output capacity.
  iree_byte_span_t storage;
  // First uncommitted byte in |storage|.
  iree_host_size_t byte_offset;
  // Number of complete operations in |storage|.
  uint32_t operation_count;
  // Header fields copied during finalization.
  iree_hal_amd_xdna_aie2p_transaction_target_t target;
} iree_hal_amd_xdna_aie2p_transaction_writer_t;

// Initializes |out_writer| over caller-owned |storage|.
iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_initialize(
    const iree_hal_amd_xdna_aie2p_transaction_target_t* target,
    iree_byte_span_t storage,
    iree_hal_amd_xdna_aie2p_transaction_writer_t* out_writer);

// Appends one complete 32-bit configuration-register write.
iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_append_write32(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer, uint32_t address,
    uint32_t value);

// Appends one masked 32-bit configuration-register update.
iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_append_mask_write32(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer, uint32_t address,
    uint32_t value, uint32_t mask);

// Appends consecutive 32-bit configuration-register writes.
//
// |word_data| must contain one or more complete little-endian words. |column|
// and |row| are the context-relative destination encoded by the native block
// operation and must agree with the tile bits already present in |address|.
iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_append_block_write32(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer, uint32_t address,
    uint8_t column, uint8_t row, iree_const_byte_span_t word_data);

// Appends one firmware task-completion-token wait.
iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_append_dma_task_wait(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer,
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* wait);

// Finalizes the header and returns the written prefix of |storage|.
iree_status_t iree_hal_amd_xdna_aie2p_transaction_writer_finalize(
    iree_hal_amd_xdna_aie2p_transaction_writer_t* writer,
    iree_const_byte_span_t* out_transaction);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_TRANSACTION_H_
