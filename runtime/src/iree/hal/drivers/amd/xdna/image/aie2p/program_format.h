// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P ARRAY and CONTROL program record wire format.
//
// These records are embedded in the generic XDNA image program framing. The
// decoded values are independent of compiler planning objects and native
// transaction representations. Variable block-write word storage is borrowed
// from the encoded record and remains little-endian.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_PROGRAM_FORMAT_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_PROGRAM_FORMAT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/format.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// AIE2P payload operation selected by the generic program-record type.
typedef enum iree_hal_amd_xdna_aie2p_program_record_type_e {
  IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32 = 1,
  IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32 = 2,
  IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32 = 3,
  IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD = 4,
  IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT = 5,
} iree_hal_amd_xdna_aie2p_program_record_type_t;

// DMA direction encoded by a task-wait record.
typedef enum iree_hal_amd_xdna_aie2p_dma_direction_e {
  IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_STREAM_TO_MEMORY = 0,
  IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_MEMORY_TO_STREAM = 1,
} iree_hal_amd_xdna_aie2p_dma_direction_t;

enum {
  // Serialized byte length of a complete register-write record.
  IREE_HAL_AMD_XDNA_AIE2P_REGISTER_WRITE32_RECORD_SIZE = 16,
  // Serialized byte length of a complete masked register-write record.
  IREE_HAL_AMD_XDNA_AIE2P_REGISTER_MASK_WRITE32_RECORD_SIZE = 20,
  // Serialized header byte length preceding block-write words.
  IREE_HAL_AMD_XDNA_AIE2P_REGISTER_BLOCK_WRITE32_HEADER_SIZE = 16,
  // Serialized byte length of a complete tile-program-load record.
  IREE_HAL_AMD_XDNA_AIE2P_TILE_PROGRAM_LOAD_RECORD_SIZE = 12,
  // Serialized byte length of a complete DMA-task-wait record.
  IREE_HAL_AMD_XDNA_AIE2P_DMA_TASK_WAIT_RECORD_SIZE = 16,
};

// One complete 32-bit configuration-register write.
typedef struct iree_hal_amd_xdna_aie2p_register_write32_t {
  // Absolute AIE array register address.
  uint32_t address;
  // Complete value written to the register.
  uint32_t value;
} iree_hal_amd_xdna_aie2p_register_write32_t;

// One masked 32-bit configuration-register update.
typedef struct iree_hal_amd_xdna_aie2p_register_mask_write32_t {
  // Absolute AIE array register address.
  uint32_t address;
  // Register bits replaced by |value|.
  uint32_t mask;
  // Positioned register value; bits outside |mask| are ignored.
  uint32_t value;
} iree_hal_amd_xdna_aie2p_register_mask_write32_t;

// One contiguous sequence of 32-bit configuration-register writes.
typedef struct iree_hal_amd_xdna_aie2p_register_block_write32_t {
  // Absolute address of the first register word.
  uint32_t address;
  // Number of consecutive words in |word_data|.
  uint32_t word_count;
  // Borrowed little-endian word bytes within the decoded record storage.
  iree_const_byte_span_t word_data;
} iree_hal_amd_xdna_aie2p_register_block_write32_t;

// One resident tile program loaded while its destination core remains reset.
typedef struct iree_hal_amd_xdna_aie2p_tile_program_load_t {
  // Absolute ELF program-header ordinal naming the TILE payload.
  uint32_t program_header_ordinal;
} iree_hal_amd_xdna_aie2p_tile_program_load_t;

// One firmware task-completion-token wait for a shim DMA task.
typedef struct iree_hal_amd_xdna_aie2p_dma_task_wait_t {
  // Context-relative first tile column in the waited range.
  uint8_t column;
  // Context-relative first tile row in the waited range.
  uint8_t row;
  // DMA direction whose task issued the completion token.
  iree_hal_amd_xdna_aie2p_dma_direction_t direction;
  // Direction-local DMA channel ordinal.
  uint8_t dma_channel;
  // Number of consecutive columns in the waited range.
  uint8_t column_count;
  // Number of consecutive rows in the waited range.
  uint8_t row_count;
} iree_hal_amd_xdna_aie2p_dma_task_wait_t;

// Decoded AIE2P program record.
//
// Any borrowed spans remain valid only while the source |storage| passed to
// iree_hal_amd_xdna_aie2p_decode_program_record remains valid.
typedef struct iree_hal_amd_xdna_aie2p_program_record_t {
  // Record operation selecting one value member.
  iree_hal_amd_xdna_aie2p_program_record_type_t type;
  // Operation payload selected by |type|.
  union {
    // REGISTER_WRITE32 payload.
    iree_hal_amd_xdna_aie2p_register_write32_t register_write32;
    // REGISTER_MASK_WRITE32 payload.
    iree_hal_amd_xdna_aie2p_register_mask_write32_t register_mask_write32;
    // REGISTER_BLOCK_WRITE32 payload.
    iree_hal_amd_xdna_aie2p_register_block_write32_t register_block_write32;
    // TILE_PROGRAM_LOAD payload.
    iree_hal_amd_xdna_aie2p_tile_program_load_t tile_program_load;
    // DMA_TASK_WAIT payload.
    iree_hal_amd_xdna_aie2p_dma_task_wait_t dma_task_wait;
  } value;
} iree_hal_amd_xdna_aie2p_program_record_t;

// Calculates the complete serialized block-write record byte length.
iree_status_t iree_hal_amd_xdna_aie2p_measure_register_block_write32(
    iree_host_size_t word_count, iree_host_size_t* out_byte_length);

// Encodes one complete register-write program record.
iree_status_t iree_hal_amd_xdna_aie2p_encode_register_write32(
    const iree_hal_amd_xdna_aie2p_register_write32_t* value,
    iree_byte_span_t storage);

// Encodes one complete masked register-write program record.
iree_status_t iree_hal_amd_xdna_aie2p_encode_register_mask_write32(
    const iree_hal_amd_xdna_aie2p_register_mask_write32_t* value,
    iree_byte_span_t storage);

// Encodes one complete block-write program record from host-order words.
iree_status_t iree_hal_amd_xdna_aie2p_encode_register_block_write32(
    uint32_t address, iree_host_size_t word_count, const uint32_t* words,
    iree_byte_span_t storage);

// Encodes one complete tile-program-load program record.
iree_status_t iree_hal_amd_xdna_aie2p_encode_tile_program_load(
    const iree_hal_amd_xdna_aie2p_tile_program_load_t* value,
    iree_byte_span_t storage);

// Encodes one complete DMA-task-wait program record.
iree_status_t iree_hal_amd_xdna_aie2p_encode_dma_task_wait(
    const iree_hal_amd_xdna_aie2p_dma_task_wait_t* value,
    iree_byte_span_t storage);

// Decodes and locally validates one complete AIE2P program record.
iree_status_t iree_hal_amd_xdna_aie2p_decode_program_record(
    iree_const_byte_span_t storage,
    iree_hal_amd_xdna_aie2p_program_record_t* out_record);

// Reads one host-order word from a decoded block-write record.
iree_status_t iree_hal_amd_xdna_aie2p_program_record_read_block_word(
    const iree_hal_amd_xdna_aie2p_program_record_t* record,
    iree_host_size_t word_ordinal, uint32_t* out_word);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_PROGRAM_FORMAT_H_
