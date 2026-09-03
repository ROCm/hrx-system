// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed executable AIE2P array and invocation-control programs.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_PROGRAM_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_PROGRAM_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/target/arch/amd/xdna/elf_format.h"

#ifdef __cplusplus
extern "C" {
#endif

// AIE2P payload-ABI operation executed by the native XDNA loader.
typedef enum loom_aie2p_program_record_type_e {
  LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32 = 1,
  LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32 = 2,
  LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32 = 3,
  LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD = 4,
  LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT = 5,
} loom_aie2p_program_record_type_t;

// One complete 32-bit configuration-register write.
typedef struct loom_aie2p_program_register_write32_t {
  // Absolute AIE array register address.
  uint32_t address;
  // Complete value written to the register.
  uint32_t value;
} loom_aie2p_program_register_write32_t;

// One masked 32-bit configuration-register update.
typedef struct loom_aie2p_program_register_mask_write32_t {
  // Absolute AIE array register address.
  uint32_t address;
  // Register bits replaced by value.
  uint32_t mask;
  // Positioned register value; bits outside mask are ignored.
  uint32_t value;
} loom_aie2p_program_register_mask_write32_t;

// One contiguous sequence of 32-bit configuration-register writes.
typedef struct loom_aie2p_program_register_block_write32_t {
  // Absolute address of the first register word.
  uint32_t address;
  // Arena-owned words written to consecutive addresses.
  const uint32_t* words;
  // Number of words in the block.
  iree_host_size_t word_count;
} loom_aie2p_program_register_block_write32_t;

// One tile image loaded while its destination core remains reset.
typedef struct loom_aie2p_program_tile_program_load_t {
  // Index relative to the array payload's first tile program header.
  uint32_t tile_program_index;
} loom_aie2p_program_tile_program_load_t;

// One firmware task-completion-token wait for a shim DMA task.
typedef struct loom_aie2p_program_dma_task_wait_t {
  // First physical tile in the waited range.
  loom_xdna_tile_coordinate_t coordinate;
  // DMA direction whose task issued the completion token.
  loom_aie2p_array_dma_direction_t direction;
  // Direction-local DMA channel ordinal.
  uint8_t dma_channel;
  // Number of consecutive columns in the waited range.
  uint8_t column_count;
  // Number of consecutive rows in the waited range.
  uint8_t row_count;
} loom_aie2p_program_dma_task_wait_t;

// One typed operation in an array or invocation-control program.
typedef struct loom_aie2p_program_record_t {
  // Payload-ABI operation kind selecting one union member.
  loom_aie2p_program_record_type_t type;
  union {
    // REGISTER_WRITE32 payload.
    loom_aie2p_program_register_write32_t register_write32;
    // REGISTER_MASK_WRITE32 payload.
    loom_aie2p_program_register_mask_write32_t register_mask_write32;
    // REGISTER_BLOCK_WRITE32 payload.
    loom_aie2p_program_register_block_write32_t register_block_write32;
    // TILE_PROGRAM_LOAD payload.
    loom_aie2p_program_tile_program_load_t tile_program_load;
    // DMA_TASK_WAIT payload.
    loom_aie2p_program_dma_task_wait_t dma_task_wait;
  } value;
} loom_aie2p_program_record_t;

// Runtime relocation targeting one word range in a block-write record.
typedef struct loom_aie2p_program_relocation_t {
  // Control-program record containing the target block write.
  uint32_t target_record_index;
  // First target word within the block-write payload.
  uint32_t target_word_index;
  // Dense entry-relative binding ordinal supplying the runtime value.
  uint32_t binding_ordinal;
  // Runtime field interpretation.
  loom_xdna_elf_relocation_kind_t kind;
  // Number of consecutive target bytes.
  uint8_t field_byte_width;
  // Signed addend applied to the supplied runtime value.
  int64_t addend;
  // Minimum permitted relocated unsigned value.
  uint64_t minimum_value;
  // Maximum permitted relocated unsigned value.
  uint64_t maximum_value;
  // Required relocated-value alignment in bytes.
  uint64_t required_alignment;
} loom_aie2p_program_relocation_t;

// Arena-owned executable programs materialized from one physical array plan.
typedef struct loom_aie2p_array_program_t {
  // Resident array configuration and core activation records.
  const loom_aie2p_program_record_t* array_records;
  // Number of array configuration records.
  iree_host_size_t array_record_count;
  // Per-invocation shim DMA and completion records.
  const loom_aie2p_program_record_t* control_records;
  // Number of invocation-control records.
  iree_host_size_t control_record_count;
  // Runtime patches into control records.
  const loom_aie2p_program_relocation_t* relocations;
  // Number of runtime patches.
  iree_host_size_t relocation_count;
  // Number of tile program headers referenced by the array program.
  uint32_t tile_program_count;
} loom_aie2p_array_program_t;

// Encoded canonical XDNA payloads and their resolved relocation table rows.
typedef struct loom_aie2p_encoded_array_program_t {
  // Complete encoded ARRAY program-header payload.
  iree_const_byte_span_t array_payload;
  // Complete encoded CONTROL program-header payload.
  iree_const_byte_span_t control_payload;
  // Arena-owned fixed-width runtime relocation rows.
  const loom_xdna_elf_relocation_record_t* relocations;
  // Number of runtime relocation rows.
  iree_host_size_t relocation_count;
} loom_aie2p_encoded_array_program_t;

// Materializes executable AIE2P array and invocation-control operations.
//
// The array program resets resident cores, initializes locks and routing,
// programs circular compute DMA rings, loads each tile image, and activates the
// cores. The control program patches and queues one finite shim DMA task per
// external channel and waits for every egress completion token.
iree_status_t loom_aie2p_array_program_build(
    const loom_aie2p_array_plan_t* plan, iree_arena_allocator_t* arena,
    loom_aie2p_array_program_t* out_program);

// Encodes one typed program after final ELF program-header placement.
//
// Tile program indices are resolved relative to
// |first_tile_program_header_ordinal|. Runtime relocation targets are resolved
// against |control_program_header_ordinal| and the encoded control record
// offsets.
iree_status_t loom_aie2p_array_program_encode(
    const loom_aie2p_array_program_t* program,
    uint32_t first_tile_program_header_ordinal,
    uint32_t control_program_header_ordinal, iree_arena_allocator_t* arena,
    loom_aie2p_encoded_array_program_t* out_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_PROGRAM_H_
