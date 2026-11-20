// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed executable AIE2P array and invocation-control programs.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_H_

#include "iree/base/api.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiler-owned operation awaiting native command emission.
typedef enum loom_aie2p_program_record_type_e {
  LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32 = 1,
  LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WRITE32 = 2,
  LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32 = 3,
  LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD = 4,
  LOOM_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT = 5,
  LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WAIT32 = 6,
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

// Waits until (register & mask) == value before issuing later commands.
typedef struct loom_aie2p_program_register_mask_wait32_t {
  // Absolute AIE array register address.
  uint32_t address;
  // Register bits participating in the comparison.
  uint32_t mask;
  // Required positioned value; bits outside the mask are zero.
  uint32_t value;
} loom_aie2p_program_register_mask_wait32_t;

// One contiguous sequence of 32-bit configuration-register writes.
typedef struct loom_aie2p_program_register_block_write32_t {
  // Absolute address of the first register word.
  uint32_t address;
  // Words written to consecutive addresses, borrowed until emission completes.
  const uint32_t* words;
  // Number of words in the block.
  iree_host_size_t word_count;
} loom_aie2p_program_register_block_write32_t;

// One resident tile program loaded while its destination core remains reset.
typedef struct loom_aie2p_program_tile_program_load_t {
  // Index into the entry's linked worker table.
  uint32_t tile_program_index;
} loom_aie2p_program_tile_program_load_t;

// One firmware task-completion-token wait for an array DMA task.
typedef struct loom_aie2p_program_dma_task_wait_t {
  // First physical tile in the waited range.
  loom_xdna_tile_coordinate_t coordinate;
  // DMA direction whose task issued the completion token.
  loom_xdna_dma_direction_t direction;
  // Direction-local DMA channel ordinal.
  uint8_t dma_channel;
  // Number of consecutive columns in the waited range.
  uint8_t column_count;
  // Number of consecutive rows in the waited range.
  uint8_t row_count;
} loom_aie2p_program_dma_task_wait_t;

// One typed operation in an array or invocation-control program.
typedef struct loom_aie2p_program_record_t {
  // Native operation kind selecting one union member.
  loom_aie2p_program_record_type_t type;
  union {
    // REGISTER_WRITE32 payload.
    loom_aie2p_program_register_write32_t register_write32;
    // REGISTER_MASK_WRITE32 payload.
    loom_aie2p_program_register_mask_write32_t register_mask_write32;
    // REGISTER_MASK_WAIT32 payload.
    loom_aie2p_program_register_mask_wait32_t register_mask_wait32;
    // REGISTER_BLOCK_WRITE32 payload.
    loom_aie2p_program_register_block_write32_t register_block_write32;
    // TILE_PROGRAM_LOAD payload.
    loom_aie2p_program_tile_program_load_t tile_program_load;
    // DMA_TASK_WAIT payload.
    loom_aie2p_program_dma_task_wait_t dma_task_wait;
  } value;
} loom_aie2p_program_record_t;

// Runtime shim-address relocation targeting a block-write word pair.
typedef struct loom_aie2p_program_relocation_t {
  // Control-program record containing the target block write.
  uint32_t target_record_index;
  // First target word within the block-write payload.
  uint32_t target_word_index;
  // Dense entry-relative binding ordinal supplying the runtime value.
  uint32_t binding_ordinal;
  // Signed addend applied to the supplied runtime value.
  int64_t addend;
  // Minimum permitted relocated unsigned value.
  uint64_t minimum_value;
  // Maximum permitted relocated unsigned value.
  uint64_t maximum_value;
  // Required relocated-value alignment in bytes.
  uint64_t required_alignment;
} loom_aie2p_program_relocation_t;

// One shim DMA descriptor reading immutable loader data during initialization.
typedef struct loom_aie2p_program_upload_patch_t {
  // Initialization record publishing the complete shim descriptor.
  uint32_t record_index;
  // First byte in the immutable DMA allocation.
  uint32_t source_offset;
  // Number of bytes accessed by this descriptor.
  uint32_t byte_length;
} loom_aie2p_program_upload_patch_t;

// Equivalent initialization using DMA in place of inline firmware writes.
typedef struct loom_aie2p_program_upload_t {
  // Initialization including the loader's temporary circuits and queue drains.
  const loom_aie2p_program_record_t* records;
  // Number of initialization records.
  iree_host_size_t record_count;
  // Immutable bytes in device DMA order, shared across cold invocations.
  iree_const_byte_span_t contents;
  // Shim address fixups in increasing initialization-record order.
  const loom_aie2p_program_upload_patch_t* patches;
  // Number of shim DMA descriptors referencing contents.
  iree_host_size_t patch_count;
} loom_aie2p_program_upload_t;

// Executable array initialization and invocation-control programs.
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
  // Optional same-epoch timer interval retained from ordered measure commands.
  struct {
    // True when the invocation declares a complete measurement interval.
    bool present;
    // Control-record boundary immediately before measured work.
    uint32_t begin_record;
    // Control-record boundary after measured work, possibly record_count.
    uint32_t end_record;
  } measurement;
  // Optional bulk loader for product directories with room for its DMA backing.
  const loom_aie2p_program_upload_t* upload;
} loom_aie2p_array_program_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_H_
