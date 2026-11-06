// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_PROGRAM_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_PROGRAM_H_

#include "iree/hal/driver/amd/gpu/queue/fixup.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/commands.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_hal_amd_gpu_sdma_program_segment_kind_e {
  IREE_HAL_AMD_GPU_SDMA_PROGRAM_PACKETS,
  IREE_HAL_AMD_GPU_SDMA_PROGRAM_COPY,
  IREE_HAL_AMD_GPU_SDMA_PROGRAM_FILL,
  IREE_HAL_AMD_GPU_SDMA_PROGRAM_RETIRE,
} iree_hal_amd_gpu_sdma_program_segment_kind_t;

// Dense invocation values supply all external facts. A constant extent has
// length_value == UINT32_MAX; otherwise length is an additive displacement.
// FILL uses source as an address of two repeated pattern DWORDs, allowing byte
// edges without a compute blit or a read/modify/write of surrounding payload.
typedef struct iree_hal_amd_gpu_sdma_program_range_t {
  // Dense source address input, including the executor's private-data input.
  uint32_t source_value;
  // Dense target address input.
  uint32_t target_value;
  // Dense logical extent input, or UINT32_MAX for a constant extent.
  uint32_t length_value;
  // Repeated DWORD for FILL; zero for COPY.
  uint32_t pattern;
  // Displacement added to the source input, in bytes.
  uint64_t source_offset;
  // Displacement added to the target input, in bytes.
  uint64_t target_offset;
  // Constant extent or displacement added to length_value, in bytes.
  uint64_t length;
} iree_hal_amd_gpu_sdma_program_range_t;

// Immutable native template block or bounded expansion of a late-bound range.
// PACKETS never splits a packet and its fixups are relative to this block.
// RETIRE is a native carrier visibility boundary, not a host/global queue join.
typedef struct iree_hal_amd_gpu_sdma_program_segment_t {
  // Selects the initialized representation below.
  iree_hal_amd_gpu_sdma_program_segment_kind_t kind;
  union {
    // Native words and replacements coalesced into one indivisible block.
    struct {
      // First DWORD in the program's immutable command section.
      uint32_t word_offset;
      // Complete native packets, at most MAX_COMMAND_WORDS.
      uint32_t word_count;
      // First block-relative fixup in the program's immutable table.
      uint32_t fixup_offset;
      // Number of disjoint DWORD replacements in this block.
      uint32_t fixup_count;
    } packets;
    // Resolved only once dependency commitment supplies native addresses.
    iree_hal_amd_gpu_sdma_program_range_t range;
  };
} iree_hal_amd_gpu_sdma_program_segment_t;

typedef struct iree_hal_amd_gpu_sdma_program_t {
  // Sole host allocation owning all sealed arrays; NULL before sealing.
  void* storage;
  // Native format fields captured by the family, independently of shader ISA.
  amdf_queue_format_features_t features;
  // Immutable segment array owned by storage.
  const iree_hal_amd_gpu_sdma_program_segment_t* segments;
  // Number of initialized segments.
  uint32_t segment_count;
  // Immutable native packet templates owned by storage.
  const uint32_t* words;
  // Disjoint block-relative replacements owned by storage.
  const iree_hal_amd_gpu_fixup_t* fixups;
  // Captured update/pattern bytes and private control initialization.
  iree_const_byte_span_t data;
} iree_hal_amd_gpu_sdma_program_t;

// An invocation's unpublished suffix. Speculative emission updates a private
// copy; the submission owner commits it only after native acceptance.
typedef struct iree_hal_amd_gpu_sdma_program_cursor_t {
  // Next segment, or program.segment_count at the end.
  uint32_t segment;
  // Published prefix of a range segment, in bytes.
  uint64_t offset;
} iree_hal_amd_gpu_sdma_program_cursor_t;

typedef enum iree_hal_amd_gpu_sdma_program_progress_e {
  IREE_HAL_AMD_GPU_SDMA_PROGRAM_MORE,
  IREE_HAL_AMD_GPU_SDMA_PROGRAM_BOUNDARY,
  IREE_HAL_AMD_GPU_SDMA_PROGRAM_END,
} iree_hal_amd_gpu_sdma_program_progress_t;

// Emits at most word_capacity (up to MAX_COMMAND_WORDS) without allocation or
// native calls. Input and extent validity were established by recording and
// invocation capture.
// BOUNDARY requires this invocation's checked native retirement before emitting
// its successor; unrelated ready invocations remain eligible. END still needs
// the queue's native completion tail, including for an empty span.
iree_hal_amd_gpu_sdma_program_progress_t iree_hal_amd_gpu_sdma_program_emit(
    const iree_hal_amd_gpu_sdma_program_t* program, const uint64_t* values,
    iree_hal_amd_gpu_sdma_program_cursor_t* cursor, uint32_t word_capacity,
    uint32_t* words, uint32_t* out_word_count);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_PROGRAM_H_
