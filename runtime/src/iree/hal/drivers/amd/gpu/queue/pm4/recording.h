// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RECORDING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RECORDING_H_

#include "iree/hal/drivers/amd/gpu/queue/pm4/fixup.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_pm4_recording_chunk_t
    iree_hal_amd_gpu_pm4_recording_chunk_t;

// Temporary append storage. Chunk boundaries have no native execution meaning;
// sealing copies these bytes once into the program's resident block layout.
typedef struct iree_hal_amd_gpu_pm4_recording_stream_t {
  // First initialized chunk, or NULL for an empty stream.
  iree_hal_amd_gpu_pm4_recording_chunk_t* head;
  // Current append chunk, or NULL before the first allocation.
  iree_hal_amd_gpu_pm4_recording_chunk_t* tail;
  // Complete logical extent, including alignment holes, in bytes.
  uint32_t length;
} iree_hal_amd_gpu_pm4_recording_stream_t;

// Recording-time native bytes and relocations, discarded after sealing. There
// are no operation records or invocation cursors. Commands and shader-readable
// data occupy separate ranges so command fetch never walks argument payloads.
typedef struct iree_hal_amd_gpu_pm4_recording_t {
  // Allocator for temporary chunks only.
  iree_allocator_t host_allocator;
  // Encoded native commands in execution order.
  iree_hal_amd_gpu_pm4_recording_stream_t commands;
  // Kernargs, indirect launch data, and captured update bytes.
  iree_hal_amd_gpu_pm4_recording_stream_t data;
  // Independent fixups with recording-time section-relative offsets.
  iree_hal_amd_gpu_pm4_recording_stream_t fixups;
  // Maximum alignment requested for private data.
  uint32_t alignment;
} iree_hal_amd_gpu_pm4_recording_t;

void iree_hal_amd_gpu_pm4_recording_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_recording_t* out_recording);
void iree_hal_amd_gpu_pm4_recording_deinitialize(
    iree_hal_amd_gpu_pm4_recording_t* recording);

// Appends uninitialized native words. The caller initializes every word before
// sealing. Allocation and the hardware block extent are the only failures.
iree_status_t iree_hal_amd_gpu_pm4_recording_commands(
    iree_hal_amd_gpu_pm4_recording_t* recording, uint32_t word_count,
    uint32_t* out_offset, uint32_t** out_words);

// Appends zero-initialized private data with the requested native alignment.
iree_status_t iree_hal_amd_gpu_pm4_recording_data(
    iree_hal_amd_gpu_pm4_recording_t* recording, uint32_t byte_length,
    uint32_t alignment, uint32_t* out_offset, void** out_data);

// Records a complete 32- or 64-bit replacement. target_data selects the private
// data section. addend_data makes the addend relative to that section's final
// placement; otherwise it is an ordinary immediate displacement. shift is zero
// for pointers and eight for native instruction-address register pairs.
iree_status_t iree_hal_amd_gpu_pm4_recording_fixup(
    iree_hal_amd_gpu_pm4_recording_t* recording, bool target_data,
    uint32_t target_offset, uint32_t source, uint64_t addend, bool addend_data,
    uint32_t shift, bool wide);

// Returns the final private-data offset after the complete command extent has
// been appended. The final instance base satisfies recording->alignment.
uint32_t iree_hal_amd_gpu_pm4_recording_data_offset(
    const iree_hal_amd_gpu_pm4_recording_t* recording);
uint32_t iree_hal_amd_gpu_pm4_recording_fixup_count(
    const iree_hal_amd_gpu_pm4_recording_t* recording);

// Copies admitted recording bytes and resolves section-relative fixups once.
// Destination covers data_offset + data.length; table covers fixup_count.
// Both outputs may be write-combined device memory and are only written.
void iree_hal_amd_gpu_pm4_recording_write(
    const iree_hal_amd_gpu_pm4_recording_t* recording, uint8_t* destination,
    iree_hal_amd_gpu_pm4_fixup_t* table);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RECORDING_H_
