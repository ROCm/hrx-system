// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_RECORDING_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_RECORDING_H_

#include "iree/hal/driver/amd/gpu/queue/recording.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/barrier.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold append state. Sealing retains only native templates, indexed expansion
// facts, and captured data; no public command replay or resource lookup occurs
// during emission. Packet blocks have fixups relative to their own byte zero.
typedef struct iree_hal_amd_gpu_sdma_recording_t {
  // Temporary chunked command/data storage and its host allocator.
  iree_hal_amd_gpu_recording_t storage;
  // Native family facts, independent of the shader target.
  amdf_queue_format_features_t features;
  // Mutable segment vector, released after sealing.
  iree_hal_amd_gpu_sdma_program_segment_t* segments;
  // Initialized segment extent.
  uint32_t segment_count;
  // Allocated segment extent.
  uint32_t segment_capacity;
  // Block-relative address replacements, released after sealing.
  iree_hal_amd_gpu_fixup_t* fixups;
  // Initialized replacement extent.
  uint32_t fixup_count;
  // Allocated replacement extent.
  uint32_t fixup_capacity;
} iree_hal_amd_gpu_sdma_recording_t;

void iree_hal_amd_gpu_sdma_recording_initialize(
    amdf_queue_format_features_t features, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_sdma_recording_t* out_recording);
void iree_hal_amd_gpu_sdma_recording_deinitialize(
    iree_hal_amd_gpu_sdma_recording_t* recording);

// Appends an indivisible sequence of complete native packets. word_count is
// nonzero and at most MAX_COMMAND_WORDS. Returned offset is section-relative;
// pointer() consumes it while this block remains current.
iree_status_t iree_hal_amd_gpu_sdma_recording_commands(
    iree_hal_amd_gpu_sdma_recording_t* recording, uint32_t word_count,
    uint32_t* out_offset, uint32_t** out_words);
iree_status_t iree_hal_amd_gpu_sdma_recording_pointer(
    iree_hal_amd_gpu_sdma_recording_t* recording, uint32_t byte_offset,
    uint32_t value, uint64_t addend);

// Appends a late native range expansion. The recorder has already qualified
// and deduplicated its inputs, equality constraints, and private pattern data.
iree_status_t iree_hal_amd_gpu_sdma_recording_range(
    iree_hal_amd_gpu_sdma_recording_t* recording,
    iree_hal_amd_gpu_sdma_program_segment_kind_t kind,
    iree_hal_amd_gpu_sdma_program_range_t range);

// Ends this invocation's native submission for a KERNEL cache boundary.
// Consecutive boundaries coalesce. The executor resumes only after retirement.
iree_status_t iree_hal_amd_gpu_sdma_recording_retire(
    iree_hal_amd_gpu_sdma_recording_t* recording);

// Appends the selected native execution/cache boundary. USER emits an ordered
// NOP and optional GCR; KERNEL cache work ends a submission instead.
iree_status_t iree_hal_amd_gpu_sdma_recording_barrier(
    iree_hal_amd_gpu_sdma_recording_t* recording,
    iree_hal_amd_gpu_sdma_barrier_profile_t profile,
    iree_hal_amd_gpu_sdma_barrier_t barrier);

// Seals into one host allocation, independently owned by the result. On
// success the caller frees program.storage with storage.host_allocator after
// all invocations retire. Append storage remains owned by recording.
iree_status_t iree_hal_amd_gpu_sdma_recording_seal(
    const iree_hal_amd_gpu_sdma_recording_t* recording,
    iree_hal_amd_gpu_sdma_program_t* out_program);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_RECORDING_H_
