// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RECORDING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RECORDING_H_

#include "iree/hal/drivers/amd/gpu/queue/recording.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// PM4 block geometry over shared append-only native commands/data/fixups.
typedef struct iree_hal_amd_gpu_pm4_recording_t {
  // Temporary native bytes, with no execution or lifetime policy.
  iree_hal_amd_gpu_recording_t storage;
  // Invocation value containing the selected resident instance's base address.
  uint32_t instance_value;
  // Byte offset of the current native block within the command section.
  uint32_t block_offset;
  // First block's complete native extent, established when that block closes.
  uint32_t entry_word_count;
  // Host-side predecessor edge awaiting this block's final native extent.
  uint32_t* pending_branch;
} iree_hal_amd_gpu_pm4_recording_t;

void iree_hal_amd_gpu_pm4_recording_initialize(
    iree_allocator_t host_allocator, uint32_t instance_value,
    iree_hal_amd_gpu_pm4_recording_t* out_recording);

// Appends an indivisible sequence of complete native packets. The caller
// initializes every word before sealing. A sequence that does not fit closes
// the current block with a tail call and starts its successor. Each sequence
// must itself fit a native block including its aligned terminal edge.
iree_status_t iree_hal_amd_gpu_pm4_recording_commands(
    iree_hal_amd_gpu_pm4_recording_t* recording, uint32_t word_count,
    uint32_t* out_offset, uint32_t** out_words);

// Closes the final native block and resolves every incoming block extent.
// A USER program returns to its primary ring. A KERNEL program chains to the
// invocation's 32-DWORD continuation, supplied by continuation_value. After
// this succeeds, entry_word_count describes the first block for submission.
iree_status_t iree_hal_amd_gpu_pm4_recording_end(
    iree_hal_amd_gpu_pm4_recording_t* recording, uint32_t continuation_value,
    bool chain);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_RECORDING_H_
