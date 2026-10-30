// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_TIMESTAMP_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_TIMESTAMP_H_

#include "iree/hal/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/queue/recording.h"

#ifdef __cplusplus
extern "C" {
#endif

// Recording-only destinations of posted MEC clock writes. A confirmed MEC
// write drains the entire cohort; shader execution dependencies do not. Native
// programs retain only the resulting packets and invocation-private scratch.
typedef struct iree_hal_amd_gpu_timestamp_writes_t {
  // Borrowed exact eight-byte destinations since the last write drain.
  iree_hal_buffer_ref_t* pending;
  // Number of pending destinations.
  uint32_t count;
  // Allocated destination capacity, reused across recording boundaries.
  uint32_t capacity;
  // Private-data offset of a DWORD owned by each invocation, valid if capacity.
  uint32_t scratch_offset;
} iree_hal_amd_gpu_timestamp_writes_t;

// Captures one destination and lazily reserves invocation-owned drain scratch.
// The recorder retains the buffer through its ordinary command inputs.
iree_status_t iree_hal_amd_gpu_timestamp_writes_append(
    iree_hal_amd_gpu_recording_t* recording, iree_hal_buffer_ref_t target,
    iree_hal_amd_gpu_timestamp_writes_t* writes);

// Releases recording-only destinations. The native scratch remains in the
// sealed program and follows its existing per-invocation ownership.
void iree_hal_amd_gpu_timestamp_writes_deinitialize(
    iree_allocator_t allocator, iree_hal_amd_gpu_timestamp_writes_t* writes);

// Whether this payload dependency can consume an outstanding clock write.
// Same-slot/allocation ranges preserve byte precision. Distinct binding or
// imported-buffer identities may alias and conservatively require delivery.
bool iree_hal_amd_gpu_timestamp_writes_overlap(
    const iree_hal_amd_gpu_timestamp_writes_t* writes,
    const iree_hal_barrier_t* barrier);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_TIMESTAMP_H_
