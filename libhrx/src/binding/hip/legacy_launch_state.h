// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_BINDING_HIP_LEGACY_LAUNCH_STATE_H_
#define IREE_EXPERIMENTAL_STREAMING_BINDING_HIP_LEGACY_LAUNCH_STATE_H_

#include <stddef.h>
#include <stdint.h>

#include "binding/hip/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// One configured legacy launch removed from the current thread's stack.
// The frame owns |argument_data| until it is deinitialized.
typedef struct iree_hip_legacy_launch_frame_t {
  // Grid dimensions in workgroups.
  dim3 grid_dimension;
  // Block dimensions in threads.
  dim3 block_dimension;
  // Dynamic shared memory requested for the launch, in bytes.
  size_t shared_memory_bytes;
  // Stream on which the launch is to be enqueued.
  hipStream_t stream;
  // Owned native-ABI argument storage assembled by hipSetupArgument.
  uint8_t* argument_data;
  // Number of initialized bytes in |argument_data|.
  size_t argument_length;
  // Allocated capacity of |argument_data|, in bytes.
  size_t argument_capacity;
} iree_hip_legacy_launch_frame_t;

// Pushes a call configuration onto the current thread's legacy launch stack.
hipError_t iree_hip_legacy_launch_state_push(dim3 grid_dimension,
                                             dim3 block_dimension,
                                             size_t shared_memory_bytes,
                                             hipStream_t stream);

// Writes an argument into the top frame at its native kernarg byte offset.
hipError_t iree_hip_legacy_launch_state_setup_argument(const void* argument,
                                                       size_t size,
                                                       size_t offset);

// Removes the top frame and transfers its argument storage to |out_frame|.
// Returns hipErrorMissingConfiguration when the current thread has no frame.
hipError_t iree_hip_legacy_launch_state_pop(
    iree_hip_legacy_launch_frame_t* out_frame);

// Releases argument storage transferred by iree_hip_legacy_launch_state_pop.
void iree_hip_legacy_launch_frame_deinitialize(
    iree_hip_legacy_launch_frame_t* frame);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_BINDING_HIP_LEGACY_LAUNCH_STATE_H_
