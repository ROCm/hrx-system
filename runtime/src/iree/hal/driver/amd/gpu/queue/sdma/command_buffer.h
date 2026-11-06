// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_COMMAND_BUFFER_H_

#include "iree/hal/driver/amd/gpu/queue/inputs.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/barrier.h"
#include "iree/hal/driver/amd/gpu/queue/sdma/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a TRANSFER recorder for the family's selected native carrier. It
// owns immutable host templates and captured update bytes; invocation-private
// native command/control storage belongs to the executing queue. No compute
// program, device allocation, or queue acquisition occurs while recording.
iree_status_t iree_hal_amd_gpu_sdma_command_buffer_create(
    const iree_hal_queue_family_t* family,
    amdf_queue_format_features_t features,
    iree_hal_amd_gpu_sdma_barrier_profile_t barriers,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_buffer);

// Sealed facts borrowed through the recording's retained lifetime.
const iree_hal_amd_gpu_sdma_program_t*
iree_hal_amd_gpu_sdma_command_buffer_program(iree_hal_command_buffer_t* buffer);
iree_host_size_t iree_hal_amd_gpu_sdma_command_buffer_invocation_size(
    iree_hal_command_buffer_t* buffer);
iree_host_size_t iree_hal_amd_gpu_sdma_command_buffer_operation_count(
    iree_hal_command_buffer_t* buffer);

typedef struct iree_hal_amd_gpu_sdma_command_invocation_t {
  // Retained recording dominating its immutable program and direct resources.
  iree_hal_command_buffer_t* command_buffer;
  // Captured binding table in submission-owned host storage.
  iree_hal_amd_gpu_command_binding_t* bindings;
  // Dense resolved values; input zero is this invocation's private GPU data.
  uint64_t* values;
  // Binding buffers borrow caller lifetime; descriptors are always captured.
  bool borrow_bindings;
} iree_hal_amd_gpu_sdma_command_invocation_t;

// Captures a publicly validated table into invocation_size bytes of storage.
// Failure leaves a deinitializable prefix. Private GPU data is 64-byte aligned
// and initialized from program.data before emission; each invocation owns its
// own data until native retirement, including posted timestamp uses.
iree_status_t iree_hal_amd_gpu_sdma_command_invocation_initialize(
    iree_hal_command_buffer_t* buffer, iree_hal_buffer_binding_table_t bindings,
    iree_hal_queue_execute_flags_t flags, void* storage,
    iree_hal_amd_gpu_sdma_command_invocation_t* out_invocation);
void iree_hal_amd_gpu_sdma_command_invocation_deinitialize(
    iree_hal_amd_gpu_sdma_command_invocation_t* invocation);

// Resolves each unique native input after dependencies commit its backing.
// This neither replays commands nor touches tensor payloads.
iree_status_t iree_hal_amd_gpu_sdma_command_invocation_resolve(
    const iree_hal_amd_gpu_sdma_command_invocation_t* invocation,
    uint32_t queue_ordinal, uint64_t private_address);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SDMA_COMMAND_BUFFER_H_
