// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_COMMAND_BUFFER_H_

#include "iree/hal/drivers/amd/gpu/queue/aql/program.h"
#include "iree/hal/drivers/amd/gpu/queue/inputs.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable native packet extent. Indirect producer/target pairs never
// cross a block boundary. The publisher reads this descriptor, not packets.
typedef struct iree_hal_amd_gpu_aql_program_block_t {
  // Byte offset from resident instance zero, aligned to a native packet.
  uint32_t offset;
  // Complete 64-byte packets copied by one materializer launch.
  uint32_t packet_count;
} iree_hal_amd_gpu_aql_program_block_t;

typedef struct iree_hal_amd_gpu_aql_command_program_t {
  // Native ownership shared across protocols and retained by this recording.
  iree_hal_amd_gpu_program_t resident;
  // Immutable indexed native blocks, owned by the command buffer.
  iree_hal_amd_gpu_aql_program_block_t* blocks;
  // Number of complete blocks; zero is an empty recording.
  uint32_t block_count;
  // Unique input count including instance base at index zero.
  uint32_t value_count;
  // Public dispatches, independent of internal materializer/fixup dispatches.
  uint32_t dispatch_count;
} iree_hal_amd_gpu_aql_command_program_t;

// Seals resident templates and private arguments. maximum_packet_count is
// derived from the family's provisioned ring geometry, leaving publication
// space for waits, initialization, and a terminal point. The recording retains
// the device; UNRETAINED applies only to caller payload resources.
iree_status_t iree_hal_amd_gpu_aql_command_buffer_create(
    const iree_hal_queue_family_t* family,
    iree_hal_amd_gpu_aql_program_pool_t* pool,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity, uint32_t maximum_packet_count,
    uint32_t maximum_group_segment_byte_length, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_buffer);

// Returns immutable sealed execution facts; the caller retains command_buffer.
iree_hal_amd_gpu_aql_command_program_t*
iree_hal_amd_gpu_aql_command_buffer_program(
    iree_hal_command_buffer_t* command_buffer);

typedef struct iree_hal_amd_gpu_aql_command_invocation_t {
  // Pending suffix and nonterminal native pieces retain independent ownership.
  iree_atomic_ref_count_t ref_count;
  // Retained recording dominating its resident program and direct resources.
  iree_hal_command_buffer_t* command_buffer;
  // Borrowed sealed facts in command_buffer.
  iree_hal_amd_gpu_aql_command_program_t* program;
  // Copied/qualified public table in submission-owned storage.
  iree_hal_amd_gpu_command_binding_t* bindings;
  // Exclusive unpublished instance, then causally reusable after final publish.
  iree_hal_amd_gpu_program_instance_t* instance;
  // Next unpublished block; only the publisher accesses this cursor.
  uint32_t block_ordinal;
  // Final acceptance installed this invocation's native writer coordinate.
  bool published;
  // Caller borrows buffer lifetime; table entries are always copied.
  bool borrow_bindings;
} iree_hal_amd_gpu_aql_command_invocation_t;

// Captures validated table entries in binding_count slots. A failure leaves a
// deinitializable retained prefix. The caller owns the initial suffix
// reference.
iree_status_t iree_hal_amd_gpu_aql_command_invocation_initialize(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings, void* storage,
    iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_aql_command_invocation_t* out_invocation);

// Called after all accepted native pieces join; returns the instance and
// payloads without waiting. Cancellation preserves prior causal coordinates;
// published use is released.
void iree_hal_amd_gpu_aql_command_invocation_deinitialize(
    iree_hal_amd_gpu_aql_command_invocation_t* invocation);

// Writes value_count uint64_t values once after address commitment. Cost is
// proportional to unique inputs, independent of recorded packet/fixup counts.
iree_status_t iree_hal_amd_gpu_aql_command_invocation_write(
    const iree_hal_amd_gpu_aql_command_invocation_t* invocation,
    uint32_t queue_ordinal, uint64_t* values);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_COMMAND_BUFFER_H_
