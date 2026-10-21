// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_COMMAND_BUFFER_H_

#include "iree/hal/drivers/amd/gpu/profile.h"
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
  // Immutable dispatch subrange for independently retiring profile samples.
  struct {
    // First public dispatch in the recording.
    uint32_t first;
    // Public dispatches in this block, excluding builtin kernels.
    uint32_t count;
  } dispatches;
} iree_hal_amd_gpu_aql_program_block_t;

// One native dispatch and bounded private data. The byte template and fixup
// table follow this descriptor in the same host allocation. Each invocation
// copies and patches its own argument-ring storage; no native instance exists.
typedef struct iree_hal_amd_gpu_aql_host_template_t {
  // Complete packet/private-data extent copied into each invocation.
  uint32_t byte_length;
  // Required power-of-two alignment of the invocation's upload address.
  uint32_t alignment;
  // Independent DWORD replacements applied after resolving unique inputs.
  uint32_t fixup_count;
  // Immutable fixup-table offset, also the invocation's dense-value offset.
  uint32_t fixup_offset;
} iree_hal_amd_gpu_aql_host_template_t;

typedef struct iree_hal_amd_gpu_aql_command_program_t {
  // Native growth owns these facts independently of the recording's resources.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this program and its immutable block/template metadata.
  iree_allocator_t host_allocator;
  // Native ownership for resident recordings; zero for bounded host templates.
  iree_hal_amd_gpu_program_t resident;
  // Immutable indexed native blocks, owned by this program.
  iree_hal_amd_gpu_aql_program_block_t* blocks;
  // Optional host-only attribution; its presence selects tagged templates.
  iree_hal_amd_gpu_profile_commands_t* profile;
  // Number of complete blocks; zero is an empty recording.
  uint32_t block_count;
  // Unique input count including instance base at index zero.
  uint32_t value_count;
  // Public dispatches, independent of internal materializer/fixup dispatches.
  uint32_t dispatch_count;
  // Public encoded operations, independent of native packet count or folding.
  uint32_t operation_count;
  // Optional bounded native template, owned instead of resident storage.
  iree_hal_amd_gpu_aql_host_template_t* host_template;
} iree_hal_amd_gpu_aql_command_program_t;

// Seals native templates and private arguments. maximum_packet_count is
// derived from the family's provisioned ring geometry, leaving publication
// space for waits, initialization, and a terminal point. The recording and its
// independently retained program borrow the device; UNRETAINED applies only to
// caller payload resources.
iree_status_t iree_hal_amd_gpu_aql_command_buffer_create(
    const iree_hal_queue_family_t* family,
    iree_hal_amd_gpu_aql_program_pool_t* pool,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_host_size_t binding_capacity, uint32_t maximum_packet_count,
    uint32_t maximum_group_segment_byte_length, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_buffer);

// Retains/releases native program facts independently of caller resources.
// The recording owns one reference; cold clone growth may outlive invocations.
void iree_hal_amd_gpu_aql_command_program_retain(
    iree_hal_amd_gpu_aql_command_program_t* program);
void iree_hal_amd_gpu_aql_command_program_release(
    iree_hal_amd_gpu_aql_command_program_t* program);

// Returns borrowed sealed execution facts; the caller retains command_buffer.
iree_hal_amd_gpu_aql_command_program_t*
iree_hal_amd_gpu_aql_command_buffer_program(
    iree_hal_command_buffer_t* command_buffer);

typedef struct iree_hal_amd_gpu_aql_command_invocation_t {
  // Pending suffix and nonterminal native pieces retain independent ownership.
  iree_atomic_ref_count_t ref_count;
  // Retained recording dominating its resident program and direct resources.
  iree_hal_command_buffer_t* command_buffer;
  // Borrowed sealed facts retained by command_buffer.
  iree_hal_amd_gpu_aql_command_program_t* program;
  // Copied/qualified public table in submission-owned storage.
  iree_hal_amd_gpu_command_binding_t* bindings;
  // Exclusive unpublished resident instance, then causally reusable after final
  // publication. NULL for a bounded host template.
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

// Resolves unique input values once after upload address commitment. A bounded
// host template is copied and patched into that invocation-private upload;
// resident programs receive a dense value table for native fixup execution.
iree_status_t iree_hal_amd_gpu_aql_command_invocation_write(
    const iree_hal_amd_gpu_aql_command_invocation_t* invocation,
    uint32_t queue_ordinal, uint64_t upload_address, void* upload);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_COMMAND_BUFFER_H_
