// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PROGRAM_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PROGRAM_H_

#include "iree/hal/drivers/amd/gpu/code.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/indirect.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/materialize.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/ring.h"
#include "iree/hal/drivers/amd/gpu/queue/blit.h"
#include "iree/hal/drivers/amd/gpu/queue/program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Concrete AQL executor. Resident ownership is shared with other protocols;
// native descriptor launches and ring materialization belong to AQL.
typedef struct iree_hal_amd_gpu_aql_program_pool_t {
  // Shared placement, causal instance ownership, and explicit trim registry.
  iree_hal_amd_gpu_program_pool_t storage;
  // One immutable descriptor/instruction allocation for all builtin kernels.
  iree_hal_amd_fabric_memory_t code;
  // Temporary initialization bytes, published before the device is exposed.
  iree_hal_amd_gpu_code_upload_t* code_upload;
  // Native descriptor for resident instance initialization and copy/fill.
  uint64_t blit;
  // Native descriptor for independent complete-DWORD replacements.
  uint64_t fixup;
  // Native descriptor for write-only packet materialization.
  uint64_t materialize;
  // Native descriptor for tagged completion operands and private samples.
  uint64_t materialize_profile;
  // Native descriptor for execution-time grid/hidden-count publication.
  uint64_t indirect;
} iree_hal_amd_gpu_aql_program_pool_t;

// Cold construction before queue creation. A failure leaves a deinitializable
// partial owner. Every queue and resident recording is joined before teardown.
iree_status_t iree_hal_amd_gpu_aql_program_pool_initialize(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    uint32_t queue_count, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_aql_program_pool_t* out_pool);
void iree_hal_amd_gpu_aql_program_pool_deinitialize(
    iree_hal_amd_gpu_aql_program_pool_t* pool);

// Initial invocation upload. Instrumented recordings extend materialize to
// materialize_profile_arguments_t; the dense uint64_t value table follows
// that complete argument record. Later chunks upload only their materializer
// arguments. The first native chunk releases the table independently of later
// work.
typedef struct iree_hal_amd_gpu_aql_program_arguments_t {
  // Optional initial resident copy, including the first instance's fixups.
  iree_hal_amd_gpu_blit_arguments_t copy;
  // Invocation-wide disjoint replacements, applied once before any payload.
  iree_hal_amd_gpu_fixup_arguments_t fixup;
  // First native packet block and its exact reserved ring slots.
  iree_hal_amd_gpu_aql_materialize_arguments_t materialize;
} iree_hal_amd_gpu_aql_program_arguments_t;

// Instrumented materialization extends the last argument record by 16 bytes
// before the value table. Ordinary invocation uploads retain their compact ABI.
enum iree_hal_amd_gpu_aql_program_flag_bits_e {
  IREE_HAL_AMD_GPU_AQL_PROGRAM_FIRST = 1u << 0,
  IREE_HAL_AMD_GPU_AQL_PROGRAM_INITIALIZE = 1u << 1,
  IREE_HAL_AMD_GPU_AQL_PROGRAM_PROFILE = 1u << 2,
};
typedef uint32_t iree_hal_amd_gpu_aql_program_flags_t;

// Counts host-owned prelude packets; block packets and the terminal barrier
// are separate. The unpublished instance remains reserved through emission.
uint32_t iree_hal_amd_gpu_aql_program_prefix_count(
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance, bool first);

// Emits one fully reserved block's host prelude. Resident templates are copied
// exclusively by the GPU. upload is coherent native argument storage and its
// value table is already written. The caller establishes ownership before
// emission and publishes a terminal completion after the block's slots.
void iree_hal_amd_gpu_aql_program_emit(
    const iree_hal_amd_gpu_aql_program_pool_t* pool,
    const iree_hal_amd_gpu_program_t* program,
    const iree_hal_amd_gpu_program_instance_t* instance, uint32_t block_offset,
    uint32_t packet_count, iree_hal_amd_gpu_aql_ring_t* ring, uint64_t position,
    uint64_t argument_address, uint64_t signal_address, void* upload,
    iree_hal_amd_gpu_aql_program_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PROGRAM_H_
