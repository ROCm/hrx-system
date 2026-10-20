// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PROGRAM_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PROGRAM_H_

#include "iree/hal/drivers/amd/gpu/queue/aql/indirect.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/materialize.h"
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
  // Native descriptor for resident instance initialization and copy/fill.
  uint64_t blit;
  // Native descriptor for independent complete-DWORD replacements.
  uint64_t fixup;
  // Native descriptor for write-only packet materialization.
  uint64_t materialize;
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PROGRAM_H_
