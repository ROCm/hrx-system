// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_CODE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_CODE_H_

#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_aql_code_t iree_hal_amd_gpu_aql_code_t;

// Owns one cold native queue, control page, and event for instruction-cache
// publication. No thread is created. Loading callers serialize only cache
// commands; memory allocation, code upload, and ordinary dispatch are separate.
// The device outlives this owner. Partial construction remains in out_code so
// the caller can join native ownership with free even when allocation fails.
iree_status_t iree_hal_amd_gpu_aql_code_allocate(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    const iree_hal_amd_fabric_memory_options_t* control,
    iree_allocator_t host_allocator, iree_hal_amd_gpu_aql_code_t** out_code);

// Invalidates cached instructions after the caller has published all code data.
// Waits on this loading caller for native completion, independently of workload
// queues. The aligned range belongs to the native code allocation. A failure
// stops further publication; unproven native use preserves memory's handles
// without handing their release to the failed executable's rollback.
iree_status_t iree_hal_amd_gpu_aql_code_publish(
    iree_hal_amd_gpu_aql_code_t* code, iree_hal_amd_fabric_memory_t* memory);

// Called after all loaders have returned. Success joins native notification
// tails and frees the owner; failure preserves it and its context. NULL is
// safe.
amdf_status_t iree_hal_amd_gpu_aql_code_free(iree_hal_amd_gpu_aql_code_t* code);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_CODE_H_
