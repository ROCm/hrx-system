// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_H_

#include "iree/hal/drivers/amd/xdna/command_arena.h"
#include "iree/hal/drivers/amd/xdna/executable_storage.h"
#include "iree/hal/drivers/amd/xdna/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// One admitted function and its reusable invocation storage. The owning
// executable dominates this borrowed pointer.
typedef struct iree_hal_amd_xdna_function_t iree_hal_amd_xdna_function_t;

// Exclusive mutable backing for one native invocation. The owning executable
// stays retained until the invocation is returned after checked retirement.
typedef struct iree_hal_amd_xdna_invocation_t iree_hal_amd_xdna_invocation_t;

// Copies the artifact, admits its image, and prepares native storage. The
// family and context are borrowed from the parent device for the executable
// lifetime.
iree_status_t iree_hal_amd_xdna_executable_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    iree_hal_amd_xdna_command_arena_t* command_arena,
    const iree_hal_executable_load_params_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Resolves a public token and captures stable native binding slots. The caller
// owns |out_bindings| with space for bindings.count rows and retains the
// executable and logical buffers until checked completion. This neither reads
// the queue-published slot payload nor mutates executable backing.
iree_status_t iree_hal_amd_xdna_executable_resolve(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amd_xdna_executable_binding_t* out_bindings,
    iree_hal_amd_xdna_function_t** out_function);

// After queue prerequisites resolve, loads each captured native slot once,
// acquires exclusive invocation storage, patches the addresses, and publishes
// the changed ranges. Cold growth creates private mutable backing while sharing
// the function's immutable allocation closure. Warm reuse performs no
// allocation. A native publication retry reuses the prepared invocation and
// does not call this again. On failure no invocation is returned. On success
// the command borrows |out_invocation| until the caller returns it after
// checked native retirement.
iree_status_t iree_hal_amd_xdna_function_prepare(
    iree_hal_amd_xdna_function_t* function,
    iree_hal_amd_xdna_executable_binding_t* bindings,
    iree_hal_amd_xdna_invocation_t** out_invocation,
    amdf_xdna_kernel_command_t* out_command);

// Returns exclusively owned storage after rejection or checked retirement.
// Safe concurrently with acquisition on another queue; performs no native work.
// A NULL invocation requires no action.
void iree_hal_amd_xdna_invocation_release(
    iree_hal_amd_xdna_invocation_t* invocation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_H_
