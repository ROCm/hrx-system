// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_EXECUTABLE_H_
#define IREE_HAL_DRIVER_AMD_XDNA_EXECUTABLE_H_

#include "iree/hal/driver/amd/xdna/command_arena.h"
#include "iree/hal/driver/amd/xdna/executable_storage.h"
#include "iree/hal/driver/amd/xdna/memory.h"

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

// Resolves an external function token into immutable executable-owned state.
iree_status_t iree_hal_amd_xdna_executable_lookup(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    iree_hal_amd_xdna_function_t** out_function);

// Returns the admitted function's binding-table extent.
iree_host_size_t iree_hal_amd_xdna_function_binding_count(
    const iree_hal_amd_xdna_function_t* function);

// Queries whether an in-range binding ordinal names a resource.
bool iree_hal_amd_xdna_function_uses_binding(
    const iree_hal_amd_xdna_function_t* function, uint32_t ordinal);

// Validates an external direct reference for an in-range, used binding ordinal
// and captures its stable native slot without reading its address payload.
iree_status_t iree_hal_amd_xdna_function_capture_binding(
    iree_hal_amd_xdna_function_t* function,
    const iree_hal_queue_family_t* family, uint32_t ordinal,
    iree_hal_buffer_ref_t reference,
    iree_hal_amd_xdna_executable_binding_t* out_binding);

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

// Selects a measured command for an exclusively prepared invocation. First use
// allocates a command copy with fixed timer IDs; later uses neither allocate
// nor patch unless the ordinary invocation's bindings changed. Native retry
// retains the prepared command unchanged. An undeclared timer interval is
// UNAVAILABLE.
iree_status_t iree_hal_amd_xdna_invocation_prepare_measurement(
    iree_hal_amd_xdna_invocation_t* invocation,
    amdf_xdna_kernel_command_t* out_command);

// Borrowed attribution for an admitted invocation. Its executable dominates
// the name and must be retained when this identity outlives queue retirement.
typedef struct iree_hal_amd_xdna_invocation_info_t {
  // Owning executable, borrowed.
  iree_hal_executable_t* executable;
  // Process-local stable identity, independent of pointer reuse.
  uint64_t executable_id;
  // Dense executable function ordinal.
  uint32_t function_ordinal;
  // Borrowed function name from the admitted image.
  iree_string_view_t name;
} iree_hal_amd_xdna_invocation_info_t;

// Returns immutable attribution without reflection lookup or allocation.
iree_hal_amd_xdna_invocation_info_t iree_hal_amd_xdna_invocation_info(
    const iree_hal_amd_xdna_invocation_t* invocation);

// Identifies which of the invocation's two prepared forms was submitted.
bool iree_hal_amd_xdna_invocation_is_measured(
    const iree_hal_amd_xdna_invocation_t* invocation,
    const amdf_xdna_kernel_command_t* command);

// Returns exclusively owned storage after rejection or checked retirement.
// Safe concurrently with acquisition on another queue; performs no native work.
// A NULL invocation requires no action.
void iree_hal_amd_xdna_invocation_release(
    iree_hal_amd_xdna_invocation_t* invocation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_XDNA_EXECUTABLE_H_
