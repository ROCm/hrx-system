// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_H_

#include "iree/hal/drivers/amd/xdna/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// One admitted function and its prepared native backing. The owning executable
// dominates this borrowed pointer. Only its queue's execution owner may bind
// it.
typedef struct iree_hal_amd_xdna_function_t iree_hal_amd_xdna_function_t;

// Copies the artifact, admits its image, and prepares native storage. The
// family and context are borrowed from the parent device for the executable
// lifetime.
iree_status_t iree_hal_amd_xdna_executable_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    const iree_hal_executable_load_params_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Resolves a public token and captures prepared native bindings. The caller
// owns |out_bindings| with space for bindings.count rows and retains the
// executable and logical buffers until checked completion. This does not mutate
// backing.
iree_status_t iree_hal_amd_xdna_executable_resolve(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_hal_buffer_ref_list_t bindings,
    iree_hal_amd_xdna_executable_binding_t* out_bindings,
    iree_hal_amd_xdna_function_t** out_function);

// Patches and publishes invocation zero after prior native users have retired.
// |bindings| was captured by executable_resolve. The command borrows function
// storage through checked native completion.
iree_status_t iree_hal_amd_xdna_function_prepare(
    iree_hal_amd_xdna_function_t* function,
    const iree_hal_amd_xdna_executable_binding_t* bindings,
    amdf_xdna_kernel_command_t* out_command);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_H_
