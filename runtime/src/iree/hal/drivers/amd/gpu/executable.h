// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_EXECUTABLE_H_

#include "iree/hal/drivers/amd/fabric/asan.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/feedback.h"
#include "iree/hal/drivers/amd/gpu/kernarg.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Captured construction facts for one exact executable family. Selection is
// cold device/family work, independent of the image's serialized frontend.
typedef struct iree_hal_amd_gpu_executable_options_t {
  // Qualified physical compiler identity, including active feature modes.
  const iree_hal_amd_gpu_target_identity_t* target;
  // Native family ordinal; its command protocol is independent of target ISA.
  uint32_t native_family_ordinal;
  // Qualified owned backing with read/write/execute access for context->device
  // at native access ordinal zero, plus a writable persistent host view. The
  // selected scope determines physical placement; this loader picks no pool.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Group-local placement guarantee established with the native memory policy.
  iree_hal_pool_placement_t placement;
  // Borrowed device service for native allocation and deferred final cleanup.
  // The parent joins this service before releasing its native context.
  iree_hal_memory_maintenance_t* maintenance;
  // Device-owned shared report channel, or NULL when feedback is disabled.
  iree_hal_amd_gpu_feedback_t* feedback;
  // Optional shadow namespace used by code-object globals and data ranges.
  iree_hal_amd_fabric_asan_t* asan;
} iree_hal_amd_gpu_executable_options_t;

// Copies the public artifact, admits its image, and loads native backing for
// one PM4 family. The family belongs to a sealed device group; context and
// family remain borrowed through final service cleanup. Other option storage is
// consumed during construction. Success returns one owning HAL resource
// reference; failure leaves out_executable NULL after joining rollback. Final
// release enqueues an embedded cleanup entry on the shared device service.
//
// Native services named by launch/argument metadata remain requirements for
// dispatch admission. Code upload flushes host writes; each queue establishes
// instruction-cache visibility before shader use. Module lifecycle kernels,
// required external linkage, and unprovisioned runtime configuration globals
// are rejected because loading alone cannot provide their required services.
// Global buffers follow HAL's executable-owned borrow contract and never extend
// the executable's lifetime implicitly.
iree_status_t iree_hal_amd_gpu_executable_create(
    const iree_hal_queue_family_t* family,
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_executable_options_t* options,
    const iree_hal_executable_load_params_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Resolves an external function token into executable-owned immutable state.
// Exact-family compatibility is established at the public dispatch boundary.
iree_status_t iree_hal_amd_gpu_executable_lookup_pm4(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    const iree_hal_amd_gpu_pm4_function_t** out_function);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_EXECUTABLE_H_
