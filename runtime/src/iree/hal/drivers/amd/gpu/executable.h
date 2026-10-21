// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_EXECUTABLE_H_

#include "iree/hal/drivers/amd/fabric/asan.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/code.h"
#include "iree/hal/drivers/amd/gpu/feedback.h"
#include "iree/hal/drivers/amd/gpu/kernarg.h"
#include "iree/hal/drivers/amd/gpu/profile.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/code.h"
#include "iree/hal/drivers/amd/gpu/queue/aql/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/tsan.h"

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
  // at native access ordinal zero. Host visibility is optional; placement is
  // always device-local. The loader never substitutes host backing.
  const iree_hal_amd_fabric_memory_options_t* memory;
  // Device-owned instruction publication used by AQL, absent for PM4.
  iree_hal_amd_gpu_aql_code_t* aql_code;
  // Upload resources borrowed through image construction only.
  struct {
    // Host-visible initialization backing, used only without a local host view.
    const iree_hal_amd_fabric_memory_options_t* memory;
    // Existing transfer owner with live publication and completion sources.
    iree_hal_amd_gpu_sdma_queue_t* queue;
    // Native family accepting the upload copies.
    uint32_t family_ordinal;
  } upload;
  // Group-local placement guarantee established with the native memory policy.
  iree_hal_pool_placement_t placement;
  // Borrowed device capacity notification, live through global buffer uses.
  iree_async_notification_t* notification;
  // Borrowed sealed-group completion tracker for pools backed by globals.
  iree_async_frontier_tracker_t* frontier_tracker;
  // Device-owned shared report channel, or NULL when feedback is disabled.
  iree_hal_amd_gpu_feedback_t* feedback;
  // Optional shadow namespace used by code-object globals and data ranges.
  iree_hal_amd_fabric_asan_t* asan;
  // Number of provisioned queue shadow records; zero when TSAN is disabled.
  uint32_t tsan_count;
  // Borrowed device-owned queue records, consumed during image placement.
  const iree_hal_amd_gpu_tsan_t* tsan;
} iree_hal_amd_gpu_executable_options_t;

// Copies the public artifact, admits its image, and loads native backing for
// one native compute family. The family belongs to a sealed device group;
// context and family remain borrowed through final native cleanup. Other option
// storage is consumed during construction. Success returns one owning HAL
// resource reference; failure leaves out_executable NULL after joining
// rollback. Final release retires host metadata inline and submits the native
// backing batch using embedded release storage. The context joins that batch
// before teardown.
//
// Native services named by launch/argument metadata remain requirements for
// dispatch admission. Code upload flushes host writes. AQL loading completes
// instruction-cache invalidation; PM4 establishes it before shader use.
// Module lifecycle kernels, required external linkage, and unprovisioned
// runtime configuration globals are rejected because loading alone cannot
// provide their required services. Global buffers follow HAL's executable-owned
// borrow contract and never extend the executable's lifetime implicitly.
iree_status_t iree_hal_amd_gpu_executable_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_executable_options_t* options,
    const iree_hal_executable_load_params_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Resolves an external function token into executable-owned immutable state.
// Exact-family compatibility is established at the public dispatch boundary.
iree_status_t iree_hal_amd_gpu_executable_lookup_pm4(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    const iree_hal_amd_gpu_pm4_function_t** out_function);

// Selects the prepared placement for a provisioned queue in this exact family.
// The owning queue supplies its construction ordinal; only the external
// function token requires validation. Ordinary images share one placement.
iree_status_t iree_hal_amd_gpu_executable_lookup_pm4_for_queue(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    uint32_t queue_ordinal,
    const iree_hal_amd_gpu_pm4_function_t** out_function);

// Selects a placed AMDHSA descriptor for an AQL queue. Exact-family protocol
// compatibility is established at the public dispatch boundary; this validates
// only the caller's function token. No PM4 register state is allocated or read.
iree_status_t iree_hal_amd_gpu_executable_lookup_aql_for_queue(
    iree_hal_executable_t* executable, iree_hal_executable_function_t token,
    uint32_t queue_ordinal,
    const iree_hal_amd_gpu_aql_function_t** out_function);

// Borrows queue-indexed image bases for recording native address relocations.
// NULL means the image is shared. The recording retains the executable through
// every invocation, so no per-dispatch copy of this table is necessary.
const uint64_t* iree_hal_amd_gpu_executable_queue_image_addresses(
    iree_hal_executable_t* executable);

// Captures admitted launch attribution and lazily prepares host-only reflection
// on the recording/submission caller. The result borrows executable metadata;
// recorded and completed profile records retain it without keeping native code
// allocations alive. The caller supplies command-buffer identity when needed.
iree_status_t iree_hal_amd_gpu_executable_profile_dispatch(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_amd_gpu_dispatch_geometry_t* geometry,
    iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_profile_dispatch_t* out_dispatch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_EXECUTABLE_H_
