// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BARRIER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BARRIER_H_

#include "iree/base/api.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/drivers/amd/gpu/target/identity.h"
#include "iree/hal/queue.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable cache policy for an admitted MEC GCR command format. ISA selects
// cache fields, independently of the native queue's packet language. Native
// family admission establishes support for ACQUIRE_MEM GCR and execution joins.
typedef struct iree_hal_amd_gpu_pm4_barrier_profile_t {
  // Lower-cache writeback/invalidation for local shader dependencies.
  uint32_t shader;
  // Forward lower-cache and L2 writeback to the system coherence domain.
  uint32_t release;
  // Forward writeback/invalidation preserving unrelated dirty cache lines.
  uint32_t acquire;
  // CP accesses bypass shader L2 and require maintenance when crossing clients.
  bool command_processor_bypasses_l2;
} iree_hal_amd_gpu_pm4_barrier_profile_t;

// Selects a supported physical ISA's cache layout during family construction.
// This neither discovers native capabilities nor creates queue resources.
iree_status_t iree_hal_amd_gpu_pm4_barrier_profile_initialize(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_barrier_profile_t* out_profile);

enum iree_hal_amd_gpu_pm4_barrier_execution_bits_e {
  IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE = 1u << 0,
  IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA = 1u << 1,
};

// Captured, whole-cache boundary consumed by direct submission and recording.
// Contains no borrowed descriptor, buffer, recipe, or topology pointer. Zero
// requests no optional payload work. Mandatory instruction publication and
// control-memory ordering belong to their owners and remain independent.
typedef struct iree_hal_amd_gpu_pm4_barrier_t {
  // Target-specific ACQUIRE_MEM GCR fields, excluding instruction invalidation.
  uint32_t cache_control;
  // Local engines that must complete before the cache action or later commands.
  uint32_t execution;
} iree_hal_amd_gpu_pm4_barrier_t;

// Captures a validated boundary without allocation, resource retention, or
// native queries. NULL selects conservative system release/acquire; an empty
// list explicitly elides optional barriers. Range effects are promoted globally
// and need no copied recipe or buffer. Public HAL validation has already
// checked effect coverage, queue executors, and descriptor storage. The same
// function applies at either position around an operation: position never
// selects the direction of a visibility action.
iree_hal_amd_gpu_pm4_barrier_t iree_hal_amd_gpu_pm4_barrier_prepare(
    const iree_hal_amd_gpu_pm4_barrier_profile_t* profile,
    const iree_hal_barrier_list_t* barriers);

// DWORD count for ordinary emission, including independent execution joins.
static inline uint32_t iree_hal_amd_gpu_pm4_barrier_word_count(
    iree_hal_amd_gpu_pm4_barrier_t barrier) {
  return ((barrier.execution & IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_COMPUTE)
              ? 2u
              : 0u) +
         ((barrier.execution & IREE_HAL_AMD_GPU_PM4_BARRIER_EXECUTION_DMA)
              ? 7u
              : 0u) +
         (barrier.cache_control ? 8u : 0u);
}

// Emits local execution joins followed by a synchronous cache action. The
// caller owns sufficient command storage and has established external waits.
// Native instruction publication can be combined by ORing GLI_INV=1 into a
// local copy's cache_control immediately before this ordinary emission.
uint32_t* iree_hal_amd_gpu_pm4_barrier_emit(
    uint32_t* words, iree_hal_amd_gpu_pm4_barrier_t barrier);

// Prepares an actual completion tail: emits the seven-DWORD DMA join if needed
// and converts payload cache actions to unshifted RELEASE_MEM fields. The
// caller then emits its existing RELEASE_MEM with |out_cache_control|, which
// supplies compute completion and orders cache work before the completion
// write. This is not an ordinary barrier: subsequent command processing must
// not depend on the asynchronous release without its own wait. Consumes a
// prepared payload boundary without added instruction invalidation.
uint32_t* iree_hal_amd_gpu_pm4_barrier_emit_tail(
    uint32_t* words, iree_hal_amd_gpu_pm4_barrier_t barrier,
    uint32_t* out_cache_control);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_BARRIER_H_
