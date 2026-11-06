// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_SCRATCH_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_SCRATCH_H_

#include "iree/hal/driver/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable geometry of command-owned architected private storage. An absent
// native capability leaves all fields zero and rejects nonempty scratch.
typedef struct iree_hal_amd_gpu_pm4_scratch_profile_t {
  // Native COMPUTE_TMPRING_SIZE DWORD address.
  uint32_t size_register;
  // Maximum per-wave bytes representable by the native register.
  uint32_t maximum_wave_byte_length;
  // Full hardware scratch slot count per shader engine.
  uint32_t waves_per_engine;
  // Number of equal contiguous shader-engine partitions.
  uint32_t engine_count;
} iree_hal_amd_gpu_pm4_scratch_profile_t;

// Captures physical geometry before exposing queues or recorders. No native
// allocation occurs. Unsupported capability is a valid empty profile; malformed
// native geometry is reported before accepting shader work.
iree_status_t iree_hal_amd_gpu_pm4_scratch_profile_initialize(
    const amdf_gpu_endpoint_info_t* target,
    amdf_queue_format_features_t features,
    iree_hal_amd_gpu_pm4_scratch_profile_t* out_profile);

// One physical generation, owned by its queue or a proven retirement record.
typedef struct iree_hal_amd_gpu_pm4_scratch_storage_t {
  // Replaced generation sharing this owner's retirement boundary.
  struct iree_hal_amd_gpu_pm4_scratch_storage_t* next;
  // Device-readable/writable physical backing; no host mapping is required.
  iree_hal_amd_fabric_memory_t memory;
  // Native device address, initialized after acquisition succeeds.
  uint64_t address;
  // Last accepted queue epoch using this generation, or zero before use.
  uint64_t last_use;
  // Bytes per hardware wave slot, a multiple of 256.
  uint32_t wave_byte_length;
  // Release storage secured before native acquisition, never at retirement.
  iree_hal_amd_gpu_memory_release_t release;
  // Host allocator owning this record through native release completion.
  iree_allocator_t host_allocator;
} iree_hal_amd_gpu_pm4_scratch_storage_t;

iree_status_t iree_hal_amd_gpu_pm4_scratch_storage_prepare(
    uint32_t wave_byte_length, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_scratch_storage_t** out_storage);
iree_status_t iree_hal_amd_gpu_pm4_scratch_storage_initialize(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_scratch_storage_t* storage);

// Releases a proven-retired list using each record's embedded native request.
// The caller has exclusive ownership; callbacks reclaim individual records.
void iree_hal_amd_gpu_pm4_scratch_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_scratch_storage_t* storage);

// Exclusive publisher state. No atomics or per-dispatch references are needed:
// a later native FIFO completion dominates every prior scratch generation.
typedef struct iree_hal_amd_gpu_pm4_scratch_t {
  // Physical slot and register geometry captured at device construction.
  iree_hal_amd_gpu_pm4_scratch_profile_t profile;
  // Current reusable generation, or NULL before first demand/after idle trim.
  iree_hal_amd_gpu_pm4_scratch_storage_t* current;
  // Replaced live generations awaiting transfer to an accepted retirement.
  iree_hal_amd_gpu_pm4_scratch_storage_t* retired;
  // Zero for unchanged USER state, seven for binding, nine for join+binding.
  uint32_t binding_word_count;
} iree_hal_amd_gpu_pm4_scratch_t;

// Consumes acquired capacity. Already-retired backing returns asynchronously;
// live predecessors remain owned until a later accepted FIFO completion.
void iree_hal_amd_gpu_pm4_scratch_install(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_scratch_storage_t* storage, uint64_t completed,
    iree_hal_amd_gpu_pm4_scratch_t* scratch);

// Returns exact prefix capacity; required is the admitted per-wave extent.
// KERNEL streams reestablish register state at each immutable entry.
static inline uint32_t iree_hal_amd_gpu_pm4_scratch_word_count(
    const iree_hal_amd_gpu_pm4_scratch_t* scratch, uint32_t required,
    bool kernel) {
  return !required                     ? 0
         : scratch->binding_word_count ? scratch->binding_word_count
         : kernel                      ? 7
                                       : 0;
}

// Emits the previously sized prefix without mutating accepted native state.
uint32_t* iree_hal_amd_gpu_pm4_scratch_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_scratch_t* scratch,
    uint32_t word_count);

// Commits accepted use and transfers replaced generations to this publication's
// retirement record. Zero required leaves current binding/use unchanged.
iree_hal_amd_gpu_pm4_scratch_storage_t* iree_hal_amd_gpu_pm4_scratch_publish(
    iree_hal_amd_gpu_pm4_scratch_t* scratch, uint64_t epoch, uint32_t required);

// Explicit trim returns only generations whose last use is proven complete.
// Native queue shutdown may pass UINT64_MAX after joining all hardware access.
void iree_hal_amd_gpu_pm4_scratch_trim(iree_hal_amd_gpu_context_t* context,
                                       uint64_t completed,
                                       iree_hal_amd_gpu_pm4_scratch_t* scratch);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_PM4_SCRATCH_H_
