// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_
#define IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold storage budget for the serialized workgroup-race shadow arena.
typedef struct iree_hal_amd_gpu_tsan_options_t {
  // Bytes per queue; instrumented dispatches reuse the complete arena. The
  // compiler's unsigned 32-bit byte offsets permit at most 2^32 bytes.
  iree_device_size_t shadow_size;
} iree_hal_amd_gpu_tsan_options_t;

// Initializes a 512 MiB arena per queue.
void iree_hal_amd_gpu_tsan_options_initialize(
    iree_hal_amd_gpu_tsan_options_t* out_options);

// Runtime-published TSAN configuration read by instrumented device code.
//
// Queue-scoped executable variants publish one shared shadow arena. The
// compiler sizes each workgroup record from the final dispatch LDS size.
// Instrumented dispatches execute serially within the queue; dispatch IDs
// distinguish reused generations without host mutation of the arena.
typedef struct iree_hal_amd_gpu_tsan_config_t {
  // Size of this record in bytes for forward-compatible parsing.
  uint32_t record_length;
  // ABI version of this record layout.
  uint32_t abi_version;
  // Enabled (bit 0) and queue-shadow (bit 1) runtime facilities.
  uint32_t flags;
  // Reserved for future TSAN configuration. Must be zero.
  uint32_t reserved0;
  // Device-visible base of the queue shadow arena.
  uint64_t shadow_base;
  // Total bytes in the queue shadow arena.
  uint64_t shadow_size;
} iree_hal_amd_gpu_tsan_config_t;
static_assert(sizeof(iree_hal_amd_gpu_tsan_config_t) == 32,
              "TSAN config size is part of the device ABI");
static_assert(offsetof(iree_hal_amd_gpu_tsan_config_t, shadow_base) == 16,
              "TSAN config shadow fields must follow the fixed header");

// One queue's native shadow owner, retained by its parent device until every
// queue and executable cleanup has joined. Ordinary devices allocate no state.
typedef struct iree_hal_amd_gpu_tsan_t {
  // Native backing for the mutable queue shadow arena.
  iree_hal_amd_fabric_memory_t memory;
  // Immutable shader configuration copied into this queue's executable variant.
  iree_hal_amd_gpu_tsan_config_t config;
} iree_hal_amd_gpu_tsan_t;

// Validates the public byte budget and prepares the shader configuration
// without allocating. Addresses remain zero until initialize establishes
// backing.
iree_status_t iree_hal_amd_gpu_tsan_calculate_config(
    const iree_hal_amd_gpu_tsan_options_t* options,
    iree_hal_amd_gpu_tsan_config_t* out_config);

// Allocates and publishes one queue's shadow on the native allocation owner.
// Uses host-visible device-local or unified backing for initialization. Native
// accesses use this context's GPU address domain. Failure leaves
// deinitializable state; the parent joins GPU uses before deinitialization.
iree_status_t iree_hal_amd_gpu_tsan_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const iree_hal_amd_gpu_tsan_options_t* options,
    iree_hal_amd_gpu_tsan_t* out_tsan);

// Releases native backing after queue execution and executable cleanup join.
void iree_hal_amd_gpu_tsan_deinitialize(
    const iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_tsan_t* tsan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_TSAN_H_
