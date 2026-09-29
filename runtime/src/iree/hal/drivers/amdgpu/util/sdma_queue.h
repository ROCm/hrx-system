// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_QUEUE_H_

#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/sdma_capabilities.h"
#include "iree/hal/drivers/amdgpu/util/sdma_ring.h"

#ifdef __cplusplus
extern "C" {
#endif

// The caller owns the HSA lifetime and selects the physical device. These
// borrowed references and callback storage must outlive the native queue.
typedef struct iree_hal_amdgpu_sdma_queue_params_t {
  const iree_hal_amdgpu_libhsa_t* libhsa;
  hsa_agent_t agent;
  iree_hal_amdgpu_gfxip_version_t gfxip_version;
  uint32_t capacity_bytes;
  uint32_t engine_id;  // UINT32_MAX lets ROCr select an available engine.
  hsa_amd_queue_priority_t priority;
  void (*error_callback)(hsa_status_t, hsa_queue_t*, void*);
  void* error_callback_data;
} iree_hal_amdgpu_sdma_queue_params_t;

typedef struct iree_hal_amdgpu_sdma_queue_t {
  const iree_hal_amdgpu_libhsa_t* libhsa;
  hsa_agent_t agent;
  hsa_queue_t* handle;
  uint32_t engine_id;
  iree_hal_amdgpu_sdma_capabilities_t capabilities;
  iree_hal_amdgpu_sdma_ring_t ring;
} iree_hal_amdgpu_sdma_queue_t;

// Creates an unpublished queue in caller-owned storage. On failure, destroys
// any partial native queue and leaves output unchanged. No payload allocation.
iree_status_t iree_hal_amdgpu_sdma_queue_initialize(
    const iree_hal_amdgpu_sdma_queue_params_t* params,
    iree_hal_amdgpu_sdma_queue_t* out_queue);
// Caller must retire all work and stop every producer before destruction.
// Like hsa_queue_destroy, failed native teardown is an invariant violation.
void iree_hal_amdgpu_sdma_queue_deinitialize(
    iree_hal_amdgpu_sdma_queue_t* queue);
// Convert a completed pair of SDMA timestamps to elapsed nanoseconds through
// the shared HSA clock domain. Does not wait for or infer completion.
iree_status_t iree_hal_amdgpu_sdma_queue_elapsed_ns(
    const iree_hal_amdgpu_sdma_queue_t* queue, uint64_t start_tick,
    uint64_t end_tick, double* out_elapsed_ns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_QUEUE_H_
