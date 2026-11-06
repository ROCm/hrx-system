// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_HSA_QUEUE_H_
#define IREE_HAL_DRIVER_AMDGPU_HSA_QUEUE_H_

#include "iree/base/api.h"
#include "iree/hal/driver/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable parameters used to create one native HSA queue.
typedef struct iree_hal_amdgpu_hsa_queue_params_t {
  // HSA API table used to create and destroy the queue.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // HSA agent on which the queue is created.
  hsa_agent_t agent;
  // Power-of-two queue capacity in packets.
  uint32_t packet_count;
  // Native producer and dispatch behavior of the queue.
  hsa_queue_type32_t type;
  // Native dispatch and wavefront scheduling priority.
  hsa_amd_queue_priority_t priority;
  // Number of bits in |compute_unit_mask|. Zero leaves the native queue
  // unmasked.
  uint32_t compute_unit_mask_bit_count;
  // Exact native compute-unit mask applied before publication. NULL exactly
  // when |compute_unit_mask_bit_count| is zero.
  const uint32_t* compute_unit_mask;
  // Callback invoked by HSA for asynchronous queue errors.
  void (*error_callback)(hsa_status_t status, hsa_queue_t* source, void* data);
  // Application data passed to |error_callback|.
  void* error_callback_data;
} iree_hal_amdgpu_hsa_queue_params_t;

// Creates an unpublished native HSA queue with the exact |params|.
//
// The queue uses the runtime's default private and group segment sizing. On
// success |out_queue| receives ownership of one HSA queue reference. On
// failure |out_queue| is unchanged and any partially created queue is
// destroyed before returning. Cooperative queues accept only normal priority,
// are created without applying a compute-unit mask to ROCr's agent-shared
// queue, and require the complete execution-resource set.
iree_status_t iree_hal_amdgpu_hsa_queue_allocate(
    const iree_hal_amdgpu_hsa_queue_params_t* params, hsa_queue_t** out_queue);

// Destroys |queue| and asserts the HSA teardown invariant.
void iree_hal_amdgpu_hsa_queue_free(const iree_hal_amdgpu_libhsa_t* libhsa,
                                    hsa_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMDGPU_HSA_QUEUE_H_
