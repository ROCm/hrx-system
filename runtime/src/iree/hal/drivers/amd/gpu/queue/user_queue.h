// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_USER_QUEUE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_USER_QUEUE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Cold native storage embedded in the concrete queue, separate from its hot
// reservation state. The context outlives every acquired native handle and any
// failed cleanup. This owns no HAL resource reference or additional allocation.
typedef struct iree_hal_amd_gpu_user_queue_t {
  // Borrowed native device and immutable API tables.
  const iree_hal_amd_gpu_context_t* context;
  // Owned native queue, consumed by successful or native-error removal.
  amdf_user_queue_t* queue;
  // Owned host producer mapping, removed before the queue.
  amdf_user_queue_mapping_t* mapping;
  // Optional exact-device producer view, removed before the host view.
  amdf_user_queue_mapping_t* device_mapping;
  // Terminal native removal error after queue-handle consumption. Retaining
  // this failure prevents a later cleanup attempt from certifying quiescence.
  amdf_status_t removal_status;
  // Achieved native primary-ring locality, captured once during construction.
  iree_hal_queue_placement_t placement;
  // Native release selected once before any command publication.
  amdf_cache_transition_executor_t host_publish_executor;
  // Direct host ordering before publishing a valid header or producer index.
  iree_hal_host_cache_fence_t host_publish_fence;
} iree_hal_amd_gpu_user_queue_t;

// Acquires the selected native USER queue and its host producer view. The
// caller has admitted the native family, format, SINGLE producer mode, host
// capability, ring capacity and scratch requirements in create_info. This is
// cold work; successful construction leaves no deferred native setup on
// publication.
//
// Initializes out_storage even on failure. The caller always deinitializes it
// before releasing its containing allocation or context; partial acquisition
// remains visible so failed native rollback cannot lose its owner. Success
// copies the native publication addresses to out_mapping_info. A non-NULL
// out_device_mapping_info additionally acquires the exact device's view, whose
// producer capability must have been requested in create_info.
// Protocol-specific ring state is initialized by the caller. Failure leaves
// both outputs unchanged.
iree_status_t iree_hal_amd_gpu_user_queue_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const amdf_gpu_user_queue_create_info_t* create_info,
    iree_hal_amd_gpu_user_queue_t* out_storage,
    amdf_user_queue_mapping_info_t* out_mapping_info,
    amdf_user_queue_mapping_info_t* out_device_mapping_info);

// Executes the captured release before a format commits its header/index.
// NONE needs no call. HOST_API preserves the exact native failure; the caller
// terminally fails its existing queue owner without reclaiming accepted work.
amdf_status_t iree_hal_amd_gpu_user_queue_flush(
    const iree_hal_amd_gpu_user_queue_t* storage);

// Joins the final already-published primary-ring index, then removes its device
// and host mappings and native queue. Publishers, callbacks and execution uses
// have separately joined. Prefix completion alone does not consume trailing
// notification/padding packets; this cold wait joins that last native access.
// A zero index also handles partial initialization without any publication.
//
// Success permits releasing this embedded owner. Failure preserves remaining
// handles, the original native status domain, and the context's required
// lifetime. A failed native removal consumes its queue handle but remains a
// terminal failure on subsequent calls; it never certifies safe reclamation.
// This is exclusively owned teardown, never a submission or retirement action.
amdf_status_t iree_hal_amd_gpu_user_queue_deinitialize(
    iree_hal_amd_gpu_user_queue_t* storage, uint64_t published_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_USER_QUEUE_H_
