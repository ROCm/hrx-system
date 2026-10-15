// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_RING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_RING_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/gpu/context.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/ring.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Cold native storage embedded in the concrete queue, separate from its hot
// reservation state. The context outlives every acquired native handle and any
// failed cleanup. This owns no HAL resource reference or additional allocation.
typedef struct iree_hal_amd_gpu_pm4_ring_storage_t {
  // Borrowed native device and immutable API tables.
  const iree_hal_amd_gpu_context_t* context;
  // Owned native queue, consumed by successful or native-error removal.
  amdf_user_queue_t* queue;
  // Owned host producer mapping, removed before the queue.
  amdf_user_queue_mapping_t* mapping;
  // Terminal native removal error after queue-handle consumption. Retaining
  // this failure prevents a later cleanup attempt from certifying quiescence.
  amdf_status_t removal_status;
} iree_hal_amd_gpu_pm4_ring_storage_t;

// Acquires the selected PM4 USER queue and its host producer view. The caller
// has admitted the native family, format, SINGLE producer mode, host
// capability, ring capacity and scratch requirements in create_info. This is
// cold work; successful construction leaves no deferred native setup on
// publication.
//
// Initializes out_storage even on failure. The caller always deinitializes it
// before releasing its containing allocation or context; partial acquisition
// remains visible so failed native rollback cannot lose its owner. Success
// initializes an empty out_ring and copies the native publication addresses to
// out_mapping_info. Failure leaves those two outputs unchanged.
iree_status_t iree_hal_amd_gpu_pm4_ring_storage_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const amdf_gpu_user_queue_create_info_t* create_info,
    iree_hal_amd_gpu_pm4_ring_storage_t* out_storage,
    iree_hal_amd_gpu_pm4_ring_t* out_ring,
    amdf_user_queue_mapping_info_t* out_mapping_info);

// Joins the final already-published primary-ring index, then removes its host
// mapping and native queue. All host publishers, callbacks and execution uses
// have separately joined. Prefix completion alone does not consume trailing
// notification/padding packets; this cold wait joins that last native access.
// A zero index also handles partial initialization without any publication.
//
// Success permits releasing this embedded owner. Failure preserves remaining
// handles, the original native status domain, and the context's required
// lifetime. A failed native removal consumes its queue handle but remains a
// terminal failure on subsequent calls; it never certifies safe reclamation.
// This is exclusively owned teardown, never a submission or retirement action.
amdf_status_t iree_hal_amd_gpu_pm4_ring_storage_deinitialize(
    iree_hal_amd_gpu_pm4_ring_storage_t* storage, uint64_t published_words);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_RING_H_
