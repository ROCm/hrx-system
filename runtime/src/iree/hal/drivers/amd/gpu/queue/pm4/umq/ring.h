// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_RING_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_UMQ_RING_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Primary PM4 command storage accessed only by the queue's current publisher.
// Native allocation, mapping, write-index publication and the MMIO doorbell
// belong to the enclosing UMQ. This state never reads a native read pointer or
// waits for capacity. Completion returns exact consumed-prefix credit; later
// notification packets and padding are not covered by an earlier release.
typedef struct iree_hal_amd_gpu_pm4_ring_t {
  // Borrowed writable primary ring, live through native consumption/removal.
  uint32_t* words;
  // Power-of-two capacity in DWORDs, admitted from the native mapping.
  uint32_t capacity;
  // End of reserved storage in the monotonic native DWORD domain. The sole
  // publisher completes these stores before publishing this index to hardware.
  uint64_t reserved_words;
} iree_hal_amd_gpu_pm4_ring_t;

// Contiguous body within one primary-ring publication. Padding is initialized
// by reservation; the infallible encoder fills the complete body before the
// enclosing queue publishes ring.reserved_words.
typedef struct iree_hal_amd_gpu_pm4_ring_reservation_t {
  // Borrowed body destination; no packet in the body straddles ring wrap.
  uint32_t* words;
  // Monotonic body start in DWORDs, after any wrap padding. Adding a release
  // packet's exclusive body-relative end gives its exact consumed prefix.
  uint64_t position;
} iree_hal_amd_gpu_pm4_ring_reservation_t;

// Binds newly created, empty native storage. Capacity and pointer alignment
// have already been established by UMQ construction; this allocates nothing.
void iree_hal_amd_gpu_pm4_ring_initialize(
    uint32_t* words, uint32_t capacity, iree_hal_amd_gpu_pm4_ring_t* out_ring);

// Reserves one bounded body, including native eight-DWORD publication alignment
// and wrap padding. False means insufficient consumed-prefix credit and leaves
// the ring, its storage and out_reservation unchanged. The enclosing queue
// parks on its already-enrolled completion notification instead of polling.
//
// The caller supplies an admitted nonempty body whose padded extent is at most
// 16384 DWORDs and strictly less than half the ring capacity. Each publication
// includes an execution-completion release followed only by an optional
// eight-DWORD event notification and alignment padding. The caller records the
// release's exact end before publication. The completion consumer supplies only
// a proven prefix in consumed_words, never a speculative value or the end of
// later event packets.
// At least one DWORD always remains unconsumed, as required by the native ring.
//
// The current publisher has exclusive access through encoding and publication.
// No fallible work follows successful reservation and no second reservation
// begins before publication. Payload lifetime is independent of ring credit.
bool iree_hal_amd_gpu_pm4_ring_try_reserve(
    iree_hal_amd_gpu_pm4_ring_t* ring, uint64_t consumed_words,
    uint32_t word_count,
    iree_hal_amd_gpu_pm4_ring_reservation_t* out_reservation);

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
