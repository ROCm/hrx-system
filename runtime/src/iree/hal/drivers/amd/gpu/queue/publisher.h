// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PUBLISHER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PUBLISHER_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Link embedded in a captured, causally ready operation. Unresolved dependency
// waits do not enter this list. Storage remains owned by the operation through
// publication and its separate retirement contract. The publisher never
// allocates or retains an operation, nor interprets its native command format.
typedef struct iree_hal_amd_gpu_publication_entry_t {
  // Next ready entry while linked; otherwise exclusively owned by the caller.
  struct iree_hal_amd_gpu_publication_entry_t* next;
} iree_hal_amd_gpu_publication_entry_t;

// Exclusively owned ready prefix. An empty head makes the tail irrelevant.
typedef struct iree_hal_amd_gpu_publication_list_t {
  // First remaining operation, or NULL after the prefix is consumed.
  iree_hal_amd_gpu_publication_entry_t* head;
  // Last operation when head is non-NULL.
  iree_hal_amd_gpu_publication_entry_t* tail;
} iree_hal_amd_gpu_publication_list_t;

// Transfers one exact queue's native producer between ready API callers and
// its device's single service executor. Only the current owner accesses the
// native producer. This object contains no native state or per-queue worker.
//
// The slim mutex protects only ownership and list links. Capture, allocation,
// encoding, driver calls, doorbells and retirement all run outside it. Every
// operation below takes this short lock; none waits for GPU capacity or work.
// The embedding queue separates this metadata from native producer state and
// completion cursors to avoid sharing their independently written cache lines.
typedef struct iree_hal_amd_gpu_publisher_t {
  // Serializes enqueue with the current owner's final handoff.
  iree_slim_mutex_t mutex;
  // Implementation-owned idle/caller/service discriminator.
  uint32_t owner;
  // Ready work appended while another publisher owns the native producer.
  iree_hal_amd_gpu_publication_list_t pending;
} iree_hal_amd_gpu_publisher_t;

void iree_hal_amd_gpu_publisher_initialize(
    iree_hal_amd_gpu_publisher_t* out_publisher);

// All producers and the service executor have joined; no owner or work remains.
void iree_hal_amd_gpu_publisher_deinitialize(
    iree_hal_amd_gpu_publisher_t* publisher);

// Enqueues a completely captured, ready operation. True grants the caller
// exclusive native publication ownership for this entry immediately. False
// transfers the entry to the pending list; the caller must not access it again.
// The current owner guarantees service progress, so this caller needs no wake.
// Concurrent entries are ordered by this metadata lock, not HAL causality.
bool iree_hal_amd_gpu_publisher_enqueue(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* entry);

// Relinquishes the native producer after the owner's last access to it. An
// unpublished remainder is prepended ahead of entries enqueued concurrently.
// This supports bounded service turns and deferral on native capacity.
//
// True transfers responsibility to the device service executor. A calling
// producer must arrange that service turn before returning; a service owner
// may continue or park on its already-enrolled capacity notification. False
// leaves the queue idle so the next ready caller can publish immediately.
// Neither result permits further access to native producer state.
bool iree_hal_amd_gpu_publisher_handoff(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_list_t remainder);

// Detaches pending work for the single service executor in O(1). An empty
// result is a stale wake or another caller's ownership and needs no handoff.
// A nonempty result grants exclusive native publication ownership until the
// service calls handoff, even while new callers append to the pending list.
//
// The executor enrolls in its notification before checking this predicate and
// capacity, then cancels the wait when proceeding or commits it when parking.
// Each entry's next link must be captured before native publication: immediate
// completion may retire and free the entry before publication returns.
iree_hal_amd_gpu_publication_list_t iree_hal_amd_gpu_publisher_take(
    iree_hal_amd_gpu_publisher_t* publisher);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PUBLISHER_H_
