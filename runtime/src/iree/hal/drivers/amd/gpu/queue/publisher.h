// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PUBLISHER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PUBLISHER_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Link embedded in a captured operation. Once next returns an entry, no
// producer can access its link again: native retirement may immediately free
// the operation. Unresolved dependencies are parked outside this FIFO.
typedef struct iree_hal_amd_gpu_publication_entry_t {
  // Release-published successor, or zero until the producer finishes linking.
  iree_atomic_intptr_t next;
} iree_hal_amd_gpu_publication_entry_t;

// Transfers one exact queue's admission/native producer between API callers
// and the device service executor. Enqueue and each consumption step have a
// bounded number of atomic operations; no mutex, allocation, or reverse walk
// is required. A producer paused during linking never blocks the consumer.
//
// Caller/service ownership protects head and ready. The permanent stub lets
// the consumer detach the final real entry without borrowing its storage until
// another submission arrives. The embedding queue isolates this cache line
// from independently written native producer and completion state.
typedef struct iree_alignas(64) iree_hal_amd_gpu_publisher_t {
  // Atomic caller ownership, reserved service turn, and notification bits.
  iree_atomic_int32_t state;
  // Last reserved FIFO link; producers exchange it before linking successors.
  iree_atomic_intptr_t tail;
  // Permanent link used when the FIFO is empty or its final entry detaches.
  iree_hal_amd_gpu_publication_entry_t stub;
  // Next FIFO entry to detach, accessed only by the exclusive publisher.
  iree_hal_amd_gpu_publication_entry_t* head;
  // Detached entry awaiting publication, including capacity deferral.
  iree_hal_amd_gpu_publication_entry_t* ready;
} iree_hal_amd_gpu_publisher_t;

void iree_hal_amd_gpu_publisher_initialize(
    iree_hal_amd_gpu_publisher_t* out_publisher);

// Every producer, publication owner, and entry has joined before teardown.
void iree_hal_amd_gpu_publisher_deinitialize(
    iree_hal_amd_gpu_publisher_t* publisher);

// Transfers a completely captured operation. True grants exclusive publication
// ownership: the caller consumes at most one entry with next and then calls
// handoff. That entry may be an older capture, or NULL across an unfinished
// producer link. An idle caller receives its own entry immediately.
//
// False transfers all entry ownership; the caller must not access it again.
// The active publisher owns progress. Host enqueue order is not HAL causality.
bool iree_hal_amd_gpu_publisher_enqueue(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* entry);

// Claims pending work for the single service executor. False is a stale wake
// or a caller's active ownership. True requires a final handoff even when next
// returns NULL. Service bounds its work independently of the FIFO length.
bool iree_hal_amd_gpu_publisher_acquire_service(
    iree_hal_amd_gpu_publisher_t* publisher);

// Detaches the next captured operation under exclusive publication ownership.
// NULL means empty or an unfinished producer link; the owner calls handoff
// without spinning. The producer finishing that link guarantees future
// progress. A returned entry has no remaining producer references and may be
// freed.
iree_hal_amd_gpu_publication_entry_t* iree_hal_amd_gpu_publisher_next(
    iree_hal_amd_gpu_publisher_t* publisher);

// Relinquishes publication after its owner's last native-state access. A
// non-NULL remainder is the current detached entry blocked on native capacity;
// it precedes all entries still in the FIFO. No suffix traversal is needed.
//
// True requests a service turn. A caller arranges that turn before returning;
// service may continue or park on its already-enrolled capacity notification.
// False leaves the publisher idle. A racing enqueue either marks this handoff
// or acquires idle ownership itself, so an unfinished link cannot lose a wake.
bool iree_hal_amd_gpu_publisher_handoff(
    iree_hal_amd_gpu_publisher_t* publisher,
    iree_hal_amd_gpu_publication_entry_t* remainder);

// Cold shutdown query after external producers have joined. No publication
// owner or notified work remains; accepted operation lifetime is independent.
bool iree_hal_amd_gpu_publisher_is_idle(
    iree_hal_amd_gpu_publisher_t* publisher);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PUBLISHER_H_
