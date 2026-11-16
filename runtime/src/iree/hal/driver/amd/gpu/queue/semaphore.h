// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_SEMAPHORE_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_SEMAPHORE_H_

#include "iree/hal/driver/amd/gpu/queue/signal.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Immutable device-owned policy for host observation of native signals.
// ACTIVE observes through the caller deadline; YIELD uses a bounded budget
// before returning to ordinary host notification. NONE observes no native word.
// Retirement remains authoritative for cleanup, frontiers, and shader failure;
// a native value alone never satisfies the public timeline. The semaphore and
// its callers borrow this context through the parent device's lifetime.
typedef struct iree_hal_amd_gpu_semaphore_wait_context_t {
  // Qualified native signal domain whose coherent words the host may observe.
  const void* domain;
  // Value before native completion: zero for PM4 or one for AQL.
  uint64_t pending_value;
  // Attempts bounded retirement after a native value changes. Contention
  // returns immediately; the ordinary observer continues independently.
  void (*try_retire)(void* user_data);
  // Borrowed retirement owner, live through all semaphore calls.
  void* user_data;
} iree_hal_amd_gpu_semaphore_wait_context_t;

// Creates a HAL timeline with independently retained native signal
// acceleration. The proactor is borrowed through destruction. Frontier storage
// is allocated inline at creation; only completed signals contribute to that
// frontier. The same allocation constructs a separately retained submission-
// availability view. Its final reference releases the shared allocation after
// the completed timeline is destroyed. This object owns no native backing,
// queue, worker, or device reference. A NULL wait_context uses only public
// timeline observation.
iree_status_t iree_hal_amd_gpu_semaphore_create(
    iree_async_proactor_t* proactor,
    const iree_hal_amd_gpu_semaphore_wait_context_t* wait_context,
    uint64_t initial_value, uint8_t frontier_capacity,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

// Returns whether the semaphore provides this native signal contract.
// Type identity alone does not establish native addressability or visibility.
bool iree_hal_amd_gpu_semaphore_isa(iree_hal_semaphore_t* semaphore);

typedef enum iree_hal_amd_gpu_semaphore_wait_state_e {
  // No completed value or published producer covers the requested value.
  IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_DEFERRED = 0,
  // The completed host timeline covers the requested value.
  IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_COMPLETED,
  // A retained native producer can satisfy the wait without host retirement.
  IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE,
  // Publication covers the value but no exact compatible producer exists.
  // Wait for the requested public value without borrowing native storage.
  IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_HOST,
} iree_hal_amd_gpu_semaphore_wait_state_t;

// One acquired dependency. Only a native result owns a signal.
typedef struct iree_hal_amd_gpu_semaphore_wait_t {
  // Resolution at the instant of acquisition, including sticky failure checks.
  iree_hal_amd_gpu_semaphore_wait_state_t state;
  // Owned native reader reference, released after execution or quiescence.
  // Retains the complete causal snapshot and producer retirement outcome.
  iree_hal_amd_gpu_signal_t* signal;
} iree_hal_amd_gpu_semaphore_wait_t;

// One accepted producer, stored in its submission's capture. The submission
// retains both the semaphore and signal until it withdraws this association.
// Index membership borrows those owners and never pins additional native cells.
typedef struct iree_hal_amd_gpu_semaphore_producer_t {
  // Parent in the semaphore's value-ordered index, protected by its mutex.
  struct iree_hal_amd_gpu_semaphore_producer_t* parent;
  // Lower-value subtree, protected by the semaphore mutex.
  struct iree_hal_amd_gpu_semaphore_producer_t* left;
  // Higher-value subtree, protected by the semaphore mutex.
  struct iree_hal_amd_gpu_semaphore_producer_t* right;
  // Exact public timeline value produced by this submission.
  uint64_t value;
  // Borrowed native producer; NULL before publication or after withdrawal.
  iree_hal_amd_gpu_signal_t* signal;
  // AVL subtree height, with a missing child having height zero.
  uint32_t height;
} iree_hal_amd_gpu_semaphore_producer_t;

// Resolves completed waits using atomic loads. An unresolved wait pins its
// native reader under the producer index lock only for an exact producer
// value. A later producer may depend on this consumer and cannot replace the
// requested dependency; HOST observes the original public timeline instead.
// The caller has established the semaphore type and retains the native domain
// through the reader's final release. Domain identity includes addressing,
// atomic width, completion protocol, visibility, forward progress, and backing
// lifetime. Foreign
// signals never escape the index lock: HOST uses the ordinary completed HAL
// timeline, never submission availability, and owns no producer storage.
// A native result permits the domain's native wait operation. Its retained
// snapshot supplies the complete native-domain causal history; producer
// retirement separately joins the producer's payload release and completion
// outcome. The caller keeps the semaphore live through this call. Failure
// leaves an empty deferred result. No allocation, native call, or retry loop is
// performed.
iree_status_t iree_hal_amd_gpu_semaphore_acquire_wait(
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value, const void* domain,
    iree_hal_amd_gpu_semaphore_wait_t* out_wait);

// Publishes an irrevocably accepted native producer. KERNEL publication has
// succeeded; a USER publisher has committed the complete command sequence and
// must publish its write index and doorbell without another fallible step.
// Captured work with unresolved dependencies is not an eligible producer.
// The caller supplies zero-initialized producer storage and retains both
// semaphore and signal until unpublish_signal. A publication reference keeps
// signal live until retirement ownership transfers. Every live exact producer
// remains available independently of subsequent publications.
// Completion racing publication cannot reinstall an already completed point.
// Submission publication does not advance the completed timeline or frontier.
// On failure the native work still owns its references until quiescence.
iree_status_t iree_hal_amd_gpu_semaphore_publish_signal(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_hal_amd_gpu_signal_t* signal,
    iree_hal_amd_gpu_semaphore_producer_t* producer);

// Withdraws a producer after public completion covers its value or the
// semaphore has failed, and before its capture or native owner is released.
// An observed available point therefore cannot become unavailable. Readers
// already acquired under the same mutex retain their independent native use.
// Zero-initialized unpublished records are accepted; no allocation or callback
// occurs. The caller still owns semaphore and signal through return.
void iree_hal_amd_gpu_semaphore_unpublish_signal(
    iree_hal_semaphore_t* semaphore,
    iree_hal_amd_gpu_semaphore_producer_t* producer);

// Publishes a proven native completion. Producer associations remain borrowed
// until their owning submissions withdraw them during retirement.
// Observers may arrive out of order even when GPU execution was ordered: a
// later already-observed completion subsumes this observation. This is not a
// substitute for the application's obligation to order semaphore signals.
// Ordinary host signals use the strict HAL signal API. The capture keeps the
// semaphore live through return and separately releases its producer reference.
iree_status_t iree_hal_amd_gpu_semaphore_publish_completed(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier);

// Acquires the already constructed view for native submission admission.
// Its value is covered by a published native producer or the completed
// timeline. Live producers retain their exact associations until retirement;
// completion then covers the withdrawn points. A higher published value does
// not supply an exact native dependency for skipped timeline values.
//
// First acquisition activates availability notifications under the metadata
// lock and publishes the current predicate. No allocation or semaphore creation
// occurs. The returned view is retained; terminal failure returns no view. A
// caller registers an ordinary async wait and then retries admission on
// service. It never treats availability as execution completion. Consumers
// without qualified native compatibility wait on the completed semaphore
// instead. The capture retains the original semaphore independently through
// resumption.
iree_status_t iree_hal_amd_gpu_semaphore_acquire_submission_semaphore(
    iree_hal_semaphore_t* semaphore, iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_SEMAPHORE_H_
