// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SEMAPHORE_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SEMAPHORE_H_

#include "iree/hal/drivers/amd/gpu/queue/pm4/signal.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a HAL timeline with independently retained PM4 signal acceleration.
// The proactor is borrowed through destruction. Frontier storage is allocated
// inline at creation; only completed signals contribute to that frontier.
// Submission availability has separate, lazily allocated host wait state.
// This object owns no native backing, queue, worker, or device reference.
iree_status_t iree_hal_amd_gpu_pm4_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    uint8_t frontier_capacity, iree_allocator_t host_allocator,
    iree_hal_semaphore_t** out_semaphore);

// Returns whether the semaphore provides this native signal contract.
// Type identity alone does not establish native addressability or visibility.
bool iree_hal_amd_gpu_pm4_semaphore_isa(iree_hal_semaphore_t* semaphore);

typedef enum iree_hal_amd_gpu_pm4_semaphore_wait_state_e {
  // No completed value or published producer covers the requested value.
  IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_DEFERRED = 0,
  // The completed host timeline covers the requested value.
  IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_COMPLETED,
  // A retained native producer can satisfy the wait without host retirement.
  IREE_HAL_AMD_GPU_PM4_SEMAPHORE_WAIT_NATIVE,
} iree_hal_amd_gpu_pm4_semaphore_wait_state_t;

// One acquired dependency. Completed and deferred results own no signal.
typedef struct iree_hal_amd_gpu_pm4_semaphore_wait_t {
  // Resolution at the instant of acquisition, including sticky failure checks.
  iree_hal_amd_gpu_pm4_semaphore_wait_state_t state;
  // Owned native reader reference, released after execution or quiescence.
  iree_hal_amd_gpu_pm4_signal_t* signal;
  // Producer coordinate proven by waiting on signal. This is a causal lower
  // bound, not the producer's complete transitive frontier. An epoch of zero
  // means no topology coordinate is assigned and cannot justify wait elision.
  iree_async_frontier_entry_t producer;
} iree_hal_amd_gpu_pm4_semaphore_wait_t;

// Resolves a wait and pins its native reader under the cache replacement lock.
// The caller has established the semaphore type. A native result is a pinned
// producer candidate: the consumer must also match its qualified domain to
// signal->pool->domain before using the address. An incompatible candidate is
// released and waited on through the ordinary completed HAL timeline, never
// through submission availability. Domain identity includes addressing,
// atomic width, visibility, and forward progress.
// A native result permits waiting for signal value one, or eliding that wait
// when an already established causal frontier dominates a nonzero producer
// epoch. The owning capture keeps the semaphore live through this call. Failure
// leaves an empty deferred result. No allocation, native call, or retry loop
// is performed.
iree_status_t iree_hal_amd_gpu_pm4_semaphore_acquire_wait(
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    iree_hal_amd_gpu_pm4_semaphore_wait_t* out_wait);

// Publishes an already admitted native producer. Its commands, including the
// release of signal, must be visible to the hardware before this call. Merely
// captured work with unresolved dependencies is not an eligible producer.
// Borrows signal and retains it only if it becomes the cached producer. A
// publication reference must keep signal live if execution finishes first.
// Higher values replace older cached values; existing readers remain retained.
// Completion racing publication cannot reinstall an already completed point.
// Submission publication does not advance the completed timeline or frontier.
// On failure the native work still owns its references until quiescence.
iree_status_t iree_hal_amd_gpu_pm4_semaphore_publish_signal(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_async_frontier_entry_t producer,
    iree_hal_amd_gpu_pm4_signal_t* signal);

// Publishes a proven native completion and returns any covered cache reference.
// Observers may arrive out of order even when GPU execution was ordered: a
// later already-observed completion subsumes this observation. This is not a
// substitute for the application's obligation to order semaphore signals.
// Ordinary host signals use the strict HAL signal API. The capture keeps the
// semaphore live through return and separately releases its producer reference.
iree_status_t iree_hal_amd_gpu_pm4_semaphore_publish_completed(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier);

// Acquires the lazily allocated host timeline for native submission admission.
// Its value is covered either by a cached native producer or by the completed
// timeline. Replacement by a higher producer preserves that predicate; only
// completion or failure removes the last applicable cache reference.
//
// This cold path allocates outside the metadata lock, retains the returned HAL
// semaphore, and propagates allocation or terminal failure. A caller registers
// an ordinary async wait on the returned semaphore and then retries admission
// on service. It never treats availability as execution completion. Consumers
// without qualified native compatibility wait on the real semaphore instead.
// The capture retains the original semaphore independently through resumption.
iree_status_t iree_hal_amd_gpu_pm4_semaphore_acquire_submission_semaphore(
    iree_hal_semaphore_t* semaphore, iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_SEMAPHORE_H_
