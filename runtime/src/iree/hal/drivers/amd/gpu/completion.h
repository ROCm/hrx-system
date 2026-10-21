// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_COMPLETION_H_
#define IREE_HAL_DRIVERS_AMD_GPU_COMPLETION_H_

#include "iree/async/frontier.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reports the outcome and lifetime of one native queue operation. A failure
// may be reported before retirement: retired=false fails dependents but leaves
// every borrowed resource live. Exactly one retired=true callback ends those
// borrows after publication and all accepted native users have joined. Rejected
// work can retire immediately. A failed device may never prove retirement.
// Callbacks consume status and borrow the completed queue frontier for the
// call. The frontier may be NULL when no native work retired. Calls may run
// before submit returns; the submitter separately joins its borrowed arguments.
typedef struct iree_hal_amd_gpu_retirement_callback_t {
  // Failure notification or final ownership handoff, without an executor hop.
  void (*fn)(void* user_data, iree_status_t status,
             const iree_async_single_frontier_t* frontier, bool retired);
  // Borrowed owner, live until the final callback returns its ownership.
  void* user_data;
} iree_hal_amd_gpu_retirement_callback_t;

// A coherent host word whose change may retire already-published work. Idle
// sources leave this empty and rely on native completion events. This is only
// a wake hint: the source still establishes native retirement before releasing
// resources. No driver call or MMIO read is permitted to observe this word.
typedef struct iree_hal_amd_gpu_completion_wait_t {
  // Borrowed CPU-cacheable atomic word, or NULL for event-only observation.
  // Storage remains live until the next callback, source replacement, or join.
  const iree_atomic_int32_t* address;
  // Last consumed value. A change requests another authoritative source drain.
  int32_t value;
} iree_hal_amd_gpu_completion_wait_t;

// On success, consumes a bounded batch of authoritative completion records.
// Returns true when another batch is immediately ready or this batch retired
// resources that may unblock another source. Outstanding GPU work alone does
// not request another round. Native events are hints, never completion facts.
// The completion worker and active host waiters share a serialized drain.
// A callback may publish continuations, but cannot synchronously wait for
// another source in this drain. Recursive try_drain calls return immediately.
// out_wait starts empty; the source may provide its next coherent wake hint.
// An already-ready record blocked on another source's retirement supplies no
// hint until that dependency progresses. KERNEL status queries remain
// event-only.
//
// A native wait failure is delivered once to every source before worker exit.
// Queue sources record terminal failure; the return value is then ignored.
// Failure does not establish native quiescence or permit releasing GPU storage.
typedef bool (*iree_hal_amd_gpu_completion_fn_t)(
    void* user_data, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_wait_t* out_wait);

// One completion source, such as queue retirement or shader reports. Callback
// bindings are immutable and borrowed until source replacement or worker join.
// The drain mutex protects table replacement, callbacks, and wait-hint
// observation, including the lifetime of each borrowed coherent word.
typedef struct iree_hal_amd_gpu_completion_source_t {
  // Bounded observation or terminal native-failure delivery.
  iree_hal_amd_gpu_completion_fn_t fn;
  // Borrowed source state, live through the callback's final access.
  void* user_data;
  // Consumer-owned next wake hint, replaced after each authoritative drain.
  iree_hal_amd_gpu_completion_wait_t wait;
} iree_hal_amd_gpu_completion_source_t;

// One high-priority completion worker per device. Its source array has
// no submission-time activation or registration traffic. The worker waits
// on coherent completion words for a bounded interval after actual progress,
// then on the context's native event. GPU producers and host stop signals share
// that event. No timer, polling thread, or per-queue worker is created.
// The embedding owner joins accepted work before stopping this worker.
typedef struct iree_hal_amd_gpu_completion_t {
  // Borrowed context whose event and API table outlive the worker join.
  const iree_hal_amd_gpu_context_t* context;
  // Number of sources, protected by drain_mutex.
  iree_host_size_t source_count;
  // Borrowed contiguous source array, replaced only under drain_mutex.
  iree_hal_amd_gpu_completion_source_t* sources;
  // Maximum active observation after progress, in ns; zero parks immediately.
  iree_duration_t spin_ns;
  // Owned thread, or NULL after failed initialization or a successful join.
  iree_thread_t* thread;
  // Stop and source-replacement requests interrupt active coherent observation.
  iree_atomic_int32_t state;
  // Serializes source retirement and every borrowed wait-hint observation.
  iree_slim_mutex_t drain_mutex;
} iree_hal_amd_gpu_completion_t;

// Attempts one bounded source round without waiting for another observer.
// Returns whether this round made progress; contention or a stopped observer
// returns false. An active caller uses this after observing native completion,
// then joins the public timeline. It does not repeatedly scan pending sources.
// The caller keeps the enclosing device alive through the final access; owner
// teardown joins all such callers before deinitializing the drain mutex.
bool iree_hal_amd_gpu_completion_try_drain(
    iree_hal_amd_gpu_completion_t* completion);

// Starts the sole native event waiter with an explicit affinity and high
// priority hint. Sources must be ready before entry; callbacks may run before
// this returns. Allocation and thread startup are cold, fallible operations.
// Failure runs no callback and leaves out_completion safe to deinitialize.
iree_status_t iree_hal_amd_gpu_completion_initialize(
    const iree_hal_amd_gpu_context_t* context, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources, iree_duration_t spin_ns,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_completion_t* out_completion);

// Replaces the borrowed source table after joining all callbacks and coherent
// observations using the old table. Interrupts active observation, including
// an infinite observation policy, without stopping the worker or invoking a
// native API. The cold owner then signals the context's native event to rebuild
// wake hints for remaining queues and handles any native wake failure. The
// replacement remains live until the next replacement or deinitialization; its
// wait hints start empty. The cold owner serializes replacements with
// deinitialization and separately joins removed sources' accepted work and
// external producers. Called outside source callbacks, with no allocation or
// native operation inside the handoff.
void iree_hal_amd_gpu_completion_set_sources(
    iree_hal_amd_gpu_completion_t* completion, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources);

// Requests exit after the current source round and wakes a parked worker.
// Called only after accepted work and producers have joined. This does not
// join or release any ownership; deinitialization joins final worker access.
// Native wake failure preserves the request and permits retry. The raw status
// lets a void device destructor diagnose failure without constructing and
// discarding an IREE status.
amdf_status_t iree_hal_amd_gpu_completion_request_stop(
    iree_hal_amd_gpu_completion_t* completion);

// Stops and joins the final source access. Called by one external owner after
// accepted work and producers have joined, never from a source callback.
//
// This does not wait for GPU work or consume notification tails on the owner's
// behalf. A failed native stop wake returns an error and preserves the entire
// live owner for retry. A successful return permits releasing borrowed source
// state and the context, subject to their separate native producer lifetime
// joins.
iree_status_t iree_hal_amd_gpu_completion_deinitialize(
    iree_hal_amd_gpu_completion_t* completion);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_COMPLETION_H_
