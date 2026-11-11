// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_COMPLETION_H_
#define IREE_HAL_DRIVER_AMD_GPU_COMPLETION_H_

#include "iree/async/frontier.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/driver/amd/gpu/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amd_gpu_completion_scope_t
    iree_hal_amd_gpu_completion_scope_t;

// Reports the outcome and lifetime of one native queue operation. A failure
// may be reported before retirement: retired=false permits internal failure
// handling but leaves every borrowed resource live. Exactly one retired=true
// callback carries the first failure and ends those borrows after publication
// and all accepted native users have joined. Rejected work can retire
// immediately. A failed device may never prove retirement.
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

// A coherent host word whose change requests another source drain. Pending
// work keeps observation active; passive publication and notification hints
// are observed only while another source has pending work. When all sources
// are idle, the worker parks on the native event. This is only a wake hint:
// the source still establishes retirement before releasing resources. No
// driver call or MMIO read is permitted to observe this word.
typedef struct iree_hal_amd_gpu_completion_wait_t {
  // Borrowed CPU-cacheable 32- or 64-bit atomic word, or NULL for event-only
  // observation. Width is selected by is_64_bit.
  // Storage belongs to signal when supplied; otherwise it remains live until
  // source replacement or worker join, including across helper callbacks.
  const void* address;
  // Last consumed value. A change requests another authoritative source drain.
  int64_t value;
  // Optional borrowed signal owning the word. The worker retains it under the
  // drain mutex before observing, preventing reuse or trim during those reads.
  struct iree_hal_amd_gpu_signal_t* signal;
  // Word width; false selects a signed 32-bit load, true a signed 64-bit load.
  bool is_64_bit;
  // Outstanding work that warrants active observation. Passive hints do not
  // keep an otherwise idle worker awake.
  bool pending;
} iree_hal_amd_gpu_completion_wait_t;

// One bounded CPU copy of already-retired staging bytes. Only the completion
// worker executes this outside the drain lock. Its source retains the staging
// and destination until return; helpers may retire other sources meanwhile.
// This callback cannot allocate, perform I/O, or invoke application code.
typedef struct iree_hal_amd_gpu_completion_copy_t {
  // Copies already-visible bytes and marks the source's copy consumed.
  void (*fn)(void* user_data);
  // Borrowed source-owned copy state, joined before replacement or teardown.
  void* user_data;
} iree_hal_amd_gpu_completion_copy_t;

// On success, consumes a bounded batch of authoritative completion records.
// Returns true when another batch is immediately ready or this batch retired
// resources that may unblock another source. Outstanding GPU work alone does
// not request another round. Native events are hints, never completion facts.
// The completion worker and active host waiters share short metadata exclusion.
// A source releases that exclusion with scope_unlock before native calls,
// resource destruction, or arbitrary callbacks. It remains exclusively owned
// until return; recursive helpers can retire other sources.
// out_wait starts empty; the source may provide its next coherent wake hint.
// A non-NULL out_copy admits one worker-owned staging copy. NULL leaves that
// copy pending: helpers never execute bulk copies, and each worker round admits
// at most one. The source is excluded until its copy returns; ordinary source
// retirement then releases the capture and publishes completion.
// An already-ready record blocked on another source's retirement supplies no
// hint until that dependency progresses. KERNEL status queries remain
// event-only.
//
// A native execution/observation failure is delivered once to every source
// before the worker joins the parent's terminal ownership and exits.
// Queue sources record terminal failure; the return value is then ignored.
// Failure does not establish native quiescence or permit releasing GPU storage.
typedef bool (*iree_hal_amd_gpu_completion_fn_t)(
    void* user_data, amdf_status_t wait_status,
    iree_hal_amd_gpu_completion_scope_t* scope,
    iree_hal_amd_gpu_completion_wait_t* out_wait,
    iree_hal_amd_gpu_completion_copy_t* out_copy);

// One completion source, such as queue retirement or shader reports. Only the
// worker stores hints here; helpers discard their hints. The drain mutex
// protects bindings, hints, and active invocation ownership. Replacement joins
// removed sources and coherent observation, and relocates retained invocations
// before returning the old table's ownership.
typedef struct iree_hal_amd_gpu_completion_source_t {
  // Bounded observation or terminal delivery; NULL while borrowed or disabled.
  iree_hal_amd_gpu_completion_fn_t fn;
  // Borrowed source state, live through the callback's final access.
  void* user_data;
  // Worker-owned next wake hint, unchanged throughout unlocked observation.
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
  iree_hal_amd_gpu_context_t* context;
  // Number of sources, stable under drain_mutex and coherent observation.
  iree_host_size_t source_count;
  // Borrowed contiguous array; active callbacks can outlive a table
  // replacement.
  iree_hal_amd_gpu_completion_source_t* sources;
  // Maximum active observation after progress, in ns; zero parks immediately.
  iree_duration_t spin_ns;
  // Owned thread, or NULL after failed initialization or a successful join.
  iree_thread_t* thread;
  // Stop/replacement requests and the worker's unlocked source borrow.
  iree_atomic_int32_t state;
  // Protects source metadata, table swaps, and shared shader report parsing.
  // Individual sources retain ownership across unlocked callouts.
  iree_slim_mutex_t drain_mutex;
  // Intrusive caller-stack ownership, linked and relocated under drain_mutex.
  iree_hal_amd_gpu_completion_scope_t* active_scopes;
  // First native execution/observation failure. Cold membership can observe it
  // before the terminal worker joins, without taking the drain mutex.
  iree_atomic_int64_t failure;
  // Callback join progress, posted only while a callback join is waiting.
  // Its epoch survives another joiner's enrollment after a callback returns.
  iree_notification_t callout_joined;
} iree_hal_amd_gpu_completion_t;

// One source invocation, initially under drain_mutex. A callout can release
// that mutex while retaining exclusive source ownership. The source binding is
// absent until finish restores it, so another drainer skips this source.
// This scope lives on the invoking thread's stack and owns no allocation.
struct iree_hal_amd_gpu_completion_scope_t {
  // Borrowed device completion owner.
  iree_hal_amd_gpu_completion_t* completion;
  // Borrowed table entry, retargeted under drain_mutex during replacement.
  iree_hal_amd_gpu_completion_source_t* source;
  // Binding restored after the final access; NULL when explicitly disabled.
  iree_hal_amd_gpu_completion_fn_t fn;
  // Next active invocation, or NULL at the end of the list.
  iree_hal_amd_gpu_completion_scope_t* next;
  // Link pointing at this invocation; NULL until its first unlocked callout.
  iree_hal_amd_gpu_completion_scope_t** previous;
  // Whether drain_mutex is currently released by this invocation.
  bool unlocked;
  // Whether replacement changed this round's table, including address reuse.
  bool relocated;
  // Whether this invocation belongs to the worker that consumes native wakes.
  bool is_worker;
};

// Leaves metadata exclusion before native work, releases, or callouts. Repeated
// calls while unlocked do nothing. Source ownership lasts through finish.
void iree_hal_amd_gpu_completion_scope_unlock(
    iree_hal_amd_gpu_completion_scope_t* scope);

// Temporarily rejoins metadata exclusion for shared report parsing. This does
// not release the source's exclusive ownership.
void iree_hal_amd_gpu_completion_scope_lock(
    iree_hal_amd_gpu_completion_scope_t* scope);

// Rejoins metadata exclusion and ends the source borrow. Returns whether a
// worker enrolled for this helper's return; the caller wakes it after releasing
// drain_mutex and finishing all table access. Worker-owned scopes never consume
// another helper's wake enrollment.
bool iree_hal_amd_gpu_completion_scope_finish(
    iree_hal_amd_gpu_completion_scope_t* scope);

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
    iree_hal_amd_gpu_context_t* context, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources, iree_duration_t spin_ns,
    iree_thread_affinity_t affinity, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_completion_t* out_completion);

// Replaces the borrowed source table after joining removed callbacks, staging
// copies, and coherent observations using the old table. Retained callbacks
// continue with their ownership retargeted to the replacement. Interrupts
// observation, including an infinite observation policy, without stopping the
// worker or invoking a native API. After every replacement, including removal
// of idle sources, the owner signals the context's native event to revisit
// retained work and rebuild wake hints, and handles any native wake failure.
// The worker may consume an earlier event while replacement excludes its drain
// or a ready copy. The replacement remains live until the next replacement
// or deinitialization; it is a distinct array with initially empty wait hints.
// Callback/user-data pairs uniquely identify sources, and retained sources keep
// their relative order. The cold owner serializes
// replacements with deinitialization and separately joins removed sources'
// accepted work and external producers. A retained callback may replace the
// table after scope_unlock; removing its own active source would self-join.
// No allocation or native operation occurs inside the handoff.
void iree_hal_amd_gpu_completion_set_sources(
    iree_hal_amd_gpu_completion_t* completion, iree_host_size_t source_count,
    iree_hal_amd_gpu_completion_source_t* sources);

// Disables one source and returns whether its final invocation is still active.
// user_data must uniquely identify this source. Serialized with replacement;
// the owner joins accepted work and external producers before removal. An
// active callback retains a disabled entry (NULL fn, same user_data) in every
// replacement until join_source returns. Its relative order remains stable so
// terminal failure delivery can continue after it without repeating a source.
// Other sources can be replaced or removed while the exact join waits.
bool iree_hal_amd_gpu_completion_remove_source(
    iree_hal_amd_gpu_completion_t* completion,
    iree_hal_amd_gpu_completion_fn_t fn, void* user_data);

// Joins this disabled source's final callback/copy access, outside the owner's
// table-replacement exclusion. The owner keeps source state and its disabled
// table entry live through return, then may omit the entry and reclaim state.
// Called outside this source's own invocation; there is no native operation.
void iree_hal_amd_gpu_completion_join_source(
    iree_hal_amd_gpu_completion_t* completion, void* user_data);

// Cold membership check after installing a source. Either the source receives
// a later native failure or this returns the failure already delivered before
// installation. Ordinary retirement and submission do not call this function.
amdf_status_t iree_hal_amd_gpu_completion_query_status(
    iree_hal_amd_gpu_completion_t* completion);

// Latches the first native error and excludes new healthy helper drains. Pure
// host publication: no drain lock, native call, or publisher wait. The parent
// arms its independent monitor first, then wakes host release waiters after
// this publication, and uses request_stop to interrupt a parked native worker.
void iree_hal_amd_gpu_completion_request_failure(
    iree_hal_amd_gpu_completion_t* completion, amdf_status_t status);

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

#endif  // IREE_HAL_DRIVER_AMD_GPU_COMPLETION_H_
