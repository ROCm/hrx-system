// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_HEARTBEAT_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_HEARTBEAT_H_

#include "experimental/loom_serve/http/server.h"
#include "experimental/loom_serve/runtime/device.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_text_heartbeat_snapshot_t {
  // Current application activity; literals outlive the reporter.
  const char* phase;
  // Start of the current activity, including a possibly unfinished GPU wait.
  iree_time_t phase_start;
  // Most recent completed epoch, or service startup before any model work.
  iree_time_t last_completion;
  // Requests currently holding retained rows.
  iree_host_size_t active_rows;
  // Claimed requests waiting without a device row or completion reservation.
  iree_host_size_t queued_requests;
  // Pool accounting in token positions; reserved and resident are disjoint.
  struct {
    // Addressable token capacity; zero selects dense comparison storage.
    iree_host_size_t capacity;
    // Additional positions reserved for active completion, including tail COW.
    iree_host_size_t reserved;
    // Unique resident positions, including idle rows and explicit pins.
    iree_host_size_t resident;
  } pool;
  // Host payload owned by explicitly suspended endpoints, not device backing.
  iree_host_size_t checkpoint_host_bytes;
  // Elastic mutable-state backing; weights and transient workspace are
  // separate.
  loom_serve_memory_statistics_t state_memory;
  // Prepared parameters, independently reclaimable from retained state.
  loom_serve_memory_statistics_t weight_memory;
  // Shared owner accounting, including other models and pooled command scratch.
  loom_serve_device_memory_statistics_t device_memory;
  // Active requests still consuming prompt input.
  iree_host_size_t prefill_rows;
  // Active requests generating output.
  iree_host_size_t decode_rows;
  // Rows paused because their peer has not returned output credit.
  iree_host_size_t backpressured_rows;
  // Issued epoch identity; a failed epoch never increments completed_epochs.
  uint64_t issued_epochs;
  // Fully completed model epochs.
  uint64_t completed_epochs;
  // Total target weight traversals, including each isolated control span.
  uint64_t traversals;
  // Prompt tokens consumed by completed epochs.
  uint64_t prefill_tokens;
  // Pending generated tokens consumed by completed epochs.
  uint64_t decode_tokens;
  // Selected output tokens published to response processing, including EOS.
  uint64_t output_tokens;
  // Sum of completed model epoch wall times, excluding host text processing.
  iree_duration_t model_duration;
  // Completed speculative work, separate from committed output progress.
  struct {
    // Candidate tokens produced across the three proposal rounds.
    uint64_t proposed_tokens;
    // Draft inputs consumed by the target, excluding each pending anchor.
    uint64_t accepted_inputs;
  } mtp;
  // Span count of the most recently issued epoch.
  iree_host_size_t epoch_spans;
  // Input count of the most recently issued epoch, including proposed inputs.
  iree_host_size_t epoch_tokens;
} loom_serve_text_heartbeat_snapshot_t;

typedef struct loom_serve_text_heartbeat_t {
  // Protects copied statistics and callback publication, never model work.
  iree_slim_mutex_t mutex;
  // Borrowed host I/O owner, outliving every accepted operation.
  iree_async_proactor_t* proactor;
  // Borrowed nonblocking service wake target, protected by mutex.
  loom_serve_http_observer_t observer;
  // Borrowed prequoted deployment name, stable through deinitialization.
  iree_string_view_t name_json;
  // Immutable reporting period in nanoseconds.
  iree_duration_t interval;
  // Owner has requested shutdown; no further timers are armed.
  bool stopping;
  // Callback publication consumed by the serialized service owner.
  struct {
    // Intrusive timer, stop handoff and cancellation receipt ownership bits.
    uint32_t pending;
    // A callback outcome awaits consumption, even after its operation retired.
    bool ready;
    // Owned callback failures, returned exactly once by advance.
    iree_status_t status;
  } events;
  // Private timer and the two lifetimes required for prompt shutdown.
  struct {
    // One deadline, reused only after successful terminal callback return.
    iree_async_timer_operation_t operation;
    // Poll-owner handoff for requesting cancellation without blocking a worker.
    iree_async_nop_operation_t stop;
    // Receipt retiring any native cancellation key before timer storage dies.
    iree_async_cancel_request_t cancellation;
  } timer;
  // Previous report, accessed only by serialized service turns.
  struct {
    // Report timestamp in monotonic nanoseconds.
    iree_time_t time;
    // Consumed prompt-token counter at the previous report.
    uint64_t prefill;
    // Published output-token counter at the previous report.
    uint64_t output;
  } previous;
  // Application-owned counters copied under mutex before reporting.
  loom_serve_text_heartbeat_snapshot_t snapshot;
} loom_serve_text_heartbeat_t;

// Initializes copied statistics and synchronization without accepting work.
// The application owner updates snapshot under mutex; no callback accesses the
// model. An initialized but unconfigured reporter can be deinitialized.
void loom_serve_text_heartbeat_initialize(
    loom_serve_text_heartbeat_t* heartbeat);

// Configures borrowed I/O and name storage before the first advance. Zero
// interval disables reporting. The owner installs observer under mutex before
// advancing, then clears it only after all accepted work has been consumed.
void loom_serve_text_heartbeat_configure(loom_serve_text_heartbeat_t* heartbeat,
                                         iree_async_proactor_t* proactor,
                                         iree_string_view_t name_json,
                                         iree_duration_t interval);

// Whether a callback or unconsumed outcome still borrows this reporter.
// Reporting alone is not model activity that can free admission capacity.
bool loom_serve_text_heartbeat_is_pending(
    loom_serve_text_heartbeat_t* heartbeat);

// Consumes readiness, reports copied statistics and arms the next deadline.
// Stopping is sticky and cancels the timer through the poll owner. Continue
// advancing through pending work before clearing the observer or destroying
// storage; no task worker or callback waits for another operation.
iree_status_t loom_serve_text_heartbeat_advance(
    loom_serve_text_heartbeat_t* heartbeat, bool stopping, bool* out_progress);

// Publishes final copied statistics and releases synchronization. All accepted
// work and callback outcomes have already been consumed; this never waits.
void loom_serve_text_heartbeat_deinitialize(
    loom_serve_text_heartbeat_t* heartbeat);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_HEARTBEAT_H_
