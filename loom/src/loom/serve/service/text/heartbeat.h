// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_HEARTBEAT_H_
#define LOOM_SERVE_SERVICE_TEXT_HEARTBEAT_H_

#include "iree/base/threading/mutex.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/host/timer.h"
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/serve/telemetry/stream.h"

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
  // Monotonic receipt time of the oldest admitted request, if active_rows > 0.
  iree_time_t oldest_active_arrival;
  // Monotonic receipt time of the oldest queued request, if queued_requests >
  // 0.
  iree_time_t oldest_queued_arrival;
  // Pool accounting in token positions; reserved and resident are disjoint.
  struct {
    // Addressable token capacity; zero selects dense comparison storage.
    iree_host_size_t capacity;
    // Additional positions reserved for active completion, including tail COW.
    iree_host_size_t reserved;
    // Unique resident positions, including idle rows, cached endpoints and
    // pins.
    iree_host_size_t resident;
  } pool;
  // Idle endpoint retention, independent of active row capacity.
  struct {
    // Configured detached endpoint bound.
    iree_host_size_t capacity;
    // Retained endpoints, including at most one temporary admission exchange.
    iree_host_size_t endpoints;
  } session_cache;
  // Intermediate consumed-token prefixes, separate from completed histories.
  struct {
    // Configured evictable frontier bound.
    iree_host_size_t capacity;
    // Retained frontiers, including a capture awaiting readiness.
    iree_host_size_t endpoints;
  } prefix_cache;
  // Host payload owned by explicitly suspended endpoints, not device backing.
  iree_host_size_t checkpoint_host_bytes;
  // Elastic mutable-state backing; weights and transient workspace are
  // separate.
  loom_serve_memory_statistics_t state_memory;
  // Prepared parameters, independently reclaimable from retained state.
  loom_serve_memory_statistics_t weight_memory;
  // Cumulative parameter read/preparation submissions, not device timings.
  loom_serve_weight_activity_t weight_activity;
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
  // Sum of completed model epoch wall times, excluding host text processing.
  iree_duration_t model_duration;
  // Completed speculative work, separate from committed output progress.
  struct {
    // Candidate tokens produced across completed verifications.
    uint64_t proposed_tokens;
    // Draft inputs consumed by the target, excluding each pending anchor.
    uint64_t accepted_inputs;
  } draft;
  // Span count of the most recently issued epoch.
  iree_host_size_t epoch_spans;
  // Input count of the most recently issued epoch, including proposed inputs.
  iree_host_size_t epoch_tokens;
} loom_serve_text_heartbeat_snapshot_t;

typedef struct loom_serve_text_heartbeat_t {
  // Protects copied statistics, never model work.
  iree_slim_mutex_t mutex;
  // Periodic readiness and callback lifetime, independent of model counters.
  loom_serve_timer_t timer;
  // Borrowed owner-only stream, alive through deinitialization.
  loom_serve_telemetry_stream_t* stream;
  // Borrowed owner counters independent of reporting interval and readers.
  const loom_serve_telemetry_counters_t* counters;
  // Application-owned counters copied under mutex before reporting.
  loom_serve_text_heartbeat_snapshot_t snapshot;
} loom_serve_text_heartbeat_t;

// Initializes copied statistics and synchronization without accepting work.
// The application owner updates snapshot under mutex; no callback accesses the
// model. An initialized but unconfigured reporter can be deinitialized.
void loom_serve_text_heartbeat_initialize(
    loom_serve_text_heartbeat_t* heartbeat);

// Copies the last published counters without accessing model/session state.
// Safe while a finite CPU task owns mutable rows or device work is pending.
loom_serve_text_heartbeat_snapshot_t loom_serve_text_heartbeat_read(
    loom_serve_text_heartbeat_t* heartbeat);

// Configures borrowed I/O and observation storage before the first advance.
// Zero interval disables reporting. The owner installs the timer observer
// before advancing, then clears it only after all accepted work has been
// consumed.
void loom_serve_text_heartbeat_configure(
    loom_serve_text_heartbeat_t* heartbeat, iree_async_proactor_t* proactor,
    loom_serve_telemetry_stream_t* stream,
    const loom_serve_telemetry_counters_t* counters, iree_duration_t interval);

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

#endif  // LOOM_SERVE_SERVICE_TEXT_HEARTBEAT_H_
