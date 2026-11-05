// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_TEXT_STATISTICS_H_
#define LOOM_SERVICE_TEXT_STATISTICS_H_

#include "loom/runtime/parameter/activity.h"
#include "loom/runtime/storage/statistics.h"
#include "loom/runtime/telemetry/sample.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_service_text_statistics_t {
  // Current application activity; literals outlive the reporter.
  const char* phase;
  // Start of the current activity, including a possibly unfinished GPU wait.
  iree_time_t phase_start;
  // Most recent completed epoch, or service startup before any model work.
  iree_time_t last_completion;
  // Requests currently holding retained rows.
  iree_host_size_t active_rows;
  // Allocated resident owners, including idle reusable records.
  iree_host_size_t session_capacity;
  // Configured resident ceiling; zero leaves concurrency memory-limited.
  iree_host_size_t max_sessions;
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
  // Cold automatic-cache sizing target; not a reservation or physical usage.
  uint64_t cache_target_bytes;
  // Selected additional initial state and checkpoint-map extent. Resident
  // growth can add COW working set independently of this cold estimate.
  uint64_t cache_additional_bytes;
  // Elastic mutable-state backing; weights and transient workspace are
  // separate.
  loom_runtime_storage_statistics_t state_memory;
  // Prepared parameters, independently reclaimable from retained state.
  loom_runtime_storage_usage_t parameter_memory;
  // Cumulative parameter read/preparation submissions, not device timings.
  loom_runtime_parameter_activity_t parameter_activity;
  // Shared owner accounting, including other models and pooled command scratch.
  loom_runtime_storage_shared_memory_record_t pool_memory;
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
} loom_service_text_statistics_t;

// Projects owner-published facts without traversing live model state.
iree_status_t loom_service_text_sample_statistics(
    const loom_service_text_statistics_t* statistics,
    loom_runtime_telemetry_sample_sink_t sink);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_TEXT_STATISTICS_H_
