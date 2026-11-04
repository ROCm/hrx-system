// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_TELEMETRY_LATENCY_H_
#define LOOM_SERVICE_TELEMETRY_LATENCY_H_

#include "loom/runtime/telemetry/stream.h"
#include "loom/service/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { LOOM_SERVICE_TELEMETRY_LATENCY_BUCKET_COUNT = 20 };

// Single-owner duration distribution. All-zero storage is an empty histogram.
// Disjoint buckets keep each observation to one increment; readers form
// cumulative counts without modifying the producer's storage.
typedef struct loom_service_telemetry_latency_histogram_t {
  // Counts for increasing inclusive upper bounds; the last bucket is unbounded.
  uint64_t buckets[LOOM_SERVICE_TELEMETRY_LATENCY_BUCKET_COUNT];
  // Sum of all observed durations in nanoseconds, independent of bucketing.
  uint64_t sum_ns;
} loom_service_telemetry_latency_histogram_t;

// Model protocol receipt through observed request milestones, not peer receipt
// or device time. Stored separately from compact work counters and event
// records. Updated and copied only by the serialized serving owner, without
// allocation, locking, formatting, or consulting telemetry readers.
typedef struct loom_service_telemetry_latency_t {
  // Exactly one duration per terminal request, indexed by its existing outcome.
  loom_service_telemetry_latency_histogram_t
      outcomes[LOOM_SERVICE_TELEMETRY_FAILED + 1];
  // One duration per request whose first meaningful output reached the carrier,
  // regardless of its eventual terminal outcome. Empty/error responses do not
  // contribute a sample.
  loom_service_telemetry_latency_histogram_t first_output;
} loom_service_telemetry_latency_t;

// Returns the inclusive upper bound in nanoseconds for a finite bucket. The
// caller supplies an index below LATENCY_BUCKET_COUNT - 1; the final unbounded
// bucket has no numeric bound.
uint64_t loom_service_telemetry_latency_bound_ns(iree_host_size_t index);

// Observes one nonnegative duration at its owning milestone. The caller
// supplies trusted request lifetime state and records each milestone exactly
// once.
void loom_service_telemetry_latency_observe(
    loom_service_telemetry_latency_histogram_t* histogram,
    uint64_t duration_ns);

// Records the first meaningful output accepted by the service consumer, using
// one timestamp for the histogram and event. The serialized owner calls this
// once after a successful payload copy; framing, empty deltas, errors, and
// control responses are not model output. For HTTP this is carrier acceptance,
// not peer receipt. A later failure does not undo this milestone.
void loom_service_telemetry_latency_publish_output(
    loom_service_telemetry_latency_t* latency,
    loom_runtime_telemetry_stream_t* stream, uint64_t request,
    iree_time_t arrival_time);

// Adds a complete owner snapshot, including buckets and exact sums. Source and
// target are distinct; neither is being concurrently updated.
void loom_service_telemetry_latency_merge(
    loom_service_telemetry_latency_t* target,
    const loom_service_telemetry_latency_t* source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_TELEMETRY_LATENCY_H_
