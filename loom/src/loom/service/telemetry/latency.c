// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/service/telemetry/latency.h"

#include <string.h>

#include "loom/service/telemetry/record.h"

static const uint64_t loom_service_telemetry_latency_bounds_ns[] = {
    1000000,      2500000,      5000000,      10000000,      25000000,
    50000000,     100000000,    250000000,    500000000,     1000000000,
    2500000000,   5000000000,   10000000000,  25000000000,   50000000000,
    100000000000, 250000000000, 500000000000, 1000000000000,
};
static_assert(IREE_ARRAYSIZE(loom_service_telemetry_latency_bounds_ns) + 1 ==
                  LOOM_SERVICE_TELEMETRY_LATENCY_BUCKET_COUNT,
              "finite bounds plus the unbounded bucket");

uint64_t loom_service_telemetry_latency_bound_ns(iree_host_size_t index) {
  return loom_service_telemetry_latency_bounds_ns[index];
}

void loom_service_telemetry_latency_observe(
    loom_service_telemetry_latency_histogram_t* histogram,
    uint64_t duration_ns) {
  iree_host_size_t first = 0;
  iree_host_size_t last =
      IREE_ARRAYSIZE(loom_service_telemetry_latency_bounds_ns);
  while (first < last) {
    const iree_host_size_t middle = first + (last - first) / 2;
    if (duration_ns <= loom_service_telemetry_latency_bounds_ns[middle]) {
      last = middle;
    } else {
      first = middle + 1;
    }
  }
  ++histogram->buckets[first];
  histogram->sum_ns += duration_ns;
}

void loom_service_telemetry_latency_publish_output(
    loom_service_telemetry_latency_t* latency,
    loom_runtime_telemetry_stream_t* stream, uint64_t request,
    iree_time_t arrival_time) {
  loom_runtime_telemetry_record_t record = {
      .time_ns = iree_time_now(),
      .request = request,
      .event = &LOOM_SERVICE_TELEMETRY_FIRST_OUTPUT};
  loom_service_telemetry_first_output_record_t telemetry_payload = {0};
  telemetry_payload.duration_ns = record.time_ns - arrival_time;
  loom_service_telemetry_latency_observe(&latency->first_output,
                                         telemetry_payload.duration_ns);
  memcpy(record.data.words, &telemetry_payload, sizeof(telemetry_payload));
  loom_runtime_telemetry_stream_append(stream, &record);
}

static void loom_service_telemetry_latency_histogram_merge(
    loom_service_telemetry_latency_histogram_t* target,
    const loom_service_telemetry_latency_histogram_t* source) {
  for (iree_host_size_t i = 0; i < LOOM_SERVICE_TELEMETRY_LATENCY_BUCKET_COUNT;
       ++i) {
    target->buckets[i] += source->buckets[i];
  }
  target->sum_ns += source->sum_ns;
}

void loom_service_telemetry_latency_merge(
    loom_service_telemetry_latency_t* target,
    const loom_service_telemetry_latency_t* source) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(target->outcomes); ++i) {
    loom_service_telemetry_latency_histogram_merge(&target->outcomes[i],
                                                   &source->outcomes[i]);
  }
  loom_service_telemetry_latency_histogram_merge(&target->first_output,
                                                 &source->first_output);
}
