// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_TELEMETRY_SAMPLE_H_
#define LOOM_RUNTIME_TELEMETRY_SAMPLE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_runtime_telemetry_metric_kind_e {
  // Current ownership or occupancy; samples may decrease.
  LOOM_RUNTIME_TELEMETRY_GAUGE = 0,
  // Cumulative work within the source incarnation, independent of readers.
  LOOM_RUNTIME_TELEMETRY_COUNTER = 1,
} loom_runtime_telemetry_metric_kind_t;

// Immutable source-owned numeric semantics. Names are stable lowercase ASCII
// identifiers; units describe the integer value, without implicit conversion.
// A topic's field order and meaning are stable for its schema version.
typedef struct loom_runtime_telemetry_metric_t {
  // Field identifier, unique within the topic.
  const char* name;
  // Unit such as requests, tokens, bytes, pixels, or nanoseconds.
  const char* unit;
  // Gauge versus cumulative counter; neither implies a retirement reduction.
  loom_runtime_telemetry_metric_kind_t kind;
} loom_runtime_telemetry_metric_t;

typedef struct loom_runtime_telemetry_topic_t {
  // Stable source-owned namespace, independent of a transport or model name.
  const char* name;
  // Schema revision for the field meanings and order.
  uint32_t version;
  // Number of fields and corresponding sampled values.
  iree_host_size_t count;
  // Immutable field descriptors, live through the synchronous sink call.
  const loom_runtime_telemetry_metric_t* fields;
} loom_runtime_telemetry_topic_t;

// Synchronous consumer of copied owner facts. Descriptors and values are
// borrowed only during emit; an exporter copies or formats before returning.
// The callback can fail only for its output/allocation, not model execution.
// Providers sample on their serialized owner, never wait for a device, and
// never expose task-borrowed state. No history is synthesized from samples.
// This requested observation contract is not a hot event-publication callback.
typedef struct loom_runtime_telemetry_sample_sink_t {
  // Consumer context, borrowed through the sampling call.
  void* self;
  // Consumes exactly topic->count unsigned 64-bit values.
  iree_status_t (*emit)(void* self, const loom_runtime_telemetry_topic_t* topic,
                        const uint64_t* values);
} loom_runtime_telemetry_sample_sink_t;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_TELEMETRY_SAMPLE_H_
