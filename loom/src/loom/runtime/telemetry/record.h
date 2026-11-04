// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_TELEMETRY_RECORD_H_
#define LOOM_RUNTIME_TELEMETRY_RECORD_H_

#include "iree/base/api.h"
#include "loom/runtime/telemetry/field.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_runtime_telemetry_record_t loom_runtime_telemetry_record_t;

// Immutable event schema with process lifetime. Interpretation runs on copied
// records in consumers, never during publication. It exposes borrowed native
// fields and propagates only the consumer's output or allocation failure.
typedef struct loom_runtime_telemetry_event_t {
  // Stable event name, independent of deployment/model identifiers.
  const char* name;
  // Consumer-only native interpretation of the copied payload.
  iree_status_t (*read_fields)(const loom_runtime_telemetry_record_t* record,
                               loom_runtime_telemetry_field_sink_t sink);
} loom_runtime_telemetry_event_t;

// Fixed-size owned observation. Its schema interprets opaque payload bytes only
// after a consumer copies the record. This is an in-process representation, not
// a portable binary wire format; publication never formats or consults a
// reader. Explicit alignment keeps its extent a whole number of atomic words
// even on ABIs whose pointers and uint64_t have four-byte alignment.
struct iree_alignas(8) loom_runtime_telemetry_record_t {
  // Monotonic host timestamp in nanoseconds, not a device-clock timestamp.
  iree_time_t time_ns;
  // Producer-scoped lifetime ID; zero for observations without a request.
  uint64_t request;
  // Process-lifetime schema, safe to dereference after the producer retires.
  const loom_runtime_telemetry_event_t* event;
  // Source-owned payload; unused bytes are initialized by the producer.
  struct {
    // Native bytes decoded via memcpy, without retaining producer storage.
    uint64_t words[16];
  } data;
};

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_TELEMETRY_RECORD_H_
