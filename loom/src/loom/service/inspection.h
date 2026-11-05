// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_INSPECTION_H_
#define LOOM_SERVICE_INSPECTION_H_

#include "loom/service/telemetry/record.h"

// Copied scheduler ownership, not hardware utilization. Finite active requests
// include preparation, capacity waiting and encoding; text active requests
// hold admitted rows. No request content or borrowed mutable buffers escape.
typedef struct loom_service_snapshot_t {
  // Modality-owned phase label with process lifetime.
  const char* phase;
  // Requests waiting outside the active invocation/cohort.
  iree_host_size_t queued_requests;
  // Requests retained by the active invocation/cohort, including reset peers
  // whose accepted work still has to retire.
  iree_host_size_t active_requests;
  // Nanoseconds since the oldest queued request was accepted; zero if empty.
  iree_duration_t oldest_queued_age;
  // Nanoseconds since the oldest active request was accepted, including prior
  // queue residence; zero if empty. This is ownership age, not device time.
  iree_duration_t oldest_active_age;
} loom_service_snapshot_t;

// Scheduler-owned metadata, independent of mutable CPU/device task payloads.
typedef struct loom_service_request_snapshot_t {
  // Existing model-protocol request ID; zero marks an unused owner slot.
  uint64_t id;
  // Text session lifetime, or zero when not assigned/applicable.
  uint64_t session;
  // Ownership phase with process lifetime, not a hardware utilization claim.
  const char* phase;
  // Immutable protocol receipt time, including time before model admission.
  iree_time_t arrival_time;
  // Completed work published by the owner, never read from in-flight outputs.
  loom_service_telemetry_counters_t counters;
} loom_service_request_snapshot_t;

typedef struct loom_service_session_snapshot_t {
  // Logical lifetime ID; zero marks an unused or unpublished owner slot.
  uint64_t id;
  // Client cache key borrowed only through the serialized inspection call.
  iree_string_view_t name;
  // Current admitted request ID; zero means an idle retained history.
  uint64_t request;
  // Storage owner label with process lifetime, independent of native backing.
  const char* storage;
  // Last owner-published completed position, not an in-flight device value.
  uint64_t position;
  // Referenced token positions, including shared prefixes; not unique bytes.
  uint64_t resident_tokens;
  // Page-rounded active completion bound; zero for idle histories.
  uint64_t reserved_tokens;
  // Cumulative completed work across retained continuations.
  loom_service_telemetry_counters_t counters;
} loom_service_session_snapshot_t;

// Queries run only on the serialized service owner. They copy metadata without
// allocating, submitting work, waiting, or accessing task-owned mutable state.
// False means index is past the table end; true with ID zero is an empty slot.
// Returned views remain borrowed only until that owner turn ends. Table indices
// are not lifetime identities: each page observes current ownership, and churn
// may move entries between pages. NULL callbacks mean unavailable capability.
typedef struct loom_service_inspection_t {
  // Model-admitted requests, excluding frontend-held and tokenizer utilities.
  bool (*request_at)(void* self, iree_host_size_t index,
                     loom_service_request_snapshot_t* out_snapshot);
  // Retained logical sessions, independent of reusable device rows and pins.
  bool (*session_at)(void* self, iree_host_size_t index,
                     loom_service_session_snapshot_t* out_snapshot);
} loom_service_inspection_t;

#endif  // LOOM_SERVICE_INSPECTION_H_
