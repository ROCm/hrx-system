// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_INSPECTION_H_
#define LOOM_SERVE_SERVICE_INSPECTION_H_

#include "loom/serve/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scheduler-owned metadata, independent of mutable CPU/device task payloads.
typedef struct loom_serve_http_request_snapshot_t {
  // Existing model-protocol request ID; zero marks an unused owner slot.
  uint64_t id;
  // Text session lifetime, or zero when not assigned/applicable.
  uint64_t session;
  // Ownership phase with process lifetime, not a hardware utilization claim.
  const char* phase;
  // Immutable protocol receipt time, including time before model admission.
  iree_time_t arrival_time;
  // Completed work published by the owner, never read from in-flight outputs.
  loom_serve_telemetry_counters_t counters;
} loom_serve_http_request_snapshot_t;

typedef struct loom_serve_http_session_snapshot_t {
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
  loom_serve_telemetry_counters_t counters;
} loom_serve_http_session_snapshot_t;

// Queries run only on the serialized service owner. They copy metadata without
// allocating, submitting work, waiting, or accessing task-owned mutable state.
// False means index is past the table end; true with ID zero is an empty slot.
// Returned views remain borrowed only until that owner turn ends. Table indices
// are not lifetime identities: each page observes current ownership, and churn
// may move entries between pages. NULL callbacks mean unavailable capability.
typedef struct loom_serve_http_inspection_t {
  // Model-admitted requests, excluding router-held and tokenizer utilities.
  bool (*request_at)(void* self, iree_host_size_t index,
                     loom_serve_http_request_snapshot_t* out_snapshot);
  // Retained logical sessions, independent of reusable device rows and pins.
  bool (*session_at)(void* self, iree_host_size_t index,
                     loom_serve_http_session_snapshot_t* out_snapshot);
} loom_serve_http_inspection_t;

typedef enum loom_serve_http_inspection_kind_e {
  LOOM_SERVE_HTTP_INSPECT_REQUESTS,
  LOOM_SERVE_HTTP_INSPECT_SESSIONS,
} loom_serve_http_inspection_kind_t;

// Parses a nonempty decimal owner-table offset without signs or wrapping.
bool loom_serve_http_inspection_offset(iree_string_view_t value,
                                       iree_host_size_t* out_offset);

// Appends data and next_offset JSON members, each with a leading comma. The
// caller already selected a present capability and owns the model through this
// call. Examines at most 64 slots including holes; next_offset is null once the
// table end is observed. No query or record survives the call. |now| is shared
// by every age in this page. Failure can only come from output allocation.
iree_status_t loom_serve_http_inspection_format(
    const loom_serve_http_inspection_t* inspection, void* self,
    loom_serve_http_inspection_kind_t kind, iree_host_size_t offset,
    iree_time_t now, iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_INSPECTION_H_
