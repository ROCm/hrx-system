// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_DIAGNOSTICS_H_
#define EXPERIMENTAL_ID4_PIPELINE_DIAGNOSTICS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Pipeline diagnostic event kind.
typedef enum id4_pipeline_diagnostic_event_kind_e {
  // Lifecycle event such as load, plan, prepare, or issue.
  ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE = 0,
  // Structured plan event.
  ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PLAN = 1,
  // Parameter slab planning or loading event.
  ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB = 2,
} id4_pipeline_diagnostic_event_kind_t;

// Structured diagnostic event emitted by pipeline infrastructure.
typedef struct id4_pipeline_diagnostic_event_t {
  // Kind of event being emitted.
  id4_pipeline_diagnostic_event_kind_t kind;
  // Stage name associated with the event.
  iree_string_view_t stage_name;
  // Stable event key within the stage.
  iree_string_view_t key;
  // Human-readable event message.
  iree_string_view_t message;
} id4_pipeline_diagnostic_event_t;

// Function pointer used to consume a diagnostic event.
typedef iree_status_t(IREE_API_PTR* id4_pipeline_diagnostics_emit_fn_t)(
    void* user_data, const id4_pipeline_diagnostic_event_t* event);

// Caller-owned diagnostic sink.
typedef struct id4_pipeline_diagnostics_sink_t {
  // Callback invoked for each event.
  id4_pipeline_diagnostics_emit_fn_t emit;
  // Caller-owned pointer passed to the callback.
  void* user_data;
} id4_pipeline_diagnostics_sink_t;

// Emits a diagnostic event if |sink| has a callback.
iree_status_t id4_pipeline_diagnostics_emit(
    id4_pipeline_diagnostics_sink_t* sink,
    const id4_pipeline_diagnostic_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_DIAGNOSTICS_H_
