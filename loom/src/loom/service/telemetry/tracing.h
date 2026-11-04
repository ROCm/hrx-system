// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_TELEMETRY_TRACING_H_
#define LOOM_SERVICE_TELEMETRY_TRACING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Service categories share the application's process-wide tracing namespace.
// These describe elapsed operations, independently of worker thread ownership.
typedef enum loom_service_trace_section_e {
  LOOM_SERVICE_TRACE_SECTION_PREPARE = 1,
  LOOM_SERVICE_TRACE_SECTION_WARMUP = 2,
  LOOM_SERVICE_TRACE_SECTION_DEACTIVATE = 3,
  LOOM_SERVICE_TRACE_SECTION_RETIRE = 4,
  LOOM_SERVICE_TRACE_SECTION_EXECUTE = 5,
} loom_service_trace_section_t;

// Publishes category names before observing model lifecycles. Repeated setup
// preserves the same names across multiple owners in one process.
void loom_service_trace_initialize(void);

// Enters an independently overlapping "model-<operation>:<name>" section.
// The provider consumes the deployment name during entry. The operation owner
// leaves the returned ID exactly once at its terminal result, including failure
// or cancellation; no trace-only waiter is needed. Execution sections span an
// accepted model unit through observed completion, not CPU preparation or
// response encoding. Callers guard section storage and entry with IREE_TRACE.
iree_section_id_t loom_service_trace_section_enter(
    loom_service_trace_section_t section, iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVICE_TELEMETRY_TRACING_H_
