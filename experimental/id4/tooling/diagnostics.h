// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_TOOLING_DIAGNOSTICS_H_
#define EXPERIMENTAL_ID4_TOOLING_DIAGNOSTICS_H_

#include <stdio.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// File-backed diagnostics sink writing one JSON object per event.
typedef struct id4_tooling_diagnostics_file_sink_t {
  // Allocator used for owned path storage.
  iree_allocator_t host_allocator;
  // Owned path to the JSONL event log.
  iree_string_view_t event_log_path;
  // Open event log file, or NULL when uninitialized.
  FILE* event_log_file;
} id4_tooling_diagnostics_file_sink_t;

// Initializes |out_sink| to write diagnostic events into |directory|.
iree_status_t id4_tooling_diagnostics_file_sink_initialize(
    iree_string_view_t directory, iree_allocator_t host_allocator,
    id4_tooling_diagnostics_file_sink_t* out_file_sink,
    id4_pipeline_diagnostics_sink_t* out_sink);

// Flushes and closes |file_sink|.
iree_status_t id4_tooling_diagnostics_file_sink_deinitialize(
    id4_tooling_diagnostics_file_sink_t* file_sink);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_TOOLING_DIAGNOSTICS_H_
