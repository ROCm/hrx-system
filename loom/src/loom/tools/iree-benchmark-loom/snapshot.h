// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact benchmark snapshot aggregation over typed benchmark events.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_SNAPSHOT_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_SNAPSHOT_H_

#include "iree/base/api.h"
#include "loom/tools/iree-benchmark-loom/event.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_benchmark_loom_snapshot_sink_t {
  // Opaque aggregation state allocated by initialize and freed by deinitialize.
  void* state;
} iree_benchmark_loom_snapshot_sink_t;

// Initializes an empty compact snapshot aggregator.
//
// The snapshot sink copies the event data needed for final output, so callers
// may append the JSON document after the benchmark run and borrowed event
// payloads have gone out of scope.
iree_status_t iree_benchmark_loom_snapshot_sink_initialize(
    iree_allocator_t allocator,
    iree_benchmark_loom_snapshot_sink_t* out_snapshot);

// Releases all storage owned by |snapshot|.
void iree_benchmark_loom_snapshot_sink_deinitialize(
    iree_benchmark_loom_snapshot_sink_t* snapshot);

// Initializes an event sink that appends lifecycle events into |snapshot|. The
// returned sink borrows |snapshot| until the snapshot is deinitialized.
void iree_benchmark_loom_snapshot_event_sink_initialize(
    iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_benchmark_loom_event_sink_t* out_sink);

// Appends the complete compact JSON snapshot document into |output|.
iree_status_t iree_benchmark_loom_snapshot_sink_append_json(
    const iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_string_builder_t* output);

// Writes the complete compact JSON snapshot document to |path|.
iree_status_t iree_benchmark_loom_snapshot_sink_write(
    const iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_string_view_t path);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_SNAPSHOT_H_
