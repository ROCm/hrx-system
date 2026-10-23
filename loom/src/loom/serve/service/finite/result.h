// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_RESULT_H_
#define LOOM_SERVE_SERVICE_FINITE_RESULT_H_

#include "iree/vm/buffer.h"
#include "loom/serve/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable cohort metadata and independently retained device output. The
// result owns its logical output storage, not the source model, commands, or
// parameters.
// Device consumers import ready before reading and retain their buffer views
// through actual queue retirement. The application device group outlives all
// results and their consumers; a shared group alone grants no device access.
typedef struct loom_serve_finite_result_t loom_serve_finite_result_t;

// Checks an external source plan once and copies its row metadata. The LE i64
// header is [physical output bytes, result bytes, request count, row count],
// followed by [first row, row count, input tokens] per request, then
// [byte offset, F32 count] per row. Rows lie within the result prefix of
// output. expected_rows is protocol-derived, one entry per original request. A
// newly created result has no device storage until the producer calls bind.
// Copying separates retained rows from source buffers reused by a later
// invocation.
iree_status_t loom_serve_finite_result_create(
    iree_vm_buffer_t* plan, iree_host_size_t output_capacity,
    iree_host_size_t result_capacity, iree_host_size_t request_count,
    const iree_host_size_t* expected_rows, iree_allocator_t host_allocator,
    loom_serve_finite_result_t** out_result);

// Captures the accepted producer frontier and retains the exact output view,
// including its offset and storage-release obligations. The caller supplies
// storage independent of the producer's retirement-tracked views: a queue
// allocation root or an owning logical suballocation, never a view whose
// release keeps the producer model alive. Called once after successful
// submission; output has the checked physical extent. Its source pool outlives
// the result and every later buffer consumer. The producer separately retains
// its tracked view through completion or terminal retirement.
void loom_serve_finite_result_bind(loom_serve_finite_result_t* result,
                                   loom_serve_execution_t* execution,
                                   iree_hal_buffer_t* output);

void loom_serve_finite_result_retain(loom_serve_finite_result_t* result);

// Releases caller ownership without waiting. Queue-captured views preserve
// the exact logical storage. The storage owner keeps its source pool alive
// until the final capture retires. NULL is accepted. There is no result cache
// in this layer.
void loom_serve_finite_result_release(loom_serve_finite_result_t* result);

// Source-declared output extent, including padding, not allocator overhead.
iree_host_size_t loom_serve_finite_result_output_length(
    const loom_serve_finite_result_t* result);

// Live result prefix; also the exact size of an explicit host snapshot.
iree_host_size_t loom_serve_finite_result_length(
    const loom_serve_finite_result_t* result);

// Borrowed producer readiness, stable through result release. Import this
// frontier before using device rows; no host wait is required for composition.
loom_serve_execution_frontier_t loom_serve_finite_result_ready(
    const loom_serve_finite_result_t* result);

// Trusted queries into the checked plan. Request and row ordinals retain the
// original cohort associations even when other requests have been cancelled.
uint64_t loom_serve_finite_result_input_tokens(
    const loom_serve_finite_result_t* result, iree_host_size_t request);
iree_hal_buffer_ref_t loom_serve_finite_result_row(
    const loom_serve_finite_result_t* result, iree_host_size_t request,
    iree_host_size_t row);

// Explicit host materialization for consumers such as HTTP vector encoders.
// Device-to-device composition uses the result directly and never creates one.
typedef struct loom_serve_finite_snapshot_t loom_serve_finite_snapshot_t;

// Copies checked row metadata, allocates exact host result storage, and submits
// one download on the producer's feedback branch. Called by the serialized
// owner immediately after submission when host output is required. Queue
// capture protects the destination even after failed readiness or early caller
// release. On success the caller owns the snapshot; its rows become readable
// after successful returned readiness. Snapshot readers retain no device output
// after the download capture retires.
iree_status_t loom_serve_finite_result_download(
    const loom_serve_finite_result_t* result,
    loom_serve_finite_snapshot_t** out_snapshot,
    loom_serve_execution_frontier_t* out_frontier);

// Releases caller ownership, independently of any accepted download capture.
// NULL is accepted. Row borrowers must have retired first.
void loom_serve_finite_snapshot_release(loom_serve_finite_snapshot_t* snapshot);

// Queries the snapshot's copied metadata without retaining device storage.
uint64_t loom_serve_finite_snapshot_input_tokens(
    const loom_serve_finite_snapshot_t* snapshot, iree_host_size_t request);
iree_const_byte_span_t loom_serve_finite_snapshot_row(
    const loom_serve_finite_snapshot_t* snapshot, iree_host_size_t request,
    iree_host_size_t row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_FINITE_RESULT_H_
