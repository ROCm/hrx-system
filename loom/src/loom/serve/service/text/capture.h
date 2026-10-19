// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CAPTURE_H_
#define LOOM_SERVE_SERVICE_TEXT_CAPTURE_H_

#include "iree/vm/buffer.h"
#include "loom/serve/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independently owned opaque output from one known-input epoch. Span ordinals
// preserve the submitted cohort order; byte layout belongs to the source.
// Captures retain their allocation and execution domain, never model weights,
// VM code, or mutable rows. The application device group outlives consumers.
typedef struct loom_serve_text_capture_t loom_serve_text_capture_t;

// Validates a source plan before row mutation or device submission. The plan is
// an array of LE i64 triples [span index, residual byte offset, byte length],
// grouped in span order with at least one nonempty range for each span. Ranges
// are concatenated in source order without interpreting their element types.
// Copies all metadata; source may immediately reuse its plan buffer.
iree_status_t loom_serve_text_capture_create(
    iree_vm_buffer_t* plan, iree_host_size_t span_count,
    iree_device_size_t residual_length, iree_allocator_t host_allocator,
    loom_serve_text_capture_t** out_capture);

// Allocates independent pooled storage and queues capture after the producer,
// before any later epoch can overwrite residual. Called once by its serialized
// owner. Partial submission retains queue ownership even on failure; caller
// release is nonblocking and never substitutes for retirement.
iree_status_t loom_serve_text_capture_submit(loom_serve_text_capture_t* capture,
                                             loom_serve_execution_t* execution,
                                             iree_hal_buffer_t* residual);

void loom_serve_text_capture_retain(loom_serve_text_capture_t* capture);
void loom_serve_text_capture_release(loom_serve_text_capture_t* capture);

// Borrowed readiness and device views, valid through capture release. A device
// consumer imports readiness and retains its view through queue retirement.
loom_serve_execution_frontier_t loom_serve_text_capture_ready(
    const loom_serve_text_capture_t* capture);
iree_hal_buffer_ref_t loom_serve_text_capture_span(
    const loom_serve_text_capture_t* capture, iree_host_size_t span);

// Explicit host materialization for transport consumers. It copies metadata
// and owns exact byte storage independently of the model and device capture.
typedef struct loom_serve_text_capture_snapshot_t
    loom_serve_text_capture_snapshot_t;

// Enqueues an independent feedback download. Destination storage survives early
// release or failed readiness until the accepted queue capture actually
// retires. On success span bytes become readable after the returned frontier
// succeeds.
iree_status_t loom_serve_text_capture_download(
    loom_serve_text_capture_t* capture,
    loom_serve_text_capture_snapshot_t** out_snapshot,
    loom_serve_execution_frontier_t* out_frontier);
void loom_serve_text_capture_snapshot_release(
    loom_serve_text_capture_snapshot_t* snapshot);
iree_const_byte_span_t loom_serve_text_capture_snapshot_span(
    const loom_serve_text_capture_snapshot_t* snapshot, iree_host_size_t span);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_CAPTURE_H_
