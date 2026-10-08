// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_STORAGE_SNAPSHOT_H_
#define EXPERIMENTAL_LOOM_SERVE_STORAGE_SNAPSHOT_H_

#include "experimental/loom_serve/runtime/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packed host-owned contents in caller-defined logical order. No device
// addresses or physical block identities survive in a completed snapshot.
typedef struct loom_serve_snapshot_t loom_serve_snapshot_t;

// A validated range in a caller-owned logical transfer plan. Buffer indices
// address the supplied exported roots, including their retirement tracking.
typedef struct loom_serve_snapshot_range_t {
  // Root slot in the buffer table.
  iree_host_size_t buffer_index;
  // Byte offset relative to that root view.
  iree_device_size_t offset;
  // Nonzero number of bytes packed at this logical position.
  iree_device_size_t length;
} loom_serve_snapshot_range_t;

// Enqueues capture of the concatenated ranges after preceding execution work.
// Returns an owned snapshot on successful submission, without waiting. Its
// payload is usable only after the resulting execution frontier succeeds.
// The caller retains source backing and excludes mutation until that frontier
// completes. Failed readiness does not authorize unmapping source backing.
// Metadata and root handles are borrowed only during this call. Accepted queue
// operations retain the payload even if submission fails or the caller releases
// its snapshot reference before completion.
iree_status_t loom_serve_snapshot_capture(
    loom_serve_execution_t* execution, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* buffers, iree_host_size_t range_count,
    const loom_serve_snapshot_range_t* ranges,
    loom_serve_snapshot_t** out_snapshot, iree_allocator_t host_allocator);

// Enqueues restore of a completed snapshot into caller-admitted backed ranges
// with the same logical lengths and order used for capture. Physical buffers
// and offsets may differ. Returns without waiting; queued operations retain
// the payload independently of caller ownership. The caller retains target
// backing and orders consumers after the resulting execution frontier.
// Platform/submission failure is terminal, not a partially resumed session.
iree_status_t loom_serve_snapshot_restore(
    loom_serve_snapshot_t* snapshot, loom_serve_execution_t* execution,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers,
    iree_host_size_t range_count, const loom_serve_snapshot_range_t* ranges);

// Retained host payload bytes, excluding the small ownership header. Null is
// accepted and reports zero; these bytes are not device pool commitment.
iree_host_size_t loom_serve_snapshot_size(
    const loom_serve_snapshot_t* snapshot);

// Releases caller ownership without waiting for queued captures or restores.
// The host payload is freed after its final queue/caller reference. Accepts
// NULL.
void loom_serve_snapshot_release(loom_serve_snapshot_t* snapshot);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_STORAGE_SNAPSHOT_H_
