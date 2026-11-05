// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_STORAGE_SNAPSHOT_H_
#define LOOM_SERVE_RUNTIME_STORAGE_SNAPSHOT_H_

#include "loom/serve/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Packed host-owned contents in caller-defined logical order. No device
// addresses or physical block identities survive in a completed snapshot.
typedef struct loom_serve_snapshot_t loom_serve_snapshot_t;

// One exported root and the execution domain ordering its transfers. Multiple
// roots may share a domain or physical pool without changing logical byte
// order.
typedef struct loom_serve_snapshot_buffer_t {
  // Borrowed root, including the caller's retirement tracking; NULL omits it.
  iree_hal_buffer_t* buffer;
  // Borrowed domain whose queues can access this root, live through retirement.
  loom_serve_execution_t* execution;
} loom_serve_snapshot_buffer_t;

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

// Enqueues capture of the concatenated ranges after each root's preceding work.
// Returns an owned snapshot on successful submission, without waiting. Its
// payload is usable only after every participating execution frontier succeeds.
// The caller retains source backing and excludes mutation until all frontiers
// complete. Failed readiness does not authorize unmapping source backing.
// Metadata and root handles are borrowed only during this call. Accepted queue
// operations retain the payload even if submission fails or the caller releases
// its snapshot reference before completion.
iree_status_t loom_serve_snapshot_capture(
    iree_host_size_t buffer_count, const loom_serve_snapshot_buffer_t* buffers,
    iree_host_size_t range_count, const loom_serve_snapshot_range_t* ranges,
    iree_allocator_t host_allocator, loom_serve_snapshot_t** out_snapshot);

// Enqueues restore of the captured logical byte sequence into caller-admitted
// backed ranges. Their total length must match the snapshot; range
// partitioning, physical buffers, owners, and offsets may differ. Each target
// domain must already follow capture completion through its execution
// dependencies or an observed completed snapshot. Returns without waiting;
// queued operations retain the payload independently of caller ownership. The
// caller retains target backing and orders consumers after every participating
// execution frontier. Platform/submission failure is terminal, not a partially
// resumed session.
iree_status_t loom_serve_snapshot_restore(
    loom_serve_snapshot_t* snapshot, iree_host_size_t buffer_count,
    const loom_serve_snapshot_buffer_t* buffers, iree_host_size_t range_count,
    const loom_serve_snapshot_range_t* ranges);

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

#endif  // LOOM_SERVE_RUNTIME_STORAGE_SNAPSHOT_H_
