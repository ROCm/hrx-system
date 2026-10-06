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

// Captures the concatenated ranges after preceding execution work. The caller
// excludes mutation until return. All accepted transfers retire before return,
// including on failed readiness; the image is published only on success.
// Metadata and roots are borrowed only for the call. This is a cold, blocking
// maintenance boundary, not an inference hot path.
iree_status_t loom_serve_snapshot_capture(
    loom_serve_execution_t* execution, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* buffers, iree_host_size_t range_count,
    const loom_serve_snapshot_range_t* ranges,
    loom_serve_snapshot_t** out_snapshot, iree_allocator_t host_allocator);

// Restores into caller-admitted backed ranges with the same logical lengths
// and order used for capture. Physical buffers and offsets may differ. The
// caller excludes consumers until success and keeps the image on failure.
// Platform/submission failure is terminal, not a partially resumed session.
iree_status_t loom_serve_snapshot_restore(
    const loom_serve_snapshot_t* snapshot, loom_serve_execution_t* execution,
    iree_host_size_t buffer_count, iree_hal_buffer_t* const* buffers,
    iree_host_size_t range_count, const loom_serve_snapshot_range_t* ranges);

// Retained host payload bytes, excluding the small ownership header. Null is
// accepted and reports zero; these bytes are not device pool commitment.
iree_host_size_t loom_serve_snapshot_size(
    const loom_serve_snapshot_t* snapshot);
void loom_serve_snapshot_destroy(loom_serve_snapshot_t* snapshot);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_STORAGE_SNAPSHOT_H_
