// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_STATISTICS_H_
#define LOOM_RUNTIME_STORAGE_STATISTICS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive usage across distinct reservations or banks on one serialized owner.
// Views borrowing overlapping owners cannot be summed. Historical peaks are
// not additive and belong to the storage owner that observes their transitions.
typedef struct loom_runtime_storage_usage_t {
  // Reserved virtual bytes, including final slab alignment.
  uint64_t reserved_bytes;
  // Unique backing bytes currently published by the owner.
  uint64_t committed_bytes;
  // Budget charged to accepted commitments not yet published by the owner.
  uint64_t pending_bytes;
  // Backing selected for return, until logical release is published.
  uint64_t releasing_bytes;
  // Cumulative backing returns published by the owner.
  uint64_t released_bytes;
} loom_runtime_storage_usage_t;

// Statistics maintained by one storage owner, not reconstructed from views of
// independently activated or retired banks.
typedef struct loom_runtime_storage_statistics_t {
  // Reserved virtual bytes, including final slab alignment.
  uint64_t reserved_bytes;
  // Unique backing bytes; pool-wide queries include cached idle slabs and
  // pending or abandoned native retirement. Per-reservation groups contain
  // only live slabs.
  uint64_t committed_bytes;
  // Pool-only committed charge whose native backing could not be returned.
  // Monotonic; neither reusable nor pending, and not attributed to live roots.
  uint64_t abandoned_bytes;
  // Physically committed idle slabs available to any reservation in the pool.
  // Only pool-wide queries include this shared cache; per-reservation is zero.
  uint64_t cached_bytes;
  // Budget charged to accepted native commitments not yet published by owner.
  uint64_t pending_bytes;
  // Backing selected for return. Per-buffer accounting ends at logical return;
  // pool-wide accounting also includes outstanding native release.
  uint64_t releasing_bytes;
  // Maximum simultaneously committed physical bytes.
  uint64_t peak_bytes;
  // Cumulative logical backing returns by reservations, including cache reuse.
  // This is not cumulative native disposal or available physical capacity.
  uint64_t released_bytes;
} loom_runtime_storage_statistics_t;

typedef struct loom_runtime_storage_shared_memory_record_t {
  // Entire shared device owner's retained backing, not this model's charge.
  uint64_t retained_committed_bytes;
  // Physical workspace backing, including reusable idle ranges.
  uint64_t workspace_committed_bytes;
  // Workspace backing charged to live reservations.
  uint64_t workspace_live_bytes;
  // Shared workspace slab count.
  uint64_t workspace_slab_count;
  // Cumulative successful workspace reuse across all device users.
  uint64_t workspace_reuse_count;
} loom_runtime_storage_shared_memory_record_t;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_STORAGE_STATISTICS_H_
