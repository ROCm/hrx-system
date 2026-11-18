// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_PARTITION_H_
#define LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_PARTITION_H_

#include "loom/transforms/kernel/private_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_private_storage_interval_t {
  // Direct memory access, or NULL for a cell required by a later copy.
  loom_private_storage_access_t* access;
  // Allocation root supplied by shared region facts.
  loom_private_storage_allocation_t* allocation;
  // Exact inclusive root-relative byte offset.
  int64_t begin;
  // Exact exclusive root-relative byte offset.
  int64_t end;
  // Accessed scalar representation.
  loom_type_t type;
} loom_private_storage_interval_t;

// Construction-only scalar intervals. The final partition coalesces equal cells
// and binds each direct access to its cell in the plan.
typedef struct loom_private_storage_intervals_t {
  // Intervals in discovery order until partitioning.
  loom_private_storage_interval_t* values;
  // Number of populated intervals.
  iree_host_size_t count;
  // Arena-backed interval capacity.
  iree_host_size_t capacity;
} loom_private_storage_intervals_t;

// Appends a typed access or projected demand in the construction arena.
static inline iree_status_t loom_private_storage_intervals_append(
    iree_arena_allocator_t* arena, loom_private_storage_intervals_t* intervals,
    loom_private_storage_interval_t interval) {
  if (intervals->count == intervals->capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, intervals->count, intervals->count + 1,
                              sizeof(*intervals->values), &intervals->capacity,
                              (void**)&intervals->values));
  }
  intervals->values[intervals->count++] = interval;
  return iree_ok_status();
}

// Completes the sparse partition and binds each transfer to its source cells.
// The returned plan retains no references to the construction scratch.
iree_status_t loom_private_storage_partition(
    loom_private_storage_plan_t* plan,
    loom_private_storage_intervals_t* intervals,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_PRIVATE_STORAGE_PARTITION_H_
