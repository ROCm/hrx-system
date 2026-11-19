// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_MEMORY_RESERVATION_H_
#define LOOM_RUNTIME_STORAGE_MEMORY_RESERVATION_H_

#include "iree/hal/memory/tlsf.h"
#include "loom/runtime/storage/backing_directory.h"
#include "loom/runtime/storage/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mapping serialization and failure are properties of the native reservation,
// including when several independently resident buffers occupy its ranges.
typedef struct loom_runtime_memory_target_t {
  // Retained last mapping completion; source allocation remains independent.
  iree_hal_semaphore_t* mapping_completion;
  // Owned incomplete-edit failure requiring checked whole-target detach.
  iree_status_t failure;
  // Terminal checked removal permits physical cleanup, never further execution.
  bool detached;
} loom_runtime_memory_target_t;

struct loom_runtime_virtual_buffer_t {
  // Borrowed physical owner outliving this reservation.
  loom_runtime_memory_pool_t* pool;
  // Borrowed caller accounting group, independent of the shared pool budget.
  loom_runtime_storage_statistics_t* statistics;
  // Owned stable virtual reservation or range view, independent of backing.
  iree_hal_buffer_t* handle;
  // Previous standalone reservation retained through cold growth mapping.
  // Failed growth keeps both targets until checked teardown.
  iree_hal_buffer_t* previous_handle;
  // Shared native-target state, or owned trailing state for a standalone root.
  loom_runtime_memory_target_t* target;
  // Borrowed address allocator, NULL when this buffer owns its reservation.
  loom_runtime_memory_arena_t* arena;
  // Address lease returned only after consumers and backing edits retire.
  iree_hal_memory_tlsf_block_index_t block_index;
  // Validated physical geometry, independent of other reservations.
  loom_runtime_memory_geometry_t geometry;
  // Required virtual base alignment, preserved across reservation growth.
  iree_device_size_t alignment;
  // Native-aligned reservation extent, including its exact short tail.
  iree_device_size_t byte_length;
  // Number of slots spanning the reservation, including an optional short tail.
  iree_host_size_t slab_count;
  // Regular slots with physical or accepted backing; excludes a short tail.
  iree_host_size_t backed_regular_count;
  // Physical ownership kind of regular mapping slots.
  loom_runtime_backing_kind_t regular_kind;
  // Physical ownership kind of the final possibly shorter mapping slot.
  loom_runtime_backing_kind_t tail_kind;
  // Native cost of one dedicated regular extent; zero for reusable classes.
  uint64_t regular_cost;
  // Native cost of the final dedicated extent, including alignment padding.
  uint64_t tail_cost;
  // Independent page ownership in sparse groups of adjacent mappings.
  loom_runtime_backing_directory_t backing;
};

static inline iree_device_size_t loom_runtime_memory_slab_length(
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index) {
  return iree_min(buffer->geometry.mapping_size,
                  buffer->byte_length - index * buffer->geometry.mapping_size);
}

static inline uint64_t loom_runtime_memory_range_length(
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index,
    iree_host_size_t count) {
  return (count - 1) * buffer->geometry.mapping_size +
         loom_runtime_memory_slab_length(buffer, index + count - 1);
}

static inline iree_host_size_t loom_runtime_memory_range_end(
    const loom_runtime_memory_range_t* range) {
  const uint64_t stride = range->buffer->geometry.mapping_size;
  return range->length ? (range->offset + range->length - 1) / stride + 1
                       : range->offset / stride;
}

static inline loom_runtime_backing_cursor_t loom_runtime_memory_range_iterate(
    const loom_runtime_memory_range_t* range) {
  return loom_runtime_backing_directory_iterate(
      &range->buffer->backing,
      range->offset / range->buffer->geometry.mapping_size,
      loom_runtime_memory_range_end(range));
}

static inline uint64_t loom_runtime_memory_slab_cost(
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index) {
  return index + 1 == buffer->slab_count ? buffer->tail_cost
                                         : buffer->regular_cost;
}

// Accumulates the backing classes and native costs of a contiguous interval.
void loom_runtime_memory_range_add_demand(
    loom_runtime_memory_demand_t* demand,
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index,
    iree_host_size_t count);

// Records the physical assignment before native submission. An allocation
// failure can leave an assigned prefix; clearing the interval cancels it.
iree_status_t loom_runtime_memory_assign_backing(
    loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index,
    iree_host_size_t count, loom_runtime_backing_slot_t slot);
void loom_runtime_memory_clear_backing(loom_runtime_virtual_buffer_t* buffer,
                                       iree_host_size_t index,
                                       iree_host_size_t count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_STORAGE_MEMORY_RESERVATION_H_
