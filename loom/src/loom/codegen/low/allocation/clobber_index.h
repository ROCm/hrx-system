// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Producer-owned index of implicit physical-register writes.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_CLOBBER_INDEX_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_CLOBBER_INDEX_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_low_descriptor_set_t;

// One physical-write event retained only while constructing the final index.
typedef struct loom_low_allocation_clobber_event_t
    loom_low_allocation_clobber_event_t;

// Transient program-ordered physical-write events. The enclosing producer
// releases |scratch_arena| after build transfers the compact index to its
// decision arena.
typedef struct loom_low_allocation_clobber_builder_t {
  // Arena owning growable event storage and build scratch.
  iree_arena_allocator_t* scratch_arena;
  // Program-ordered physical-write events.
  loom_low_allocation_clobber_event_t* events;
  // Number of initialized events.
  iree_host_size_t event_count;
  // Allocated event capacity.
  iree_host_size_t event_capacity;
  // Last recorded program point, valid when |event_count| is nonzero.
  uint32_t last_point;
  // Whether any event permits a definition at its exact point.
  bool has_permitted_definitions;
} loom_low_allocation_clobber_builder_t;

// Direct location range for every register class sharing one storage
// namespace. Aliasing classes and explicit physical-register classes share
// identical records.
typedef struct loom_low_allocation_clobber_storage_t {
  // First prefix offset for location zero in |location_range_starts|.
  uint32_t location_range_start;
  // Number of directly indexed locations. Larger locations have no clobbers.
  uint32_t location_count;
} loom_low_allocation_clobber_storage_t;

// Half-open point range belonging to one physical storage location.
typedef struct loom_low_allocation_clobber_range_t {
  // First point in |points|.
  uint32_t start;
  // One past the last point in |points|.
  uint32_t end;
} loom_low_allocation_clobber_range_t;

// Compact immutable physical-write index retained for allocation decisions.
// Points within each location range are in ascending program order.
typedef struct loom_low_allocation_clobber_index_t {
  // Physical-write program points grouped by storage location.
  const uint32_t* points;
  // Events that permit a definition exactly at their point. The words are
  // NULL when every event forbids definitions.
  iree_bitmap_t permitted_definitions;
  // Prefix point offsets for every indexed storage location.
  const uint32_t* location_range_starts;
  // Direct storage ranges indexed by descriptor register class.
  const loom_low_allocation_clobber_storage_t* storage_by_reg_class;
} loom_low_allocation_clobber_index_t;

// Initializes an empty builder using |scratch_arena| for transient storage.
void loom_low_allocation_clobber_builder_initialize(
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_clobber_builder_t* out_builder);

// Records one physical write in nondecreasing program-point order.
iree_status_t loom_low_allocation_clobber_builder_record(
    loom_low_allocation_clobber_builder_t* builder, uint32_t storage_key,
    uint32_t location, uint32_t point, bool permits_definition);

// Builds the direct immutable index in |arena|. All builder storage remains
// transient and may be released immediately after this returns.
iree_status_t loom_low_allocation_clobber_builder_build(
    loom_low_allocation_clobber_builder_t* builder,
    const struct loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_allocation_clobber_index_t* out_index);

// Returns the clobber point range for one physical storage location.
static inline loom_low_allocation_clobber_range_t
loom_low_allocation_clobber_index_range(
    const loom_low_allocation_clobber_index_t* index,
    uint16_t descriptor_reg_class_id, uint32_t location) {
  if (index->points == NULL) {
    return (loom_low_allocation_clobber_range_t){0};
  }
  const loom_low_allocation_clobber_storage_t* storage =
      &index->storage_by_reg_class[descriptor_reg_class_id];
  if (location >= storage->location_count) {
    return (loom_low_allocation_clobber_range_t){0};
  }
  const uint32_t range_index = storage->location_range_start + location;
  return (loom_low_allocation_clobber_range_t){
      /*.start=*/index->location_range_starts[range_index],
      /*.end=*/index->location_range_starts[range_index + 1u],
  };
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_CLOBBER_INDEX_H_
