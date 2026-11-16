// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable byte-range packing for source function storage.

#ifndef LOOM_ANALYSIS_SOURCE_STORAGE_PACKING_H_
#define LOOM_ANALYSIS_SOURCE_STORAGE_PACKING_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_source_storage_packing_t loom_source_storage_packing_t;
typedef struct loom_storage_interference_t loom_storage_interference_t;

// Queries whether two source allocation roots require distinct byte ranges.
// The callback must consume retained compiler facts and must not recover the
// relation by walking source IR. Queries are synchronous and may use
// |user_data| as function-local scratch.
typedef iree_status_t (*loom_source_storage_packing_interference_fn_t)(
    void* user_data, loom_value_id_t lhs_root_value_id,
    loom_value_id_t rhs_root_value_id, bool* out_interferes);

typedef struct loom_source_storage_packing_interference_callback_t {
  // Callback invoked for source allocation pairs whose byte ranges overlap.
  loom_source_storage_packing_interference_fn_t fn;
  // Caller-owned state passed to |fn|.
  void* user_data;
} loom_source_storage_packing_interference_callback_t;

// Returns an interference callback wrapping |fn| and |user_data|.
static inline loom_source_storage_packing_interference_callback_t
loom_source_storage_packing_interference_callback_make(
    loom_source_storage_packing_interference_fn_t fn, void* user_data) {
  return (loom_source_storage_packing_interference_callback_t){
      /*.fn=*/fn,
      /*.user_data=*/user_data,
  };
}

typedef struct loom_source_storage_packing_requirement_t {
  // Packed high-water extent in bytes.
  uint64_t byte_length;
  // Strongest base alignment required by any packed allocation.
  uint64_t byte_alignment;
} loom_source_storage_packing_requirement_t;

// A fixed byte range owned by another participant in the same backing store.
// Transport descriptors, service state, and externally placed data can reserve
// storage without inventing source allocation roots.
typedef struct loom_source_storage_packing_range_t {
  // First reserved byte relative to the backing store.
  uint64_t byte_offset;
  // Number of reserved bytes.
  uint64_t byte_length;
} loom_source_storage_packing_range_t;

// Creates a packing segment owned by |arena|. Copies |reserved_ranges|, which
// remain occupied for the segment's entire lifetime. Their ownership and
// mutual compatibility are established by the caller; packing only excludes
// their union. An empty interference callback keeps every source allocation
// disjoint. Reserved ranges and subsequently placed allocations contribute to
// the aggregate extent.
iree_status_t loom_source_storage_packing_allocate(
    loom_source_storage_packing_interference_callback_t interference,
    const loom_source_storage_packing_range_t* reserved_ranges,
    iree_host_size_t reserved_range_count, iree_arena_allocator_t* arena,
    loom_source_storage_packing_t** out_packing);

// Creates a private storage segment using the shared allocation-instance proof.
// The analysis remains live while allocations are appended; the completed
// packing and its requirement remain usable after that analysis retires.
iree_status_t loom_source_storage_packing_allocate_private(
    const loom_storage_interference_t* interference,
    iree_arena_allocator_t* arena, loom_source_storage_packing_t** out_packing);

// Appends one source allocation and returns its immutable packed byte offset.
//
// Each allocation root must be appended at most once. The allocation is placed
// within this segment only; callers use distinct packings for memory spaces or
// other storage classes that cannot share a backing allocation.
//
// The allocation is placed at the lowest aligned offset that does not overlap
// any interfering allocation already in |packing|. Noninterfering ranges may
// overlap partially or completely. A failure leaves all previously published
// offsets and the aggregate requirement unchanged.
// Byte alignment is a verified nonzero power of two.
iree_status_t loom_source_storage_packing_append(
    loom_source_storage_packing_t* packing, loom_value_id_t root_value_id,
    uint64_t byte_length, uint64_t byte_alignment, uint64_t* out_byte_offset);

// Places one additional lifetime-long reservation at the lowest aligned free
// offset. The range cannot overlap a source allocation, a fixed range, or any
// earlier reservation. Invocation-long allocations, compiled worker storage,
// and generated service state use this when no function-local interference
// relation applies. Source correspondence stays with the owning construction;
// no IR identity is synthesized. Previously published offsets remain unchanged.
// Optional admitted excluded_ranges constrain only this placement, without
// occupying storage for later allocations. Physical bank conflicts use these
// temporary exclusions while sharing the same canonical backing allocation.
iree_status_t loom_source_storage_packing_reserve(
    loom_source_storage_packing_t* packing, uint64_t byte_length,
    uint64_t byte_alignment,
    const loom_source_storage_packing_range_t* excluded_ranges,
    iree_host_size_t excluded_range_count, uint64_t* out_byte_offset);

// Returns the aggregate extent and base alignment for |packing|.
loom_source_storage_packing_requirement_t
loom_source_storage_packing_requirement(
    const loom_source_storage_packing_t* packing);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SOURCE_STORAGE_PACKING_H_
