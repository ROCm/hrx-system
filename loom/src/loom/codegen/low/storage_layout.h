// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent fixed layout for function-local low storage.
//
// low.storage.reserve ops declare byte-addressable storage owned by the current
// low function. This layout packs reservations by storage space in function
// body order, honoring each reservation's byte alignment before assigning its
// stable byte offset. Targets project the generic storage spaces onto their ABI
// frame, private, scratch, local, or workgroup storage mechanisms.

#ifndef LOOM_CODEGEN_LOW_STORAGE_LAYOUT_H_
#define LOOM_CODEGEN_LOW_STORAGE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bitset of function-local storage spaces.
typedef uint32_t loom_low_storage_space_set_t;

// Empty storage-space set.
#define LOOM_LOW_STORAGE_SPACE_SET_NONE ((loom_low_storage_space_set_t)0u)
// Set bit for host stack-frame storage.
#define LOOM_LOW_STORAGE_SPACE_SET_STACK \
  ((loom_low_storage_space_set_t)1u << LOOM_STORAGE_SPACE_STACK)
// Set bit for per-lane spill/scratch storage.
#define LOOM_LOW_STORAGE_SPACE_SET_SCRATCH \
  ((loom_low_storage_space_set_t)1u << LOOM_STORAGE_SPACE_SCRATCH)
// Set bit for target-private per-invocation storage.
#define LOOM_LOW_STORAGE_SPACE_SET_PRIVATE \
  ((loom_low_storage_space_set_t)1u << LOOM_STORAGE_SPACE_PRIVATE)
// Set bit for workgroup-local shared storage.
#define LOOM_LOW_STORAGE_SPACE_SET_WORKGROUP \
  ((loom_low_storage_space_set_t)1u << LOOM_STORAGE_SPACE_WORKGROUP)
// Set containing every currently defined function-local storage space.
#define LOOM_LOW_STORAGE_SPACE_SET_ALL                                     \
  (LOOM_LOW_STORAGE_SPACE_SET_STACK | LOOM_LOW_STORAGE_SPACE_SET_SCRATCH | \
   LOOM_LOW_STORAGE_SPACE_SET_PRIVATE | LOOM_LOW_STORAGE_SPACE_SET_WORKGROUP)

// Returns the singleton set for |space|, or NONE when |space| is invalid.
static inline loom_low_storage_space_set_t loom_low_storage_space_set_for(
    loom_storage_space_t space) {
  if (!loom_storage_space_is_valid(space)) {
    return LOOM_LOW_STORAGE_SPACE_SET_NONE;
  }
  return (loom_low_storage_space_set_t)1u << (uint32_t)space;
}

// Returns true when |set| contains |space|.
static inline bool loom_low_storage_space_set_contains(
    loom_low_storage_space_set_t set, loom_storage_space_t space) {
  const loom_low_storage_space_set_t singleton =
      loom_low_storage_space_set_for(space);
  return singleton != LOOM_LOW_STORAGE_SPACE_SET_NONE &&
         iree_all_bits_set(set, singleton);
}

// Writes the canonical storage-space names in |set| into |out_names| in stable
// declaration order and returns the number of names written.
iree_host_size_t loom_low_storage_space_set_names(
    loom_low_storage_space_set_t set, iree_host_size_t capacity,
    iree_string_view_t* out_names);

// Hoists static reservations in a verified low function |body| into its entry
// prefix after its live-in/resource declarations, preserving their original
// function body order and SSA identities.
// Frame construction calls this before target lowering embeds storage offsets.
// Subsequent spill reservations can then append to the prefix without changing
// the offsets of existing storage, including target-merged storage spaces.
// Storage views and executable operations retain their order and placement.
iree_status_t loom_low_storage_layout_hoist_reservations(
    loom_module_t* module, loom_region_t* body, iree_arena_allocator_t* arena);

typedef struct loom_low_storage_layout_space_sizes_t {
  // Bytes reserved in function stack-frame storage.
  uint64_t stack_bytes;
  // Bytes reserved in per-lane spill/scratch storage.
  uint64_t scratch_bytes;
  // Bytes reserved in target-private per-invocation storage.
  uint64_t private_bytes;
  // Bytes reserved in workgroup-local shared storage.
  uint64_t workgroup_bytes;
} loom_low_storage_layout_space_sizes_t;

typedef struct loom_low_storage_layout_reservation_t {
  // Function-local storage space containing this reservation.
  loom_storage_space_t space;
  // Byte offset assigned within |space|.
  uint64_t byte_offset;
  // Reservation size in bytes.
  uint64_t byte_size;
  // Reservation alignment in bytes.
  uint64_t byte_alignment;
} loom_low_storage_layout_reservation_t;

// Packs one verified reservation shape into |sizes| and returns its stable
// generic-space placement. This is the common primitive for authored storage
// ops and compiler-planned reservations that do not yet have SSA handles.
// Verification and reservation producers establish that the aggregate fits u64.
void loom_low_storage_layout_pack_reservation(
    loom_storage_space_t space, uint64_t byte_size, uint64_t byte_alignment,
    loom_low_storage_layout_space_sizes_t* sizes,
    loom_low_storage_layout_reservation_t* out_reservation);

typedef struct loom_low_storage_layout_record_t {
  // SSA value produced by the low.storage.reserve op.
  loom_value_id_t storage_value_id;
  // Layout assigned to |storage_value_id|.
  loom_low_storage_layout_reservation_t reservation;
} loom_low_storage_layout_record_t;

// One flattened handle, independent of target placement of its root
// reservation.
typedef struct loom_low_storage_layout_handle_t {
  // SSA result of a reservation or view in this function.
  loom_value_id_t value_id;
  // Root record ordinal in declaration order.
  uint32_t reservation_ordinal;
  // Byte offset from the root reservation to this handle.
  uint64_t byte_offset;
  // Static byte length visible through this handle.
  uint64_t byte_length;
} loom_low_storage_layout_handle_t;

// Sparse index over storage handles, sized by their count rather than the
// module's SSA value domain. Target projections borrow this index unchanged.
typedef struct loom_low_storage_layout_index_t {
  // Arena-owned flattened handles, with reservations preceding views.
  const loom_low_storage_layout_handle_t* handles;
  // Open-addressed handle ordinals, or UINT32_MAX for an empty bucket.
  const uint32_t* buckets;
  // Power-of-two bucket count, or zero for a function without storage.
  iree_host_size_t bucket_count;
} loom_low_storage_layout_index_t;

typedef struct loom_low_storage_layout_t {
  // Total bytes reserved in each function-local storage space.
  loom_low_storage_layout_space_sizes_t space_sizes;
  // Strongest reservation alignment in each space, indexed by storage space.
  uint64_t minimum_alignments[LOOM_STORAGE_SPACE_COUNT_];
  // Arena-owned records in function body declaration order.
  const loom_low_storage_layout_record_t* records;
  // Number of entries in |records|.
  iree_host_size_t record_count;
  // Flattened references shared by all consumers of this layout.
  loom_low_storage_layout_index_t index;
} loom_low_storage_layout_t;

// Unresolved view captured during the owning function traversal. Sources may
// appear later in block declaration order; finish resolves each view once.
typedef struct loom_low_storage_layout_view_t {
  // SSA result of the view.
  loom_value_id_t value_id;
  // Immediate source handle, before view-chain flattening.
  loom_value_id_t source_value_id;
  // Byte offset within the immediate source.
  uint64_t byte_offset;
  // Static byte length visible through the view.
  uint64_t byte_length;
} loom_low_storage_layout_view_t;

// Mutable builder fed by one traversal of a verified function's storage ops.
typedef struct loom_low_storage_layout_builder_t {
  // Packed byte sizes accumulated for each storage space.
  loom_low_storage_layout_space_sizes_t space_sizes;
  // Strongest reservation alignment in each space, indexed by storage space.
  uint64_t minimum_alignments[LOOM_STORAGE_SPACE_COUNT_];
  // Arena-owned records accumulated in declaration order.
  loom_low_storage_layout_record_t* records;
  // Number of initialized records.
  iree_host_size_t record_count;
  // Allocated capacity of |records|.
  iree_host_size_t record_capacity;
  // Arena-owned unresolved views in function body order.
  loom_low_storage_layout_view_t* views;
  // Number of initialized views.
  iree_host_size_t view_count;
  // Allocated capacity of |views|.
  iree_host_size_t view_capacity;
} loom_low_storage_layout_builder_t;

typedef struct loom_low_storage_layout_reference_t {
  // Root reservation containing the referenced storage bytes.
  loom_low_storage_layout_reservation_t reservation;
  // Byte offset from |reservation| to the referenced storage view.
  uint64_t byte_offset;
  // Static byte length visible through the referenced storage handle.
  uint64_t byte_length;
} loom_low_storage_layout_reference_t;

// Placement requirement for one complete function-local storage space.
typedef struct loom_low_storage_layout_requirement_t {
  // Packed byte length, including padding between reservations.
  uint64_t byte_length;
  // Strongest reservation alignment, or zero for an empty space.
  uint64_t minimum_alignment;
} loom_low_storage_layout_requirement_t;

// Measures one verified storage space from the retained layout. This excludes
// padding imposed by a target outside the function-local storage domain.
loom_low_storage_layout_requirement_t loom_low_storage_layout_requirement(
    const loom_low_storage_layout_t* layout, loom_storage_space_t space);

// Initializes an empty one-pass layout builder.
void loom_low_storage_layout_builder_initialize(
    loom_low_storage_layout_builder_t* out_builder);

// Captures one verified low.storage.reserve or low.storage.view in |builder|.
// Each handle must be appended exactly once before finish. Reservations retain
// declaration order; views need not follow their sources in declaration order.
// Verification and reservation producers establish that the aggregate fits u64.
// Only arena growth can fail.
iree_status_t loom_low_storage_layout_builder_append(
    const loom_module_t* module, const loom_op_t* op,
    iree_arena_allocator_t* arena, loom_low_storage_layout_builder_t* builder);

// Resolves all captured views and publishes an immutable, arena-owned layout.
// Rebuild after storage IR mutation; consumers never consult the source IR.
iree_status_t loom_low_storage_layout_builder_finish(
    const loom_low_storage_layout_builder_t* builder,
    iree_arena_allocator_t* arena, loom_low_storage_layout_t* out_layout);

// Accumulates one verified low.storage.reserve into |sizes| without retaining a
// record. Returns false on aggregate overflow, leaving |sizes| unchanged. The
// owning verifier or reservation producer diagnoses the failing operation.
bool loom_low_storage_layout_accumulate_reservation(
    const loom_module_t* module, const loom_op_t* reserve_op,
    loom_low_storage_layout_space_sizes_t* sizes);

// Resolves a captured handle using |index| and root |records|. Projections may
// replace reservation placements while preserving record order and borrowing
// the index. Both arrays and the handle belong to the same immutable function.
void loom_low_storage_layout_lookup_reference(
    const loom_low_storage_layout_index_t* index,
    const loom_low_storage_layout_record_t* records,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reference_t* out_reference);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_STORAGE_LAYOUT_H_
