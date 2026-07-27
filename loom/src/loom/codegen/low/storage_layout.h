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

typedef struct loom_low_storage_layout_record_t {
  // SSA value produced by the low.storage.reserve op.
  loom_value_id_t storage_value_id;
  // Layout assigned to |storage_value_id|.
  loom_low_storage_layout_reservation_t reservation;
} loom_low_storage_layout_record_t;

typedef struct loom_low_storage_layout_t {
  // Total bytes reserved in each function-local storage space.
  loom_low_storage_layout_space_sizes_t space_sizes;
  // Arena-owned records in function body declaration order.
  const loom_low_storage_layout_record_t* records;
  // Number of entries in |records|.
  iree_host_size_t record_count;
} loom_low_storage_layout_t;

// Mutable one-pass builder for a verified function-local storage layout.
typedef struct loom_low_storage_layout_builder_t {
  // Packed byte sizes accumulated for each storage space.
  loom_low_storage_layout_space_sizes_t space_sizes;
  // Arena-owned records accumulated in declaration order.
  loom_low_storage_layout_record_t* records;
  // Number of initialized records.
  iree_host_size_t record_count;
  // Allocated capacity of |records|.
  iree_host_size_t record_capacity;
} loom_low_storage_layout_builder_t;

typedef struct loom_low_storage_layout_reference_t {
  // Root reservation containing the referenced storage bytes.
  loom_low_storage_layout_reservation_t reservation;
  // Byte offset from |reservation| to the referenced storage view.
  uint64_t byte_offset;
  // Static byte length visible through the referenced storage handle.
  uint64_t byte_length;
} loom_low_storage_layout_reference_t;

// Initializes an empty one-pass layout builder.
void loom_low_storage_layout_builder_initialize(
    loom_low_storage_layout_builder_t* out_builder);

// Packs one verified low.storage.reserve into |builder|. Aggregate byte-size
// overflow and arena growth are returned as status.
iree_status_t loom_low_storage_layout_builder_append(
    const loom_module_t* module, const loom_op_t* reserve_op,
    iree_arena_allocator_t* arena, loom_low_storage_layout_builder_t* builder);

// Publishes the current arena-owned builder contents as an immutable layout.
void loom_low_storage_layout_builder_finish(
    const loom_low_storage_layout_builder_t* builder,
    loom_low_storage_layout_t* out_layout);

// Accumulates one verified low.storage.reserve into |sizes| without retaining a
// record. Aggregate byte-size overflow is returned as status.
iree_status_t loom_low_storage_layout_accumulate_reservation(
    const loom_module_t* module, const loom_op_t* reserve_op,
    loom_low_storage_layout_space_sizes_t* sizes);

// Looks up a verified low.storage.reserve result in |layout|.
void loom_low_storage_layout_lookup_reservation(
    const loom_low_storage_layout_t* layout, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation);

// Resolves a verified low.storage.reserve or low.storage.view handle against
// |layout|. The layout and storage value must belong to the same immutable
// function.
void loom_low_storage_layout_lookup_reference(
    const loom_low_storage_layout_t* layout, const loom_module_t* module,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reference_t* out_reference);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_STORAGE_LAYOUT_H_
