// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_storage_packing.h"

#include <stdint.h>

typedef struct loom_source_storage_packing_allocation_t {
  // Source allocation root value.
  loom_value_id_t root_value_id;
  // Packed byte offset in the storage segment.
  uint64_t byte_offset;
  // Allocation extent in bytes.
  uint64_t byte_length;
} loom_source_storage_packing_allocation_t;

struct loom_source_storage_packing_t {
  // Arena owning the packing and allocation records.
  iree_arena_allocator_t* arena;
  // Retained-fact query defining which allocation roots interfere.
  loom_source_storage_packing_interference_callback_t interference;
  // Aggregate packed extent and base alignment.
  loom_source_storage_packing_requirement_t requirement;
  // Packed source allocations in stable append order.
  loom_source_storage_packing_allocation_t* allocations;
  // Number of initialized allocation records.
  iree_host_size_t allocation_count;
  // Allocated record capacity.
  iree_host_size_t allocation_capacity;
};

iree_status_t loom_source_storage_packing_create(
    loom_source_storage_packing_interference_callback_t interference,
    iree_arena_allocator_t* arena,
    loom_source_storage_packing_t** out_packing) {
  IREE_ASSERT_ARGUMENT(interference.fn);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_packing);
  *out_packing = NULL;

  loom_source_storage_packing_t* packing = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*packing), (void**)&packing));
  *packing = (loom_source_storage_packing_t){
      .arena = arena,
      .interference = interference,
  };
  *out_packing = packing;
  return iree_ok_status();
}

static iree_status_t loom_source_storage_packing_find_byte_offset(
    const loom_source_storage_packing_t* packing, loom_value_id_t root_value_id,
    uint64_t byte_length, uint64_t byte_alignment, uint64_t* out_byte_offset) {
  *out_byte_offset = 0;
  uint64_t candidate_offset = 0;
  if (!iree_is_power_of_two_uint64(byte_alignment) ||
      !iree_checked_align_u64(candidate_offset, byte_alignment,
                              &candidate_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source storage packing alignment overflows");
  }

  for (;;) {
    uint64_t candidate_end = 0;
    if (!iree_checked_add_u64(candidate_offset, byte_length, &candidate_end) ||
        candidate_end > INT64_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source storage packing exceeds INT64_MAX");
    }
    uint64_t next_candidate_offset = candidate_offset;
    for (iree_host_size_t i = 0; i < packing->allocation_count; ++i) {
      const loom_source_storage_packing_allocation_t* allocation =
          &packing->allocations[i];
      bool interferes = true;
      IREE_RETURN_IF_ERROR(packing->interference.fn(
          packing->interference.user_data, root_value_id,
          allocation->root_value_id, &interferes));
      if (!interferes) {
        continue;
      }
      uint64_t allocation_end = 0;
      const bool valid_allocation = iree_checked_add_u64(
          allocation->byte_offset, allocation->byte_length, &allocation_end);
      IREE_ASSERT(valid_allocation);
      if (candidate_offset < allocation_end &&
          allocation->byte_offset < candidate_end) {
        next_candidate_offset = iree_max(next_candidate_offset, allocation_end);
      }
    }
    if (next_candidate_offset == candidate_offset) {
      *out_byte_offset = candidate_offset;
      return iree_ok_status();
    }
    if (!iree_checked_align_u64(next_candidate_offset, byte_alignment,
                                &candidate_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source storage packing alignment overflows");
    }
  }
}

iree_status_t loom_source_storage_packing_append(
    loom_source_storage_packing_t* packing, loom_value_id_t root_value_id,
    uint64_t byte_length, uint64_t byte_alignment, uint64_t* out_byte_offset) {
  IREE_ASSERT_ARGUMENT(packing);
  IREE_ASSERT_ARGUMENT(out_byte_offset);
  *out_byte_offset = 0;

  uint64_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_source_storage_packing_find_byte_offset(
      packing, root_value_id, byte_length, byte_alignment, &byte_offset));
  uint64_t allocation_end = 0;
  if (!iree_checked_add_u64(byte_offset, byte_length, &allocation_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source storage packing exceeds INT64_MAX");
  }
  const iree_host_size_t minimum_capacity = packing->allocation_count + 1;
  if (minimum_capacity > packing->allocation_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        packing->arena, packing->allocation_count,
        iree_max(minimum_capacity, 4u), sizeof(*packing->allocations),
        &packing->allocation_capacity, (void**)&packing->allocations));
  }
  packing->allocations[packing->allocation_count++] =
      (loom_source_storage_packing_allocation_t){
          .root_value_id = root_value_id,
          .byte_offset = byte_offset,
          .byte_length = byte_length,
      };
  packing->requirement.byte_length =
      iree_max(packing->requirement.byte_length, allocation_end);
  packing->requirement.byte_alignment =
      iree_max(packing->requirement.byte_alignment, byte_alignment);
  *out_byte_offset = byte_offset;
  return iree_ok_status();
}

loom_source_storage_packing_requirement_t
loom_source_storage_packing_requirement(
    const loom_source_storage_packing_t* packing) {
  IREE_ASSERT_ARGUMENT(packing);
  return packing->requirement;
}
