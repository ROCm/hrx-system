// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_storage_packing.h"

#include <stdint.h>

#include "loom/analysis/storage_interference.h"

typedef struct loom_source_storage_packing_allocation_t {
  // Source allocation root value.
  loom_value_id_t root_value_id;
  // Packed byte offset in the storage segment.
  uint64_t byte_offset;
  // Exclusive end of the allocation, admitted when appended.
  uint64_t byte_end;
} loom_source_storage_packing_allocation_t;

typedef struct loom_source_storage_packing_reservation_t {
  // First occupied byte in the backing store.
  uint64_t byte_offset;
  // Exclusive end of the reservation, admitted at construction.
  uint64_t byte_end;
} loom_source_storage_packing_reservation_t;

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
  // Fixed and automatically placed lifetime-long reservations.
  loom_source_storage_packing_reservation_t* reservations;
  // Number of initialized reservation records.
  iree_host_size_t reservation_count;
  // Allocated reservation record capacity.
  iree_host_size_t reservation_capacity;
};

iree_status_t loom_source_storage_packing_allocate(
    loom_source_storage_packing_interference_callback_t interference,
    const loom_source_storage_packing_range_t* reserved_ranges,
    iree_host_size_t reserved_range_count, iree_arena_allocator_t* arena,
    loom_source_storage_packing_t** out_packing) {
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
  if (reserved_range_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, reserved_range_count, sizeof(*packing->reservations),
        (void**)&packing->reservations));
    packing->reservation_capacity = reserved_range_count;
    for (iree_host_size_t i = 0; i < reserved_range_count; ++i) {
      const loom_source_storage_packing_range_t range = reserved_ranges[i];
      uint64_t byte_end = 0;
      if (!iree_checked_add_u64(range.byte_offset, range.byte_length,
                                &byte_end) ||
          byte_end > INT64_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "reserved storage range exceeds INT64_MAX");
      }
      if (!range.byte_length) {
        continue;
      }
      packing->reservations[packing->reservation_count++] =
          (loom_source_storage_packing_reservation_t){
              .byte_offset = range.byte_offset,
              .byte_end = byte_end,
          };
      packing->requirement.byte_length =
          iree_max(packing->requirement.byte_length, byte_end);
    }
  }
  *out_packing = packing;
  return iree_ok_status();
}

static iree_status_t loom_source_storage_packing_private_interference(
    void* user_data, loom_value_id_t lhs_root_value_id,
    loom_value_id_t rhs_root_value_id, bool* out_interferes) {
  *out_interferes = !loom_storage_interference_prove_private_nonoverlap(
      (const loom_storage_interference_t*)user_data, lhs_root_value_id,
      rhs_root_value_id);
  return iree_ok_status();
}

iree_status_t loom_source_storage_packing_allocate_private(
    const loom_storage_interference_t* interference,
    iree_arena_allocator_t* arena,
    loom_source_storage_packing_t** out_packing) {
  return loom_source_storage_packing_allocate(
      loom_source_storage_packing_interference_callback_make(
          loom_source_storage_packing_private_interference,
          (void*)interference),
      NULL, 0, arena, out_packing);
}

static iree_status_t loom_source_storage_packing_find_byte_offset(
    const loom_source_storage_packing_t* packing,
    const loom_value_id_t* root_value_id, uint64_t byte_length,
    uint64_t byte_alignment,
    const loom_source_storage_packing_range_t* excluded_ranges,
    iree_host_size_t excluded_range_count, uint64_t* out_byte_offset) {
  *out_byte_offset = 0;
  uint64_t candidate_offset = 0;
  if (!byte_length) {
    return iree_ok_status();
  }

  for (;;) {
    uint64_t candidate_end = 0;
    if (!iree_checked_add_u64(candidate_offset, byte_length, &candidate_end) ||
        candidate_end > INT64_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source storage packing exceeds INT64_MAX");
    }
    uint64_t next_candidate_offset = candidate_offset;
    for (iree_host_size_t i = 0; i < excluded_range_count; ++i) {
      const loom_source_storage_packing_range_t range = excluded_ranges[i];
      const uint64_t end = range.byte_offset + range.byte_length;
      if (range.byte_length && candidate_offset < end &&
          range.byte_offset < candidate_end) {
        next_candidate_offset = iree_max(next_candidate_offset, end);
      }
    }
    for (iree_host_size_t i = 0; i < packing->reservation_count; ++i) {
      const loom_source_storage_packing_reservation_t reservation =
          packing->reservations[i];
      if (candidate_offset < reservation.byte_end &&
          reservation.byte_offset < candidate_end) {
        next_candidate_offset =
            iree_max(next_candidate_offset, reservation.byte_end);
      }
    }
    for (iree_host_size_t i = 0; i < packing->allocation_count; ++i) {
      const loom_source_storage_packing_allocation_t* allocation =
          &packing->allocations[i];
      if (allocation->byte_offset == allocation->byte_end ||
          candidate_offset >= allocation->byte_end ||
          allocation->byte_offset >= candidate_end) {
        continue;
      }
      bool interferes = true;
      if (root_value_id && packing->interference.fn) {
        IREE_RETURN_IF_ERROR(packing->interference.fn(
            packing->interference.user_data, *root_value_id,
            allocation->root_value_id, &interferes));
      }
      if (interferes) {
        next_candidate_offset =
            iree_max(next_candidate_offset, allocation->byte_end);
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
      packing, &root_value_id, byte_length, byte_alignment, NULL, 0,
      &byte_offset));
  const uint64_t allocation_end = byte_offset + byte_length;
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
          .byte_end = allocation_end,
      };
  packing->requirement.byte_length =
      iree_max(packing->requirement.byte_length, allocation_end);
  packing->requirement.byte_alignment =
      iree_max(packing->requirement.byte_alignment, byte_alignment);
  *out_byte_offset = byte_offset;
  return iree_ok_status();
}

iree_status_t loom_source_storage_packing_reserve(
    loom_source_storage_packing_t* packing, uint64_t byte_length,
    uint64_t byte_alignment,
    const loom_source_storage_packing_range_t* excluded_ranges,
    iree_host_size_t excluded_range_count, uint64_t* out_byte_offset) {
  *out_byte_offset = 0;
  uint64_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_source_storage_packing_find_byte_offset(
      packing, NULL, byte_length, byte_alignment, excluded_ranges,
      excluded_range_count, &byte_offset));
  if (packing->reservation_count == packing->reservation_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        packing->arena, packing->reservation_count,
        packing->reservation_count + 1, sizeof(*packing->reservations),
        &packing->reservation_capacity, (void**)&packing->reservations));
  }
  const uint64_t byte_end = byte_offset + byte_length;
  packing->reservations[packing->reservation_count++] =
      (loom_source_storage_packing_reservation_t){
          .byte_offset = byte_offset,
          .byte_end = byte_end,
      };
  packing->requirement.byte_length =
      iree_max(packing->requirement.byte_length, byte_end);
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
