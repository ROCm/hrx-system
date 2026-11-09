// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/ranges.h"

typedef struct loom_runtime_range_claim_t {
  // One owner from the reservation and one from every lifetime-tracked view.
  iree_atomic_ref_count_t ref_count;
  // Borrowed owner, protected by the enclosing model's root retirement.
  loom_runtime_ranges_t* ranges;
  // First logical unit in the contiguous interval.
  iree_host_size_t begin;
  // Number of logical units protected through final queue release.
  iree_host_size_t count;
} loom_runtime_range_claim_t;

typedef struct loom_runtime_range_t {
  // Generic prepared-resource reference prefix.
  iree_vm_ref_object_t ref_object;
  // Allocator owning this handle and its trailing view table.
  iree_allocator_t host_allocator;
  // Current interval, separate from cached views to avoid an ownership cycle.
  loom_runtime_range_claim_t* claim;
  // Number of initialized cached views, including partial construction.
  iree_host_size_t buffer_count;
  // Owned full-root views, reused by every continuing epoch of this cohort.
  iree_hal_buffer_t* buffers[];
} loom_runtime_range_t;

iree_status_t loom_runtime_ranges_initialize(
    iree_host_size_t capacity, iree_host_size_t root_count,
    iree_hal_buffer_t* const* roots, iree_allocator_t host_allocator,
    loom_runtime_ranges_t* out_ranges) {
  *out_ranges = (loom_runtime_ranges_t){.host_allocator = host_allocator,
                                        .root_count = root_count,
                                        .roots = roots};
  iree_slim_mutex_initialize(&out_ranges->mutex);
  if (!capacity) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator,
      capacity / IREE_BITMAP_BITS_PER_WORD +
          (capacity % IREE_BITMAP_BITS_PER_WORD != 0),
      sizeof(uint64_t), (void**)&out_ranges->occupied.words));
  out_ranges->occupied.bit_count = capacity;
  return iree_ok_status();
}

void loom_runtime_ranges_deinitialize(loom_runtime_ranges_t* ranges) {
  iree_slim_mutex_lock(&ranges->mutex);
  IREE_ASSERT(iree_bitmap_none_set(ranges->occupied));
  iree_slim_mutex_unlock(&ranges->mutex);
  iree_allocator_free(ranges->host_allocator, ranges->occupied.words);
  iree_slim_mutex_deinitialize(&ranges->mutex);
}

void loom_runtime_ranges_set_observer(loom_runtime_ranges_t* ranges,
                                      loom_runtime_wake_callback_t observer) {
  iree_slim_mutex_lock(&ranges->mutex);
  ranges->observer = observer;
  iree_slim_mutex_unlock(&ranges->mutex);
}

uint64_t loom_runtime_ranges_generation(loom_runtime_ranges_t* ranges) {
  iree_slim_mutex_lock(&ranges->mutex);
  const uint64_t generation = ranges->generation;
  iree_slim_mutex_unlock(&ranges->mutex);
  return generation;
}

static void loom_runtime_range_claim_release(
    loom_runtime_range_claim_t* claim) {
  if (!claim) {
    return;
  }
  if (iree_atomic_ref_count_dec(&claim->ref_count) != 1) {
    return;
  }
  loom_runtime_ranges_t* ranges = claim->ranges;
  iree_slim_mutex_lock(&ranges->mutex);
  iree_bitmap_reset_span(ranges->occupied, claim->begin, claim->count);
  ++ranges->generation;
  if (ranges->observer.wake) {
    ranges->observer.wake(ranges->observer.user_data);
  }
  iree_slim_mutex_unlock(&ranges->mutex);
  iree_allocator_free(ranges->host_allocator, claim);
}

static void loom_runtime_range_view_release(void* user_data,
                                            iree_hal_buffer_t* buffer) {
  loom_runtime_range_claim_release(user_data);
}

void loom_runtime_range_reset(iree_vm_ref_t reservation) {
  loom_runtime_range_t* range = reservation.object;
  if (!range) {
    return;
  }
  for (iree_host_size_t i = 0; i < range->buffer_count; ++i) {
    iree_hal_buffer_release(range->buffers[i]);
    range->buffers[i] = NULL;
  }
  range->buffer_count = 0;
  loom_runtime_range_claim_release(range->claim);
  range->claim = NULL;
}

static void loom_runtime_range_destroy(void* pointer) {
  loom_runtime_range_t* range = pointer;
  loom_runtime_range_reset((iree_vm_ref_t){.object = pointer});
  iree_allocator_free(range->host_allocator, range);
}

static const iree_vm_ref_type_table_t range_types;
static const iree_vm_ref_type_descriptor_t range_type = {
    .table = &range_types,
    .destroy = loom_runtime_range_destroy,
    .type_name = IREE_SVL("range"),
};

static const iree_vm_ref_type_t range_type_values[] = {&range_type};
static const iree_vm_ref_type_table_t range_types = {
    .structure_size = sizeof(iree_vm_ref_type_table_t),
    .namespace_name = IREE_SVL("storage"),
    .types = {range_type_values, IREE_ARRAYSIZE(range_type_values)},
};

bool loom_runtime_range_is_active(iree_vm_ref_t reservation) {
  return reservation.object &&
         ((loom_runtime_range_t*)reservation.object)->claim != NULL;
}

iree_host_size_t loom_runtime_range_begin(iree_vm_ref_t reservation) {
  return ((loom_runtime_range_t*)reservation.object)->claim->begin;
}

iree_host_size_t loom_runtime_range_buffer_count(iree_vm_ref_t reservation) {
  return ((loom_runtime_range_t*)reservation.object)->buffer_count;
}

iree_hal_buffer_t* loom_runtime_range_buffer(iree_vm_ref_t reservation,
                                             iree_host_size_t ordinal) {
  return ((loom_runtime_range_t*)reservation.object)->buffers[ordinal];
}

iree_status_t loom_runtime_ranges_try_acquire(loom_runtime_ranges_t* ranges,
                                              iree_host_size_t count,
                                              iree_vm_ref_t* reservation,
                                              bool* out_acquired) {
  *out_acquired = loom_runtime_range_is_active(*reservation);
  if (*out_acquired) {
    return iree_ok_status();
  }
  iree_slim_mutex_lock(&ranges->mutex);
  const iree_host_size_t begin =
      iree_bitmap_find_first_unset_span(ranges->occupied, 0, count);
  if (begin != ranges->occupied.bit_count) {
    iree_bitmap_set_span(ranges->occupied, begin, count);
  }
  iree_slim_mutex_unlock(&ranges->mutex);
  if (begin == ranges->occupied.bit_count) {
    return iree_ok_status();
  }

  loom_runtime_range_claim_t* claim = NULL;
  iree_status_t status = iree_allocator_malloc(ranges->host_allocator,
                                               sizeof(*claim), (void**)&claim);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&ranges->mutex);
    iree_bitmap_reset_span(ranges->occupied, begin, count);
    iree_slim_mutex_unlock(&ranges->mutex);
    return status;
  }
  iree_atomic_ref_count_init(&claim->ref_count);
  claim->ranges = ranges;
  claim->begin = begin;
  claim->count = count;
  loom_runtime_range_t* range = reservation->object;
  if (!range) {
    status = iree_allocator_malloc_array(
        ranges->host_allocator, 1,
        sizeof(*range) + ranges->root_count * sizeof(*range->buffers),
        (void**)&range);
    if (iree_status_is_ok(status)) {
      iree_vm_ref_object_initialize(&range->ref_object);
      range->host_allocator = ranges->host_allocator;
    }
  }
  for (iree_host_size_t i = 0;
       i < ranges->root_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_subspan_buffer_create_with_callback(
        ranges->roots[i], iree_hal_buffer_byte_offset(ranges->roots[i]),
        iree_hal_buffer_byte_length(ranges->roots[i]),
        (iree_hal_buffer_release_callback_t){loom_runtime_range_view_release,
                                             claim},
        ranges->host_allocator, &range->buffers[i]);
    if (iree_status_is_ok(status)) {
      iree_atomic_ref_count_inc(&claim->ref_count);
      ++range->buffer_count;
    }
  }
  if (iree_status_is_ok(status)) {
    range->claim = claim;
    if (!reservation->object) {
      void* pointer = range;
      *reservation = iree_vm_ref_from_ptr_move(&pointer, &range_type);
    }
    *out_acquired = true;
  } else {
    if (range) {
      loom_runtime_range_reset((iree_vm_ref_t){.object = range});
      if (!reservation->object) {
        iree_allocator_free(ranges->host_allocator, range);
      }
    }
    loom_runtime_range_claim_release(claim);
  }
  return status;
}
