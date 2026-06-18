// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/memory_pool.h"

#include <string.h>

void iree_hip_pool_allocation_tracker_initialize(
    iree_hip_pool_allocation_tracker_t* tracker,
    iree_hip_pool_allocation_slot_t* slots, size_t capacity) {
  IREE_ASSERT_ARGUMENT(tracker);
  tracker->slots = slots;
  tracker->capacity = capacity;
  tracker->live_count = 0;
  tracker->high_water_mark = 0;
  tracker->free_hint = 0;
  if (slots && capacity > 0) {
    memset(slots, 0, capacity * sizeof(*slots));
  }
}

void iree_hip_pool_allocation_tracker_reset(
    iree_hip_pool_allocation_tracker_t* tracker) {
  IREE_ASSERT_ARGUMENT(tracker);
  if (tracker->slots && tracker->high_water_mark > 0) {
    memset(tracker->slots, 0,
           tracker->high_water_mark * sizeof(tracker->slots[0]));
  }
  tracker->live_count = 0;
  tracker->high_water_mark = 0;
  tracker->free_hint = 0;
}

bool iree_hip_pool_allocation_tracker_insert(
    iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    size_t size) {
  return iree_hip_pool_allocation_tracker_insert_sized(tracker, address, size,
                                                       size);
}

bool iree_hip_pool_allocation_tracker_insert_sized(
    iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    size_t size, size_t requested_size) {
  IREE_ASSERT_ARGUMENT(tracker);
  if (!tracker->slots || address == 0 || size == 0 || requested_size == 0) {
    return false;
  }
  if (requested_size > size) return false;
  if (size - 1 > UINTPTR_MAX - address) return false;
  if (tracker->live_count >= tracker->capacity) return false;

  for (size_t i = 0; i < tracker->high_water_mark; ++i) {
    const iree_hip_pool_allocation_slot_t* slot = &tracker->slots[i];
    if (!slot->live) continue;
    const bool overlaps = address >= slot->address
                              ? address - slot->address < slot->size
                              : slot->address - address < size;
    if (overlaps) return false;
  }

  const size_t capacity = tracker->capacity;
  size_t slot_index = SIZE_MAX;
  for (size_t scan_count = 0; scan_count < capacity; ++scan_count) {
    const size_t index = (tracker->free_hint + scan_count) % capacity;
    if (!tracker->slots[index].live) {
      slot_index = index;
      break;
    }
  }
  if (slot_index == SIZE_MAX) return false;

  tracker->slots[slot_index].address = address;
  tracker->slots[slot_index].size = size;
  tracker->slots[slot_index].requested_size = requested_size;
  tracker->slots[slot_index].live = true;
  ++tracker->live_count;
  if (slot_index >= tracker->high_water_mark) {
    tracker->high_water_mark = slot_index + 1;
  }
  tracker->free_hint = (slot_index + 1) % capacity;
  return true;
}

bool iree_hip_pool_allocation_tracker_release(
    iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    size_t* out_size) {
  IREE_ASSERT_ARGUMENT(tracker);
  if (out_size) *out_size = 0;
  if (!tracker->slots || address == 0) return false;

  for (size_t i = 0; i < tracker->high_water_mark; ++i) {
    iree_hip_pool_allocation_slot_t* slot = &tracker->slots[i];
    if (!slot->live || slot->address != address) continue;
    if (out_size) *out_size = slot->size;
    slot->live = false;
    slot->address = 0;
    slot->size = 0;
    slot->requested_size = 0;
    --tracker->live_count;
    tracker->free_hint = i;
    return true;
  }
  return false;
}

bool iree_hip_pool_allocation_tracker_find(
    const iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    uintptr_t* out_base, size_t* out_size) {
  IREE_ASSERT_ARGUMENT(tracker);
  if (!tracker->slots || address == 0) return false;

  for (size_t i = 0; i < tracker->high_water_mark; ++i) {
    const iree_hip_pool_allocation_slot_t* slot = &tracker->slots[i];
    if (!slot->live) continue;
    if (address < slot->address) continue;
    const uintptr_t offset = address - slot->address;
    if (offset >= slot->size) continue;
    if (out_base) *out_base = slot->address;
    if (out_size) *out_size = slot->requested_size;
    return true;
  }
  return false;
}

bool iree_hip_pool_allocation_tracker_find_physical(
    const iree_hip_pool_allocation_tracker_t* tracker, uintptr_t address,
    uintptr_t* out_base, size_t* out_size) {
  IREE_ASSERT_ARGUMENT(tracker);
  if (!tracker->slots || address == 0) return false;

  for (size_t i = 0; i < tracker->high_water_mark; ++i) {
    const iree_hip_pool_allocation_slot_t* slot = &tracker->slots[i];
    if (!slot->live) continue;
    if (address < slot->address) continue;
    const uintptr_t offset = address - slot->address;
    if (offset >= slot->size) continue;
    if (out_base) *out_base = slot->address;
    if (out_size) *out_size = slot->size;
    return true;
  }
  return false;
}

size_t iree_hip_pool_allocation_tracker_live_size(
    const iree_hip_pool_allocation_tracker_t* tracker) {
  IREE_ASSERT_ARGUMENT(tracker);
  if (!tracker->slots) return 0;
  size_t total_size = 0;
  for (size_t i = 0; i < tracker->high_water_mark; ++i) {
    const iree_hip_pool_allocation_slot_t* slot = &tracker->slots[i];
    if (!slot->live) continue;
    if (slot->size > SIZE_MAX - total_size) return SIZE_MAX;
    total_size += slot->size;
  }
  return total_size;
}

bool iree_hip_pool_align_size(size_t value, size_t alignment,
                              size_t* out_aligned_value) {
  if (!out_aligned_value || alignment == 0 ||
      (alignment & (alignment - 1)) != 0) {
    return false;
  }
  return iree_host_size_checked_align(value, alignment, out_aligned_value);
}
