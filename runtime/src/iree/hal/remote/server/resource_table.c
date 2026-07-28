// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/resource_table.h"

#include "iree/hal/api.h"

// The Slot field in a resource ID is 24 bits wide.
#define IREE_HAL_REMOTE_RESOURCE_TABLE_MAX_CAPACITY (1u << 24)

struct iree_hal_remote_resource_table_slot_t {
  // Retained resource, or NULL when the slot is unoccupied.
  void* resource;
  // Generation assigned to the current or most recent resource.
  uint16_t generation;
  // Resource type assigned while |resource| is non-NULL.
  iree_hal_remote_resource_type_t resource_type;
};

static bool iree_hal_remote_resource_table_slot_matches(
    const iree_hal_remote_resource_table_slot_t* slot,
    iree_hal_remote_resource_id_t resource_id) {
  return slot->resource != NULL &&
         slot->resource_type == IREE_HAL_REMOTE_RESOURCE_ID_TYPE(resource_id) &&
         slot->generation ==
             IREE_HAL_REMOTE_RESOURCE_ID_GENERATION(resource_id);
}

iree_status_t iree_hal_remote_resource_table_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_hal_remote_resource_table_t* out_table) {
  memset(out_table, 0, sizeof(*out_table));
  if (capacity > IREE_HAL_REMOTE_RESOURCE_TABLE_MAX_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "resource table capacity %u exceeds 24-bit slot encoding", capacity);
  }

  iree_status_t status = iree_allocator_malloc_array(host_allocator, capacity,
                                                     sizeof(*out_table->slots),
                                                     (void**)&out_table->slots);
  if (iree_status_is_ok(status)) {
    out_table->capacity = capacity;
    out_table->next_slot = 0;
  }
  return status;
}

void iree_hal_remote_resource_table_deinitialize(
    iree_hal_remote_resource_table_t* table, iree_allocator_t host_allocator) {
  if (!table->slots) return;
  for (uint32_t i = 0; i < table->capacity; ++i) {
    iree_hal_resource_release(table->slots[i].resource);
  }
  iree_allocator_free(host_allocator, table->slots);
  memset(table, 0, sizeof(*table));
}

iree_status_t iree_hal_remote_resource_table_assign(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_type_t resource_type, void* resource,
    iree_hal_remote_resource_id_t* out_resource_id) {
  *out_resource_id = 0;

  // Linear scan from next_slot for a free entry.
  uint32_t capacity = table->capacity;
  uint32_t start = table->next_slot;
  for (uint32_t i = 0; i < capacity; ++i) {
    uint32_t slot = (start + i) % capacity;
    iree_hal_remote_resource_table_slot_t* table_slot = &table->slots[slot];
    if (!table_slot->resource && table_slot->generation != UINT16_MAX) {
      iree_hal_resource_retain(resource);
      table_slot->resource = resource;
      table_slot->resource_type = resource_type;
      ++table_slot->generation;
      table->next_slot = (slot + 1) % capacity;

      *out_resource_id = ((uint64_t)resource_type << 56) |
                         ((uint64_t)table_slot->generation << 32) |
                         ((uint64_t)slot & 0xFFFFFF);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "resource table full (capacity=%u)", capacity);
}

void* iree_hal_remote_resource_table_lookup(
    const iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_type_t expected_type,
    iree_hal_remote_resource_id_t resource_id) {
  if (IREE_HAL_REMOTE_RESOURCE_ID_TYPE(resource_id) != expected_type) {
    return NULL;
  }
  uint32_t slot = IREE_HAL_REMOTE_RESOURCE_ID_SLOT(resource_id);
  if (slot >= table->capacity) return NULL;

  const iree_hal_remote_resource_table_slot_t* table_slot = &table->slots[slot];
  return iree_hal_remote_resource_table_slot_matches(table_slot, resource_id)
             ? table_slot->resource
             : NULL;
}

void iree_hal_remote_resource_table_release(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_id_t resource_id) {
  void* resource = iree_hal_remote_resource_table_detach(table, resource_id);
  iree_hal_resource_release(resource);
}

void* iree_hal_remote_resource_table_detach(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_id_t resource_id) {
  uint32_t slot = IREE_HAL_REMOTE_RESOURCE_ID_SLOT(resource_id);
  if (slot >= table->capacity) return NULL;

  iree_hal_remote_resource_table_slot_t* table_slot = &table->slots[slot];
  if (!iree_hal_remote_resource_table_slot_matches(table_slot, resource_id)) {
    return NULL;
  }

  void* resource = table_slot->resource;
  table_slot->resource = NULL;
  table_slot->resource_type = 0;
  return resource;
}
