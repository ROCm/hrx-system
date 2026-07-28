// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/epoch_semaphore_map.h"

#include <inttypes.h>
#include <string.h>

typedef uint8_t iree_hal_remote_server_epoch_semaphore_slot_state_t;
enum iree_hal_remote_server_epoch_semaphore_slot_state_e {
  IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_EMPTY = 0,
  IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_OCCUPIED = 1,
  IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_TOMBSTONE = 2,
};

struct iree_hal_remote_server_epoch_semaphore_slot_t {
  // Retained local semaphore, valid only while the slot is occupied.
  iree_hal_semaphore_t* semaphore;
  // Remote frontier axis identifying the semaphore.
  iree_async_axis_t axis;
  // Remote frontier epoch identifying the semaphore.
  uint64_t epoch;
  // Current slot occupancy state.
  iree_hal_remote_server_epoch_semaphore_slot_state_t state;
};

static uint64_t iree_hal_remote_server_epoch_semaphore_map_mix_u64(
    uint64_t value) {
  value ^= value >> 33;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33;
  value *= UINT64_C(0xc4ceb9fe1a85ec53);
  value ^= value >> 33;
  return value;
}

static iree_host_size_t iree_hal_remote_server_epoch_semaphore_map_hash(
    iree_async_axis_t axis, uint64_t epoch, iree_host_size_t capacity) {
  uint64_t hash = iree_hal_remote_server_epoch_semaphore_map_mix_u64(axis);
  hash ^= iree_hal_remote_server_epoch_semaphore_map_mix_u64(epoch);
  return (iree_host_size_t)(hash & (capacity - 1));
}

static iree_host_size_t
iree_hal_remote_server_epoch_semaphore_map_calculate_capacity(
    iree_host_size_t minimum_capacity) {
  iree_host_size_t capacity = 64;
  while (capacity < minimum_capacity) {
    if (capacity > IREE_HOST_SIZE_MAX / 2) return 0;
    capacity *= 2;
  }
  return capacity;
}

// Finds an existing slot or the first slot at which the key can be inserted.
static iree_host_size_t iree_hal_remote_server_epoch_semaphore_map_find_slot(
    const iree_hal_remote_server_epoch_semaphore_map_t* map,
    iree_async_axis_t axis, uint64_t epoch, bool* out_found) {
  iree_host_size_t slot = iree_hal_remote_server_epoch_semaphore_map_hash(
      axis, epoch, map->capacity);
  iree_host_size_t first_tombstone = IREE_HOST_SIZE_MAX;
  while (map->slots[slot].state !=
         IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_EMPTY) {
    if (map->slots[slot].state ==
        IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_TOMBSTONE) {
      if (first_tombstone == IREE_HOST_SIZE_MAX) first_tombstone = slot;
    } else if (map->slots[slot].axis == axis &&
               map->slots[slot].epoch == epoch) {
      *out_found = true;
      return slot;
    }
    slot = (slot + 1) & (map->capacity - 1);
  }
  *out_found = false;
  return first_tombstone == IREE_HOST_SIZE_MAX ? slot : first_tombstone;
}

static iree_status_t iree_hal_remote_server_epoch_semaphore_map_resize(
    iree_hal_remote_server_epoch_semaphore_map_t* map,
    iree_host_size_t minimum_capacity, iree_allocator_t host_allocator) {
  iree_host_size_t new_capacity =
      iree_hal_remote_server_epoch_semaphore_map_calculate_capacity(
          minimum_capacity);
  if (new_capacity == 0) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "epoch semaphore map capacity overflow");
  }

  iree_hal_remote_server_epoch_semaphore_slot_t* new_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, new_capacity, sizeof(*new_slots), (void**)&new_slots));

  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    if (map->slots[i].state !=
        IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_OCCUPIED) {
      continue;
    }
    iree_hal_remote_server_epoch_semaphore_slot_t old_slot = map->slots[i];
    iree_host_size_t new_slot = iree_hal_remote_server_epoch_semaphore_map_hash(
        old_slot.axis, old_slot.epoch, new_capacity);
    while (new_slots[new_slot].state ==
           IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_OCCUPIED) {
      new_slot = (new_slot + 1) & (new_capacity - 1);
    }
    new_slots[new_slot] = old_slot;
  }

  iree_allocator_free(host_allocator, map->slots);
  map->slots = new_slots;
  map->used_count = map->count;
  map->capacity = new_capacity;
  return iree_ok_status();
}

void iree_hal_remote_server_epoch_semaphore_map_move(
    iree_hal_remote_server_epoch_semaphore_map_t* map,
    iree_hal_remote_server_epoch_semaphore_map_t* out_map) {
  memcpy(out_map, map, sizeof(*out_map));
  memset(map, 0, sizeof(*map));
}

void iree_hal_remote_server_epoch_semaphore_map_deinitialize(
    iree_hal_remote_server_epoch_semaphore_map_t* map,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    if (map->slots[i].state ==
        IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_OCCUPIED) {
      iree_hal_semaphore_release(map->slots[i].semaphore);
    }
  }
  iree_allocator_free(host_allocator, map->slots);
  memset(map, 0, sizeof(*map));
}

iree_status_t iree_hal_remote_server_epoch_semaphore_map_insert(
    iree_hal_remote_server_epoch_semaphore_map_t* map, iree_async_axis_t axis,
    uint64_t epoch, iree_hal_semaphore_t* semaphore,
    iree_allocator_t host_allocator) {
  bool found = false;
  if (map->capacity != 0) {
    iree_hal_remote_server_epoch_semaphore_map_find_slot(map, axis, epoch,
                                                         &found);
    if (found) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "duplicate signal frontier entry axis=0x%016" PRIx64
          " epoch=%" PRIu64,
          axis, epoch);
    }
  }

  iree_host_size_t minimum_used_capacity = map->used_count + 1;
  iree_host_size_t growth_threshold = map->capacity - map->capacity / 4;
  if (minimum_used_capacity >= growth_threshold) {
    if (map->count >= IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "epoch semaphore map capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_hal_remote_server_epoch_semaphore_map_resize(
        map, (map->count + 1) * 2, host_allocator));
  }

  iree_host_size_t slot = iree_hal_remote_server_epoch_semaphore_map_find_slot(
      map, axis, epoch, &found);
  if (map->slots[slot].state ==
      IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_EMPTY) {
    ++map->used_count;
  }
  map->slots[slot].semaphore = semaphore;
  map->slots[slot].axis = axis;
  map->slots[slot].epoch = epoch;
  map->slots[slot].state = IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_OCCUPIED;
  ++map->count;
  iree_hal_semaphore_retain(semaphore);
  return iree_ok_status();
}

iree_hal_semaphore_t* iree_hal_remote_server_epoch_semaphore_map_lookup(
    const iree_hal_remote_server_epoch_semaphore_map_t* map,
    iree_async_axis_t axis, uint64_t epoch) {
  if (map->capacity == 0) return NULL;
  bool found = false;
  iree_host_size_t slot = iree_hal_remote_server_epoch_semaphore_map_find_slot(
      map, axis, epoch, &found);
  return found ? map->slots[slot].semaphore : NULL;
}

iree_hal_semaphore_t* iree_hal_remote_server_epoch_semaphore_map_remove(
    iree_hal_remote_server_epoch_semaphore_map_t* map, iree_async_axis_t axis,
    uint64_t epoch) {
  if (map->capacity == 0) return NULL;
  bool found = false;
  iree_host_size_t slot = iree_hal_remote_server_epoch_semaphore_map_find_slot(
      map, axis, epoch, &found);
  if (!found) return NULL;

  iree_hal_semaphore_t* semaphore = map->slots[slot].semaphore;
  map->slots[slot].semaphore = NULL;
  map->slots[slot].axis = 0;
  map->slots[slot].epoch = 0;
  map->slots[slot].state =
      IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_SLOT_TOMBSTONE;
  --map->count;
  return semaphore;
}
