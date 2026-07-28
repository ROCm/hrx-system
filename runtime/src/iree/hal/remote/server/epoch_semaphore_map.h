// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_MAP_H_
#define IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_MAP_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_epoch_semaphore_slot_t
    iree_hal_remote_server_epoch_semaphore_slot_t;

// Maps remote frontier entries to retained local completion semaphores.
//
// The zero value is an initialized empty map. The map is thread-compatible;
// callers must serialize access while a map may be mutated.
typedef struct iree_hal_remote_server_epoch_semaphore_map_t {
  // Open-addressed slots owned by the map.
  iree_hal_remote_server_epoch_semaphore_slot_t* slots;
  // Number of occupied slots.
  iree_host_size_t count;
  // Number of occupied or tombstone slots.
  iree_host_size_t used_count;
  // Power-of-two capacity of |slots|.
  iree_host_size_t capacity;
} iree_hal_remote_server_epoch_semaphore_map_t;

// Moves |map| into |out_map| and resets |map| to the empty state.
void iree_hal_remote_server_epoch_semaphore_map_move(
    iree_hal_remote_server_epoch_semaphore_map_t* map,
    iree_hal_remote_server_epoch_semaphore_map_t* out_map);

// Releases all retained semaphores and resets |map| to the empty state.
void iree_hal_remote_server_epoch_semaphore_map_deinitialize(
    iree_hal_remote_server_epoch_semaphore_map_t* map,
    iree_allocator_t host_allocator);

// Inserts |semaphore| at |axis| and |epoch| and retains it.
//
// Returns IREE_STATUS_ALREADY_EXISTS when the frontier entry is already
// present. Allocation failure leaves all existing mappings unchanged.
iree_status_t iree_hal_remote_server_epoch_semaphore_map_insert(
    iree_hal_remote_server_epoch_semaphore_map_t* map, iree_async_axis_t axis,
    uint64_t epoch, iree_hal_semaphore_t* semaphore,
    iree_allocator_t host_allocator);

// Returns the borrowed semaphore mapped to |axis| and |epoch|, or NULL.
iree_hal_semaphore_t* iree_hal_remote_server_epoch_semaphore_map_lookup(
    const iree_hal_remote_server_epoch_semaphore_map_t* map,
    iree_async_axis_t axis, uint64_t epoch);

// Removes the mapping at |axis| and |epoch|, or returns NULL when absent.
//
// The returned semaphore transfers the map's retained reference to the caller.
iree_hal_semaphore_t* iree_hal_remote_server_epoch_semaphore_map_remove(
    iree_hal_remote_server_epoch_semaphore_map_t* map, iree_async_axis_t axis,
    uint64_t epoch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_EPOCH_SEMAPHORE_MAP_H_
