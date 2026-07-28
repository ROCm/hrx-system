// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_RESOURCE_TABLE_H_
#define IREE_HAL_REMOTE_SERVER_RESOURCE_TABLE_H_

#include "iree/base/api.h"
#include "iree/hal/remote/protocol/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Slot-based resource table with typed generation counters for ABA prevention.
//
// Stores retained HAL resources (buffers, semaphores, executables, etc.)
// indexed by the Slot[23:0] field of a resource_id. Each occupied slot records
// the assigned resource type and generation independently of the untrusted ID.
// Lookup, detach, and release require all three values to match.
//
// Resource IDs encode: Type[63:56] | Flags[55:48] | Generation[47:32] |
//                      Proactor[31:24] | Slot[23:0]
//
// A slot is permanently retired when its 16-bit generation is exhausted so
// that a stale ID can never become valid again after counter wraparound.
typedef struct iree_hal_remote_resource_table_slot_t
    iree_hal_remote_resource_table_slot_t;

typedef struct iree_hal_remote_resource_table_t {
  // Typed resource slots owned by the table.
  iree_hal_remote_resource_table_slot_t* slots;
  // Total number of slots in |slots|.
  uint32_t capacity;
  // Slot at which the next free-slot scan begins.
  uint32_t next_slot;
} iree_hal_remote_resource_table_t;

// Initializes a resource table with the given capacity.
// All slots start empty (NULL). The table must be deinitialized when done.
iree_status_t iree_hal_remote_resource_table_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_hal_remote_resource_table_t* out_table);

// Releases all retained resources and frees the table storage.
// Safe to call on a zero-initialized table (no-op).
void iree_hal_remote_resource_table_deinitialize(
    iree_hal_remote_resource_table_t* table, iree_allocator_t host_allocator);

// Assigns a resource to a free slot and returns the full resource_id.
// The resource is retained by the table. |resource_type| is encoded into
// the Type[63:56] field of the returned ID.
iree_status_t iree_hal_remote_resource_table_assign(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_type_t resource_type, void* resource,
    iree_hal_remote_resource_id_t* out_resource_id);

// Looks up a resource by resource_id. The encoded type, |expected_type|, stored
// type, generation, and slot occupancy must all match. Returns NULL on any
// mismatch.
void* iree_hal_remote_resource_table_lookup(
    const iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_type_t expected_type,
    iree_hal_remote_resource_id_t resource_id);

// Releases a resource slot by resource_id. The encoded type, stored type,
// generation, and slot occupancy must all match. Silently ignores mismatches
// because stale releases are benign.
void iree_hal_remote_resource_table_release(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_id_t resource_id);

// Detaches a resource slot by resource_id and transfers the table's retained
// reference to the caller. The encoded type, stored type, generation, and slot
// occupancy must all match. Returns NULL on mismatch. The caller must release
// the returned HAL resource.
void* iree_hal_remote_resource_table_detach(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_id_t resource_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_RESOURCE_TABLE_H_
