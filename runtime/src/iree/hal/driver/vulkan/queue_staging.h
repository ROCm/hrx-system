// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_VULKAN_QUEUE_STAGING_H_
#define IREE_HAL_DRIVER_VULKAN_QUEUE_STAGING_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Queue-owned physical staging storage shared by host transfers and file I/O.
// A loan lasts through both native I/O and device access. Releasing capacity
// invokes waiting transfers outside the ring lock; the ring schedules no work.

typedef void(IREE_API_PTR* iree_hal_vulkan_queue_staging_waiter_fn_t)(
    void* user_data);

typedef struct iree_hal_vulkan_queue_staging_waiter_t
    iree_hal_vulkan_queue_staging_waiter_t;
struct iree_hal_vulkan_queue_staging_waiter_t {
  // Next waiter in the ring-local list.
  iree_hal_vulkan_queue_staging_waiter_t* next;

  // Callback invoked after staging capacity may be available.
  iree_hal_vulkan_queue_staging_waiter_fn_t fn;

  // User data passed to |fn|.
  void* user_data;

  // Resource retained while the waiter is linked.
  iree_hal_resource_t* resource;

  // Whether this waiter is currently linked on a staging ring.
  bool is_queued;
};

typedef struct iree_hal_vulkan_queue_staging_slot_t {
  // Slot ordinal in the queue-local staging ring.
  uint32_t ordinal;

  // Byte offset of this slot within the staging buffer.
  iree_device_size_t buffer_offset;

  // Host mapped byte span for this slot.
  iree_byte_span_t host_span;

  // Whether a transfer currently owns this slot.
  bool in_use;
} iree_hal_vulkan_queue_staging_slot_t;

typedef struct iree_hal_vulkan_queue_staging_ring_t {
  // Mutex protecting slot ownership and waiters.
  iree_slim_mutex_t mutex;

  // Host allocator used for ring metadata and imported host backing.
  iree_allocator_t host_allocator;

  // Backing buffer for all slots.
  iree_hal_buffer_t* buffer;

  // Persistent host mapping of |buffer|.
  iree_hal_buffer_mapping_t mapping;

  // Owned ordinary host pages when imported into |buffer|. Native I/O uses
  // this original address instead of a potentially different driver mapping.
  iree_byte_span_t host_allocation;

  // Byte length of each staging slot.
  iree_device_size_t slot_size;

  // Number of slots in |slots|.
  uint32_t slot_count;

  // Head of transfers waiting for capacity.
  iree_hal_vulkan_queue_staging_waiter_t* waiter_head;

  // Tail link for transfers waiting for capacity.
  iree_hal_vulkan_queue_staging_waiter_t** waiter_tail;

  // Slot table.
  iree_hal_vulkan_queue_staging_slot_t slots[];
} iree_hal_vulkan_queue_staging_ring_t;

// Allocates and persistently maps the ring. The queue owns its lifetime and
// retires every loan and waiter before freeing it.
iree_status_t iree_hal_vulkan_queue_staging_ring_allocate(
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t slot_size, uint32_t slot_count,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_queue_staging_ring_t** out_ring);

void iree_hal_vulkan_queue_staging_ring_free(
    iree_hal_vulkan_queue_staging_ring_t* ring);

// Borrows one free slot without waiting. The loan exclusively owns its contents
// and returns it with release after all accesses complete.
bool iree_hal_vulkan_queue_staging_ring_try_acquire(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_slot_t** out_slot);

// Registers a placed waiter only while all slots are occupied. Returns false
// if capacity became available and the caller should retry acquisition. A
// queued waiter retains resource through cancellation or callback return.
bool iree_hal_vulkan_queue_staging_ring_queue_waiter(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_waiter_t* waiter,
    iree_hal_vulkan_queue_staging_waiter_fn_t fn, void* user_data,
    iree_hal_resource_t* resource);

// Removes a linked waiter. A callback already taken by release may still run;
// its resource reference keeps the placed waiter and user_data live.
void iree_hal_vulkan_queue_staging_ring_cancel_waiter(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_waiter_t* waiter);

// Returns a loan and wakes FIFO waiters while capacity remains available.
// Callbacks may acquire/release slots, cancel waiters, or requeue themselves.
void iree_hal_vulkan_queue_staging_ring_release(
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_slot_t* slot);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_VULKAN_QUEUE_STAGING_H_
