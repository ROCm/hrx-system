// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/event_timestamp_pool.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

// Bytes per slot: one 64-bit device clock tick.
#define IREE_HAL_STREAMING_EVENT_TIMESTAMP_SLOT_SIZE ((iree_device_size_t)8)

// Slots carved out of the first slab. Each subsequent slab doubles this up to
// IREE_HAL_STREAMING_EVENT_TIMESTAMP_MAX_SLOT_COUNT.
#define IREE_HAL_STREAMING_EVENT_TIMESTAMP_INITIAL_SLOT_COUNT 256

// Ceiling on slots per slab, bounding a single slab allocation.
#define IREE_HAL_STREAMING_EVENT_TIMESTAMP_MAX_SLOT_COUNT 8192

struct iree_hal_streaming_event_timestamp_slot_t {
  // Live references to this slot. Atomic rather than guarded by the pool
  // mutex, so a reader can take one under the event mutex a point is published
  // behind without nesting the pool mutex underneath it. Reaching zero returns
  // the slot to the pool, which the releasing thread does under that mutex.
  iree_atomic_ref_count_t ref_count;
  // Slab this slot was carved out of, and through it the owning pool. Never
  // changes after the slab is built.
  iree_hal_streaming_event_timestamp_slab_t* slab;
  // Byte offset of this slot's tick within the slab buffer. Always a multiple
  // of 8.
  iree_device_size_t offset;
  // Next slot on whichever pool list holds this slot, NULL at the tail.
  // Written only under the pool mutex and meaningless while the slot is held.
  iree_hal_streaming_event_timestamp_slot_t* next;
  // Semaphore whose |retire_value| retires the device write outstanding
  // against this slot, retained. NULL unless the slot is on the pending list.
  iree_hal_semaphore_t* retire_semaphore;
  // Payload value of |retire_semaphore| that retires the outstanding write.
  uint64_t retire_value;
};

struct iree_hal_streaming_event_timestamp_slab_t {
  // Pool that owns this slab, so a slot can be released without naming it.
  iree_hal_streaming_event_timestamp_pool_t* pool;
  // Next slab in the pool's slab list, NULL at the tail.
  iree_hal_streaming_event_timestamp_slab_t* next;
  // Device allocation holding |slot_count| tick slots, retained by the pool.
  iree_hal_buffer_t* buffer;
  // Number of slots carved out of |buffer|.
  iree_host_size_t slot_count;
  // Slot records for this slab, allocated in the same block as the slab.
  iree_hal_streaming_event_timestamp_slot_t slots[];
};

void iree_hal_streaming_event_timestamp_pool_initialize(
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_streaming_event_timestamp_pool_t* out_pool) {
  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->device_allocator = device_allocator;
  iree_hal_allocator_retain(device_allocator);
  out_pool->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&out_pool->mutex);
  out_pool->next_slab_slot_count =
      IREE_HAL_STREAMING_EVENT_TIMESTAMP_INITIAL_SLOT_COUNT;
}

void iree_hal_streaming_event_timestamp_pool_deinitialize(
    iree_hal_streaming_event_timestamp_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t pooled_slot_count = 0;
  for (iree_hal_streaming_event_timestamp_slab_t* slab = pool->slabs;
       slab != NULL; slab = slab->next) {
    pooled_slot_count += slab->slot_count;
  }
  iree_host_size_t returned_slot_count = 0;
  for (iree_hal_streaming_event_timestamp_slot_t* slot = pool->free_slots;
       slot != NULL; slot = slot->next) {
    ++returned_slot_count;
  }
  for (iree_hal_streaming_event_timestamp_slot_t* slot = pool->pending_slots;
       slot != NULL; slot = slot->next) {
    ++returned_slot_count;
  }

  // A slot record is storage inside its slab, so a holder that has not returned
  // its slot points into that slab. Every slot is back by the time the owning
  // context is destroyed: a record draws its slot from the pool of the
  // recording event's own context, the event retains that context, and every
  // copy of a point is released before the frame holding it returns. A pool
  // reaching here with a slot out therefore keeps its slabs: freeing storage
  // that holder still names would corrupt memory wherever the assertion is
  // compiled out, and the leak this leaves shows up in every build.
  IREE_ASSERT(returned_slot_count == pooled_slot_count,
              "every timestamp slot must be released before its pool");
  if (returned_slot_count == pooled_slot_count) {
    for (iree_hal_streaming_event_timestamp_slot_t* slot = pool->pending_slots;
         slot != NULL; slot = slot->next) {
      iree_hal_semaphore_release(slot->retire_semaphore);
    }
    pool->pending_slots = NULL;
    pool->free_slots = NULL;
    iree_hal_streaming_event_timestamp_slab_t* slab = pool->slabs;
    while (slab != NULL) {
      iree_hal_streaming_event_timestamp_slab_t* next_slab = slab->next;
      iree_hal_buffer_release(slab->buffer);
      iree_allocator_free(pool->host_allocator, slab);
      slab = next_slab;
    }
    pool->slabs = NULL;
  }

  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_hal_allocator_release(pool->device_allocator);
  pool->device_allocator = NULL;

  IREE_TRACE_ZONE_END(z0);
}

// Allocates a slab of |slot_count| slots and pushes every one of its slots onto
// the pool free list. Called with |pool->mutex| held.
static iree_status_t iree_hal_streaming_event_timestamp_pool_grow(
    iree_hal_streaming_event_timestamp_pool_t* pool,
    iree_host_size_t slot_count) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_event_timestamp_slab_t* slab = NULL;
  const iree_host_size_t slab_size =
      sizeof(*slab) + slot_count * sizeof(slab->slots[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(pool->host_allocator, slab_size, (void**)&slab));

  // The device writes a tick into this buffer at a queue point and the host
  // maps it back, so the allocation has to serve both: DEVICE_VISIBLE is what
  // states the device writes here, and every other host allocation this binding
  // hands the device asks for it beside HOST_LOCAL. HOST_COHERENT is a
  // requirement here and not a preference: the read path never invalidates, so
  // non-coherent memory would return stale ticks.
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
                IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      pool->device_allocator, params,
      slot_count * IREE_HAL_STREAMING_EVENT_TIMESTAMP_SLOT_SIZE, &buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(pool->host_allocator, slab);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  slab->pool = pool;
  slab->next = pool->slabs;
  slab->buffer = buffer;
  slab->slot_count = slot_count;
  pool->slabs = slab;

  // Reversed so the free list hands out ascending offsets, keeping slots for
  // records made together on the same few cache lines.
  for (iree_host_size_t i = slot_count; i > 0; --i) {
    iree_hal_streaming_event_timestamp_slot_t* slot = &slab->slots[i - 1];
    slot->slab = slab;
    slot->offset = (iree_device_size_t)(i - 1) *
                   IREE_HAL_STREAMING_EVENT_TIMESTAMP_SLOT_SIZE;
    iree_atomic_ref_count_init_value(&slot->ref_count, 0);
    slot->retire_semaphore = NULL;
    slot->retire_value = 0;
    slot->next = pool->free_slots;
    pool->free_slots = slot;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Moves every pending slot whose device write has retired onto the free list.
// Called with |pool->mutex| held. A failed semaphore counts as retired: the
// queue that would have made the write is in a failure state, so none is
// coming.
static void iree_hal_streaming_event_timestamp_pool_drain_pending(
    iree_hal_streaming_event_timestamp_pool_t* pool) {
  iree_hal_streaming_event_timestamp_slot_t* slot = pool->pending_slots;
  iree_hal_streaming_event_timestamp_slot_t* still_pending = NULL;
  while (slot != NULL) {
    iree_hal_streaming_event_timestamp_slot_t* next = slot->next;
    uint64_t current_value = 0;
    iree_status_t status =
        iree_hal_semaphore_query(slot->retire_semaphore, &current_value);
    const bool write_retired =
        !iree_status_is_ok(status) || current_value >= slot->retire_value;
    iree_status_ignore(status);
    if (write_retired) {
      iree_hal_semaphore_release(slot->retire_semaphore);
      slot->retire_semaphore = NULL;
      slot->retire_value = 0;
      slot->next = pool->free_slots;
      pool->free_slots = slot;
    } else {
      slot->next = still_pending;
      still_pending = slot;
    }
    slot = next;
  }
  pool->pending_slots = still_pending;
}

iree_status_t iree_hal_streaming_event_timestamp_pool_acquire(
    iree_hal_streaming_event_timestamp_pool_t* pool,
    iree_hal_streaming_event_timestamp_slot_t** out_slot) {
  *out_slot = NULL;
  iree_slim_mutex_lock(&pool->mutex);

  iree_status_t status = iree_ok_status();
  if (pool->free_slots == NULL) {
    // Recover slots whose writes have retired since they were released before
    // asking the device for more memory.
    iree_hal_streaming_event_timestamp_pool_drain_pending(pool);
  }
  if (pool->free_slots == NULL) {
    const iree_host_size_t slot_count = pool->next_slab_slot_count;
    status = iree_hal_streaming_event_timestamp_pool_grow(pool, slot_count);
    if (iree_status_is_ok(status) &&
        slot_count < IREE_HAL_STREAMING_EVENT_TIMESTAMP_MAX_SLOT_COUNT) {
      pool->next_slab_slot_count = slot_count * 2;
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_streaming_event_timestamp_slot_t* slot = pool->free_slots;
    pool->free_slots = slot->next;
    slot->next = NULL;
    iree_atomic_ref_count_init(&slot->ref_count);
    *out_slot = slot;
  }

  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

void iree_hal_streaming_event_timestamp_slot_retain(
    iree_hal_streaming_event_timestamp_slot_t* slot) {
  if (slot) {
    iree_atomic_ref_count_inc(&slot->ref_count);
  }
}

void iree_hal_streaming_event_timestamp_slot_release(
    iree_hal_streaming_event_timestamp_slot_t* slot,
    iree_hal_semaphore_t* retire_semaphore, uint64_t retire_value) {
  if (!slot) return;
  if (iree_atomic_ref_count_dec(&slot->ref_count) != 1) return;

  iree_hal_streaming_event_timestamp_pool_t* pool = slot->slab->pool;
  iree_slim_mutex_lock(&pool->mutex);

  // Queried here so the common already-retired case skips the pending list
  // rather than making the next acquire pay for the query.
  bool write_retired = retire_semaphore == NULL;
  if (!write_retired) {
    uint64_t current_value = 0;
    iree_status_t status =
        iree_hal_semaphore_query(retire_semaphore, &current_value);
    write_retired = !iree_status_is_ok(status) || current_value >= retire_value;
    iree_status_ignore(status);
  }

  if (write_retired) {
    slot->retire_semaphore = NULL;
    slot->retire_value = 0;
    slot->next = pool->free_slots;
    pool->free_slots = slot;
  } else {
    iree_hal_semaphore_retain(retire_semaphore);
    slot->retire_semaphore = retire_semaphore;
    slot->retire_value = retire_value;
    slot->next = pool->pending_slots;
    pool->pending_slots = slot;
  }

  iree_slim_mutex_unlock(&pool->mutex);
}

iree_hal_buffer_t* iree_hal_streaming_event_timestamp_slot_buffer(
    const iree_hal_streaming_event_timestamp_slot_t* slot) {
  return slot->slab->buffer;
}

iree_device_size_t iree_hal_streaming_event_timestamp_slot_offset(
    const iree_hal_streaming_event_timestamp_slot_t* slot) {
  return slot->offset;
}
