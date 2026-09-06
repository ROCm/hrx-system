// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_EVENT_TIMESTAMP_POOL_H_
#define IREE_EXPERIMENTAL_STREAMING_EVENT_TIMESTAMP_POOL_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Suballocator handing out the 8-byte device-writable tick slots that
// timing-enabled event records capture their device clock into. Slots are
// carved out of shared device slabs and recycled through a free list, so a
// record costs a free-list pop rather than a device allocation.

// One 8-byte device timestamp slot. Reference counted: the record that acquires
// the slot holds the only reference and hands it to the point it commits, and
// each reader that acquires a copy of that point takes one more, so a reader
// can read a tick without a concurrent record recycling the slot underneath it.
// The last reference returns the slot to its pool.
typedef struct iree_hal_streaming_event_timestamp_slot_t
    iree_hal_streaming_event_timestamp_slot_t;

// One device allocation a run of slots is carved out of.
typedef struct iree_hal_streaming_event_timestamp_slab_t
    iree_hal_streaming_event_timestamp_slab_t;

// Per-context pool of device timestamp slots. Thread-safe once initialized: the
// lists and slabs are guarded by |mutex|, which callers never hold, and the
// allocators are constant from initialize onward. Initialization and teardown
// are exclusive to the context that owns the pool, so neither takes the mutex.
typedef struct iree_hal_streaming_event_timestamp_pool_t {
  // Allocator slab storage comes from, retained for the life of the pool.
  iree_hal_allocator_t* device_allocator;
  // Allocator the host-side slab and slot records come from.
  iree_allocator_t host_allocator;
  // Guards every field below, and each slot record's list linkage and
  // retirement condition. A slot's reference count is not among them: it is
  // atomic so a reader can take a reference under the event mutex the record's
  // point is published behind without nesting this one underneath it.
  iree_slim_mutex_t mutex;
  // Head of the slab list. Slabs are released only at deinitialize, so a slot's
  // storage never moves while it is in use.
  iree_hal_streaming_event_timestamp_slab_t* slabs;
  // LIFO of slots that are free to hand out, linked through slot->next.
  iree_hal_streaming_event_timestamp_slot_t* free_slots;
  // Released slots whose device write may still be outstanding, linked through
  // slot->next. Drained on acquire once the write has retired.
  iree_hal_streaming_event_timestamp_slot_t* pending_slots;
  // Slot count for the next slab; doubles per slab.
  iree_host_size_t next_slab_slot_count;
} iree_hal_streaming_event_timestamp_pool_t;

// Initializes |out_pool| against |device_allocator|, which is retained. No
// device memory is allocated until the first slot is acquired.
void iree_hal_streaming_event_timestamp_pool_initialize(
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_streaming_event_timestamp_pool_t* out_pool);

// Releases every slab and slot record owned by |pool|. A slot whose device
// write is still outstanding does not block this: dropping the pool's
// reference to a slab buffer does not free it while a queue submission still
// holds one, and the amdgpu queue takes that reference for a timestamp's
// target buffer on both its immediate and its deferred path. The retention is
// a fact of the queue implementation; iree_hal_queue_timestamp does not
// promise it.
//
// Every slot must already have been released. That holds because a record
// draws its slot from the pool of the recording event's own context, the event
// retains that context, and every copy of a point a reader takes is released
// before the frame that took it returns: a pool reached here has outlived
// every event that could still name one of its slots. A slot record lives
// inside its slab, so a pool that still has one out keeps its slabs: freeing
// storage the holder points into would be a use-after-free, and the leak this
// leaves is visible.
void iree_hal_streaming_event_timestamp_pool_deinitialize(
    iree_hal_streaming_event_timestamp_pool_t* pool);

// Acquires an 8-byte slot from |pool| with one reference, growing the pool by a
// slab when no slot is free. The slot's contents are undefined until a record
// writes it.
iree_status_t iree_hal_streaming_event_timestamp_pool_acquire(
    iree_hal_streaming_event_timestamp_pool_t* pool,
    iree_hal_streaming_event_timestamp_slot_t** out_slot);

// Takes one more reference to |slot|; NULL is ignored.
void iree_hal_streaming_event_timestamp_slot_retain(
    iree_hal_streaming_event_timestamp_slot_t* slot);

// Drops one reference to |slot|; NULL is ignored. The last reference returns it
// to its pool: straight to the free list when |retire_semaphore| is NULL or has
// already reached |retire_value|, and otherwise onto a pending list the pool
// drains before it grows, so a slot a submission may still write is never
// handed out again. A NULL |retire_semaphore| states that no write is
// outstanding, which only the record that failed to enqueue can say.
void iree_hal_streaming_event_timestamp_slot_release(
    iree_hal_streaming_event_timestamp_slot_t* slot,
    iree_hal_semaphore_t* retire_semaphore, uint64_t retire_value);

// Returns the slab buffer |slot| lives in. Not retained: the buffer outlives
// the slot, and any queue submission referencing it retains it itself.
iree_hal_buffer_t* iree_hal_streaming_event_timestamp_slot_buffer(
    const iree_hal_streaming_event_timestamp_slot_t* slot);

// Returns the byte offset of |slot|'s tick within its slab buffer. Always a
// multiple of 8.
iree_device_size_t iree_hal_streaming_event_timestamp_slot_offset(
    const iree_hal_streaming_event_timestamp_slot_t* slot);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_EVENT_TIMESTAMP_POOL_H_
