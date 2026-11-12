// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_LIBHRX_EVENT_TIMESTAMP_H_
#define LIBHRX_SRC_LIBHRX_EVENT_TIMESTAMP_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Facts converting a pair of ticks captured on one queue family into a
// duration. Populated or zeroed as a unit: a zero |frequency_hz| means the
// family advertises no clock whose ticks this layer can convert.
typedef struct hrx_event_timestamp_domain_t {
  // Ticks per second of the domain, or 0 when the device advertises none.
  uint64_t frequency_hz;
  // Number of low bits defined in a tick, in [1, 64]; the counter wraps at
  // this width. Zero exactly when |frequency_hz| is zero.
  uint32_t valid_bits;
} hrx_event_timestamp_domain_t;

// Reads the recording queue family's clock. Ticks from different families
// are not comparable merely because they have the same frequency and width.
// Returns a zeroed domain if the family has no timestamp clock or covers more
// than one physical device. A NULL family publishes no facts.
// Synchronization: none (reads immutable device facts).
hrx_event_timestamp_domain_t hrx_event_timestamp_domain_query(
    const iree_hal_queue_family_spec_t* family);

// Converts the interval between two ticks captured in |domain| to milliseconds.
// Only the low |domain.valid_bits| of a tick are defined and the counter wraps
// there, so the difference is reduced modulo that width; a width of 64 makes
// the reduction the identity.
//
// Reading that reduced difference as a signed offset from the counter's top bit
// is this layer's choice and not something the device facts state. It is what
// makes a pair captured in order a positive duration and a reversed pair a
// negative one, and what it costs is that an interval longer than half the
// counter range reports negative: out of reach at 64 bits and 100 MHz, but 21
// seconds on a 32-bit counter at the same rate.
//
// |domain| must be populated and both ticks must belong to the same family.
// Synchronization: none (pure arithmetic).
float hrx_event_timestamp_domain_elapsed_ms(hrx_event_timestamp_domain_t domain,
                                            uint64_t start_tick,
                                            uint64_t stop_tick);

// Suballocator handing out the 8-byte device-writable tick slots that
// timing-enabled event records capture their device clock into. Slots are
// carved out of shared device slabs and recycled through a free list, so a
// record costs a free-list pop rather than a device allocation.

// One 8-byte device timestamp slot. Reference counted: the record that acquires
// the slot holds the only reference and hands it to the point it commits, and
// each reader that acquires a copy of that point takes one more, so a reader
// can read a tick without a concurrent record recycling the slot underneath it.
// The last reference returns the slot to its pool.
typedef struct hrx_event_timestamp_slot_t hrx_event_timestamp_slot_t;

// One device allocation a run of slots is carved out of.
typedef struct hrx_event_timestamp_slab_t hrx_event_timestamp_slab_t;

// Pool of device timestamp slots owned by a device or binding context.
// Thread-safe once initialized: the lists and slabs are guarded by |mutex|,
// and the allocators are constant from initialize onward. Initialization and
// teardown are exclusive to the owner, so neither takes the mutex.
typedef struct hrx_event_timestamp_pool_t {
  // Explicit coherent host-visible source, retained for the pool lifetime.
  iree_hal_pool_t* memory_pool;
  // Allocator the host-side slab and slot records come from.
  iree_allocator_t host_allocator;
  // Guards every field below, and each slot record's list linkage and
  // retirement condition. A slot's reference count is not among them: it is
  // atomic so a reader can take a reference under the event mutex the record's
  // point is published behind without nesting this one underneath it.
  iree_slim_mutex_t mutex;
  // Head of the slab list. Slabs are released only at deinitialize, so a slot's
  // storage never moves while it is in use.
  hrx_event_timestamp_slab_t* slabs;
  // LIFO of slots that are free to hand out, linked through slot->next.
  hrx_event_timestamp_slot_t* free_slots;
  // Released slots whose device write may still be outstanding, linked through
  // slot->next. Drained on acquire once the write has retired.
  hrx_event_timestamp_slot_t* pending_slots;
  // Slot count for the next slab; doubles per slab.
  iree_host_size_t next_slab_slot_count;
} hrx_event_timestamp_pool_t;

// Initializes |out_pool| against |memory_pool|, which is retained. No
// device memory is allocated until the first slot is acquired.
void hrx_event_timestamp_pool_initialize(iree_hal_pool_t* memory_pool,
                                         iree_allocator_t host_allocator,
                                         hrx_event_timestamp_pool_t* out_pool);

// Releases every slab and slot record owned by |pool|. A slot whose device
// write is still outstanding does not block this: dropping the pool's
// reference to a slab buffer does not free it while a queue submission still
// holds one. Users of this pool must submit through queues that retain their
// timestamp target buffers until the writes retire.
//
// Every slot must already have been released. Each record and its readers
// keep the owning device or context alive until returning their slots. Slot
// metadata lives inside the slab, so an outstanding slot reference prevents
// teardown from freeing that slab's metadata.
void hrx_event_timestamp_pool_deinitialize(hrx_event_timestamp_pool_t* pool);

// Acquires an 8-byte slot from |pool| with one reference, growing the pool by a
// slab when no slot is free. The slot's contents are undefined until a record
// writes it.
iree_status_t hrx_event_timestamp_pool_acquire(
    hrx_event_timestamp_pool_t* pool, hrx_event_timestamp_slot_t** out_slot);

// Takes one more reference to |slot|; NULL is ignored.
void hrx_event_timestamp_slot_retain(hrx_event_timestamp_slot_t* slot);

// Drops one reference to |slot|; NULL is ignored. The last reference returns it
// to its pool: straight to the free list when |retire_semaphore| is NULL or has
// successfully reached |retire_value|, and otherwise onto a pending list the
// pool drains before it grows, so a slot a submission may still write is never
// handed out again. A NULL |retire_semaphore| states that no write is
// outstanding, which only the record that failed to enqueue can say. A failed
// semaphore does not prove that an accepted write stopped; its slot is retained
// until pool teardown instead of being reused by another record.
void hrx_event_timestamp_slot_release(hrx_event_timestamp_slot_t* slot,
                                      iree_hal_semaphore_t* retire_semaphore,
                                      uint64_t retire_value);

// Returns the slab buffer |slot| lives in. Not retained: the buffer outlives
// the slot, and any queue submission referencing it retains it itself.
iree_hal_buffer_t* hrx_event_timestamp_slot_buffer(
    const hrx_event_timestamp_slot_t* slot);

// Returns the byte offset of |slot|'s tick within its slab buffer. Always a
// multiple of 8.
iree_device_size_t hrx_event_timestamp_slot_offset(
    const hrx_event_timestamp_slot_t* slot);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_EVENT_TIMESTAMP_H_
