// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "event_timestamp.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/math.h"

hrx_event_timestamp_domain_t hrx_event_timestamp_domain_query(
    const iree_hal_queue_family_spec_t* family) {
  const hrx_event_timestamp_domain_t none = {0};
  if (!family) {
    return none;
  }
  // Unrelated compute or transfer families need not share this clock. This
  // family must identify one physical device so its records can be compared.
  if (iree_math_count_ones_u64(family->physical_device_affinity) != 1) {
    return none;
  }
  if (family->timestamp_frequency_hz == 0 ||
      family->timestamp_valid_bits == 0 || family->timestamp_valid_bits > 64) {
    return none;
  }
  const hrx_event_timestamp_domain_t domain = {
      .frequency_hz = family->timestamp_frequency_hz,
      .valid_bits = family->timestamp_valid_bits,
  };
  return domain;
}

float hrx_event_timestamp_domain_elapsed_ms(hrx_event_timestamp_domain_t domain,
                                            uint64_t start_tick,
                                            uint64_t stop_tick) {
  const uint64_t mask = domain.valid_bits == 64
                            ? UINT64_MAX
                            : ((1ull << domain.valid_bits) - 1ull);
  const uint64_t delta = (stop_tick - start_tick) & mask;
  // The counter's top bit, defined even for a full-width domain.
  const uint64_t sign_bit = (mask >> 1) + 1ull;
  // Formed as a double because the scaling below is floating point and the
  // result is a float. The offset itself fits int64_t at every width the
  // domain can carry: for delta >= sign_bit, mask - delta is at most
  // INT64_MAX, so -(mask - delta) - 1 bottoms out at exactly INT64_MIN.
  const double signed_delta =
      delta >= sign_bit ? -(double)(mask - delta) - 1.0 : (double)delta;
  return (float)(signed_delta * 1000.0 / (double)domain.frequency_hz);
}

// Bytes per slot: one 64-bit device clock tick.
#define HRX_EVENT_TIMESTAMP_SLOT_SIZE ((iree_device_size_t)8)

// Slots carved out of the first slab. Each subsequent slab doubles this up to
// HRX_EVENT_TIMESTAMP_MAX_SLOT_COUNT.
#define HRX_EVENT_TIMESTAMP_INITIAL_SLOT_COUNT 256

// Ceiling on slots per slab, bounding a single slab allocation.
#define HRX_EVENT_TIMESTAMP_MAX_SLOT_COUNT 8192

struct hrx_event_timestamp_slot_t {
  // Live references to this slot. Atomic rather than guarded by the pool
  // mutex, so a reader can take one under the event mutex a point is published
  // behind without nesting the pool mutex underneath it. Reaching zero returns
  // the slot to the pool, which the releasing thread does under that mutex.
  iree_atomic_ref_count_t ref_count;
  // Slab this slot was carved out of, and through it the owning pool. Never
  // changes after the slab is built.
  hrx_event_timestamp_slab_t* slab;
  // Byte offset of this slot's tick within the slab buffer. Always a multiple
  // of 8.
  iree_device_size_t offset;
  // Next slot on whichever pool list holds this slot, NULL at the tail.
  // Written only under the pool mutex and meaningless while the slot is held.
  hrx_event_timestamp_slot_t* next;
  // Semaphore whose |retire_value| retires the device write outstanding
  // against this slot, retained. NULL unless the slot is on the pending list.
  iree_hal_semaphore_t* retire_semaphore;
  // Payload value of |retire_semaphore| that retires the outstanding write.
  uint64_t retire_value;
};

struct hrx_event_timestamp_slab_t {
  // Pool that owns this slab, so a slot can be released without naming it.
  hrx_event_timestamp_pool_t* pool;
  // Next slab in the pool's slab list, NULL at the tail.
  hrx_event_timestamp_slab_t* next;
  // Device allocation holding |slot_count| tick slots, retained by the pool.
  iree_hal_buffer_t* buffer;
  // Number of slots carved out of |buffer|.
  iree_host_size_t slot_count;
  // Slot records for this slab, allocated in the same block as the slab.
  hrx_event_timestamp_slot_t slots[];
};

void hrx_event_timestamp_pool_initialize(iree_hal_pool_t* memory_pool,
                                         iree_allocator_t host_allocator,
                                         hrx_event_timestamp_pool_t* out_pool) {
  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->memory_pool = memory_pool;
  iree_hal_pool_retain(memory_pool);
  out_pool->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&out_pool->mutex);
  out_pool->next_slab_slot_count = HRX_EVENT_TIMESTAMP_INITIAL_SLOT_COUNT;
}

void hrx_event_timestamp_pool_set_source(hrx_event_timestamp_pool_t* pool,
                                         iree_hal_pool_t* memory_pool) {
  iree_slim_mutex_lock(&pool->mutex);
  if (!pool->memory_pool) {
    pool->memory_pool = memory_pool;
    iree_hal_pool_retain(memory_pool);
  }
  iree_slim_mutex_unlock(&pool->mutex);
}

void hrx_event_timestamp_pool_deinitialize(hrx_event_timestamp_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t pooled_slot_count = 0;
  for (hrx_event_timestamp_slab_t* slab = pool->slabs; slab != NULL;
       slab = slab->next) {
    pooled_slot_count += slab->slot_count;
  }
  iree_host_size_t returned_slot_count = 0;
  for (hrx_event_timestamp_slot_t* slot = pool->free_slots; slot != NULL;
       slot = slot->next) {
    ++returned_slot_count;
  }
  for (hrx_event_timestamp_slot_t* slot = pool->pending_slots; slot != NULL;
       slot = slot->next) {
    ++returned_slot_count;
  }

  // A slot record is storage inside its slab, so a holder that has not returned
  // its slot points into that slab. Every slot is back by the time the owning
  // device or context is destroyed: a record keeps its owning device/context
  // alive, and every copy of a point is released before its owner. A pool
  // reaching here with a slot out therefore keeps its slabs: freeing storage
  // that holder still names would corrupt memory wherever the assertion is
  // compiled out, and the leak this leaves shows up in every build.
  IREE_ASSERT(returned_slot_count == pooled_slot_count,
              "every timestamp slot must be released before its pool");
  if (returned_slot_count == pooled_slot_count) {
    for (hrx_event_timestamp_slot_t* slot = pool->pending_slots; slot != NULL;
         slot = slot->next) {
      iree_hal_semaphore_release(slot->retire_semaphore);
    }
    pool->pending_slots = NULL;
    pool->free_slots = NULL;
    hrx_event_timestamp_slab_t* slab = pool->slabs;
    while (slab != NULL) {
      hrx_event_timestamp_slab_t* next_slab = slab->next;
      iree_hal_buffer_release(slab->buffer);
      iree_allocator_free(pool->host_allocator, slab);
      slab = next_slab;
    }
    pool->slabs = NULL;
  }

  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_hal_pool_release(pool->memory_pool);
  pool->memory_pool = NULL;

  IREE_TRACE_ZONE_END(z0);
}

// Allocates a slab of |slot_count| slots and pushes every one of its slots onto
// the pool free list. Called with |pool->mutex| held.
static iree_status_t hrx_event_timestamp_pool_grow(
    hrx_event_timestamp_pool_t* pool, iree_host_size_t slot_count) {
  IREE_TRACE_ZONE_BEGIN(z0);

  hrx_event_timestamp_slab_t* slab = NULL;
  const iree_host_size_t slab_size =
      sizeof(*slab) + slot_count * sizeof(slab->slots[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(pool->host_allocator, slab_size, (void**)&slab));

  // The source contract requires coherent public host access for every device
  // family. The read path therefore needs no per-slot invalidation.
  iree_hal_buffer_t* buffer = NULL;
  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(
          slot_count * HRX_EVENT_TIMESTAMP_SLOT_SIZE,
          /*min_alignment=*/HRX_EVENT_TIMESTAMP_SLOT_SIZE);
  iree_status_t status = iree_hal_pool_allocate_buffer(
      pool->memory_pool, request, iree_infinite_timeout(), &buffer);
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
    hrx_event_timestamp_slot_t* slot = &slab->slots[i - 1];
    slot->slab = slab;
    slot->offset = (iree_device_size_t)(i - 1) * HRX_EVENT_TIMESTAMP_SLOT_SIZE;
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
// Called with |pool->mutex| held. A failed public semaphore does not prove that
// an accepted device write has stopped. Its slot remains pending until
// teardown.
static void hrx_event_timestamp_pool_drain_pending(
    hrx_event_timestamp_pool_t* pool) {
  hrx_event_timestamp_slot_t* slot = pool->pending_slots;
  hrx_event_timestamp_slot_t* still_pending = NULL;
  while (slot != NULL) {
    hrx_event_timestamp_slot_t* next = slot->next;
    uint64_t current_value = 0;
    iree_status_t status =
        iree_hal_semaphore_query(slot->retire_semaphore, &current_value);
    const bool write_retired =
        iree_status_is_ok(status) && current_value >= slot->retire_value;
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

iree_status_t hrx_event_timestamp_pool_acquire(
    hrx_event_timestamp_pool_t* pool, hrx_event_timestamp_slot_t** out_slot) {
  *out_slot = NULL;
  iree_slim_mutex_lock(&pool->mutex);

  iree_status_t status = iree_ok_status();
  if (pool->free_slots == NULL) {
    // Recover slots whose writes have retired since they were released before
    // asking the device for more memory.
    hrx_event_timestamp_pool_drain_pending(pool);
  }
  if (pool->free_slots == NULL) {
    const iree_host_size_t slot_count = pool->next_slab_slot_count;
    status = hrx_event_timestamp_pool_grow(pool, slot_count);
    if (iree_status_is_ok(status) &&
        slot_count < HRX_EVENT_TIMESTAMP_MAX_SLOT_COUNT) {
      pool->next_slab_slot_count = slot_count * 2;
    }
  }
  if (iree_status_is_ok(status)) {
    hrx_event_timestamp_slot_t* slot = pool->free_slots;
    pool->free_slots = slot->next;
    slot->next = NULL;
    iree_atomic_ref_count_init(&slot->ref_count);
    *out_slot = slot;
  }

  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

void hrx_event_timestamp_slot_retain(hrx_event_timestamp_slot_t* slot) {
  if (slot) {
    iree_atomic_ref_count_inc(&slot->ref_count);
  }
}

void hrx_event_timestamp_slot_release(hrx_event_timestamp_slot_t* slot,
                                      iree_hal_semaphore_t* retire_semaphore,
                                      uint64_t retire_value) {
  if (!slot) {
    return;
  }
  if (iree_atomic_ref_count_dec(&slot->ref_count) != 1) {
    return;
  }

  hrx_event_timestamp_pool_t* pool = slot->slab->pool;
  iree_slim_mutex_lock(&pool->mutex);

  // Queried here so the common already-retired case skips the pending list
  // rather than making the next acquire pay for the query.
  bool write_retired = retire_semaphore == NULL;
  if (!write_retired) {
    uint64_t current_value = 0;
    iree_status_t status =
        iree_hal_semaphore_query(retire_semaphore, &current_value);
    write_retired = iree_status_is_ok(status) && current_value >= retire_value;
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

iree_hal_buffer_t* hrx_event_timestamp_slot_buffer(
    const hrx_event_timestamp_slot_t* slot) {
  return slot->slab->buffer;
}

iree_device_size_t hrx_event_timestamp_slot_offset(
    const hrx_event_timestamp_slot_t* slot) {
  return slot->offset;
}

iree_status_t hrx_event_timestamp_elapsed_ms(
    hrx_event_timestamp_domain_t domain,
    const hrx_event_timestamp_slot_t* start,
    const hrx_event_timestamp_slot_t* stop, float* out_ms) {
  uint64_t start_tick = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
      hrx_event_timestamp_slot_buffer(start),
      hrx_event_timestamp_slot_offset(start), &start_tick, sizeof(start_tick)));
  uint64_t stop_tick = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
      hrx_event_timestamp_slot_buffer(stop),
      hrx_event_timestamp_slot_offset(stop), &stop_tick, sizeof(stop_tick)));
  *out_ms =
      hrx_event_timestamp_domain_elapsed_ms(domain, start_tick, stop_tick);
  return iree_ok_status();
}
