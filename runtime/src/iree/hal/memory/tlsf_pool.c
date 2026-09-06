// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf_pool.h"

#include "iree/async/frontier.h"
#include "iree/async/notification.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/memory/tracing.h"

enum {
  IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY = 4,
  IREE_HAL_TLSF_POOL_INLINE_TRANSACTION_CAPACITY = 8,
};

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tlsf_pool_release_node_t {
  // Intrusive next pointer in pool->pending_release_head or
  // pool-owned release-node lists.
  struct iree_hal_tlsf_pool_release_node_t* next;

  // Stable slab object holding |block_index|.
  struct iree_hal_tlsf_pool_slab_t* slab;

  // TLSF block handle owned by this reservation.
  iree_hal_memory_tlsf_block_index_t block_index;

  // Backing bytes charged to this reservation.
  iree_device_size_t charged_length;

  // Backing offset within |slab| for this reservation.
  iree_device_size_t backing_offset;

  // ASAN backing layout for this reservation.
  iree_hal_asan_allocation_layout_t asan_layout;
} iree_hal_tlsf_pool_release_node_t;

typedef struct iree_hal_tlsf_pool_slab_t {
  // Offset allocator for the address range backed by |slab|.
  iree_hal_memory_tlsf_t tlsf;

  // Physical memory backing |tlsf|'s offset range.
  iree_hal_slab_t slab;

  // Current position in pool->slabs, updated after slab-array compaction.
  uint16_t index;
} iree_hal_tlsf_pool_slab_t;

typedef struct iree_hal_tlsf_pool_allocation_t {
  // Index into pool->slabs identifying the owning TLSF instance.
  uint16_t slab_index;

  // Allocation returned by the owning TLSF instance.
  iree_hal_memory_tlsf_allocation_t allocation;
} iree_hal_tlsf_pool_allocation_t;

typedef struct iree_hal_tlsf_pool_t {
  // Base pool resource for vtable dispatch and ref counting.
  iree_hal_pool_t base;

  // Provider used to acquire additional slabs as the pool grows.
  iree_hal_slab_provider_t* slab_provider;

  // Notification signaled when a reservation release may unblock waiters.
  iree_async_notification_t* notification;

  // Guards TLSF mutation and the slab array.
  iree_slim_mutex_t mutex;

  // Template options used when initializing each new TLSF slab.
  iree_hal_memory_tlsf_options_t slab_options;

  // Minimum backing byte length requested for newly grown slabs.
  iree_device_size_t backing_slab_length;

  // Maximum user-visible reservation length served by the TLSF slabs.
  iree_device_size_t max_reservation_size;

  // Dynamic array of committed slab pointers. Protected by |mutex|.
  iree_hal_tlsf_pool_slab_t** slabs;

  // Number of initialized entries in |slabs|. Protected by |mutex|.
  uint32_t slab_count;

  // Allocated entry capacity of |slabs|. Protected by |mutex|.
  iree_host_size_t slab_capacity;

  // Preferred slab to try first for the next allocation. Protected by |mutex|.
  uint16_t preferred_slab_index;

  // Bounded set of slabs that recently received releases. Protected by |mutex|.
  uint16_t
      reuse_candidate_slab_indices[IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY];

  // Valid entries in |reuse_candidate_slab_indices|. Protected by |mutex|.
  uint8_t reuse_candidate_slab_count;

  // Next candidate slot to overwrite when the bounded set is full.
  uint8_t reuse_candidate_slab_cursor;

  // Approximate committed bytes across all slabs for lock-free stats queries.
  iree_atomic_int64_t bytes_committed;

  // Approximate committed slab count for lock-free stats queries.
  iree_atomic_int32_t committed_slab_count;

  // Pending release nodes pushed by queue-retirement paths.
  iree_atomic_intptr_t pending_release_head;

  // Free list of release nodes ready for reuse. Protected by |mutex|.
  iree_hal_tlsf_pool_release_node_t* release_node_free_head;

  // Head of poisoned ASAN releases retained before TLSF reuse.
  iree_hal_tlsf_pool_release_node_t* quarantine_head;

  // Tail of poisoned ASAN releases retained before TLSF reuse.
  iree_hal_tlsf_pool_release_node_t* quarantine_tail;

  // Backing bytes currently retained in the ASAN quarantine list.
  iree_device_size_t quarantine_size;

  // Optional completion predicate for try-before-fence reuse.
  iree_hal_pool_epoch_query_t epoch_query;

  // Host allocator used for pool metadata.
  iree_allocator_t host_allocator;

  // Stable named-memory stream for logical reservations from this pool.
  iree_hal_memory_trace_t trace;

  // Immutable memory properties provided by |slab_provider|.
  iree_hal_slab_provider_properties_t slab_properties;

  // ASAN policy used to shape hidden backing ranges.
  iree_hal_asan_pool_options_t asan_options;

  // Logical byte budget for live reservations. 0 means unlimited.
  iree_device_size_t budget_limit;

  // Byte size of each release node including inline frontier storage.
  iree_host_size_t release_node_size;

  // Byte offset of inline frontier storage within a release node.
  iree_host_size_t release_frontier_offset;

  // Scratch storage used under |mutex| to hold rejected block indices while a
  // reserve call keeps searching for a satisfiable block.
  iree_hal_memory_tlsf_block_index_t* rejected_block_indices;
  iree_host_size_t rejected_block_capacity;

  // Approximate live reservation bytes for lock-free stats queries.
  iree_atomic_int64_t bytes_reserved;

  // Approximate live reservation count for lock-free stats queries.
  iree_atomic_int32_t reservation_count;

  // Total successful reservation acquisitions.
  iree_atomic_int64_t reserve_count;

  // Total reservation releases.
  iree_atomic_int64_t release_count;

  // Reserves that hit frontier-dominated reuse.
  iree_atomic_int64_t reuse_count;

  // Reserves where dominance check failed.
  iree_atomic_int64_t reuse_miss_count;

  // Reserves from fresh (never-used) blocks.
  iree_atomic_int64_t fresh_count;

  // Reserves that returned EXHAUSTED.
  iree_atomic_int64_t exhausted_count;

  // Reserves that returned OVER_BUDGET.
  iree_atomic_int64_t over_budget_count;

  // Reserves that returned NEEDS_WAIT.
  iree_atomic_int64_t wait_count;
} iree_hal_tlsf_pool_t;

typedef struct iree_hal_tlsf_pool_materialize_state_t
    iree_hal_tlsf_pool_materialize_state_t;

// Per-buffer element in an owning materialization transaction.
typedef struct iree_hal_tlsf_pool_materialize_element_t {
  // Shared transaction state controlling the ownership commit.
  iree_hal_tlsf_pool_materialize_state_t* state;

  // Reservation released when the committed buffer is destroyed.
  iree_hal_pool_reservation_t reservation;

  // Materialized buffer staged until the complete transaction succeeds.
  iree_hal_buffer_t* buffer;
} iree_hal_tlsf_pool_materialize_element_t;

// Shared state for an owning materialization transaction.
struct iree_hal_tlsf_pool_materialize_state_t {
  // Borrowed from the wrapped buffer's creator. Pool owners must keep the pool
  // alive until all buffers sourced from it are destroyed.
  iree_hal_pool_t* pool;

  // Host allocator used for this state object.
  iree_allocator_t host_allocator;

  // Number of materialized buffers still referencing this transaction.
  iree_atomic_int32_t reference_count;

  // True after every buffer was materialized and reservation ownership moved.
  bool ownership_committed;

  // Per-buffer transaction elements.
  iree_hal_tlsf_pool_materialize_element_t elements[];
};

// Staged result for one reservation acquisition. Transactions use staging so
// public output arrays remain untouched unless the operation succeeds.
typedef struct iree_hal_tlsf_pool_acquire_element_t {
  // Reservation produced for the request.
  iree_hal_pool_reservation_t reservation;

  // Acquisition metadata produced for the request.
  iree_hal_pool_acquire_info_t info;
} iree_hal_tlsf_pool_acquire_element_t;

static const iree_hal_pool_vtable_t iree_hal_tlsf_pool_vtable;
static void iree_hal_tlsf_pool_destroy(iree_hal_pool_t* base_pool);

static const char* IREE_HAL_TLSF_POOL_TRACE_ID = "iree-hal-tlsf-pool";

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

static inline iree_async_frontier_t* iree_hal_tlsf_pool_release_node_frontier(
    const iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node) {
  return (iree_async_frontier_t*)((uint8_t*)node +
                                  pool->release_frontier_offset);
}

static void iree_hal_tlsf_pool_push_pending_release(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node) {
  intptr_t expected =
      iree_atomic_load(&pool->pending_release_head, iree_memory_order_relaxed);
  do {
    node->next = (iree_hal_tlsf_pool_release_node_t*)expected;
  } while (!iree_atomic_compare_exchange_weak(
      &pool->pending_release_head, &expected, (intptr_t)node,
      iree_memory_order_release, iree_memory_order_relaxed));
}

static iree_hal_tlsf_pool_release_node_t*
iree_hal_tlsf_pool_take_pending_releases(iree_hal_tlsf_pool_t* pool) {
  return (iree_hal_tlsf_pool_release_node_t*)iree_atomic_exchange(
      &pool->pending_release_head, 0, iree_memory_order_acquire);
}

static bool iree_hal_tlsf_pool_try_charge_reservation(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t charged_length) {
  if (pool->budget_limit == 0) {
    iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)charged_length,
                          iree_memory_order_relaxed);
    return true;
  }
  int64_t expected =
      iree_atomic_load(&pool->bytes_reserved, iree_memory_order_relaxed);
  for (;;) {
    const iree_device_size_t current = (iree_device_size_t)expected;
    if (current > pool->budget_limit ||
        charged_length > pool->budget_limit - current) {
      return false;
    }
    const int64_t desired = (int64_t)(current + charged_length);
    if (iree_atomic_compare_exchange_weak(&pool->bytes_reserved, &expected,
                                          desired, iree_memory_order_relaxed,
                                          iree_memory_order_relaxed)) {
      return true;
    }
  }
}

static void iree_hal_tlsf_pool_uncharge_reservation(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t charged_length) {
  iree_atomic_fetch_add(&pool->bytes_reserved, -(int64_t)charged_length,
                        iree_memory_order_relaxed);
}

static bool iree_hal_tlsf_pool_adjust_charged_reservation(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t* charged_length,
    iree_device_size_t actual_length) {
  if (actual_length <= *charged_length) {
    iree_hal_tlsf_pool_uncharge_reservation(pool,
                                            *charged_length - actual_length);
    *charged_length = actual_length;
    return true;
  }
  const iree_device_size_t additional_length = actual_length - *charged_length;
  if (!iree_hal_tlsf_pool_try_charge_reservation(pool, additional_length)) {
    return false;
  }
  *charged_length = actual_length;
  return true;
}

static void iree_hal_tlsf_pool_note_reuse_candidate(iree_hal_tlsf_pool_t* pool,
                                                    uint16_t slab_index)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  for (uint8_t i = 0; i < pool->reuse_candidate_slab_count; ++i) {
    if (pool->reuse_candidate_slab_indices[i] == slab_index) return;
  }
  if (pool->reuse_candidate_slab_count <
      IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY) {
    pool->reuse_candidate_slab_indices[pool->reuse_candidate_slab_count++] =
        slab_index;
    return;
  }
  pool->reuse_candidate_slab_indices[pool->reuse_candidate_slab_cursor] =
      slab_index;
  pool->reuse_candidate_slab_cursor =
      (uint8_t)((pool->reuse_candidate_slab_cursor + 1) %
                IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY);
}

static iree_status_t iree_hal_tlsf_pool_acquire_release_node(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t** out_node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* node = pool->release_node_free_head;
  if (node) {
    pool->release_node_free_head = node->next;
  } else {
    IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_tlsf_pool_grow_release_nodes");
    iree_status_t status = iree_allocator_malloc(
        pool->host_allocator, pool->release_node_size, (void**)&node);
    IREE_TRACE_ZONE_END(z0);
    if (!iree_status_is_ok(status)) return status;
  }
  node->next = NULL;
  node->slab = NULL;
  node->block_index = 0;
  node->charged_length = 0;
  node->backing_offset = 0;
  memset(&node->asan_layout, 0, sizeof(node->asan_layout));
  iree_async_frontier_initialize(
      iree_hal_tlsf_pool_release_node_frontier(pool, node), 0);
  *out_node = node;
  return iree_ok_status();
}

static void iree_hal_tlsf_pool_recycle_release_node(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  if (!node) return;
  node->next = pool->release_node_free_head;
  pool->release_node_free_head = node;
}

static void iree_hal_tlsf_pool_free_release_node_list(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    iree_allocator_free(pool->host_allocator, node);
    node = next;
  }
}

static void iree_hal_tlsf_pool_free_release_nodes(iree_hal_tlsf_pool_t* pool)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_free_release_node_list(pool, pool->release_node_free_head);
  pool->release_node_free_head = NULL;
}

static void iree_hal_tlsf_pool_return_release_node_to_tlsf(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_async_frontier_t* death_frontier =
      iree_hal_tlsf_pool_release_node_frontier(pool, node);
  iree_hal_tlsf_pool_slab_t* slab = node->slab;
  const uint16_t slab_index = slab->index;
  iree_hal_memory_tlsf_free(
      &slab->tlsf, node->block_index,
      death_frontier->entry_count > 0 ? death_frontier : NULL);
  if (death_frontier->entry_count == 0) {
    pool->preferred_slab_index = slab_index;
  } else {
    iree_hal_tlsf_pool_note_reuse_candidate(pool, slab_index);
  }
  iree_hal_tlsf_pool_recycle_release_node(pool, node);
}

static bool iree_hal_tlsf_pool_asan_quarantine_is_enabled(
    const iree_hal_tlsf_pool_t* pool) {
  return iree_hal_asan_pool_options_is_enabled(&pool->asan_options) &&
         pool->asan_options.quarantine_size > 0;
}

static void iree_hal_tlsf_pool_add_quarantine_size(iree_hal_tlsf_pool_t* pool,
                                                   iree_device_size_t length)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  if (length > IREE_DEVICE_SIZE_MAX - pool->quarantine_size) {
    pool->quarantine_size = IREE_DEVICE_SIZE_MAX;
  } else {
    pool->quarantine_size += length;
  }
}

static void iree_hal_tlsf_pool_subtract_quarantine_size(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t length)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  pool->quarantine_size -= iree_min(pool->quarantine_size, length);
}

static void iree_hal_tlsf_pool_quarantine_release_node(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  node->next = NULL;
  if (pool->quarantine_tail) {
    pool->quarantine_tail->next = node;
  } else {
    pool->quarantine_head = node;
  }
  pool->quarantine_tail = node;
  iree_hal_tlsf_pool_add_quarantine_size(pool, node->charged_length);

  while (pool->quarantine_size > pool->asan_options.quarantine_size &&
         pool->quarantine_head) {
    iree_hal_tlsf_pool_release_node_t* release_node = pool->quarantine_head;
    pool->quarantine_head = release_node->next;
    if (!pool->quarantine_head) {
      pool->quarantine_tail = NULL;
    }
    iree_hal_tlsf_pool_subtract_quarantine_size(pool,
                                                release_node->charged_length);
    release_node->next = NULL;
    iree_hal_tlsf_pool_return_release_node_to_tlsf(pool, release_node);
  }
}

static void iree_hal_tlsf_pool_drain_pending_releases(
    iree_hal_tlsf_pool_t* pool)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* node =
      iree_hal_tlsf_pool_take_pending_releases(pool);
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    if (iree_hal_tlsf_pool_asan_quarantine_is_enabled(pool)) {
      iree_hal_tlsf_pool_quarantine_release_node(pool, node);
    } else {
      node->next = NULL;
      iree_hal_tlsf_pool_return_release_node_to_tlsf(pool, node);
    }
    node = next;
  }
}

static void iree_hal_tlsf_pool_flush_quarantine(iree_hal_tlsf_pool_t* pool)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* node = pool->quarantine_head;
  pool->quarantine_head = NULL;
  pool->quarantine_tail = NULL;
  pool->quarantine_size = 0;
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    node->next = NULL;
    iree_hal_tlsf_pool_return_release_node_to_tlsf(pool, node);
    node = next;
  }
}

static bool iree_hal_tlsf_pool_frontier_is_satisfied(
    const iree_hal_tlsf_pool_t* pool,
    const iree_async_frontier_t* requester_frontier,
    const iree_async_frontier_t* death_frontier,
    iree_hal_memory_tlsf_block_flags_t block_flags) {
  if (!death_frontier) {
    return (block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED) == 0;
  }
  if (block_flags & IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED) {
    return false;
  }
  if (requester_frontier) {
    const iree_async_frontier_comparison_t comparison =
        iree_async_frontier_compare(requester_frontier, death_frontier);
    if (comparison == IREE_ASYNC_FRONTIER_AFTER ||
        comparison == IREE_ASYNC_FRONTIER_EQUAL) {
      return true;
    }
  }
  if (!pool->epoch_query.fn) return false;

  iree_host_size_t requester_index = 0;
  for (uint8_t i = 0; i < death_frontier->entry_count; ++i) {
    const iree_async_axis_t axis = death_frontier->entries[i].axis;
    const uint64_t epoch = death_frontier->entries[i].epoch;
    while (requester_frontier &&
           requester_index < requester_frontier->entry_count &&
           requester_frontier->entries[requester_index].axis < axis) {
      ++requester_index;
    }
    if (requester_frontier &&
        requester_index < requester_frontier->entry_count &&
        requester_frontier->entries[requester_index].axis == axis &&
        requester_frontier->entries[requester_index].epoch >= epoch) {
      continue;
    }
    if (!pool->epoch_query.fn(pool->epoch_query.user_data, axis, epoch)) {
      return false;
    }
  }
  return true;
}

static void iree_hal_tlsf_pool_restore_rejected_blocks(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_slab_t* slab,
    uint32_t rejected_block_count)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  for (uint32_t i = 0; i < rejected_block_count; ++i) {
    iree_hal_memory_tlsf_restore(&slab->tlsf, pool->rejected_block_indices[i]);
  }
}

static iree_status_t iree_hal_tlsf_pool_ensure_rejected_capacity(
    iree_hal_tlsf_pool_t* pool, iree_host_size_t capacity)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  if (pool->rejected_block_capacity >= capacity) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
      pool->host_allocator, capacity, sizeof(*pool->rejected_block_indices),
      &pool->rejected_block_capacity, (void**)&pool->rejected_block_indices));
  return iree_ok_status();
}

static iree_status_t iree_hal_tlsf_pool_aligned_slab_length(
    const iree_hal_tlsf_pool_t* pool, iree_device_size_t* out_slab_length) {
  const iree_device_size_t alignment =
      pool->slab_options.alignment
          ? pool->slab_options.alignment
          : (iree_device_size_t)IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  const iree_device_size_t unaligned_length = pool->backing_slab_length;
  if (unaligned_length > IREE_DEVICE_SIZE_MAX - (alignment - 1)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "slab length %" PRIdsz
                            " overflows when aligned to %" PRIdsz,
                            unaligned_length, alignment);
  }
  const iree_device_size_t required_length =
      iree_device_align(unaligned_length, alignment);
  *out_slab_length = required_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_tlsf_pool_append_slab(iree_hal_tlsf_pool_t* pool,
                                                    uint16_t* out_slab_index)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  *out_slab_index = 0;
  if (pool->slab_count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "TLSF pool reached maximum slab count (%" PRIu32
                            ")",
                            (uint32_t)UINT16_MAX);
  }
  if (pool->slab_count >= pool->slab_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        pool->host_allocator, pool->slab_count + 1, sizeof(*pool->slabs),
        &pool->slab_capacity, (void**)&pool->slabs));
  }

  iree_device_size_t slab_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_tlsf_pool_aligned_slab_length(pool, &slab_length));

  iree_hal_tlsf_pool_slab_t* slab_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator, sizeof(*slab_entry), (void**)&slab_entry));
  memset(slab_entry, 0, sizeof(*slab_entry));

  iree_hal_slab_t slab;
  iree_status_t status = iree_hal_slab_provider_acquire_slab(
      pool->slab_provider, slab_length, &slab);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(pool->host_allocator, slab_entry);
    return status;
  }

  iree_hal_memory_tlsf_options_t tlsf_options = pool->slab_options;
  tlsf_options.range_length = slab.length;
  status = iree_hal_memory_tlsf_initialize(tlsf_options, pool->host_allocator,
                                           &slab_entry->tlsf);
  if (!iree_status_is_ok(status)) {
    iree_hal_slab_provider_release_slab(pool->slab_provider, &slab);
    iree_allocator_free(pool->host_allocator, slab_entry);
    return status;
  }
  slab_entry->slab = slab;

  const uint16_t slab_index = (uint16_t)pool->slab_count;
  slab_entry->index = slab_index;
  pool->slabs[slab_index] = slab_entry;
  ++pool->slab_count;
  pool->preferred_slab_index = slab_index;
  iree_atomic_store(&pool->committed_slab_count, (int32_t)pool->slab_count,
                    iree_memory_order_release);
  iree_atomic_fetch_add(&pool->bytes_committed, (int64_t)slab.length,
                        iree_memory_order_relaxed);
  *out_slab_index = slab_index;
  return iree_ok_status();
}

static void iree_hal_tlsf_pool_deinitialize_slabs(iree_hal_tlsf_pool_t* pool) {
  for (uint32_t i = 0; i < pool->slab_count; ++i) {
    iree_hal_tlsf_pool_slab_t* slab = pool->slabs[i];
    if (!slab) continue;
    if (slab->tlsf.block_storage) {
      iree_hal_memory_tlsf_deinitialize(&slab->tlsf);
    }
    if (slab->slab.length > 0) {
      iree_hal_slab_provider_release_slab(pool->slab_provider, &slab->slab);
    }
    iree_allocator_free(pool->host_allocator, slab);
  }
  iree_allocator_free(pool->host_allocator, pool->slabs);
  pool->slabs = NULL;
  pool->slab_count = 0;
  pool->slab_capacity = 0;
  pool->preferred_slab_index = 0;
  pool->reuse_candidate_slab_count = 0;
  pool->reuse_candidate_slab_cursor = 0;
  iree_atomic_store(&pool->committed_slab_count, 0, iree_memory_order_release);
  iree_atomic_store(&pool->bytes_committed, 0, iree_memory_order_release);
}

static iree_status_t iree_hal_tlsf_pool_return_allocation(
    iree_hal_tlsf_pool_t* pool,
    const iree_hal_tlsf_pool_allocation_t* pool_allocation,
    iree_device_size_t byte_length, iree_device_size_t charged_length,
    const iree_hal_asan_allocation_layout_t* asan_layout,
    iree_hal_pool_acquire_result_t result,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_slab_t* slab = pool->slabs[pool_allocation->slab_index];
  const iree_hal_memory_tlsf_allocation_t* allocation =
      &pool_allocation->allocation;
  iree_hal_tlsf_pool_release_node_t* release_node = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_tlsf_pool_acquire_release_node(pool, &release_node));
  release_node->slab = slab;
  release_node->block_index = allocation->block_index;
  release_node->charged_length = charged_length;
  release_node->backing_offset = allocation->offset;
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    release_node->asan_layout = *asan_layout;
  }
  pool->preferred_slab_index = pool_allocation->slab_index;

  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->offset =
      allocation->offset +
      (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)
           ? asan_layout->user_offset
           : 0);
  out_reservation->byte_length = byte_length;
  out_reservation->block_handle = (uint64_t)(uintptr_t)release_node;
  out_reservation->slab_index = pool_allocation->slab_index;

  memset(out_info, 0, sizeof(*out_info));
  out_info->result = result;

  iree_atomic_fetch_add(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);
  switch (result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
      iree_atomic_fetch_add(&pool->reuse_count, 1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_atomic_fetch_add(&pool->fresh_count, 1, iree_memory_order_relaxed);
      break;
    default:
      IREE_ASSERT(false, "invalid successful TLSF pool result: %u", result);
      break;
  }
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &slab->slab, allocation->offset,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED, asan_layout);
  }
  iree_hal_memory_trace_alloc(
      &pool->trace, (uint8_t*)slab->slab.base_ptr + out_reservation->offset,
      out_reservation->byte_length);
  *out_result = result;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_tlsf_pool_create(
    iree_hal_tlsf_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification,
    iree_hal_pool_epoch_query_t epoch_query, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(slab_provider);
  IREE_ASSERT_ARGUMENT(notification);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (options.tlsf_options.range_length == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range_length must be > 0");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_slab_provider_validate_asan_options(slab_provider,
                                                       &options.asan));

  iree_hal_memory_tlsf_options_t tlsf_options = options.tlsf_options;
  if (tlsf_options.alignment == 0) {
    tlsf_options.alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  }
  iree_device_size_t backing_slab_length = tlsf_options.range_length;
  if (iree_hal_asan_pool_options_is_enabled(&options.asan)) {
    const iree_device_size_t backing_length_alignment = iree_max(
        options.asan.backing_alignment ? options.asan.backing_alignment
                                       : options.asan.shadow_granule_size,
        options.asan.shadow_granule_size);
    tlsf_options.alignment =
        iree_max(tlsf_options.alignment, backing_length_alignment);
    iree_hal_asan_allocation_layout_t slab_layout;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_asan_calculate_allocation_layout(
                &options.asan, options.tlsf_options.range_length,
                tlsf_options.alignment, &slab_layout));
    backing_slab_length = slab_layout.backing_length;
  }

  iree_hal_tlsf_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_hal_pool_initialize(&iree_hal_tlsf_pool_vtable, &pool->base);
  iree_slim_mutex_initialize(&pool->mutex);
  iree_atomic_store(&pool->pending_release_head, 0, iree_memory_order_relaxed);
  pool->host_allocator = host_allocator;
  pool->epoch_query = epoch_query;
  pool->budget_limit = options.budget_limit;
  pool->slab_options = tlsf_options;
  pool->backing_slab_length = backing_slab_length;
  pool->max_reservation_size = options.tlsf_options.range_length;
  pool->asan_options = options.asan;
  iree_atomic_store(&pool->bytes_committed, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pool->committed_slab_count, 0, iree_memory_order_relaxed);

  iree_hal_slab_provider_retain(slab_provider);
  pool->slab_provider = slab_provider;
  iree_async_notification_retain(notification);
  pool->notification = notification;
  iree_hal_slab_provider_query_properties(slab_provider,
                                          &pool->slab_properties);

  iree_status_t status = iree_hal_memory_trace_initialize_pool(
      options.trace_name, IREE_HAL_TLSF_POOL_TRACE_ID, host_allocator,
      &pool->trace);
  if (iree_status_is_ok(status)) {
    uint16_t initial_slab_index = 0;
    iree_slim_mutex_lock(&pool->mutex);
    status = iree_hal_tlsf_pool_append_slab(pool, &initial_slab_index);
    iree_slim_mutex_unlock(&pool->mutex);
  }
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_tlsf_pool_release_node_t), &pool->release_node_size,
        IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                  iree_alignof(iree_async_frontier_entry_t),
                                  &pool->release_frontier_offset),
        IREE_STRUCT_FIELD(pool->slabs[0]->tlsf.frontier_capacity,
                          iree_async_frontier_entry_t, NULL));
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_tlsf_pool_destroy((iree_hal_pool_t*)pool);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_pool = (iree_hal_pool_t*)pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_tlsf_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;

  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_tlsf_pool_drain_pending_releases(pool);
  iree_hal_tlsf_pool_flush_quarantine(pool);
  iree_hal_tlsf_pool_free_release_nodes(pool);
  iree_slim_mutex_unlock(&pool->mutex);

  iree_hal_tlsf_pool_deinitialize_slabs(pool);
  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_allocator_free(pool->host_allocator, pool->rejected_block_indices);
  iree_hal_memory_trace_deinitialize(&pool->trace);
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_release(pool->slab_provider);
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_allocator_free(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_tlsf_pool_try_acquire_from_slab(
    iree_hal_tlsf_pool_t* pool, uint16_t slab_index,
    iree_device_size_t allocation_length,
    const iree_async_frontier_t* requester_frontier, bool record_reuse_miss,
    iree_hal_tlsf_pool_allocation_t* out_allocation,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_status_t status = iree_ok_status();
  iree_hal_tlsf_pool_slab_t* slab = pool->slabs[slab_index];
  uint32_t rejected_block_count = 0;
  *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;

  while (iree_status_is_ok(status)) {
    iree_hal_memory_tlsf_allocation_t allocation;
    iree_hal_memory_tlsf_allocate_result_t allocation_result =
        IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED;
    status = iree_hal_memory_tlsf_try_allocate(&slab->tlsf, allocation_length,
                                               &allocation, &allocation_result);
    if (!iree_status_is_ok(status) ||
        allocation_result == IREE_HAL_MEMORY_TLSF_ALLOCATE_EXHAUSTED) {
      break;
    }

    if (!iree_hal_tlsf_pool_frontier_is_satisfied(pool, requester_frontier,
                                                  allocation.death_frontier,
                                                  allocation.block_flags)) {
      if (record_reuse_miss) {
        iree_atomic_fetch_add(&pool->reuse_miss_count, 1,
                              iree_memory_order_relaxed);
      }
      status = iree_hal_tlsf_pool_ensure_rejected_capacity(
          pool, (iree_host_size_t)rejected_block_count + 1);
      if (!iree_status_is_ok(status)) {
        iree_hal_memory_tlsf_restore(&slab->tlsf, allocation.block_index);
        break;
      }
      pool->rejected_block_indices[rejected_block_count++] =
          allocation.block_index;
      continue;
    }

    out_allocation->slab_index = slab_index;
    out_allocation->allocation = allocation;
    *out_result = allocation.death_frontier ? IREE_HAL_POOL_ACQUIRE_OK
                                            : IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    break;
  }

  iree_hal_tlsf_pool_restore_rejected_blocks(pool, slab, rejected_block_count);
  return status;
}

static iree_status_t iree_hal_tlsf_pool_validate_reservation_request(
    const iree_hal_tlsf_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request) {
  const iree_device_size_t size = request->allocation_size;
  const iree_device_size_t alignment =
      request->params.min_alignment ? request->params.min_alignment : 1;
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation size must be > 0");
  }
  if (!iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment (%" PRIdsz
                            ") must be a power of two",
                            alignment);
  }
  const iree_device_size_t pool_alignment =
      pool->slab_options.alignment
          ? pool->slab_options.alignment
          : (iree_device_size_t)IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  if (alignment > pool_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment %" PRIdsz
                            " exceeds TLSF pool alignment %" PRIdsz,
                            alignment, pool_alignment);
  }
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_asan_allocation_layout_t asan_layout;
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        &pool->asan_options, size, alignment, &asan_layout));
  }
  if (size > pool->max_reservation_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reservation size %" PRIdsz
                            " exceeds TLSF pool limit %" PRIdsz,
                            size, pool->max_reservation_size);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tlsf_pool_acquire_one_reservation_locked(
    iree_hal_tlsf_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  const iree_device_size_t size = request->allocation_size;
  const iree_device_size_t alignment =
      request->params.min_alignment ? request->params.min_alignment : 1;
  iree_hal_asan_allocation_layout_t asan_layout = {0};
  iree_device_size_t allocation_length = size;
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        &pool->asan_options, size, alignment, &asan_layout));
    allocation_length = asan_layout.backing_length;
  }
  iree_device_size_t charged_length = allocation_length;
  if (!iree_hal_tlsf_pool_try_charge_reservation(pool, charged_length)) {
    iree_atomic_fetch_add(&pool->over_budget_count, 1,
                          iree_memory_order_relaxed);
    memset(out_reservation, 0, sizeof(*out_reservation));
    memset(out_info, 0, sizeof(*out_info));
    out_info->result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    return iree_ok_status();
  }

  iree_hal_tlsf_pool_allocation_t selected_allocation;
  iree_hal_pool_acquire_result_t selected_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  bool has_selected_allocation = false;
  bool growth_required = false;

  iree_status_t status = iree_ok_status();
  const uint16_t preferred_slab_index = pool->preferred_slab_index;
  if (preferred_slab_index < pool->slab_count) {
    status = iree_hal_tlsf_pool_try_acquire_from_slab(
        pool, preferred_slab_index, allocation_length, requester_frontier,
        /*record_reuse_miss=*/true, &selected_allocation, &selected_result);
    has_selected_allocation = selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
                              selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  }

  for (uint8_t i = 0; i < pool->reuse_candidate_slab_count &&
                      iree_status_is_ok(status) && !has_selected_allocation;
       ++i) {
    const uint16_t slab_index = pool->reuse_candidate_slab_indices[i];
    if (slab_index == preferred_slab_index || slab_index >= pool->slab_count) {
      continue;
    }
    status = iree_hal_tlsf_pool_try_acquire_from_slab(
        pool, slab_index, allocation_length, requester_frontier,
        /*record_reuse_miss=*/true, &selected_allocation, &selected_result);
    has_selected_allocation = selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
                              selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  }

  if (iree_status_is_ok(status) && !has_selected_allocation) {
    if (iree_all_bits_set(flags, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH)) {
      growth_required = true;
    } else {
      uint16_t slab_index = 0;
      status = iree_hal_tlsf_pool_append_slab(pool, &slab_index);
      if (iree_status_is_ok(status)) {
        status = iree_hal_tlsf_pool_try_acquire_from_slab(
            pool, slab_index, allocation_length, requester_frontier,
            /*record_reuse_miss=*/true, &selected_allocation, &selected_result);
        has_selected_allocation =
            selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
            selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    if (has_selected_allocation) {
      iree_hal_tlsf_pool_slab_t* slab =
          pool->slabs[selected_allocation.slab_index];
      if (!iree_hal_tlsf_pool_adjust_charged_reservation(
              pool, &charged_length, selected_allocation.allocation.length)) {
        iree_atomic_fetch_add(&pool->over_budget_count, 1,
                              iree_memory_order_relaxed);
        iree_hal_memory_tlsf_restore(
            &slab->tlsf, selected_allocation.allocation.block_index);
        memset(out_reservation, 0, sizeof(*out_reservation));
        memset(out_info, 0, sizeof(*out_info));
        out_info->result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
        *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
        iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
        charged_length = 0;
      } else {
        if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
          status = iree_hal_asan_extend_allocation_layout(
              selected_allocation.allocation.length, &asan_layout);
        }
        if (iree_status_is_ok(status)) {
          status = iree_hal_tlsf_pool_return_allocation(
              pool, &selected_allocation, size, charged_length, &asan_layout,
              selected_result, out_reservation, out_info, out_result);
        }
        if (!iree_status_is_ok(status)) {
          if (charged_length > 0) {
            iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
            charged_length = 0;
          }
          iree_hal_memory_tlsf_restore(
              &slab->tlsf, selected_allocation.allocation.block_index);
        }
      }
    } else {
      iree_atomic_fetch_add(&pool->exhausted_count, 1,
                            iree_memory_order_relaxed);
      memset(out_reservation, 0, sizeof(*out_reservation));
      memset(out_info, 0, sizeof(*out_info));
      if (growth_required) {
        out_info->flags |= IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED;
      }
      out_info->result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
      charged_length = 0;
    }
  } else {
    iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
    charged_length = 0;
  }
  return status;
}

// Rolls back a reservation acquired by the current transaction before it was
// made visible to the caller.
static void iree_hal_tlsf_pool_rollback_reservation_locked(
    iree_hal_tlsf_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
    const iree_hal_pool_acquire_info_t* info)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* release_node =
      (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservation->block_handle;
  iree_hal_tlsf_pool_slab_t* slab = release_node->slab;
  iree_hal_memory_trace_free(
      &pool->trace, (uint8_t*)slab->slab.base_ptr + reservation->offset);
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &slab->slab, release_node->backing_offset,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED, &release_node->asan_layout);
  }
  iree_hal_memory_tlsf_restore(&slab->tlsf, release_node->block_index);
  iree_hal_tlsf_pool_uncharge_reservation(pool, release_node->charged_length);
  iree_hal_tlsf_pool_recycle_release_node(pool, release_node);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, -1, iree_memory_order_relaxed);
  switch (info->result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
      iree_atomic_fetch_add(&pool->reuse_count, -1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_atomic_fetch_add(&pool->fresh_count, -1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      iree_atomic_fetch_add(&pool->wait_count, -1, iree_memory_order_relaxed);
      break;
    default:
      IREE_ASSERT(false, "invalid TLSF rollback result: %u", info->result);
      break;
  }
}

static iree_status_t iree_hal_tlsf_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_result_t* out_result) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_tlsf_pool_validate_reservation_request(pool, &requests[i]));
  }

  iree_hal_tlsf_pool_acquire_element_t
      inline_elements[IREE_HAL_TLSF_POOL_INLINE_TRANSACTION_CAPACITY];
  iree_hal_tlsf_pool_acquire_element_t* elements = inline_elements;
  bool elements_allocated = false;
  iree_status_t status = iree_ok_status();
  if (request_count > IREE_ARRAYSIZE(inline_elements)) {
    status = iree_allocator_malloc_array(pool->host_allocator, request_count,
                                         sizeof(*elements), (void**)&elements);
    elements_allocated = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    memset(elements, 0, request_count * sizeof(*elements));
  }

  iree_host_size_t acquired_count = 0;
  bool did_rollback = false;
  iree_hal_pool_acquire_result_t transaction_result =
      IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&pool->mutex);
    iree_hal_tlsf_pool_drain_pending_releases(pool);
    while (acquired_count < request_count && iree_status_is_ok(status)) {
      iree_hal_pool_acquire_result_t item_result = IREE_HAL_POOL_ACQUIRE_NONE;
      status = iree_hal_tlsf_pool_acquire_one_reservation_locked(
          pool, &requests[acquired_count], requester_frontier, flags,
          &elements[acquired_count].reservation, &elements[acquired_count].info,
          &item_result);
      if (!iree_status_is_ok(status)) break;
      switch (item_result) {
        case IREE_HAL_POOL_ACQUIRE_OK:
          if (transaction_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH) {
            transaction_result = IREE_HAL_POOL_ACQUIRE_OK;
          }
          break;
        case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
          break;
        case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
          transaction_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
          break;
        case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
        case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
          transaction_result = item_result;
          break;
        case IREE_HAL_POOL_ACQUIRE_NONE:
        default:
          status = iree_make_status(
              IREE_STATUS_INTERNAL,
              "TLSF pool produced unknown acquire result %u", item_result);
          break;
      }
      if (!iree_status_is_ok(status) ||
          item_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
          item_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
        break;
      }
      ++acquired_count;
    }
    if (!iree_status_is_ok(status) || acquired_count != request_count) {
      did_rollback = acquired_count != 0;
      for (iree_host_size_t i = 0; i < acquired_count; ++i) {
        iree_hal_tlsf_pool_rollback_reservation_locked(
            pool, &elements[i].reservation, &elements[i].info);
        memset(&elements[i], 0, sizeof(elements[i]));
      }
    }
    iree_slim_mutex_unlock(&pool->mutex);
  }
  if (did_rollback) {
    iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
  }

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_infos[i] = elements[i].info;
    }
    if (acquired_count == request_count) {
      for (iree_host_size_t i = 0; i < request_count; ++i) {
        out_reservations[i] = elements[i].reservation;
      }
    }
    *out_result = transaction_result;
  }
  if (elements_allocated) {
    iree_allocator_free(pool->host_allocator, elements);
  }
  return status;
}

static void iree_hal_tlsf_pool_release_one_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  iree_hal_tlsf_pool_release_node_t* release_node =
      (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservation->block_handle;
  iree_async_frontier_t* release_frontier =
      iree_hal_tlsf_pool_release_node_frontier(pool, release_node);
  iree_hal_tlsf_pool_slab_t* slab = release_node->slab;

  if (death_frontier && death_frontier->entry_count > 0) {
    if (death_frontier->entry_count <= slab->tlsf.frontier_capacity) {
      if (release_frontier != death_frontier) {
        memcpy(release_frontier, death_frontier,
               sizeof(iree_async_frontier_t) +
                   (iree_host_size_t)death_frontier->entry_count *
                       sizeof(iree_async_frontier_entry_t));
      }
    } else {
      // Sentinel count larger than capacity. The drain path forwards this
      // header to TLSF free(), which marks the block tainted without reading
      // entries.
      iree_async_frontier_initialize(
          release_frontier, (uint8_t)(slab->tlsf.frontier_capacity + 1u));
    }
  } else {
    iree_async_frontier_initialize(release_frontier, 0);
  }

  iree_hal_memory_trace_free(
      &pool->trace, (uint8_t*)slab->slab.base_ptr + reservation->offset);
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_slab_provider_advise_asan_range(
        pool->slab_provider, &slab->slab, release_node->backing_offset,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED, &release_node->asan_layout);
  }

  iree_hal_tlsf_pool_push_pending_release(pool, release_node);

  iree_hal_tlsf_pool_uncharge_reservation(pool, release_node->charged_length);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);
}

static void iree_hal_tlsf_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_tlsf_pool_release_one_reservation(base_pool, &reservations[i],
                                               death_frontier);
  }
  iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static void iree_hal_tlsf_pool_buffer_release(void* user_data,
                                              iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_tlsf_pool_materialize_element_t* element =
      (iree_hal_tlsf_pool_materialize_element_t*)user_data;
  iree_hal_tlsf_pool_materialize_state_t* state = element->state;
  if (state->ownership_committed) {
    iree_hal_pool_release_reservations(state->pool, 1, &element->reservation,
                                       NULL);
  }
  const int32_t previous_count = iree_atomic_fetch_sub(
      &state->reference_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous_count > 0);
  if (previous_count == 1) {
    iree_allocator_free(state->host_allocator, state);
  }
}

static iree_status_t iree_hal_tlsf_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;

  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    if (reservations[i].byte_length < requests[i].allocation_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reservation %" PRIhsz " has %" PRIdsz
          " bytes but its allocation request requires %" PRIdsz,
          i, reservations[i].byte_length, requests[i].allocation_size);
    }
    if (!reservations[i].block_handle) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reservation %" PRIhsz " has no TLSF release node", i);
    }
    iree_hal_tlsf_pool_release_node_t* release_node =
        (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservations[i]
            .block_handle;
    if (!release_node->slab) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "reservation %" PRIhsz " has no TLSF slab", i);
    }
  }

  const bool transfer_ownership = iree_all_bits_set(
      flags, IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP);
  iree_hal_tlsf_pool_materialize_state_t* state = NULL;
  iree_hal_buffer_t*
      inline_buffers[IREE_HAL_TLSF_POOL_INLINE_TRANSACTION_CAPACITY] = {0};
  iree_hal_buffer_t** staged_buffers = inline_buffers;
  bool staged_buffers_allocated = false;
  iree_status_t status = iree_ok_status();
  if (transfer_ownership) {
    if (reservation_count > INT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "materialization count exceeds INT32_MAX");
    }
    iree_host_size_t state_size = 0;
    if (!iree_host_size_checked_mul_add(
            reservation_count, sizeof(iree_hal_tlsf_pool_materialize_element_t),
            sizeof(*state), &state_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "materialization state size overflow");
    }
    status =
        iree_allocator_malloc(pool->host_allocator, state_size, (void**)&state);
    if (!iree_status_is_ok(status)) return status;
    memset(state, 0, state_size);
    state->pool = base_pool;
    state->host_allocator = pool->host_allocator;
    iree_atomic_store(&state->reference_count, (int32_t)reservation_count,
                      iree_memory_order_relaxed);
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      state->elements[i].state = state;
      state->elements[i].reservation = reservations[i];
    }
  } else if (reservation_count > IREE_ARRAYSIZE(inline_buffers)) {
    status = iree_allocator_malloc_array(
        pool->host_allocator, reservation_count, sizeof(*staged_buffers),
        (void**)&staged_buffers);
    if (!iree_status_is_ok(status)) return status;
    staged_buffers_allocated = true;
    memset(staged_buffers, 0, reservation_count * sizeof(*staged_buffers));
  }

  iree_host_size_t materialized_count = 0;
  while (materialized_count < reservation_count && iree_status_is_ok(status)) {
    iree_hal_tlsf_pool_release_node_t* release_node =
        (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)
            reservations[materialized_count]
                .block_handle;
    // The release node owns a stable slab pointer for the reservation lifetime.
    // Trim may compact the slab array but cannot release a live slab.
    const iree_hal_slab_t* slab = &release_node->slab->slab;
    iree_hal_buffer_release_callback_t release_callback =
        iree_hal_buffer_release_callback_null();
    iree_hal_buffer_t** staged_buffer = &staged_buffers[materialized_count];
    if (state) {
      release_callback.fn = iree_hal_tlsf_pool_buffer_release;
      release_callback.user_data = &state->elements[materialized_count];
      staged_buffer = &state->elements[materialized_count].buffer;
    }
    status = iree_hal_slab_provider_wrap_buffer(
        pool->slab_provider, slab, reservations[materialized_count].offset,
        reservations[materialized_count].byte_length,
        requests[materialized_count].params, release_callback, staged_buffer);
    if (iree_status_is_ok(status)) ++materialized_count;
  }

  if (iree_status_is_ok(status)) {
    if (state) state->ownership_committed = true;
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      out_buffers[i] = state ? state->elements[i].buffer : staged_buffers[i];
    }
  } else {
    if (state) {
      iree_atomic_store(&state->reference_count, (int32_t)materialized_count,
                        iree_memory_order_relaxed);
    }
    for (iree_host_size_t i = 0; i < materialized_count; ++i) {
      iree_hal_buffer_release(state ? state->elements[i].buffer
                                    : staged_buffers[i]);
    }
    if (state && materialized_count == 0) {
      iree_allocator_free(pool->host_allocator, state);
    }
  }
  if (staged_buffers_allocated) {
    iree_allocator_free(pool->host_allocator, staged_buffers);
  }
  return status;
}

static void iree_hal_tlsf_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_tlsf_pool_t* pool = (const iree_hal_tlsf_pool_t*)base_pool;
  out_capabilities->memory_type = pool->slab_properties.memory_type;
  out_capabilities->supported_usage = pool->slab_properties.supported_usage;
  out_capabilities->queue_family_affinity =
      pool->slab_properties.queue_family_affinity;
  out_capabilities->atomic_operations = pool->slab_properties.atomic_operations;
  out_capabilities->min_allocation_size = 1;
  out_capabilities->max_allocation_size = pool->max_reservation_size;
}

static void iree_hal_tlsf_pool_query_stats(const iree_hal_pool_t* base_pool,
                                           iree_hal_pool_stats_t* out_stats) {
  const iree_hal_tlsf_pool_t* pool = (const iree_hal_tlsf_pool_t*)base_pool;
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  out_stats->bytes_committed = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_committed, iree_memory_order_relaxed);
  out_stats->bytes_free =
      out_stats->bytes_committed > out_stats->bytes_reserved
          ? out_stats->bytes_committed - out_stats->bytes_reserved
          : 0;
  out_stats->budget_limit = pool->budget_limit;
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count = (uint32_t)iree_atomic_load(
      &pool->committed_slab_count, iree_memory_order_relaxed);
  out_stats->reserve_count = (uint64_t)iree_atomic_load(
      &pool->reserve_count, iree_memory_order_relaxed);
  out_stats->release_count = (uint64_t)iree_atomic_load(
      &pool->release_count, iree_memory_order_relaxed);
  out_stats->reuse_count =
      (uint64_t)iree_atomic_load(&pool->reuse_count, iree_memory_order_relaxed);
  out_stats->reuse_miss_count = (uint64_t)iree_atomic_load(
      &pool->reuse_miss_count, iree_memory_order_relaxed);
  out_stats->fresh_count =
      (uint64_t)iree_atomic_load(&pool->fresh_count, iree_memory_order_relaxed);
  out_stats->exhausted_count = (uint64_t)iree_atomic_load(
      &pool->exhausted_count, iree_memory_order_relaxed);
  out_stats->over_budget_count = (uint64_t)iree_atomic_load(
      &pool->over_budget_count, iree_memory_order_relaxed);
  out_stats->wait_count =
      (uint64_t)iree_atomic_load(&pool->wait_count, iree_memory_order_relaxed);
}

static void iree_hal_tlsf_pool_release_slab_locked(iree_hal_tlsf_pool_t* pool,
                                                   uint16_t slab_index)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_slab_t* slab = pool->slabs[slab_index];
  const iree_device_size_t slab_length = slab->slab.length;
  iree_hal_memory_tlsf_deinitialize(&slab->tlsf);
  iree_hal_slab_provider_release_slab(pool->slab_provider, &slab->slab);
  iree_allocator_free(pool->host_allocator, slab);

  const uint32_t trailing_slab_count = pool->slab_count - slab_index - 1;
  if (trailing_slab_count) {
    memmove(&pool->slabs[slab_index], &pool->slabs[slab_index + 1],
            (iree_host_size_t)trailing_slab_count * sizeof(pool->slabs[0]));
  }
  --pool->slab_count;
  pool->slabs[pool->slab_count] = NULL;
  for (uint32_t i = slab_index; i < pool->slab_count; ++i) {
    pool->slabs[i]->index = (uint16_t)i;
  }
  iree_atomic_store(&pool->committed_slab_count, (int32_t)pool->slab_count,
                    iree_memory_order_release);
  iree_atomic_fetch_sub(&pool->bytes_committed, (int64_t)slab_length,
                        iree_memory_order_relaxed);
}

static void iree_hal_tlsf_pool_trim_unused_slabs_locked(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t min_bytes_to_keep)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  uint32_t slab_index = 0;
  while (slab_index < pool->slab_count) {
    iree_hal_tlsf_pool_slab_t* slab = pool->slabs[slab_index];
    const iree_async_frontier_t* death_frontier = NULL;
    iree_hal_memory_tlsf_block_flags_t block_flags =
        IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_NONE;
    const bool is_reclaimable =
        iree_hal_memory_tlsf_query_full_free_block(&slab->tlsf, &death_frontier,
                                                   &block_flags) &&
        iree_hal_tlsf_pool_frontier_is_satisfied(
            pool, /*requester_frontier=*/NULL, death_frontier, block_flags);
    const iree_device_size_t committed = (iree_device_size_t)iree_atomic_load(
        &pool->bytes_committed, iree_memory_order_relaxed);
    if (is_reclaimable && committed >= min_bytes_to_keep &&
        slab->slab.length <= committed - min_bytes_to_keep) {
      iree_hal_tlsf_pool_release_slab_locked(pool, (uint16_t)slab_index);
      continue;
    }
    ++slab_index;
  }
  pool->preferred_slab_index = 0;
  pool->reuse_candidate_slab_count = 0;
  pool->reuse_candidate_slab_cursor = 0;
}

IREE_API_EXPORT iree_status_t iree_hal_tlsf_pool_trim_to(
    iree_hal_pool_t* base_pool, iree_device_size_t min_bytes_to_keep) {
  if (!iree_hal_resource_is(base_pool, &iree_hal_tlsf_pool_vtable)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pool is not a TLSF pool");
  }
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_tlsf_pool_drain_pending_releases(pool);
  iree_hal_tlsf_pool_flush_quarantine(pool);
  iree_hal_tlsf_pool_trim_unused_slabs_locked(pool, min_bytes_to_keep);
  iree_hal_tlsf_pool_free_release_nodes(pool);
  iree_slim_mutex_unlock(&pool->mutex);
  iree_hal_slab_provider_trim(pool->slab_provider,
                              IREE_HAL_SLAB_PROVIDER_TRIM_FLAG_EXCESS);
  return iree_ok_status();
}

static iree_status_t iree_hal_tlsf_pool_trim(iree_hal_pool_t* base_pool) {
  return iree_hal_tlsf_pool_trim_to(base_pool, /*min_bytes_to_keep=*/0);
}

static iree_async_notification_t* iree_hal_tlsf_pool_notification(
    iree_hal_pool_t* base_pool) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  return pool->notification;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_tlsf_pool_vtable = {
    .destroy = iree_hal_tlsf_pool_destroy,
    .acquire_reservations = iree_hal_tlsf_pool_acquire_reservations,
    .release_reservations = iree_hal_tlsf_pool_release_reservations,
    .materialize_reservations = iree_hal_tlsf_pool_materialize_reservations,
    .query_capabilities = iree_hal_tlsf_pool_query_capabilities,
    .query_stats = iree_hal_tlsf_pool_query_stats,
    .trim = iree_hal_tlsf_pool_trim,
    .notification = iree_hal_tlsf_pool_notification,
};
