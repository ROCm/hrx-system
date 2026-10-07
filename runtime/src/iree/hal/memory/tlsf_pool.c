// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf_pool.h"

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/memory/buffer_range.h"
#include "iree/hal/memory/maintenance.h"
#include "iree/hal/memory/tlsf_pool_reservation.h"
#include "iree/hal/memory/tracing.h"

enum {
  IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY = 4,
  IREE_HAL_TLSF_POOL_INLINE_TRANSACTION_CAPACITY = 8,
};

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tlsf_pool_slab_t {
  // Next owned slab, or next detached slab awaiting release outside the mutex.
  struct iree_hal_tlsf_pool_slab_t* next;

  // Previous owned slab, allowing constant-time removal of unused backing.
  struct iree_hal_tlsf_pool_slab_t* previous;

  // Next slab whose free state must be checked by maintenance.
  struct iree_hal_tlsf_pool_slab_t* candidate_next;

  // Whether this slab is already linked in the return-candidate list.
  bool return_candidate;

  // Cold transaction preparations preserving this slab until their next
  // attempt.
  uint32_t preparation_count;

  // Offset allocator for the address range backed by |slab|.
  iree_hal_memory_tlsf_t tlsf;

  // Prepared backing range; borrowed from source_range for a finite arena.
  iree_hal_pool_buffer_range_t range;

  // Ordinary parent reservation owned until this slab is returned.
  iree_hal_pool_reservation_t reservation;

  // Exact history returned with reservation before TLSF metadata is freed.
  const iree_async_frontier_t* release_frontier;
} iree_hal_tlsf_pool_slab_t;

typedef struct iree_hal_tlsf_pool_allocation_t {
  // Owning slab, stable across directory changes while this allocation is live.
  iree_hal_tlsf_pool_slab_t* slab;

  // Allocation returned by the owning TLSF instance.
  iree_hal_memory_tlsf_allocation_t allocation;
} iree_hal_tlsf_pool_allocation_t;

// One pending candidate remembered while searching for immediately usable
// bytes. Selection and consumption both happen within the same mutation
// critical section.
typedef struct iree_hal_tlsf_pool_pending_candidate_t {
  // Slab owning the candidate, or NULL if none has been found.
  iree_hal_tlsf_pool_slab_t* slab;
  // Exact raw candidate; its block and history remain stable under the mutex.
  iree_hal_memory_tlsf_candidate_t candidate;
} iree_hal_tlsf_pool_pending_candidate_t;

// Per-transaction metadata preparation, performed outside the mutation mutex.
typedef struct iree_hal_tlsf_pool_metadata_t {
  // Caller-owned storage available for a retried transaction.
  struct {
    // Detached split metadata, consumed only by a compatible slab.
    iree_hal_memory_tlsf_growth_t blocks;
    // Release records allocated outside the mutex and not yet consumed.
    iree_hal_tlsf_pool_release_node_t* release_nodes;
  } prepared;
  // Requirements captured while inspecting eligible capacity under the mutex.
  struct {
    // Slab whose metadata requirements were inspected under the mutex.
    iree_hal_tlsf_pool_slab_t* slab;
    // Split metadata geometry, or an empty descriptor when capacity suffices.
    iree_hal_memory_tlsf_growth_t blocks;
    // Whether eligible capacity needs another stable release record.
    bool release_node;
  } required;
} iree_hal_tlsf_pool_metadata_t;

static bool iree_hal_tlsf_pool_metadata_is_required(
    const iree_hal_tlsf_pool_metadata_t* metadata) {
  return metadata->required.blocks.first_block ||
         metadata->required.release_node;
}

typedef struct iree_hal_tlsf_pool_t {
  // Base pool resource for vtable dispatch and ref counting.
  iree_hal_pool_t base;

  // Retained ordinary source of reservations and prepared backing views.
  iree_hal_pool_t* backing_pool;

  // Retained finite backing; empty for a growing pool.
  iree_hal_pool_buffer_range_t source_range;

  // Guards TLSF mutation and the owned slab inventory.
  iree_slim_mutex_t mutex;

  // Template options used when initializing each new TLSF slab.
  iree_hal_memory_tlsf_options_t slab_options;

  // Minimum backing byte length requested for newly grown slabs.
  iree_device_size_t backing_slab_length;

  // Maximum user-visible reservation length, or zero for an unbounded source.
  iree_device_size_t max_reservation_size;

  // Slabs published in allocation order. Protected by |mutex|.
  struct {
    // First owned slab, or NULL when empty.
    iree_hal_tlsf_pool_slab_t* head;
    // Last owned slab, used for allocation-free publication.
    iree_hal_tlsf_pool_slab_t* tail;
    // Number of owned slabs.
    uint32_t count;
  } slabs;

  // Slabs changed by growth or returned blocks. Protected by |mutex|.
  iree_hal_tlsf_pool_slab_t* return_candidates;

  // Coalesced cold work. Its mutex guards only scheduling and final joining.
  struct {
    // Protects requested/pending and the callback's final notification access.
    iree_slim_mutex_t mutex;
    // Joins this pool's work during caller-quiescent destruction.
    iree_notification_t notification;
    // Reusable entry on the backing pool's captured maintenance owner.
    iree_hal_memory_maintenance_entry_t entry;
    // Whether a producer requested another pass over changed metadata.
    bool requested;
    // Whether the entry is queued or executing.
    bool pending;
  } maintenance;

  // Preferred slab to try first for the next allocation. Protected by |mutex|.
  iree_hal_tlsf_pool_slab_t* preferred_slab;

  // Bounded set of slabs that recently received releases. Protected by |mutex|.
  struct {
    // Direct pointers into the owned inventory, cleared when trimming it.
    iree_hal_tlsf_pool_slab_t*
        entries[IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY];
    // Number of valid entries.
    uint8_t count;
    // Next entry to overwrite when all entries are occupied.
    uint8_t cursor;
  } reuse_candidates;

  // Approximate committed bytes across all slabs for lock-free stats queries.
  iree_atomic_int64_t bytes_committed;

  // Approximate committed slab count for lock-free stats queries.
  iree_atomic_int32_t committed_slab_count;

  // Pending release nodes pushed by queue-retirement paths.
  iree_atomic_intptr_t pending_release_head;

  // Dedicated backing returned by callers. Protected by |mutex|.
  struct {
    // Returns ready for cold parent release.
    iree_hal_tlsf_pool_release_node_t* returns;
    // Unrepresentable history retained until caller-quiescent destruction.
    iree_hal_tlsf_pool_release_node_t* tainted;
  } dedicated;

  // Free list of release nodes ready for reuse. Protected by |mutex|.
  iree_hal_tlsf_pool_release_node_t* release_node_free_head;

  // Returned ranges withheld by ASAN. Mutation is protected by |mutex|;
  // counters support lock-free observation independently of list mutation.
  struct {
    // Oldest retained range.
    iree_hal_tlsf_pool_release_node_t* head;
    // Newest retained range.
    iree_hal_tlsf_pool_release_node_t* tail;
    // Backing bytes currently withheld from reuse.
    iree_atomic_int64_t size;
    // Ranges removed by pressure or explicit trim.
    iree_atomic_int64_t eviction_count;
  } quarantine;

  // Host allocator used for pool metadata.
  iree_allocator_t host_allocator;

  // Stable named-memory stream for logical reservations from this pool.
  iree_hal_memory_trace_t trace;

  // Immutable source capabilities captured during construction.
  iree_hal_pool_capabilities_t capabilities;

  // ASAN policy used to shape hidden backing ranges.
  iree_hal_asan_pool_options_t asan_options;

  // Logical byte budget for live reservations. 0 means unlimited.
  iree_device_size_t budget_limit;

  // Stable record geometry shared by suballocations and dedicated ranges.
  iree_hal_tlsf_pool_reservation_layout_t reservation_layout;

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

// Staged result for one reservation acquisition. Transactions use staging so
// public output arrays remain untouched unless the operation succeeds.
typedef struct iree_hal_tlsf_pool_acquire_element_t {
  // Captured request geometry, immutable across transaction retries.
  iree_hal_tlsf_pool_request_geometry_t geometry;

  // Private dedicated reservation retained across cold transaction retries.
  struct {
    // Stable record owning the prepared parent token and view.
    iree_hal_tlsf_pool_release_node_t* node;
    // Original source eligibility, including any inherited prerequisite.
    iree_hal_pool_acquire_result_t result;
  } dedicated;

  // Backing preserved across a cold retry, without withholding free blocks.
  iree_hal_tlsf_pool_slab_t* preparing_slab;

  // Reservation produced for the request.
  iree_hal_pool_reservation_t reservation;

  // Acquisition metadata produced for the request.
  iree_hal_pool_acquire_info_t info;
} iree_hal_tlsf_pool_acquire_element_t;

static const iree_hal_pool_vtable_t iree_hal_tlsf_pool_vtable;
static void iree_hal_tlsf_pool_destroy(iree_hal_pool_t* base_pool);

static const char* IREE_HAL_TLSF_POOL_TRACE_ID = "iree-hal-tlsf-pool";

static bool iree_hal_tlsf_pool_query_completed_epoch(void* user_data,
                                                     iree_async_axis_t axis,
                                                     uint64_t epoch) {
  return iree_async_frontier_tracker_query_epoch(user_data, axis, epoch);
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

static void iree_hal_tlsf_pool_schedule_maintenance(
    iree_hal_tlsf_pool_t* pool) {
  if (!pool->backing_pool) {
    return;
  }
  iree_slim_mutex_lock(&pool->maintenance.mutex);
  pool->maintenance.requested = true;
  const bool enqueue = !pool->maintenance.pending;
  pool->maintenance.pending = true;
  iree_slim_mutex_unlock(&pool->maintenance.mutex);
  if (enqueue) {
    iree_hal_memory_maintenance_enqueue(pool->base.maintenance,
                                        &pool->maintenance.entry);
  }
}

static bool iree_hal_tlsf_pool_maintenance_is_idle(void* user_data) {
  iree_hal_tlsf_pool_t* pool = user_data;
  iree_slim_mutex_lock(&pool->maintenance.mutex);
  const bool idle = !pool->maintenance.pending;
  iree_slim_mutex_unlock(&pool->maintenance.mutex);
  return idle;
}

static void iree_hal_tlsf_pool_note_return_candidate(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_slab_t* slab)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  if (pool->backing_pool && !slab->return_candidate) {
    slab->return_candidate = true;
    slab->candidate_next = pool->return_candidates;
    pool->return_candidates = slab;
  }
}

static iree_hal_tlsf_pool_slab_t* iree_hal_tlsf_pool_slab_from_release_node(
    const iree_hal_tlsf_pool_release_node_t* node) {
  return (
      iree_hal_tlsf_pool_slab_t*)((uint8_t*)node->range -
                                  offsetof(iree_hal_tlsf_pool_slab_t, range));
}

static inline iree_async_frontier_t* iree_hal_tlsf_pool_release_node_frontier(
    const iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node) {
  return (iree_async_frontier_t*)((uint8_t*)node +
                                  pool->reservation_layout.frontier.offset);
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

static void iree_hal_tlsf_pool_note_reuse_candidate(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_slab_t* slab)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  for (uint8_t i = 0; i < pool->reuse_candidates.count; ++i) {
    if (pool->reuse_candidates.entries[i] == slab) {
      return;
    }
  }
  if (pool->reuse_candidates.count <
      IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY) {
    pool->reuse_candidates.entries[pool->reuse_candidates.count++] = slab;
    return;
  }
  pool->reuse_candidates.entries[pool->reuse_candidates.cursor] = slab;
  pool->reuse_candidates.cursor =
      (uint8_t)((pool->reuse_candidates.cursor + 1) %
                IREE_HAL_TLSF_POOL_REUSE_CANDIDATE_CAPACITY);
}

static iree_hal_tlsf_pool_release_node_t*
iree_hal_tlsf_pool_acquire_release_node(iree_hal_tlsf_pool_t* pool,
                                        iree_hal_tlsf_pool_metadata_t* metadata)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* node = pool->release_node_free_head;
  if (node) {
    pool->release_node_free_head = node->next;
  } else {
    node = metadata->prepared.release_nodes;
    metadata->prepared.release_nodes = node->next;
  }
  memset(node, 0, sizeof(*node));
  iree_async_frontier_initialize(
      iree_hal_tlsf_pool_release_node_frontier(pool, node), 0);
  return node;
}

static void iree_hal_tlsf_pool_recycle_release_node(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  if (!node) {
    return;
  }
  node->next = pool->release_node_free_head;
  pool->release_node_free_head = node;
}

static void iree_hal_tlsf_pool_free_release_node_list(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node) {
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    iree_allocator_free(pool->host_allocator, node);
    node = next;
  }
}

static iree_hal_tlsf_pool_release_node_t*
iree_hal_tlsf_pool_take_free_release_nodes(iree_hal_tlsf_pool_t* pool)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* nodes = pool->release_node_free_head;
  pool->release_node_free_head = NULL;
  return nodes;
}

static void iree_hal_tlsf_pool_return_release_node_to_tlsf(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_async_frontier_t* death_frontier =
      iree_hal_tlsf_pool_release_node_frontier(pool, node);
  if (node->block_index == IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    iree_hal_tlsf_pool_release_node_t** list =
        iree_hal_tlsf_pool_dedicated_merge_return_frontier(
            node, &pool->reservation_layout)
            ? &pool->dedicated.returns
            : &pool->dedicated.tainted;
    node->next = *list;
    *list = node;
    return;
  }
  iree_hal_tlsf_pool_slab_t* slab =
      iree_hal_tlsf_pool_slab_from_release_node(node);
  iree_hal_memory_tlsf_free(
      &slab->tlsf, node->block_index,
      death_frontier->entry_count > 0 ? death_frontier : NULL);
  iree_hal_tlsf_pool_note_return_candidate(pool, slab);
  if (death_frontier->entry_count == 0) {
    pool->preferred_slab = slab;
  } else {
    iree_hal_tlsf_pool_note_reuse_candidate(pool, slab);
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
  const iree_device_size_t size = (iree_device_size_t)iree_atomic_load(
      &pool->quarantine.size, iree_memory_order_relaxed);
  iree_atomic_store(&pool->quarantine.size,
                    length > IREE_DEVICE_SIZE_MAX - size ? IREE_DEVICE_SIZE_MAX
                                                         : size + length,
                    iree_memory_order_relaxed);
}

static void iree_hal_tlsf_pool_record_quarantine_eviction(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t length)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  const iree_device_size_t size = (iree_device_size_t)iree_atomic_load(
      &pool->quarantine.size, iree_memory_order_relaxed);
  iree_atomic_store(&pool->quarantine.size, size - iree_min(size, length),
                    iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->quarantine.eviction_count, 1,
                        iree_memory_order_relaxed);
}

static void iree_hal_tlsf_pool_quarantine_release_node(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  node->next = NULL;
  if (pool->quarantine.tail) {
    pool->quarantine.tail->next = node;
  } else {
    pool->quarantine.head = node;
  }
  pool->quarantine.tail = node;
  iree_hal_tlsf_pool_add_quarantine_size(pool, node->charged_length);

  while ((iree_device_size_t)iree_atomic_load(&pool->quarantine.size,
                                              iree_memory_order_relaxed) >
             pool->asan_options.quarantine_size &&
         pool->quarantine.head) {
    iree_hal_tlsf_pool_release_node_t* release_node = pool->quarantine.head;
    pool->quarantine.head = release_node->next;
    if (!pool->quarantine.head) {
      pool->quarantine.tail = NULL;
    }
    iree_hal_tlsf_pool_record_quarantine_eviction(pool,
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
  iree_hal_tlsf_pool_release_node_t* node = pool->quarantine.head;
  pool->quarantine.head = NULL;
  pool->quarantine.tail = NULL;
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    iree_hal_tlsf_pool_record_quarantine_eviction(pool, node->charged_length);
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
  if (!pool->base.epoch_query.fn) {
    return false;
  }

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
    if (!pool->base.epoch_query.fn(pool->base.epoch_query.user_data, axis,
                                   epoch)) {
      return false;
    }
  }
  return true;
}

// Acquires and prepares an ordinary backing range outside the mutation mutex.
// Setup rollback returns the original prerequisite even when the requester
// already covers it: requester eligibility is not global completion.
static iree_status_t iree_hal_tlsf_pool_prepare_slab(
    iree_hal_tlsf_pool_t* pool, const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags, iree_hal_tlsf_pool_slab_t** out_slab,
    iree_hal_pool_acquire_result_t* out_result) {
  *out_slab = NULL;
  *out_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  iree_hal_tlsf_pool_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                             sizeof(*slab), (void**)&slab));
  memset(slab, 0, sizeof(*slab));
  iree_status_t status = iree_ok_status();
  iree_hal_pool_acquire_info_t info = {0};
  if (pool->backing_pool) {
    const iree_hal_pool_reservation_request_t request = {
        .params =
            {
                .type = pool->capabilities.memory_type,
                .access = pool->capabilities.allowed_access,
                .usage = pool->capabilities.supported_usage,
                .queue_family_affinity =
                    pool->capabilities.queue_family_affinity,
                .min_alignment =
                    iree_min(pool->slab_options.alignment,
                             pool->capabilities.max_allocation_alignment),
            },
        .allocation_size = pool->backing_slab_length,
    };
    status = iree_hal_pool_acquire_reservations(
        pool->backing_pool, 1, &request, requester_frontier, flags,
        &slab->reservation, &info, out_result);
    if (iree_status_is_ok(status) &&
        (*out_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
         *out_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET)) {
      iree_allocator_free(pool->host_allocator, slab);
      return iree_ok_status();
    }
    iree_hal_buffer_t* buffer = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_pool_materialize_reservations(
          pool->backing_pool, 1, &request, &slab->reservation,
          IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_pool_buffer_range_initialize(
          buffer, 0, IREE_HAL_WHOLE_BUFFER, request.params.min_alignment,
          &pool->asan_options, &slab->range);
      slab->range.memory.reuse_frontier = info.reuse_frontier;
    }
    iree_hal_buffer_release(buffer);
  } else {
    slab->range = pool->source_range;
  }
  if (iree_status_is_ok(status)) {
    iree_hal_memory_tlsf_options_t options = pool->slab_options;
    options.range_length = slab->range.length;
    options.initial_frontier = slab->range.memory.reuse_frontier;
    status = iree_hal_memory_tlsf_initialize(options, pool->host_allocator,
                                             &slab->tlsf);
  }
  if (iree_status_is_ok(status)) {
    slab->release_frontier = info.reuse_frontier;
    *out_slab = slab;
  } else {
    if (pool->backing_pool && info.result != IREE_HAL_POOL_ACQUIRE_NONE) {
      iree_hal_pool_release_reservations(
          pool->backing_pool, 1, &slab->reservation, info.reuse_frontier);
      iree_hal_pool_buffer_range_deinitialize(&slab->range);
    }
    iree_allocator_free(pool->host_allocator, slab);
  }
  return status;
}

// Publishes prepared backing with no allocation or native work under the mutex.
static void iree_hal_tlsf_pool_publish_slab(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_slab_t* slab_entry)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  slab_entry->previous = pool->slabs.tail;
  if (pool->slabs.tail) {
    pool->slabs.tail->next = slab_entry;
  } else {
    pool->slabs.head = slab_entry;
  }
  pool->slabs.tail = slab_entry;
  ++pool->slabs.count;
  iree_hal_tlsf_pool_note_return_candidate(pool, slab_entry);
  pool->preferred_slab = slab_entry;
  iree_atomic_fetch_add(&pool->committed_slab_count, 1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->bytes_committed,
                        (int64_t)slab_entry->range.length,
                        iree_memory_order_relaxed);
}

static void iree_hal_tlsf_pool_destroy_slab(iree_hal_tlsf_pool_t* pool,
                                            iree_hal_tlsf_pool_slab_t* slab) {
  if (pool->backing_pool) {
    iree_hal_pool_release_reservations(
        pool->backing_pool, 1, &slab->reservation, slab->release_frontier);
    iree_hal_pool_buffer_range_deinitialize(&slab->range);
  }
  iree_hal_memory_tlsf_deinitialize(&slab->tlsf);
  iree_allocator_free(pool->host_allocator, slab);
}

// Returns the history of the entire parent reservation, including bytes that
// endpoint alignment excluded from the child's managed interval.
static bool iree_hal_tlsf_pool_query_full_free_slab(
    iree_hal_tlsf_pool_slab_t* slab, const iree_async_frontier_t** out_frontier,
    iree_hal_memory_tlsf_block_flags_t* out_flags) {
  if (!iree_hal_memory_tlsf_query_full_free_block(&slab->tlsf, out_frontier,
                                                  out_flags)) {
    return false;
  }
  if (slab->reservation.byte_length &&
      (slab->range.offset ||
       slab->range.length != slab->reservation.byte_length)) {
    iree_hal_memory_tlsf_merge_full_free_frontier(
        &slab->tlsf, slab->range.memory.reuse_frontier);
    iree_hal_memory_tlsf_query_full_free_block(&slab->tlsf, out_frontier,
                                               out_flags);
  }
  return true;
}

// Removes only ownership metadata. Parent calls and metadata destruction run
// after dropping the mutation mutex on the captured maintenance owner.
static void iree_hal_tlsf_pool_detach_slab(iree_hal_tlsf_pool_t* pool,
                                           iree_hal_tlsf_pool_slab_t* slab)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  if (slab->previous) {
    slab->previous->next = slab->next;
  } else {
    pool->slabs.head = slab->next;
  }
  if (slab->next) {
    slab->next->previous = slab->previous;
  } else {
    pool->slabs.tail = slab->previous;
  }
  --pool->slabs.count;
  if (pool->preferred_slab == slab) {
    pool->preferred_slab = pool->slabs.head;
  }
  for (uint8_t i = 0; i < pool->reuse_candidates.count; ++i) {
    if (pool->reuse_candidates.entries[i] == slab) {
      pool->reuse_candidates.entries[i] =
          pool->reuse_candidates.entries[--pool->reuse_candidates.count];
      break;
    }
  }
  iree_atomic_fetch_sub(&pool->committed_slab_count, 1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_sub(&pool->bytes_committed, (int64_t)slab->range.length,
                        iree_memory_order_relaxed);
}

// Consumes changed slabs without walking unrelated live backing. A tainted
// frontier remains owned until caller-quiescent destruction can retire it.
static iree_hal_tlsf_pool_slab_t* iree_hal_tlsf_pool_take_unused_slabs(
    iree_hal_tlsf_pool_t* pool, iree_device_size_t min_bytes_to_keep)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_slab_t* retired = NULL;
  iree_hal_tlsf_pool_slab_t* slab = pool->return_candidates;
  pool->return_candidates = NULL;
  while (slab) {
    iree_hal_tlsf_pool_slab_t* next = slab->candidate_next;
    slab->candidate_next = NULL;
    slab->return_candidate = false;
    iree_hal_memory_tlsf_block_flags_t flags;
    const iree_device_size_t committed = (iree_device_size_t)iree_atomic_load(
        &pool->bytes_committed, iree_memory_order_relaxed);
    if (!slab->preparation_count &&
        iree_hal_tlsf_pool_query_full_free_slab(slab, &slab->release_frontier,
                                                &flags) &&
        !iree_any_bit_set(flags, IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED)) {
      if (committed >= min_bytes_to_keep &&
          slab->range.length <= committed - min_bytes_to_keep) {
        iree_hal_tlsf_pool_detach_slab(pool, slab);
        slab->next = retired;
        retired = slab;
      } else {
        // A one-shot trim floor does not disable ordinary idle return.
        iree_hal_tlsf_pool_note_return_candidate(pool, slab);
      }
    }
    slab = next;
  }
  return retired;
}

static void iree_hal_tlsf_pool_destroy_slab_list(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_slab_t* slab) {
  while (slab) {
    iree_hal_tlsf_pool_slab_t* next = slab->next;
    iree_hal_tlsf_pool_destroy_slab(pool, slab);
    slab = next;
  }
}

static iree_hal_tlsf_pool_release_node_t*
iree_hal_tlsf_pool_take_dedicated_returns(iree_hal_tlsf_pool_t* pool)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* nodes = pool->dedicated.returns;
  pool->dedicated.returns = NULL;
  return nodes;
}

static void iree_hal_tlsf_pool_release_dedicated_list(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node) {
  while (node) {
    iree_hal_tlsf_pool_release_node_t* next = node->next;
    const iree_device_size_t length =
        iree_hal_tlsf_pool_dedicated_backing_length(node,
                                                    &pool->reservation_layout);
    iree_hal_tlsf_pool_dedicated_release(pool->backing_pool, node,
                                         &pool->reservation_layout,
                                         pool->host_allocator);
    iree_atomic_fetch_sub(&pool->committed_slab_count, 1,
                          iree_memory_order_relaxed);
    iree_atomic_fetch_sub(&pool->bytes_committed, (int64_t)length,
                          iree_memory_order_relaxed);
    node = next;
  }
}

static void iree_hal_tlsf_pool_maintain(
    iree_hal_memory_maintenance_entry_t* entry) {
  iree_hal_tlsf_pool_t* pool =
      (iree_hal_tlsf_pool_t*)((uint8_t*)entry - offsetof(iree_hal_tlsf_pool_t,
                                                         maintenance.entry));
  for (;;) {
    iree_slim_mutex_lock(&pool->maintenance.mutex);
    if (!pool->maintenance.requested) {
      pool->maintenance.pending = false;
      iree_notification_post(&pool->maintenance.notification, IREE_ALL_WAITERS);
      iree_slim_mutex_unlock(&pool->maintenance.mutex);
      return;
    }
    pool->maintenance.requested = false;
    iree_slim_mutex_unlock(&pool->maintenance.mutex);

    iree_slim_mutex_lock(&pool->mutex);
    iree_hal_tlsf_pool_drain_pending_releases(pool);
    iree_hal_tlsf_pool_slab_t* retired =
        iree_hal_tlsf_pool_take_unused_slabs(pool, 0);
    iree_hal_tlsf_pool_release_node_t* dedicated =
        iree_hal_tlsf_pool_take_dedicated_returns(pool);
    iree_slim_mutex_unlock(&pool->mutex);
    iree_hal_tlsf_pool_destroy_slab_list(pool, retired);
    iree_hal_tlsf_pool_release_dedicated_list(pool, dedicated);
  }
}

static void iree_hal_tlsf_pool_deinitialize_slabs(iree_hal_tlsf_pool_t* pool) {
  iree_hal_tlsf_pool_slab_t* slab = pool->slabs.head;
  while (slab) {
    iree_hal_tlsf_pool_slab_t* next = slab->next;
    iree_hal_memory_tlsf_block_flags_t flags;
    const bool is_free = iree_hal_tlsf_pool_query_full_free_slab(
        slab, &slab->release_frontier, &flags);
    IREE_ASSERT(is_free, "pool destruction requires returned reservations");
    if (iree_any_bit_set(flags, IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED)) {
      // Destruction requires caller quiescence for unrepresentable history.
      // Ordinary trim cannot discard this prerequisite.
      slab->release_frontier = NULL;
    }
    iree_hal_tlsf_pool_destroy_slab(pool, slab);
    slab = next;
  }
  memset(&pool->slabs, 0, sizeof(pool->slabs));
  pool->return_candidates = NULL;
  pool->preferred_slab = NULL;
  pool->reuse_candidates.count = 0;
  pool->reuse_candidates.cursor = 0;
  iree_atomic_store(&pool->committed_slab_count, 0, iree_memory_order_release);
  iree_atomic_store(&pool->bytes_committed, 0, iree_memory_order_release);
}

static void iree_hal_tlsf_pool_return_reservation(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_release_node_t* node,
    iree_device_size_t byte_length, iree_hal_pool_acquire_result_t result,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  *out_reservation = (iree_hal_pool_reservation_t){
      .offset = node->backing_offset + node->asan_layout.user_offset,
      .byte_length = byte_length,
      .block_handle = (uint64_t)(uintptr_t)node,
  };
  iree_async_frontier_t* frontier =
      iree_hal_tlsf_pool_release_node_frontier(pool, node);
  *out_info = (iree_hal_pool_acquire_info_t){
      .result = result,
      .reuse_frontier = frontier->entry_count ? frontier : NULL,
  };
  iree_atomic_fetch_add(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);
  switch (result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
      iree_atomic_fetch_add(&pool->reuse_count, 1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_atomic_fetch_add(&pool->fresh_count, 1, iree_memory_order_relaxed);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      iree_atomic_fetch_add(&pool->wait_count, 1, iree_memory_order_relaxed);
      break;
    default:
      IREE_ASSERT(false, "invalid successful TLSF pool result: %u", result);
      break;
  }
  *out_result = result;
}

static void iree_hal_tlsf_pool_return_allocation(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_metadata_t* metadata,
    const iree_hal_tlsf_pool_allocation_t* pool_allocation,
    iree_device_size_t byte_length, iree_device_size_t charged_length,
    const iree_hal_asan_allocation_layout_t* asan_layout,
    iree_hal_pool_acquire_result_t result,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_slab_t* slab = pool_allocation->slab;
  const iree_hal_memory_tlsf_allocation_t* allocation =
      &pool_allocation->allocation;
  iree_hal_tlsf_pool_release_node_t* release_node =
      iree_hal_tlsf_pool_acquire_release_node(pool, metadata);
  release_node->range = &slab->range;
  release_node->block_index = allocation->block_index;
  release_node->charged_length = charged_length;
  release_node->backing_offset = allocation->offset;
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    release_node->asan_layout = *asan_layout;
  }
  pool->preferred_slab = slab;

  if (allocation->death_frontier) {
    // The reservation owns its exact prerequisite alongside its release record
    // until explicitly returned, including while native views borrow it.
    iree_async_frontier_t* reuse_frontier =
        iree_hal_tlsf_pool_release_node_frontier(pool, release_node);
    memcpy(reuse_frontier, allocation->death_frontier,
           sizeof(*reuse_frontier) +
               (iree_host_size_t)allocation->death_frontier->entry_count *
                   sizeof(iree_async_frontier_entry_t));
  }
  iree_hal_tlsf_pool_return_reservation(pool, release_node, byte_length, result,
                                        out_reservation, out_info, out_result);
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_tlsf_pool_calculate_backing_geometry(
    const iree_hal_tlsf_pool_options_t* options,
    iree_device_size_t* out_alignment, iree_device_size_t* out_length) {
  iree_device_size_t alignment = options->tlsf_options.alignment;
  if (!alignment) {
    alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  }
  if (alignment < IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT ||
      !iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "TLSF alignment must be a power of two at least 16");
  }
  iree_device_size_t length = options->tlsf_options.range_length;
  if (iree_hal_asan_pool_options_is_enabled(&options->asan)) {
    alignment =
        iree_max(alignment, iree_max(options->asan.backing_alignment,
                                     options->asan.shadow_granule_size));
    iree_hal_asan_allocation_layout_t layout;
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        &options->asan, length, alignment, &layout));
    length = layout.backing_length;
  }
  if (!iree_device_size_checked_align(length, alignment, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TLSF backing size overflows alignment");
  }
  *out_alignment = alignment;
  *out_length = length;
  return iree_ok_status();
}

static iree_status_t iree_hal_tlsf_pool_resolve_backing_request(
    const iree_hal_pool_capabilities_t* capabilities,
    const iree_hal_tlsf_pool_options_t* options,
    iree_hal_pool_reservation_request_t* out_request) {
  iree_device_size_t alignment = 0;
  iree_device_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_tlsf_pool_calculate_backing_geometry(
      options, &alignment, &length));
  if (!options->tlsf_options.range_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TLSF range_length must be > 0");
  }
  if (alignment > capabilities->max_allocation_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TLSF alignment exceeds backing pool support");
  }
  // Native address alignment and storage-relative maintenance endpoints are
  // independent. Reserve complete granules without asking the parent to
  // strengthen its native address guarantee.
  const iree_device_size_t granule =
      iree_max(alignment, capabilities->maintenance_alignment);
  if (!iree_device_size_checked_align(length, granule, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TLSF backing size overflows maintenance granule");
  }
  alignment = iree_min(granule, capabilities->max_allocation_alignment);
  if (capabilities->max_allocation_size &&
      length > capabilities->max_allocation_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TLSF slab size exceeds backing pool support");
  }
  *out_request = (iree_hal_pool_reservation_request_t){
      .params =
          {
              .type = capabilities->memory_type,
              .access = capabilities->allowed_access,
              .usage = capabilities->supported_usage,
              .queue_family_affinity = capabilities->queue_family_affinity,
              .min_alignment = alignment,
          },
      .allocation_size = length,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_tlsf_pool_query_backing_request(
    iree_hal_pool_t* backing_pool, const iree_hal_tlsf_pool_options_t* options,
    iree_hal_pool_reservation_request_t* out_request) {
  IREE_RETURN_IF_ERROR(
      iree_hal_pool_validate_asan_options(backing_pool, &options->asan));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(backing_pool, &capabilities);
  return iree_hal_tlsf_pool_resolve_backing_request(&capabilities, options,
                                                    out_request);
}

static iree_status_t iree_hal_tlsf_pool_create_impl(
    iree_hal_tlsf_pool_options_t options,
    iree_hal_pool_buffer_range_t source_range, iree_hal_pool_t* backing_pool,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_hal_pool_epoch_query_t epoch_query, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (options.tlsf_options.range_length == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range_length must be > 0");
  }
  if (backing_pool) {
    if (!backing_pool->maintenance) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "TLSF backing pool has no memory maintenance owner");
    }
    if (options.tlsf_options.initial_frontier) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "backing reservations supply their own reuse frontier");
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_pool_validate_asan_options(backing_pool, &options.asan));
  }

  iree_hal_memory_tlsf_options_t tlsf_options = options.tlsf_options;
  iree_hal_pool_capabilities_t capabilities = {0};
  iree_device_size_t backing_slab_length = 0;
  if (backing_pool) {
    iree_hal_pool_query_capabilities(backing_pool, &capabilities);
    iree_hal_pool_reservation_request_t request;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_tlsf_pool_resolve_backing_request(&capabilities, &options,
                                                       &request));
    tlsf_options.alignment = iree_max(request.params.min_alignment,
                                      capabilities.maintenance_alignment);
    backing_slab_length = request.allocation_size;
  } else {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_tlsf_pool_calculate_backing_geometry(
                &options, &tlsf_options.alignment, &backing_slab_length));
    backing_slab_length = source_range.length;
    tlsf_options.range_length = source_range.length;
    capabilities = (iree_hal_pool_capabilities_t){
        .memory_type = iree_hal_buffer_memory_type(source_range.buffer),
        .allowed_access = iree_hal_buffer_allowed_access(source_range.buffer),
        .supported_usage = iree_hal_buffer_allowed_usage(source_range.buffer),
        .queue_family_affinity =
            iree_hal_buffer_allocation_placement(source_range.buffer)
                .queue_family_affinity,
        .atomic_operations = source_range.memory.backing->atomic_operations,
        .max_allocation_size = source_range.length,
        .max_allocation_alignment = tlsf_options.alignment,
        .maintenance_alignment =
            source_range.memory.backing->maintenance_alignment,
    };
    tlsf_options.alignment =
        iree_max(tlsf_options.alignment, capabilities.maintenance_alignment);
  }
  if (!tlsf_options.frontier_capacity) {
    tlsf_options.frontier_capacity =
        IREE_HAL_MEMORY_TLSF_DEFAULT_FRONTIER_CAPACITY;
  }

  iree_device_size_t max_reservation_size = capabilities.max_allocation_size;
  if (max_reservation_size) {
    max_reservation_size &= ~(tlsf_options.alignment - 1);
    if (iree_hal_asan_pool_options_is_enabled(&options.asan)) {
      iree_hal_asan_allocation_layout_t minimum_layout;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_asan_calculate_allocation_layout(&options.asan, 1, 1,
                                                        &minimum_layout));
      max_reservation_size -=
          minimum_layout.user_offset + options.asan.redzone_size;
    }
  }

  iree_hal_tlsf_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  iree_async_notification_t* notification = NULL;
  iree_status_t status = iree_async_notification_create(
      proactor, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification);
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_initialize(
        &iree_hal_tlsf_pool_vtable,
        backing_pool ? backing_pool->memory_contract
                     : source_range.memory.contract,
        notification,
        backing_pool ? backing_pool->wait_sources
                     : (iree_hal_pool_wait_source_list_t){0},
        frontier_tracker, host_allocator, &pool->base);
  }
  iree_async_notification_release(notification);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, pool);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  iree_slim_mutex_initialize(&pool->mutex);
  iree_slim_mutex_initialize(&pool->maintenance.mutex);
  iree_notification_initialize(&pool->maintenance.notification);
  pool->maintenance.entry.fn = iree_hal_tlsf_pool_maintain;
  iree_atomic_store(&pool->pending_release_head, 0, iree_memory_order_relaxed);
  pool->host_allocator = host_allocator;
  pool->source_range = source_range;
  iree_hal_buffer_retain(source_range.buffer);
  pool->base.epoch_query = epoch_query;
  pool->base.maintenance = backing_pool
                               ? backing_pool->maintenance
                               : source_range.memory.backing->maintenance;
  pool->budget_limit = options.budget_limit;
  pool->slab_options = tlsf_options;
  pool->reservation_layout =
      iree_hal_tlsf_pool_reservation_layout(tlsf_options.frontier_capacity);
  pool->backing_slab_length = backing_slab_length;
  pool->max_reservation_size = max_reservation_size;
  pool->asan_options = options.asan;
  pool->base.asan_enabled =
      iree_hal_asan_pool_options_is_enabled(&options.asan);
  iree_atomic_store(&pool->bytes_committed, 0, iree_memory_order_relaxed);
  iree_atomic_store(&pool->committed_slab_count, 0, iree_memory_order_relaxed);

  iree_hal_pool_retain(backing_pool);
  pool->backing_pool = backing_pool;
  pool->capabilities = capabilities;

  status = iree_hal_memory_trace_initialize_pool(options.trace_name,
                                                 IREE_HAL_TLSF_POOL_TRACE_ID,
                                                 host_allocator, &pool->trace);
  iree_hal_tlsf_pool_slab_t* initial_slab = NULL;
  if (iree_status_is_ok(status) && source_range.buffer) {
    iree_hal_pool_acquire_result_t result;
    status = iree_hal_tlsf_pool_prepare_slab(
        pool, NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE, &initial_slab, &result);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&pool->mutex);
      iree_hal_tlsf_pool_publish_slab(pool, initial_slab);
      iree_slim_mutex_unlock(&pool->mutex);
    }
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

IREE_API_EXPORT iree_status_t iree_hal_tlsf_pool_create(
    iree_hal_pool_t* backing_pool, const iree_hal_tlsf_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(backing_pool);
  *out_pool = NULL;
  return iree_hal_tlsf_pool_create_impl(
      *options, (iree_hal_pool_buffer_range_t){0}, backing_pool,
      backing_pool->notification->proactor, backing_pool->frontier_tracker,
      backing_pool->epoch_query, host_allocator, out_pool);
}

IREE_API_EXPORT iree_status_t iree_hal_tlsf_pool_create_from_buffer(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, const iree_hal_tlsf_pool_options_t* options,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  *out_pool = NULL;
  if (options->tlsf_options.range_length != 0 ||
      options->tlsf_options.initial_frontier) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "finite TLSF geometry and history come from its buffer");
  }
  iree_hal_tlsf_pool_options_t resolved = *options;
  iree_device_size_t alignment = resolved.tlsf_options.alignment;
  if (!alignment) {
    alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  }
  if (alignment < IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT ||
      !iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "TLSF alignment must be a power of two at least 16");
  }
  iree_hal_asan_allocation_layout_t minimum_layout = {0};
  if (iree_hal_asan_pool_options_is_enabled(&resolved.asan)) {
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        &resolved.asan, 1, 1, &minimum_layout));
    alignment =
        iree_max(alignment, iree_max(resolved.asan.backing_alignment,
                                     resolved.asan.shadow_granule_size));
  }
  iree_hal_pool_buffer_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_pool_buffer_range_initialize(
      buffer, offset, length, alignment, &resolved.asan, &range));
  resolved.tlsf_options.alignment = alignment;
  resolved.tlsf_options.range_length = range.length;
  resolved.tlsf_options.initial_frontier = range.memory.reuse_frontier;
  iree_hal_pool_epoch_query_t epoch_query = {
      .fn = iree_hal_tlsf_pool_query_completed_epoch,
      .user_data = range.memory.backing->tracker,
  };
  iree_status_t status = iree_ok_status();
  if (minimum_layout.backing_length > range.length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "buffer range cannot hold a guarded allocation");
  } else {
    status = iree_hal_tlsf_pool_create_impl(
        resolved, range, NULL, range.memory.backing->notification->proactor,
        range.memory.backing->tracker, epoch_query, host_allocator, out_pool);
  }
  iree_hal_pool_buffer_range_deinitialize(&range);
  return status;
}

static void iree_hal_tlsf_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;

  iree_notification_await(&pool->maintenance.notification,
                          iree_hal_tlsf_pool_maintenance_is_idle, pool,
                          iree_infinite_timeout());
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_tlsf_pool_drain_pending_releases(pool);
  iree_hal_tlsf_pool_flush_quarantine(pool);
  while (pool->dedicated.tainted) {
    iree_hal_tlsf_pool_release_node_t* node = pool->dedicated.tainted;
    pool->dedicated.tainted = node->next;
    // Destruction requires caller quiescence for unrepresentable history.
    iree_async_frontier_initialize(
        iree_hal_tlsf_pool_release_node_frontier(pool, node), 0);
    node->next = pool->dedicated.returns;
    pool->dedicated.returns = node;
  }
  iree_hal_tlsf_pool_release_node_t* dedicated =
      iree_hal_tlsf_pool_take_dedicated_returns(pool);
  iree_hal_tlsf_pool_release_node_t* release_nodes =
      iree_hal_tlsf_pool_take_free_release_nodes(pool);
  iree_slim_mutex_unlock(&pool->mutex);
  iree_hal_tlsf_pool_free_release_node_list(pool, release_nodes);
  iree_hal_tlsf_pool_release_dedicated_list(pool, dedicated);

  iree_hal_tlsf_pool_deinitialize_slabs(pool);
  iree_notification_deinitialize(&pool->maintenance.notification);
  iree_slim_mutex_deinitialize(&pool->maintenance.mutex);
  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_hal_memory_trace_deinitialize(&pool->trace);
  iree_hal_pool_deinitialize(base_pool);
  iree_hal_pool_release(pool->backing_pool);
  iree_hal_pool_buffer_range_deinitialize(&pool->source_range);
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_allocator_free(host_allocator, pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static void iree_hal_tlsf_pool_acquire_candidate_locked(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_slab_t* slab,
    iree_hal_memory_tlsf_candidate_t candidate,
    iree_device_size_t aligned_length, iree_hal_pool_acquire_result_t result,
    iree_hal_tlsf_pool_metadata_t* metadata,
    iree_hal_tlsf_pool_allocation_t* out_allocation,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  const iree_hal_memory_tlsf_growth_t growth =
      iree_hal_memory_tlsf_query_growth(&slab->tlsf, candidate.block_index,
                                        aligned_length);
  if (growth.first_block && (!metadata->prepared.blocks.storage ||
                             !iree_hal_memory_tlsf_apply_growth(
                                 &slab->tlsf, &metadata->prepared.blocks))) {
    metadata->required.blocks = growth;
  }
  if (!pool->release_node_free_head && !metadata->prepared.release_nodes) {
    metadata->required.release_node = true;
  }
  if (iree_hal_tlsf_pool_metadata_is_required(metadata)) {
    metadata->required.slab = slab;
    return;
  }
  iree_hal_memory_tlsf_allocate_block(&slab->tlsf, candidate.block_index,
                                      aligned_length,
                                      &out_allocation->allocation);
  out_allocation->slab = slab;
  *out_result = result;
}

static void iree_hal_tlsf_pool_try_acquire_from_slab(
    iree_hal_tlsf_pool_t* pool, iree_hal_tlsf_pool_slab_t* slab,
    iree_device_size_t allocation_length,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_tlsf_pool_pending_candidate_t* pending_candidate,
    iree_hal_tlsf_pool_metadata_t* metadata,
    iree_hal_tlsf_pool_allocation_t* out_allocation,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  const iree_device_size_t aligned_length =
      iree_device_align(allocation_length, pool->slab_options.alignment);
  iree_hal_memory_tlsf_block_index_t after_block =
      IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
  *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  for (;;) {
    const iree_hal_memory_tlsf_candidate_t candidate =
        iree_hal_memory_tlsf_query_free_block(&slab->tlsf, aligned_length,
                                              after_block);
    if (candidate.block_index == IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
      break;
    }
    after_block = candidate.block_index;
    if (!iree_hal_tlsf_pool_frontier_is_satisfied(pool, requester_frontier,
                                                  candidate.death_frontier,
                                                  candidate.block_flags)) {
      if (pending_candidate && !pending_candidate->slab &&
          candidate.death_frontier &&
          !iree_any_bit_set(candidate.block_flags,
                            IREE_HAL_MEMORY_TLSF_BLOCK_FLAG_TAINTED)) {
        pending_candidate->slab = slab;
        pending_candidate->candidate = candidate;
      }
      iree_atomic_fetch_add(&pool->reuse_miss_count, 1,
                            iree_memory_order_relaxed);
      continue;
    }
    iree_hal_tlsf_pool_acquire_candidate_locked(
        pool, slab, candidate, aligned_length,
        candidate.death_frontier ? IREE_HAL_POOL_ACQUIRE_OK
                                 : IREE_HAL_POOL_ACQUIRE_OK_FRESH,
        metadata, out_allocation, out_result);
    break;
  }
}

static bool iree_hal_tlsf_pool_request_is_dedicated(
    const iree_hal_tlsf_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request) {
  return pool->backing_pool &&
         (request->allocation_size > pool->slab_options.range_length ||
          request->params.min_alignment > pool->slab_options.alignment);
}

static void iree_hal_tlsf_pool_acquire_one_reservation_locked(
    iree_hal_tlsf_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_tlsf_pool_metadata_t* metadata,
    iree_hal_tlsf_pool_acquire_element_t* element,
    iree_hal_pool_acquire_result_t* out_result)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  const iree_device_size_t size = request->allocation_size;
  iree_hal_pool_reservation_t* out_reservation = &element->reservation;
  iree_hal_pool_acquire_info_t* out_info = &element->info;
  iree_hal_asan_allocation_layout_t asan_layout = element->geometry.asan;
  const iree_device_size_t allocation_length = element->geometry.length;
  iree_device_size_t charged_length =
      element->dedicated.node ? element->dedicated.node->charged_length
                              : allocation_length;
  if (!iree_hal_tlsf_pool_try_charge_reservation(pool, charged_length)) {
    iree_atomic_fetch_add(&pool->over_budget_count, 1,
                          iree_memory_order_relaxed);
    memset(out_reservation, 0, sizeof(*out_reservation));
    memset(out_info, 0, sizeof(*out_info));
    out_info->result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
    return;
  }

  if (iree_hal_tlsf_pool_request_is_dedicated(pool, request)) {
    if (element->dedicated.node) {
      iree_hal_tlsf_pool_return_reservation(
          pool, element->dedicated.node, size, element->dedicated.result,
          out_reservation, out_info, out_result);
    } else {
      iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
      out_info->flags = IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED;
      out_info->result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
      *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    }
    return;
  }

  iree_hal_tlsf_pool_allocation_t selected_allocation;
  iree_hal_pool_acquire_result_t selected_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  bool has_selected_allocation = false;
  bool growth_required = false;
  iree_hal_tlsf_pool_pending_candidate_t pending_storage = {0};
  iree_hal_tlsf_pool_pending_candidate_t* pending_candidate =
      iree_any_bit_set(flags, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER)
          ? &pending_storage
          : NULL;

  iree_hal_tlsf_pool_slab_t* preferred_slab = pool->preferred_slab;
  if (preferred_slab) {
    iree_hal_tlsf_pool_try_acquire_from_slab(
        pool, preferred_slab, allocation_length, requester_frontier,
        pending_candidate, metadata, &selected_allocation, &selected_result);
    has_selected_allocation = selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
                              selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  }

  for (uint8_t i = 0; i < pool->reuse_candidates.count &&
                      !iree_hal_tlsf_pool_metadata_is_required(metadata) &&
                      !has_selected_allocation;
       ++i) {
    iree_hal_tlsf_pool_slab_t* slab = pool->reuse_candidates.entries[i];
    if (slab == preferred_slab) {
      continue;
    }
    iree_hal_tlsf_pool_try_acquire_from_slab(
        pool, slab, allocation_length, requester_frontier, pending_candidate,
        metadata, &selected_allocation, &selected_result);
    has_selected_allocation = selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
                              selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  }

  // The preferred and recent-release slabs are fast candidates, not the full
  // capacity inventory. Search the remainder before requesting more backing.
  for (iree_hal_tlsf_pool_slab_t* slab = pool->slabs.head;
       slab && !iree_hal_tlsf_pool_metadata_is_required(metadata) &&
       !has_selected_allocation;
       slab = slab->next) {
    bool was_searched = slab == preferred_slab;
    for (uint8_t j = 0; j < pool->reuse_candidates.count; ++j) {
      was_searched |= slab == pool->reuse_candidates.entries[j];
    }
    if (was_searched) {
      continue;
    }
    iree_hal_tlsf_pool_try_acquire_from_slab(
        pool, slab, allocation_length, requester_frontier, pending_candidate,
        metadata, &selected_allocation, &selected_result);
    has_selected_allocation = selected_result == IREE_HAL_POOL_ACQUIRE_OK ||
                              selected_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  }

  if (!iree_hal_tlsf_pool_metadata_is_required(metadata) &&
      !has_selected_allocation) {
    if (pending_storage.slab) {
      iree_hal_tlsf_pool_acquire_candidate_locked(
          pool, pending_storage.slab, pending_storage.candidate,
          iree_device_align(allocation_length, pool->slab_options.alignment),
          IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT, metadata, &selected_allocation,
          &selected_result);
      has_selected_allocation =
          selected_result != IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    }
    growth_required = pool->backing_pool && !has_selected_allocation &&
                      !iree_hal_tlsf_pool_metadata_is_required(metadata);
  }

  if (has_selected_allocation) {
    iree_hal_tlsf_pool_slab_t* slab = selected_allocation.slab;
    if (!iree_hal_tlsf_pool_adjust_charged_reservation(
            pool, &charged_length, selected_allocation.allocation.length)) {
      iree_atomic_fetch_add(&pool->over_budget_count, 1,
                            iree_memory_order_relaxed);
      iree_hal_memory_tlsf_restore(&slab->tlsf,
                                   selected_allocation.allocation.block_index);
      iree_hal_tlsf_pool_note_return_candidate(pool, slab);
      memset(out_reservation, 0, sizeof(*out_reservation));
      memset(out_info, 0, sizeof(*out_info));
      out_info->result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
      *out_result = IREE_HAL_POOL_ACQUIRE_OVER_BUDGET;
      iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
      charged_length = 0;
    } else {
      if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
        // Selection guarantees a complete aligned range for this layout.
        asan_layout.backing_length = selected_allocation.allocation.length;
        asan_layout.right_redzone_length = asan_layout.backing_length -
                                           asan_layout.user_offset -
                                           asan_layout.user_length;
      }
      iree_hal_tlsf_pool_return_allocation(
          pool, metadata, &selected_allocation, size, charged_length,
          &asan_layout, selected_result, out_reservation, out_info, out_result);
    }
  } else {
    memset(out_reservation, 0, sizeof(*out_reservation));
    memset(out_info, 0, sizeof(*out_info));
    if (growth_required || iree_hal_tlsf_pool_metadata_is_required(metadata)) {
      out_info->flags |= IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED;
    }
    out_info->result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
    charged_length = 0;
  }
}

// Rolls back a reservation acquired by the current transaction before it was
// made visible to the caller.
static void iree_hal_tlsf_pool_rollback_reservation_locked(
    iree_hal_tlsf_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
    const iree_hal_pool_acquire_info_t* info)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(&pool->mutex)) {
  iree_hal_tlsf_pool_release_node_t* release_node =
      (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservation->block_handle;
  iree_hal_tlsf_pool_uncharge_reservation(pool, release_node->charged_length);
  if (release_node->block_index != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE) {
    iree_hal_tlsf_pool_slab_t* slab =
        iree_hal_tlsf_pool_slab_from_release_node(release_node);
    iree_hal_memory_tlsf_restore(&slab->tlsf, release_node->block_index);
    iree_hal_tlsf_pool_note_return_candidate(pool, slab);
    iree_hal_tlsf_pool_recycle_release_node(pool, release_node);
  }
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

// Publishes tracing only after the entire reservation transaction commits.
static void iree_hal_tlsf_pool_commit_reservation(
    iree_hal_tlsf_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation) {
  iree_hal_tlsf_pool_release_node_t* node =
      (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservation->block_handle;
  // Each live release record has a unique identity, independent of native
  // address representation or overlapping offsets in different backing ranges.
  iree_hal_memory_trace_alloc(&pool->trace, node, reservation->byte_length);
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

  iree_hal_tlsf_pool_acquire_element_t
      inline_elements[IREE_HAL_TLSF_POOL_INLINE_TRANSACTION_CAPACITY];
  const bool growth_allowed =
      !iree_any_bit_set(flags, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH);
  if (request_count > IREE_ARRAYSIZE(inline_elements) && !growth_allowed) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      iree_hal_tlsf_pool_request_geometry_t geometry = {0};
      IREE_RETURN_IF_ERROR(iree_hal_tlsf_pool_reservation_geometry(
          &requests[i], pool->slab_options.alignment, &pool->capabilities,
          &pool->asan_options, &geometry));
    }
    // Staging large transactions is cold work even when all backing and
    // persistent allocator metadata are already available.
    iree_atomic_fetch_add(&pool->exhausted_count, 1, iree_memory_order_relaxed);
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_infos[i] = (iree_hal_pool_acquire_info_t){
          .result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED,
          .flags = IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED,
      };
    }
    *out_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
    return iree_ok_status();
  }
  iree_hal_tlsf_pool_acquire_element_t* elements = inline_elements;
  bool elements_allocated = false;
  iree_status_t status = iree_ok_status();
  if (request_count > IREE_ARRAYSIZE(inline_elements)) {
    status = iree_allocator_malloc_array(pool->host_allocator, request_count,
                                         sizeof(*elements), (void**)&elements);
    elements_allocated = iree_status_is_ok(status);
  }
  iree_host_size_t initialized_count = 0;
  if (iree_status_is_ok(status)) {
    memset(elements, 0, request_count * sizeof(*elements));
    initialized_count = request_count;
  }
  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_tlsf_pool_reservation_geometry(
        &requests[i], pool->slab_options.alignment, &pool->capabilities,
        &pool->asan_options, &elements[i].geometry);
  }
  iree_host_size_t acquired_count = 0;
  iree_host_size_t preparing_count = 0;
  iree_hal_pool_acquire_result_t transaction_result =
      IREE_HAL_POOL_ACQUIRE_OK_FRESH;
  iree_hal_tlsf_pool_slab_t* prepared_slab = NULL;
  iree_hal_tlsf_pool_metadata_t metadata = {0};
  bool needs_preparation = true;
  while (iree_status_is_ok(status) && needs_preparation) {
    acquired_count = 0;
    transaction_result = IREE_HAL_POOL_ACQUIRE_OK_FRESH;
    bool published_growth = false;
    bool needs_growth = false;
    bool needs_dedicated = false;
    memset(&metadata.required, 0, sizeof(metadata.required));
    iree_slim_mutex_lock(&pool->mutex);
    for (iree_host_size_t i = 0; i < preparing_count; ++i) {
      --elements[i].preparing_slab->preparation_count;
      iree_hal_tlsf_pool_note_return_candidate(pool,
                                               elements[i].preparing_slab);
    }
    preparing_count = 0;
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      elements[i].preparing_slab = NULL;
      memset(&elements[i].reservation, 0, sizeof(elements[i].reservation));
      memset(&elements[i].info, 0, sizeof(elements[i].info));
    }
    iree_hal_tlsf_pool_drain_pending_releases(pool);
    while (acquired_count < request_count) {
      iree_hal_pool_acquire_result_t item_result = IREE_HAL_POOL_ACQUIRE_NONE;
      iree_hal_tlsf_pool_acquire_one_reservation_locked(
          pool, &requests[acquired_count], requester_frontier, flags, &metadata,
          &elements[acquired_count], &item_result);
      if (iree_hal_tlsf_pool_metadata_is_required(&metadata)) {
        transaction_result = item_result;
        break;
      }
      if (item_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED &&
          iree_any_bit_set(elements[acquired_count].info.flags,
                           IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED) &&
          growth_allowed) {
        if (iree_hal_tlsf_pool_request_is_dedicated(
                pool, &requests[acquired_count])) {
          needs_dedicated = true;
        } else if (prepared_slab) {
          iree_hal_tlsf_pool_publish_slab(pool, prepared_slab);
          prepared_slab = NULL;
          published_growth = true;
          continue;
        } else {
          needs_growth = true;
        }
      }
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
        default:
          IREE_ASSERT(false, "invalid TLSF acquisition result: %u",
                      item_result);
          break;
      }
      if (item_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
          item_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
        break;
      }
      ++acquired_count;
    }
    needs_preparation =
        growth_allowed && (needs_growth || needs_dedicated ||
                           iree_hal_tlsf_pool_metadata_is_required(&metadata));
    if (acquired_count != request_count) {
      for (iree_host_size_t i = 0; i < acquired_count; ++i) {
        // Restored blocks remain available to competing callers, but their
        // backing must survive preparation or each retry can erase its own
        // progress toward a multi-slab transaction.
        const iree_hal_tlsf_pool_release_node_t* node =
            (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)elements[i]
                .reservation.block_handle;
        iree_hal_tlsf_pool_slab_t* slab =
            node->block_index != IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE
                ? iree_hal_tlsf_pool_slab_from_release_node(node)
                : NULL;
        iree_hal_tlsf_pool_rollback_reservation_locked(
            pool, &elements[i].reservation, &elements[i].info);
        memset(&elements[i].reservation, 0, sizeof(elements[i].reservation));
        memset(&elements[i].info, 0, sizeof(elements[i].info));
        if (needs_preparation && slab) {
          elements[preparing_count++].preparing_slab = slab;
          ++slab->preparation_count;
        }
      }
    }
    const bool needs_maintenance =
        pool->return_candidates != NULL || pool->dedicated.returns != NULL;
    if (needs_preparation && metadata.required.slab) {
      elements[preparing_count++].preparing_slab = metadata.required.slab;
      ++metadata.required.slab->preparation_count;
    }
    iree_slim_mutex_unlock(&pool->mutex);
    if (needs_maintenance) {
      iree_hal_tlsf_pool_schedule_maintenance(pool);
    }
    // A restored prefix was never visible to competing acquisitions. Only new
    // backing creates capacity progress; a rollback-only signal would let a
    // failed retry wake itself indefinitely.
    if (published_growth) {
      iree_async_notification_signal_if_observed(pool->base.notification,
                                                 INT32_MAX);
    }
    if (iree_status_is_ok(status) && needs_preparation &&
        metadata.required.blocks.first_block) {
      iree_allocator_free(pool->host_allocator,
                          metadata.prepared.blocks.storage);
      metadata.prepared.blocks = metadata.required.blocks;
      status = iree_hal_memory_tlsf_prepare_growth(&metadata.prepared.blocks,
                                                   pool->host_allocator);
    }
    if (iree_status_is_ok(status) && needs_preparation &&
        metadata.required.release_node) {
      // The accepted candidate needs a record. Prepare the remaining batch's
      // records together so each missing record does not cause another replay.
      for (iree_host_size_t i = acquired_count;
           i < request_count && iree_status_is_ok(status); ++i) {
        if (iree_hal_tlsf_pool_request_is_dedicated(pool, &requests[i])) {
          continue;
        }
        iree_hal_tlsf_pool_release_node_t* node = NULL;
        status = iree_allocator_malloc(pool->host_allocator,
                                       pool->reservation_layout.node_size,
                                       (void**)&node);
        if (iree_status_is_ok(status)) {
          node->next = metadata.prepared.release_nodes;
          metadata.prepared.release_nodes = node;
        }
      }
    }
    if (iree_status_is_ok(status) && needs_dedicated) {
      iree_hal_tlsf_pool_acquire_element_t* element = &elements[acquired_count];
      const iree_hal_pool_reservation_request_t backing_request = {
          .params =
              {
                  .type = pool->capabilities.memory_type,
                  .access = pool->capabilities.allowed_access,
                  .usage = pool->capabilities.supported_usage,
                  .queue_family_affinity =
                      pool->capabilities.queue_family_affinity,
                  .min_alignment =
                      iree_min(element->geometry.alignment,
                               pool->capabilities.max_allocation_alignment),
              },
          .allocation_size = element->geometry.length,
      };
      status = iree_hal_tlsf_pool_dedicated_acquire(
          pool->backing_pool, &backing_request, &pool->asan_options,
          &element->geometry.asan, &pool->reservation_layout,
          requester_frontier, flags, pool->host_allocator,
          &element->dedicated.node, &element->dedicated.result);
      if (iree_status_is_ok(status)) {
        if (element->dedicated.node) {
          iree_atomic_fetch_add(
              &pool->bytes_committed,
              (int64_t)iree_hal_tlsf_pool_dedicated_backing_length(
                  element->dedicated.node, &pool->reservation_layout),
              iree_memory_order_relaxed);
          iree_atomic_fetch_add(&pool->committed_slab_count, 1,
                                iree_memory_order_relaxed);
        } else {
          transaction_result = element->dedicated.result;
          element->info.result = transaction_result;
          element->info.flags = IREE_HAL_POOL_ACQUIRE_FLAG_NONE;
          if (transaction_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
            iree_atomic_fetch_add(&pool->over_budget_count, 1,
                                  iree_memory_order_relaxed);
          }
          needs_preparation = false;
        }
      }
    }
    if (iree_status_is_ok(status) && needs_growth) {
      iree_hal_pool_acquire_result_t backing_result;
      status = iree_hal_tlsf_pool_prepare_slab(pool, requester_frontier, flags,
                                               &prepared_slab, &backing_result);
      if (iree_status_is_ok(status) && !prepared_slab) {
        transaction_result = backing_result;
        elements[acquired_count].info.result = backing_result;
        elements[acquired_count].info.flags = IREE_HAL_POOL_ACQUIRE_FLAG_NONE;
        if (backing_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
          iree_atomic_fetch_add(&pool->over_budget_count, 1,
                                iree_memory_order_relaxed);
        }
        needs_preparation = false;
      }
    }
  }
  if (preparing_count) {
    iree_slim_mutex_lock(&pool->mutex);
    for (iree_host_size_t i = 0; i < preparing_count; ++i) {
      --elements[i].preparing_slab->preparation_count;
      iree_hal_tlsf_pool_note_return_candidate(pool,
                                               elements[i].preparing_slab);
    }
    iree_slim_mutex_unlock(&pool->mutex);
    iree_hal_tlsf_pool_schedule_maintenance(pool);
  }
  iree_allocator_free(pool->host_allocator, metadata.prepared.blocks.storage);
  iree_hal_tlsf_pool_free_release_node_list(pool,
                                            metadata.prepared.release_nodes);
  if (prepared_slab) {
    iree_hal_tlsf_pool_destroy_slab(pool, prepared_slab);
  }

  if (!iree_status_is_ok(status) || acquired_count != request_count) {
    for (iree_host_size_t i = 0; i < initialized_count; ++i) {
      if (elements[i].dedicated.node) {
        iree_hal_tlsf_pool_release_dedicated_list(pool,
                                                  elements[i].dedicated.node);
      }
    }
  }

  if (iree_status_is_ok(status)) {
    if (transaction_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED) {
      iree_atomic_fetch_add(&pool->exhausted_count, 1,
                            iree_memory_order_relaxed);
    }
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      out_infos[i] = elements[i].info;
    }
    if (acquired_count == request_count) {
      for (iree_host_size_t i = 0; i < request_count; ++i) {
        iree_hal_tlsf_pool_commit_reservation(pool, &elements[i].reservation);
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
  const uint8_t frontier_capacity = pool->reservation_layout.frontier.capacity;

  if (death_frontier && death_frontier->entry_count > 0) {
    if (death_frontier->entry_count <= frontier_capacity) {
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
      iree_async_frontier_initialize(release_frontier,
                                     (uint8_t)(frontier_capacity + 1u));
    }
  } else {
    iree_async_frontier_initialize(release_frontier, 0);
  }

  iree_hal_memory_trace_free(&pool->trace, release_node);

  // Publication transfers the node to acquisition, which may recycle it.
  const iree_device_size_t charged_length = release_node->charged_length;
  iree_hal_tlsf_pool_push_pending_release(pool, release_node);

  iree_hal_tlsf_pool_uncharge_reservation(pool, charged_length);
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
  iree_hal_tlsf_pool_schedule_maintenance(pool);
  iree_async_notification_signal_if_observed(pool->base.notification,
                                             INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static void iree_hal_tlsf_pool_advise_asan_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_asan_range_advice_flags_t flags) {
  (void)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_tlsf_pool_release_node_t* node =
        (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservations[i]
            .block_handle;
    iree_hal_pool_buffer_range_advise_asan(node->range, node->backing_offset,
                                           flags, &node->asan_layout);
  }
}

static void iree_hal_tlsf_pool_query_reservation_views(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_reservation_view_t* out_views) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_tlsf_pool_release_node_t* release_node =
        (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)reservations[i]
            .block_handle;
    iree_hal_pool_buffer_range_query_reservation_view(
        release_node->range, reservations[i].offset,
        reservations[i].byte_length,
        iree_hal_tlsf_pool_reservation_frontier(
            release_node, pool->reservation_layout.frontier.offset),
        &out_views[i]);
  }
}

static iree_status_t iree_hal_tlsf_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  return iree_hal_tlsf_pool_reservation_materialize(
      base_pool, pool->reservation_layout.frontier.offset, reservation_count,
      requests, reservations, flags, pool->host_allocator, out_buffers);
}

static void iree_hal_tlsf_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_tlsf_pool_t* pool = (const iree_hal_tlsf_pool_t*)base_pool;
  *out_capabilities = pool->capabilities;
  out_capabilities->min_allocation_size = 1;
  out_capabilities->max_allocation_size = pool->max_reservation_size;
}

static iree_status_t iree_hal_tlsf_pool_validate_asan(
    const iree_hal_pool_t* base_pool,
    const iree_hal_asan_pool_options_t* options) {
  const iree_hal_tlsf_pool_t* pool = (const iree_hal_tlsf_pool_t*)base_pool;
  if (pool->backing_pool) {
    return iree_hal_pool_validate_asan_options(pool->backing_pool, options);
  }
  const iree_hal_buffer_range_advice_t* advice =
      pool->source_range.memory.backing->advice;
  if (!advice) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer has no native ASAN range advice");
  }
  return advice->validate_asan(advice->user_data, options);
}

static void iree_hal_tlsf_pool_query_stats(const iree_hal_pool_t* base_pool,
                                           iree_hal_pool_stats_t* out_stats) {
  const iree_hal_tlsf_pool_t* pool = (const iree_hal_tlsf_pool_t*)base_pool;
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  out_stats->bytes_committed = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_committed, iree_memory_order_relaxed);
  out_stats->bytes_quarantined = (iree_device_size_t)iree_atomic_load(
      &pool->quarantine.size, iree_memory_order_relaxed);
  out_stats->bytes_free =
      out_stats->bytes_committed > out_stats->bytes_reserved
          ? out_stats->bytes_committed - out_stats->bytes_reserved
          : 0;
  out_stats->bytes_free -=
      iree_min(out_stats->bytes_free, out_stats->bytes_quarantined);
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
  out_stats->quarantine_eviction_count = (uint64_t)iree_atomic_load(
      &pool->quarantine.eviction_count, iree_memory_order_relaxed);
}

static void iree_hal_tlsf_pool_trim(iree_hal_pool_t* base_pool,
                                    iree_hal_pool_trim_flags_t flags,
                                    iree_device_size_t min_bytes_to_keep) {
  iree_hal_tlsf_pool_t* pool = (iree_hal_tlsf_pool_t*)base_pool;
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_tlsf_pool_drain_pending_releases(pool);
  iree_hal_tlsf_pool_flush_quarantine(pool);
  iree_hal_tlsf_pool_slab_t* retired_slabs = NULL;
  if (pool->backing_pool) {
    for (iree_hal_tlsf_pool_slab_t* slab = pool->slabs.head; slab;
         slab = slab->next) {
      iree_hal_tlsf_pool_note_return_candidate(pool, slab);
    }
    retired_slabs =
        iree_hal_tlsf_pool_take_unused_slabs(pool, min_bytes_to_keep);
  }
  iree_hal_tlsf_pool_release_node_t* release_nodes =
      iree_hal_tlsf_pool_take_free_release_nodes(pool);
  const bool needs_maintenance =
      pool->return_candidates != NULL || pool->dedicated.returns != NULL;
  iree_slim_mutex_unlock(&pool->mutex);
  iree_hal_tlsf_pool_free_release_node_list(pool, release_nodes);
  iree_hal_tlsf_pool_destroy_slab_list(pool, retired_slabs);
  if (needs_maintenance) {
    iree_hal_tlsf_pool_schedule_maintenance(pool);
  }
  (void)flags;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_tlsf_pool_vtable = {
    .destroy = iree_hal_tlsf_pool_destroy,
    .acquire_reservations = iree_hal_tlsf_pool_acquire_reservations,
    .release_reservations = iree_hal_tlsf_pool_release_reservations,
    .query_reservation_views = iree_hal_tlsf_pool_query_reservation_views,
    .materialize_reservations = iree_hal_tlsf_pool_materialize_reservations,
    .query_capabilities = iree_hal_tlsf_pool_query_capabilities,
    .validate_asan = iree_hal_tlsf_pool_validate_asan,
    .query_stats = iree_hal_tlsf_pool_query_stats,
    .trim = iree_hal_tlsf_pool_trim,
    .advise_asan_reservations = iree_hal_tlsf_pool_advise_asan_reservations,
};
