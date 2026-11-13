// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory.h"

#include <stdlib.h>

#include "iree/async/semaphore.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/memory/tlsf.h"
#include "iree/hal/pool_wait.h"
#include "loom/runtime/storage/backing_directory.h"
#include "loom/runtime/storage/backing_pages.h"

struct loom_runtime_memory_pool_t {
  // Allocator owning pool, buffer, and slab metadata.
  iree_allocator_t host_allocator;
  // Borrowed sealed group outliving all source pools and virtual buffers.
  iree_hal_device_group_t* group;
  // Retained mapping-source pool, used only by this physical budget.
  iree_hal_pool_t* storage_pool;
  // Physical admission survives logical return and unpublished native failure.
  iree_hal_memory_budget_t* budget;
  // Serialized parent/slot inventory; one HAL owner per native allocation.
  loom_runtime_backing_pages_t* pages;
  // Exact reusable whole-weight geometry.
  iree_hal_pool_reservation_request_t weight_request;
  // Fixed physical mapping slot geometry, independent of logical tensor blocks.
  iree_hal_pool_reservation_request_t packed_request;
  // Borrowed queue accepting batched allocation and mapping operations.
  iree_hal_queue_t* queue;
  // Exact virtual-target scope; family storage trails this object.
  iree_hal_pool_scope_t scope;
  // All sites admitted by the virtual-target scope in this sealed group.
  iree_hal_memory_scope_t mapping_scope;
  // Capacity observation independent of any model's completion semaphore.
  iree_hal_pool_wait_t* capacity_wait;
  // Protects capacity completion and pool observer lifetime only.
  iree_slim_mutex_t capacity_mutex;
  // Outer teardown waits for a committed capacity observation to retire.
  iree_notification_t capacity_notification;
  // A committed capacity observation owns its callback storage.
  bool capacity_pending;
  // The owner withdrew an observation after its returned backing retired.
  bool capacity_withdrawn;
  // Callback-owned failure transferred to the serialized owner on collection.
  iree_status_t capacity_status;
  // The capacity source failed and cannot admit another observation.
  bool capacity_failed;
  // Finite physical admission ceiling selected when the pool is created.
  uint64_t limit_bytes;
  // Accounting across all buffers, with physical allocations counted once.
  loom_runtime_storage_statistics_t statistics;
  // Logical live and accepted backing, independent of physical retention.
  loom_runtime_memory_demand_t demand;
  // Accepted physical cost no longer retained by inventory or permanent loss.
  uint64_t retiring_bytes;
  // Explicit physical trim, reapplied until accepted detachments return.
  struct {
    // Total native bytes the caller wishes to retain, including live storage.
    uint64_t target_bytes;
    // Reapply while accepted private unmaps may return more idle parents.
    bool pending;
  } trim;
  // Owner-only list of accepted operations awaiting result publication.
  loom_runtime_memory_operation_t* pending_operations;
  // Callback-to-owner handoff of operations ready for publication.
  iree_atomic_slist_t completed_operations;
  // Owner wake copied into operations and capacity observations.
  struct {
    // Nonblocking callback; detachment joins every accepted copy.
    void (*wake)(void* user_data);
    // Borrowed scheduling owner, never accessed after detachment.
    void* user_data;
  } observer;
  // Borrowed owner of cold capacity reclamation, separate from slab mechanics.
  loom_runtime_memory_reclaimer_t reclaimer;
  // Owned shared capacity-observation or reclamation failure.
  iree_status_t failure;
};

// Mapping serialization and failure are properties of the native reservation,
// including when several independently resident buffers occupy its ranges.
typedef struct loom_runtime_memory_target_t {
  // Retained last mapping completion; source allocation remains independent.
  iree_hal_semaphore_t* mapping_completion;
  // Owned incomplete-edit failure requiring checked whole-target detach.
  iree_status_t failure;
  // Terminal checked removal permits physical cleanup, never further execution.
  bool detached;
} loom_runtime_memory_target_t;

struct loom_runtime_memory_arena_t {
  // Borrowed physical owner outliving the arena and every allocated range.
  loom_runtime_memory_pool_t* pool;
  // Owned address interval, independent of physical payload and range views.
  iree_hal_buffer_t* handle;
  // Shared native mapping order and failure domain for every allocated range.
  loom_runtime_memory_target_t target;
  // Native geometry governing independent ranges and their physical slabs.
  loom_runtime_memory_geometry_t geometry;
  // Address-only allocation metadata; leases return after actual retirement.
  iree_hal_memory_tlsf_t ranges;
};

struct loom_runtime_virtual_buffer_t {
  // Borrowed physical owner outliving this reservation.
  loom_runtime_memory_pool_t* pool;
  // Borrowed caller accounting group, independent of the shared pool budget.
  loom_runtime_storage_statistics_t* statistics;
  // Owned stable virtual reservation or range view, independent of backing.
  iree_hal_buffer_t* handle;
  // Previous standalone reservation retained through cold growth mapping.
  // Failed growth keeps both targets until checked teardown.
  iree_hal_buffer_t* previous_handle;
  // Shared native-target state, or owned trailing state for a standalone root.
  loom_runtime_memory_target_t* target;
  // Borrowed address allocator, NULL when this buffer owns its reservation.
  loom_runtime_memory_arena_t* arena;
  // Address lease returned only after consumers and backing edits retire.
  iree_hal_memory_tlsf_block_index_t block_index;
  // Validated physical geometry, independent of other reservations.
  loom_runtime_memory_geometry_t geometry;
  // Required virtual base alignment, preserved across reservation growth.
  iree_device_size_t alignment;
  // Native-aligned reservation extent, including its exact short tail.
  iree_device_size_t byte_length;
  // Number of slots spanning the reservation, including an optional short tail.
  iree_host_size_t slab_count;
  // Regular slots with physical or accepted backing; excludes a short tail.
  iree_host_size_t backed_regular_count;
  // Physical ownership kind of regular mapping slots.
  loom_runtime_backing_kind_t regular_kind;
  // Physical ownership kind of the final possibly shorter mapping slot.
  loom_runtime_backing_kind_t tail_kind;
  // Native cost of one dedicated regular extent; zero for reusable classes.
  uint64_t regular_cost;
  // Native cost of the final dedicated extent, including alignment padding.
  uint64_t tail_cost;
  // Sparse physical-slot state, proportional to touched slots, not extent.
  loom_runtime_backing_directory_t backing;
};

typedef struct loom_runtime_memory_edit_t {
  // Borrowed reservation held alive by the operation's caller.
  loom_runtime_virtual_buffer_t* buffer;
  // Slab index within buffer.
  iree_host_size_t index;
  // Result written exclusively by the last queue completion before publication.
  loom_runtime_backing_entry_t result;
} loom_runtime_memory_edit_t;

typedef struct loom_runtime_memory_stage_t {
  // Borrowed enclosing transaction through the last completion claim.
  loom_runtime_memory_operation_t* operation;
  // Owned private timeline; caller readiness cannot fail accepted ownership.
  iree_hal_semaphore_t* semaphore;
  // Embedded observation retained until this stage completes.
  iree_async_semaphore_timepoint_t timepoint;
  // Owned terminal result, published before dropping the completion claim.
  iree_status_t status;
} loom_runtime_memory_stage_t;

struct loom_runtime_memory_operation_t {
  // Submission and both private stage callbacks each own one claim.
  iree_atomic_int32_t pending;
  // Source readiness/map completion, or private unmap/slot-return readiness.
  loom_runtime_memory_stage_t stages[2];
  // Complete native growth captured before source IO.
  loom_runtime_backing_growth_t growth;
  // Unique pending-parent readiness, captured in trailing storage.
  iree_hal_semaphore_list_t allocation_waits;
  // Mapping runs in trailing storage, independent of logical edit ownership.
  iree_hal_memory_mapping_t* mappings;
  // Owned predecessor mapping references, at most one per edited reservation.
  iree_hal_semaphore_list_t mapping_waits;
  // Borrowed pool outliving the operation and every reservation.
  loom_runtime_memory_pool_t* pool;
  // Next operation in the owner-only pending publication list.
  loom_runtime_memory_operation_t* pending_next;
  // Previous link naming this operation in the owner-only pending list.
  loom_runtime_memory_operation_t** pending_previous_link;
  // Callback-to-owner handoff after the final native completion claim.
  iree_atomic_slist_entry_t completed_entry;
  // Readiness signaled after native mapping publication, retained by this job.
  iree_hal_semaphore_t* readiness;
  // Caller-selected readiness payload.
  uint64_t value;
  // Release existing physical ownership instead of constructing new backing.
  bool releasing;
  // Cold reservation growth, with unchanged physical ownership; null for
  // ordinary commitments and returns. Both stages are mapping operations.
  loom_runtime_virtual_buffer_t* growing;
  // Joins completion and observer access before the caller releases the job.
  iree_slim_mutex_t mutex;
  // Outer-thread wait only; shared task workers never wait here.
  iree_notification_t notification;
  // Both queue operations and submission have released captured ownership;
  // protected by mutex.
  bool complete;
  // Serialized owner has installed native results and returned unmapped slots;
  // protected by mutex.
  bool published;
  // Owned terminal result, independent of other operations on the same pool.
  iree_status_t status;
  // Native edits failed after backing could change; affected targets require
  // checked detach before their physical sources can be released.
  bool invalidates_backing;
  // Optional owner scheduling callback, protected by mutex.
  void (*wake)(void* user_data);
  // Borrowed callback payload, joined by detachment or final observation.
  void* user_data;
  // Pool-owner scheduling copy, independent of the caller's retirement wake.
  struct {
    // Protected by the operation mutex; never reenters the pool.
    void (*wake)(void* user_data);
    // Borrowed owner, joined by pool observer detachment.
    void* user_data;
  } observer;
  // Number of distinct selected slabs in the trailing edit array.
  iree_host_size_t count;
  // Exact native work; no worker traverses mutable serving metadata.
  loom_runtime_memory_edit_t edits[];
};

static void loom_runtime_memory_pool_collect(loom_runtime_memory_pool_t* pool);

static void loom_runtime_memory_capacity_completed(void* user_data,
                                                   iree_status_t status) {
  loom_runtime_memory_pool_t* pool = user_data;
  iree_slim_mutex_lock(&pool->capacity_mutex);
  if (pool->capacity_withdrawn && iree_status_is_cancelled(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  pool->capacity_status = iree_status_join(pool->capacity_status, status);
  pool->capacity_pending = false;
  pool->capacity_withdrawn = false;
  iree_notification_post(&pool->capacity_notification, IREE_ALL_WAITERS);
  if (pool->observer.wake) {
    pool->observer.wake(pool->observer.user_data);
  }
  iree_slim_mutex_unlock(&pool->capacity_mutex);
}

static bool loom_runtime_memory_capacity_is_idle(void* user_data) {
  loom_runtime_memory_pool_t* pool = user_data;
  iree_slim_mutex_lock(&pool->capacity_mutex);
  const bool idle = !pool->capacity_pending;
  iree_slim_mutex_unlock(&pool->capacity_mutex);
  return idle;
}

void loom_runtime_memory_demand_add(loom_runtime_memory_demand_t* demand,
                                    loom_runtime_memory_demand_t additional) {
  demand->packed_bytes += additional.packed_bytes;
  demand->weight_bytes += additional.weight_bytes;
  demand->dedicated_bytes += additional.dedicated_bytes;
  demand->packed_count += additional.packed_count;
  demand->dedicated_cost += additional.dedicated_cost;
}

static uint64_t loom_runtime_memory_demand_total(
    loom_runtime_memory_demand_t demand) {
  return demand.packed_bytes + demand.weight_bytes + demand.dedicated_bytes;
}

// Native credits are authoritative even while unrelated constructors are
// pending. Inventory owns retained parents; all remaining live charges are
// actual native retirement. Logical counters never authorize physical reuse.
static void loom_runtime_memory_pool_observe_retirement(
    loom_runtime_memory_pool_t* pool) {
  const uint64_t charged = iree_hal_memory_budget_charged(pool->budget);
  const uint64_t abandoned = iree_hal_memory_budget_abandoned(pool->budget);
  const uint64_t retained = loom_runtime_backing_pages_retained(pool->pages);
  // Observation only: source statistics never authorize admission or reuse.
  iree_hal_pool_stats_t native;
  iree_hal_pool_query_stats(pool->storage_pool, &native);
  pool->statistics.peak_bytes =
      iree_max(pool->statistics.peak_bytes, native.bytes_committed);
  pool->statistics.abandoned_bytes = abandoned;
  pool->retiring_bytes =
      charged > retained + abandoned ? charged - retained - abandoned : 0;
}

static uint64_t loom_runtime_memory_growth_cost(
    const loom_runtime_memory_pool_t* pool,
    loom_runtime_memory_demand_t demand) {
  return loom_runtime_backing_pages_growth_cost(
      pool->pages, demand.packed_count,
      demand.weight_bytes / pool->weight_request.allocation_size,
      demand.dedicated_cost);
}

static uint64_t loom_runtime_memory_pool_projected_bytes(
    const loom_runtime_memory_pool_t* pool,
    loom_runtime_memory_demand_t additional) {
  const uint64_t charged = iree_hal_memory_budget_charged(pool->budget);
  const uint64_t growth = loom_runtime_memory_growth_cost(pool, additional);
  return growth > UINT64_MAX - charged ? UINT64_MAX : charged + growth;
}

static void loom_runtime_memory_native_wake(void* user_data) {
  loom_runtime_memory_pool_t* pool = user_data;
  iree_slim_mutex_lock(&pool->capacity_mutex);
  if (pool->observer.wake) {
    pool->observer.wake(pool->observer.user_data);
  }
  iree_slim_mutex_unlock(&pool->capacity_mutex);
}

// Arms a one-shot capacity wake only while native retirement can make progress.
// Prepare precedes the final query so a concurrent free cannot lose its wake.
// True preserves an observed wait even if its callback races this return: one
// capacity change does not establish that every native release has retired.
static bool loom_runtime_memory_pool_observe_capacity(
    loom_runtime_memory_pool_t* pool) {
  if (pool->capacity_failed) {
    return false;
  }
  if (!loom_runtime_memory_capacity_is_idle(pool)) {
    loom_runtime_memory_pool_observe_retirement(pool);
    if (!pool->retiring_bytes) {
      // A native return can finish before this observation is armed. Withdraw
      // the obsolete wait when no further native free remains to signal it.
      iree_slim_mutex_lock(&pool->capacity_mutex);
      pool->capacity_withdrawn = pool->capacity_pending;
      iree_slim_mutex_unlock(&pool->capacity_mutex);
      iree_hal_pool_wait_cancel(pool->capacity_wait);
    }
    return true;
  }
  iree_hal_pool_wait_prepare(pool->capacity_wait);
  loom_runtime_memory_pool_observe_retirement(pool);
  if (!pool->retiring_bytes) {
    iree_hal_pool_wait_abort(pool->capacity_wait);
    return false;
  }
  iree_slim_mutex_lock(&pool->capacity_mutex);
  pool->capacity_pending = true;
  iree_slim_mutex_unlock(&pool->capacity_mutex);
  iree_hal_pool_wait_commit(pool->capacity_wait, iree_infinite_timeout(),
                            (iree_hal_pool_wait_callback_t){
                                loom_runtime_memory_capacity_completed, pool});
  return true;
}

iree_status_t loom_runtime_memory_pool_allocate(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_hal_pool_t* storage_pool, iree_hal_queue_t* queue,
    loom_runtime_memory_geometry_t weight_geometry,
    loom_runtime_memory_geometry_t state_geometry, uint64_t limit_bytes,
    iree_allocator_t host_allocator, loom_runtime_memory_pool_t** out_pool) {
  *out_pool = NULL;
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(storage_pool, &capabilities);
  limit_bytes = limit_bytes ? limit_bytes : UINT64_MAX;
  if (capabilities.physical_capacity) {
    limit_bytes = iree_min(limit_bytes, capabilities.physical_capacity);
  }
  if (limit_bytes == UINT64_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "elastic storage requires a backing capacity; the selected pool "
        "does not report one, so specify its memory_bytes budget");
  }
  loom_runtime_memory_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*pool), scope.family_count,
      sizeof(iree_hal_pool_family_access_t), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  pool->host_allocator = host_allocator;
  pool->group = group;
  pool->storage_pool = storage_pool;
  iree_hal_pool_retain(storage_pool);
  pool->queue = queue;
  pool->scope = scope;
  pool->scope.families = (iree_hal_pool_family_access_t*)(pool + 1);
  memcpy((void*)pool->scope.families, scope.families,
         scope.family_count * sizeof(*scope.families));
  pool->mapping_scope = (iree_hal_memory_scope_t){
      iree_hal_device_group_memory_domain(group), IREE_HAL_MEMORY_SCOPE_ANY};
  pool->limit_bytes = limit_bytes;
  iree_slim_mutex_initialize(&pool->capacity_mutex);
  iree_atomic_slist_initialize(&pool->completed_operations);
  iree_notification_initialize(&pool->capacity_notification);
  pool->weight_request = iree_hal_make_pool_reservation_request(
      weight_geometry.mapping_size, weight_geometry.granularity);
  pool->packed_request = iree_hal_make_pool_reservation_request(
      state_geometry.mapping_size, state_geometry.granularity);
  const uint64_t parent_granularity =
      iree_max(state_geometry.granularity, capabilities.maintenance_alignment);
  const iree_hal_pool_reservation_request_t parent_request =
      iree_hal_make_pool_reservation_request(
          iree_device_align(state_geometry.backing_size, parent_granularity),
          iree_min(parent_granularity, capabilities.max_allocation_alignment));
  iree_status_t status = iree_hal_memory_budget_create(
      limit_bytes, storage_pool->notification, host_allocator, &pool->budget);
  if (iree_status_is_ok(status)) {
    status = loom_runtime_backing_pages_allocate(
        storage_pool,
        iree_hal_queue_family_device(iree_hal_queue_family(queue)),
        pool->packed_request, parent_request, pool->weight_request,
        loom_runtime_memory_native_wake, pool, host_allocator, &pool->pages);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_wait_allocate(storage_pool, host_allocator,
                                         &pool->capacity_wait);
  }
  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    loom_runtime_memory_pool_free(pool);
  }
  return status;
}

void loom_runtime_memory_pool_free(loom_runtime_memory_pool_t* pool) {
  if (!pool) {
    return;
  }
  IREE_ASSERT(!pool->statistics.reserved_bytes &&
              !pool->statistics.committed_bytes && !pool->pending_operations);
  if (!loom_runtime_memory_capacity_is_idle(pool)) {
    iree_hal_pool_wait_cancel(pool->capacity_wait);
    iree_notification_await(&pool->capacity_notification,
                            loom_runtime_memory_capacity_is_idle, pool,
                            iree_infinite_timeout());
  }
  iree_hal_pool_wait_free(pool->capacity_wait);
  iree_status_free(pool->capacity_status);
  iree_status_free(pool->failure);
  iree_notification_deinitialize(&pool->capacity_notification);
  iree_atomic_slist_deinitialize(&pool->completed_operations);
  iree_slim_mutex_deinitialize(&pool->capacity_mutex);
  loom_runtime_backing_pages_free(pool->pages);
  iree_hal_memory_budget_release(pool->budget);
  iree_hal_pool_release(pool->storage_pool);
  iree_allocator_free(pool->host_allocator, pool);
}

loom_runtime_storage_statistics_t loom_runtime_memory_pool_statistics(
    loom_runtime_memory_pool_t* pool) {
  loom_runtime_memory_pool_collect(pool);
  loom_runtime_memory_pool_observe_retirement(pool);
  loom_runtime_storage_statistics_t statistics = pool->statistics;
  iree_hal_pool_stats_t native;
  iree_hal_pool_query_stats(pool->storage_pool, &native);
  statistics.committed_bytes = native.bytes_committed;
  statistics.cached_bytes = loom_runtime_backing_pages_cached(pool->pages);
  statistics.releasing_bytes += pool->retiring_bytes;
  return statistics;
}

loom_runtime_storage_statistics_t loom_runtime_memory_pool_snapshot(
    const loom_runtime_memory_pool_t* pool) {
  return pool->statistics;
}

uint64_t loom_runtime_memory_pool_retained_bytes(
    const loom_runtime_memory_pool_t* pool) {
  return iree_hal_memory_budget_charged(pool->budget);
}

static void loom_runtime_memory_pool_trim_cache(
    loom_runtime_memory_pool_t* pool, uint64_t target_bytes,
    loom_runtime_memory_demand_t additional) {
  const uint64_t projected =
      loom_runtime_memory_pool_projected_bytes(pool, additional);
  const uint64_t excess =
      projected > target_bytes ? projected - target_bytes : 0;
  const uint64_t to_return =
      excess > pool->retiring_bytes ? excess - pool->retiring_bytes : 0;
  loom_runtime_backing_pages_trim(
      pool->pages, to_return, additional.packed_count,
      additional.weight_bytes / pool->weight_request.allocation_size);
  loom_runtime_memory_pool_observe_retirement(pool);
}

void loom_runtime_memory_pool_trim(loom_runtime_memory_pool_t* pool,
                                   uint64_t target_bytes) {
  pool->trim.target_bytes = target_bytes;
  pool->trim.pending = true;
  loom_runtime_memory_pool_collect(pool);
}

uint64_t loom_runtime_memory_pool_limit(
    const loom_runtime_memory_pool_t* pool) {
  return pool->limit_bytes;
}

uint64_t loom_runtime_memory_pool_live_bytes(
    const loom_runtime_memory_pool_t* pool) {
  return loom_runtime_memory_demand_total(pool->demand);
}

iree_status_t loom_runtime_memory_pool_query_status(
    loom_runtime_memory_pool_t* pool) {
  loom_runtime_memory_pool_collect(pool);
  return iree_status_clone(pool->failure);
}

loom_runtime_wake_callback_t loom_runtime_memory_pool_set_observer(
    loom_runtime_memory_pool_t* pool, void (*wake)(void* user_data),
    void* user_data) {
  iree_slim_mutex_lock(&pool->capacity_mutex);
  const loom_runtime_wake_callback_t previous = {pool->observer.wake,
                                                 pool->observer.user_data};
  pool->observer.wake = wake;
  pool->observer.user_data = user_data;
  iree_slim_mutex_unlock(&pool->capacity_mutex);
  for (loom_runtime_memory_operation_t* operation = pool->pending_operations;
       operation; operation = operation->pending_next) {
    iree_slim_mutex_lock(&operation->mutex);
    operation->observer.wake = wake;
    operation->observer.user_data = user_data;
    iree_slim_mutex_unlock(&operation->mutex);
  }
  return previous;
}

bool loom_runtime_memory_pool_is_pending(loom_runtime_memory_pool_t* pool) {
  loom_runtime_memory_pool_collect(pool);
  if (pool->pending_operations ||
      loom_runtime_backing_pages_is_pending(pool->pages)) {
    return true;
  }
  return loom_runtime_memory_pool_observe_capacity(pool);
}

iree_status_t loom_runtime_memory_pool_await(loom_runtime_memory_pool_t* pool) {
  while (loom_runtime_memory_pool_is_pending(pool)) {
    if (pool->pending_operations) {
      loom_runtime_memory_operation_await(pool->pending_operations);
    } else if (loom_runtime_backing_pages_is_pending(pool->pages)) {
      loom_runtime_backing_pages_await(pool->pages);
    } else {
      iree_notification_await(&pool->capacity_notification,
                              loom_runtime_memory_capacity_is_idle, pool,
                              iree_infinite_timeout());
    }
  }
  // A callback can finish after the last collection but before observation
  // finds no remaining retirement. Join its published status as well.
  loom_runtime_memory_pool_collect(pool);
  return iree_status_clone(pool->failure);
}

void loom_runtime_memory_pool_set_reclaimer(
    loom_runtime_memory_pool_t* pool,
    loom_runtime_memory_reclaimer_t reclaimer) {
  pool->reclaimer = reclaimer;
}

iree_status_t loom_runtime_memory_pool_prepare(
    loom_runtime_memory_pool_t* pool, loom_runtime_memory_demand_t additional,
    loom_runtime_memory_admission_t* out_admission) {
  *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_CAPACITY;
  loom_runtime_memory_pool_collect(pool);
  IREE_RETURN_IF_ERROR(iree_status_clone(pool->failure));
  const uint64_t additional_bytes =
      loom_runtime_memory_demand_total(additional);
  if (!additional_bytes) {
    *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
    return iree_ok_status();
  }
  if (additional_bytes > pool->limit_bytes) {
    return iree_ok_status();
  }
  // Explicit trim can change class ceilings. Resolve it before promising
  // a complete no-yield cohort; collection midway through its streamed commits
  // must not destroy the capacity on which that promise depends.
  if (pool->trim.pending) {
    *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_DEFERRED;
    return iree_ok_status();
  }
  if (loom_runtime_memory_pool_projected_bytes(pool, additional) >
          pool->limit_bytes &&
      pool->reclaimer.fn) {
    const uint64_t target = pool->limit_bytes - additional_bytes;
    IREE_TRACE_ZONE_BEGIN_NAMED(z_reclaim, "loom_serve_memory_reclaim");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_reclaim, additional_bytes);
    iree_status_t status =
        pool->reclaimer.fn(pool->reclaimer.user_data, target);
    IREE_TRACE_ZONE_END(z_reclaim);
    if (!iree_status_is_ok(status)) {
      if (iree_status_is_ok(pool->failure)) {
        pool->failure = iree_status_clone(status);
      }
      return status;
    }
    loom_runtime_memory_pool_collect(pool);
    IREE_RETURN_IF_ERROR(iree_status_clone(pool->failure));
  }
  // Another class's idle bytes cannot satisfy this demand. Return them without
  // joining unrelated acquisitions; their native disposal must finish before
  // those bytes become available for a different physical extent.
  if (loom_runtime_memory_pool_projected_bytes(pool, additional) >
      pool->limit_bytes) {
    loom_runtime_memory_pool_trim_cache(pool, pool->limit_bytes, additional);
  }
  // Reclamation can schedule a physical trim. Resolve that boundary before
  // returning the same no-yield promise as an immediately available cohort.
  bool deferred = pool->trim.pending;
  if (loom_runtime_memory_pool_projected_bytes(pool, additional) >
      pool->limit_bytes) {
    deferred |= loom_runtime_memory_pool_observe_capacity(pool) ||
                pool->pending_operations != NULL;
  }
  if (!pool->trim.pending && loom_runtime_memory_pool_projected_bytes(
                                 pool, additional) <= pool->limit_bytes) {
    *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
  } else if (deferred) {
    *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_DEFERRED;
  } else if (pool->statistics.abandoned_bytes &&
             loom_runtime_memory_pool_projected_bytes(pool, additional) -
                     pool->statistics.abandoned_bytes <=
                 pool->limit_bytes) {
    // No native return can satisfy this attempt. Report loss only when this
    // demand would otherwise fit; surviving capacity remains usable.
    return iree_status_annotate_f(
        iree_hal_memory_budget_query_loss(pool->budget),
        "physical admission is blocked by %" PRIu64
        " bytes of abandoned backing within a %" PRIu64 " byte limit",
        pool->statistics.abandoned_bytes, pool->limit_bytes);
  }
  return iree_ok_status();
}

// Physical matching is independent of the reservation's role. In particular,
// a bulk root's short tail can share sparse state backing.
static loom_runtime_backing_kind_t loom_runtime_memory_pool_kind(
    const loom_runtime_memory_pool_t* pool, iree_device_size_t size,
    iree_device_size_t alignment) {
  if (size <= pool->packed_request.allocation_size &&
      alignment <= pool->packed_request.min_alignment) {
    return LOOM_RUNTIME_BACKING_PACKED;
  }
  if (size == pool->weight_request.allocation_size &&
      alignment <= pool->weight_request.min_alignment) {
    return LOOM_RUNTIME_BACKING_WEIGHT;
  }
  return LOOM_RUNTIME_BACKING_DEDICATED;
}

static iree_device_size_t loom_runtime_memory_slab_length(
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index) {
  return iree_min(buffer->geometry.mapping_size,
                  buffer->byte_length - index * buffer->geometry.mapping_size);
}

void loom_runtime_virtual_buffer_select_backing(
    loom_runtime_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  if (!length) {
    return;
  }
  const iree_device_size_t slab_size = buffer->geometry.mapping_size;
  const iree_host_size_t begin = offset / slab_size;
  const iree_host_size_t end = (offset + length - 1) / slab_size + 1;
  if (end - begin < buffer->backing.capacity) {
    for (iree_host_size_t i = begin; i < end; ++i) {
      loom_runtime_backing_entry_t* entry =
          loom_runtime_backing_directory_find(&buffer->backing, i);
      if (entry && (entry->physical || entry->in_flight)) {
        entry->counted = true;
      }
    }
  } else {
    for (iree_host_size_t i = 0; i < buffer->backing.capacity; ++i) {
      loom_runtime_backing_entry_t* entry = &buffer->backing.entries[i];
      if (entry->occupied && entry->index >= begin && entry->index < end &&
          (entry->physical || entry->in_flight)) {
        entry->counted = true;
      }
    }
  }
}

iree_status_t loom_runtime_virtual_buffer_visit_backing(
    loom_runtime_virtual_buffer_t* buffer,
    iree_status_t (*visitor)(void* user_data, iree_device_size_t offset,
                             iree_device_size_t length),
    void* user_data) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < buffer->backing.capacity; ++i) {
    loom_runtime_backing_entry_t* entry = &buffer->backing.entries[i];
    if (entry->counted) {
      entry->counted = false;
      if (iree_status_is_ok(status) && visitor) {
        status =
            visitor(user_data, entry->index * buffer->geometry.mapping_size,
                    loom_runtime_memory_slab_length(buffer, entry->index));
      }
    }
  }
  return status;
}

iree_status_t loom_runtime_memory_ranges_footprint(
    iree_host_size_t range_count, const loom_runtime_memory_range_t* ranges,
    loom_runtime_memory_footprint_t* out_footprint) {
  *out_footprint = (loom_runtime_memory_footprint_t){0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t r = 0; r < range_count && iree_status_is_ok(status);
       ++r) {
    loom_runtime_virtual_buffer_t* buffer = ranges[r].buffer;
    const iree_device_size_t slab_size = buffer->geometry.mapping_size;
    const iree_host_size_t end =
        (ranges[r].offset + ranges[r].length - 1) / slab_size + 1;
    for (iree_host_size_t i = ranges[r].offset / slab_size;
         i < end && iree_status_is_ok(status); ++i) {
      loom_runtime_backing_entry_t* entry = NULL;
      status =
          loom_runtime_backing_directory_insert(&buffer->backing, i, &entry);
      if (iree_status_is_ok(status) && !entry->counted) {
        entry->counted = true;
        const uint64_t length = loom_runtime_memory_slab_length(buffer, i);
        out_footprint->total_bytes +=
            iree_min(length, UINT64_MAX - out_footprint->total_bytes);
        if (!entry->physical && !entry->in_flight) {
          out_footprint->missing_bytes +=
              iree_min(length, UINT64_MAX - out_footprint->missing_bytes);
        }
      }
    }
  }
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    loom_runtime_virtual_buffer_t* buffer = ranges[r].buffer;
    const iree_device_size_t slab_size = buffer->geometry.mapping_size;
    const iree_host_size_t end =
        (ranges[r].offset + ranges[r].length - 1) / slab_size + 1;
    for (iree_host_size_t i = ranges[r].offset / slab_size; i < end; ++i) {
      loom_runtime_backing_entry_t* entry =
          loom_runtime_backing_directory_find(&buffer->backing, i);
      if (entry && entry->counted) {
        entry->counted = false;
        if (!entry->physical && !entry->in_flight && !entry->keep) {
          loom_runtime_backing_directory_remove(&buffer->backing, entry);
        }
      }
    }
  }
  for (iree_host_size_t r = 0; r < range_count && iree_status_is_ok(status);
       ++r) {
    status = loom_runtime_backing_directory_shrink(&ranges[r].buffer->backing);
  }
  return status;
}

static uint64_t* loom_runtime_memory_slab_demand(
    loom_runtime_memory_demand_t* demand,
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index) {
  const loom_runtime_backing_kind_t kind = index + 1 == buffer->slab_count
                                               ? buffer->tail_kind
                                               : buffer->regular_kind;
  switch (kind) {
    case LOOM_RUNTIME_BACKING_PACKED:
      return &demand->packed_bytes;
    case LOOM_RUNTIME_BACKING_WEIGHT:
      return &demand->weight_bytes;
    default:
      return &demand->dedicated_bytes;
  }
}

static uint64_t loom_runtime_memory_slab_cost(
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index) {
  return index + 1 == buffer->slab_count ? buffer->tail_cost
                                         : buffer->regular_cost;
}

static void loom_runtime_memory_slab_add_demand(
    loom_runtime_memory_demand_t* demand,
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index) {
  *loom_runtime_memory_slab_demand(demand, buffer, index) +=
      loom_runtime_memory_slab_length(buffer, index);
  const loom_runtime_backing_kind_t kind = index + 1 == buffer->slab_count
                                               ? buffer->tail_kind
                                               : buffer->regular_kind;
  demand->packed_count += kind == LOOM_RUNTIME_BACKING_PACKED;
  demand->dedicated_cost += loom_runtime_memory_slab_cost(buffer, index);
}

static iree_status_t loom_runtime_virtual_buffer_allocate_metadata(
    loom_runtime_memory_pool_t* pool, loom_runtime_memory_geometry_t geometry,
    iree_device_size_t length, iree_device_size_t alignment,
    loom_runtime_storage_statistics_t* statistics,
    loom_runtime_memory_arena_t* arena,
    loom_runtime_virtual_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (!length || length > INT64_MAX - geometry.mapping_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid virtual buffer extent");
  }
  if (!alignment || geometry.granularity % alignment) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "virtual reservation cannot guarantee alignment %" PRIdsz, alignment);
  }
  const uint64_t count =
      (length + geometry.mapping_size - 1) / geometry.mapping_size;
  const iree_device_size_t tail = length % geometry.mapping_size;
  const iree_device_size_t byte_length =
      length - tail +
      (tail ? iree_device_align(iree_max(tail, geometry.minimum_size),
                                geometry.granularity)
            : 0);
  const iree_host_size_t base_size =
      sizeof(loom_runtime_virtual_buffer_t) +
      (arena ? 0 : sizeof(loom_runtime_memory_target_t));
  if (count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "virtual buffer has too many physical slots");
  }
  loom_runtime_virtual_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(pool->host_allocator, base_size, (void**)&buffer));
  buffer->pool = pool;
  loom_runtime_backing_directory_initialize(pool->host_allocator,
                                            &buffer->backing);
  buffer->statistics = statistics;
  buffer->arena = arena;
  buffer->target =
      arena ? &arena->target : (loom_runtime_memory_target_t*)(buffer + 1);
  buffer->geometry = geometry;
  buffer->alignment = alignment;
  buffer->byte_length = byte_length;
  buffer->slab_count = (iree_host_size_t)count;
  buffer->regular_kind = loom_runtime_memory_pool_kind(
      pool, geometry.mapping_size, geometry.granularity);
  buffer->tail_kind = loom_runtime_memory_pool_kind(
      pool, loom_runtime_memory_slab_length(buffer, count - 1),
      geometry.granularity);
  iree_status_t status = iree_ok_status();
  if (buffer->regular_kind == LOOM_RUNTIME_BACKING_DEDICATED) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(geometry.mapping_size,
                                               geometry.granularity);
    status = iree_hal_slab_pool_query_allocation_cost(
        pool->storage_pool, 1, &request, &buffer->regular_cost);
  }
  if (iree_status_is_ok(status) &&
      buffer->tail_kind == LOOM_RUNTIME_BACKING_DEDICATED) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(
            loom_runtime_memory_slab_length(buffer, count - 1),
            geometry.granularity);
    status = iree_hal_slab_pool_query_allocation_cost(
        pool->storage_pool, 1, &request, &buffer->tail_cost);
  }
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_allocator_free(pool->host_allocator, buffer);
  }
  return status;
}

iree_status_t loom_runtime_virtual_buffer_allocate(
    loom_runtime_memory_pool_t* pool, loom_runtime_memory_geometry_t geometry,
    iree_device_size_t length, iree_device_size_t alignment,
    loom_runtime_storage_statistics_t* statistics,
    loom_runtime_virtual_buffer_t** out_buffer) {
  *out_buffer = NULL;
  loom_runtime_virtual_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_virtual_buffer_allocate_metadata(
      pool, geometry, length, alignment, statistics, NULL, &buffer));
  // Loom replaces regular slots in mapping_size units, but packed tails may
  // source any native-granularity slice from a shared physical parent.
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = buffer->byte_length,
      .min_alignment = alignment,
      .mapping_granularity = geometry.granularity,
  };
  iree_status_t status = iree_hal_virtual_buffer_create(
      pool->group, pool->scope,
      (iree_hal_virtual_buffer_sources_t){1, &pool->storage_pool}, &options,
      pool->host_allocator, &buffer->handle);
  if (iree_status_is_ok(status)) {
    pool->statistics.reserved_bytes += buffer->byte_length;
    statistics->reserved_bytes += buffer->byte_length;
    *out_buffer = buffer;
  } else {
    loom_runtime_backing_directory_deinitialize(&buffer->backing);
    iree_allocator_free(pool->host_allocator, buffer);
  }
  return status;
}

iree_status_t loom_runtime_memory_arena_allocate(
    loom_runtime_memory_pool_t* pool, loom_runtime_memory_geometry_t geometry,
    iree_device_size_t length, loom_runtime_memory_arena_t** out_arena) {
  *out_arena = NULL;
  if (!length || length > INT64_MAX - geometry.granularity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid virtual arena extent");
  }
  length = iree_device_align(length, geometry.granularity);
  loom_runtime_memory_arena_t* arena = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                             sizeof(*arena), (void**)&arena));
  arena->pool = pool;
  arena->geometry = geometry;
  const iree_hal_memory_tlsf_options_t range_options = {
      .range_length = length,
      .alignment = geometry.granularity,
      // Range allocation is registration work; metadata grows with live banks.
      .initial_block_capacity = 8,
      // Ranges are returned only after their complete consumer lease retires.
      .frontier_capacity = 1,
  };
  iree_status_t status = iree_hal_memory_tlsf_initialize(
      range_options, pool->host_allocator, &arena->ranges);
  if (iree_status_is_ok(status)) {
    const iree_hal_virtual_buffer_options_t options = {
        .byte_length = length,
        .min_alignment = geometry.granularity,
        .mapping_granularity = geometry.granularity,
    };
    status = iree_hal_virtual_buffer_create(
        pool->group, pool->scope,
        (iree_hal_virtual_buffer_sources_t){1, &pool->storage_pool}, &options,
        pool->host_allocator, &arena->handle);
    if (!iree_status_is_ok(status)) {
      iree_hal_memory_tlsf_deinitialize(&arena->ranges);
    }
  }
  if (iree_status_is_ok(status)) {
    pool->statistics.reserved_bytes += length;
    *out_arena = arena;
  } else {
    iree_allocator_free(pool->host_allocator, arena);
  }
  return status;
}

iree_hal_buffer_t* loom_runtime_memory_arena_handle(
    const loom_runtime_memory_arena_t* arena) {
  return arena->handle;
}

iree_status_t loom_runtime_virtual_buffer_allocate_in_arena(
    loom_runtime_memory_arena_t* arena, iree_device_size_t length,
    iree_device_size_t alignment, loom_runtime_storage_statistics_t* statistics,
    loom_runtime_virtual_buffer_t** out_buffer) {
  *out_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_status_clone(arena->target.failure));
  IREE_ASSERT(!arena->target.detached);
  loom_runtime_virtual_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_virtual_buffer_allocate_metadata(
      arena->pool, arena->geometry, length, alignment, statistics, arena,
      &buffer));
  iree_hal_memory_tlsf_allocation_t allocation;
  iree_status_t status = iree_hal_memory_tlsf_allocate(
      &arena->ranges, buffer->byte_length, &allocation);
  if (iree_status_is_ok(status)) {
    buffer->block_index = allocation.block_index;
    status = iree_hal_buffer_subspan(
        arena->handle, allocation.offset, buffer->byte_length,
        arena->pool->host_allocator, &buffer->handle);
    if (!iree_status_is_ok(status)) {
      iree_hal_memory_tlsf_free(&arena->ranges, allocation.block_index, NULL);
    }
  }
  if (iree_status_is_ok(status)) {
    statistics->reserved_bytes += buffer->byte_length;
    *out_buffer = buffer;
  } else {
    loom_runtime_backing_directory_deinitialize(&buffer->backing);
    iree_allocator_free(arena->pool->host_allocator, buffer);
  }
  return status;
}

iree_status_t loom_runtime_memory_arena_detach(
    loom_runtime_memory_arena_t* arena) {
  loom_runtime_memory_pool_collect(arena->pool);
  // Whole-target removal requires retirement of every accepted range edit and
  // every consumer, not just the bank whose edit reported the failure.
  IREE_ASSERT(!arena->target.mapping_completion);
  iree_status_t status =
      iree_hal_virtual_buffer_detach(arena->handle, arena->pool->mapping_scope);
  if (iree_status_is_ok(status)) {
    arena->target.detached = true;
  } else if (iree_status_is_ok(arena->target.failure)) {
    arena->target.failure = iree_status_clone(status);
  }
  return status;
}

void loom_runtime_memory_arena_free(loom_runtime_memory_arena_t* arena) {
  if (!arena) {
    return;
  }
  IREE_ASSERT(!arena->ranges.allocation_count);
  IREE_ASSERT(!arena->target.mapping_completion);
  arena->pool->statistics.reserved_bytes -=
      iree_hal_buffer_byte_length(arena->handle);
  iree_hal_buffer_release(arena->handle);
  iree_status_free(arena->target.failure);
  iree_hal_memory_tlsf_deinitialize(&arena->ranges);
  iree_allocator_free(arena->pool->host_allocator, arena);
}

iree_hal_buffer_t* loom_runtime_virtual_buffer_handle(
    const loom_runtime_virtual_buffer_t* buffer) {
  return buffer->handle;
}

iree_device_size_t loom_runtime_virtual_buffer_mapping_size(
    const loom_runtime_virtual_buffer_t* buffer) {
  return buffer->geometry.mapping_size;
}

iree_host_size_t loom_runtime_virtual_buffer_slab_count(
    const loom_runtime_virtual_buffer_t* buffer) {
  return buffer->slab_count;
}

iree_hal_buffer_binding_t loom_runtime_virtual_buffer_backing(
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t slab_index) {
  const loom_runtime_backing_entry_t* entry =
      loom_runtime_backing_directory_find_const(&buffer->backing, slab_index);
  return (iree_hal_buffer_binding_t){
      loom_runtime_backing_pages_buffer((loom_runtime_backing_slot_t){
          entry->physical, entry->physical_offset}),
      entry->physical_offset,
      loom_runtime_memory_slab_length(buffer, slab_index)};
}

loom_runtime_memory_demand_t loom_runtime_virtual_buffer_missing_demand(
    const loom_runtime_virtual_buffer_t* buffer) {
  loom_runtime_memory_demand_t demand = {0};
  const uint64_t regular_count =
      buffer->byte_length / buffer->geometry.mapping_size;
  const uint64_t regular_bytes =
      (regular_count - buffer->backed_regular_count) *
      buffer->geometry.mapping_size;
  switch (buffer->regular_kind) {
    case LOOM_RUNTIME_BACKING_PACKED:
      demand.packed_bytes = regular_bytes;
      demand.packed_count = regular_count - buffer->backed_regular_count;
      break;
    case LOOM_RUNTIME_BACKING_WEIGHT:
      demand.weight_bytes = regular_bytes;
      break;
    default:
      demand.dedicated_bytes = regular_bytes;
      demand.dedicated_cost =
          (regular_count - buffer->backed_regular_count) * buffer->regular_cost;
      break;
  }
  if (regular_count < buffer->slab_count) {
    const loom_runtime_backing_entry_t* tail =
        loom_runtime_backing_directory_find_const(&buffer->backing,
                                                  regular_count);
    if (!tail || (!tail->physical && !tail->in_flight)) {
      loom_runtime_memory_slab_add_demand(&demand, buffer, regular_count);
    }
  }
  return demand;
}

bool loom_runtime_virtual_buffer_is_backed(
    const loom_runtime_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  if (!length) {
    return true;
  }
  const uint64_t begin = offset / buffer->geometry.mapping_size;
  const uint64_t end =
      (offset + length - 1) / buffer->geometry.mapping_size + 1;
  for (uint64_t i = begin; i < end; ++i) {
    const loom_runtime_backing_entry_t* entry =
        loom_runtime_backing_directory_find_const(&buffer->backing, i);
    if (!entry || !entry->physical) {
      return false;
    }
  }
  return true;
}

// Publishes one native result on the serialized pool owner. Native work never
// mutates this accounting or another operation's slab metadata.
static void loom_runtime_memory_slab_publish(
    loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index,
    const loom_runtime_backing_entry_t* result) {
  loom_runtime_memory_pool_t* pool = buffer->pool;
  loom_runtime_backing_entry_t* entry =
      loom_runtime_backing_directory_find(&buffer->backing, index);
  IREE_ASSERT(entry);
  const uint64_t length = loom_runtime_memory_slab_length(buffer, index);
  if ((entry->physical || entry->in_flight) && !result->physical) {
    *loom_runtime_memory_slab_demand(&pool->demand, buffer, index) -= length;
    pool->demand.packed_count -=
        (index + 1 == buffer->slab_count
             ? buffer->tail_kind
             : buffer->regular_kind) == LOOM_RUNTIME_BACKING_PACKED;
    pool->demand.dedicated_cost -= loom_runtime_memory_slab_cost(buffer, index);
    if (length == buffer->geometry.mapping_size) {
      --buffer->backed_regular_count;
    }
  }
  if (entry->physical && !result->physical) {
    pool->statistics.committed_bytes -= length;
    pool->statistics.released_bytes += length;
    buffer->statistics->committed_bytes -= length;
    buffer->statistics->released_bytes += length;
  } else if (!entry->physical && result->physical) {
    pool->statistics.committed_bytes += length;
    pool->statistics.peak_bytes =
        iree_max(pool->statistics.peak_bytes, pool->statistics.committed_bytes);
    buffer->statistics->committed_bytes += length;
    buffer->statistics->peak_bytes = iree_max(
        buffer->statistics->peak_bytes, buffer->statistics->committed_bytes);
  }
  const bool keep = entry->keep;
  *entry = *result;
  entry->index = index;
  entry->occupied = true;
  entry->keep |= keep;
  entry->in_flight = false;
  if (!entry->physical && !entry->keep) {
    loom_runtime_backing_directory_remove(&buffer->backing, entry);
  }
}

static void loom_runtime_memory_pool_collect(loom_runtime_memory_pool_t* pool) {
  loom_runtime_backing_pages_collect(pool->pages);
  iree_slim_mutex_lock(&pool->capacity_mutex);
  iree_status_t capacity_status = pool->capacity_status;
  pool->capacity_status = iree_ok_status();
  iree_slim_mutex_unlock(&pool->capacity_mutex);
  pool->capacity_failed |= !iree_status_is_ok(capacity_status);
  pool->failure = iree_status_join(pool->failure, capacity_status);
  iree_atomic_slist_entry_t* completed = NULL;
  iree_atomic_slist_flush(&pool->completed_operations,
                          IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_LIFO,
                          &completed, NULL);
  while (completed) {
    iree_atomic_slist_entry_t* next = completed->next;
    loom_runtime_memory_operation_t* operation = iree_containerof(
        completed, loom_runtime_memory_operation_t, completed_entry);
    // The completion link is visible before readiness and wake callbacks run.
    // Join the operation mutex once so publication owns the completed result
    // and no callback retains access after the caller observes |published|.
    iree_slim_mutex_lock(&operation->mutex);
    iree_slim_mutex_unlock(&operation->mutex);
    iree_hal_semaphore_t* mapping_completion =
        operation->stages[operation->releasing ? 0 : 1].semaphore;
    if (operation->growing) {
      loom_runtime_virtual_buffer_t* buffer = operation->growing;
      iree_hal_semaphore_release(buffer->target->mapping_completion);
      buffer->target->mapping_completion = NULL;
      if (iree_status_is_ok(operation->status)) {
        const uint64_t previous_length =
            iree_hal_buffer_byte_length(buffer->previous_handle);
        pool->statistics.reserved_bytes -= previous_length;
        buffer->statistics->reserved_bytes -= previous_length;
        iree_hal_buffer_release(buffer->previous_handle);
        buffer->previous_handle = NULL;
      } else {
        buffer->target->failure = iree_status_clone(operation->status);
      }
    }
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      const loom_runtime_memory_edit_t* edit = &operation->edits[i];
      if (operation->growing) {
        loom_runtime_backing_entry_t* entry =
            loom_runtime_backing_directory_find(&edit->buffer->backing,
                                                edit->index);
        IREE_ASSERT(entry);
        entry->in_flight = false;
        continue;
      }
      if (edit->buffer->target->mapping_completion == mapping_completion) {
        iree_hal_semaphore_release(edit->buffer->target->mapping_completion);
        edit->buffer->target->mapping_completion = NULL;
      }
      const uint64_t length =
          loom_runtime_memory_slab_length(edit->buffer, edit->index);
      loom_runtime_backing_entry_t result = edit->result;
      if (!result.mapped) {
        loom_runtime_backing_pages_return((loom_runtime_backing_slot_t){
            result.physical, result.physical_offset});
        result.physical = NULL;
        result.physical_offset = 0;
      }
      loom_runtime_memory_slab_publish(edit->buffer, edit->index, &result);
      if (operation->invalidates_backing &&
          iree_status_is_ok(edit->buffer->target->failure)) {
        edit->buffer->target->failure = iree_status_clone(operation->status);
      }
      if (operation->releasing) {
        edit->buffer->statistics->releasing_bytes -= length;
        pool->statistics.releasing_bytes -= length;
      } else {
        edit->buffer->statistics->pending_bytes -= length;
        pool->statistics.pending_bytes -= length;
      }
    }
    *operation->pending_previous_link = operation->pending_next;
    if (operation->pending_next) {
      operation->pending_next->pending_previous_link =
          operation->pending_previous_link;
    }
    operation->pending_next = NULL;
    operation->pending_previous_link = NULL;
    // Another thread may have flushed this operation and still be publishing
    // while its caller waits. Publish the terminal state under the operation
    // join and wake that caller only after the collector's final access.
    iree_slim_mutex_lock(&operation->mutex);
    operation->published = true;
    iree_notification_post(&operation->notification, IREE_ALL_WAITERS);
    iree_slim_mutex_unlock(&operation->mutex);
    completed = next;
  }
  loom_runtime_memory_pool_observe_retirement(pool);
  if (pool->trim.pending) {
    loom_runtime_memory_pool_trim_cache(pool, pool->trim.target_bytes,
                                        (loom_runtime_memory_demand_t){0});
    pool->trim.pending = pool->statistics.releasing_bytes != 0;
  }
}

static void loom_runtime_memory_operation_complete(
    loom_runtime_memory_operation_t* operation) {
  if (iree_atomic_fetch_sub(&operation->pending, 1,
                            iree_memory_order_acq_rel) != 1) {
    return;
  }
  const bool first_ok = iree_status_is_ok(operation->stages[0].status);
  for (iree_host_size_t i = 0; !operation->growing && i < operation->count;
       ++i) {
    loom_runtime_backing_entry_t* result = &operation->edits[i].result;
    if (!operation->releasing || first_ok) {
      result->mapped = !operation->releasing && first_ok;
    }
  }
  iree_status_t status = iree_status_join(operation->stages[0].status,
                                          operation->stages[1].status);
  operation->stages[0].status = operation->stages[1].status = iree_ok_status();
  operation->invalidates_backing =
      !iree_status_is_ok(status) && (operation->releasing || first_ok);
  // Readiness can wake the owner on another thread. Its completion query must
  // join publication and our last callback access before it can free us.
  iree_slim_mutex_lock(&operation->mutex);
  iree_atomic_slist_push(&operation->pool->completed_operations,
                         &operation->completed_entry);
  if (operation->readiness && iree_status_is_ok(status)) {
    status =
        iree_hal_semaphore_signal(operation->readiness, operation->value, NULL);
  }
  if (operation->readiness && !iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(operation->readiness, iree_status_clone(status));
  }
  operation->status = status;
  operation->complete = true;
  iree_notification_post(&operation->notification, IREE_ALL_WAITERS);
  if (operation->wake) {
    operation->wake(operation->user_data);
  }
  if (operation->observer.wake) {
    operation->observer.wake(operation->observer.user_data);
  }
  iree_slim_mutex_unlock(&operation->mutex);
}

static int loom_runtime_memory_edit_compare(const void* lhs, const void* rhs) {
  const loom_runtime_memory_edit_t* a = lhs;
  const loom_runtime_memory_edit_t* b = rhs;
  if (a->buffer != b->buffer) {
    return (uintptr_t)a->buffer < (uintptr_t)b->buffer ? -1 : 1;
  }
  return (a->index > b->index) - (a->index < b->index);
}

// Native readiness establishes each parent's source view. Consecutive logical
// slots sharing contiguous physical storage become one mapping, while edits
// retain their independent publication, return, and failure ownership.
static iree_host_size_t loom_runtime_memory_operation_build_mappings(
    loom_runtime_memory_operation_t* operation) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    const loom_runtime_memory_edit_t* edit = &operation->edits[i];
    const loom_runtime_backing_entry_t* result = &edit->result;
    const iree_device_size_t length =
        loom_runtime_memory_slab_length(edit->buffer, edit->index);
    const loom_runtime_memory_edit_t* previous =
        i ? &operation->edits[i - 1] : NULL;
    if (previous && previous->buffer == edit->buffer &&
        previous->index + 1 == edit->index &&
        previous->result.physical == result->physical &&
        previous->result.physical_offset +
                loom_runtime_memory_slab_length(previous->buffer,
                                                previous->index) ==
            result->physical_offset) {
      operation->mappings[count - 1].target_range.byte_length += length;
      continue;
    }
    iree_hal_buffer_t* source =
        loom_runtime_backing_pages_buffer((loom_runtime_backing_slot_t){
            result->physical, result->physical_offset});
    operation->mappings[count++] = (iree_hal_memory_mapping_t){
        .target = edit->buffer->handle,
        .scope = operation->pool->mapping_scope,
        .target_range = {edit->index * edit->buffer->geometry.mapping_size,
                         length},
        .source = operation->releasing ? NULL : source,
        .source_offset = operation->releasing ? 0 : result->physical_offset,
        .access = operation->releasing ? IREE_HAL_MEMORY_ACCESS_NONE
                                       : IREE_HAL_MEMORY_ACCESS_ALL,
        .operation = operation->releasing
                         ? IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND
                         : IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
        .previous_source = operation->releasing ? source : NULL,
        .previous_source_offset =
            operation->releasing ? result->physical_offset : 0,
    };
  }
  return count;
}

// Sources are resolved only after their private native acquisition completed.
// Warm claims skip the allocation queue entirely; pending claims wait once per
// parent and continue directly into mapping without creating per-page views.
static void loom_runtime_memory_stage_completed(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_runtime_memory_stage_t* stage = user_data;
  loom_runtime_memory_operation_t* operation = stage->operation;
  stage->status = status;
  if (stage == &operation->stages[0] && !operation->growing) {
    iree_status_t next_status = iree_status_clone(status);
    if (iree_status_is_ok(next_status)) {
      if (operation->releasing) {
        next_status =
            iree_hal_semaphore_signal(operation->stages[1].semaphore, 1, NULL);
      } else {
        const iree_host_size_t mapping_count =
            loom_runtime_memory_operation_build_mappings(operation);
        uint64_t value = 1;
        const iree_hal_semaphore_list_t signal = {
            1, &operation->stages[1].semaphore, &value};
        next_status =
            iree_hal_queue_map(operation->pool->queue, operation->mapping_waits,
                               signal, mapping_count, operation->mappings, 0);
      }
    }
    for (iree_host_size_t i = 0; i < operation->mapping_waits.count; ++i) {
      iree_hal_semaphore_release(operation->mapping_waits.semaphores[i]);
    }
    operation->mapping_waits.count = 0;
    if (!iree_status_is_ok(next_status)) {
      iree_hal_semaphore_fail(operation->stages[1].semaphore, next_status);
    }
  }
  loom_runtime_memory_operation_complete(operation);
}

static iree_status_t loom_runtime_memory_operation_allocate(
    loom_runtime_memory_pool_t* pool, iree_host_size_t count,
    loom_runtime_memory_operation_t** out_operation) {
  *out_operation = NULL;
  loom_runtime_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      pool->host_allocator,
      sizeof(*operation) + sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t),
      count,
      sizeof(loom_runtime_memory_edit_t) + sizeof(iree_hal_memory_mapping_t) +
          2 * (sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t)),
      (void**)&operation));
  operation->pool = pool;
  operation->count = count;
  operation->mappings = (iree_hal_memory_mapping_t*)(operation->edits + count);
  operation->mapping_waits.semaphores =
      (iree_hal_semaphore_t**)(operation->mappings + count);
  operation->mapping_waits.payload_values =
      (uint64_t*)(operation->mapping_waits.semaphores + count + 1);
  operation->allocation_waits.semaphores =
      (iree_hal_semaphore_t**)(operation->mapping_waits.payload_values + count +
                               1);
  operation->allocation_waits.payload_values =
      (uint64_t*)(operation->allocation_waits.semaphores + count);
  iree_slim_mutex_initialize(&operation->mutex);
  iree_notification_initialize(&operation->notification);
  iree_hal_device_t* device =
      iree_hal_queue_family_device(iree_hal_queue_family(pool->queue));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    operation->stages[i].operation = operation;
    operation->stages[i].timepoint.callback =
        loom_runtime_memory_stage_completed;
    operation->stages[i].timepoint.user_data = &operation->stages[i];
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &operation->stages[i].semaphore);
  }
  if (iree_status_is_ok(status)) {
    *out_operation = operation;
  } else {
    operation->published = true;
    loom_runtime_memory_operation_free(operation);
  }
  return status;
}

static iree_status_t loom_runtime_memory_operation_prepare_backing(
    loom_runtime_memory_operation_t* operation,
    loom_runtime_memory_demand_t additional) {
  loom_runtime_memory_pool_t* pool = operation->pool;
  const uint64_t cost = loom_runtime_memory_growth_cost(pool, additional);
  // The serialized owner admitted this complete cohort before capture. Native
  // callbacks can only return capacity while it prepares the accepted work.
  const bool admitted = iree_hal_memory_budget_try_reserve(
      pool->budget, cost, &operation->growth.credit);
  IREE_ASSERT(admitted);
  iree_status_t status = iree_ok_status();
  // Whole-parent reuse is assigned first, matching complete-cohort admission.
  // Packed mappings then use the remaining slots; dedicated extents cannot
  // consume reusable capacity from either class.
  for (int pass = 0; pass < 2 && iree_status_is_ok(status); ++pass) {
    for (iree_host_size_t i = 0;
         i < operation->count && iree_status_is_ok(status); ++i) {
      loom_runtime_memory_edit_t* edit = &operation->edits[i];
      const loom_runtime_backing_kind_t kind =
          edit->index + 1 == edit->buffer->slab_count
              ? edit->buffer->tail_kind
              : edit->buffer->regular_kind;
      if ((kind == LOOM_RUNTIME_BACKING_WEIGHT) != (pass == 0)) {
        continue;
      }
      const iree_hal_pool_reservation_request_t request =
          iree_hal_make_pool_reservation_request(
              loom_runtime_memory_slab_length(edit->buffer, edit->index),
              edit->buffer->geometry.granularity);
      loom_runtime_backing_slot_t slot;
      status = loom_runtime_backing_pages_claim(
          pool->pages, kind, request,
          loom_runtime_memory_slab_cost(edit->buffer, edit->index),
          &operation->growth, &slot);
      edit->result.physical = slot.parent;
      edit->result.physical_offset = slot.offset;
    }
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      loom_runtime_backing_entry_t* result = &operation->edits[i].result;
      loom_runtime_backing_pages_return((loom_runtime_backing_slot_t){
          result->physical, result->physical_offset});
      result->physical = NULL;
      result->physical_offset = 0;
    }
    loom_runtime_backing_pages_abort(&operation->growth);
  }
  return status;
}

static void loom_runtime_memory_operation_enqueue(
    loom_runtime_memory_pool_t* pool,
    loom_runtime_memory_operation_t* operation,
    iree_status_t preparation_status) {
  // Growth and trim capture sparse-directory order and have no indexed backing
  // consumer. Commitments preserve caller order for streamed source I/O.
  if ((operation->growing || operation->releasing) && operation->count > 1) {
    qsort(operation->edits, operation->count, sizeof(operation->edits[0]),
          loom_runtime_memory_edit_compare);
  }
  operation->observer.wake = pool->observer.wake;
  operation->observer.user_data = pool->observer.user_data;
  operation->pending_next = pool->pending_operations;
  operation->pending_previous_link = &pool->pending_operations;
  if (operation->pending_next) {
    operation->pending_next->pending_previous_link = &operation->pending_next;
  }
  pool->pending_operations = operation;
  iree_atomic_store(&operation->pending, 3, iree_memory_order_relaxed);
  uint64_t value = 1;
  const iree_hal_semaphore_list_t first = {1, &operation->stages[0].semaphore,
                                           &value};
  const iree_hal_semaphore_list_t second = {1, &operation->stages[1].semaphore,
                                            &value};
  iree_hal_semaphore_t* mapping_completion =
      operation->stages[operation->releasing ? 0 : 1].semaphore;
  const uint64_t observation =
      loom_runtime_backing_pages_begin_observation(pool->pages);
  if (operation->growing) {
    operation->growing->target->mapping_completion = mapping_completion;
    iree_hal_semaphore_retain(mapping_completion);
  }
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    loom_runtime_memory_edit_t* edit = &operation->edits[i];
    if (!operation->growing &&
        edit->buffer->target->mapping_completion != mapping_completion) {
      if (edit->buffer->target->mapping_completion) {
        const iree_host_size_t index = operation->mapping_waits.count++;
        operation->mapping_waits.semaphores[index] =
            edit->buffer->target->mapping_completion;
        operation->mapping_waits.payload_values[index] = 1;
      }
      edit->buffer->target->mapping_completion = mapping_completion;
      iree_hal_semaphore_retain(mapping_completion);
    }
    if (!operation->growing && !operation->releasing && edit->result.physical) {
      iree_hal_semaphore_t* ready = loom_runtime_backing_pages_observe(
          (loom_runtime_backing_slot_t){edit->result.physical,
                                        edit->result.physical_offset},
          observation);
      if (ready) {
        const iree_host_size_t index = operation->allocation_waits.count++;
        operation->allocation_waits.semaphores[index] = ready;
        operation->allocation_waits.payload_values[index] = 1;
      }
    }
  }
  iree_status_t status = preparation_status;
  for (iree_host_size_t i = 0; i < 2; ++i) {
    loom_runtime_memory_stage_t* stage = &operation->stages[i];
    iree_status_t acquired = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)stage->semaphore, 1, &stage->timepoint);
    if (!iree_status_is_ok(acquired)) {
      status = iree_status_join(status, iree_status_clone(acquired));
      loom_runtime_memory_stage_completed(stage, &stage->timepoint, acquired);
    }
  }
  loom_runtime_backing_pages_submit(&operation->growth);
  iree_host_size_t mapping_count = 0;
  if (iree_status_is_ok(status)) {
    if (operation->releasing || operation->growing) {
      mapping_count = loom_runtime_memory_operation_build_mappings(operation);
      status = iree_hal_queue_map(pool->queue, operation->mapping_waits, first,
                                  mapping_count, operation->mappings, 0);
    } else if (operation->allocation_waits.count) {
      status = iree_hal_queue_barrier(pool->queue, operation->allocation_waits,
                                      first, NULL, 0);
    } else {
      status =
          iree_hal_semaphore_signal(operation->stages[0].semaphore, 1, NULL);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(operation->stages[0].semaphore,
                            iree_status_clone(status));
    iree_hal_semaphore_fail(operation->stages[1].semaphore, status);
  } else if (operation->growing) {
    for (iree_host_size_t i = 0; i < mapping_count; ++i) {
      operation->mappings[i].target = operation->growing->previous_handle;
      operation->mappings[i].previous_source = operation->mappings[i].source;
      operation->mappings[i].previous_source_offset =
          operation->mappings[i].source_offset;
      operation->mappings[i].source = NULL;
      operation->mappings[i].source_offset = 0;
      operation->mappings[i].access = IREE_HAL_MEMORY_ACCESS_NONE;
      operation->mappings[i].operation =
          IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
    }
    status = iree_hal_queue_map(pool->queue, first, second, mapping_count,
                                operation->mappings, 0);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_fail(operation->stages[1].semaphore, status);
    }
  }
  // Non-growing continuations own predecessor references through map capture.
  // Growth has already captured both mapping batches on this owner.
  if (operation->growing) {
    for (iree_host_size_t i = 0; i < operation->mapping_waits.count; ++i) {
      iree_hal_semaphore_release(operation->mapping_waits.semaphores[i]);
    }
    operation->mapping_waits.count = 0;
  }
  loom_runtime_memory_operation_complete(operation);
}

iree_status_t loom_runtime_virtual_buffer_grow_async(
    loom_runtime_virtual_buffer_t* buffer, iree_device_size_t length,
    iree_hal_semaphore_t* readiness, uint64_t value,
    loom_runtime_memory_operation_t** out_operation) {
  *out_operation = NULL;
  // Shared-arena ranges have independent consumers; replacing their parent's
  // address interval is not a single-buffer growth operation.
  IREE_ASSERT(!buffer->arena);
  loom_runtime_memory_pool_t* pool = buffer->pool;
  loom_runtime_memory_pool_collect(pool);
  IREE_RETURN_IF_ERROR(iree_status_clone(buffer->target->failure));
  if (length <= buffer->byte_length) {
    return iree_ok_status();
  }
  const iree_device_size_t slab_size = buffer->geometry.mapping_size;
  if (buffer->byte_length % slab_size || length > INT64_MAX - slab_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "growth requires a whole-slab reservation extent");
  }
  IREE_ASSERT(!buffer->target->mapping_completion && !buffer->previous_handle);
  const iree_device_size_t byte_length = iree_device_align(length, slab_size);
  const uint64_t count = byte_length / slab_size;
  if (count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "virtual buffer has too many physical slots");
  }
  // Preserve the native source-slice contract used by the original
  // reservation; mapping_size remains Loom's logical replacement unit.
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = byte_length,
      .min_alignment = buffer->alignment,
      .mapping_granularity = buffer->geometry.granularity,
  };
  iree_hal_buffer_t* handle = NULL;
  iree_status_t status = iree_hal_virtual_buffer_create(
      pool->group, pool->scope,
      (iree_hal_virtual_buffer_sources_t){1, &pool->storage_pool}, &options,
      pool->host_allocator, &handle);
  loom_runtime_memory_operation_t* operation = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_runtime_memory_operation_allocate(
        pool, buffer->backed_regular_count, &operation);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(handle);
    return status;
  }
  operation->growing = buffer;
  operation->readiness = readiness;
  operation->value = value;
  iree_hal_semaphore_retain(readiness);
  iree_host_size_t edit_count = 0;
  for (iree_host_size_t i = 0; i < buffer->backing.capacity; ++i) {
    loom_runtime_backing_entry_t* entry = &buffer->backing.entries[i];
    if (entry->occupied) {
      IREE_ASSERT(!entry->in_flight);
    }
    if (entry->occupied && entry->physical) {
      operation->edits[edit_count++] =
          (loom_runtime_memory_edit_t){buffer, entry->index, *entry};
      entry->in_flight = true;
    }
  }
  IREE_ASSERT(edit_count == operation->count);
  buffer->previous_handle = buffer->handle;
  buffer->handle = handle;
  buffer->slab_count = (iree_host_size_t)count;
  buffer->byte_length = byte_length;
  buffer->tail_kind = buffer->regular_kind;
  buffer->tail_cost = buffer->regular_cost;
  pool->statistics.reserved_bytes += byte_length;
  buffer->statistics->reserved_bytes += byte_length;
  *out_operation = operation;
  loom_runtime_memory_operation_enqueue(pool, operation, iree_ok_status());
  return iree_ok_status();
}

iree_status_t loom_runtime_memory_pool_commit_async(
    loom_runtime_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_runtime_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_runtime_memory_admission_t* out_admission,
    loom_runtime_memory_operation_t** out_operation) {
  return loom_runtime_memory_pool_commit_with_demand_async(
      pool, range_count, ranges, (loom_runtime_memory_demand_t){0}, readiness,
      value, out_admission, out_operation);
}

iree_status_t loom_runtime_memory_pool_commit_with_demand_async(
    loom_runtime_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_runtime_memory_range_t* ranges,
    loom_runtime_memory_demand_t immediate, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_runtime_memory_admission_t* out_admission,
    loom_runtime_memory_operation_t** out_operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_admission = LOOM_RUNTIME_MEMORY_ADMISSION_CAPACITY;
  *out_operation = NULL;
  loom_runtime_memory_pool_collect(pool);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t r = 0; r < range_count && iree_status_is_ok(status);
       ++r) {
    IREE_ASSERT(!ranges[r].buffer->target->detached);
    status = iree_status_clone(ranges[r].buffer->target->failure);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  iree_host_size_t count = 0;
  loom_runtime_memory_demand_t additional = {0};
  for (iree_host_size_t r = 0; r < range_count && iree_status_is_ok(status);
       ++r) {
    loom_runtime_virtual_buffer_t* buffer = ranges[r].buffer;
    const iree_device_size_t slab_size = buffer->geometry.mapping_size;
    const iree_host_size_t end =
        (ranges[r].offset + ranges[r].length - 1) / slab_size + 1;
    for (iree_host_size_t i = ranges[r].offset / slab_size;
         i < end && iree_status_is_ok(status); ++i) {
      loom_runtime_backing_entry_t* entry = NULL;
      status =
          loom_runtime_backing_directory_insert(&buffer->backing, i, &entry);
      if (!iree_status_is_ok(status)) {
        break;
      }
      IREE_ASSERT(!entry->in_flight);
      if (!entry->physical && !entry->pending_commit) {
        entry->pending_commit = true;
        ++count;
        loom_runtime_memory_slab_add_demand(&additional, buffer, i);
      }
    }
  }
  loom_runtime_memory_demand_t demand = additional;
  loom_runtime_memory_demand_add(&demand, immediate);
  loom_runtime_memory_admission_t admission =
      LOOM_RUNTIME_MEMORY_ADMISSION_CAPACITY;
  if (iree_status_is_ok(status)) {
    status = loom_runtime_memory_pool_prepare(pool, demand, &admission);
  }
  const bool admitted = admission == LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
  loom_runtime_memory_operation_t* operation = NULL;
  if (iree_status_is_ok(status) && admitted && count) {
    status = loom_runtime_memory_operation_allocate(pool, count, &operation);
  }
  iree_host_size_t edit_index = 0;
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    loom_runtime_virtual_buffer_t* buffer = ranges[r].buffer;
    const iree_device_size_t slab_size = buffer->geometry.mapping_size;
    const iree_host_size_t end =
        (ranges[r].offset + ranges[r].length - 1) / slab_size + 1;
    for (iree_host_size_t i = ranges[r].offset / slab_size; i < end; ++i) {
      loom_runtime_backing_entry_t* entry =
          loom_runtime_backing_directory_find(&buffer->backing, i);
      if (!entry || !entry->pending_commit) {
        continue;
      }
      entry->pending_commit = false;
      if (operation) {
        operation->edits[edit_index++] =
            (loom_runtime_memory_edit_t){.buffer = buffer, .index = i};
        entry->in_flight = true;
        const uint64_t length = loom_runtime_memory_slab_length(buffer, i);
        buffer->statistics->pending_bytes += length;
        if (length == buffer->geometry.mapping_size) {
          ++buffer->backed_regular_count;
        }
      } else if (!entry->physical && !entry->keep) {
        loom_runtime_backing_directory_remove(&buffer->backing, entry);
      }
    }
  }
  IREE_ASSERT(!operation || edit_index == count);
  if (!operation && iree_status_is_ok(status)) {
    for (iree_host_size_t r = 0; r < range_count && iree_status_is_ok(status);
         ++r) {
      status =
          loom_runtime_backing_directory_shrink(&ranges[r].buffer->backing);
    }
  }
  if (operation) {
    operation->readiness = readiness;
    operation->value = value;
    iree_hal_semaphore_retain(readiness);
    pool->statistics.pending_bytes +=
        loom_runtime_memory_demand_total(additional);
    loom_runtime_memory_demand_add(&pool->demand, additional);
    *out_operation = operation;
    loom_runtime_memory_operation_enqueue(
        pool, operation,
        loom_runtime_memory_operation_prepare_backing(operation, additional));
  } else if (iree_status_is_ok(status) && admitted && readiness) {
    status = iree_hal_semaphore_signal(readiness, value, NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_admission = admission;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_runtime_memory_pool_trim_async(
    loom_runtime_memory_pool_t* pool, iree_host_size_t buffer_count,
    loom_runtime_virtual_buffer_t* const* buffers,
    iree_hal_semaphore_t* readiness, uint64_t value,
    loom_runtime_memory_operation_t** out_operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_operation = NULL;
  loom_runtime_memory_pool_collect(pool);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t b = 0; b < buffer_count && iree_status_is_ok(status);
       ++b) {
    IREE_ASSERT(!buffers[b]->target->detached);
    status = iree_status_clone(buffers[b]->target->failure);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  iree_host_size_t count = 0;
  for (iree_host_size_t b = 0; b < buffer_count; ++b) {
    for (iree_host_size_t i = 0; i < buffers[b]->backing.capacity; ++i) {
      const loom_runtime_backing_entry_t* entry =
          &buffers[b]->backing.entries[i];
      if (entry->occupied && !entry->keep && entry->physical) {
        IREE_ASSERT(!entry->in_flight);
        ++count;
      }
    }
  }
  for (iree_host_size_t b = 0; b < buffer_count; ++b) {
    loom_runtime_backing_directory_discard_empty(&buffers[b]->backing);
  }
  if (!count) {
    iree_status_t status =
        readiness ? iree_hal_semaphore_signal(readiness, value, NULL)
                  : iree_ok_status();
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  loom_runtime_memory_operation_t* operation = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loom_runtime_memory_operation_allocate(pool, count, &operation));
  operation->count = count;
  operation->releasing = true;
  operation->readiness = readiness;
  operation->value = value;
  iree_hal_semaphore_retain(readiness);
  iree_host_size_t index = 0;
  for (iree_host_size_t b = 0; b < buffer_count; ++b) {
    loom_runtime_virtual_buffer_t* buffer = buffers[b];
    for (iree_host_size_t i = 0; i < buffer->backing.capacity; ++i) {
      loom_runtime_backing_entry_t* entry = &buffer->backing.entries[i];
      if (entry->occupied && !entry->keep && entry->physical) {
        operation->edits[index++] =
            (loom_runtime_memory_edit_t){buffer, entry->index, *entry};
        entry->in_flight = true;
        const uint64_t length =
            loom_runtime_memory_slab_length(buffer, entry->index);
        buffer->statistics->releasing_bytes += length;
        pool->statistics.releasing_bytes += length;
      }
    }
  }
  *out_operation = operation;
  loom_runtime_memory_operation_enqueue(pool, operation, iree_ok_status());
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void loom_runtime_memory_operation_set_observer(
    loom_runtime_memory_operation_t* operation, void (*wake)(void* user_data),
    void* user_data) {
  if (operation) {
    iree_slim_mutex_lock(&operation->mutex);
    operation->wake = wake;
    operation->user_data = user_data;
    iree_slim_mutex_unlock(&operation->mutex);
  }
}

static bool loom_runtime_memory_operation_is_published(void* user_data) {
  loom_runtime_memory_operation_t* operation = user_data;
  iree_slim_mutex_lock(&operation->mutex);
  const bool published = operation->published;
  iree_slim_mutex_unlock(&operation->mutex);
  return published;
}

bool loom_runtime_memory_operation_is_complete(
    loom_runtime_memory_operation_t* operation) {
  if (!operation || loom_runtime_memory_operation_is_published(operation)) {
    return true;
  }
  loom_runtime_memory_pool_collect(operation->pool);
  return loom_runtime_memory_operation_is_published(operation);
}

static bool loom_runtime_memory_operation_is_retired(void* user_data) {
  loom_runtime_memory_operation_t* operation = user_data;
  iree_slim_mutex_lock(&operation->mutex);
  const bool complete = operation->complete;
  iree_slim_mutex_unlock(&operation->mutex);
  return complete;
}

void loom_runtime_memory_operation_await(
    loom_runtime_memory_operation_t* operation) {
  if (operation) {
    iree_notification_await(&operation->notification,
                            loom_runtime_memory_operation_is_retired, operation,
                            iree_infinite_timeout());
    loom_runtime_memory_pool_collect(operation->pool);
    iree_notification_await(&operation->notification,
                            loom_runtime_memory_operation_is_published,
                            operation, iree_infinite_timeout());
  }
}

iree_status_t loom_runtime_memory_operation_query_status(
    const loom_runtime_memory_operation_t* operation) {
  return operation ? iree_status_clone(operation->status) : iree_ok_status();
}

iree_hal_buffer_binding_t loom_runtime_memory_operation_backing(
    const loom_runtime_memory_operation_t* operation,
    iree_host_size_t edit_index) {
  const loom_runtime_backing_entry_t* entry =
      &operation->edits[edit_index].result;
  const loom_runtime_memory_edit_t* edit = &operation->edits[edit_index];
  return (iree_hal_buffer_binding_t){
      loom_runtime_backing_pages_buffer((loom_runtime_backing_slot_t){
          entry->physical, entry->physical_offset}),
      entry->physical_offset,
      loom_runtime_memory_slab_length(edit->buffer, edit->index)};
}

void loom_runtime_memory_operation_free(
    loom_runtime_memory_operation_t* operation) {
  if (operation) {
    IREE_ASSERT(loom_runtime_memory_operation_is_published(operation));
    iree_notification_deinitialize(&operation->notification);
    iree_slim_mutex_deinitialize(&operation->mutex);
    iree_hal_semaphore_release(operation->stages[0].semaphore);
    iree_hal_semaphore_release(operation->stages[1].semaphore);
    iree_hal_semaphore_release(operation->readiness);
    iree_status_free(operation->status);
    iree_allocator_free(operation->pool->host_allocator, operation);
  }
}

iree_status_t loom_runtime_memory_pool_try_commit(
    loom_runtime_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_runtime_memory_range_t* ranges, bool* out_admitted) {
  *out_admitted = false;
  loom_runtime_memory_admission_t admission;
  loom_runtime_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_memory_pool_commit_async(
      pool, range_count, ranges, NULL, 0, &admission, &operation));
  *out_admitted = admission == LOOM_RUNTIME_MEMORY_ADMISSION_AVAILABLE;
  loom_runtime_memory_operation_await(operation);
  iree_status_t status = loom_runtime_memory_operation_query_status(operation);
  loom_runtime_memory_operation_free(operation);
  return status;
}

iree_status_t loom_runtime_virtual_buffer_commit(
    loom_runtime_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  const loom_runtime_memory_range_t range = {buffer, offset, length};
  bool admitted = false;
  IREE_RETURN_IF_ERROR(
      loom_runtime_memory_pool_try_commit(buffer->pool, 1, &range, &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "physical pool cannot back the requested range");
  }
  return iree_ok_status();
}

void loom_runtime_virtual_buffer_begin_trim(
    loom_runtime_virtual_buffer_t* buffer) {
  loom_runtime_backing_directory_discard_empty(&buffer->backing);
  for (iree_host_size_t i = 0; i < buffer->backing.capacity; ++i) {
    if (buffer->backing.entries[i].occupied) {
      buffer->backing.entries[i].keep = false;
    }
  }
}

iree_status_t loom_runtime_virtual_buffer_keep(
    loom_runtime_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, uint64_t* out_added_bytes) {
  if (out_added_bytes) {
    *out_added_bytes = 0;
  }
  if (!length) {
    return iree_ok_status();
  }
  const iree_device_size_t slab_size = buffer->geometry.mapping_size;
  const iree_host_size_t end = (offset + length - 1) / slab_size + 1;
  for (iree_host_size_t i = offset / slab_size; i < end; ++i) {
    loom_runtime_backing_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(
        loom_runtime_backing_directory_insert(&buffer->backing, i, &entry));
    if (!entry->keep) {
      if (out_added_bytes) {
        *out_added_bytes += loom_runtime_memory_slab_length(buffer, i);
      }
      entry->keep = true;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_runtime_virtual_buffer_trim(
    loom_runtime_virtual_buffer_t* buffer) {
  loom_runtime_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_memory_pool_trim_async(
      buffer->pool, 1, &buffer, NULL, 0, &operation));
  loom_runtime_memory_operation_await(operation);
  iree_status_t status = loom_runtime_memory_operation_query_status(operation);
  loom_runtime_memory_operation_free(operation);
  // Synchronous trimming is an explicit outer capacity-reclamation boundary.
  return iree_status_join(status, loom_runtime_memory_pool_await(buffer->pool));
}

iree_status_t loom_runtime_virtual_buffer_free(
    loom_runtime_virtual_buffer_t* buffer) {
  if (!buffer) {
    return iree_ok_status();
  }
  loom_runtime_memory_pool_t* pool = buffer->pool;
  loom_runtime_memory_pool_collect(pool);
  loom_runtime_backing_directory_discard_empty(&buffer->backing);
  if (buffer->arena && !buffer->target->detached) {
    // A range cannot clear the native target's failure by detaching siblings.
    // Its owner completes ordinary asynchronous trim before returning the
    // lease; freeing metadata must never wait on another bank's mapping
    // frontier.
    IREE_RETURN_IF_ERROR(iree_status_clone(buffer->target->failure));
    IREE_ASSERT(!buffer->backing.count);
  }
  if (buffer->previous_handle) {
    IREE_RETURN_IF_ERROR(iree_hal_virtual_buffer_detach(buffer->previous_handle,
                                                        pool->mapping_scope));
    const uint64_t previous_length =
        iree_hal_buffer_byte_length(buffer->previous_handle);
    pool->statistics.reserved_bytes -= previous_length;
    buffer->statistics->reserved_bytes -= previous_length;
    iree_hal_buffer_release(buffer->previous_handle);
    buffer->previous_handle = NULL;
  }
  // Checked removal remains available after a partial failed mapping batch.
  // Sources stay owned if detach fails, so the caller can retry cleanup. A
  // clean, empty reservation has no mapping to remove; in particular its
  // construction failure cleanup needs no native edit.
  if (!buffer->arena &&
      (buffer->backing.count || !iree_status_is_ok(buffer->target->failure))) {
    IREE_RETURN_IF_ERROR(
        iree_hal_virtual_buffer_detach(buffer->handle, pool->mapping_scope));
  }
  for (iree_host_size_t i = 0; i < buffer->backing.capacity; ++i) {
    loom_runtime_backing_entry_t* entry = &buffer->backing.entries[i];
    if (!entry->occupied) {
      continue;
    }
    IREE_ASSERT(!entry->in_flight);
    IREE_ASSERT(entry->physical);
    const iree_host_size_t index = entry->index;
    loom_runtime_backing_pages_return(
        (loom_runtime_backing_slot_t){entry->physical, entry->physical_offset});
    entry->keep = false;
    const loom_runtime_backing_entry_t empty = {0};
    loom_runtime_memory_slab_publish(buffer, index, &empty);
  }
  IREE_ASSERT(!buffer->backing.count);
  iree_hal_buffer_release(buffer->handle);
  if (buffer->arena) {
    iree_hal_memory_tlsf_free(&buffer->arena->ranges, buffer->block_index,
                              NULL);
  } else {
    IREE_ASSERT(!buffer->target->mapping_completion);
    iree_status_free(buffer->target->failure);
    pool->statistics.reserved_bytes -= buffer->byte_length;
  }
  buffer->statistics->reserved_bytes -= buffer->byte_length;
  loom_runtime_backing_directory_deinitialize(&buffer->backing);
  iree_allocator_free(pool->host_allocator, buffer);
  return iree_ok_status();
}
