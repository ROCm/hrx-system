// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/memory.h"

#include "iree/async/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf.h"
#include "iree/hal/pool_wait.h"

struct loom_serve_memory_pool_t {
  // Allocator owning pool, buffer, and slab metadata.
  iree_allocator_t host_allocator;
  // Borrowed sealed group outliving all source pools and virtual buffers.
  iree_hal_device_group_t* group;
  // Retained mapping-source pool, used only by this physical budget.
  iree_hal_pool_t* storage_pool;
  // Borrowed outer cache; the class records own the complete chain.
  iree_hal_pool_t* slab_cache;
  // Immutable physical classes, each retaining only its exact extent.
  struct {
    // Owned cache; null when this class duplicates an earlier class.
    iree_hal_pool_t* cache;
    // Exact native extent and alignment admitted by this cache.
    iree_hal_pool_reservation_request_t request;
    // Physical ceiling; never shrinks while source operations are outstanding.
    uint64_t admitted_bytes;
  } classes[LOOM_SERVE_MEMORY_CLASS_COUNT];
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
  // Physical admission ceiling; UINT64_MAX means unbounded.
  uint64_t limit_bytes;
  // Accounting across all buffers, with physical allocations counted once.
  loom_serve_memory_statistics_t statistics;
  // Published and accepted backing, including failed demand until a joined
  // source observation, partitioned by physical class.
  loom_serve_memory_demand_t demand;
  // Exact native extents admitted since the last joined source observation.
  uint64_t uncached_bytes;
  // Failed accepted demand remains charged until every source operation has
  // published; a disposed prefix cannot masquerade as reusable cached backing.
  loom_serve_memory_demand_t failed_demand;
  // Upper bound on returned backing still charged until native disposal.
  uint64_t retiring_bytes;
  // Explicit physical trim, reapplied until accepted detachments return.
  struct {
    // Total native bytes the caller wishes to retain, including live storage.
    uint64_t target_bytes;
    // Apply once all accepted source operations have published.
    bool pending;
  } trim;
  // Owner-only list of queue operations awaiting result publication.
  loom_serve_memory_operation_t* operations;
  // Owner wake copied into operations and capacity observations.
  struct {
    // Nonblocking callback; detachment joins every accepted copy.
    void (*wake)(void* user_data);
    // Borrowed scheduling owner, never accessed after detachment.
    void* user_data;
  } observer;
  // Borrowed owner of cold capacity reclamation, separate from slab mechanics.
  loom_serve_memory_reclaimer_t reclaimer;
  // Owned shared capacity-observation or reclamation failure.
  iree_status_t failure;
};

typedef struct loom_serve_memory_slab_t {
  // Owned physical handle, or null for uncommitted storage.
  iree_hal_buffer_t* physical;
  // Whether physical is currently mapped into its virtual slot.
  bool mapped;
  // Whether a current retained range requires this slab.
  bool keep;
  // Missing slab selected by the current serialized commitment transaction.
  bool pending_commit;
  // Temporary union mark scoped to one serialized footprint query.
  bool counted;
  // Accepted asynchronous operation excludes another editor of this slot.
  bool in_flight;
} loom_serve_memory_slab_t;

// Mapping serialization and failure are properties of the native reservation,
// including when several independently resident buffers occupy its ranges.
typedef struct loom_serve_memory_target_t {
  // Retained last mapping completion; source allocation remains independent.
  iree_hal_semaphore_t* mapping_completion;
  // Owned incomplete-edit failure requiring checked whole-target detach.
  iree_status_t failure;
  // Terminal checked removal permits physical cleanup, never further execution.
  bool detached;
} loom_serve_memory_target_t;

struct loom_serve_memory_arena_t {
  // Borrowed physical owner outliving the arena and every allocated range.
  loom_serve_memory_pool_t* pool;
  // Owned address interval, independent of physical payload and range views.
  iree_hal_buffer_t* handle;
  // Shared native mapping order and failure domain for every allocated range.
  loom_serve_memory_target_t target;
  // Native geometry governing independent ranges and their physical slabs.
  loom_serve_memory_geometry_t geometry;
  // Address-only allocation metadata; leases return after actual retirement.
  iree_hal_memory_tlsf_t ranges;
};

struct loom_serve_virtual_buffer_t {
  // Borrowed physical owner outliving this reservation.
  loom_serve_memory_pool_t* pool;
  // Borrowed caller accounting group, independent of the shared pool budget.
  loom_serve_memory_statistics_t* statistics;
  // Owned stable virtual reservation or range view, independent of backing.
  iree_hal_buffer_t* handle;
  // Shared native-target state, or owned trailing state for a standalone root.
  loom_serve_memory_target_t* target;
  // Borrowed address allocator, NULL when this buffer owns its reservation.
  loom_serve_memory_arena_t* arena;
  // Address lease returned only after consumers and backing edits retire.
  iree_hal_memory_tlsf_block_index_t block_index;
  // Validated physical geometry, independent of other reservations.
  loom_serve_memory_geometry_t geometry;
  // Native-aligned reservation extent, including its exact short tail.
  iree_device_size_t byte_length;
  // Number of slots spanning the reservation, including an optional short tail.
  iree_host_size_t slab_count;
  // Regular slots with physical or accepted backing; excludes a short tail.
  iree_host_size_t backed_regular_count;
  // Physical class of regular slots, or CLASS_COUNT for exact bypass.
  uint8_t regular_class;
  // Physical class of the last slot, which may be shorter than a regular slab.
  uint8_t tail_class;
  // Physical ownership and retained-range marks for each virtual slot.
  loom_serve_memory_slab_t slabs[];
};

typedef struct loom_serve_memory_edit_t {
  // Borrowed reservation held alive by the operation's caller.
  loom_serve_virtual_buffer_t* buffer;
  // Slab index within buffer.
  iree_host_size_t index;
  // Result written exclusively by the last queue completion before publication.
  loom_serve_memory_slab_t result;
} loom_serve_memory_edit_t;

typedef struct loom_serve_memory_stage_t {
  // Borrowed enclosing transaction through the last completion claim.
  loom_serve_memory_operation_t* operation;
  // Owned private timeline; caller readiness cannot fail accepted ownership.
  iree_hal_semaphore_t* semaphore;
  // Embedded observation retained until this stage completes.
  iree_async_semaphore_timepoint_t timepoint;
  // Owned terminal result, published before dropping the completion claim.
  iree_status_t status;
} loom_serve_memory_stage_t;

struct loom_serve_memory_operation_t {
  // Submission and both private stage callbacks each own one claim.
  iree_atomic_int32_t pending;
  // Allocation/map or unmap/deallocation completion, respectively.
  loom_serve_memory_stage_t stages[2];
  // Batched source buffers in trailing storage; ownership moves into edits.
  iree_hal_buffer_t** sources;
  // Prepared allocation requests in trailing storage.
  iree_hal_pool_reservation_request_t* requests;
  // Captured virtual mapping updates in trailing storage.
  iree_hal_memory_mapping_t* mappings;
  // Owned predecessor mapping references plus a borrowed allocation stage.
  // Trailing storage holds at most one predecessor per edited reservation.
  iree_hal_semaphore_list_t mapping_waits;
  // Borrowed pool outliving the operation and every reservation.
  loom_serve_memory_pool_t* pool;
  // Owner-only link while results still need publication.
  loom_serve_memory_operation_t* next;
  // Readiness signaled after native mapping publication, retained by this job.
  iree_hal_semaphore_t* readiness;
  // Caller-selected readiness payload.
  uint64_t value;
  // Release existing physical ownership instead of constructing new backing.
  bool releasing;
  // Joins completion and observer access before the caller releases the job.
  iree_slim_mutex_t mutex;
  // Outer-thread wait only; shared task workers never wait here.
  iree_notification_t notification;
  // Both queue operations and submission have released captured ownership.
  bool complete;
  // Serialized owner has installed native results and returned budget credit.
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
  loom_serve_memory_edit_t edits[];
};

static void loom_serve_memory_pool_collect(loom_serve_memory_pool_t* pool);

static void loom_serve_memory_capacity_completed(void* user_data,
                                                 iree_status_t status) {
  loom_serve_memory_pool_t* pool = user_data;
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

static bool loom_serve_memory_capacity_is_idle(void* user_data) {
  loom_serve_memory_pool_t* pool = user_data;
  iree_slim_mutex_lock(&pool->capacity_mutex);
  const bool idle = !pool->capacity_pending;
  iree_slim_mutex_unlock(&pool->capacity_mutex);
  return idle;
}

void loom_serve_memory_demand_add(loom_serve_memory_demand_t* demand,
                                  loom_serve_memory_demand_t additional) {
  for (iree_host_size_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
    demand->cached_bytes[i] += additional.cached_bytes[i];
  }
  demand->uncached_bytes += additional.uncached_bytes;
}

static uint64_t loom_serve_memory_demand_total(
    loom_serve_memory_demand_t demand) {
  uint64_t total = demand.uncached_bytes;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
    total += demand.cached_bytes[i];
  }
  return total;
}

// Only this owner can acquire, return, or trim this source. Once all captured
// operations have published, live ownership plus each exact cache's idle
// bytes is stable. Native bytes above it are pending return or permanent loss.
// Concurrent acquisition snapshots cannot safely supply that subtraction.
static void loom_serve_memory_pool_observe_retirement(
    loom_serve_memory_pool_t* pool) {
  if (!pool->operations) {
    for (iree_host_size_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
      pool->demand.cached_bytes[i] -= pool->failed_demand.cached_bytes[i];
    }
    pool->demand.uncached_bytes -= pool->failed_demand.uncached_bytes;
    pool->failed_demand = (loom_serve_memory_demand_t){0};
  }
  uint64_t retained_bytes = pool->demand.uncached_bytes;
  if (!pool->operations) {
    for (iree_host_size_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
      if (!pool->classes[i].cache) {
        continue;
      }
      iree_hal_pool_stats_t cached;
      iree_hal_pool_query_stats(pool->classes[i].cache, &cached);
      pool->classes[i].admitted_bytes =
          pool->demand.cached_bytes[i] + cached.bytes_free;
      retained_bytes += pool->classes[i].admitted_bytes;
    }
    pool->uncached_bytes = pool->demand.uncached_bytes;
  }
  iree_hal_pool_stats_t native;
  iree_hal_pool_query_stats(pool->storage_pool, &native);
  pool->statistics.peak_bytes =
      iree_max(pool->statistics.peak_bytes, native.bytes_committed);
  if (!pool->operations) {
    // Accepted constructors can lose backing before publishing their failure.
    // Until this join, their protected demand already covers that charge.
    pool->statistics.abandoned_bytes = native.bytes_abandoned;
    const uint64_t settled_bytes = retained_bytes + native.bytes_abandoned;
    pool->retiring_bytes = native.bytes_committed > settled_bytes
                               ? native.bytes_committed - settled_bytes
                               : 0;
  }
}

static uint64_t loom_serve_memory_pool_projected_bytes(
    const loom_serve_memory_pool_t* pool,
    loom_serve_memory_demand_t additional) {
  uint64_t total = pool->uncached_bytes + additional.uncached_bytes +
                   pool->retiring_bytes + pool->statistics.abandoned_bytes;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
    total +=
        iree_max(pool->classes[i].admitted_bytes,
                 pool->demand.cached_bytes[i] + additional.cached_bytes[i]);
  }
  return total;
}

// Arms a one-shot capacity wake only while native retirement can make progress.
// Prepare precedes the final query so a concurrent free cannot lose its wake.
// True preserves an observed wait even if its callback races this return: one
// capacity change does not establish that every native release has retired.
static bool loom_serve_memory_pool_observe_capacity(
    loom_serve_memory_pool_t* pool) {
  if (pool->capacity_failed) {
    return false;
  }
  if (!loom_serve_memory_capacity_is_idle(pool)) {
    loom_serve_memory_pool_observe_retirement(pool);
    if (!pool->retiring_bytes) {
      // Pending acquisitions can obscure a concurrent native return. Once
      // their ownership is published, withdraw any obsolete capacity wait;
      // there may be no further native free to signal it.
      iree_slim_mutex_lock(&pool->capacity_mutex);
      pool->capacity_withdrawn = pool->capacity_pending;
      iree_slim_mutex_unlock(&pool->capacity_mutex);
      iree_hal_pool_wait_cancel(pool->capacity_wait);
    }
    return true;
  }
  iree_hal_pool_wait_prepare(pool->capacity_wait);
  loom_serve_memory_pool_observe_retirement(pool);
  if (!pool->retiring_bytes) {
    iree_hal_pool_wait_abort(pool->capacity_wait);
    return false;
  }
  iree_slim_mutex_lock(&pool->capacity_mutex);
  pool->capacity_pending = true;
  iree_slim_mutex_unlock(&pool->capacity_mutex);
  iree_hal_pool_wait_commit(pool->capacity_wait, iree_infinite_timeout(),
                            (iree_hal_pool_wait_callback_t){
                                loom_serve_memory_capacity_completed, pool});
  return true;
}

iree_status_t loom_serve_memory_pool_query_geometry(
    const iree_hal_virtual_buffer_capabilities_t* capabilities,
    iree_hal_pool_t* storage_pool, iree_device_size_t slab_size,
    loom_serve_memory_geometry_t* out_geometry) {
  *out_geometry = (loom_serve_memory_geometry_t){0};
  const iree_hal_memory_mapping_operations_t required =
      (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_BIND) |
      (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND);
  if (!capabilities->max_byte_length ||
      !iree_all_bits_set(capabilities->operations, required)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "elastic backing requires bind and unbind support");
  }
  const iree_hal_pool_mapping_requirements_t requirements =
      iree_hal_pool_query_mapping_requirements(storage_pool);
  const iree_device_size_t minimum =
      iree_max(capabilities->reservation_granularity,
               iree_max(capabilities->mapping_granularity,
                        requirements.minimum_granularity));
  iree_hal_pool_capabilities_t source_capabilities;
  iree_hal_pool_query_capabilities(storage_pool, &source_capabilities);
  const iree_device_size_t minimum_size = iree_device_align(
      iree_max(minimum, source_capabilities.min_allocation_size), minimum);
  if (!slab_size) {
    slab_size = iree_max(minimum_size, requirements.recommended_granularity);
  }
  if (!iree_device_size_is_valid_alignment(slab_size) ||
      slab_size > INT64_MAX || slab_size % minimum) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "slab size must be a power-of-two multiple of %" PRIdsz, minimum);
  }
  if (source_capabilities.max_allocation_size &&
      slab_size > source_capabilities.max_allocation_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "slab size %" PRIdsz
                            " exceeds backing source limit %" PRIdsz,
                            slab_size, source_capabilities.max_allocation_size);
  }
  if (minimum > source_capabilities.max_allocation_alignment) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "backing source cannot satisfy slab alignment");
  }
  if (slab_size < minimum_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "slab size is smaller than source minimum %" PRIdsz,
                            minimum_size);
  }
  *out_geometry = (loom_serve_memory_geometry_t){
      .slab_size = slab_size,
      .granularity = minimum,
      .minimum_size = minimum_size,
  };
  return iree_ok_status();
}

iree_status_t loom_serve_memory_pool_allocate(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_hal_pool_t* storage_pool, iree_hal_queue_t* queue,
    const loom_serve_memory_geometry_t
        geometries[LOOM_SERVE_MEMORY_CLASS_COUNT],
    uint64_t limit_bytes, iree_allocator_t host_allocator,
    loom_serve_memory_pool_t** out_pool) {
  *out_pool = NULL;
  loom_serve_memory_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*pool), scope.family_count,
      sizeof(iree_hal_pool_family_access_t), (void**)&pool));
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
  pool->limit_bytes = limit_bytes ? limit_bytes : UINT64_MAX;
  iree_slim_mutex_initialize(&pool->capacity_mutex);
  iree_notification_initialize(&pool->capacity_notification);
  pool->slab_cache = storage_pool;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < LOOM_SERVE_MEMORY_CLASS_COUNT && iree_status_is_ok(status); ++i) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(geometries[i].slab_size,
                                               geometries[i].granularity);
    if (i &&
        request.allocation_size == pool->classes[0].request.allocation_size &&
        request.min_alignment == pool->classes[0].request.min_alignment) {
      continue;
    }
    pool->classes[i].request = request;
    const iree_hal_slab_cache_options_t cache_options = {
        .slab = request,
        .minimum_cached_size = request.allocation_size,
        .max_count = UINT32_MAX,
        .trace_name = IREE_SVL("loom-serve-physical-slabs"),
    };
    status =
        iree_hal_slab_cache_create(pool->slab_cache, &cache_options,
                                   host_allocator, &pool->classes[i].cache);
    if (iree_status_is_ok(status)) {
      pool->slab_cache = pool->classes[i].cache;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_wait_allocate(pool->slab_cache, host_allocator,
                                         &pool->capacity_wait);
  }
  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    loom_serve_memory_pool_free(pool);
  }
  return status;
}

void loom_serve_memory_pool_free(loom_serve_memory_pool_t* pool) {
  if (!pool) {
    return;
  }
  IREE_ASSERT(!pool->statistics.reserved_bytes &&
              !pool->statistics.committed_bytes && !pool->operations);
  if (!loom_serve_memory_capacity_is_idle(pool)) {
    iree_hal_pool_wait_cancel(pool->capacity_wait);
    iree_notification_await(&pool->capacity_notification,
                            loom_serve_memory_capacity_is_idle, pool,
                            iree_infinite_timeout());
  }
  iree_hal_pool_wait_free(pool->capacity_wait);
  iree_status_free(pool->capacity_status);
  iree_status_free(pool->failure);
  iree_notification_deinitialize(&pool->capacity_notification);
  iree_slim_mutex_deinitialize(&pool->capacity_mutex);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
    iree_hal_pool_release(pool->classes[i].cache);
  }
  iree_hal_pool_release(pool->storage_pool);
  iree_allocator_free(pool->host_allocator, pool);
}

loom_serve_memory_statistics_t loom_serve_memory_pool_statistics(
    loom_serve_memory_pool_t* pool) {
  loom_serve_memory_pool_collect(pool);
  loom_serve_memory_pool_observe_retirement(pool);
  loom_serve_memory_statistics_t statistics = pool->statistics;
  const uint64_t retiring = pool->retiring_bytes;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
    if (!pool->classes[i].cache) {
      continue;
    }
    iree_hal_pool_stats_t cached;
    iree_hal_pool_query_stats(pool->classes[i].cache, &cached);
    statistics.cached_bytes += cached.bytes_free;
  }
  statistics.committed_bytes +=
      retiring + statistics.cached_bytes + statistics.abandoned_bytes;
  statistics.releasing_bytes += retiring;
  return statistics;
}

// Trims only idle capacity not required by this demand. Each cache's total
// includes bypassed entries, so its local floor subtracts only the selected
// class's eviction. Requery before each trim: returning an outer cached slab
// also releases the corresponding inner bypass entry.
static void loom_serve_memory_pool_trim_cache(
    loom_serve_memory_pool_t* pool, uint64_t target_bytes,
    loom_serve_memory_demand_t additional) {
  uint64_t projected = loom_serve_memory_pool_projected_bytes(pool, additional);
  for (iree_host_size_t i = 0;
       i < LOOM_SERVE_MEMORY_CLASS_COUNT && projected > target_bytes; ++i) {
    if (!pool->classes[i].cache) {
      continue;
    }
    const uint64_t required =
        pool->demand.cached_bytes[i] + additional.cached_bytes[i];
    const uint64_t retained = pool->classes[i].admitted_bytes;
    if (retained <= required) {
      continue;
    }
    const uint64_t size = pool->classes[i].request.allocation_size;
    const uint64_t excess = projected - target_bytes;
    const uint64_t removable = retained - required;
    const uint64_t count =
        iree_min(removable / size, excess / size + (excess % size != 0));
    if (!count) {
      continue;
    }
    iree_hal_pool_stats_t cached;
    iree_hal_pool_query_stats(pool->classes[i].cache, &cached);
    const uint64_t bytes = count * size;
    iree_hal_pool_trim(pool->classes[i].cache, IREE_HAL_POOL_TRIM_FLAG_ALL,
                       cached.bytes_committed - bytes);
    projected -= bytes;
  }
  loom_serve_memory_pool_observe_retirement(pool);
}

void loom_serve_memory_pool_trim(loom_serve_memory_pool_t* pool,
                                 uint64_t target_bytes) {
  pool->trim.target_bytes = target_bytes;
  pool->trim.pending = true;
  loom_serve_memory_pool_collect(pool);
}

uint64_t loom_serve_memory_pool_limit(const loom_serve_memory_pool_t* pool) {
  return pool->limit_bytes;
}

uint64_t loom_serve_memory_pool_live_bytes(
    const loom_serve_memory_pool_t* pool) {
  return loom_serve_memory_demand_total(pool->demand);
}

iree_status_t loom_serve_memory_pool_query_status(
    loom_serve_memory_pool_t* pool) {
  loom_serve_memory_pool_collect(pool);
  return iree_status_clone(pool->failure);
}

loom_serve_wake_callback_t loom_serve_memory_pool_set_observer(
    loom_serve_memory_pool_t* pool, void (*wake)(void* user_data),
    void* user_data) {
  iree_slim_mutex_lock(&pool->capacity_mutex);
  const loom_serve_wake_callback_t previous = {pool->observer.wake,
                                               pool->observer.user_data};
  pool->observer.wake = wake;
  pool->observer.user_data = user_data;
  iree_slim_mutex_unlock(&pool->capacity_mutex);
  for (loom_serve_memory_operation_t* operation = pool->operations; operation;
       operation = operation->next) {
    iree_slim_mutex_lock(&operation->mutex);
    operation->observer.wake = wake;
    operation->observer.user_data = user_data;
    iree_slim_mutex_unlock(&operation->mutex);
  }
  return previous;
}

bool loom_serve_memory_pool_is_pending(loom_serve_memory_pool_t* pool) {
  loom_serve_memory_pool_collect(pool);
  if (pool->operations) {
    return true;
  }
  return loom_serve_memory_pool_observe_capacity(pool);
}

iree_status_t loom_serve_memory_pool_await(loom_serve_memory_pool_t* pool) {
  while (loom_serve_memory_pool_is_pending(pool)) {
    if (pool->operations) {
      loom_serve_memory_operation_await(pool->operations);
    } else {
      iree_notification_await(&pool->capacity_notification,
                              loom_serve_memory_capacity_is_idle, pool,
                              iree_infinite_timeout());
    }
  }
  // A callback can finish after the last collection but before observation
  // finds no remaining retirement. Join its published status as well.
  loom_serve_memory_pool_collect(pool);
  return iree_status_clone(pool->failure);
}

void loom_serve_memory_pool_set_reclaimer(
    loom_serve_memory_pool_t* pool, loom_serve_memory_reclaimer_t reclaimer) {
  pool->reclaimer = reclaimer;
}

iree_status_t loom_serve_memory_pool_prepare(
    loom_serve_memory_pool_t* pool, loom_serve_memory_demand_t additional,
    loom_serve_memory_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_MEMORY_ADMISSION_CAPACITY;
  loom_serve_memory_pool_collect(pool);
  IREE_RETURN_IF_ERROR(iree_status_clone(pool->failure));
  const uint64_t additional_bytes = loom_serve_memory_demand_total(additional);
  if (!additional_bytes) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
    return iree_ok_status();
  }
  if (additional_bytes > pool->limit_bytes) {
    return iree_ok_status();
  }
  // Explicit trim can change class ceilings. Resolve it before promising
  // a complete no-yield cohort; collection midway through its streamed commits
  // must not destroy the capacity on which that promise depends.
  if (pool->trim.pending) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_DEFERRED;
    return iree_ok_status();
  }
  if (loom_serve_memory_pool_projected_bytes(pool, additional) >
          pool->limit_bytes &&
      pool->reclaimer.fn) {
    const uint64_t target = pool->limit_bytes - additional_bytes;
    iree_status_t status = pool->reclaimer.fn(
        pool->reclaimer.user_data, target > pool->statistics.pending_bytes
                                       ? target - pool->statistics.pending_bytes
                                       : 0);
    if (!iree_status_is_ok(status)) {
      if (iree_status_is_ok(pool->failure)) {
        pool->failure = iree_status_clone(status);
      }
      return status;
    }
    loom_serve_memory_pool_collect(pool);
    IREE_RETURN_IF_ERROR(iree_status_clone(pool->failure));
  }
  // Another class's idle bytes cannot satisfy this demand. At a joined source
  // epoch they can be returned, but native disposal must finish before those
  // bytes become available for a different physical extent.
  if (!pool->operations && loom_serve_memory_pool_projected_bytes(
                               pool, additional) > pool->limit_bytes) {
    loom_serve_memory_pool_trim_cache(pool, pool->limit_bytes, additional);
  }
  // Reclamation can schedule a physical trim. Resolve that boundary before
  // returning the same no-yield promise as an immediately available cohort.
  bool deferred = pool->trim.pending;
  if (loom_serve_memory_pool_projected_bytes(pool, additional) >
      pool->limit_bytes) {
    deferred |= loom_serve_memory_pool_observe_capacity(pool) ||
                pool->operations != NULL;
  }
  if (!pool->trim.pending && loom_serve_memory_pool_projected_bytes(
                                 pool, additional) <= pool->limit_bytes) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
  } else if (deferred) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_DEFERRED;
  } else if (pool->statistics.abandoned_bytes &&
             loom_serve_memory_pool_projected_bytes(pool, additional) -
                     pool->statistics.abandoned_bytes <=
                 pool->limit_bytes) {
    // No native return can satisfy this attempt. Report loss only when this
    // demand would otherwise fit; surviving capacity remains usable.
    return iree_status_annotate_f(
        iree_hal_pool_query_abandonment_status(pool->storage_pool),
        "physical admission is blocked by %" PRIu64
        " bytes of abandoned backing within a %" PRIu64 " byte limit",
        pool->statistics.abandoned_bytes, pool->limit_bytes);
  }
  return iree_ok_status();
}

// Physical matching is independent of the reservation's role. In particular,
// a bulk root's short tail can reuse an exact sparse-storage slab.
static uint8_t loom_serve_memory_pool_class(
    const loom_serve_memory_pool_t* pool, iree_device_size_t size,
    iree_device_size_t alignment) {
  for (uint8_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
    if (pool->classes[i].cache &&
        size == pool->classes[i].request.allocation_size &&
        alignment <= pool->classes[i].request.min_alignment) {
      return i;
    }
  }
  return LOOM_SERVE_MEMORY_CLASS_COUNT;
}

static iree_device_size_t loom_serve_memory_slab_length(
    const loom_serve_virtual_buffer_t* buffer, iree_host_size_t index) {
  return iree_min(buffer->geometry.slab_size,
                  buffer->byte_length - index * buffer->geometry.slab_size);
}

loom_serve_memory_footprint_t loom_serve_memory_ranges_footprint(
    iree_host_size_t range_count, const loom_serve_memory_range_t* ranges) {
  loom_serve_memory_footprint_t footprint = {0};
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    const loom_serve_memory_range_t* range = &ranges[r];
    const iree_device_size_t slab_size = range->buffer->geometry.slab_size;
    const iree_host_size_t end =
        (range->offset + range->length - 1) / slab_size + 1;
    for (iree_host_size_t i = range->offset / slab_size; i < end; ++i) {
      loom_serve_memory_slab_t* slab = &range->buffer->slabs[i];
      if (!slab->counted) {
        slab->counted = true;
        const uint64_t length = loom_serve_memory_slab_length(range->buffer, i);
        footprint.total_bytes +=
            iree_min(length, UINT64_MAX - footprint.total_bytes);
        if (!slab->physical && !slab->in_flight) {
          footprint.missing_bytes +=
              iree_min(length, UINT64_MAX - footprint.missing_bytes);
        }
      }
    }
  }
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    const loom_serve_memory_range_t* range = &ranges[r];
    const iree_device_size_t slab_size = range->buffer->geometry.slab_size;
    const iree_host_size_t end =
        (range->offset + range->length - 1) / slab_size + 1;
    for (iree_host_size_t i = range->offset / slab_size; i < end; ++i) {
      range->buffer->slabs[i].counted = false;
    }
  }
  return footprint;
}

static uint64_t* loom_serve_memory_slab_demand(
    loom_serve_memory_demand_t* demand,
    const loom_serve_virtual_buffer_t* buffer, iree_host_size_t index) {
  const uint8_t physical_class = index + 1 == buffer->slab_count
                                     ? buffer->tail_class
                                     : buffer->regular_class;
  return physical_class == LOOM_SERVE_MEMORY_CLASS_COUNT
             ? &demand->uncached_bytes
             : &demand->cached_bytes[physical_class];
}

static iree_status_t loom_serve_virtual_buffer_allocate_metadata(
    loom_serve_memory_pool_t* pool, loom_serve_memory_geometry_t geometry,
    iree_device_size_t length, iree_device_size_t alignment,
    loom_serve_memory_statistics_t* statistics,
    loom_serve_memory_arena_t* arena,
    loom_serve_virtual_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (!length || length > INT64_MAX - geometry.slab_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid virtual buffer extent");
  }
  if (!alignment || geometry.granularity % alignment) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "virtual reservation cannot guarantee alignment %" PRIdsz, alignment);
  }
  const uint64_t count = (length + geometry.slab_size - 1) / geometry.slab_size;
  const iree_device_size_t tail = length % geometry.slab_size;
  const iree_device_size_t byte_length =
      length - tail +
      (tail ? iree_device_align(iree_max(tail, geometry.minimum_size),
                                geometry.granularity)
            : 0);
  const iree_host_size_t base_size =
      sizeof(loom_serve_virtual_buffer_t) +
      (arena ? 0 : sizeof(loom_serve_memory_target_t));
  if (count >
      (IREE_HOST_SIZE_MAX - base_size) / sizeof(loom_serve_memory_slab_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "virtual buffer metadata is too large");
  }
  loom_serve_virtual_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator,
      base_size + count * sizeof(loom_serve_memory_slab_t), (void**)&buffer));
  buffer->pool = pool;
  buffer->statistics = statistics;
  buffer->arena = arena;
  buffer->target = arena ? &arena->target
                         : (loom_serve_memory_target_t*)(buffer->slabs + count);
  buffer->geometry = geometry;
  buffer->byte_length = byte_length;
  buffer->slab_count = (iree_host_size_t)count;
  buffer->regular_class = loom_serve_memory_pool_class(pool, geometry.slab_size,
                                                       geometry.granularity);
  buffer->tail_class = loom_serve_memory_pool_class(
      pool, loom_serve_memory_slab_length(buffer, count - 1),
      geometry.granularity);
  *out_buffer = buffer;
  return iree_ok_status();
}

iree_status_t loom_serve_virtual_buffer_allocate(
    loom_serve_memory_pool_t* pool, loom_serve_memory_geometry_t geometry,
    iree_device_size_t length, iree_device_size_t alignment,
    loom_serve_memory_statistics_t* statistics,
    loom_serve_virtual_buffer_t** out_buffer) {
  *out_buffer = NULL;
  loom_serve_virtual_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_allocate_metadata(
      pool, geometry, length, alignment, statistics, NULL, &buffer));
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = buffer->byte_length,
      .min_alignment = alignment,
      .mapping_granularity = geometry.slab_size,
  };
  iree_status_t status =
      iree_hal_virtual_buffer_create(pool->group, pool->scope, &options,
                                     pool->host_allocator, &buffer->handle);
  if (iree_status_is_ok(status)) {
    pool->statistics.reserved_bytes += buffer->byte_length;
    statistics->reserved_bytes += buffer->byte_length;
    *out_buffer = buffer;
  } else {
    iree_allocator_free(pool->host_allocator, buffer);
  }
  return status;
}

iree_status_t loom_serve_memory_arena_allocate(
    loom_serve_memory_pool_t* pool, loom_serve_memory_geometry_t geometry,
    iree_device_size_t length, loom_serve_memory_arena_t** out_arena) {
  *out_arena = NULL;
  if (!length || length > INT64_MAX - geometry.granularity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid virtual arena extent");
  }
  length = iree_device_align(length, geometry.granularity);
  loom_serve_memory_arena_t* arena = NULL;
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
    status =
        iree_hal_virtual_buffer_create(pool->group, pool->scope, &options,
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

iree_hal_buffer_t* loom_serve_memory_arena_handle(
    const loom_serve_memory_arena_t* arena) {
  return arena->handle;
}

iree_status_t loom_serve_virtual_buffer_allocate_in_arena(
    loom_serve_memory_arena_t* arena, iree_device_size_t length,
    iree_device_size_t alignment, loom_serve_memory_statistics_t* statistics,
    loom_serve_virtual_buffer_t** out_buffer) {
  *out_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_status_clone(arena->target.failure));
  IREE_ASSERT(!arena->target.detached);
  loom_serve_virtual_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_virtual_buffer_allocate_metadata(
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
    iree_allocator_free(arena->pool->host_allocator, buffer);
  }
  return status;
}

iree_status_t loom_serve_memory_arena_detach(loom_serve_memory_arena_t* arena) {
  loom_serve_memory_pool_collect(arena->pool);
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

void loom_serve_memory_arena_free(loom_serve_memory_arena_t* arena) {
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

iree_hal_buffer_t* loom_serve_virtual_buffer_handle(
    const loom_serve_virtual_buffer_t* buffer) {
  return buffer->handle;
}

iree_device_size_t loom_serve_virtual_buffer_slab_size(
    const loom_serve_virtual_buffer_t* buffer) {
  return buffer->geometry.slab_size;
}

iree_host_size_t loom_serve_virtual_buffer_slab_count(
    const loom_serve_virtual_buffer_t* buffer) {
  return buffer->slab_count;
}

loom_serve_memory_demand_t loom_serve_virtual_buffer_missing_demand(
    const loom_serve_virtual_buffer_t* buffer) {
  loom_serve_memory_demand_t demand = {0};
  const uint64_t regular_count =
      buffer->byte_length / buffer->geometry.slab_size;
  const uint64_t regular_bytes =
      (regular_count - buffer->backed_regular_count) *
      buffer->geometry.slab_size;
  if (buffer->regular_class == LOOM_SERVE_MEMORY_CLASS_COUNT) {
    demand.uncached_bytes = regular_bytes;
  } else {
    demand.cached_bytes[buffer->regular_class] = regular_bytes;
  }
  if (regular_count < buffer->slab_count) {
    const loom_serve_memory_slab_t* tail = &buffer->slabs[regular_count];
    if (!tail->physical && !tail->in_flight) {
      *loom_serve_memory_slab_demand(&demand, buffer, regular_count) +=
          loom_serve_memory_slab_length(buffer, regular_count);
    }
  }
  return demand;
}

// Publishes one native result on the serialized pool owner. Native work never
// mutates this accounting or another operation's slab metadata.
static void loom_serve_memory_slab_publish(
    loom_serve_virtual_buffer_t* buffer, iree_host_size_t index,
    const loom_serve_memory_slab_t* result) {
  loom_serve_memory_pool_t* pool = buffer->pool;
  loom_serve_memory_slab_t* slab = &buffer->slabs[index];
  const uint64_t length = loom_serve_memory_slab_length(buffer, index);
  if ((slab->physical || slab->in_flight) && !result->physical) {
    *loom_serve_memory_slab_demand(&pool->demand, buffer, index) -= length;
    if (length == buffer->geometry.slab_size) {
      --buffer->backed_regular_count;
    }
  }
  if (slab->physical && !result->physical) {
    pool->statistics.committed_bytes -= length;
    pool->statistics.released_bytes += length;
    buffer->statistics->committed_bytes -= length;
    buffer->statistics->released_bytes += length;
  } else if (!slab->physical && result->physical) {
    pool->statistics.committed_bytes += length;
    pool->statistics.peak_bytes =
        iree_max(pool->statistics.peak_bytes, pool->statistics.committed_bytes);
    buffer->statistics->committed_bytes += length;
    buffer->statistics->peak_bytes = iree_max(
        buffer->statistics->peak_bytes, buffer->statistics->committed_bytes);
  }
  slab->physical = result->physical;
  slab->mapped = result->mapped;
  slab->in_flight = false;
}

static void loom_serve_memory_pool_collect(loom_serve_memory_pool_t* pool) {
  iree_slim_mutex_lock(&pool->capacity_mutex);
  iree_status_t capacity_status = pool->capacity_status;
  pool->capacity_status = iree_ok_status();
  iree_slim_mutex_unlock(&pool->capacity_mutex);
  pool->capacity_failed |= !iree_status_is_ok(capacity_status);
  pool->failure = iree_status_join(pool->failure, capacity_status);
  loom_serve_memory_operation_t** link = &pool->operations;
  while (*link) {
    loom_serve_memory_operation_t* operation = *link;
    iree_slim_mutex_lock(&operation->mutex);
    const bool complete = operation->complete;
    iree_slim_mutex_unlock(&operation->mutex);
    if (!complete) {
      link = &operation->next;
      continue;
    }
    iree_hal_semaphore_t* mapping_completion =
        operation->stages[operation->releasing ? 0 : 1].semaphore;
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      const loom_serve_memory_edit_t* edit = &operation->edits[i];
      if (edit->buffer->target->mapping_completion == mapping_completion) {
        iree_hal_semaphore_release(edit->buffer->target->mapping_completion);
        edit->buffer->target->mapping_completion = NULL;
      }
      const uint64_t length =
          loom_serve_memory_slab_length(edit->buffer, edit->index);
      loom_serve_memory_slab_publish(edit->buffer, edit->index, &edit->result);
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
        if (!edit->result.physical) {
          // Publication drops root ownership, but not accepted physical charge.
          // A failed batch can leave cached growth or asynchronous native
          // disposal. Retain its demand until all source work has published.
          *loom_serve_memory_slab_demand(&pool->demand, edit->buffer,
                                         edit->index) += length;
          *loom_serve_memory_slab_demand(&pool->failed_demand, edit->buffer,
                                         edit->index) += length;
        }
      }
    }
    operation->published = true;
    *link = operation->next;
    operation->next = NULL;
  }
  if (!pool->operations) {
    loom_serve_memory_pool_observe_retirement(pool);
    if (pool->trim.pending) {
      loom_serve_memory_pool_trim_cache(pool, pool->trim.target_bytes,
                                        (loom_serve_memory_demand_t){0});
      pool->trim.pending = false;
    }
  }
}

static void loom_serve_memory_operation_complete(
    loom_serve_memory_operation_t* operation) {
  if (iree_atomic_fetch_sub(&operation->pending, 1,
                            iree_memory_order_acq_rel) != 1) {
    return;
  }
  const bool first_ok = iree_status_is_ok(operation->stages[0].status);
  const bool second_ok = iree_status_is_ok(operation->stages[1].status);
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    loom_serve_memory_slab_t* result = &operation->edits[i].result;
    if (operation->releasing) {
      if (first_ok) {
        result->mapped = false;
      }
      if (second_ok) {
        iree_hal_buffer_release(operation->sources[i]);
        result->physical = NULL;
      }
    } else if (first_ok) {
      result->physical = operation->sources[i];
      // Failed batches can have installed a subset. Checked whole-target
      // detach owns recovery after all consumers and captured work retire.
      result->mapped = true;
    } else {
      iree_hal_buffer_release(operation->sources[i]);
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

static void loom_serve_memory_stage_completed(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  loom_serve_memory_stage_t* stage = user_data;
  stage->status = status;
  loom_serve_memory_operation_complete(stage->operation);
}

static iree_status_t loom_serve_memory_operation_allocate(
    loom_serve_memory_pool_t* pool, iree_host_size_t count,
    loom_serve_memory_operation_t** out_operation) {
  *out_operation = NULL;
  loom_serve_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      pool->host_allocator,
      sizeof(*operation) + sizeof(iree_hal_semaphore_t*) + sizeof(uint64_t),
      count,
      sizeof(loom_serve_memory_edit_t) + sizeof(iree_hal_buffer_t*) +
          sizeof(iree_hal_pool_reservation_request_t) +
          sizeof(iree_hal_memory_mapping_t) + sizeof(iree_hal_semaphore_t*) +
          sizeof(uint64_t),
      (void**)&operation));
  operation->pool = pool;
  operation->count = count;
  operation->sources = (iree_hal_buffer_t**)(operation->edits + count);
  operation->requests =
      (iree_hal_pool_reservation_request_t*)(operation->sources + count);
  operation->mappings =
      (iree_hal_memory_mapping_t*)(operation->requests + count);
  operation->mapping_waits.semaphores =
      (iree_hal_semaphore_t**)(operation->mappings + count);
  operation->mapping_waits.payload_values =
      (uint64_t*)(operation->mapping_waits.semaphores + count + 1);
  iree_slim_mutex_initialize(&operation->mutex);
  iree_notification_initialize(&operation->notification);
  iree_hal_device_t* device =
      iree_hal_queue_family_device(iree_hal_queue_family(pool->queue));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    operation->stages[i].operation = operation;
    operation->stages[i].timepoint.callback = loom_serve_memory_stage_completed;
    operation->stages[i].timepoint.user_data = &operation->stages[i];
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &operation->stages[i].semaphore);
  }
  if (iree_status_is_ok(status)) {
    *out_operation = operation;
  } else {
    operation->published = true;
    loom_serve_memory_operation_free(operation);
  }
  return status;
}

static void loom_serve_memory_operation_enqueue(
    loom_serve_memory_pool_t* pool, loom_serve_memory_operation_t* operation) {
  operation->observer.wake = pool->observer.wake;
  operation->observer.user_data = pool->observer.user_data;
  operation->next = pool->operations;
  pool->operations = operation;
  iree_atomic_store(&operation->pending, 3, iree_memory_order_relaxed);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 2; ++i) {
    loom_serve_memory_stage_t* stage = &operation->stages[i];
    iree_status_t acquired = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)stage->semaphore, 1, &stage->timepoint);
    if (!iree_status_is_ok(acquired)) {
      status = iree_status_join(status, iree_status_clone(acquired));
      loom_serve_memory_stage_completed(stage, &stage->timepoint, acquired);
    }
  }
  uint64_t value = 1;
  const iree_hal_semaphore_list_t first = {1, &operation->stages[0].semaphore,
                                           &value};
  const iree_hal_semaphore_list_t second = {1, &operation->stages[1].semaphore,
                                            &value};
  iree_hal_semaphore_t* mapping_completion =
      operation->stages[operation->releasing ? 0 : 1].semaphore;
  if (!operation->releasing) {
    operation->mapping_waits.semaphores[0] = first.semaphores[0];
    operation->mapping_waits.payload_values[0] = 1;
    operation->mapping_waits.count = 1;
  }
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    loom_serve_memory_edit_t* edit = &operation->edits[i];
    if (edit->buffer->target->mapping_completion != mapping_completion) {
      if (edit->buffer->target->mapping_completion) {
        // Transfer the reservation's predecessor reference until queue_map
        // captures it. Repeated slabs in this batch share one dependency.
        const iree_host_size_t index = operation->mapping_waits.count++;
        operation->mapping_waits.semaphores[index] =
            edit->buffer->target->mapping_completion;
        operation->mapping_waits.payload_values[index] = 1;
      }
      edit->buffer->target->mapping_completion = mapping_completion;
      iree_hal_semaphore_retain(mapping_completion);
    }
    const iree_device_size_t length =
        loom_serve_memory_slab_length(edit->buffer, edit->index);
    operation->requests[i] = iree_hal_make_pool_reservation_request(
        length, edit->buffer->geometry.granularity);
    if (operation->releasing) {
      operation->sources[i] = edit->result.physical;
    }
    operation->mappings[i] = (iree_hal_memory_mapping_t){
        .target = edit->buffer->handle,
        .scope = pool->mapping_scope,
        .target_range = {edit->index * edit->buffer->geometry.slab_size,
                         length},
        .access = operation->releasing ? IREE_HAL_MEMORY_ACCESS_NONE
                                       : IREE_HAL_MEMORY_ACCESS_ALL,
        .operation = operation->releasing
                         ? IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND
                         : IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  if (iree_status_is_ok(status)) {
    status =
        operation->releasing
            ? iree_hal_queue_map(pool->queue, operation->mapping_waits, first,
                                 operation->count, operation->mappings, 0)
            : iree_hal_queue_alloca(pool->queue,
                                    iree_hal_semaphore_list_empty(), first,
                                    pool->slab_cache, operation->count,
                                    operation->requests, operation->sources);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(operation->stages[0].semaphore,
                            iree_status_clone(status));
  } else if (operation->releasing) {
    status = iree_hal_queue_dealloca(pool->queue, first, second,
                                     operation->count, operation->sources);
  } else {
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      operation->mappings[i].source = operation->sources[i];
    }
    status = iree_hal_queue_map(pool->queue, operation->mapping_waits, second,
                                operation->count, operation->mappings, 0);
  }
  for (iree_host_size_t i = operation->releasing ? 0 : 1;
       i < operation->mapping_waits.count; ++i) {
    iree_hal_semaphore_release(operation->mapping_waits.semaphores[i]);
  }
  operation->mapping_waits.count = 0;
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(operation->stages[1].semaphore, status);
  }
  // Final handoff: an inline completion can now publish and release capture.
  loom_serve_memory_operation_complete(operation);
}

iree_status_t loom_serve_memory_pool_commit_async(
    loom_serve_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_serve_memory_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation) {
  return loom_serve_memory_pool_commit_with_demand_async(
      pool, range_count, ranges, (loom_serve_memory_demand_t){0}, readiness,
      value, out_admission, out_operation);
}

iree_status_t loom_serve_memory_pool_commit_with_demand_async(
    loom_serve_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges,
    loom_serve_memory_demand_t immediate, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_serve_memory_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_admission = LOOM_SERVE_MEMORY_ADMISSION_CAPACITY;
  *out_operation = NULL;
  loom_serve_memory_pool_collect(pool);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t r = 0; r < range_count && iree_status_is_ok(status);
       ++r) {
    IREE_ASSERT(!ranges[r].buffer->target->detached);
    status = iree_status_clone(ranges[r].buffer->target->failure);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  iree_host_size_t count = 0;
  loom_serve_memory_demand_t additional = {0};
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    const loom_serve_memory_range_t* range = &ranges[r];
    const iree_device_size_t slab_size = range->buffer->geometry.slab_size;
    const iree_host_size_t end =
        (range->offset + range->length - 1) / slab_size + 1;
    for (iree_host_size_t i = range->offset / slab_size; i < end; ++i) {
      loom_serve_memory_slab_t* slab = &range->buffer->slabs[i];
      IREE_ASSERT(!slab->in_flight);
      if (!slab->physical && !slab->pending_commit) {
        slab->pending_commit = true;
        ++count;
        *loom_serve_memory_slab_demand(&additional, range->buffer, i) +=
            loom_serve_memory_slab_length(range->buffer, i);
      }
    }
  }
  loom_serve_memory_demand_t demand = additional;
  loom_serve_memory_demand_add(&demand, immediate);
  loom_serve_memory_admission_t admission;
  status = loom_serve_memory_pool_prepare(pool, demand, &admission);
  const bool admitted = admission == LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
  loom_serve_memory_operation_t* operation = NULL;
  if (iree_status_is_ok(status) && admitted && count) {
    status = loom_serve_memory_operation_allocate(pool, count, &operation);
  }
  iree_host_size_t index = 0;
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    const loom_serve_memory_range_t* range = &ranges[r];
    const iree_device_size_t slab_size = range->buffer->geometry.slab_size;
    const iree_host_size_t end =
        (range->offset + range->length - 1) / slab_size + 1;
    for (iree_host_size_t i = range->offset / slab_size; i < end; ++i) {
      loom_serve_memory_slab_t* slab = &range->buffer->slabs[i];
      if (slab->pending_commit) {
        slab->pending_commit = false;
        if (operation) {
          operation->edits[index++] = (loom_serve_memory_edit_t){
              .buffer = range->buffer,
              .index = i,
          };
          slab->in_flight = true;
          const uint64_t length =
              loom_serve_memory_slab_length(range->buffer, i);
          range->buffer->statistics->pending_bytes += length;
          if (length == range->buffer->geometry.slab_size) {
            ++range->buffer->backed_regular_count;
          }
        }
      }
    }
  }
  if (operation) {
    operation->count = count;
    operation->readiness = readiness;
    operation->value = value;
    iree_hal_semaphore_retain(readiness);
    pool->statistics.pending_bytes +=
        loom_serve_memory_demand_total(additional);
    loom_serve_memory_demand_add(&pool->demand, additional);
    pool->uncached_bytes += additional.uncached_bytes;
    for (iree_host_size_t i = 0; i < LOOM_SERVE_MEMORY_CLASS_COUNT; ++i) {
      pool->classes[i].admitted_bytes = iree_max(
          pool->classes[i].admitted_bytes, pool->demand.cached_bytes[i]);
    }
    *out_operation = operation;
    loom_serve_memory_operation_enqueue(pool, operation);
  } else if (iree_status_is_ok(status) && admitted && readiness) {
    status = iree_hal_semaphore_signal(readiness, value, NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_admission = admission;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t loom_serve_memory_pool_trim_async(
    loom_serve_memory_pool_t* pool, iree_host_size_t buffer_count,
    loom_serve_virtual_buffer_t* const* buffers,
    iree_hal_semaphore_t* readiness, uint64_t value,
    loom_serve_memory_operation_t** out_operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_operation = NULL;
  loom_serve_memory_pool_collect(pool);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t b = 0; b < buffer_count && iree_status_is_ok(status);
       ++b) {
    IREE_ASSERT(!buffers[b]->target->detached);
    status = iree_status_clone(buffers[b]->target->failure);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  iree_host_size_t count = 0;
  for (iree_host_size_t b = 0; b < buffer_count; ++b) {
    for (iree_host_size_t i = 0; i < buffers[b]->slab_count; ++i) {
      const loom_serve_memory_slab_t* slab = &buffers[b]->slabs[i];
      if (!slab->keep && slab->physical) {
        IREE_ASSERT(!slab->in_flight);
        ++count;
      }
    }
  }
  if (!count) {
    iree_status_t status =
        readiness ? iree_hal_semaphore_signal(readiness, value, NULL)
                  : iree_ok_status();
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  loom_serve_memory_operation_t* operation = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, loom_serve_memory_operation_allocate(pool, count, &operation));
  operation->count = count;
  operation->releasing = true;
  operation->readiness = readiness;
  operation->value = value;
  iree_hal_semaphore_retain(readiness);
  iree_host_size_t index = 0;
  for (iree_host_size_t b = 0; b < buffer_count; ++b) {
    loom_serve_virtual_buffer_t* buffer = buffers[b];
    for (iree_host_size_t i = 0; i < buffer->slab_count; ++i) {
      loom_serve_memory_slab_t* slab = &buffer->slabs[i];
      if (!slab->keep && slab->physical) {
        operation->edits[index++] =
            (loom_serve_memory_edit_t){buffer, i, *slab};
        slab->in_flight = true;
        const uint64_t length = loom_serve_memory_slab_length(buffer, i);
        buffer->statistics->releasing_bytes += length;
        pool->statistics.releasing_bytes += length;
      }
    }
  }
  *out_operation = operation;
  loom_serve_memory_operation_enqueue(pool, operation);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void loom_serve_memory_operation_set_observer(
    loom_serve_memory_operation_t* operation, void (*wake)(void* user_data),
    void* user_data) {
  if (operation) {
    iree_slim_mutex_lock(&operation->mutex);
    operation->wake = wake;
    operation->user_data = user_data;
    iree_slim_mutex_unlock(&operation->mutex);
  }
}

bool loom_serve_memory_operation_is_complete(
    loom_serve_memory_operation_t* operation) {
  if (!operation || operation->published) {
    return true;
  }
  loom_serve_memory_pool_collect(operation->pool);
  return operation->published;
}

static bool loom_serve_memory_operation_is_retired(void* user_data) {
  loom_serve_memory_operation_t* operation = user_data;
  iree_slim_mutex_lock(&operation->mutex);
  const bool complete = operation->complete;
  iree_slim_mutex_unlock(&operation->mutex);
  return complete;
}

void loom_serve_memory_operation_await(
    loom_serve_memory_operation_t* operation) {
  if (operation) {
    iree_notification_await(&operation->notification,
                            loom_serve_memory_operation_is_retired, operation,
                            iree_infinite_timeout());
    loom_serve_memory_pool_collect(operation->pool);
  }
}

iree_status_t loom_serve_memory_operation_query_status(
    const loom_serve_memory_operation_t* operation) {
  return operation ? iree_status_clone(operation->status) : iree_ok_status();
}

void loom_serve_memory_operation_free(
    loom_serve_memory_operation_t* operation) {
  if (operation) {
    IREE_ASSERT(operation->published);
    iree_notification_deinitialize(&operation->notification);
    iree_slim_mutex_deinitialize(&operation->mutex);
    iree_hal_semaphore_release(operation->stages[0].semaphore);
    iree_hal_semaphore_release(operation->stages[1].semaphore);
    iree_hal_semaphore_release(operation->readiness);
    iree_status_free(operation->status);
    iree_allocator_free(operation->pool->host_allocator, operation);
  }
}

iree_status_t loom_serve_memory_pool_try_commit(
    loom_serve_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, bool* out_admitted) {
  *out_admitted = false;
  loom_serve_memory_admission_t admission;
  loom_serve_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_memory_pool_commit_async(
      pool, range_count, ranges, NULL, 0, &admission, &operation));
  *out_admitted = admission == LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
  loom_serve_memory_operation_await(operation);
  iree_status_t status = loom_serve_memory_operation_query_status(operation);
  loom_serve_memory_operation_free(operation);
  return status;
}

iree_status_t loom_serve_virtual_buffer_commit(
    loom_serve_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  const loom_serve_memory_range_t range = {buffer, offset, length};
  bool admitted = false;
  IREE_RETURN_IF_ERROR(
      loom_serve_memory_pool_try_commit(buffer->pool, 1, &range, &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "physical pool cannot back the requested range");
  }
  return iree_ok_status();
}

void loom_serve_virtual_buffer_begin_trim(loom_serve_virtual_buffer_t* buffer) {
  for (iree_host_size_t i = 0; i < buffer->slab_count; ++i) {
    buffer->slabs[i].keep = false;
  }
}

void loom_serve_virtual_buffer_keep(loom_serve_virtual_buffer_t* buffer,
                                    iree_device_size_t offset,
                                    iree_device_size_t length) {
  const iree_device_size_t slab_size = buffer->geometry.slab_size;
  const iree_host_size_t end = (offset + length - 1) / slab_size + 1;
  for (iree_host_size_t i = offset / slab_size; i < end; ++i) {
    buffer->slabs[i].keep = true;
  }
}

iree_status_t loom_serve_virtual_buffer_trim(
    loom_serve_virtual_buffer_t* buffer) {
  loom_serve_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_memory_pool_trim_async(
      buffer->pool, 1, &buffer, NULL, 0, &operation));
  loom_serve_memory_operation_await(operation);
  iree_status_t status = loom_serve_memory_operation_query_status(operation);
  loom_serve_memory_operation_free(operation);
  // Synchronous trimming is an explicit outer capacity-reclamation boundary.
  return iree_status_join(status, loom_serve_memory_pool_await(buffer->pool));
}

iree_status_t loom_serve_virtual_buffer_free(
    loom_serve_virtual_buffer_t* buffer) {
  if (!buffer) {
    return iree_ok_status();
  }
  loom_serve_memory_pool_t* pool = buffer->pool;
  loom_serve_memory_pool_collect(pool);
  if (buffer->arena && !buffer->target->detached) {
    // A range cannot clear the native target's failure by detaching siblings.
    // Healthy range cleanup uses the same ordered unmap path as eviction.
    IREE_RETURN_IF_ERROR(iree_status_clone(buffer->target->failure));
    loom_serve_virtual_buffer_begin_trim(buffer);
    loom_serve_memory_operation_t* operation = NULL;
    IREE_RETURN_IF_ERROR(loom_serve_memory_pool_trim_async(
        pool, 1, &buffer, NULL, 0, &operation));
    loom_serve_memory_operation_await(operation);
    iree_status_t status = loom_serve_memory_operation_query_status(operation);
    loom_serve_memory_operation_free(operation);
    IREE_RETURN_IF_ERROR(status);
  }
  // Checked removal remains available after a partial failed mapping batch.
  // Sources stay owned if detach fails, so the caller can retry cleanup. A
  // clean, empty reservation has no mapping to remove; in particular its
  // construction failure cleanup needs no native edit.
  if (!buffer->arena && (buffer->backed_regular_count ||
                         buffer->slabs[buffer->slab_count - 1].physical ||
                         !iree_status_is_ok(buffer->target->failure))) {
    IREE_RETURN_IF_ERROR(
        iree_hal_virtual_buffer_detach(buffer->handle, pool->mapping_scope));
  }
  for (iree_host_size_t i = 0; i < buffer->slab_count; ++i) {
    IREE_ASSERT(!buffer->slabs[i].in_flight);
    iree_hal_buffer_release(buffer->slabs[i].physical);
    const loom_serve_memory_slab_t empty = {0};
    loom_serve_memory_slab_publish(buffer, i, &empty);
  }
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
  iree_allocator_free(pool->host_allocator, buffer);
  return iree_ok_status();
}
