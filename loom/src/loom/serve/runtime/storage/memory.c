// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/memory.h"

#include "iree/async/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/pool_wait.h"

struct loom_serve_memory_pool_t {
  // Allocator owning pool, buffer, and slab metadata.
  iree_allocator_t host_allocator;
  // Borrowed sealed group outliving all source pools and virtual buffers.
  iree_hal_device_group_t* group;
  // Retained mapping-source pool, used only by this physical budget.
  iree_hal_pool_t* storage_pool;
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
  // Common independent physical allocation size in bytes.
  iree_device_size_t slab_size;
  // Minimum address alignment guaranteed by virtual reservations.
  iree_device_size_t minimum_alignment;
  // Physical admission ceiling; UINT64_MAX means unbounded.
  uint64_t limit_bytes;
  // Accounting across all buffers, with physical allocations counted once.
  loom_serve_memory_statistics_t statistics;
  // Pending mapping bytes whose allocation completion is already published.
  uint64_t acquired_pending_bytes;
  // Upper bound on returned backing still charged until native disposal.
  uint64_t retiring_bytes;
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
  // Accepted asynchronous operation excludes another editor of this slot.
  bool in_flight;
} loom_serve_memory_slab_t;

struct loom_serve_virtual_buffer_t {
  // Borrowed physical owner outliving this reservation.
  loom_serve_memory_pool_t* pool;
  // Borrowed caller accounting group, independent of the shared pool budget.
  loom_serve_memory_statistics_t* statistics;
  // Owned stable virtual reservation, independent of physical commitment.
  iree_hal_buffer_t* handle;
  // Owned incomplete-edit failure; only checked detach can retire this target.
  iree_status_t failure;
  // Number of equally sized slots spanning the reservation.
  iree_host_size_t slab_count;
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
  // Allocation completion published once; immutable after becoming true.
  iree_atomic_int32_t acquired;
  // The serialized owner has included this allocation in known-live backing.
  bool acquired_accounted;
  // Allocation/map or unmap/deallocation completion, respectively.
  loom_serve_memory_stage_t stages[2];
  // Batched source buffers in trailing storage; ownership moves into edits.
  iree_hal_buffer_t** sources;
  // Prepared allocation requests in trailing storage.
  iree_hal_pool_reservation_request_t* requests;
  // Captured virtual mapping updates in trailing storage.
  iree_hal_memory_mapping_t* mappings;
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

// Native allocation becomes visible before its queue completion. Only known
// returns can consume retirement credit; newly allocated pending bytes cannot
// create it. In-flight returns may already have freed their native payload and
// therefore do not establish a lower bound on still-owned native bytes.
static uint64_t loom_serve_memory_pool_observe_retirement(
    loom_serve_memory_pool_t* pool) {
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool->storage_pool, &stats);
  const uint64_t retained_bytes = pool->statistics.committed_bytes -
                                  pool->statistics.releasing_bytes +
                                  pool->acquired_pending_bytes;
  const uint64_t retiring_bytes = stats.bytes_committed > retained_bytes
                                      ? stats.bytes_committed - retained_bytes
                                      : 0;
  pool->retiring_bytes = iree_min(pool->retiring_bytes, retiring_bytes);
  return stats.bytes_committed;
}

static uint64_t loom_serve_memory_pool_charged_bytes(
    loom_serve_memory_pool_t* pool) {
  loom_serve_memory_pool_observe_retirement(pool);
  return pool->statistics.committed_bytes + pool->statistics.pending_bytes +
         pool->retiring_bytes;
}

// Arms a one-shot capacity wake only while native retirement can make progress.
// Prepare precedes the final query so a concurrent free cannot lose its wake.
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
    return !loom_serve_memory_capacity_is_idle(pool);
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

iree_status_t loom_serve_memory_pool_allocate(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_hal_pool_t* storage_pool, iree_hal_queue_t* queue,
    iree_device_size_t slab_size, uint64_t limit_bytes,
    iree_allocator_t host_allocator, loom_serve_memory_pool_t** out_pool) {
  *out_pool = NULL;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_RETURN_IF_ERROR(iree_hal_virtual_buffer_query_capabilities(
      group, scope, host_allocator, &capabilities));
  const iree_hal_memory_mapping_operations_t required =
      (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_BIND) |
      (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND);
  if (!capabilities.max_byte_length ||
      !iree_all_bits_set(capabilities.operations, required)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "elastic backing requires bind and unbind support");
  }
  const iree_hal_pool_mapping_requirements_t requirements =
      iree_hal_pool_query_mapping_requirements(storage_pool);
  const iree_device_size_t minimum =
      iree_max(capabilities.reservation_granularity,
               iree_max(capabilities.mapping_granularity,
                        requirements.minimum_granularity));
  if (!slab_size) {
    slab_size = iree_max(minimum, requirements.recommended_granularity);
  }
  if (!slab_size || slab_size > INT64_MAX || slab_size % minimum) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "slab size must be a multiple of %" PRIdsz,
                            minimum);
  }
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
  pool->slab_size = slab_size;
  pool->minimum_alignment = minimum;
  pool->limit_bytes = limit_bytes ? limit_bytes : UINT64_MAX;
  iree_slim_mutex_initialize(&pool->capacity_mutex);
  iree_notification_initialize(&pool->capacity_notification);
  iree_status_t status = iree_hal_pool_wait_allocate(
      storage_pool, host_allocator, &pool->capacity_wait);
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
  iree_hal_pool_release(pool->storage_pool);
  iree_allocator_free(pool->host_allocator, pool);
}

loom_serve_memory_statistics_t loom_serve_memory_pool_statistics(
    loom_serve_memory_pool_t* pool) {
  loom_serve_memory_pool_collect(pool);
  loom_serve_memory_statistics_t statistics = pool->statistics;
  const uint64_t physical = loom_serve_memory_pool_observe_retirement(pool);
  const uint64_t retiring = pool->retiring_bytes;
  statistics.committed_bytes += retiring;
  statistics.releasing_bytes += retiring;
  statistics.released_bytes -= iree_min(statistics.released_bytes, retiring);
  statistics.peak_bytes = iree_max(statistics.peak_bytes, physical);
  return statistics;
}

uint64_t loom_serve_memory_pool_limit(const loom_serve_memory_pool_t* pool) {
  return pool->limit_bytes;
}

iree_status_t loom_serve_memory_pool_query_status(
    loom_serve_memory_pool_t* pool) {
  loom_serve_memory_pool_collect(pool);
  return iree_status_clone(pool->failure);
}

void loom_serve_memory_pool_set_observer(loom_serve_memory_pool_t* pool,
                                         void (*wake)(void* user_data),
                                         void* user_data) {
  iree_slim_mutex_lock(&pool->capacity_mutex);
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
}

bool loom_serve_memory_pool_is_pending(loom_serve_memory_pool_t* pool) {
  loom_serve_memory_pool_collect(pool);
  if (pool->operations) {
    return true;
  }
  loom_serve_memory_pool_observe_capacity(pool);
  return !loom_serve_memory_capacity_is_idle(pool);
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
  return iree_status_clone(pool->failure);
}

void loom_serve_memory_pool_set_reclaimer(
    loom_serve_memory_pool_t* pool, loom_serve_memory_reclaimer_t reclaimer) {
  pool->reclaimer = reclaimer;
}

iree_status_t loom_serve_memory_pool_prepare(
    loom_serve_memory_pool_t* pool, uint64_t additional_bytes,
    loom_serve_memory_admission_t* out_admission) {
  *out_admission = LOOM_SERVE_MEMORY_ADMISSION_CAPACITY;
  loom_serve_memory_pool_collect(pool);
  IREE_RETURN_IF_ERROR(iree_status_clone(pool->failure));
  if (!additional_bytes) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
    return iree_ok_status();
  }
  if (additional_bytes > pool->limit_bytes) {
    return iree_ok_status();
  }
  const uint64_t target_bytes = pool->limit_bytes - additional_bytes;
  if (pool->statistics.pending_bytes > target_bytes) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_DEFERRED;
    return iree_ok_status();
  }
  if (loom_serve_memory_pool_charged_bytes(pool) > target_bytes &&
      pool->reclaimer.fn) {
    iree_status_t status =
        pool->reclaimer.fn(pool->reclaimer.user_data,
                           target_bytes - pool->statistics.pending_bytes);
    if (!iree_status_is_ok(status)) {
      if (iree_status_is_ok(pool->failure)) {
        pool->failure = iree_status_clone(status);
      }
      return status;
    }
  }
  loom_serve_memory_pool_collect(pool);
  IREE_RETURN_IF_ERROR(iree_status_clone(pool->failure));
  bool deferred = false;
  if (loom_serve_memory_pool_charged_bytes(pool) > target_bytes) {
    deferred = loom_serve_memory_pool_observe_capacity(pool) ||
               pool->operations != NULL;
  }
  // Capacity may have returned during observation. A committed observation
  // remains evidence of progress even if its callback already ran.
  if (loom_serve_memory_pool_charged_bytes(pool) <= target_bytes) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
  } else if (deferred) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_DEFERRED;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_virtual_buffer_allocate(
    loom_serve_memory_pool_t* pool, iree_device_size_t length,
    iree_device_size_t alignment, loom_serve_memory_statistics_t* statistics,
    loom_serve_virtual_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (!length || length > INT64_MAX - pool->slab_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "invalid virtual buffer extent");
  }
  if (!alignment || pool->minimum_alignment % alignment) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "virtual reservation cannot guarantee alignment %" PRIdsz, alignment);
  }
  const uint64_t count = (length + pool->slab_size - 1) / pool->slab_size;
  if (count > (IREE_HOST_SIZE_MAX - sizeof(loom_serve_virtual_buffer_t)) /
                  sizeof(loom_serve_memory_slab_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "virtual buffer metadata is too large");
  }
  loom_serve_virtual_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator,
      sizeof(*buffer) + count * sizeof(loom_serve_memory_slab_t),
      (void**)&buffer));
  buffer->pool = pool;
  buffer->statistics = statistics;
  buffer->slab_count = (iree_host_size_t)count;
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = count * pool->slab_size,
      .min_alignment = alignment,
      .mapping_granularity = pool->slab_size,
  };
  iree_status_t status =
      iree_hal_virtual_buffer_create(pool->group, pool->scope, &options,
                                     pool->host_allocator, &buffer->handle);
  if (iree_status_is_ok(status)) {
    pool->statistics.reserved_bytes += count * pool->slab_size;
    statistics->reserved_bytes += count * pool->slab_size;
    *out_buffer = buffer;
  } else {
    iree_allocator_free(pool->host_allocator, buffer);
  }
  return status;
}

iree_hal_buffer_t* loom_serve_virtual_buffer_handle(
    const loom_serve_virtual_buffer_t* buffer) {
  return buffer->handle;
}

// Publishes one native result on the serialized pool owner. Native work never
// mutates this accounting or another operation's slab metadata.
static void loom_serve_memory_slab_publish(
    loom_serve_virtual_buffer_t* buffer, iree_host_size_t index,
    const loom_serve_memory_slab_t* result) {
  loom_serve_memory_pool_t* pool = buffer->pool;
  loom_serve_memory_slab_t* slab = &buffer->slabs[index];
  if (slab->physical && !result->physical) {
    pool->statistics.committed_bytes -= pool->slab_size;
    pool->statistics.released_bytes += pool->slab_size;
    pool->retiring_bytes += pool->slab_size;
    buffer->statistics->committed_bytes -= pool->slab_size;
    buffer->statistics->released_bytes += pool->slab_size;
  } else if (!slab->physical && result->physical) {
    pool->statistics.committed_bytes += pool->slab_size;
    pool->statistics.peak_bytes =
        iree_max(pool->statistics.peak_bytes, pool->statistics.committed_bytes);
    buffer->statistics->committed_bytes += pool->slab_size;
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
    if (!operation->acquired_accounted &&
        iree_atomic_load(&operation->acquired, iree_memory_order_acquire)) {
      pool->acquired_pending_bytes += operation->count * pool->slab_size;
      operation->acquired_accounted = true;
    }
    iree_slim_mutex_lock(&operation->mutex);
    const bool complete = operation->complete;
    iree_slim_mutex_unlock(&operation->mutex);
    if (!complete) {
      link = &operation->next;
      continue;
    }
    for (iree_host_size_t i = 0; i < operation->count; ++i) {
      const loom_serve_memory_edit_t* edit = &operation->edits[i];
      loom_serve_memory_slab_publish(edit->buffer, edit->index, &edit->result);
      if (operation->invalidates_backing &&
          iree_status_is_ok(edit->buffer->failure)) {
        edit->buffer->failure = iree_status_clone(operation->status);
      }
      if (operation->releasing) {
        edit->buffer->statistics->releasing_bytes -= pool->slab_size;
      } else {
        edit->buffer->statistics->pending_bytes -= pool->slab_size;
        if (!edit->result.physical) {
          // Failed acquisition can leave native disposal behind its buffer
          // return. Transfer the pending charge until native counts prove it
          // has retired, including the case where no backing was allocated.
          pool->retiring_bytes += pool->slab_size;
        }
      }
    }
    if (operation->releasing) {
      pool->statistics.releasing_bytes -= operation->count * pool->slab_size;
    } else {
      pool->statistics.pending_bytes -= operation->count * pool->slab_size;
      if (operation->acquired_accounted) {
        pool->acquired_pending_bytes -= operation->count * pool->slab_size;
      }
    }
    operation->published = true;
    *link = operation->next;
    operation->next = NULL;
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
  if (operation->readiness && iree_status_is_ok(status)) {
    status =
        iree_hal_semaphore_signal(operation->readiness, operation->value, NULL);
  }
  if (operation->readiness && !iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(operation->readiness, iree_status_clone(status));
  }
  iree_slim_mutex_lock(&operation->mutex);
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
  if (stage == &stage->operation->stages[0] && !stage->operation->releasing &&
      iree_status_is_ok(status)) {
    iree_atomic_store(&stage->operation->acquired, 1,
                      iree_memory_order_release);
  }
  loom_serve_memory_operation_complete(stage->operation);
}

static iree_status_t loom_serve_memory_operation_allocate(
    loom_serve_memory_pool_t* pool, iree_host_size_t count,
    loom_serve_memory_operation_t** out_operation) {
  *out_operation = NULL;
  loom_serve_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      pool->host_allocator, sizeof(*operation), count,
      sizeof(loom_serve_memory_edit_t) + sizeof(iree_hal_buffer_t*) +
          sizeof(iree_hal_pool_reservation_request_t) +
          sizeof(iree_hal_memory_mapping_t),
      (void**)&operation));
  operation->pool = pool;
  operation->count = count;
  operation->sources = (iree_hal_buffer_t**)(operation->edits + count);
  operation->requests =
      (iree_hal_pool_reservation_request_t*)(operation->sources + count);
  operation->mappings =
      (iree_hal_memory_mapping_t*)(operation->requests + count);
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
  for (iree_host_size_t i = 0; i < operation->count; ++i) {
    loom_serve_memory_edit_t* edit = &operation->edits[i];
    operation->requests[i] =
        iree_hal_make_pool_reservation_request(pool->slab_size, 0);
    if (operation->releasing) {
      operation->sources[i] = edit->result.physical;
    }
    operation->mappings[i] = (iree_hal_memory_mapping_t){
        .target = edit->buffer->handle,
        .scope = pool->mapping_scope,
        .target_range = {edit->index * pool->slab_size, pool->slab_size},
        .access = operation->releasing ? IREE_HAL_MEMORY_ACCESS_NONE
                                       : IREE_HAL_MEMORY_ACCESS_ALL,
        .operation = operation->releasing
                         ? IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND
                         : IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  if (iree_status_is_ok(status)) {
    status = operation->releasing
                 ? iree_hal_queue_map(pool->queue,
                                      iree_hal_semaphore_list_empty(), first,
                                      operation->count, operation->mappings, 0)
                 : iree_hal_queue_alloca(
                       pool->queue, iree_hal_semaphore_list_empty(), first,
                       pool->storage_pool, operation->count,
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
    status = iree_hal_queue_map(pool->queue, first, second, operation->count,
                                operation->mappings, 0);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(operation->stages[1].semaphore, status);
  }
  // Final handoff: an inline completion can now publish and release capture.
  loom_serve_memory_operation_complete(operation);
}

iree_status_t loom_serve_memory_pool_commit_async(
    loom_serve_memory_pool_t* pool, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, bool* out_admitted,
    loom_serve_memory_operation_t** out_operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_admitted = false;
  *out_operation = NULL;
  loom_serve_memory_pool_collect(pool);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t r = 0; r < range_count && iree_status_is_ok(status);
       ++r) {
    status = iree_status_clone(ranges[r].buffer->failure);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  iree_host_size_t count = 0;
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    const loom_serve_memory_range_t* range = &ranges[r];
    const iree_host_size_t end =
        (range->offset + range->length - 1) / pool->slab_size + 1;
    for (iree_host_size_t i = range->offset / pool->slab_size; i < end; ++i) {
      loom_serve_memory_slab_t* slab = &range->buffer->slabs[i];
      IREE_ASSERT(!slab->in_flight);
      if (!slab->physical && !slab->pending_commit) {
        slab->pending_commit = true;
        ++count;
      }
    }
  }
  loom_serve_memory_admission_t admission;
  status =
      loom_serve_memory_pool_prepare(pool, count * pool->slab_size, &admission);
  const bool admitted = admission == LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE;
  loom_serve_memory_operation_t* operation = NULL;
  if (iree_status_is_ok(status) && admitted && count) {
    status = loom_serve_memory_operation_allocate(pool, count, &operation);
  }
  iree_host_size_t index = 0;
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    const loom_serve_memory_range_t* range = &ranges[r];
    const iree_host_size_t end =
        (range->offset + range->length - 1) / pool->slab_size + 1;
    for (iree_host_size_t i = range->offset / pool->slab_size; i < end; ++i) {
      loom_serve_memory_slab_t* slab = &range->buffer->slabs[i];
      if (slab->pending_commit) {
        slab->pending_commit = false;
        if (operation) {
          operation->edits[index++] = (loom_serve_memory_edit_t){
              .buffer = range->buffer,
              .index = i,
          };
          slab->in_flight = true;
          range->buffer->statistics->pending_bytes += pool->slab_size;
        }
      }
    }
  }
  if (operation) {
    operation->count = count;
    operation->readiness = readiness;
    operation->value = value;
    iree_hal_semaphore_retain(readiness);
    pool->statistics.pending_bytes += count * pool->slab_size;
    *out_operation = operation;
    loom_serve_memory_operation_enqueue(pool, operation);
  } else if (iree_status_is_ok(status) && admitted && readiness) {
    status = iree_hal_semaphore_signal(readiness, value, NULL);
  }
  *out_admitted = iree_status_is_ok(status) && admitted;
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
    status = iree_status_clone(buffers[b]->failure);
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
        buffer->statistics->releasing_bytes += pool->slab_size;
      }
    }
  }
  pool->statistics.releasing_bytes += count * pool->slab_size;
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
  loom_serve_memory_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_memory_pool_commit_async(
      pool, range_count, ranges, NULL, 0, out_admitted, &operation));
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
  const iree_device_size_t slab_size = buffer->pool->slab_size;
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
  // Checked removal remains available after a partial failed mapping batch.
  // Sources stay owned if detach fails, so the caller can retry cleanup.
  IREE_RETURN_IF_ERROR(
      iree_hal_virtual_buffer_detach(buffer->handle, pool->mapping_scope));
  for (iree_host_size_t i = 0; i < buffer->slab_count; ++i) {
    IREE_ASSERT(!buffer->slabs[i].in_flight);
    iree_hal_buffer_release(buffer->slabs[i].physical);
    const loom_serve_memory_slab_t empty = {0};
    loom_serve_memory_slab_publish(buffer, i, &empty);
  }
  iree_hal_buffer_release(buffer->handle);
  iree_status_free(buffer->failure);
  pool->statistics.reserved_bytes -= buffer->slab_count * pool->slab_size;
  buffer->statistics->reserved_bytes -= buffer->slab_count * pool->slab_size;
  iree_allocator_free(pool->host_allocator, buffer);
  return iree_ok_status();
}
