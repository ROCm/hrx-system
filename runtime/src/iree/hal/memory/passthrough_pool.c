// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/passthrough_pool.h"

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/memory/tracing.h"

enum { IREE_HAL_PASSTHROUGH_POOL_INLINE_TRANSACTION_CAPACITY = 8 };

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

typedef struct iree_hal_passthrough_pool_reservation_state_t
    iree_hal_passthrough_pool_reservation_state_t;
typedef struct iree_hal_passthrough_pool_allocation_state_t
    iree_hal_passthrough_pool_allocation_state_t;

// Queue identities may be registered after allocation. Capture the complete
// representable release history without allocating on the release path.
IREE_ASYNC_FIXED_FRONTIER_TYPE(iree_hal_passthrough_pool_frontier_t, UINT8_MAX);

typedef enum iree_hal_passthrough_pool_retirement_state_e {
  IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_LIVE = 0,
  IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_WAITING,
  IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_QUEUED,
  IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_QUARANTINED,
} iree_hal_passthrough_pool_retirement_state_t;

typedef struct iree_hal_passthrough_pool_state_t {
  // Exact usage grant used to qualify explicit native imports.
  iree_hal_buffer_usage_t allowed_usage;

  // Capacity publication owner retained through the final native completion.
  iree_async_notification_t* notification;

  // Exact retirement tracker retained through the final native completion.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Provider used to acquire one slab per reservation.
  iree_hal_slab_provider_t* slab_provider;

  // Public resource plus native allocations that still own completion state.
  iree_atomic_ref_count_t body_ref_count;

  // Protects state links, waiter disposition and shutdown publication.
  iree_slim_mutex_t retirement_mutex;

  // Every native allocation still owned, including pending retirement.
  iree_hal_passthrough_pool_allocation_state_t* allocation_head;

  // True after the caller has retired execution and begun pool teardown.
  bool shutting_down;

  // First failed retirement prerequisite, owned until pool destruction.
  iree_atomic_intptr_t failure_status;

  // Provider-owned allocator for private metadata and native retirement state.
  iree_allocator_t host_allocator;

  // Stable named-memory stream for logical reservations from this pool.
  iree_hal_memory_trace_t trace;

  // Immutable memory properties provided by |slab_provider|.
  iree_hal_slab_provider_properties_t slab_properties;

  // ASAN policy used to shape hidden backing ranges.
  iree_hal_asan_pool_options_t asan_options;

  // Whether reservations may acquire new provider-owned slabs.
  bool owned_enabled;

  // Number of exact external storage classes admitted for import.
  iree_host_size_t import_class_count;

  // Immutable admitted classes stored inline after this object.
  const iree_hal_external_storage_class_t* import_classes;

  // Approximate live reservation bytes for lock-free stats queries.
  iree_atomic_int64_t bytes_reserved;

  // Native bytes still held, including pending or quarantined returns.
  iree_atomic_int64_t bytes_committed;

  // Approximate live reservation count for lock-free stats queries.
  iree_atomic_int32_t reservation_count;

  // Approximate live slab count for lock-free stats queries.
  iree_atomic_int32_t slab_count;

  // Total reservations committed by successful transactions.
  iree_atomic_int64_t reserve_count;

  // Total reservations returned by release transactions.
  iree_atomic_int64_t release_count;
} iree_hal_passthrough_pool_state_t;

// Public metadata is reclaimed at final pool release independently of native
// cleanup. Its allocator therefore has exactly the public pool lifetime.
typedef struct iree_hal_passthrough_pool_t {
  // Base pool resource for vtable dispatch and ref counting.
  iree_hal_pool_t base;
  // Private native ownership and statistics, allocated by the provider.
  iree_hal_passthrough_pool_state_t* state;
} iree_hal_passthrough_pool_t;

static iree_hal_passthrough_pool_state_t* iree_hal_passthrough_pool_state(
    const iree_hal_pool_t* base_pool) {
  return ((const iree_hal_passthrough_pool_t*)base_pool)->state;
}

// Common native allocation state retained until placement-local release.
struct iree_hal_passthrough_pool_allocation_state_t {
  // Private source state retained until the native release callback returns.
  iree_hal_passthrough_pool_state_t* pool;

  // Slab acquired from the pool's provider for this reservation.
  iree_hal_slab_t slab;

  // Prepared facts shared by views while this slab allocation is owned.
  iree_hal_slab_buffer_backing_t buffer_backing;

  // Backing bytes charged to this reservation.
  iree_device_size_t charged_length;

  // Move-owned external transport released after the native attachment.
  iree_hal_external_storage_t external_storage;

  // Previous native allocation in the pool's retirement inventory.
  iree_hal_passthrough_pool_allocation_state_t* previous;

  // Next native allocation in the pool's retirement inventory.
  iree_hal_passthrough_pool_allocation_state_t* next;

  // Protected by the pool's retirement mutex after the allocation returns.
  iree_hal_passthrough_pool_retirement_state_t retirement_state;

  // Native return descriptor secured before acquisition is published.
  iree_hal_slab_release_request_t release_request;

  // Joins native submission and its callback before reclaiming this state.
  iree_atomic_int32_t release_pending;
};

// Per-reservation state owned by the reservation until release.
struct iree_hal_passthrough_pool_reservation_state_t {
  // Common native allocation ownership and retirement state.
  iree_hal_passthrough_pool_allocation_state_t allocation;

  // ASAN backing layout for this reservation.
  iree_hal_asan_allocation_layout_t asan_layout;

  // One reference for the live reservation token plus one reference for each
  // materialized buffer view. The last reference starts frontier retirement,
  // allowing the reservation to return before borrowed views are decommitted.
  iree_atomic_int32_t reference_count;

  // Set exactly once when the reservation token is released. Owning
  // materialized buffers consume that reservation release in their destroy
  // callback; borrowed views only drop their own reference.
  iree_atomic_int32_t reservation_released;

  // Caller-owned storage registered with the group's existing tracker.
  iree_async_frontier_waiter_t waiter;

  // Joins waiter registration and its callback before native retirement.
  iree_atomic_int32_t wait_pending;

  // Owned terminal prerequisite status until the registration handoff joins.
  iree_status_t wait_status;

  // Exact history copied when the explicit reservation token is released.
  iree_hal_passthrough_pool_frontier_t death_frontier;
};

typedef struct iree_hal_passthrough_pool_materialize_state_t
    iree_hal_passthrough_pool_materialize_state_t;

// Per-buffer element in an owning materialization transaction.
typedef struct iree_hal_passthrough_pool_materialize_element_t {
  // Shared transaction state controlling the ownership commit.
  iree_hal_passthrough_pool_materialize_state_t* state;

  // Reservation state released when the committed buffer is destroyed.
  iree_hal_passthrough_pool_reservation_state_t* reservation_state;

  // Materialized buffer staged until the complete transaction succeeds.
  iree_hal_buffer_t* buffer;
} iree_hal_passthrough_pool_materialize_element_t;

// Shared state for an owning materialization transaction.
struct iree_hal_passthrough_pool_materialize_state_t {
  // Host allocator used for this state object.
  iree_allocator_t host_allocator;

  // Number of materialized buffers still referencing this transaction.
  iree_atomic_int32_t reference_count;

  // True after every buffer was materialized and reservation ownership moved.
  bool ownership_committed;

  // Per-buffer transaction elements.
  iree_hal_passthrough_pool_materialize_element_t elements[];
};

static const iree_hal_pool_vtable_t iree_hal_passthrough_pool_vtable;
static void iree_hal_passthrough_pool_destroy(iree_hal_pool_t* base_pool);

static const char* IREE_HAL_PASSTHROUGH_POOL_TRACE_ID =
    "iree-hal-passthrough-pool";

// Returns a process-local tracing identity without interpreting opaque device
// storage as an address. Device providers may intentionally leave base_ptr
// null and use only provider_handle.
static void* iree_hal_passthrough_pool_trace_address(
    const iree_hal_slab_t* slab, iree_device_size_t offset) {
  uintptr_t base = (uintptr_t)slab->base_ptr;
  if (!base) {
    base = (uintptr_t)slab->provider_handle;
  }
  return base ? (void*)(base + (uintptr_t)offset) : NULL;
}

//===----------------------------------------------------------------------===//
// Reservation state helpers
//===----------------------------------------------------------------------===//

// Releases the private body after public destruction and every native return.
// No caller-visible resource reference is held by a retired allocation.
static void iree_hal_passthrough_pool_release_body(
    iree_hal_passthrough_pool_state_t* pool) {
  if (iree_atomic_ref_count_dec(&pool->body_ref_count) != 1) {
    return;
  }
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_hal_memory_trace_deinitialize(&pool->trace);
  iree_async_frontier_tracker_release(pool->frontier_tracker);
  iree_async_notification_release(pool->notification);
  iree_hal_slab_provider_t* provider = pool->slab_provider;
  iree_status_free((iree_status_t)iree_atomic_load(&pool->failure_status,
                                                   iree_memory_order_relaxed));
  iree_slim_mutex_deinitialize(&pool->retirement_mutex);
  iree_allocator_free(host_allocator, pool);
  iree_hal_slab_provider_release(provider);
}

static void iree_hal_passthrough_pool_link_allocation(
    iree_hal_passthrough_pool_state_t* pool,
    iree_hal_passthrough_pool_allocation_state_t* state) {
  iree_atomic_ref_count_inc(&pool->body_ref_count);
  iree_slim_mutex_lock(&pool->retirement_mutex);
  state->next = pool->allocation_head;
  if (pool->allocation_head) {
    pool->allocation_head->previous = state;
  }
  pool->allocation_head = state;
  iree_slim_mutex_unlock(&pool->retirement_mutex);
}

static void iree_hal_passthrough_pool_unlink_allocation(
    iree_hal_passthrough_pool_state_t* pool,
    iree_hal_passthrough_pool_allocation_state_t* state) {
  iree_slim_mutex_lock(&pool->retirement_mutex);
  if (state->previous) {
    state->previous->next = state->next;
  } else {
    pool->allocation_head = state->next;
  }
  if (state->next) {
    state->next->previous = state->previous;
  }
  iree_slim_mutex_unlock(&pool->retirement_mutex);
}

static void iree_hal_passthrough_pool_release_completed(void* user_data) {
  iree_hal_passthrough_pool_allocation_state_t* state = user_data;
  if (iree_atomic_fetch_sub(&state->release_pending, 1,
                            iree_memory_order_acq_rel) != 1) {
    return;
  }
  iree_hal_passthrough_pool_state_t* pool = state->pool;
  iree_hal_external_storage_release(&state->external_storage);
  if (state->charged_length) {
    iree_atomic_fetch_sub(&pool->bytes_committed,
                          (int64_t)state->charged_length,
                          iree_memory_order_relaxed);
    iree_atomic_fetch_sub(&pool->slab_count, 1, iree_memory_order_relaxed);
  }
  iree_hal_passthrough_pool_unlink_allocation(pool, state);
  iree_allocator_free(pool->host_allocator, state);
  // Physical capacity becomes available only after the native free returns.
  // Budget observers must recheck then, not just at logical reservation return.
  iree_async_notification_signal_if_observed(pool->notification,
                                             IREE_ALL_WAITERS);
  iree_hal_passthrough_pool_release_body(pool);
}

static void iree_hal_passthrough_pool_retire_allocation(
    iree_hal_passthrough_pool_allocation_state_t* state) {
  iree_hal_passthrough_pool_state_t* pool = state->pool;
  state->release_request = (iree_hal_slab_release_request_t){
      .count = 1,
      .slabs = &state->slab,
      .completion = {iree_hal_passthrough_pool_release_completed, state},
  };
  iree_atomic_store(&state->release_pending, 2, iree_memory_order_relaxed);
  iree_hal_slab_provider_release_slabs(pool->slab_provider,
                                       &state->release_request);
  iree_hal_passthrough_pool_release_completed(state);
}

static void iree_hal_passthrough_pool_frontier_join(
    iree_hal_passthrough_pool_reservation_state_t* reservation_state) {
  if (iree_atomic_fetch_sub(&reservation_state->wait_pending, 1,
                            iree_memory_order_acq_rel) != 1) {
    return;
  }
  iree_hal_passthrough_pool_allocation_state_t* state =
      &reservation_state->allocation;
  iree_hal_passthrough_pool_state_t* pool = state->pool;
  iree_status_t status = reservation_state->wait_status;
  reservation_state->wait_status = iree_ok_status();
  const bool failed = !iree_status_is_ok(status);
  // Shutdown may claim a quarantined allocation immediately after publication.
  // Keep its notification owner alive through this callback's final access.
  iree_atomic_ref_count_inc(&pool->body_ref_count);
  iree_slim_mutex_lock(&pool->retirement_mutex);
  const bool may_retire = !failed || pool->shutting_down;
  if (failed) {
    const iree_status_t failure = (iree_status_t)iree_atomic_load(
        &pool->failure_status, iree_memory_order_relaxed);
    if (iree_status_is_ok(failure)) {
      iree_atomic_store(&pool->failure_status, (intptr_t)status,
                        iree_memory_order_release);
      status = iree_ok_status();
    }
  }
  state->retirement_state =
      may_retire ? IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_QUEUED
                 : IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_QUARANTINED;
  iree_slim_mutex_unlock(&pool->retirement_mutex);
  // The first failure owns the diagnostic; later failed prerequisites do not
  // establish native retirement or replace the original cause.
  iree_status_free(status);
  if (failed) {
    iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
  }
  if (may_retire) {
    iree_hal_passthrough_pool_retire_allocation(state);
  }
  iree_hal_passthrough_pool_release_body(pool);
}

static void iree_hal_passthrough_pool_frontier_resolved(void* user_data,
                                                        iree_status_t status) {
  iree_hal_passthrough_pool_reservation_state_t* state = user_data;
  state->wait_status = status;
  iree_hal_passthrough_pool_frontier_join(state);
}

static void iree_hal_passthrough_pool_reservation_state_release_reference(
    iree_hal_passthrough_pool_reservation_state_t* reservation_state) {
  iree_hal_passthrough_pool_allocation_state_t* allocation =
      &reservation_state->allocation;
  iree_hal_passthrough_pool_state_t* pool = allocation->pool;
  const int32_t previous_count = iree_atomic_fetch_sub(
      &reservation_state->reference_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous_count > 0);
  if (previous_count != 1) {
    return;
  }

  iree_atomic_store(&reservation_state->wait_pending, 2,
                    iree_memory_order_relaxed);
  iree_slim_mutex_lock(&pool->retirement_mutex);
  allocation->retirement_state = IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_WAITING;
  iree_slim_mutex_unlock(&pool->retirement_mutex);
  iree_status_t status = iree_async_frontier_tracker_wait(
      pool->frontier_tracker,
      iree_async_fixed_frontier_as_const_frontier(
          &reservation_state->death_frontier),
      iree_hal_passthrough_pool_frontier_resolved, reservation_state,
      &reservation_state->waiter);
  if (!iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_frontier_resolved(reservation_state, status);
  }
  iree_hal_passthrough_pool_frontier_join(reservation_state);
}

static void iree_hal_passthrough_pool_reservation_state_release_reservation(
    iree_hal_passthrough_pool_reservation_state_t* reservation_state,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_passthrough_pool_allocation_state_t* allocation =
      &reservation_state->allocation;
  iree_hal_passthrough_pool_state_t* pool = allocation->pool;
  const int32_t already_released = iree_atomic_exchange(
      &reservation_state->reservation_released, 1, iree_memory_order_acq_rel);
  IREE_ASSERT_EQ(already_released, 0);

  if (death_frontier) {
    memcpy(&reservation_state->death_frontier, death_frontier,
           sizeof(*death_frontier) + death_frontier->entry_count *
                                         sizeof(death_frontier->entries[0]));
  }

  const iree_device_size_t user_offset =
      iree_hal_asan_pool_options_is_enabled(&pool->asan_options)
          ? reservation_state->asan_layout.user_offset
          : 0;
  iree_hal_memory_trace_free(
      &pool->trace,
      iree_hal_passthrough_pool_trace_address(&allocation->slab, user_offset));
  iree_atomic_fetch_add(&pool->bytes_reserved,
                        -(int64_t)allocation->charged_length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reservation_count, -1,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);

  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
}

static void iree_hal_passthrough_pool_borrowed_view_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)user_data;
  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
}

static void iree_hal_passthrough_pool_advise_asan_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_asan_range_advice_flags_t flags) {
  iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_passthrough_pool_reservation_state_t* state =
        (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
            reservations[i]
                .block_handle;
    iree_hal_slab_provider_advise_asan_range(pool->slab_provider,
                                             &state->allocation.slab, 0, flags,
                                             &state->asan_layout);
  }
}

static void iree_hal_passthrough_pool_owned_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_passthrough_pool_materialize_element_t* element =
      (iree_hal_passthrough_pool_materialize_element_t*)user_data;
  iree_hal_passthrough_pool_materialize_state_t* state = element->state;
  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      element->reservation_state;
  const bool ownership_committed = state->ownership_committed;
  iree_hal_passthrough_pool_state_t* pool = NULL;
  if (ownership_committed) {
    pool = reservation_state->allocation.pool;
    if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
      iree_hal_slab_provider_advise_asan_range(
          pool->slab_provider, &reservation_state->allocation.slab, 0,
          IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED,
          &reservation_state->asan_layout);
    }
    iree_hal_passthrough_pool_reservation_state_release_reservation(
        reservation_state, NULL);
  }
  iree_hal_passthrough_pool_reservation_state_release_reference(
      reservation_state);
  // A notification observer may immediately release the pool. Publish only
  // after the final buffer reference has moved the allocation out of LIVE.
  if (ownership_committed) {
    iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
  }
  const int32_t previous_count = iree_atomic_fetch_sub(
      &state->reference_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(previous_count > 0);
  if (previous_count == 1) {
    iree_allocator_free(state->host_allocator, state);
  }
}

//===----------------------------------------------------------------------===//
// Create / Destroy
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_passthrough_pool_create(
    iree_hal_passthrough_pool_options_t options,
    iree_hal_slab_provider_t* slab_provider,
    iree_async_notification_t* notification,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(slab_provider);
  IREE_ASSERT_ARGUMENT(notification);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_slab_provider_validate_asan_options(slab_provider,
                                                       &options.asan));

  iree_host_size_t pool_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_host_size_checked_mul_add(
              options.imports.class_count,
              sizeof(iree_hal_external_storage_class_t),
              sizeof(iree_hal_passthrough_pool_state_t), &pool_size)
              ? iree_ok_status()
              : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                 "pass-through pool metadata size overflow"));
  iree_hal_passthrough_pool_t* public_pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*public_pool),
                                (void**)&public_pool));
  iree_hal_passthrough_pool_state_t* pool = NULL;
  iree_status_t status = iree_allocator_malloc(slab_provider->host_allocator,
                                               pool_size, (void**)&pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_initialize(
        &iree_hal_passthrough_pool_vtable, options.memory_contract,
        notification, (iree_hal_pool_wait_source_list_t){0}, frontier_tracker,
        host_allocator, &public_pool->base);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(slab_provider->host_allocator, pool);
    iree_allocator_free(host_allocator, public_pool);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  public_pool->state = pool;
  pool->host_allocator = slab_provider->host_allocator;
  public_pool->base.epoch_query = options.epoch_query;
  public_pool->base.asan_enabled =
      iree_hal_asan_pool_options_is_enabled(&options.asan);
  pool->asan_options = options.asan;
  pool->owned_enabled = options.owned_mode == IREE_HAL_SLAB_POOL_OWNED_ENABLED;
  pool->import_class_count = options.imports.class_count;
  pool->import_classes = (const iree_hal_external_storage_class_t*)(pool + 1);
  iree_hal_external_storage_class_t* import_classes =
      (iree_hal_external_storage_class_t*)pool->import_classes;
  for (iree_host_size_t i = 0; i < options.imports.class_count; ++i) {
    import_classes[i] = options.imports.classes[i].storage_class;
  }
  iree_atomic_ref_count_init(&pool->body_ref_count);
  pool->notification = notification;
  iree_async_notification_retain(notification);
  pool->frontier_tracker = frontier_tracker;
  iree_async_frontier_tracker_retain(frontier_tracker);
  iree_slim_mutex_initialize(&pool->retirement_mutex);
  pool->slab_provider = slab_provider;
  iree_hal_slab_provider_retain(slab_provider);
  iree_hal_slab_provider_query_properties(slab_provider,
                                          &pool->slab_properties);
  pool->allowed_usage = options.memory_contract
                            ? options.memory_contract->allowed_usage
                            : pool->slab_properties.supported_usage;

  status = iree_hal_memory_trace_initialize_pool(
      options.trace_name, IREE_HAL_PASSTHROUGH_POOL_TRACE_ID, &pool->trace);
  if (iree_status_is_ok(status)) {
    *out_pool = &public_pool->base;
  } else {
    iree_hal_passthrough_pool_destroy(&public_pool->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_passthrough_pool_destroy(iree_hal_pool_t* base_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  iree_hal_slab_operation_t* ready_head = NULL;
  iree_slim_mutex_lock(&pool->retirement_mutex);
  pool->shutting_down = true;
  for (iree_hal_passthrough_pool_allocation_state_t* state =
           pool->allocation_head;
       state; state = state->next) {
    IREE_ASSERT(
        state->retirement_state != IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_LIVE,
        "pool destruction requires returned reservations and views");
    if (state->retirement_state ==
        IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_WAITING) {
      iree_hal_passthrough_pool_reservation_state_t* reservation_state =
          (iree_hal_passthrough_pool_reservation_state_t*)state;
      if (iree_async_frontier_tracker_cancel_wait(pool->frontier_tracker,
                                                  &reservation_state->waiter)) {
        state->retirement_state =
            IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_QUARANTINED;
      }
    }
    // Only caller-quiescent destruction permits native cleanup after a
    // failed or cancelled prerequisite. Ordinary release/trim cannot infer it.
    if (state->retirement_state ==
        IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_QUARANTINED) {
      state->retirement_state = IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_QUEUED;
      state->release_request.operation.next = ready_head;
      ready_head = &state->release_request.operation;
    }
  }
  iree_slim_mutex_unlock(&pool->retirement_mutex);
  while (ready_head) {
    iree_hal_slab_operation_t* operation = ready_head;
    ready_head = operation->next;
    iree_hal_passthrough_pool_allocation_state_t* state =
        (iree_hal_passthrough_pool_allocation_state_t*)((uint8_t*)operation -
                                                        offsetof(
                                                            iree_hal_passthrough_pool_allocation_state_t,
                                                            release_request));
    iree_hal_passthrough_pool_retire_allocation(state);
  }
  iree_allocator_t host_allocator = base_pool->host_allocator;
  iree_hal_pool_deinitialize(base_pool);
  iree_allocator_free(host_allocator, base_pool);
  iree_hal_passthrough_pool_release_body(pool);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// External storage import
//===----------------------------------------------------------------------===//

static bool iree_hal_passthrough_pool_supports_import_class(
    const iree_hal_pool_t* base_pool,
    iree_hal_external_storage_class_t storage_class) {
  const iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  for (iree_host_size_t i = 0; i < pool->import_class_count; ++i) {
    if (iree_hal_external_storage_class_equal(pool->import_classes[i],
                                              storage_class)) {
      return true;
    }
  }
  return false;
}

static bool iree_hal_passthrough_pool_query_import_requirements(
    const iree_hal_pool_t* base_pool,
    iree_hal_external_storage_class_t storage_class,
    iree_hal_pool_import_requirements_t* out_requirements) {
  const iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  if (!iree_hal_passthrough_pool_supports_import_class(base_pool,
                                                       storage_class)) {
    return false;
  }
  return iree_hal_slab_provider_query_import_requirements(
      pool->slab_provider, storage_class, out_requirements);
}

static void iree_hal_passthrough_pool_imported_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_passthrough_pool_allocation_state_t* allocation = user_data;
  iree_hal_passthrough_pool_state_t* pool = allocation->pool;
  iree_hal_memory_trace_free(&pool->trace,
                             iree_hal_passthrough_pool_trace_address(
                                 &allocation->slab, buffer->memory.offset));
  iree_slim_mutex_lock(&pool->retirement_mutex);
  IREE_ASSERT_EQ(allocation->retirement_state,
                 IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_LIVE);
  allocation->retirement_state = IREE_HAL_PASSTHROUGH_POOL_RETIREMENT_QUEUED;
  iree_slim_mutex_unlock(&pool->retirement_mutex);
  iree_hal_passthrough_pool_retire_allocation(allocation);
}

static iree_status_t iree_hal_passthrough_pool_validate_imported_slab(
    const iree_hal_passthrough_pool_state_t* pool,
    iree_hal_buffer_range_t logical_range,
    const iree_hal_imported_slab_t* imported_slab) {
  const iree_hal_slab_provider_properties_t* properties =
      &imported_slab->properties;
  if (logical_range.byte_offset < imported_slab->storage_offset ||
      logical_range.byte_offset - imported_slab->storage_offset >
          imported_slab->slab.length ||
      logical_range.byte_length >
          imported_slab->slab.length -
              (logical_range.byte_offset - imported_slab->storage_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "logical import range is outside the native attachment envelope");
  }
  if (!iree_device_size_is_power_of_two(properties->allocation_alignment) ||
      !iree_device_size_is_power_of_two(properties->maintenance_alignment)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "native import returned invalid alignment facts");
  }
  const iree_hal_atomic_operation_capabilities_t required_atomics =
      pool->slab_properties.atomic_operations;
  const iree_hal_atomic_operation_capabilities_t actual_atomics =
      properties->atomic_operations;
  const bool supports_atomics = iree_hal_atomic_operation_capabilities_support(
      actual_atomics, required_atomics);
  const iree_hal_buffer_usage_t required_usage = pool->allowed_usage;
  if (!iree_all_bits_set(properties->memory_type,
                         pool->slab_properties.memory_type) ||
      !iree_all_bits_set(properties->supported_usage, required_usage) ||
      !iree_all_bits_set(properties->queue_family_affinity,
                         pool->slab_properties.queue_family_affinity) ||
      !supports_atomics) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external storage does not satisfy the pool memory contract");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_passthrough_pool_import_buffer(
    iree_hal_pool_t* base_pool, iree_hal_external_storage_t* inout_storage,
    iree_hal_buffer_range_t logical_range, iree_hal_buffer_t** out_buffer) {
  iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  if (!iree_hal_passthrough_pool_supports_import_class(
          base_pool, inout_storage->storage_class)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "external storage class was not admitted by pool");
  }
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external storage cannot satisfy pass-through ASAN layout policy");
  }

  iree_hal_passthrough_pool_allocation_state_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      pool->host_allocator, sizeof(*allocation), (void**)&allocation));
  memset(allocation, 0, sizeof(*allocation));
  allocation->pool = pool;

  iree_hal_imported_slab_t imported_slab = {0};
  iree_status_t status = iree_hal_slab_provider_import_slab(
      pool->slab_provider, inout_storage, &imported_slab);
  const bool native_attached = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = iree_hal_passthrough_pool_validate_imported_slab(
        pool, logical_range, &imported_slab);
  }

  iree_hal_buffer_t* buffer = NULL;
  iree_device_size_t slab_offset = 0;
  if (iree_status_is_ok(status)) {
    slab_offset = logical_range.byte_offset - imported_slab.storage_offset;
    allocation->slab = imported_slab.slab;
    iree_hal_slab_buffer_backing_initialize(
        pool->slab_provider, &allocation->slab, &imported_slab.properties,
        pool->notification, pool->frontier_tracker,
        &allocation->buffer_backing);
    const iree_hal_buffer_release_callback_t release_callback = {
        .fn = iree_hal_passthrough_pool_imported_buffer_release,
        .user_data = allocation,
    };
    const iree_hal_memory_access_t allowed_access =
        base_pool->memory_contract ? base_pool->memory_contract->allowed_access
                                   : IREE_HAL_MEMORY_ACCESS_ALL;
    const iree_hal_buffer_usage_t allowed_usage =
        base_pool->memory_contract ? base_pool->memory_contract->allowed_usage
                                   : imported_slab.properties.supported_usage;
    status = iree_hal_slab_provider_wrap_buffer(
        pool->slab_provider, &allocation->slab, slab_offset,
        logical_range.byte_length, allowed_access, allowed_usage,
        release_callback, &buffer);
  }

  if (iree_status_is_ok(status)) {
    buffer->memory.contract = base_pool->memory_contract;
    buffer->memory.backing = &allocation->buffer_backing.facts;
    buffer->memory.offset = slab_offset;
    allocation->external_storage = *inout_storage;
    memset(inout_storage, 0, sizeof(*inout_storage));
    iree_hal_passthrough_pool_link_allocation(pool, allocation);
    iree_hal_memory_trace_alloc(
        &pool->trace,
        iree_hal_passthrough_pool_trace_address(&allocation->slab, slab_offset),
        logical_range.byte_length);
    *out_buffer = buffer;
  } else {
    if (native_attached) {
      iree_hal_slab_provider_release_slab_sync(pool->slab_provider,
                                               &imported_slab.slab);
    }
    iree_allocator_free(pool->host_allocator, allocation);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Reserve / Release
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_passthrough_pool_validate_reservation_request(
    const iree_hal_passthrough_pool_state_t* pool,
    const iree_hal_pool_reservation_request_t* request) {
  const iree_device_size_t size = request->allocation_size;
  const iree_device_size_t alignment =
      request->min_alignment ? request->min_alignment : 1;
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
  if (alignment > pool->slab_properties.max_allocation_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment %" PRIdsz
                            " exceeds pass-through pool alignment %" PRIdsz,
                            alignment,
                            pool->slab_properties.max_allocation_alignment);
  }
  if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
    iree_hal_asan_allocation_layout_t asan_layout;
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        &pool->asan_options, size, alignment, &asan_layout));
  }
  return iree_ok_status();
}

// Native growth is cold: all independent release records and one batch
// continuation are secured before the first native allocation can begin.
typedef struct iree_hal_passthrough_pool_acquire_t {
  // Pool borrowed through terminal completion.
  iree_hal_passthrough_pool_state_t* pool;
  // Caller geometry, retained by the caller through completion.
  const iree_hal_pool_reservation_request_t* requests;
  // Caller outputs published only after the complete batch is prepared.
  iree_hal_pool_reservation_t* out_reservations;
  // Exact acquisition facts published alongside reservations.
  iree_hal_pool_acquire_info_t* out_infos;
  // Exactly one terminal pool acquisition result.
  iree_hal_pool_acquire_callback_t completion;
  // Stable native input/output descriptor.
  iree_hal_slab_acquire_request_t native;
  // Presecured native rollback descriptor, used only after acquire joins.
  iree_hal_slab_release_request_t rollback;
  // Independently owned records, one per native slab.
  iree_hal_passthrough_pool_reservation_state_t** states;
  // Joins the active native submission and callback.
  iree_atomic_int32_t pending;
  // Native or post-acquisition preparation failure.
  iree_status_t status;
} iree_hal_passthrough_pool_acquire_t;

static void iree_hal_passthrough_pool_acquire_finish(
    iree_hal_passthrough_pool_acquire_t* acquire) {
  const iree_hal_pool_acquire_callback_t completion = acquire->completion;
  const iree_status_t status = acquire->status;
  const iree_hal_pool_acquire_result_t result =
      iree_status_is_ok(status) ? IREE_HAL_POOL_ACQUIRE_OK_FRESH
                                : IREE_HAL_POOL_ACQUIRE_NONE;
  const iree_allocator_t host_allocator = acquire->pool->host_allocator;
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < acquire->native.count; ++i) {
      iree_allocator_free(host_allocator, acquire->states[i]);
    }
  }
  iree_allocator_free(host_allocator, acquire);
  completion.fn(completion.user_data, status, result);
}

static void iree_hal_passthrough_pool_acquire_rollback_join(void* user_data) {
  iree_hal_passthrough_pool_acquire_t* acquire = user_data;
  if (iree_atomic_fetch_sub(&acquire->pending, 1, iree_memory_order_acq_rel) !=
      1) {
    return;
  }
  for (iree_host_size_t i = 0; i < acquire->native.count; ++i) {
    iree_atomic_fetch_sub(
        &acquire->pool->bytes_committed,
        (int64_t)acquire->native.slabs[i].native_allocation.byte_length,
        iree_memory_order_relaxed);
  }
  iree_atomic_fetch_sub(&acquire->pool->slab_count,
                        (int32_t)acquire->native.count,
                        iree_memory_order_relaxed);
  iree_hal_passthrough_pool_acquire_finish(acquire);
}

static void iree_hal_passthrough_pool_acquire_join(
    iree_hal_passthrough_pool_acquire_t* acquire) {
  if (iree_atomic_fetch_sub(&acquire->pending, 1, iree_memory_order_acq_rel) !=
      1) {
    return;
  }
  if (!iree_status_is_ok(acquire->status)) {
    iree_hal_passthrough_pool_acquire_finish(acquire);
    return;
  }
  iree_hal_passthrough_pool_state_t* pool = acquire->pool;
  const iree_host_size_t count = acquire->native.count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_atomic_fetch_add(
        &pool->bytes_committed,
        (int64_t)acquire->native.slabs[i].native_allocation.byte_length,
        iree_memory_order_relaxed);
  }
  iree_atomic_fetch_add(&pool->slab_count, (int32_t)count,
                        iree_memory_order_relaxed);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(acquire->status);
       ++i) {
    iree_hal_passthrough_pool_reservation_state_t* state = acquire->states[i];
    state->allocation.slab = acquire->native.slabs[i];
    state->allocation.charged_length =
        state->allocation.slab.native_allocation.byte_length;
    if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options) &&
        state->allocation.slab.length != state->asan_layout.backing_length) {
      acquire->status = iree_hal_asan_extend_allocation_layout(
          state->allocation.slab.length, &state->asan_layout);
    }
  }
  if (!iree_status_is_ok(acquire->status)) {
    acquire->rollback = (iree_hal_slab_release_request_t){
        .count = count,
        .slabs = acquire->native.slabs,
        .completion = {iree_hal_passthrough_pool_acquire_rollback_join,
                       acquire},
    };
    iree_atomic_store(&acquire->pending, 2, iree_memory_order_relaxed);
    iree_hal_slab_provider_release_slabs(pool->slab_provider,
                                         &acquire->rollback);
    iree_hal_passthrough_pool_acquire_rollback_join(acquire);
    return;
  }

  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_passthrough_pool_reservation_state_t* state = acquire->states[i];
    iree_hal_passthrough_pool_allocation_state_t* allocation =
        &state->allocation;
    iree_hal_slab_buffer_backing_initialize(
        pool->slab_provider, &allocation->slab, &pool->slab_properties,
        pool->notification, pool->frontier_tracker,
        &allocation->buffer_backing);
    iree_atomic_store(&state->reference_count, 1, iree_memory_order_relaxed);
    iree_hal_passthrough_pool_link_allocation(pool, allocation);
    iree_atomic_fetch_add(&pool->bytes_reserved,
                          (int64_t)allocation->charged_length,
                          iree_memory_order_relaxed);
    acquire->out_reservations[i] = (iree_hal_pool_reservation_t){
        .offset = iree_hal_asan_pool_options_is_enabled(&pool->asan_options)
                      ? state->asan_layout.user_offset
                      : 0,
        .byte_length = acquire->requests[i].allocation_size,
        .block_handle = (uint64_t)(uintptr_t)state,
    };
    acquire->out_infos[i] = (iree_hal_pool_acquire_info_t){
        .result = IREE_HAL_POOL_ACQUIRE_OK_FRESH,
    };
    iree_hal_memory_trace_alloc(
        &pool->trace,
        iree_hal_passthrough_pool_trace_address(
            &allocation->slab, acquire->out_reservations[i].offset),
        acquire->out_reservations[i].byte_length);
  }
  iree_atomic_fetch_add(&pool->reservation_count, (int32_t)count,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->reserve_count, (int64_t)count,
                        iree_memory_order_relaxed);
  iree_hal_passthrough_pool_acquire_finish(acquire);
}

static void iree_hal_passthrough_pool_acquire_completed(void* user_data,
                                                        iree_status_t status) {
  iree_hal_passthrough_pool_acquire_t* acquire = user_data;
  acquire->status = status;
  iree_hal_passthrough_pool_acquire_join(acquire);
}

static iree_status_t iree_hal_passthrough_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_callback_t completion,
    iree_hal_pool_reserve_flags_t flags) {
  (void)requester_frontier;
  (void)flags;
  iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  if (!pool->owned_enabled) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "import-only pool cannot acquire provider-owned backing");
  }
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_passthrough_pool_validate_reservation_request(
        pool, &requests[i]));
  }
  const iree_status_t failure = (iree_status_t)iree_atomic_load(
      &pool->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure)) {
    return iree_status_clone(failure);
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t geometry_offset = 0;
  iree_host_size_t slabs_offset = 0;
  iree_host_size_t states_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_passthrough_pool_acquire_t), &total_size,
      IREE_STRUCT_FIELD(request_count, iree_hal_slab_request_t,
                        &geometry_offset),
      IREE_STRUCT_FIELD(request_count, iree_hal_slab_t, &slabs_offset),
      IREE_STRUCT_FIELD(request_count,
                        iree_hal_passthrough_pool_reservation_state_t*,
                        &states_offset)));
  iree_hal_passthrough_pool_acquire_t* acquire = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator, total_size,
                                             (void**)&acquire));
  memset(acquire, 0, total_size);
  iree_hal_slab_request_t* geometry =
      (iree_hal_slab_request_t*)((uint8_t*)acquire + geometry_offset);
  *acquire = (iree_hal_passthrough_pool_acquire_t){
      .pool = pool,
      .requests = requests,
      .out_reservations = out_reservations,
      .out_infos = out_infos,
      .completion = completion,
      .native =
          {
              .count = request_count,
              .requests = geometry,
              .slabs = (iree_hal_slab_t*)((uint8_t*)acquire + slabs_offset),
              .completion = {iree_hal_passthrough_pool_acquire_completed,
                             acquire},
          },
      .states =
          (iree_hal_passthrough_pool_reservation_state_t**)((uint8_t*)acquire +
                                                            states_offset),
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_passthrough_pool_reservation_state_t* state = NULL;
    status = iree_allocator_malloc(pool->host_allocator, sizeof(*state),
                                   (void**)&state);
    if (iree_status_is_ok(status)) {
      memset(state, 0, sizeof(*state));
      acquire->states[i] = state;
      state->allocation.pool = pool;
      const iree_device_size_t alignment =
          requests[i].min_alignment ? requests[i].min_alignment : 1;
      if (iree_hal_asan_pool_options_is_enabled(&pool->asan_options)) {
        status = iree_hal_asan_calculate_allocation_layout(
            &pool->asan_options, requests[i].allocation_size, alignment,
            &state->asan_layout);
      }
      geometry[i] = (iree_hal_slab_request_t){
          .min_length =
              iree_hal_asan_pool_options_is_enabled(&pool->asan_options)
                  ? state->asan_layout.backing_length
                  : requests[i].allocation_size,
          .min_alignment =
              iree_max(alignment, state->asan_layout.backing_offset_alignment),
      };
    }
  }
  if (iree_status_is_ok(status)) {
    iree_atomic_store(&acquire->pending, 2, iree_memory_order_relaxed);
    iree_hal_slab_provider_acquire_slabs(pool->slab_provider, &acquire->native);
    iree_hal_passthrough_pool_acquire_join(acquire);
  } else {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      iree_allocator_free(pool->host_allocator, acquire->states[i]);
    }
    iree_allocator_free(pool->host_allocator, acquire);
  }
  return status;
}

static void iree_hal_passthrough_pool_release_one_reservation(
    iree_hal_pool_t* base_pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  (void)base_pool;

  iree_hal_passthrough_pool_reservation_state_t* reservation_state =
      (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
          reservation->block_handle;
  iree_hal_passthrough_pool_reservation_state_release_reservation(
      reservation_state, death_frontier);
}

static void iree_hal_passthrough_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_passthrough_pool_release_one_reservation(
        base_pool, &reservations[i], death_frontier);
  }
  iree_async_notification_signal_if_observed(pool->notification, INT32_MAX);
}

//===----------------------------------------------------------------------===//
// Wrap / Query / Trim / Notification
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_passthrough_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);

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
          "reservation %" PRIhsz " has no pass-through state", i);
    }
  }

  const bool transfer_ownership = iree_all_bits_set(
      flags, IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP);
  iree_hal_passthrough_pool_materialize_state_t* state = NULL;
  iree_hal_buffer_t*
      inline_buffers[IREE_HAL_PASSTHROUGH_POOL_INLINE_TRANSACTION_CAPACITY] = {
          0};
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
            reservation_count,
            sizeof(iree_hal_passthrough_pool_materialize_element_t),
            sizeof(*state), &state_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "materialization state size overflow");
    }
    status =
        iree_allocator_malloc(pool->host_allocator, state_size, (void**)&state);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    memset(state, 0, state_size);
    state->host_allocator = pool->host_allocator;
    iree_atomic_store(&state->reference_count, (int32_t)reservation_count,
                      iree_memory_order_relaxed);
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      state->elements[i].state = state;
      state->elements[i].reservation_state =
          (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
              reservations[i]
                  .block_handle;
    }
  } else if (reservation_count > IREE_ARRAYSIZE(inline_buffers)) {
    status = iree_allocator_malloc_array(
        pool->host_allocator, reservation_count, sizeof(*staged_buffers),
        (void**)&staged_buffers);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    staged_buffers_allocated = true;
    memset(staged_buffers, 0, reservation_count * sizeof(*staged_buffers));
  }

  iree_host_size_t materialized_count = 0;
  while (materialized_count < reservation_count && iree_status_is_ok(status)) {
    iree_hal_passthrough_pool_reservation_state_t* reservation_state =
        (iree_hal_passthrough_pool_reservation_state_t*)(uintptr_t)
            reservations[materialized_count]
                .block_handle;
    iree_atomic_fetch_add(&reservation_state->reference_count, 1,
                          iree_memory_order_acq_rel);
    iree_hal_buffer_release_callback_t release_callback = {
        .fn = iree_hal_passthrough_pool_borrowed_view_release,
        .user_data = reservation_state,
    };
    iree_hal_buffer_t** staged_buffer = &staged_buffers[materialized_count];
    if (state) {
      release_callback.fn = iree_hal_passthrough_pool_owned_buffer_release;
      release_callback.user_data = &state->elements[materialized_count];
      staged_buffer = &state->elements[materialized_count].buffer;
    }
    iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_ALL;
    iree_hal_buffer_usage_t allowed_usage =
        pool->slab_properties.supported_usage;
    if (base_pool->memory_contract) {
      allowed_access = base_pool->memory_contract->allowed_access;
      allowed_usage = base_pool->memory_contract->allowed_usage;
    }
    status = iree_hal_slab_provider_wrap_buffer(
        pool->slab_provider, &reservation_state->allocation.slab,
        reservations[materialized_count].offset,
        reservations[materialized_count].byte_length, allowed_access,
        allowed_usage, release_callback, staged_buffer);
    if (iree_status_is_ok(status)) {
      (*staged_buffer)->memory.contract = base_pool->memory_contract;
      (*staged_buffer)->memory.backing =
          &reservation_state->allocation.buffer_backing.facts;
      (*staged_buffer)->memory.offset = reservations[materialized_count].offset;
      ++materialized_count;
    } else {
      iree_hal_passthrough_pool_reservation_state_release_reference(
          reservation_state);
    }
  }

  if (iree_status_is_ok(status)) {
    if (state) {
      state->ownership_committed = true;
    }
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

static void iree_hal_passthrough_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  out_capabilities->memory_type = pool->slab_properties.memory_type;
  out_capabilities->allowed_access = IREE_HAL_MEMORY_ACCESS_ALL;
  out_capabilities->supported_usage = pool->slab_properties.supported_usage;
  out_capabilities->queue_family_affinity =
      pool->slab_properties.queue_family_affinity;
  out_capabilities->atomic_operations = pool->slab_properties.atomic_operations;
  out_capabilities->min_allocation_size = 0;
  out_capabilities->max_allocation_size = 0;
  out_capabilities->max_allocation_alignment =
      pool->slab_properties.max_allocation_alignment;
  out_capabilities->maintenance_alignment =
      pool->slab_properties.maintenance_alignment;
  if (base_pool->memory_contract) {
    out_capabilities->allowed_access =
        base_pool->memory_contract->allowed_access;
    out_capabilities->supported_usage =
        base_pool->memory_contract->allowed_usage;
  }
}

static iree_status_t iree_hal_passthrough_pool_validate_asan(
    const iree_hal_pool_t* base_pool,
    const iree_hal_asan_pool_options_t* options) {
  const iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  return iree_hal_slab_provider_validate_asan_options(pool->slab_provider,
                                                      options);
}

static void iree_hal_passthrough_pool_query_stats(
    const iree_hal_pool_t* base_pool, iree_hal_pool_stats_t* out_stats) {
  const iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  out_stats->bytes_free = 0;
  out_stats->bytes_committed = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_committed, iree_memory_order_relaxed);
  out_stats->budget_limit = 0;
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count =
      (uint32_t)iree_atomic_load(&pool->slab_count, iree_memory_order_relaxed);
  out_stats->reserve_count = (uint64_t)iree_atomic_load(
      &pool->reserve_count, iree_memory_order_relaxed);
  out_stats->release_count = (uint64_t)iree_atomic_load(
      &pool->release_count, iree_memory_order_relaxed);
  out_stats->reuse_count = 0;
  out_stats->reuse_miss_count = 0;
  out_stats->fresh_count = out_stats->reserve_count;
  out_stats->exhausted_count = 0;
  out_stats->over_budget_count = 0;
  out_stats->wait_count = 0;
}

static void iree_hal_passthrough_pool_trim(
    iree_hal_pool_t* base_pool, iree_hal_pool_trim_flags_t flags,
    iree_device_size_t min_bytes_to_keep) {
  iree_hal_passthrough_pool_state_t* pool =
      iree_hal_passthrough_pool_state(base_pool);
  // Every returned slab already has its exact retirement prerequisite. This
  // pool retains no idle backing and trim does not wait for native releases.
  (void)min_bytes_to_keep;
  iree_hal_slab_provider_trim(pool->slab_provider, flags);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_pool_vtable_t iree_hal_passthrough_pool_vtable = {
    .destroy = iree_hal_passthrough_pool_destroy,
    .acquire_reservations = iree_hal_passthrough_pool_acquire_reservations,
    .release_reservations = iree_hal_passthrough_pool_release_reservations,
    .materialize_reservations =
        iree_hal_passthrough_pool_materialize_reservations,
    .import_buffer = iree_hal_passthrough_pool_import_buffer,
    .query_capabilities = iree_hal_passthrough_pool_query_capabilities,
    .validate_asan = iree_hal_passthrough_pool_validate_asan,
    .query_stats = iree_hal_passthrough_pool_query_stats,
    .trim = iree_hal_passthrough_pool_trim,
    .advise_asan_reservations =
        iree_hal_passthrough_pool_advise_asan_reservations,
    .query_import_requirements =
        iree_hal_passthrough_pool_query_import_requirements,
};
