// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/heap_pool.h"

#include "iree/base/internal/math.h"
#include "iree/hal/memory_scope.h"

enum { IREE_HAL_HEAP_POOL_INLINE_TRANSACTION_CAPACITY = 8 };

typedef struct iree_hal_heap_pool_t iree_hal_heap_pool_t;

typedef struct iree_hal_heap_pool_reservation_state_t {
  // Borrowed pool that owns the reservation policy and statistics.
  iree_hal_heap_pool_t* pool;

  // Allocator-owned host storage exposed by |buffer|.
  uint8_t* data;

  // Root buffer created with the reservation and transferred or retained by
  // materialization.
  iree_hal_buffer_t* buffer;

  // Exact number of committed bytes in |data|.
  iree_device_size_t byte_length;

  // True after the complete acquisition transaction publishes this state.
  bool published;

  // Set exactly once when the reservation token is returned or destroyed with
  // its ownership-transferred root buffer.
  iree_atomic_int32_t reservation_released;
} iree_hal_heap_pool_reservation_state_t;

struct iree_hal_heap_pool_t {
  // Base pool resource for vtable dispatch and reference counting.
  iree_hal_pool_t base;

  // Allocator used for aligned payload storage.
  iree_allocator_t data_allocator;

  // Allocator used for pool, reservation, and buffer metadata.
  iree_allocator_t host_allocator;

  // Immutable capabilities reported for routing and validation.
  iree_hal_pool_capabilities_t capabilities;

  // Backing bytes charged to live reservations.
  iree_atomic_int64_t bytes_reserved;

  // Backing bytes retained by live reservations or materialized views.
  iree_atomic_int64_t bytes_committed;

  // Number of reservation tokens not yet returned.
  iree_atomic_int32_t reservation_count;

  // Number of backing allocations not yet released.
  iree_atomic_int32_t allocation_count;

  // Total successful reservation acquisitions.
  iree_atomic_int64_t reserve_count;

  // Total reservation releases.
  iree_atomic_int64_t release_count;
};

static iree_status_t iree_hal_heap_pool_query_memory_pair(
    void* user_data, iree_hal_memory_scope_id_t producer,
    iree_hal_memory_scope_id_t consumer,
    iree_hal_memory_pair_info_t* out_info) {
  (void)user_data;
  (void)producer;
  (void)consumer;
  *out_info = (iree_hal_memory_pair_info_t){
      .flags = IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE |
               IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN,
      .release = {.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE},
      .acquire = {.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE},
      .atomic_reach =
          {
              .scope_32 = IREE_HAL_ATOMIC_REACH_SYSTEM,
              .scope_64 = IREE_HAL_ATOMIC_REACH_SYSTEM,
          },
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_heap_pool_validate_host_access(
    iree_hal_pool_host_access_t host_access) {
  if (!host_access.access || !host_access.modes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "heap pool requires explicit host access and mapping modes");
  }
  if (iree_any_bit_set(host_access.access, ~IREE_HAL_MEMORY_ACCESS_ALL) ||
      iree_any_bit_set(
          host_access.modes,
          ~(IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "heap pool host access is invalid");
  }
  if (host_access.cacheability != IREE_HAL_HOST_CACHEABILITY_UNKNOWN &&
      host_access.cacheability != IREE_HAL_HOST_CACHEABILITY_WRITE_BACK) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "heap pool storage only supports write-back host caching");
  }
  return iree_ok_status();
}

static iree_hal_buffer_usage_t iree_hal_heap_pool_mapping_usage(
    iree_hal_mapping_mode_t modes) {
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM;
  if (iree_any_bit_set(modes, IREE_HAL_MAPPING_MODE_SCOPED)) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  }
  if (iree_any_bit_set(modes, IREE_HAL_MAPPING_MODE_PERSISTENT)) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  }
  return usage;
}

static void iree_hal_heap_pool_record_reservation_release(
    iree_hal_heap_pool_reservation_state_t* state) {
  const int32_t already_released = iree_atomic_exchange(
      &state->reservation_released, 1, iree_memory_order_acq_rel);
  IREE_ASSERT_EQ(already_released, 0);
  iree_hal_heap_pool_t* pool = state->pool;
  iree_atomic_fetch_sub(&pool->bytes_reserved, (int64_t)state->byte_length,
                        iree_memory_order_relaxed);
  iree_atomic_fetch_sub(&pool->reservation_count, 1, iree_memory_order_relaxed);
  iree_atomic_fetch_add(&pool->release_count, 1, iree_memory_order_relaxed);
}

static void iree_hal_heap_pool_buffer_release(void* user_data,
                                              iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_heap_pool_reservation_state_t* state =
      (iree_hal_heap_pool_reservation_state_t*)user_data;
  iree_hal_heap_pool_t* pool = state->pool;
  if (state->published && !iree_atomic_load(&state->reservation_released,
                                            iree_memory_order_acquire)) {
    iree_hal_heap_pool_record_reservation_release(state);
  }
  iree_allocator_free_aligned(pool->data_allocator, state->data);
  if (state->published) {
    iree_atomic_fetch_sub(&pool->bytes_committed, (int64_t)state->byte_length,
                          iree_memory_order_relaxed);
    iree_atomic_fetch_sub(&pool->allocation_count, 1,
                          iree_memory_order_relaxed);
  }
  iree_allocator_free(pool->host_allocator, state);
}

static void iree_hal_heap_pool_destroy(iree_hal_pool_t* base_pool) {
  iree_hal_heap_pool_t* pool = (iree_hal_heap_pool_t*)base_pool;
  IREE_ASSERT_EQ(
      iree_atomic_load(&pool->allocation_count, iree_memory_order_relaxed), 0);
  iree_allocator_t host_allocator = pool->host_allocator;
  iree_hal_pool_deinitialize(base_pool);
  iree_allocator_free(host_allocator, pool);
}

static iree_status_t iree_hal_heap_pool_validate_reservation_request(
    const iree_hal_pool_reservation_request_t* request) {
  const iree_device_size_t alignment =
      request->min_alignment ? request->min_alignment : 1;
  if (!request->allocation_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation size must be > 0");
  }
  if (request->allocation_size > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "heap reservation exceeds host address space");
  }
  if (!iree_device_size_is_power_of_two(alignment) ||
      alignment > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "heap reservation alignment is invalid");
  }
  return iree_ok_status();
}

static void iree_hal_heap_pool_discard_unpublished_reservation(
    iree_hal_heap_pool_t* pool, iree_hal_heap_pool_reservation_state_t* state) {
  if (!state) {
    return;
  }
  if (state->buffer) {
    iree_hal_buffer_release(state->buffer);
  } else {
    iree_allocator_free_aligned(pool->data_allocator, state->data);
    iree_allocator_free(pool->host_allocator, state);
  }
}

static iree_status_t iree_hal_heap_pool_acquire_one_reservation(
    iree_hal_heap_pool_t* pool,
    const iree_hal_pool_reservation_request_t* request,
    iree_hal_heap_pool_reservation_state_t** out_state) {
  *out_state = NULL;
  iree_hal_heap_pool_reservation_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(pool->host_allocator,
                                             sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->pool = pool;
  state->byte_length = request->allocation_size;
  iree_atomic_store(&state->reservation_released, 0, iree_memory_order_relaxed);
  const iree_host_size_t alignment = (iree_host_size_t)iree_max(
      request->min_alignment, IREE_HAL_HEAP_BUFFER_ALIGNMENT);
  iree_status_t status = iree_allocator_malloc_aligned(
      pool->data_allocator, (iree_host_size_t)state->byte_length, alignment,
      /*offset=*/0, (void**)&state->data);
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_release_callback_t release_callback = {
        .fn = iree_hal_heap_pool_buffer_release,
        .user_data = state,
    };
    status = iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(), pool->capabilities.memory_type,
        pool->capabilities.allowed_access, pool->capabilities.supported_usage,
        state->byte_length,
        iree_make_byte_span(state->data, (iree_host_size_t)state->byte_length),
        release_callback, pool->host_allocator, &state->buffer);
  }
  if (iree_status_is_ok(status)) {
    state->buffer->memory.contract = pool->base.memory_contract;
    *out_state = state;
  } else {
    iree_hal_heap_pool_discard_unpublished_reservation(pool, state);
  }
  return status;
}

static iree_status_t iree_hal_heap_pool_acquire_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reservation_t* out_reservations,
    iree_hal_pool_acquire_info_t* out_infos,
    iree_hal_pool_acquire_callback_t completion,
    iree_hal_pool_reserve_flags_t flags) {
  (void)requester_frontier;
  (void)flags;
  iree_hal_heap_pool_t* pool = (iree_hal_heap_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < request_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_heap_pool_validate_reservation_request(&requests[i]));
  }

  iree_hal_heap_pool_reservation_state_t*
      inline_states[IREE_HAL_HEAP_POOL_INLINE_TRANSACTION_CAPACITY] = {0};
  iree_hal_heap_pool_reservation_state_t** states = inline_states;
  bool states_allocated = false;
  iree_status_t status = iree_ok_status();
  if (request_count > IREE_ARRAYSIZE(inline_states)) {
    status = iree_allocator_malloc_array(pool->host_allocator, request_count,
                                         sizeof(*states), (void**)&states);
    states_allocated = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      memset(states, 0, request_count * sizeof(*states));
    }
  }

  iree_host_size_t acquired_count = 0;
  while (acquired_count < request_count && iree_status_is_ok(status)) {
    status = iree_hal_heap_pool_acquire_one_reservation(
        pool, &requests[acquired_count], &states[acquired_count]);
    if (iree_status_is_ok(status)) {
      ++acquired_count;
    }
  }

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      iree_hal_heap_pool_reservation_state_t* state = states[i];
      state->published = true;
      out_reservations[i] = (iree_hal_pool_reservation_t){
          .byte_length = state->byte_length,
          .block_handle = (uint64_t)(uintptr_t)state,
      };
      out_infos[i] = (iree_hal_pool_acquire_info_t){
          .result = IREE_HAL_POOL_ACQUIRE_OK_FRESH,
      };
      iree_atomic_fetch_add(&pool->bytes_reserved, (int64_t)state->byte_length,
                            iree_memory_order_relaxed);
      iree_atomic_fetch_add(&pool->bytes_committed, (int64_t)state->byte_length,
                            iree_memory_order_relaxed);
      iree_atomic_fetch_add(&pool->reservation_count, 1,
                            iree_memory_order_relaxed);
      iree_atomic_fetch_add(&pool->allocation_count, 1,
                            iree_memory_order_relaxed);
      iree_atomic_fetch_add(&pool->reserve_count, 1, iree_memory_order_relaxed);
    }
  } else {
    for (iree_host_size_t i = 0; i < acquired_count; ++i) {
      iree_hal_heap_pool_discard_unpublished_reservation(pool, states[i]);
    }
  }
  if (states_allocated) {
    iree_allocator_free(pool->host_allocator, states);
  }
  if (iree_status_is_ok(status)) {
    completion.fn(completion.user_data, iree_ok_status(),
                  IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  }
  return status;
}

static void iree_hal_heap_pool_release_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_t* reservations,
    const iree_async_frontier_t* death_frontier) {
  (void)base_pool;
  if (death_frontier && death_frontier->entry_count) {
    IREE_ASSERT(false,
                "host-only heap reservations cannot carry queue frontiers");
    return;
  }
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    iree_hal_heap_pool_reservation_state_t* state =
        (iree_hal_heap_pool_reservation_state_t*)(uintptr_t)reservations[i]
            .block_handle;
    iree_hal_buffer_t* buffer = state->buffer;
    iree_hal_heap_pool_record_reservation_release(state);
    iree_hal_buffer_release(buffer);
  }
}

static iree_status_t iree_hal_heap_pool_materialize_reservations(
    iree_hal_pool_t* base_pool, iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffers) {
  iree_hal_heap_pool_t* pool = (iree_hal_heap_pool_t*)base_pool;
  for (iree_host_size_t i = 0; i < reservation_count; ++i) {
    if (!reservations[i].block_handle || reservations[i].offset != 0 ||
        reservations[i].byte_length < requests[i].allocation_size) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid heap pool reservation");
    }
  }

  if (iree_all_bits_set(
          flags,
          IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP)) {
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      iree_hal_heap_pool_reservation_state_t* state =
          (iree_hal_heap_pool_reservation_state_t*)(uintptr_t)reservations[i]
              .block_handle;
      out_buffers[i] = state->buffer;
    }
    return iree_ok_status();
  }

  iree_hal_buffer_t*
      inline_buffers[IREE_HAL_HEAP_POOL_INLINE_TRANSACTION_CAPACITY] = {0};
  iree_hal_buffer_t** staged_buffers = inline_buffers;
  bool staged_buffers_allocated = false;
  iree_status_t status = iree_ok_status();
  if (reservation_count > IREE_ARRAYSIZE(inline_buffers)) {
    status = iree_allocator_malloc_array(
        pool->host_allocator, reservation_count, sizeof(*staged_buffers),
        (void**)&staged_buffers);
    staged_buffers_allocated = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      memset(staged_buffers, 0, reservation_count * sizeof(*staged_buffers));
    }
  }
  iree_host_size_t materialized_count = 0;
  while (materialized_count < reservation_count && iree_status_is_ok(status)) {
    iree_hal_heap_pool_reservation_state_t* state =
        (iree_hal_heap_pool_reservation_state_t*)(uintptr_t)
            reservations[materialized_count]
                .block_handle;
    status = iree_hal_buffer_subspan(
        state->buffer, reservations[materialized_count].offset,
        reservations[materialized_count].byte_length, pool->host_allocator,
        &staged_buffers[materialized_count]);
    if (iree_status_is_ok(status)) {
      ++materialized_count;
    }
  }
  if (iree_status_is_ok(status)) {
    memcpy(out_buffers, staged_buffers,
           reservation_count * sizeof(*out_buffers));
  } else {
    for (iree_host_size_t i = 0; i < materialized_count; ++i) {
      iree_hal_buffer_release(staged_buffers[i]);
    }
  }
  if (staged_buffers_allocated) {
    iree_allocator_free(pool->host_allocator, staged_buffers);
  }
  return status;
}

static void iree_hal_heap_pool_query_capabilities(
    const iree_hal_pool_t* base_pool,
    iree_hal_pool_capabilities_t* out_capabilities) {
  const iree_hal_heap_pool_t* pool = (const iree_hal_heap_pool_t*)base_pool;
  *out_capabilities = pool->capabilities;
}

static void iree_hal_heap_pool_query_stats(const iree_hal_pool_t* base_pool,
                                           iree_hal_pool_stats_t* out_stats) {
  const iree_hal_heap_pool_t* pool = (const iree_hal_heap_pool_t*)base_pool;
  out_stats->bytes_reserved = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_reserved, iree_memory_order_relaxed);
  out_stats->bytes_committed = (iree_device_size_t)iree_atomic_load(
      &pool->bytes_committed, iree_memory_order_relaxed);
  out_stats->reservation_count = (uint32_t)iree_atomic_load(
      &pool->reservation_count, iree_memory_order_relaxed);
  out_stats->slab_count = (uint32_t)iree_atomic_load(&pool->allocation_count,
                                                     iree_memory_order_relaxed);
  out_stats->reserve_count = (uint64_t)iree_atomic_load(
      &pool->reserve_count, iree_memory_order_relaxed);
  out_stats->release_count = (uint64_t)iree_atomic_load(
      &pool->release_count, iree_memory_order_relaxed);
  out_stats->fresh_count = out_stats->reserve_count;
  out_stats->exhausted_count = 0;
}

static void iree_hal_heap_pool_trim(iree_hal_pool_t* base_pool,
                                    iree_hal_pool_trim_flags_t flags,
                                    iree_device_size_t min_bytes_to_keep) {
  (void)base_pool;
  (void)flags;
  (void)min_bytes_to_keep;
}

static const iree_hal_pool_vtable_t iree_hal_heap_pool_vtable = {
    .destroy = iree_hal_heap_pool_destroy,
    .acquire_reservations = iree_hal_heap_pool_acquire_reservations,
    .release_reservations = iree_hal_heap_pool_release_reservations,
    .materialize_reservations = iree_hal_heap_pool_materialize_reservations,
    .query_capabilities = iree_hal_heap_pool_query_capabilities,
    .query_stats = iree_hal_heap_pool_query_stats,
    .trim = iree_hal_heap_pool_trim,
};

IREE_API_EXPORT iree_status_t iree_hal_heap_pool_create(
    iree_hal_pool_host_access_t host_access, iree_allocator_t data_allocator,
    iree_allocator_t host_allocator, iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_heap_pool_validate_host_access(host_access));
  host_access.cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;

  iree_hal_heap_pool_t* pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));
  memset(pool, 0, sizeof(*pool));
  pool->data_allocator = data_allocator;
  pool->host_allocator = host_allocator;
  pool->capabilities = (iree_hal_pool_capabilities_t){
      .memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                     IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                     IREE_HAL_MEMORY_TYPE_HOST_CACHED,
      .allowed_access = host_access.access,
      .supported_usage = iree_hal_heap_pool_mapping_usage(host_access.modes),
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      .atomic_operations = iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL),
      .min_allocation_size = 1,
      .max_allocation_size = IREE_HOST_SIZE_MAX,
      .max_allocation_alignment =
          iree_min(IREE_HOST_SIZE_MAX, IREE_DEVICE_SIZE_MAX) / 2 + 1,
      .maintenance_alignment = 1,
  };

  iree_hal_memory_contract_t* contract = NULL;
  iree_status_t status = iree_hal_memory_contract_create(
      pool, /*scope_count=*/2, iree_hal_heap_buffer_binding_layout(),
      host_allocator, &contract);
  if (iree_status_is_ok(status)) {
    contract->allowed_access = pool->capabilities.allowed_access;
    contract->allowed_usage = pool->capabilities.supported_usage;
    contract->host = host_access;
    contract->scopes[1].usage = pool->capabilities.supported_usage;
    contract->scopes[1].interfaces = UINT32_C(1)
                                     << IREE_HAL_BUFFER_INTERFACE_HOST;
    contract->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_HOST] = 0;
    status = iree_hal_memory_contract_initialize_transitions(
        contract, iree_hal_heap_pool_query_memory_pair, NULL);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_initialize(
        &iree_hal_heap_pool_vtable, contract, /*notification=*/NULL,
        (iree_hal_pool_wait_source_list_t){0}, /*frontier_tracker=*/NULL,
        host_allocator, &pool->base);
  }
  iree_hal_memory_contract_release(contract);

  if (iree_status_is_ok(status)) {
    *out_pool = &pool->base;
  } else {
    iree_allocator_free(host_allocator, pool);
  }
  return status;
}
