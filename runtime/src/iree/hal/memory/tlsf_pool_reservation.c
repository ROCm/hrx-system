// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf_pool_reservation.h"

// Dedicated backing extends only dedicated records. Ordinary suballocation
// records retain their compact shape and refer to their slab's prepared range.
typedef struct iree_hal_tlsf_pool_dedicated_backing_t {
  // Parent reservation held for this complete allocation epoch.
  iree_hal_pool_reservation_t reservation;
  // Prepared parent view retained until the dedicated record is returned.
  iree_hal_pool_buffer_range_t range;
} iree_hal_tlsf_pool_dedicated_backing_t;

iree_hal_tlsf_pool_reservation_layout_t iree_hal_tlsf_pool_reservation_layout(
    uint8_t frontier_capacity) {
  const iree_host_size_t frontier_offset =
      iree_host_align(sizeof(iree_hal_tlsf_pool_release_node_t),
                      iree_alignof(iree_async_frontier_entry_t));
  return (iree_hal_tlsf_pool_reservation_layout_t){
      .node_size = frontier_offset + sizeof(iree_async_frontier_t) +
                   frontier_capacity * sizeof(iree_async_frontier_entry_t),
      .frontier =
          {
              .offset = frontier_offset,
              .capacity = frontier_capacity,
          },
  };
}

static iree_hal_tlsf_pool_dedicated_backing_t*
iree_hal_tlsf_pool_dedicated_backing(
    const iree_hal_tlsf_pool_release_node_t* node,
    const iree_hal_tlsf_pool_reservation_layout_t* layout) {
  return (iree_hal_tlsf_pool_dedicated_backing_t*)((uint8_t*)node +
                                                   layout->node_size);
}

iree_status_t iree_hal_tlsf_pool_reservation_geometry(
    const iree_hal_pool_reservation_request_t* request,
    iree_device_size_t minimum_alignment,
    const iree_hal_pool_capabilities_t* capabilities,
    const iree_hal_asan_pool_options_t* asan,
    iree_hal_tlsf_pool_request_geometry_t* out_geometry) {
  const iree_device_size_t size = request->allocation_size;
  const iree_device_size_t alignment =
      request->min_alignment ? request->min_alignment : 1;
  if (!size || !iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TLSF reservation requires nonzero size and "
                            "power-of-two alignment");
  }
  if (alignment > capabilities->max_allocation_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation alignment exceeds backing support");
  }
  out_geometry->alignment = iree_max(alignment, minimum_alignment);
  out_geometry->length = size;
  if (iree_hal_asan_pool_options_is_enabled(asan)) {
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        asan, size, alignment, &out_geometry->asan));
    out_geometry->length = out_geometry->asan.backing_length;
  }
  if (!iree_device_size_checked_align(out_geometry->length,
                                      out_geometry->alignment,
                                      &out_geometry->length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TLSF reservation size overflows alignment");
  }
  const iree_device_size_t backing_limit = capabilities->max_allocation_size;
  if (backing_limit && out_geometry->length > backing_limit) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reservation backing exceeds source capacity");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_tlsf_pool_dedicated_prepare(
    iree_hal_pool_t* backing_pool,
    const iree_hal_pool_reservation_request_t* backing_request,
    const iree_hal_asan_pool_options_t* asan,
    const iree_hal_asan_allocation_layout_t* asan_layout,
    const iree_hal_tlsf_pool_reservation_layout_t* layout,
    iree_hal_pool_reservation_t reservation, iree_hal_pool_acquire_info_t info,
    const iree_async_frontier_t* initial_frontier,
    iree_allocator_t host_allocator,
    iree_hal_tlsf_pool_release_node_t** out_node) {
  *out_node = NULL;
  iree_hal_tlsf_pool_release_node_t* node = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      layout->node_size + sizeof(iree_hal_tlsf_pool_dedicated_backing_t),
      (void**)&node);
  if (!iree_status_is_ok(status)) {
    iree_hal_pool_release_reservations(backing_pool, 1, &reservation,
                                       info.reuse_frontier);
    return status;
  }
  iree_hal_tlsf_pool_dedicated_backing_t* backing =
      iree_hal_tlsf_pool_dedicated_backing(node, layout);
  backing->reservation = reservation;
  status = iree_hal_pool_buffer_range_prepare(
      backing_pool, backing_request, &backing->reservation,
      backing_request->min_alignment, asan, &backing->range);
  backing->range.memory.reuse_frontier = info.reuse_frontier;
  if (iree_status_is_ok(status) &&
      iree_hal_asan_pool_options_is_enabled(asan)) {
    node->asan_layout = *asan_layout;
    node->asan_layout.backing_length = backing->range.length;
    node->asan_layout.right_redzone_length = backing->range.length -
                                             node->asan_layout.user_offset -
                                             node->asan_layout.user_length;
  }
  if (iree_status_is_ok(status)) {
    node->range = &backing->range;
    node->block_index = IREE_HAL_MEMORY_TLSF_BLOCK_INDEX_NONE;
    node->charged_length = backing->range.length;
    iree_async_frontier_t* frontier =
        iree_hal_tlsf_pool_reservation_frontier(node, layout->frontier.offset);
    if (initial_frontier) {
      memcpy(frontier, initial_frontier,
             sizeof(*frontier) + initial_frontier->entry_count *
                                     sizeof(iree_async_frontier_entry_t));
    } else {
      iree_async_frontier_initialize(frontier, 0);
    }
    *out_node = node;
  } else {
    iree_hal_pool_buffer_range_deinitialize(&backing->range);
    iree_hal_pool_release_reservations(backing_pool, 1, &backing->reservation,
                                       info.reuse_frontier);
    iree_allocator_free(host_allocator, node);
  }
  return status;
}

bool iree_hal_tlsf_pool_dedicated_merge_return_frontier(
    iree_hal_tlsf_pool_release_node_t* node,
    const iree_hal_tlsf_pool_reservation_layout_t* layout) {
  iree_async_frontier_t* frontier =
      iree_hal_tlsf_pool_reservation_frontier(node, layout->frontier.offset);
  if (frontier->entry_count > layout->frontier.capacity) {
    return false;
  }
  const iree_hal_tlsf_pool_dedicated_backing_t* backing =
      iree_hal_tlsf_pool_dedicated_backing(node, layout);
  return backing->range.length == backing->reservation.byte_length ||
         !backing->range.memory.reuse_frontier ||
         iree_async_frontier_merge(frontier, layout->frontier.capacity,
                                   backing->range.memory.reuse_frontier);
}

iree_device_size_t iree_hal_tlsf_pool_dedicated_backing_length(
    const iree_hal_tlsf_pool_release_node_t* node,
    const iree_hal_tlsf_pool_reservation_layout_t* layout) {
  return iree_hal_tlsf_pool_dedicated_backing(node, layout)
      ->reservation.byte_length;
}

void iree_hal_tlsf_pool_dedicated_release(
    iree_hal_pool_t* backing_pool, iree_hal_tlsf_pool_release_node_t* node,
    const iree_hal_tlsf_pool_reservation_layout_t* layout,
    iree_allocator_t host_allocator) {
  iree_hal_tlsf_pool_dedicated_backing_t* backing =
      iree_hal_tlsf_pool_dedicated_backing(node, layout);
  const iree_async_frontier_t* frontier =
      iree_hal_tlsf_pool_reservation_frontier(node, layout->frontier.offset);
  iree_hal_pool_buffer_range_deinitialize(&backing->range);
  iree_hal_pool_release_reservations(backing_pool, 1, &backing->reservation,
                                     frontier->entry_count ? frontier : NULL);
  iree_allocator_free(host_allocator, node);
}

enum { IREE_HAL_TLSF_POOL_RESERVATION_INLINE_CAPACITY = 8 };

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

static void iree_hal_tlsf_pool_buffer_release(void* user_data,
                                              iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_hal_tlsf_pool_materialize_element_t* element =
      (iree_hal_tlsf_pool_materialize_element_t*)user_data;
  iree_hal_tlsf_pool_materialize_state_t* state = element->state;
  if (state->ownership_committed) {
    iree_hal_pool_advise_asan_reservations(
        state->pool, 1, &element->reservation,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
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

iree_status_t iree_hal_tlsf_pool_reservation_materialize(
    iree_hal_pool_t* base_pool, iree_host_size_t frontier_offset,
    iree_host_size_t reservation_count,
    const iree_hal_pool_reservation_request_t* requests,
    const iree_hal_pool_reservation_t* reservations,
    iree_hal_pool_materialize_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffers) {
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
    if (!release_node->range) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "reservation %" PRIhsz " has no prepared range",
                              i);
    }
  }

  const bool transfer_ownership = iree_all_bits_set(
      flags, IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP);
  iree_hal_tlsf_pool_materialize_state_t* state = NULL;
  iree_hal_buffer_t*
      inline_buffers[IREE_HAL_TLSF_POOL_RESERVATION_INLINE_CAPACITY] = {0};
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
    status = iree_allocator_malloc(host_allocator, state_size, (void**)&state);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    memset(state, 0, state_size);
    state->pool = base_pool;
    state->host_allocator = host_allocator;
    iree_atomic_store(&state->reference_count, (int32_t)reservation_count,
                      iree_memory_order_relaxed);
    for (iree_host_size_t i = 0; i < reservation_count; ++i) {
      state->elements[i].state = state;
      state->elements[i].reservation = reservations[i];
    }
  } else if (reservation_count > IREE_ARRAYSIZE(inline_buffers)) {
    status = iree_allocator_malloc_array(host_allocator, reservation_count,
                                         sizeof(*staged_buffers),
                                         (void**)&staged_buffers);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    staged_buffers_allocated = true;
    memset(staged_buffers, 0, reservation_count * sizeof(*staged_buffers));
  }

  iree_host_size_t materialized_count = 0;
  while (materialized_count < reservation_count && iree_status_is_ok(status)) {
    iree_hal_tlsf_pool_release_node_t* release_node =
        (iree_hal_tlsf_pool_release_node_t*)(uintptr_t)
            reservations[materialized_count]
                .block_handle;
    const iree_hal_pool_buffer_range_t* range = release_node->range;
    iree_hal_buffer_release_callback_t release_callback =
        iree_hal_buffer_release_callback_null();
    iree_hal_buffer_t** staged_buffer = &staged_buffers[materialized_count];
    if (state) {
      release_callback.fn = iree_hal_tlsf_pool_buffer_release;
      release_callback.user_data = &state->elements[materialized_count];
      staged_buffer = &state->elements[materialized_count].buffer;
    }
    status = iree_hal_pool_buffer_range_materialize(
        range, reservations[materialized_count].offset,
        reservations[materialized_count].byte_length,
        iree_hal_tlsf_pool_reservation_frontier(release_node, frontier_offset),
        release_callback, host_allocator, staged_buffer);
    if (iree_status_is_ok(status)) {
      if ((*staged_buffer)->memory.reuse_frontier &&
          (*staged_buffer)->memory.reuse_frontier->entry_count == 0) {
        (*staged_buffer)->memory.reuse_frontier = NULL;
      }
      ++materialized_count;
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
      iree_allocator_free(host_allocator, state);
    }
  }
  if (staged_buffers_allocated) {
    iree_allocator_free(host_allocator, staged_buffers);
  }
  return status;
}
