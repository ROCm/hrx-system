// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_memory.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"

static void iree_hal_amdgpu_host_queue_populate_memory_event_pool_stats(
    iree_hal_pool_t* pool, iree_hal_profile_memory_event_t* event) {
  if (!pool) {
    return;
  }
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  event->flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_STATS;
  event->pool_bytes_reserved = stats.bytes_reserved;
  event->pool_bytes_free = stats.bytes_free;
  event->pool_bytes_committed = stats.bytes_committed;
  event->pool_budget_limit = stats.budget_limit;
  event->pool_reservation_count = stats.reservation_count;
  event->pool_slab_count = stats.slab_count;
}

static void iree_hal_amdgpu_host_queue_record_memory_event(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_profile_memory_event_type_t type,
    iree_hal_profile_memory_event_flags_t flags, uint32_t result,
    iree_hal_pool_t* pool, iree_hal_buffer_t* buffer,
    iree_device_size_t alignment,
    const iree_hal_pool_reservation_t* reservation, iree_device_size_t length,
    uint64_t submission_id, uint32_t frontier_entry_count) {
  if (!iree_hal_amdgpu_logical_device_should_record_profile_memory_events(
          queue->logical_device)) {
    return;
  }

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  const iree_hal_buffer_allocation_profile_t allocation_profile =
      iree_hal_buffer_allocation_profile(buffer);
  event.type = type;
  event.flags = flags;
  event.result = result;
  event.allocation_id = allocation_profile.id;
  event.pool_id = (uint64_t)(uintptr_t)pool;
  event.submission_id = submission_id;
  event.physical_device_ordinal =
      iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
  event.queue_ordinal = iree_hal_amdgpu_host_queue_profile_queue_ordinal(queue);
  event.frontier_entry_count = frontier_entry_count;
  event.memory_type = iree_hal_buffer_memory_type(buffer);
  event.buffer_usage = iree_hal_buffer_allowed_usage(buffer);
  event.length = length;
  event.alignment = alignment ? alignment : 1;
  if (reservation) {
    event.flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION;
    event.backing_id = reservation->block_handle;
    event.offset = reservation->offset;
    if (type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA &&
        type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA) {
      event.length = reservation->byte_length;
    }
  }
  iree_hal_amdgpu_host_queue_populate_memory_event_pool_stats(pool, &event);
  // Session numbers belong to their originating device. A foreign queue
  // records this allocation in its own active capture using the shared ID.
  const uint64_t session_id =
      iree_hal_buffer_allocation_placement(buffer).device ==
              queue->logical_device
          ? allocation_profile.session_id
          : 0;
  iree_hal_amdgpu_logical_device_record_profile_memory_event_for_session(
      queue->logical_device, session_id, &event);
}

static uint32_t iree_hal_amdgpu_host_size_saturate_u32(iree_host_size_t value) {
  return value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
}

static iree_device_size_t iree_hal_amdgpu_alloca_transaction_total_length(
    const iree_hal_amdgpu_alloca_transaction_t* transaction) {
  iree_device_size_t total_length = 0;
  for (iree_host_size_t i = 0; i < transaction->request_count; ++i) {
    const iree_device_size_t length = transaction->requests[i].allocation_size;
    if (length > UINT64_MAX - total_length) {
      return UINT64_MAX;
    }
    total_length += length;
  }
  return total_length;
}

static iree_device_size_t iree_hal_amdgpu_dealloca_transaction_total_length(
    const iree_hal_amdgpu_dealloca_transaction_t* transaction) {
  iree_device_size_t total_length = 0;
  for (iree_host_size_t i = 0; i < transaction->buffer_count; ++i) {
    const iree_device_size_t length =
        iree_hal_buffer_byte_length(transaction->buffers[i]);
    if (length > UINT64_MAX - total_length) {
      return UINT64_MAX;
    }
    total_length += length;
  }
  return total_length;
}

static iree_hal_amdgpu_host_queue_profile_event_info_t
iree_hal_amdgpu_host_queue_alloca_profile_event_info(
    const iree_hal_amdgpu_alloca_transaction_t* transaction) {
  return (iree_hal_amdgpu_host_queue_profile_event_info_t){
      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA,
      .allocation_id =
          transaction->request_count == 1
              ? iree_hal_buffer_allocation_profile(transaction->buffers[0]).id
              : 0,
      .payload_length =
          iree_hal_amdgpu_alloca_transaction_total_length(transaction),
      .operation_count =
          iree_hal_amdgpu_host_size_saturate_u32(transaction->request_count),
  };
}

static iree_hal_amdgpu_host_queue_profile_event_info_t
iree_hal_amdgpu_host_queue_dealloca_profile_event_info(
    const iree_hal_amdgpu_dealloca_transaction_t* transaction) {
  return (iree_hal_amdgpu_host_queue_profile_event_info_t){
      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA,
      .allocation_id =
          transaction->buffer_count == 1
              ? iree_hal_buffer_allocation_profile(transaction->buffers[0]).id
              : 0,
      .payload_length =
          iree_hal_amdgpu_dealloca_transaction_total_length(transaction),
      .operation_count =
          iree_hal_amdgpu_host_size_saturate_u32(transaction->buffer_count),
  };
}

static void iree_hal_amdgpu_host_queue_commit_transient_buffers(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t status) {
  (void)user_data;
  if (!iree_status_is_ok(status)) {
    return;
  }
  for (uint16_t i = entry->signal_semaphore_count; i < entry->count; ++i) {
    iree_hal_amdgpu_transient_buffer_commit(
        (iree_hal_buffer_t*)entry->resources[i]);
  }
}

static void iree_hal_amdgpu_host_queue_decommit_transient_buffers(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t status) {
  (void)user_data;
  if (!iree_status_is_ok(status)) {
    return;
  }
  for (uint16_t i = entry->signal_semaphore_count; i < entry->count; ++i) {
    iree_hal_buffer_allocation_decommit(
        (iree_hal_buffer_t*)entry->resources[i]);
  }
}

typedef struct iree_hal_amdgpu_host_queue_release_reservations_state_t {
  // Queue whose frontier owns the reuse metadata release.
  iree_hal_amdgpu_host_queue_t* queue;
  // Exact source pool receiving the reservation transaction.
  iree_hal_pool_t* pool;
  // Number of reservations and corresponding buffers in the transaction.
  iree_host_size_t buffer_count;
  // Borrowed transient buffers used for profiling metadata.
  iree_hal_buffer_t* const* buffers;
  // Caller-owned storage containing detached reservation tokens.
  iree_hal_pool_reservation_t* reservations;
} iree_hal_amdgpu_host_queue_release_reservations_state_t;

static void iree_hal_amdgpu_host_queue_release_transient_reservations(
    void* user_data, const iree_async_frontier_t* queue_frontier,
    uint64_t submission_id) {
  iree_hal_amdgpu_host_queue_release_reservations_state_t* state =
      (iree_hal_amdgpu_host_queue_release_reservations_state_t*)user_data;
  iree_hal_pool_release_reservations(state->pool, state->buffer_count,
                                     state->reservations, queue_frontier);
  for (iree_host_size_t i = 0; i < state->buffer_count; ++i) {
    iree_hal_buffer_t* buffer = state->buffers[i];
    iree_hal_amdgpu_host_queue_record_memory_event(
        state->queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX,
        state->pool, buffer, /*alignment=*/1, &state->reservations[i],
        iree_hal_buffer_byte_length(buffer), submission_id,
        queue_frontier ? queue_frontier->entry_count : 0);
    iree_hal_amdgpu_host_queue_record_memory_event(
        state->queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX,
        state->pool, buffer, /*alignment=*/1, &state->reservations[i],
        iree_hal_buffer_byte_length(buffer), submission_id,
        /*frontier_entry_count=*/0);
  }
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_alloca_request(
    const iree_hal_pool_capabilities_t* capabilities, iree_host_size_t index,
    const iree_hal_pool_reservation_request_t* request) {
  if (IREE_UNLIKELY(capabilities->max_allocation_size != 0 &&
                    request->allocation_size >
                        capabilities->max_allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "allocation request %" PRIhsz " size %" PRIdsz
                            " exceeds source pool maximum %" PRIdsz,
                            index, request->allocation_size,
                            capabilities->max_allocation_size);
  }
  if (IREE_UNLIKELY(
          !iree_device_size_is_valid_alignment(request->min_alignment) ||
          request->min_alignment > capabilities->max_allocation_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation request %" PRIhsz " alignment %" PRIdsz
                            " exceeds source pool maximum %" PRIdsz,
                            index, request->min_alignment,
                            capabilities->max_allocation_alignment);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_prepare_alloca_buffers(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* pool,
    iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_pool_reservation_request_t* out_canonical_requests,
    iree_hal_buffer_t** out_buffers) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);

  iree_status_t status = iree_ok_status();
  iree_host_size_t prepared_count = 0;
  while (prepared_count < request_count && iree_status_is_ok(status)) {
    out_canonical_requests[prepared_count] = requests[prepared_count];
    status = iree_hal_amdgpu_host_queue_validate_alloca_request(
        &capabilities, prepared_count, &out_canonical_requests[prepared_count]);
    if (!iree_status_is_ok(status)) {
      break;
    }

    const iree_hal_buffer_placement_t placement = {
        .device = queue->logical_device,
        .queue_family_affinity = capabilities.queue_family_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
    };
    const iree_hal_memory_access_t allowed_access =
        pool->memory_contract ? pool->memory_contract->allowed_access
                              : capabilities.allowed_access;
    const iree_hal_buffer_usage_t allowed_usage =
        pool->memory_contract ? pool->memory_contract->allowed_usage
                              : capabilities.supported_usage;
    status = iree_hal_amdgpu_transient_buffer_create(
        placement, capabilities.memory_type, allowed_access, allowed_usage,
        out_canonical_requests[prepared_count].allocation_size,
        out_canonical_requests[prepared_count].allocation_size, pool,
        queue->transient_buffer_pool, &out_buffers[prepared_count]);
    if (iree_status_is_ok(status)) {
      uint64_t session_id = 0;
      uint64_t allocation_id =
          iree_hal_amdgpu_logical_device_allocate_profile_memory_allocation_id(
              queue->logical_device, &session_id);
      if (!allocation_id) {
        allocation_id = iree_hal_buffer_allocation_next_id();
      }
      iree_hal_amdgpu_transient_buffer_set_profile_allocation(
          out_buffers[prepared_count], session_id, allocation_id);
      ++prepared_count;
    }
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < prepared_count; ++i) {
      iree_hal_buffer_release(out_buffers[i]);
      out_buffers[i] = NULL;
    }
  }
  return status;
}

// Records one event per allocation while using the exact source pool supplied
// separately. Kept distinct from the generic helper above so profiling never
// has to recover policy objects from transient buffer state.
static void iree_hal_amdgpu_host_queue_record_alloca_pool_events(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* pool,
    const iree_hal_amdgpu_alloca_transaction_t* transaction,
    iree_hal_profile_memory_event_type_t type,
    iree_hal_profile_memory_event_flags_t flags, uint64_t submission_id,
    bool has_reservations) {
  for (iree_host_size_t i = 0; i < transaction->request_count; ++i) {
    const iree_async_frontier_t* item_frontier =
        transaction->acquire_infos[i].reuse_frontier;
    iree_hal_amdgpu_host_queue_record_memory_event(
        queue, type, flags, transaction->acquire_result, pool,
        transaction->buffers[i], transaction->requests[i].min_alignment,
        has_reservations ? &transaction->reservations[i] : NULL,
        transaction->requests[i].allocation_size, submission_id,
        item_frontier ? item_frontier->entry_count : 0);
  }
}

iree_status_t iree_hal_amdgpu_host_queue_resolve_alloca_transaction(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction) {
  transaction->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
  transaction->wait_resolution = *resolution;
  iree_async_frontier_initialize(transaction->wait_frontier, 0);
  iree_hal_amdgpu_fixed_frontier_t requester_frontier_storage;
  const iree_async_frontier_t* requester_frontier =
      iree_hal_amdgpu_host_queue_pool_requester_frontier(
          queue, resolution, &requester_frontier_storage);
  iree_hal_profile_memory_event_flags_t reserve_event_flags =
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
  if (transaction->acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
    reserve_event_flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER;
  } else if (transaction->acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
             transaction->acquire_result == IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
    reserve_event_flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION;
  }
  iree_hal_amdgpu_host_queue_record_alloca_pool_events(
      queue, allocation_pool, transaction,
      IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE, reserve_event_flags,
      /*submission_id=*/0, transaction->reservations_held);

  switch (transaction->acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      return iree_ok_status();
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT: {
      for (iree_host_size_t i = 0; i < transaction->request_count; ++i) {
        const iree_async_frontier_t* wait_frontier =
            transaction->acquire_infos[i].reuse_frontier;
        if (transaction->acquire_infos[i].result !=
            IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
          continue;
        }
        if (IREE_UNLIKELY(!wait_frontier || wait_frontier->entry_count == 0)) {
          // Preserve each reservation's original frontier while rolling back
          // the malformed transaction.
          for (iree_host_size_t j = 0; j < transaction->request_count; ++j) {
            iree_hal_pool_release_reservations(
                allocation_pool, 1, &transaction->reservations[j],
                transaction->acquire_infos[j].reuse_frontier);
          }
          transaction->reservations_held = false;
          return iree_make_status(IREE_STATUS_INTERNAL,
                                  "waitable pool reservation %" PRIhsz
                                  " has no dependency edge",
                                  i);
        }
        if (!iree_async_frontier_merge(transaction->wait_frontier, UINT8_MAX,
                                       wait_frontier)) {
          // A transaction-wide dependency is not representable. Preserve each
          // reservation's original frontier while rolling back the rejected
          // transaction.
          for (iree_host_size_t j = 0; j < transaction->request_count; ++j) {
            iree_hal_pool_release_reservations(
                allocation_pool, 1, &transaction->reservations[j],
                transaction->acquire_infos[j].reuse_frontier);
          }
          transaction->reservations_held = false;
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "allocation transaction wait frontier exceeds 255 axes");
        }
      }
      iree_hal_amdgpu_wait_resolution_t candidate_resolution = *resolution;
      if (!iree_hal_pool_requires_asan_advice(allocation_pool) &&
          iree_hal_amdgpu_host_queue_append_pool_wait_frontier_barriers(
              queue, requester_frontier, transaction->wait_frontier,
              &candidate_resolution)) {
        transaction->wait_resolution = candidate_resolution;
      } else {
        transaction->readiness =
            IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT;
        iree_hal_amdgpu_host_queue_record_alloca_pool_events(
            queue, allocation_pool, transaction,
            IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
            IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
                IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER,
            /*submission_id=*/0, /*has_reservations=*/true);
      }
      return iree_ok_status();
    }
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      transaction->readiness =
          IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION;
      iree_hal_amdgpu_host_queue_record_alloca_pool_events(
          queue, allocation_pool, transaction,
          IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION,
          /*submission_id=*/0, /*has_reservations=*/false);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized pool acquire result %u",
                              transaction->acquire_result);
  }
}

iree_status_t iree_hal_amdgpu_host_queue_materialize_alloca_transaction(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction) {
  memset(transaction->backing_buffers, 0,
         transaction->request_count * sizeof(*transaction->backing_buffers));
  IREE_RETURN_IF_ERROR(iree_hal_pool_materialize_reservations(
      allocation_pool, transaction->request_count, transaction->requests,
      transaction->reservations, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      transaction->backing_buffers));
  transaction->backing_buffers_held = true;
  // Legacy pool capabilities do not identify a native owner. Qualify the
  // application-selected storage before commitment consumes its native table.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       !allocation_pool->memory_contract && i < transaction->request_count &&
       iree_status_is_ok(status);
       ++i) {
    iree_hal_buffer_t* backing = transaction->backing_buffers[i];
    iree_hal_buffer_t* root = iree_hal_buffer_allocated_buffer(backing);
    if ((!iree_hal_amdgpu_buffer_isa(root) &&
         !iree_hal_amdgpu_transient_buffer_isa(root)) ||
        iree_hal_buffer_allocation_placement(backing).device !=
            queue->logical_device ||
        !backing->memory.bindings[IREE_HAL_AMDGPU_BUFFER_BINDING_DEVICE_ADDRESS]
             .device_address) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "allocation pool must provide prepared storage from the AMDGPU "
          "queue's device");
    }
  }
  IREE_RETURN_IF_ERROR(status);
  iree_hal_amdgpu_host_queue_record_alloca_pool_events(
      queue, allocation_pool, transaction,
      IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
      /*submission_id=*/0, /*has_reservations=*/true);
  return iree_ok_status();
}

void iree_hal_amdgpu_host_queue_release_alloca_transaction(
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction) {
  if (!transaction) {
    return;
  }
  if (transaction->backing_buffers_held) {
    for (iree_host_size_t i = 0; i < transaction->request_count; ++i) {
      iree_hal_buffer_release(transaction->backing_buffers[i]);
      transaction->backing_buffers[i] = NULL;
    }
    transaction->backing_buffers_held = false;
  }
  if (transaction->reservations_held) {
    for (iree_host_size_t i = 0; i < transaction->request_count; ++i) {
      iree_hal_pool_release_reservations(
          allocation_pool, 1, &transaction->reservations[i],
          transaction->acquire_infos[i].reuse_frontier);
    }
    transaction->reservations_held = false;
  }
}

typedef struct iree_hal_amdgpu_host_queue_alloca_profile_state_t {
  // Queue publishing the allocation epoch.
  iree_hal_amdgpu_host_queue_t* queue;
  // Pool supplying the transaction's reservations.
  iree_hal_pool_t* pool;
  // Borrowed until the synchronous submission callback returns.
  const iree_hal_amdgpu_alloca_transaction_t* transaction;
} iree_hal_amdgpu_host_queue_alloca_profile_state_t;

static void iree_hal_amdgpu_host_queue_record_committed_alloca(
    void* user_data, const iree_async_frontier_t* queue_frontier,
    uint64_t submission_id) {
  (void)queue_frontier;
  const iree_hal_amdgpu_host_queue_alloca_profile_state_t* state =
      (const iree_hal_amdgpu_host_queue_alloca_profile_state_t*)user_data;
  iree_hal_amdgpu_host_queue_record_alloca_pool_events(
      state->queue, state->pool, state->transaction,
      IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, submission_id,
      /*has_reservations=*/true);
}

static uint64_t iree_hal_amdgpu_host_queue_finish_alloca_materialization(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_barrier_submission_t* submission) {
  // Guarded allocations reach this point only after actual reuse completion.
  // No failure remains between publishing their shadow and committing backing.
  iree_hal_pool_advise_asan_reservations(
      allocation_pool, transaction->request_count, transaction->reservations,
      IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
  for (iree_host_size_t i = 0; i < transaction->request_count; ++i) {
    iree_hal_amdgpu_transient_buffer_attach_reservation(
        transaction->buffers[i], allocation_pool,
        &transaction->reservations[i]);
    iree_hal_amdgpu_transient_buffer_stage_backing(
        transaction->buffers[i], transaction->backing_buffers[i]);
    transaction->backing_buffers[i] = NULL;
  }
  transaction->reservations_held = false;
  transaction->backing_buffers_held = false;

  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_alloca_profile_event_info(transaction);
  iree_hal_amdgpu_host_queue_alloca_profile_state_t profile_state = {
      .queue = queue,
      .pool = allocation_pool,
      .transaction = transaction,
  };
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_barrier_submission(
          queue, &transaction->wait_resolution, signal_semaphore_list,
          (iree_hal_amdgpu_reclaim_action_t){
              .fn = iree_hal_amdgpu_host_queue_commit_transient_buffers,
          },
          (iree_hal_resource_t* const*)transaction->buffers,
          transaction->request_count, &profile_event_info,
          (iree_hal_amdgpu_host_queue_post_commit_callback_t){
              .fn = iree_hal_amdgpu_host_queue_record_committed_alloca,
              .user_data = &profile_state,
          },
          /*resource_set=*/NULL, submission_flags, submission);
  profile_event_info.submission_id = submission_epoch;
  iree_hal_amdgpu_host_queue_record_profile_queue_event(
      queue, &transaction->wait_resolution, signal_semaphore_list,
      &profile_event_info);
  transaction->acquire_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  return submission_epoch;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_alloca_materialization(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  *out_ready = false;
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_alloca_profile_event_info(transaction);
  iree_hal_amdgpu_host_queue_barrier_submission_t submission;
  iree_status_t status =
      iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
          queue, &transaction->wait_resolution, signal_semaphore_list,
          transaction->request_count, &profile_event_info, out_ready,
          &submission);
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_release_alloca_transaction(allocation_pool,
                                                          transaction);
    return status;
  }
  iree_hal_amdgpu_host_queue_finish_alloca_materialization(
      queue, transaction, signal_semaphore_list, allocation_pool,
      submission_flags, &submission);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_submit_alloca_transaction(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  *out_ready = false;
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_alloca_profile_event_info(transaction);
  iree_hal_amdgpu_host_queue_barrier_submission_t submission;
  iree_status_t status =
      iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
          queue, &transaction->wait_resolution, signal_semaphore_list,
          transaction->request_count, &profile_event_info, out_ready,
          &submission);
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_release_alloca_transaction(allocation_pool,
                                                          transaction);
    return status;
  }

  status = iree_hal_amdgpu_host_queue_materialize_alloca_transaction(
      queue, allocation_pool, transaction);
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_finish_alloca_materialization(
        queue, transaction, signal_semaphore_list, allocation_pool,
        submission_flags, &submission);
  } else {
    iree_hal_amdgpu_host_queue_fail_barrier_submission(queue, &submission);
    iree_hal_amdgpu_host_queue_release_alloca_transaction(allocation_pool,
                                                          transaction);
  }
  return status;
}

// Sanitizer marking is an execution-visible release effect. Its host-side
// implementation must run after device waits and before the user signals become
// resolvable by either host waiters or another device queue. Ordinary dealloca
// can publish causal reservations at submission; this diagnostic path keeps
// their tokens until the barrier completes and publishes its signals on host.
typedef struct iree_hal_amdgpu_sanitized_dealloca_t {
  // Reclaim-owned state, released after the completion action.
  iree_hal_resource_t resource;
  // Allocator for this state and the cloned signal list.
  iree_allocator_t host_allocator;
  // Queue owning this operation and its profiling identity.
  iree_hal_amdgpu_host_queue_t* queue;
  // Source pool, borrowed for the queued allocation lifetimes.
  iree_hal_pool_t* pool;
  // User signals published only after the logical release effects complete.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Exact operation epoch assigned before the barrier is published.
  uint64_t submission_id;
  // Number of retained buffer roots, zero before submission admission.
  iree_host_size_t buffer_count;
  // Retained roots whose allocation epochs are returned on completion.
  iree_hal_buffer_t* buffers[];
} iree_hal_amdgpu_sanitized_dealloca_t;

static void iree_hal_amdgpu_sanitized_dealloca_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amdgpu_sanitized_dealloca_t* state =
      (iree_hal_amdgpu_sanitized_dealloca_t*)resource;
  for (iree_host_size_t i = 0; i < state->buffer_count; ++i) {
    iree_hal_buffer_release(state->buffers[i]);
  }
  iree_hal_semaphore_list_free(state->signal_semaphore_list,
                               state->host_allocator);
  iree_allocator_free(state->host_allocator, state);
}

static const iree_hal_resource_vtable_t
    iree_hal_amdgpu_sanitized_dealloca_vtable = {
        .destroy = iree_hal_amdgpu_sanitized_dealloca_destroy,
};

static void iree_hal_amdgpu_sanitized_dealloca_committed(
    void* user_data, const iree_async_frontier_t* queue_frontier,
    uint64_t submission_id) {
  iree_hal_amdgpu_sanitized_dealloca_t* state =
      (iree_hal_amdgpu_sanitized_dealloca_t*)user_data;
  state->submission_id = submission_id;
  for (iree_host_size_t i = 0; i < state->buffer_count; ++i) {
    iree_hal_buffer_t* buffer = state->buffers[i];
    iree_hal_amdgpu_host_queue_record_memory_event(
        state->queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX,
        state->pool, buffer, /*alignment=*/1, /*reservation=*/NULL,
        iree_hal_buffer_byte_length(buffer), submission_id,
        /*frontier_entry_count=*/0);
  }
}

static void iree_hal_amdgpu_sanitized_dealloca_complete(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    const iree_status_t status) {
  iree_hal_amdgpu_sanitized_dealloca_t* state =
      (iree_hal_amdgpu_sanitized_dealloca_t*)user_data;
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < state->buffer_count; ++i) {
      iree_hal_buffer_allocation_abort_dealloca(state->buffers[i]);
    }
    iree_hal_semaphore_list_fail(state->signal_semaphore_list,
                                 iree_status_clone(status));
    return;
  }
  for (iree_host_size_t i = 0; i < state->buffer_count; ++i) {
    iree_hal_buffer_t* buffer = state->buffers[i];
    iree_hal_pool_t* source_pool = NULL;
    iree_hal_pool_reservation_t reservation;
    iree_hal_buffer_allocation_take_dealloca_reservation(buffer, &source_pool,
                                                         &reservation);
    iree_hal_buffer_allocation_decommit(buffer);
    // The barrier has completed every prior use. Empty history here records
    // actual retirement, independent of later queue-frontier publication.
    iree_hal_pool_advise_asan_reservations(
        source_pool, 1, &reservation, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
    iree_hal_pool_release_reservations(source_pool, 1, &reservation,
                                       /*death_frontier=*/NULL);
    iree_hal_amdgpu_host_queue_record_memory_event(
        state->queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX,
        source_pool, buffer, /*alignment=*/1, &reservation,
        iree_hal_buffer_byte_length(buffer), state->submission_id,
        /*frontier_entry_count=*/0);
  }
  iree_status_t signal_status = iree_hal_semaphore_list_signal(
      state->signal_semaphore_list, /*frontier=*/NULL);
  if (!iree_status_is_ok(signal_status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, signal_status);
  }
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_sanitized_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_dealloca_transaction_t* transaction,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  *out_ready = false;
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_sanitized_dealloca_t), &state_size,
      IREE_STRUCT_FIELD(transaction->buffer_count, iree_hal_buffer_t*, NULL)));
  iree_hal_amdgpu_sanitized_dealloca_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(queue->host_allocator, state_size, (void**)&state));
  iree_hal_resource_initialize(&iree_hal_amdgpu_sanitized_dealloca_vtable,
                               &state->resource);
  state->host_allocator = queue->host_allocator;
  state->queue = queue;
  state->pool = transaction->pool;
  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, queue->host_allocator,
      &state->signal_semaphore_list);
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_dealloca_profile_event_info(transaction);
  iree_hal_amdgpu_host_queue_barrier_submission_t submission;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
        queue, resolution, iree_hal_semaphore_list_empty(),
        /*operation_resource_count=*/1, &profile_event_info, out_ready,
        &submission);
  }
  if (iree_status_is_ok(status) && *out_ready) {
    const bool retain_resources = iree_any_bit_set(
        submission_flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
    state->buffer_count = transaction->buffer_count;
    for (iree_host_size_t i = 0; i < state->buffer_count; ++i) {
      state->buffers[i] = transaction->buffers[i];
      if (retain_resources) {
        iree_hal_buffer_retain(state->buffers[i]);
      }
    }
    iree_hal_resource_t* resources[] = {&state->resource};
    profile_event_info.submission_id =
        iree_hal_amdgpu_host_queue_finish_barrier_submission(
            queue, resolution, iree_hal_semaphore_list_empty(),
            (iree_hal_amdgpu_reclaim_action_t){
                .fn = iree_hal_amdgpu_sanitized_dealloca_complete,
                .user_data = state,
            },
            resources, IREE_ARRAYSIZE(resources), &profile_event_info,
            (iree_hal_amdgpu_host_queue_post_commit_callback_t){
                .fn = iree_hal_amdgpu_sanitized_dealloca_committed,
                .user_data = state,
            },
            /*resource_set=*/NULL,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &submission);
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        queue, resolution, signal_semaphore_list, &profile_event_info);
    if (!retain_resources) {
      // The cloned list owns the references needed through host publication.
      for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
        iree_hal_semaphore_release(signal_semaphore_list.semaphores[i]);
      }
    }
  } else {
    iree_hal_resource_release(&state->resource);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_dealloca_transaction_t* transaction,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  *out_ready = false;
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  if (iree_hal_amdgpu_asan_state_is_enabled(&logical_device->asan)) {
    return iree_hal_amdgpu_host_queue_submit_sanitized_dealloca(
        queue, resolution, signal_semaphore_list, transaction, submission_flags,
        out_ready);
  }
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_dealloca_profile_event_info(transaction);
  iree_hal_amdgpu_host_queue_barrier_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
      queue, resolution, signal_semaphore_list, transaction->buffer_count,
      &profile_event_info, out_ready, &submission));
  if (!*out_ready) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < transaction->buffer_count; ++i) {
    iree_hal_pool_t* source_pool = NULL;
    iree_hal_buffer_allocation_take_dealloca_reservation(
        transaction->buffers[i], &source_pool, &transaction->reservations[i]);
    IREE_ASSERT_TRUE(source_pool == transaction->pool);
  }
  iree_hal_amdgpu_host_queue_release_reservations_state_t release_state = {
      .queue = queue,
      .pool = transaction->pool,
      .buffer_count = transaction->buffer_count,
      .buffers = transaction->buffers,
      .reservations = transaction->reservations,
  };
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_barrier_submission(
          queue, resolution, signal_semaphore_list,
          (iree_hal_amdgpu_reclaim_action_t){
              .fn = iree_hal_amdgpu_host_queue_decommit_transient_buffers,
          },
          (iree_hal_resource_t* const*)transaction->buffers,
          transaction->buffer_count, &profile_event_info,
          (iree_hal_amdgpu_host_queue_post_commit_callback_t){
              .fn = iree_hal_amdgpu_host_queue_release_transient_reservations,
              .user_data = &release_state,
          },
          /*resource_set=*/NULL, submission_flags, &submission);
  profile_event_info.submission_id = submission_epoch;
  iree_hal_amdgpu_host_queue_record_profile_queue_event(
      queue, resolution, signal_semaphore_list, &profile_event_info);
  return iree_ok_status();
}
