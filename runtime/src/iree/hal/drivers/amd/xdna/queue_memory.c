// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_admission.h"
#include "iree/hal/drivers/amd/xdna/queue_state.h"
#include "iree/hal/drivers/amd/xdna/transient_buffer.h"

void iree_hal_amd_xdna_operation_release_alloca_reservations(
    iree_hal_amd_xdna_operation_t* operation) {
  if (!operation->alloca.reservations_held) {
    return;
  }
  for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
    iree_hal_pool_release_reservations(
        operation->alloca.pool, 1, &operation->alloca.reservations[i],
        operation->alloca.acquire_infos[i].reuse_frontier);
  }
  operation->alloca.reservations_held = false;
}

static void iree_hal_amd_xdna_operation_merge_memory_frontiers(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_async_frontier_t* target =
      iree_async_fixed_frontier_as_frontier(&operation->frontier.frontier);
  for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
    const iree_async_frontier_t* source =
        operation->alloca.acquire_infos[i].reuse_frontier;
    if (source && !iree_async_frontier_merge(
                      target, IREE_HAL_AMD_XDNA_FRONTIER_CAPACITY, source)) {
      operation->frontier.exact = false;
    }
  }
}

static iree_status_t iree_hal_amd_xdna_operation_prepare_frontier_wait(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_memory_wait_t* wait = operation->alloca.memory_wait;
  if (!operation->alloca.pool->frontier_tracker) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA allocation pool returned a reuse frontier without a tracker");
  }

  if (!wait->frontier) {
    iree_host_size_t frontier_size = 0;
    IREE_RETURN_IF_ERROR(iree_async_frontier_size(UINT8_MAX, &frontier_size));
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, frontier_size, (void**)&wait->frontier));
  }
  iree_async_frontier_initialize(wait->frontier, 0);
  for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
    const iree_hal_pool_acquire_info_t* info =
        &operation->alloca.acquire_infos[i];
    if (info->result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT &&
        !iree_async_frontier_merge(wait->frontier, UINT8_MAX,
                                   info->reuse_frontier)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "XDNA allocation dependency frontier exceeds %u axes", UINT8_MAX);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_operation_prepare_alloca_storage(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_status_t status = iree_ok_status();
  const bool has_reservation_views = iree_hal_pool_query_reservation_views(
      operation->alloca.pool, operation->alloca.request_count,
      operation->alloca.reservations, operation->alloca.reservation_views);
  if (!has_reservation_views) {
    status = iree_hal_pool_materialize_reservations(
        operation->alloca.pool, operation->alloca.request_count,
        operation->alloca.requests, operation->alloca.reservations,
        IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
        operation->alloca.materialized_buffers);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
        iree_hal_buffer_t* materialized =
            operation->alloca.materialized_buffers[i];
        operation->alloca.reservation_views[i] =
            (iree_hal_pool_reservation_view_t){
                .buffer = materialized,
                .byte_offset = iree_hal_buffer_byte_offset(materialized),
                .byte_length = iree_hal_buffer_byte_length(materialized),
                .memory = iree_hal_buffer_memory_view(materialized),
            };
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_pool_advise_asan_reservations(
        operation->alloca.pool, operation->alloca.request_count,
        operation->alloca.reservations,
        IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
    for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
      iree_hal_amd_xdna_transient_buffer_attach_reservation(
          operation->alloca.transient_buffers[i], operation->alloca.pool,
          &operation->alloca.reservations[i]);
    }
    operation->alloca.reservations_held = false;
    for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
      iree_hal_amd_xdna_transient_buffer_stage_backing(
          operation->alloca.transient_buffers[i],
          &operation->alloca.reservation_views[i]);
      iree_hal_buffer_release(operation->alloca.materialized_buffers[i]);
      operation->alloca.materialized_buffers[i] = NULL;
    }
    for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
      iree_hal_amd_xdna_transient_buffer_commit(
          operation->alloca.transient_buffers[i]);
    }
    iree_hal_amd_xdna_operation_merge_memory_frontiers(operation);
  } else {
    for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
      iree_hal_buffer_release(operation->alloca.materialized_buffers[i]);
      operation->alloca.materialized_buffers[i] = NULL;
    }
    iree_hal_amd_xdna_operation_release_alloca_reservations(operation);
  }
  return status;
}

static void iree_hal_amd_xdna_operation_execute_dealloca(
    iree_hal_amd_xdna_operation_t* operation) {
  for (iree_host_size_t i = 0; i < operation->dealloca.buffer_count; ++i) {
    iree_hal_pool_t* source_pool = NULL;
    iree_hal_buffer_allocation_take_dealloca_reservation(
        operation->dealloca.transient_buffers[i], &source_pool,
        &operation->dealloca.reservations[i]);
    IREE_ASSERT_TRUE(source_pool == operation->dealloca.pool);
  }
  operation->dealloca.marks_owned = false;
  for (iree_host_size_t i = 0; i < operation->dealloca.buffer_count; ++i) {
    iree_hal_buffer_allocation_decommit(
        operation->dealloca.transient_buffers[i]);
  }
  iree_hal_pool_advise_asan_reservations(
      operation->dealloca.pool, operation->dealloca.buffer_count,
      operation->dealloca.reservations,
      IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
  iree_hal_pool_release_reservations(
      operation->dealloca.pool, operation->dealloca.buffer_count,
      operation->dealloca.reservations, /*death_frontier=*/NULL);
}

// A signal can resume memory work while its producer holds queue state.
// The terminal failure is immutable, so this path never reacquires that lock.
static iree_status_t iree_hal_amd_xdna_memory_check_failure(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  return iree_status_is_ok(status)
             ? iree_status_clone(
                   iree_hal_amd_xdna_queue_failure_status(operation->queue))
             : status;
}

static void iree_hal_amd_xdna_memory_complete(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  operation->status = status;
  if (iree_status_is_ok(status)) {
    iree_hal_amd_xdna_operation_profile_ready(operation);
  }
  iree_hal_amd_xdna_operation_complete(operation);
}

static void iree_hal_amd_xdna_memory_wait_complete(void* user_data,
                                                   iree_status_t status) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  operation->alloca.memory_wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE;
  iree_hal_amd_xdna_operation_issue_memory(operation, status);
}

static void iree_hal_amd_xdna_memory_finish_acquire(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_memory_wait_t* wait = operation->alloca.memory_wait;
  iree_status_t status = operation->alloca.acquire_status;
  operation->alloca.acquire_status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    operation->alloca.reservations_held =
        operation->alloca.acquire_result == IREE_HAL_POOL_ACQUIRE_OK ||
        operation->alloca.acquire_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
        operation->alloca.acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
  }
  status = iree_hal_amd_xdna_memory_check_failure(operation, status);
  if (!iree_status_is_ok(status)) {
    iree_hal_pool_wait_abort(wait->capacity_wait);
    wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE;
    iree_hal_amd_xdna_memory_complete(operation, status);
    return;
  }

  switch (operation->alloca.acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_hal_pool_wait_abort(wait->capacity_wait);
      wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE;
      status = iree_hal_amd_xdna_operation_prepare_alloca_storage(operation);
      break;
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      iree_hal_pool_wait_abort(wait->capacity_wait);
      wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE;
      status = iree_hal_amd_xdna_operation_prepare_frontier_wait(operation);
      if (iree_status_is_ok(status)) {
        wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_FRONTIER;
        status = iree_async_frontier_tracker_wait(
            operation->alloca.pool->frontier_tracker, wait->frontier,
            iree_hal_amd_xdna_memory_wait_complete, operation,
            &wait->frontier_waiter);
        if (iree_status_is_ok(status)) {
          return;
        }
        wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE;
      }
      break;
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_CAPACITY;
      iree_hal_pool_wait_commit(
          wait->capacity_wait, iree_infinite_timeout(),
          (iree_hal_pool_wait_callback_t){
              .fn = iree_hal_amd_xdna_memory_wait_complete,
              .user_data = operation,
          });
      return;
  }
  iree_hal_amd_xdna_memory_complete(operation, status);
}

static void iree_hal_amd_xdna_memory_acquire_complete(
    void* user_data, iree_status_t status,
    iree_hal_pool_acquire_result_t result) {
  iree_hal_amd_xdna_operation_t* operation = user_data;
  operation->alloca.acquire_status = status;
  operation->alloca.acquire_result = result;
  if (iree_atomic_fetch_sub(&operation->alloca.acquire_pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_hal_amd_xdna_memory_finish_acquire(operation);
  }
}

void iree_hal_amd_xdna_operation_issue_memory(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status) {
  status = iree_hal_amd_xdna_memory_check_failure(operation, status);
  if (!iree_status_is_ok(status)) {
    iree_hal_amd_xdna_memory_complete(operation, status);
    return;
  }
  if (operation->kind == IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA) {
    iree_hal_amd_xdna_operation_execute_dealloca(operation);
    iree_hal_amd_xdna_memory_complete(operation, iree_ok_status());
    return;
  }
  if (operation->alloca.reservations_held) {
    status = iree_hal_amd_xdna_operation_prepare_alloca_storage(operation);
    iree_hal_amd_xdna_memory_complete(operation, status);
    return;
  }

  iree_hal_amd_xdna_memory_wait_t* wait = operation->alloca.memory_wait;
  iree_hal_pool_wait_prepare(wait->capacity_wait);
  wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_CAPACITY_PREPARED;
  iree_atomic_store(&operation->alloca.acquire_pending, 2,
                    iree_memory_order_relaxed);
  status = iree_hal_pool_acquire_reservations(
      operation->alloca.pool, operation->alloca.request_count,
      operation->alloca.requests,
      iree_hal_amd_xdna_frontier_state_as_frontier(&operation->frontier),
      operation->alloca.reservations, operation->alloca.acquire_infos,
      (iree_hal_pool_acquire_callback_t){
          .fn = iree_hal_amd_xdna_memory_acquire_complete,
          .user_data = operation,
      },
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER);
  if (!iree_status_is_ok(status)) {
    iree_hal_pool_wait_abort(wait->capacity_wait);
    wait->kind = IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE;
    iree_hal_amd_xdna_memory_complete(operation, status);
    return;
  }
  // No input/output storage can be released until both submission and its
  // possibly inline completion have made their final borrowed access.
  if (iree_atomic_fetch_sub(&operation->alloca.acquire_pending, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_hal_amd_xdna_memory_finish_acquire(operation);
  }
}

static iree_status_t iree_hal_amd_xdna_validate_alloca_request(
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

iree_status_t iree_hal_amd_xdna_queue_alloca(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** IREE_RESTRICT out_buffers) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);

  iree_hal_amd_xdna_operation_t* operation = NULL;
  iree_status_t status = iree_hal_amd_xdna_operation_allocate(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_ALLOCA, &operation);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  operation->alloca.pool = pool;
  operation->alloca.request_count = request_count;

  // Hold the references intended for the caller outside the captured
  // operation. Submission consumes the operation on both success and failure,
  // while the public contract requires every output entry to remain untouched
  // on synchronous failure.
  iree_hal_amd_xdna_queue_capture_t output_capture;
  iree_hal_amd_xdna_queue_capture_initialize(operation->capture.storage,
                                             &output_capture);
  iree_hal_buffer_t** captured_outputs = NULL;
  iree_host_size_t outputs_size = 0;
  status = IREE_STRUCT_LAYOUT(
      0, &outputs_size,
      IREE_STRUCT_FIELD_FAM(request_count, iree_hal_buffer_t*));
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &output_capture, outputs_size, (void**)&captured_outputs);
  }
  if (iree_status_is_ok(status)) {
    memset(captured_outputs, 0, request_count * sizeof(*captured_outputs));
  }

  iree_host_size_t storage_size = 0;
  iree_host_size_t requests_offset = 0;
  iree_host_size_t buffers_offset = 0;
  iree_host_size_t reservations_offset = 0;
  iree_host_size_t infos_offset = 0;
  iree_host_size_t views_offset = 0;
  iree_host_size_t materialized_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        0, &storage_size,
        IREE_STRUCT_FIELD_ALIGNED(
            request_count, iree_hal_pool_reservation_request_t,
            iree_alignof(iree_hal_pool_reservation_request_t),
            &requests_offset),
        IREE_STRUCT_FIELD_ALIGNED(request_count, iree_hal_buffer_t*,
                                  iree_alignof(iree_hal_buffer_t*),
                                  &buffers_offset),
        IREE_STRUCT_FIELD_ALIGNED(request_count, iree_hal_pool_reservation_t,
                                  iree_alignof(iree_hal_pool_reservation_t),
                                  &reservations_offset),
        IREE_STRUCT_FIELD_ALIGNED(request_count, iree_hal_pool_acquire_info_t,
                                  iree_alignof(iree_hal_pool_acquire_info_t),
                                  &infos_offset),
        IREE_STRUCT_FIELD_ALIGNED(
            request_count, iree_hal_pool_reservation_view_t,
            iree_alignof(iree_hal_pool_reservation_view_t), &views_offset),
        IREE_STRUCT_FIELD_ALIGNED(request_count, iree_hal_buffer_t*,
                                  iree_alignof(iree_hal_buffer_t*),
                                  &materialized_offset));
  }
  uint8_t* storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, storage_size, (void**)&storage);
  }
  if (iree_status_is_ok(status)) {
    memset(storage, 0, storage_size);
    operation->alloca.requests =
        (iree_hal_pool_reservation_request_t*)(storage + requests_offset);
    operation->alloca.transient_buffers =
        (iree_hal_buffer_t**)(storage + buffers_offset);
    operation->alloca.reservations =
        (iree_hal_pool_reservation_t*)(storage + reservations_offset);
    operation->alloca.acquire_infos =
        (iree_hal_pool_acquire_info_t*)(storage + infos_offset);
    operation->alloca.reservation_views =
        (iree_hal_pool_reservation_view_t*)(storage + views_offset);
    operation->alloca.materialized_buffers =
        (iree_hal_buffer_t**)(storage + materialized_offset);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, sizeof(*operation->alloca.memory_wait),
        (void**)&operation->alloca.memory_wait);
  }
  if (iree_status_is_ok(status)) {
    memset(operation->alloca.memory_wait, 0,
           sizeof(*operation->alloca.memory_wait));
    status = iree_hal_pool_wait_allocate(
        pool, iree_arena_allocator(&operation->capture.metadata_arena),
        &operation->alloca.memory_wait->capacity_wait);
  }

  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_pool_reservation_request_t* request =
        &operation->alloca.requests[i];
    *request = requests[i];
    status =
        iree_hal_amd_xdna_validate_alloca_request(&capabilities, i, request);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const iree_hal_buffer_placement_t placement = {
        .device = operation->device,
        .queue_family_affinity = capabilities.queue_family_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
    };
    const iree_hal_memory_access_t allowed_access =
        pool->memory_contract ? pool->memory_contract->allowed_access
                              : capabilities.allowed_access;
    const iree_hal_buffer_usage_t allowed_usage =
        pool->memory_contract ? pool->memory_contract->allowed_usage
                              : capabilities.supported_usage;
    status = iree_hal_amd_xdna_transient_buffer_create(
        placement, capabilities.memory_type, allowed_access, allowed_usage,
        request->allocation_size, request->allocation_size, pool,
        operation->metadata_block_pool,
        &operation->alloca.transient_buffers[i]);
    if (iree_status_is_ok(status)) {
      captured_outputs[i] = operation->alloca.transient_buffers[i];
      // One reference is returned to the caller and one keeps the wrapper live
      // until terminal operation completion.
      iree_hal_buffer_retain(operation->alloca.transient_buffers[i]);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_submit(operation);
    operation = NULL;
  }
  if (iree_status_is_ok(status)) {
    memcpy(out_buffers, captured_outputs,
           request_count * sizeof(*captured_outputs));
  } else if (captured_outputs) {
    for (iree_host_size_t i = 0; i < request_count; ++i) {
      iree_hal_buffer_release(captured_outputs[i]);
    }
  }
  if (operation) {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  iree_hal_amd_xdna_queue_capture_deinitialize(&output_capture);
  return status;
}

iree_status_t iree_hal_amd_xdna_queue_dealloca(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_allocate(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA, &operation));
  operation->dealloca.buffer_count = buffer_count;

  iree_host_size_t storage_size = 0;
  iree_host_size_t buffers_offset = 0;
  iree_host_size_t reservations_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD_ALIGNED(buffer_count, iree_hal_buffer_t*,
                                iree_alignof(iree_hal_buffer_t*),
                                &buffers_offset),
      IREE_STRUCT_FIELD_ALIGNED(buffer_count, iree_hal_pool_reservation_t,
                                iree_alignof(iree_hal_pool_reservation_t),
                                &reservations_offset));
  uint8_t* storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, storage_size, (void**)&storage);
  }
  if (iree_status_is_ok(status)) {
    memset(storage, 0, storage_size);
    operation->dealloca.transient_buffers =
        (iree_hal_buffer_t**)(storage + buffers_offset);
    operation->dealloca.reservations =
        (iree_hal_pool_reservation_t*)(storage + reservations_offset);
    for (iree_host_size_t i = 0; i < buffer_count; ++i) {
      operation->dealloca.transient_buffers[i] = buffers[i];
      iree_hal_buffer_retain(buffers[i]);
    }
  }

  iree_host_size_t marked_count = 0;
  while (marked_count < buffer_count && iree_status_is_ok(status)) {
    iree_hal_pool_t* source_pool = NULL;
    status = iree_hal_buffer_allocation_begin_dealloca(buffers[marked_count],
                                                       &source_pool);
    if (iree_status_is_ok(status)) {
      if (marked_count == 0) {
        operation->dealloca.pool = source_pool;
      } else if (source_pool != operation->dealloca.pool) {
        iree_hal_buffer_allocation_abort_dealloca(buffers[marked_count]);
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "XDNA deallocation transaction spans multiple source pools");
        break;
      }
      ++marked_count;
    }
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < marked_count; ++i) {
      iree_hal_buffer_allocation_abort_dealloca(buffers[i]);
    }
    iree_hal_amd_xdna_operation_discard(operation);
  } else {
    operation->dealloca.marks_owned = true;
    status = iree_hal_amd_xdna_operation_submit(operation);
  }
  return status;
}
