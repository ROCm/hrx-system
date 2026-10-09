// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_operation.h"

#include "iree/hal/drivers/amd/xdna/barrier.h"
#include "iree/hal/drivers/amd/xdna/transient_buffer.h"

static void iree_hal_amd_xdna_transfer_buffers(
    const iree_hal_amd_xdna_transfer_t* transfer,
    iree_hal_buffer_t** out_source, iree_hal_buffer_t** out_target) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  *out_source = NULL;
  *out_target = NULL;
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      if (!operation->fill.length) {
        break;
      }
      *out_target = operation->fill.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      if (!operation->update.length) {
        break;
      }
      *out_target = operation->update.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      if (!operation->copy.length) {
        break;
      }
      *out_source = operation->copy.source_buffer;
      *out_target = operation->copy.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      if (!operation->upload.length) {
        break;
      }
      *out_target = operation->upload.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      if (!operation->download.length) {
        break;
      }
      *out_source = operation->download.source_buffer;
      break;
  }
}

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

void iree_hal_amd_xdna_operation_release_resources(
    iree_hal_amd_xdna_operation_t* operation) {
  switch (operation->kind) {
    case IREE_HAL_AMD_XDNA_OPERATION_ALLOCA:
      if (operation->alloca.materialized_buffers &&
          operation->alloca.transient_buffers) {
        for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
          iree_hal_buffer_release(operation->alloca.materialized_buffers[i]);
          iree_hal_buffer_release(operation->alloca.transient_buffers[i]);
        }
      }
      iree_hal_amd_xdna_operation_release_alloca_reservations(operation);
      if (operation->alloca.memory_wait) {
        IREE_ASSERT_TRUE(operation->alloca.memory_wait->kind ==
                         IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE);
        if (operation->alloca.memory_wait->capacity_wait) {
          iree_hal_pool_wait_destroy(
              operation->alloca.memory_wait->capacity_wait);
        }
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA:
      if (operation->dealloca.transient_buffers) {
        for (iree_host_size_t i = 0; i < operation->dealloca.buffer_count;
             ++i) {
          if (operation->dealloca.marks_owned) {
            iree_hal_buffer_allocation_abort_dealloca(
                operation->dealloca.transient_buffers[i]);
          }
          iree_hal_buffer_release(operation->dealloca.transient_buffers[i]);
        }
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_DISPATCH:
      for (iree_host_size_t i = 0; i < operation->dispatch.binding_count; ++i) {
        iree_hal_buffer_release(
            operation->dispatch.bindings[i].buffer_ref.buffer);
      }
      iree_hal_amd_xdna_invocation_release(operation->dispatch.invocation);
      iree_hal_executable_release(operation->dispatch.executable);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_TRANSFER:
      for (iree_hal_amd_xdna_transfer_t* transfer = operation->transfer.head;
           transfer; transfer = transfer->next) {
        iree_hal_buffer_t* source = NULL;
        iree_hal_buffer_t* target = NULL;
        iree_hal_amd_xdna_transfer_buffers(transfer, &source, &target);
        iree_hal_buffer_release(source);
        iree_hal_buffer_release(target);
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_BARRIER:
      break;
  }
  iree_hal_semaphore_list_release(operation->waits);
}

// Queue transfers use the slab's persistent host execution binding. This
// binding exists independently of public mapping grants, which describe host
// API access and may intentionally be empty for device-only pool scopes.
static const iree_hal_buffer_vtable_t*
iree_hal_amd_xdna_buffer_execution_vtable(iree_hal_buffer_t* buffer) {
  return (const iree_hal_buffer_vtable_t*)((const iree_hal_resource_t*)buffer)
      ->vtable;
}

static iree_status_t iree_hal_amd_xdna_buffer_execution_invalidate(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  if (iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                        IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    return iree_ok_status();
  }
  const iree_device_size_t allocation_offset =
      iree_hal_buffer_byte_offset(buffer) + offset;
  const iree_hal_buffer_vtable_t* vtable =
      iree_hal_amd_xdna_buffer_execution_vtable(buffer);
  return vtable->invalidate_range(buffer, allocation_offset, length);
}

static iree_status_t iree_hal_amd_xdna_buffer_execution_flush(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  if (iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                        IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    return iree_ok_status();
  }
  const iree_device_size_t allocation_offset =
      iree_hal_buffer_byte_offset(buffer) + offset;
  const iree_hal_buffer_vtable_t* vtable =
      iree_hal_amd_xdna_buffer_execution_vtable(buffer);
  return vtable->flush_range(buffer, allocation_offset, length);
}

static iree_status_t iree_hal_amd_xdna_buffer_execution_read(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_byte_span_t* out_span) {
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_native_host_span(buffer, offset, length, out_span));
  return iree_hal_amd_xdna_buffer_execution_invalidate(buffer, offset,
                                                       out_span->data_length);
}

static iree_status_t iree_hal_amd_xdna_buffer_execution_write(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_byte_span_t* out_span) {
  return iree_hal_buffer_native_host_span(buffer, offset, length, out_span);
}

static iree_status_t iree_hal_amd_xdna_buffer_execution_publish(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  return iree_hal_amd_xdna_buffer_execution_flush(buffer, offset, length);
}

static iree_status_t iree_hal_amd_xdna_update_execute(
    const iree_hal_amd_xdna_transfer_t* transfer) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  iree_byte_span_t target;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
      operation->update.target_buffer, operation->update.target_offset,
      operation->update.length, &target));
  iree_hal_amd_xdna_queue_payload_copy(&transfer->update_payload, target.data);
  return iree_hal_amd_xdna_buffer_execution_publish(
      operation->update.target_buffer, operation->update.target_offset,
      target.data_length);
}

static iree_status_t iree_hal_amd_xdna_transfer_execute(
    const iree_hal_amd_xdna_transfer_t* transfer) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      if (!operation->fill.length) {
        return iree_ok_status();
      }
      iree_byte_span_t fill_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->fill.target_buffer, operation->fill.target_offset,
          operation->fill.length, &fill_target));
      if (operation->fill.pattern_length == 1) {
        memset(fill_target.data, *(const uint8_t*)operation->fill.pattern,
               fill_target.data_length);
      } else {
        for (iree_host_size_t offset = 0; offset < fill_target.data_length;
             offset += operation->fill.pattern_length) {
          memcpy(fill_target.data + offset, operation->fill.pattern,
                 operation->fill.pattern_length);
        }
      }
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->fill.target_buffer, operation->fill.target_offset,
          fill_target.data_length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      if (!operation->update.length) {
        return iree_ok_status();
      }
      return iree_hal_amd_xdna_update_execute(transfer);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      if (!operation->copy.length) {
        return iree_ok_status();
      }
      iree_byte_span_t copy_source;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_read(
          operation->copy.source_buffer, operation->copy.source_offset,
          operation->copy.length, &copy_source));
      iree_byte_span_t copy_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->copy.target_buffer, operation->copy.target_offset,
          operation->copy.length, &copy_target));
      const iree_host_size_t copy_length =
          iree_min(copy_source.data_length, copy_target.data_length);
      memcpy(copy_target.data, copy_source.data, copy_length);
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->copy.target_buffer, operation->copy.target_offset,
          copy_length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      if (!operation->upload.length) {
        return iree_ok_status();
      }
      iree_byte_span_t upload_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->upload.target_buffer, operation->upload.target_offset,
          operation->upload.length, &upload_target));
      memcpy(upload_target.data, operation->upload.source,
             upload_target.data_length);
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->upload.target_buffer, operation->upload.target_offset,
          upload_target.data_length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      if (!operation->download.length) {
        return iree_ok_status();
      }
      iree_byte_span_t download_source;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_read(
          operation->download.source_buffer, operation->download.source_offset,
          operation->download.length, &download_source));
      memcpy(operation->download.target, download_source.data,
             download_source.data_length);
      return iree_ok_status();
    default:
      IREE_BUILTIN_UNREACHABLE();
  }
}

void iree_hal_amd_xdna_operation_merge_memory_frontiers(
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

iree_status_t iree_hal_amd_xdna_operation_prepare_frontier_wait(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_memory_wait_t* wait = operation->alloca.memory_wait;
  if (!operation->alloca.pool->frontier_tracker) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "XDNA allocation pool returned a reuse frontier without a tracker");
  }

  iree_host_size_t entry_capacity = 0;
  for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
    const iree_hal_pool_acquire_info_t* info =
        &operation->alloca.acquire_infos[i];
    if (info->result != IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
      continue;
    }
    if (IREE_UNLIKELY(!info->reuse_frontier ||
                      !info->reuse_frontier->entry_count)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "XDNA allocation reservation %" PRIhsz
                              " requires a wait but has no dependency frontier",
                              i);
    }
    entry_capacity += info->reuse_frontier->entry_count;
    if (entry_capacity > UINT8_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "XDNA allocation dependency frontier exceeds %u axes", UINT8_MAX);
    }
  }

  if (!wait->frontier) {
    iree_host_size_t frontier_size = 0;
    IREE_RETURN_IF_ERROR(
        iree_async_frontier_size((uint8_t)entry_capacity, &frontier_size));
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, frontier_size, (void**)&wait->frontier));
  }
  iree_async_frontier_initialize(wait->frontier, 0);
  for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
    const iree_hal_pool_acquire_info_t* info =
        &operation->alloca.acquire_infos[i];
    if (info->result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT &&
        !iree_async_frontier_merge(wait->frontier, (uint8_t)entry_capacity,
                                   info->reuse_frontier)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "XDNA allocation dependency frontier exceeds %" PRIhsz " axes",
          entry_capacity);
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

static iree_status_t iree_hal_amd_xdna_operation_execute_alloca(
    iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_memory_wait_t* wait = operation->alloca.memory_wait;
  if (operation->alloca.reservations_held) {
    operation->alloca.acquire_result = IREE_HAL_POOL_ACQUIRE_OK;
    return iree_hal_amd_xdna_operation_prepare_alloca_storage(operation);
  }

  iree_hal_pool_wait_prepare(wait->capacity_wait);
  iree_status_t status = iree_hal_pool_acquire_reservations(
      operation->alloca.pool, operation->alloca.request_count,
      operation->alloca.requests,
      iree_hal_amd_xdna_frontier_state_as_frontier(&operation->frontier),
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      operation->alloca.reservations, operation->alloca.acquire_infos,
      &operation->alloca.acquire_result);
  if (!iree_status_is_ok(status)) {
    iree_hal_pool_wait_abort(wait->capacity_wait);
    return status;
  }

  operation->alloca.reservations_held =
      operation->alloca.acquire_result == IREE_HAL_POOL_ACQUIRE_OK ||
      operation->alloca.acquire_result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
      operation->alloca.acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT;
  switch (operation->alloca.acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      iree_hal_pool_wait_abort(wait->capacity_wait);
      return iree_hal_amd_xdna_operation_prepare_alloca_storage(operation);
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      iree_hal_pool_wait_abort(wait->capacity_wait);
      status = iree_hal_amd_xdna_operation_prepare_frontier_wait(operation);
      if (!iree_status_is_ok(status)) {
        iree_hal_amd_xdna_operation_release_alloca_reservations(operation);
      }
      return status;
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      // Preserve the observation prepared before acquisition. The proactor
      // owner commits it after the host service acknowledges this item.
      return iree_ok_status();
  }
  iree_hal_pool_wait_abort(wait->capacity_wait);
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unrecognized XDNA pool acquire result %u",
                          operation->alloca.acquire_result);
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

void iree_hal_amd_xdna_operation_execute_host(
    iree_hal_amd_xdna_operation_t* operation) {
  switch (operation->kind) {
    case IREE_HAL_AMD_XDNA_OPERATION_ALLOCA:
      operation->status = iree_hal_amd_xdna_operation_execute_alloca(operation);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA:
      iree_hal_amd_xdna_operation_execute_dealloca(operation);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_TRANSFER:
      for (iree_hal_amd_xdna_transfer_t* transfer = operation->transfer.head;
           transfer && iree_status_is_ok(operation->status);
           transfer = transfer->next) {
        operation->status = iree_hal_amd_xdna_transfer_execute(transfer);
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_BARRIER:
    case IREE_HAL_AMD_XDNA_OPERATION_DISPATCH:
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool iree_hal_amd_xdna_pool_supports_queue_families(
    iree_hal_queue_family_affinity_t supported,
    iree_hal_queue_family_affinity_t requested) {
  if (iree_hal_queue_family_affinity_is_any(supported)) {
    return true;
  }
  return !iree_hal_queue_family_affinity_is_any(requested) &&
         iree_all_bits_set(supported, requested);
}

static iree_status_t iree_hal_amd_xdna_validate_alloca_request(
    const iree_hal_pool_capabilities_t* capabilities, iree_host_size_t index,
    const iree_hal_pool_reservation_request_t* request) {
  const iree_device_size_t alignment =
      request->params.min_alignment ? request->params.min_alignment : 1;
  if (IREE_UNLIKELY(!iree_device_size_is_power_of_two(alignment))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation request %" PRIhsz " alignment %" PRIdsz
                            " is not a power of two",
                            index, alignment);
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(capabilities->memory_type,
                                       request->params.type))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation request %" PRIhsz
                            " memory type is not supported by the source pool",
                            index);
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(capabilities->allowed_access,
                                       request->params.access))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation request %" PRIhsz
                            " access is not supported by the source pool",
                            index);
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(capabilities->supported_usage,
                                       request->params.usage))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation request %" PRIhsz
                            " usage is not supported by the source pool",
                            index);
  }
  if (IREE_UNLIKELY(!iree_hal_amd_xdna_pool_supports_queue_families(
          capabilities->queue_family_affinity,
          request->params.queue_family_affinity))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation request %" PRIhsz
        " queue family affinity is not supported by the source pool",
        index);
  }
  if (IREE_UNLIKELY(capabilities->min_allocation_size &&
                    request->allocation_size <
                        capabilities->min_allocation_size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation request %" PRIhsz " size %" PRIdsz
                            " is smaller than source pool minimum %" PRIdsz,
                            index, request->allocation_size,
                            capabilities->min_allocation_size);
  }
  if (IREE_UNLIKELY(capabilities->max_allocation_size &&
                    request->allocation_size >
                        capabilities->max_allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "allocation request %" PRIhsz " size %" PRIdsz
                            " exceeds source pool maximum %" PRIdsz,
                            index, request->allocation_size,
                            capabilities->max_allocation_size);
  }
  if (IREE_UNLIKELY(capabilities->max_allocation_alignment &&
                    alignment > capabilities->max_allocation_alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation request %" PRIhsz " alignment %" PRIdsz
                            " exceeds source pool maximum %" PRIdsz,
                            index, alignment,
                            capabilities->max_allocation_alignment);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_xdna_queue_alloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
    iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_buffer_t** out_buffers) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);

  iree_hal_amd_xdna_operation_t* operation = NULL;
  iree_status_t status = iree_hal_amd_xdna_operation_create(
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
  iree_arena_allocator_t output_arena;
  iree_arena_initialize(operation->metadata_block_pool, &output_arena);
  iree_hal_buffer_t** captured_outputs = NULL;
  status = iree_arena_allocate_array(&output_arena, request_count,
                                     sizeof(*captured_outputs),
                                     (void**)&captured_outputs);
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
    status = iree_hal_pool_wait_create(
        pool, iree_arena_allocator(&operation->capture.metadata_arena),
        &operation->alloca.memory_wait->capacity_wait);
  }

  for (iree_host_size_t i = 0; i < request_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_pool_reservation_request_t* request =
        &operation->alloca.requests[i];
    *request = requests[i];
    if (pool->memory_contract) {
      request->params = pool->memory_contract->buffer_params;
      request->params.min_alignment = requests[i].params.min_alignment;
    } else {
      iree_hal_buffer_params_canonicalize(&request->params);
      if (iree_any_bit_set(request->params.type,
                           IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
        request->params.type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
        request->params.type |= capabilities.memory_type;
      }
    }
    status =
        iree_hal_amd_xdna_validate_alloca_request(&capabilities, i, request);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const iree_hal_buffer_placement_t placement = {
        .device = operation->device,
        .queue_family_affinity = request->params.queue_family_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
    };
    status = iree_hal_amd_xdna_transient_buffer_create(
        placement, request->params, request->allocation_size,
        request->allocation_size, pool, operation->metadata_block_pool,
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
  iree_arena_deinitialize(&output_arena);
  return status;
}

iree_status_t iree_hal_amd_xdna_queue_dealloca(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t buffer_count,
    iree_hal_buffer_t* const* buffers) {
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
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

iree_status_t iree_hal_amd_xdna_queue_transfer(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_queue_barriers_t* barriers) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_barriers_validate(barriers));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_create(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_TRANSFER, &operation));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_xdna_transfer_t* captured = NULL;
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, sizeof(*captured), (void**)&captured);
    if (!iree_status_is_ok(status)) {
      break;
    }
    *captured = (iree_hal_amd_xdna_transfer_t){0};
    captured->operation = operations[i];
    if (captured->operation.type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL &&
        captured->operation.fill.length) {
      memcpy(captured->fill_pattern, captured->operation.fill.pattern,
             captured->operation.fill.pattern_length);
      captured->operation.fill.pattern = captured->fill_pattern;
    } else if (captured->operation.type ==
                   IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE &&
               captured->operation.update.length) {
      const uint8_t* source =
          (const uint8_t*)captured->operation.update.source_buffer +
          captured->operation.update.source_offset;
      status = iree_hal_amd_xdna_queue_capture_payload(
          &operation->capture,
          iree_make_const_byte_span(
              source, (iree_host_size_t)captured->operation.update.length),
          &captured->update_payload);
      captured->operation.update.source_buffer = NULL;
      captured->operation.update.source_offset = 0;
    }
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_t* source = NULL;
      iree_hal_buffer_t* target = NULL;
      iree_hal_amd_xdna_transfer_buffers(captured, &source, &target);
      iree_hal_buffer_retain(source);
      iree_hal_buffer_retain(target);
      if (operation->transfer.tail) {
        operation->transfer.tail->next = captured;
      } else {
        operation->transfer.head = captured;
      }
      operation->transfer.tail = captured;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_submit(operation);
  } else {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}
