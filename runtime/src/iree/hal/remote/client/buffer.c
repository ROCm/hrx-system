// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/buffer.h"

#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_buffer_t
//===----------------------------------------------------------------------===//

static const iree_hal_buffer_vtable_t iree_hal_remote_client_buffer_vtable;

static void iree_hal_remote_client_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = buffer->host_allocator;

  iree_hal_remote_client_buffer_release_reservation(base_buffer,
                                                    /*death_frontier=*/NULL);

  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  iree_hal_buffer_release(buffer->backing_buffer);

  if (buffer->owns_remote_resource && buffer->resource_id != 0) {
    // Release is best-effort. If the session is already disconnected, the
    // server will clean up the resource when the session closes.
    iree_status_ignore(iree_hal_remote_client_device_release_resource(
        buffer->device, buffer->resource_id));
  }

  if (base_buffer->allocated_buffer != base_buffer) {
    iree_hal_buffer_release(base_buffer->allocated_buffer);
  }

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_remote_client_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_hal_remote_client_buffer_is_deallocated(base_buffer)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote buffer has been deallocated");
  }
  if (buffer->resource_id == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote buffer backing has not been materialized; wait for the "
        "queue_alloca signal before mapping it");
  }

  // Allocate local staging buffer. All access goes through this staging copy:
  // READ pulls data from server into staging, WRITE pushes staging to server
  // on unmap.
  uint8_t* staging = NULL;
  iree_status_t status = iree_allocator_malloc(
      buffer->host_allocator, (iree_host_size_t)local_byte_length,
      (void**)&staging);

  // If READ access, pull current buffer contents from the server.
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ)) {
    struct {
      iree_hal_remote_control_envelope_t envelope;
      iree_hal_remote_buffer_map_request_t body;
    } request;
    memset(&request, 0, sizeof(request));
    request.envelope.message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_MAP;
    request.body.buffer_id =
        iree_hal_remote_client_buffer_resource_id(base_buffer);
    request.body.memory_access = (uint32_t)memory_access;
    request.body.offset = local_byte_offset;
    request.body.length = local_byte_length;

    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    status = iree_hal_remote_client_device_control_rpc(
        buffer->device, iree_make_const_byte_span(&request, sizeof(request)),
        &response_payload, &response_lease);

    if (iree_status_is_ok(status)) {
      if (response_payload.data_length <
          sizeof(iree_hal_remote_buffer_map_response_t)) {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "BUFFER_MAP response truncated: %" PRIhsz " bytes",
                             response_payload.data_length);
      } else {
        const iree_hal_remote_buffer_map_response_t* response =
            (const iree_hal_remote_buffer_map_response_t*)response_payload.data;
        const uint8_t* response_data =
            response_payload.data + sizeof(*response);
        iree_host_size_t data_available =
            response_payload.data_length - sizeof(*response);
        if (data_available < local_byte_length) {
          status =
              iree_make_status(IREE_STATUS_INTERNAL,
                               "BUFFER_MAP response data too short: %" PRIhsz
                               " bytes, expected %" PRIdsz,
                               data_available, local_byte_length);
        } else {
          memcpy(staging, response_data, (iree_host_size_t)local_byte_length);
        }
      }
      iree_async_buffer_lease_release(&response_lease);
    }
  }

  if (iree_status_is_ok(status)) {
    mapping->contents =
        iree_make_byte_span(staging, (iree_host_size_t)local_byte_length);
    // Store mapping state so flush_range can find the staging data.
    buffer->active_mapping_data = staging;
    buffer->active_mapping_offset = local_byte_offset;
    buffer->active_mapping_length = local_byte_length;
  } else {
    iree_allocator_free(buffer->host_allocator, staging);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_client_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (iree_hal_remote_client_buffer_is_deallocated(base_buffer)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote buffer has been deallocated");
  } else if (buffer->resource_id == 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "remote buffer backing has not been "
                              "materialized; wait for the queue_alloca signal "
                              "before unmapping it");
  }

  // If WRITE access was used, push the staging data to the server.
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(mapping->impl.allowed_access,
                        IREE_HAL_MEMORY_ACCESS_WRITE)) {
    // Build request: envelope + unmap header + inline data. The data can be
    // arbitrarily large, so heap-allocate the entire request.
    iree_host_size_t header_size =
        sizeof(iree_hal_remote_control_envelope_t) +
        sizeof(iree_hal_remote_buffer_unmap_request_t);
    iree_host_size_t data_size = (iree_host_size_t)local_byte_length;
    iree_host_size_t request_size = header_size + data_size;

    uint8_t* request_buffer = NULL;
    status = iree_allocator_malloc(buffer->host_allocator, request_size,
                                   (void**)&request_buffer);
    if (iree_status_is_ok(status)) {
      memset(request_buffer, 0, header_size);

      iree_hal_remote_control_envelope_t* envelope =
          (iree_hal_remote_control_envelope_t*)request_buffer;
      envelope->message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP;

      iree_hal_remote_buffer_unmap_request_t* body =
          (iree_hal_remote_buffer_unmap_request_t*)(request_buffer +
                                                    sizeof(*envelope));
      body->buffer_id = iree_hal_remote_client_buffer_resource_id(base_buffer);
      body->offset = local_byte_offset;
      body->length = local_byte_length;

      // Copy staging data after the header.
      memcpy(request_buffer + header_size, mapping->contents.data, data_size);

      iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
      iree_async_buffer_lease_t response_lease;
      memset(&response_lease, 0, sizeof(response_lease));
      status = iree_hal_remote_client_device_control_rpc(
          buffer->device,
          iree_make_const_byte_span(request_buffer, request_size),
          &response_payload, &response_lease);
      iree_async_buffer_lease_release(&response_lease);
      iree_allocator_free(buffer->host_allocator, request_buffer);
    }
  }

  // Clear active mapping state and free the staging buffer.
  buffer->active_mapping_data = NULL;
  buffer->active_mapping_offset = 0;
  buffer->active_mapping_length = 0;
  iree_allocator_free(buffer->host_allocator, mapping->contents.data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Invalidate is a no-op: the next map_range(READ) will pull fresh data.
static iree_status_t iree_hal_remote_client_buffer_invalidate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return iree_ok_status();
}

// Flush pushes the dirty mapping data to the server without unmapping.
// Uses the active mapping state stored on the buffer during map_range
// to locate the staging data for the specified range.
static iree_status_t iree_hal_remote_client_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_hal_remote_client_buffer_is_deallocated(base_buffer)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote buffer has been deallocated");
  }
  if (buffer->resource_id == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote buffer backing has not been materialized; wait for the "
        "queue_alloca signal before flushing it");
  }

  if (!buffer->active_mapping_data) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "flush_range called without an active mapping");
  }

  // Calculate the offset within the staging buffer. local_byte_offset is
  // absolute within the buffer; the staging buffer starts at
  // active_mapping_offset.
  iree_device_size_t staging_offset =
      local_byte_offset - buffer->active_mapping_offset;
  const uint8_t* staging_data =
      buffer->active_mapping_data + (iree_host_size_t)staging_offset;

  iree_host_size_t header_size = sizeof(iree_hal_remote_control_envelope_t) +
                                 sizeof(iree_hal_remote_buffer_unmap_request_t);
  iree_host_size_t data_size = (iree_host_size_t)local_byte_length;
  iree_host_size_t request_size = 0;
  if (!iree_host_size_checked_add(header_size, data_size, &request_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "flush request size overflow");
  }

  uint8_t* request_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      buffer->host_allocator, request_size, (void**)&request_buffer);
  if (iree_status_is_ok(status)) {
    memset(request_buffer, 0, header_size);

    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)request_buffer;
    envelope->message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP;

    iree_hal_remote_buffer_unmap_request_t* body =
        (iree_hal_remote_buffer_unmap_request_t*)(request_buffer +
                                                  sizeof(*envelope));
    body->buffer_id = iree_hal_remote_client_buffer_resource_id(base_buffer);
    body->offset = local_byte_offset;
    body->length = local_byte_length;

    memcpy(request_buffer + header_size, staging_data, data_size);

    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    status = iree_hal_remote_client_device_control_rpc(
        buffer->device, iree_make_const_byte_span(request_buffer, request_size),
        &response_payload, &response_lease);
    iree_async_buffer_lease_release(&response_lease);
    iree_allocator_free(buffer->host_allocator, request_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_remote_client_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_remote_client_buffer_destroy,
    .map_range = iree_hal_remote_client_buffer_map_range,
    .unmap_range = iree_hal_remote_client_buffer_unmap_range,
    .invalidate_range = iree_hal_remote_client_buffer_invalidate_range,
    .flush_range = iree_hal_remote_client_buffer_flush_range,
};

static iree_status_t iree_hal_remote_client_buffer_create_internal(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id, iree_hal_buffer_t* root_buffer,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_hal_buffer_release_callback_t release_callback,
    bool owns_remote_resource, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_remote_client_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_placement_t placement = {
        .device = (iree_hal_device_t*)device,
        .queue_affinity = params->queue_affinity ? params->queue_affinity
                                                 : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = placement_flags,
    };
    // Remote buffers are proxies — the server manages the real memory. Default
    // unspecified access to ALL since the client doesn't constrain access (the
    // server enforces actual hardware capabilities).
    iree_hal_memory_access_t access =
        params->access ? params->access : IREE_HAL_MEMORY_ACCESS_ALL;
    iree_hal_buffer_t* allocated_buffer =
        root_buffer ? root_buffer : &buffer->base;
    iree_hal_buffer_initialize(
        placement, allocated_buffer, allocation_size, byte_offset, byte_length,
        params->type, access, params->usage,
        &iree_hal_remote_client_buffer_vtable, &buffer->base);

    buffer->host_allocator = host_allocator;
    buffer->device = device;
    buffer->resource_id = resource_id;
    buffer->backing_buffer = NULL;
    buffer->release_callback = release_callback;
    buffer->allocation_pool = NULL;
    memset(&buffer->allocation_reservation, 0,
           sizeof(buffer->allocation_reservation));
    iree_atomic_store(&buffer->allocation_reservation_armed, 0,
                      iree_memory_order_relaxed);
    iree_atomic_store(&buffer->deallocated, 0, iree_memory_order_relaxed);
    buffer->owns_remote_resource = owns_remote_resource;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  return iree_hal_remote_client_buffer_create_internal(
      device, resource_id, /*root_buffer=*/NULL, params, allocation_size,
      /*byte_offset=*/0, /*byte_length=*/allocation_size, placement_flags,
      iree_hal_buffer_release_callback_null(),
      /*owns_remote_resource=*/true, host_allocator, out_buffer);
}

iree_status_t iree_hal_remote_client_buffer_create_unbacked(
    iree_hal_remote_client_device_t* device,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  return iree_hal_remote_client_buffer_create_internal(
      device, /*resource_id=*/0, /*root_buffer=*/NULL, params, allocation_size,
      /*byte_offset=*/0, /*byte_length=*/allocation_size, placement_flags,
      iree_hal_buffer_release_callback_null(),
      /*owns_remote_resource=*/false, host_allocator, out_buffer);
}

iree_status_t iree_hal_remote_client_buffer_create_view(
    iree_hal_remote_client_device_t* device, iree_hal_buffer_t* root_buffer,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    const iree_hal_buffer_params_t* params,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(root_buffer);
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(root_buffer);
  if (!allocated_buffer) allocated_buffer = root_buffer;
  return iree_hal_remote_client_buffer_create_internal(
      device, iree_hal_remote_client_buffer_resource_id(root_buffer),
      allocated_buffer, params,
      iree_hal_buffer_allocation_size(allocated_buffer), byte_offset,
      byte_length, placement_flags, release_callback,
      /*owns_remote_resource=*/false, host_allocator, out_buffer);
}

iree_status_t iree_hal_remote_client_buffer_set_backing(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* backing_buffer) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  IREE_ASSERT_ARGUMENT(backing_buffer);
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  if (buffer->backing_buffer) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "remote buffer backing already materialized");
  }
  iree_hal_buffer_t* root_buffer =
      iree_hal_buffer_allocated_buffer(backing_buffer);
  if (!root_buffer) root_buffer = backing_buffer;
  iree_hal_buffer_retain(backing_buffer);
  buffer->backing_buffer = backing_buffer;
  buffer->resource_id = iree_hal_remote_client_buffer_resource_id(root_buffer);
  buffer->base.allocation_size = iree_hal_buffer_allocation_size(root_buffer);
  buffer->base.byte_offset = iree_hal_buffer_byte_offset(backing_buffer);
  return iree_ok_status();
}

void iree_hal_remote_client_buffer_attach_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)iree_hal_buffer_allocated_buffer(
          base_buffer);
  buffer->allocation_pool = pool;
  buffer->allocation_reservation = *reservation;
  iree_atomic_store(&buffer->allocation_reservation_armed, 1,
                    iree_memory_order_release);
}

void iree_hal_remote_client_buffer_release_reservation(
    iree_hal_buffer_t* base_buffer,
    const iree_async_frontier_t* death_frontier) {
  if (!base_buffer) return;
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)iree_hal_buffer_allocated_buffer(
          base_buffer);
  if (!buffer->allocation_pool) return;
  const int32_t was_armed = iree_atomic_exchange(
      &buffer->allocation_reservation_armed, 0, iree_memory_order_acq_rel);
  if (was_armed) {
    iree_hal_pool_release_reservation(buffer->allocation_pool,
                                      &buffer->allocation_reservation,
                                      death_frontier);
  }
}

bool iree_hal_remote_client_buffer_has_reservation(
    iree_hal_buffer_t* base_buffer) {
  if (!base_buffer) return false;
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)iree_hal_buffer_allocated_buffer(
          base_buffer);
  if (!buffer->allocation_pool) return false;
  return iree_atomic_load(&buffer->allocation_reservation_armed,
                          iree_memory_order_acquire) != 0;
}

void iree_hal_remote_client_buffer_mark_deallocated(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)iree_hal_buffer_allocated_buffer(
          base_buffer);
  iree_atomic_store(&buffer->deallocated, 1, iree_memory_order_release);
}

bool iree_hal_remote_client_buffer_is_deallocated(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)iree_hal_buffer_allocated_buffer(
          base_buffer);
  return iree_atomic_load(&buffer->deallocated, iree_memory_order_acquire) != 0;
}

iree_status_t iree_hal_remote_client_buffer_resolve_ref(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_hal_remote_resource_id_t* out_resource_id,
    iree_device_size_t* out_byte_offset) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_resource_id);
  IREE_ASSERT_ARGUMENT(out_byte_offset);
  if (iree_hal_remote_client_buffer_is_deallocated(buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote buffer has been deallocated");
  }
  iree_hal_buffer_t* root_buffer = iree_hal_buffer_allocated_buffer(buffer);
  if (!root_buffer) root_buffer = buffer;
  iree_hal_remote_client_buffer_t* remote_buffer =
      (iree_hal_remote_client_buffer_t*)root_buffer;
  if (remote_buffer->resource_id == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote buffer backing has not been materialized; wait for the "
        "queue_alloca signal before using it");
  }
  *out_resource_id = remote_buffer->resource_id;
  *out_byte_offset = iree_hal_buffer_byte_offset(buffer) + byte_offset;
  return iree_ok_status();
}
