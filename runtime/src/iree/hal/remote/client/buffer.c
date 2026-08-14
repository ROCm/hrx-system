// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/buffer.h"

#include "iree/base/threading/notification.h"
#include "iree/hal/remote/client/bulk.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/net/channel/control/frame.h"
#include "iree/net/channel/util/frame_sender.h"

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

  iree_hal_remote_resource_id_t resource_id =
      (iree_hal_remote_resource_id_t)iree_atomic_load(
          &buffer->resource_id, iree_memory_order_relaxed);
  if (buffer->owns_remote_resource && resource_id != 0) {
    // Release is best-effort. If the session is already disconnected, the
    // server will clean up the resource when the session closes.
    iree_status_ignore(iree_hal_remote_client_device_release_resource(
        buffer->device, resource_id));
  }

  if (base_buffer->allocated_buffer != base_buffer) {
    iree_hal_buffer_release(base_buffer->allocated_buffer);
  }

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_remote_client_buffer_validate_mappable(
    iree_hal_buffer_t* base_buffer, iree_hal_remote_client_buffer_t* buffer,
    const char* operation) {
  iree_status_t status = iree_ok_status();
  if (iree_hal_remote_client_buffer_is_deallocated(base_buffer)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote buffer has been deallocated");
  } else if (iree_atomic_load(&buffer->resource_id,
                              iree_memory_order_relaxed) == 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "remote buffer backing has not been "
                              "materialized; wait for the queue_alloca signal "
                              "before %s it",
                              operation);
  }
  return status;
}

typedef struct iree_hal_remote_client_buffer_write_request_header_t {
  // Control channel envelope for the write request.
  iree_hal_remote_control_envelope_t envelope;

  // BUFFER_UNMAP request header preceding the inline byte payload.
  iree_hal_remote_buffer_unmap_request_t body;
} iree_hal_remote_client_buffer_write_request_header_t;

#define IREE_HAL_REMOTE_CLIENT_MAPPING_UNTRACKED 0ull
#define IREE_HAL_REMOTE_CLIENT_MAPPING_TRACKED 1ull

typedef struct iree_hal_remote_client_buffer_bulk_wait_t {
  // Notification posted when the bulk transfer reaches a terminal state.
  iree_notification_t notification;

  // Terminal status consumed by the waiting mapper.
  iree_status_t status;

  // Non-zero after |status| has been set.
  iree_atomic_int32_t completed;
} iree_hal_remote_client_buffer_bulk_wait_t;

typedef struct iree_hal_remote_client_buffer_bulk_upload_t {
  // Device owning the client-local bulk transfer table.
  iree_hal_remote_client_device_t* device;

  // Client-allocated transfer ID to upload on the bulk channel.
  uint64_t transfer_id;
} iree_hal_remote_client_buffer_bulk_upload_t;

static bool iree_hal_remote_client_buffer_bulk_wait_complete(void* user_data) {
  iree_hal_remote_client_buffer_bulk_wait_t* wait_state =
      (iree_hal_remote_client_buffer_bulk_wait_t*)user_data;
  return iree_atomic_load(&wait_state->completed, iree_memory_order_acquire) !=
         0;
}

static void iree_hal_remote_client_buffer_bulk_complete(void* user_data,
                                                        iree_status_t status) {
  iree_hal_remote_client_buffer_bulk_wait_t* wait_state =
      (iree_hal_remote_client_buffer_bulk_wait_t*)user_data;
  wait_state->status = status;
  iree_atomic_store(&wait_state->completed, 1, iree_memory_order_release);
  iree_notification_post(&wait_state->notification, IREE_ALL_WAITERS);
}

static iree_status_t iree_hal_remote_client_buffer_bulk_upload_after_send(
    void* user_data) {
  iree_hal_remote_client_buffer_bulk_upload_t* upload =
      (iree_hal_remote_client_buffer_bulk_upload_t*)user_data;
  return iree_hal_remote_client_bulk_upload_buffer_unmap(upload->device,
                                                         upload->transfer_id);
}

static iree_status_t iree_hal_remote_client_buffer_control_payload_length(
    iree_host_size_t body_header_length, iree_host_size_t data_length,
    iree_host_size_t* out_payload_length) {
  *out_payload_length = 0;
  iree_host_size_t body_length = 0;
  iree_status_t status =
      iree_host_size_checked_add(body_header_length, data_length, &body_length)
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "remote buffer control payload size overflow");
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_add(sizeof(iree_hal_remote_control_envelope_t),
                                  body_length, out_payload_length)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "remote buffer control payload size overflow");
  }
  return status;
}

static iree_status_t iree_hal_remote_client_buffer_control_response_length(
    iree_host_size_t body_header_length, iree_host_size_t data_length,
    iree_host_size_t* out_response_length) {
  *out_response_length = 0;
  iree_host_size_t body_length = 0;
  iree_status_t status =
      iree_host_size_checked_add(body_header_length, data_length, &body_length)
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "remote buffer control response size overflow");
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_add(
          sizeof(iree_hal_remote_control_envelope_t) +
              sizeof(iree_hal_remote_control_response_prefix_t),
          body_length, out_response_length)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "remote buffer control response size overflow");
  }
  return status;
}

static bool iree_hal_remote_client_buffer_control_frame_fits_inline(
    iree_host_size_t frame_payload_length) {
  if (IREE_NET_FRAME_SENDER_INLINE_FRAME_CAPACITY <
      IREE_NET_CONTROL_FRAME_HEADER_SIZE) {
    return false;
  }
  return frame_payload_length <= IREE_NET_FRAME_SENDER_INLINE_FRAME_CAPACITY -
                                     IREE_NET_CONTROL_FRAME_HEADER_SIZE;
}

static iree_status_t iree_hal_remote_client_buffer_should_use_bulk_map_read(
    iree_host_size_t staging_length, bool* out_use_bulk) {
  *out_use_bulk = false;
  iree_host_size_t response_length = 0;
  iree_status_t status = iree_hal_remote_client_buffer_control_response_length(
      sizeof(iree_hal_remote_buffer_map_response_t), staging_length,
      &response_length);
  if (iree_status_is_ok(status)) {
    *out_use_bulk = staging_length > 0 &&
                    !iree_hal_remote_client_buffer_control_frame_fits_inline(
                        response_length);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_buffer_should_use_bulk_unmap_write(
    iree_host_size_t source_length, bool* out_use_bulk) {
  *out_use_bulk = false;
  iree_host_size_t request_length = 0;
  iree_status_t status = iree_hal_remote_client_buffer_control_payload_length(
      sizeof(iree_hal_remote_buffer_unmap_request_t), source_length,
      &request_length);
  if (iree_status_is_ok(status)) {
    *out_use_bulk = source_length > 0 &&
                    !iree_hal_remote_client_buffer_control_frame_fits_inline(
                        request_length);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_buffer_write_range(
    iree_hal_remote_client_buffer_t* buffer, iree_hal_buffer_t* base_buffer,
    iree_device_size_t local_byte_offset, iree_const_byte_span_t source_bytes) {
  bool use_bulk = false;
  iree_status_t status =
      iree_hal_remote_client_buffer_should_use_bulk_unmap_write(
          source_bytes.data_length, &use_bulk);

  uint64_t transfer_id = 0;
  if (iree_status_is_ok(status) && use_bulk) {
    status = iree_hal_remote_client_bulk_begin_buffer_unmap_write(
        buffer->device, source_bytes, &transfer_id);
  }

  iree_host_size_t request_size = 0;
  iree_host_size_t data_offset = 0;
  if (iree_status_is_ok(status)) {
    iree_host_size_t inline_length = use_bulk ? 0 : source_bytes.data_length;
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_client_buffer_write_request_header_t),
        &request_size, IREE_STRUCT_FIELD(inline_length, uint8_t, &data_offset));
  }

  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_buffer_write_request_header_t* request =
        (iree_hal_remote_client_buffer_write_request_header_t*)iree_alloca(
            request_size);
    memset(request, 0, request_size);
    request->envelope.message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_UNMAP;
    request->body.buffer_id =
        iree_hal_remote_client_buffer_resource_id(base_buffer);
    request->body.offset = local_byte_offset;
    request->body.length = source_bytes.data_length;
    request->body.flags =
        use_bulk ? IREE_HAL_REMOTE_BUFFER_UNMAP_FLAG_BULK_TRANSFER : 0;
    request->body.transfer_id = transfer_id;
    if (!use_bulk && source_bytes.data_length > 0) {
      memcpy((uint8_t*)request + data_offset, source_bytes.data,
             source_bytes.data_length);
    }

    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    if (use_bulk) {
      iree_hal_remote_client_buffer_bulk_upload_t upload = {
          .device = buffer->device,
          .transfer_id = transfer_id,
      };
      iree_hal_remote_client_device_control_rpc_after_send_t after_send = {
          .fn = iree_hal_remote_client_buffer_bulk_upload_after_send,
          .user_data = &upload,
      };
      status = iree_hal_remote_client_device_control_rpc_with_after_send(
          buffer->device, iree_make_const_byte_span(request, request_size),
          after_send, &response_payload, &response_lease);
    } else {
      status = iree_hal_remote_client_device_control_rpc(
          buffer->device, iree_make_const_byte_span(request, request_size),
          &response_payload, &response_lease);
    }
    iree_async_buffer_lease_release(&response_lease);
    if (iree_status_is_ok(status) && use_bulk) {
      iree_hal_remote_client_bulk_end_buffer_unmap(buffer->device, transfer_id);
      transfer_id = 0;
    }
  }

  if (!iree_status_is_ok(status) && transfer_id != 0) {
    iree_hal_remote_client_bulk_cancel_transfer(buffer->device, transfer_id);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_buffer_read_range(
    iree_hal_remote_client_buffer_t* buffer, iree_hal_buffer_t* base_buffer,
    iree_device_size_t local_byte_offset, iree_byte_span_t target_bytes) {
  bool use_bulk = false;
  iree_status_t status = iree_ok_status();
  if (target_bytes.data_length > 0) {
    status = iree_hal_remote_client_buffer_should_use_bulk_map_read(
        target_bytes.data_length, &use_bulk);
  }

  iree_hal_remote_client_buffer_bulk_wait_t bulk_wait;
  memset(&bulk_wait, 0, sizeof(bulk_wait));
  bool bulk_wait_initialized = false;
  uint64_t transfer_id = 0;
  if (iree_status_is_ok(status) && use_bulk) {
    iree_notification_initialize(&bulk_wait.notification);
    bulk_wait_initialized = true;
    bulk_wait.status = iree_ok_status();
    iree_atomic_store(&bulk_wait.completed, 0, iree_memory_order_relaxed);

    iree_hal_remote_client_bulk_completion_callback_t callback = {
        .fn = iree_hal_remote_client_buffer_bulk_complete,
        .user_data = &bulk_wait,
    };
    status = iree_hal_remote_client_bulk_begin_buffer_map_read(
        buffer->device, target_bytes, callback, &transfer_id);
  }

  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_map_request_t body;
  } request;
  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  if (iree_status_is_ok(status) && target_bytes.data_length > 0) {
    memset(&request, 0, sizeof(request));
    request.envelope.message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_MAP;
    request.body.buffer_id =
        iree_hal_remote_client_buffer_resource_id(base_buffer);
    request.body.memory_access = IREE_HAL_MEMORY_ACCESS_READ;
    request.body.flags =
        use_bulk ? IREE_HAL_REMOTE_BUFFER_MAP_FLAG_BULK_TRANSFER : 0;
    request.body.offset = local_byte_offset;
    request.body.length = target_bytes.data_length;
    request.body.transfer_id = transfer_id;

    status = iree_hal_remote_client_device_control_rpc(
        buffer->device, iree_make_const_byte_span(&request, sizeof(request)),
        &response_payload, &response_lease);
  }

  if (iree_status_is_ok(status) && target_bytes.data_length > 0) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_buffer_map_response_t)) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "BUFFER_MAP response truncated: %" PRIhsz " bytes",
                           response_payload.data_length);
    } else {
      const iree_hal_remote_buffer_map_response_t* response =
          (const iree_hal_remote_buffer_map_response_t*)response_payload.data;
      const uint8_t* response_data = response_payload.data + sizeof(*response);
      iree_host_size_t data_available =
          response_payload.data_length - sizeof(*response);
      if (response->mapped_offset != local_byte_offset ||
          response->mapped_length != target_bytes.data_length) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "BUFFER_MAP response range mismatch: offset %" PRIu64
            " length %" PRIu64 ", expected offset %" PRIdsz " length %" PRIhsz,
            response->mapped_offset, response->mapped_length, local_byte_offset,
            target_bytes.data_length);
      } else if (use_bulk) {
        if (response->transfer_id != transfer_id) {
          status =
              iree_make_status(IREE_STATUS_INTERNAL,
                               "BUFFER_MAP response transfer_id=%" PRIu64
                               " does not match expected transfer_id=%" PRIu64,
                               response->transfer_id, transfer_id);
        } else if (data_available != 0) {
          status = iree_make_status(IREE_STATUS_INTERNAL,
                                    "BUFFER_MAP bulk response carried %" PRIhsz
                                    " unexpected inline bytes",
                                    data_available);
        }
      } else if (response->transfer_id != 0) {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "BUFFER_MAP inline response carried unexpected "
                             "transfer_id=%" PRIu64,
                             response->transfer_id);
      } else if (data_available < target_bytes.data_length) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "BUFFER_MAP response data too short: %" PRIhsz
                                  " bytes, expected %" PRIhsz,
                                  data_available, target_bytes.data_length);
      } else {
        memcpy(target_bytes.data, response_data, target_bytes.data_length);
      }
    }
  }
  iree_async_buffer_lease_release(&response_lease);

  if (iree_status_is_ok(status) && use_bulk) {
    iree_notification_await(&bulk_wait.notification,
                            iree_hal_remote_client_buffer_bulk_wait_complete,
                            &bulk_wait, iree_infinite_timeout());
    status = bulk_wait.status;
    bulk_wait.status = iree_ok_status();
  }
  if (!iree_status_is_ok(status) && transfer_id != 0) {
    iree_hal_remote_client_bulk_cancel_transfer(buffer->device, transfer_id);
  }
  if (bulk_wait_initialized) {
    iree_status_ignore(bulk_wait.status);
    iree_notification_deinitialize(&bulk_wait.notification);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_buffer_validate_active_range(
    iree_hal_remote_client_buffer_t* buffer,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    const char* operation, iree_host_size_t* out_staging_offset,
    iree_host_size_t* out_staging_length) {
  *out_staging_offset = 0;
  *out_staging_length = 0;

  iree_status_t status = iree_ok_status();
  if (local_byte_length == 0) {
    // Zero-length cache operations carry no data and do not require staging.
  } else if (!buffer->active_mapping_data) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "remote buffer has no active mapping");
  } else if (local_byte_offset < buffer->active_mapping_offset) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "%s range begins before the active mapping",
                              operation);
  } else {
    iree_device_size_t staging_offset =
        local_byte_offset - buffer->active_mapping_offset;
    if (staging_offset > buffer->active_mapping_length ||
        local_byte_length > buffer->active_mapping_length - staging_offset) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "%s range extends beyond the active mapping",
                                operation);
    } else if (staging_offset > IREE_HOST_SIZE_MAX ||
               local_byte_length > IREE_HOST_SIZE_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "%s range cannot be represented on this host",
                                operation);
    } else {
      *out_staging_offset = (iree_host_size_t)staging_offset;
      *out_staging_length = (iree_host_size_t)local_byte_length;
    }
  }

  return status;
}

static iree_status_t iree_hal_remote_client_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)mapping_mode;

  iree_status_t status = iree_hal_remote_client_buffer_validate_mappable(
      base_buffer, buffer, "mapping");
  iree_host_size_t staging_length = 0;
  if (iree_status_is_ok(status) && local_byte_length > IREE_HOST_SIZE_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "map range cannot be represented on this host");
  }
  if (iree_status_is_ok(status)) {
    staging_length = (iree_host_size_t)local_byte_length;
  }
  const bool mapping_writes =
      iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_WRITE);
  const bool mapping_reads =
      iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ);
  bool track_mapping = false;
  if (iree_status_is_ok(status) && staging_length > 0) {
    if (!buffer->active_mapping_data) {
      track_mapping = true;
    } else if (mapping_writes) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "remote buffer already has an active write mapping");
    } else if (!iree_all_bits_set(buffer->active_mapping_access,
                                  IREE_HAL_MEMORY_ACCESS_WRITE)) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "remote buffer already has an active read mapping");
    } else if (buffer->active_untracked_mapping_count == UINT32_MAX) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "remote buffer has too many active untracked read mappings");
    }
  }

  // All access goes through a staging copy: READ pulls data from the server
  // into staging, while WRITE pushes staging back on flush or unmap.
  uint8_t* staging_data = NULL;
  if (iree_status_is_ok(status) && staging_length > 0) {
    status = iree_allocator_malloc_array(buffer->host_allocator, staging_length,
                                         sizeof(*staging_data),
                                         (void**)&staging_data);
  }

  // If READ access, pull current buffer contents from the server.
  if (iree_status_is_ok(status) && staging_length > 0 && mapping_reads) {
    status = iree_hal_remote_client_buffer_read_range(
        buffer, base_buffer, local_byte_offset,
        iree_make_byte_span(staging_data, staging_length));
  }

  if (iree_status_is_ok(status)) {
    mapping->contents = iree_make_byte_span(staging_data, staging_length);
    mapping->impl.reserved[0] = IREE_HAL_REMOTE_CLIENT_MAPPING_UNTRACKED;
    if (track_mapping) {
      buffer->active_mapping_data = staging_data;
      buffer->active_mapping_access = memory_access;
      buffer->active_mapping_offset = local_byte_offset;
      buffer->active_mapping_length = local_byte_length;
      mapping->impl.reserved[0] = IREE_HAL_REMOTE_CLIENT_MAPPING_TRACKED;
      staging_data = NULL;
    } else if (staging_length > 0) {
      ++buffer->active_untracked_mapping_count;
      staging_data = NULL;
    }
  } else {
    iree_allocator_free(buffer->host_allocator, staging_data);
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

  iree_status_t status = iree_hal_remote_client_buffer_validate_mappable(
      base_buffer, buffer, "unmapping");
  const bool mapping_writes = iree_all_bits_set(mapping->impl.allowed_access,
                                                IREE_HAL_MEMORY_ACCESS_WRITE);
  const bool mapping_tracked =
      mapping->impl.reserved[0] == IREE_HAL_REMOTE_CLIENT_MAPPING_TRACKED;
  if (iree_status_is_ok(status) && local_byte_length > 0) {
    if (mapping_tracked &&
        mapping->contents.data != buffer->active_mapping_data) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "remote buffer mapping state mismatch");
    } else if (!mapping_tracked && mapping_writes) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "remote buffer write mapping was not tracked");
    } else if (!mapping_tracked &&
               buffer->active_untracked_mapping_count == 0) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "remote buffer untracked mapping count underflow");
    }
  }

  // If WRITE access was used, push the staging data to the server.
  if (iree_status_is_ok(status) && local_byte_length > 0 && mapping_writes) {
    status = iree_hal_remote_client_buffer_write_range(
        buffer, base_buffer, local_byte_offset,
        iree_make_const_byte_span(mapping->contents.data,
                                  mapping->contents.data_length));
  }

  if (mapping_tracked &&
      mapping->contents.data == buffer->active_mapping_data) {
    buffer->active_mapping_data = NULL;
    buffer->active_mapping_access = IREE_HAL_MEMORY_ACCESS_NONE;
    buffer->active_mapping_offset = 0;
    buffer->active_mapping_length = 0;
  } else if (!mapping_tracked && local_byte_length > 0 &&
             buffer->active_untracked_mapping_count > 0) {
    --buffer->active_untracked_mapping_count;
  }
  iree_allocator_free(buffer->host_allocator, mapping->contents.data);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_client_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_remote_client_buffer_validate_mappable(
      base_buffer, buffer, "invalidating");
  iree_host_size_t staging_offset = 0;
  iree_host_size_t staging_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_buffer_validate_active_range(
        buffer, local_byte_offset, local_byte_length, "invalidate",
        &staging_offset, &staging_length);
  }
  if (iree_status_is_ok(status) && staging_length > 0 &&
      buffer->active_untracked_mapping_count != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote buffer mapping invalidation is ambiguous while untracked read "
        "mappings are active");
  }
  if (iree_status_is_ok(status) && staging_length > 0 &&
      !iree_all_bits_set(buffer->active_mapping_access,
                         IREE_HAL_MEMORY_ACCESS_READ)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote buffer mapping invalidation requires READ access");
  }
  if (iree_status_is_ok(status) && staging_length > 0) {
    status = iree_hal_remote_client_buffer_read_range(
        buffer, base_buffer, local_byte_offset,
        iree_make_byte_span(buffer->active_mapping_data + staging_offset,
                            staging_length));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Flush pushes the dirty mapping data to the server without unmapping.
// Uses the active mapping state stored on the buffer during map_range to locate
// the staging data for the specified range.
static iree_status_t iree_hal_remote_client_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_remote_client_buffer_t* buffer =
      (iree_hal_remote_client_buffer_t*)base_buffer;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_remote_client_buffer_validate_mappable(
      base_buffer, buffer, "flushing");
  iree_host_size_t staging_offset = 0;
  iree_host_size_t staging_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_buffer_validate_active_range(
        buffer, local_byte_offset, local_byte_length, "flush", &staging_offset,
        &staging_length);
  }
  if (iree_status_is_ok(status) && staging_length > 0 &&
      !iree_all_bits_set(buffer->active_mapping_access,
                         IREE_HAL_MEMORY_ACCESS_WRITE)) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "remote buffer mapping flush requires WRITE access");
  }
  if (iree_status_is_ok(status) && staging_length > 0) {
    status = iree_hal_remote_client_buffer_write_range(
        buffer, base_buffer, local_byte_offset,
        iree_make_const_byte_span(buffer->active_mapping_data + staging_offset,
                                  staging_length));
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
        .queue_affinity = params->queue_affinity,
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
    iree_atomic_store(&buffer->resource_id, resource_id,
                      iree_memory_order_relaxed);
    buffer->backing_buffer = NULL;
    buffer->release_callback = release_callback;
    buffer->allocation_pool = NULL;
    memset(&buffer->allocation_reservation, 0,
           sizeof(buffer->allocation_reservation));
    iree_atomic_store(&buffer->allocation_reservation_armed, 0,
                      iree_memory_order_relaxed);
    iree_atomic_store(&buffer->deallocated, 0, iree_memory_order_relaxed);
    buffer->owns_remote_resource = owns_remote_resource;
    buffer->active_mapping_data = NULL;
    buffer->active_mapping_access = IREE_HAL_MEMORY_ACCESS_NONE;
    buffer->active_mapping_offset = 0;
    buffer->active_mapping_length = 0;
    buffer->active_untracked_mapping_count = 0;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_device_size_t byte_length,
    iree_hal_buffer_placement_flags_t placement_flags,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  return iree_hal_remote_client_buffer_create_internal(
      device, resource_id, /*root_buffer=*/NULL, params, allocation_size,
      /*byte_offset=*/0, byte_length, placement_flags,
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
  iree_atomic_store(&buffer->resource_id,
                    iree_hal_remote_client_buffer_resource_id(root_buffer),
                    iree_memory_order_relaxed);
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

iree_status_t iree_hal_remote_client_buffer_resolve_wire_ref(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_hal_remote_resource_id_t* out_resource_id, uint64_t* out_byte_offset) {
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
  iree_hal_remote_resource_id_t resource_id =
      (iree_hal_remote_resource_id_t)iree_atomic_load(
          &remote_buffer->resource_id, iree_memory_order_relaxed);
  if (resource_id == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote buffer backing has not been materialized; wait for the "
        "queue_alloca signal before using it");
  }
  *out_resource_id = resource_id;
  *out_byte_offset =
      (uint64_t)(iree_hal_buffer_byte_offset(buffer) + byte_offset);
  return iree_ok_status();
}

iree_status_t iree_hal_remote_client_buffer_resolve_wire_range(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length,
    iree_hal_remote_resource_id_t* out_resource_id, uint64_t* out_byte_offset,
    uint64_t* out_byte_length) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_resource_id);
  IREE_ASSERT_ARGUMENT(out_byte_offset);
  IREE_ASSERT_ARGUMENT(out_byte_length);
  if (iree_hal_remote_client_buffer_is_deallocated(buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "remote buffer has been deallocated");
  }

  iree_device_size_t relative_offset = 0;
  iree_device_size_t relative_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(buffer), byte_offset,
      byte_length, &relative_offset, &relative_length));

  iree_hal_buffer_t* root_buffer = iree_hal_buffer_allocated_buffer(buffer);
  if (!root_buffer) root_buffer = buffer;
  iree_hal_remote_client_buffer_t* remote_buffer =
      (iree_hal_remote_client_buffer_t*)root_buffer;
  iree_hal_remote_resource_id_t resource_id =
      (iree_hal_remote_resource_id_t)iree_atomic_load(
          &remote_buffer->resource_id, iree_memory_order_relaxed);
  if (resource_id == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "remote buffer backing has not been materialized; wait for the "
        "queue_alloca signal before using it");
  }
  *out_resource_id = resource_id;
  *out_byte_offset =
      (uint64_t)(iree_hal_buffer_byte_offset(buffer) + relative_offset);
  *out_byte_length = (uint64_t)relative_length;
  return iree_ok_status();
}
