// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/event.h"

#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_event_t
//===----------------------------------------------------------------------===//

static const iree_hal_event_vtable_t iree_hal_remote_client_event_vtable;

typedef struct iree_hal_remote_client_event_t {
  // Base HAL resource for event lifetime management.
  iree_hal_resource_t resource;

  // Host allocator used for proxy storage.
  iree_allocator_t host_allocator;

  // Back-pointer to the owning device for release notifications.
  iree_hal_remote_client_device_t* device;

  // Server-assigned resource ID.
  iree_hal_remote_resource_id_t resource_id;
} iree_hal_remote_client_event_t;

typedef struct iree_hal_remote_client_event_create_message_t {
  // Control channel envelope for the EVENT_CREATE request.
  iree_hal_remote_control_envelope_t envelope;

  // EVENT_CREATE request body.
  iree_hal_remote_event_create_request_t body;
} iree_hal_remote_client_event_create_message_t;

static iree_hal_remote_client_event_t* iree_hal_remote_client_event_cast(
    iree_hal_event_t* base_event) {
  IREE_HAL_ASSERT_TYPE(base_event, &iree_hal_remote_client_event_vtable);
  return (iree_hal_remote_client_event_t*)base_event;
}

static void iree_hal_remote_client_event_destroy(iree_hal_event_t* base_event) {
  iree_hal_remote_client_event_t* event =
      iree_hal_remote_client_event_cast(base_event);
  iree_allocator_t host_allocator = event->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (event->resource_id != 0) {
    // Release is best-effort. If the session is already disconnected, the
    // server will clean up the resource when the session closes.
    iree_status_ignore(iree_hal_remote_client_device_release_resource(
        event->device, event->resource_id));
  }

  iree_allocator_free(host_allocator, event);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_event_vtable_t iree_hal_remote_client_event_vtable = {
    .destroy = iree_hal_remote_client_event_destroy,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_remote_client_event_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_event);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_event = NULL;

  iree_hal_remote_client_event_create_message_t message;
  memset(&message, 0, sizeof(message));
  message.envelope.message_type = IREE_HAL_REMOTE_CONTROL_EVENT_CREATE;
  message.body.queue_affinity = (uint64_t)queue_affinity;
  message.body.flags = (uint32_t)flags;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      device, iree_make_const_byte_span(&message, sizeof(message)),
      &response_payload, &response_lease);

  iree_hal_remote_resource_id_t resource_id = 0;
  if (iree_status_is_ok(status) &&
      response_payload.data_length <
          sizeof(iree_hal_remote_event_create_response_t)) {
    status =
        iree_make_status(IREE_STATUS_INTERNAL,
                         "EVENT_CREATE response too short: %" PRIhsz " bytes",
                         response_payload.data_length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_event_create_response_t* response =
        (const iree_hal_remote_event_create_response_t*)response_payload.data;
    resource_id = response->resolved_id;
    if (IREE_HAL_REMOTE_RESOURCE_ID_TYPE(resource_id) !=
        IREE_HAL_REMOTE_RESOURCE_TYPE_EVENT) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EVENT_CREATE returned non-event resource 0x%016" PRIx64,
          resource_id);
    }
  }
  iree_async_buffer_lease_release(&response_lease);

  iree_hal_remote_client_event_t* event = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_remote_client_event_vtable,
                                 &event->resource);
    event->host_allocator = host_allocator;
    event->device = device;
    event->resource_id = resource_id;
    *out_event = (iree_hal_event_t*)event;
  } else if (resource_id != 0) {
    iree_status_ignore(
        iree_hal_remote_client_device_release_resource(device, resource_id));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_remote_client_event_isa(const iree_hal_event_t* event) {
  return iree_hal_resource_is(event, &iree_hal_remote_client_event_vtable);
}

iree_hal_remote_resource_id_t iree_hal_remote_client_event_resource_id(
    const iree_hal_event_t* base_event) {
  iree_hal_remote_client_event_t* event =
      iree_hal_remote_client_event_cast((iree_hal_event_t*)base_event);
  return event->resource_id;
}
