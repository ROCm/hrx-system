// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/file.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/utils/fd_file.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/net/carrier.h"

typedef struct iree_hal_remote_client_file_t {
  iree_hal_resource_t resource;

  // Allocator used for this structure.
  iree_allocator_t host_allocator;

  // Owning remote device for server-side resource release.
  iree_hal_remote_client_device_t* device;

  // Capability lane used for queue file I/O.
  iree_hal_remote_client_file_kind_t kind;

  // Allowed HAL access bits.
  iree_hal_memory_access_t access;

  // Accessible file length in bytes.
  uint64_t length;

  // Retained external handle for HOST_ALLOCATION files.
  iree_io_file_handle_t* handle;

  // Retained common HAL file for ASYNC_FILE files.
  iree_hal_file_t* inner_file;

  // Client-local host allocation for HOST_ALLOCATION files.
  iree_byte_span_t host_allocation;

  // Server-side resource ID for REMOTE_FILE files.
  iree_hal_remote_resource_id_t remote_file_id;

  // True after this file has been referenced by a queue command.
  bool queue_referenced;

  // True if destroy should release remote_file_id on the server.
  bool owns_remote_resource;
} iree_hal_remote_client_file_t;

static const iree_hal_file_vtable_t iree_hal_remote_client_file_vtable;

static iree_hal_remote_client_file_t* iree_hal_remote_client_file_cast(
    iree_hal_file_t* IREE_RESTRICT base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_file_vtable);
  return (iree_hal_remote_client_file_t*)base_value;
}

static bool iree_hal_remote_client_file_handle_allows_access(
    iree_io_file_handle_t* handle, iree_hal_memory_access_t access) {
  const iree_io_file_access_t handle_access =
      iree_io_file_handle_access(handle);
  return (!iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_READ) ||
          iree_all_bits_set(handle_access, IREE_IO_FILE_ACCESS_READ)) &&
         (!iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE) ||
          iree_all_bits_set(handle_access, IREE_IO_FILE_ACCESS_WRITE));
}

static iree_status_t iree_hal_remote_client_file_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_client_file_kind_t kind, iree_hal_memory_access_t access,
    uint64_t length, iree_io_file_handle_t* handle, iree_hal_file_t* inner_file,
    iree_byte_span_t host_allocation,
    iree_hal_remote_resource_id_t remote_file_id, bool owns_remote_resource,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_client_file_t* file = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*file), (void**)&file);
  if (iree_status_is_ok(status)) {
    memset(file, 0, sizeof(*file));
    iree_hal_resource_initialize(&iree_hal_remote_client_file_vtable,
                                 &file->resource);
    file->host_allocator = host_allocator;
    file->device = device;
    file->kind = kind;
    file->access = access;
    file->length = length;
    file->handle = handle;
    file->inner_file = inner_file;
    file->host_allocation = host_allocation;
    file->remote_file_id = remote_file_id;
    file->owns_remote_resource = owns_remote_resource;

    iree_io_file_handle_retain(file->handle);
    iree_hal_file_retain(file->inner_file);

    *out_file = (iree_hal_file_t*)file;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_remote_client_file_import_host_allocation(
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  const iree_byte_span_t contents =
      iree_io_file_handle_value(handle).host_allocation;
  return iree_hal_remote_client_file_create(
      /*device=*/NULL, IREE_HAL_REMOTE_CLIENT_FILE_KIND_HOST_ALLOCATION, access,
      (uint64_t)contents.data_length, handle, /*inner_file=*/NULL, contents,
      /*remote_file_id=*/0, /*owns_remote_resource=*/false, host_allocator,
      out_file);
}

typedef struct iree_hal_remote_client_file_register_request_header_t {
  iree_hal_remote_control_envelope_t envelope;
  iree_hal_remote_file_register_request_t body;
} iree_hal_remote_client_file_register_request_header_t;

static iree_hal_remote_file_external_type_t
iree_hal_remote_client_file_external_type(
    iree_net_file_handle_transfer_type_t transfer_type) {
  switch (transfer_type) {
    case IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD:
      return IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_POSIX_FD;
    case IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE:
      return IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_WIN32_HANDLE;
    default:
      return IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_NONE;
  }
}

static iree_status_t iree_hal_remote_client_file_register_descriptor(
    iree_hal_remote_client_device_t* device, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file) {
  *out_file = NULL;

  iree_status_t status = iree_ok_status();
  if (!device || !device->session || !device->session_carrier) {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "remote file descriptor registration requires "
                              "an active transport carrier");
  }
  if (iree_status_is_ok(status) && !iree_io_file_handle_uses_async_io(handle)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote file descriptor registration requires an async file handle");
  }

  iree_net_file_handle_transfer_type_t transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
  iree_host_size_t transfer_payload_length = 0;
  if (iree_status_is_ok(status)) {
    status = iree_net_carrier_query_file_handle_transfer(
        device->session_carrier, handle, &transfer_type,
        &transfer_payload_length);
  }

  iree_hal_remote_file_external_type_t external_type =
      IREE_HAL_REMOTE_FILE_EXTERNAL_TYPE_NONE;
  iree_hal_remote_file_registration_capabilities_t required_capability =
      IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE;
  if (iree_status_is_ok(status)) {
    external_type = iree_hal_remote_client_file_external_type(transfer_type);
    required_capability =
        iree_hal_remote_file_registration_capability_for_external_type(
            external_type);
    if (required_capability ==
        IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE) {
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "remote transport file transfer type %u is "
                                "not supported by FILE_REGISTER",
                                (uint32_t)transfer_type);
    }
  }
  if (iree_status_is_ok(status) &&
      !iree_all_bits_set(device->file_registration_capabilities,
                         required_capability)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote FILE_REGISTER external file type %u requires capability "
        "0x%08x, but this client session supports 0x%08x",
        (uint32_t)external_type, required_capability,
        device->file_registration_capabilities);
  }
  if (iree_status_is_ok(status) && transfer_payload_length > UINT32_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "remote FILE_REGISTER transfer payload too large: %" PRIhsz " bytes",
        transfer_payload_length);
  }

  iree_host_size_t request_size = 0;
  iree_host_size_t transfer_payload_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_client_file_register_request_header_t),
        &request_size,
        IREE_STRUCT_FIELD(transfer_payload_length, uint8_t,
                          &transfer_payload_offset));
  }

  iree_hal_remote_client_file_register_request_header_t* request = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, request_size, (void**)&request);
  }
  if (iree_status_is_ok(status)) {
    memset(request, 0, request_size);
    request->envelope.message_type = IREE_HAL_REMOTE_CONTROL_FILE_REGISTER;
    request->envelope.message_flags =
        IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET;
    const uint32_t provisional_generation = (uint32_t)iree_atomic_fetch_add(
        &device->next_provisional_generation, 1, iree_memory_order_relaxed);
    request->body.provisional_id = IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
        IREE_HAL_REMOTE_RESOURCE_TYPE_FILE, provisional_generation);
    request->body.external_type = (uint32_t)external_type;
    request->body.access_flags = access;
    request->body.handle_payload_length = (uint32_t)transfer_payload_length;
  }

  uint64_t file_length = 0;
  iree_hal_file_t* metadata_file = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_fd_file_from_handle(access, handle, /*proactor=*/NULL,
                                          host_allocator, &metadata_file);
  }
  if (iree_status_is_ok(status)) {
    file_length = iree_hal_file_length(metadata_file);
  }

  iree_hal_file_t* file = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_file_create(
        device, IREE_HAL_REMOTE_CLIENT_FILE_KIND_REMOTE_FILE, access,
        file_length, /*handle=*/NULL, /*inner_file=*/NULL,
        iree_byte_span_empty(), request->body.provisional_id,
        /*owns_remote_resource=*/true, host_allocator, &file);
  }

  iree_byte_span_t transfer_payload = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    transfer_payload = iree_make_byte_span(
        (uint8_t*)request + transfer_payload_offset, transfer_payload_length);
    status = iree_net_carrier_export_file_handle(
        device->session_carrier, handle, transfer_type, transfer_payload);
  }

  bool transfer_exported = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_send_fire_and_forget(
        device, iree_make_const_byte_span(request, request_size));
  }
  if (!iree_status_is_ok(status) && transfer_exported) {
    status = iree_status_join(
        status, iree_net_carrier_release_file_handle_transfer(
                    device->session_carrier, transfer_type,
                    iree_make_const_byte_span(transfer_payload.data,
                                              transfer_payload.data_length)));
  }
  if (iree_status_is_ok(status)) {
    *out_file = file;
    file = NULL;
  }

  iree_hal_file_release(file);
  iree_hal_file_release(metadata_file);
  iree_allocator_free(host_allocator, request);
  return status;
}

static iree_status_t iree_hal_remote_client_file_import_file_descriptor(
    iree_hal_remote_client_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_async_proactor_t* proactor,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  if (device && device->file_registration_capabilities !=
                    IREE_HAL_REMOTE_FILE_REGISTRATION_CAPABILITY_NONE) {
    return iree_hal_remote_client_file_register_descriptor(
        device, access, handle, host_allocator, out_file);
  }

  if (!proactor || !iree_io_file_handle_uses_async_io(handle)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote file descriptor import requires an async proactor file handle; "
        "FILE_REGISTER external handle transfer is not implemented for this "
        "transport, so open the file with IREE_IO_FILE_MODE_ASYNC or use a "
        "host allocation or server-side remote file");
  }

  iree_hal_file_t* inner_file = NULL;
  iree_status_t status = iree_hal_fd_file_from_handle(
      access, handle, proactor, host_allocator, &inner_file);
  if (iree_status_is_ok(status) && !iree_hal_file_async_handle(inner_file)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "remote file descriptor import requires a proactor-backed async file "
        "handle");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_file_create(
        /*device=*/NULL, IREE_HAL_REMOTE_CLIENT_FILE_KIND_ASYNC_FILE, access,
        iree_hal_file_length(inner_file), /*handle=*/NULL, inner_file,
        iree_byte_span_empty(), /*remote_file_id=*/0,
        /*owns_remote_resource=*/false, host_allocator, out_file);
  }
  iree_hal_file_release(inner_file);
  (void)queue_affinity;
  return status;
}

typedef struct iree_hal_remote_client_file_open_request_header_t {
  iree_hal_remote_control_envelope_t envelope;
  iree_hal_remote_file_open_request_t body;
} iree_hal_remote_client_file_open_request_header_t;

static iree_status_t iree_hal_remote_client_file_close_control(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t remote_file_id) {
  if (!device || !device->session ||
      iree_hal_remote_client_device_load_state(device) !=
          IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {
    // Session teardown reclaims all server-side resources.
    return iree_ok_status();
  }

  iree_hal_remote_control_envelope_t envelope;
  memset(&envelope, 0, sizeof(envelope));
  envelope.message_type = IREE_HAL_REMOTE_CONTROL_FILE_CLOSE;
  envelope.message_flags = IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET;

  iree_hal_remote_file_close_t body;
  memset(&body, 0, sizeof(body));
  body.file_id = remote_file_id;

  iree_async_span_t spans[2];
  spans[0] = iree_async_span_from_ptr(&envelope, sizeof(envelope));
  spans[1] = iree_async_span_from_ptr(&body, sizeof(body));
  iree_async_span_list_t payload = iree_async_span_list_make(spans, 2);
  return iree_net_session_send_control_data_copy(
      device->session, /*flags=*/0, payload, /*operation_user_data=*/0);
}

iree_status_t iree_hal_remote_client_file_open(
    iree_hal_remote_client_device_t* device, iree_string_view_t logical_name,
    iree_hal_memory_access_t access, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t request_size = 0;
  iree_host_size_t logical_name_offset = 0;
  iree_status_t status = iree_hal_remote_client_device_check_connected(device);
  const iree_hal_memory_access_t allowed_access =
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  if (iree_status_is_ok(status) &&
      (access == 0 || (access & ~allowed_access) != 0)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote file access must contain read and/or write bits only");
  } else if (iree_status_is_ok(status) && logical_name.size > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "remote file logical name too long: %" PRIhsz,
                              logical_name.size);
  } else if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_client_file_open_request_header_t),
        &request_size,
        IREE_STRUCT_FIELD(logical_name.size, uint8_t, &logical_name_offset));
  }

  iree_hal_remote_client_file_open_request_header_t* request = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, request_size, (void**)&request);
  }

  if (iree_status_is_ok(status)) {
    memset(request, 0, sizeof(*request));
    request->envelope.message_type = IREE_HAL_REMOTE_CONTROL_FILE_OPEN;
    request->body.path_length = (uint16_t)logical_name.size;
    request->body.mode = (uint16_t)access;
    memcpy((uint8_t*)request + logical_name_offset, logical_name.data,
           logical_name.size);
  }

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_control_rpc(
        device, iree_make_const_byte_span(request, request_size),
        &response_payload, &response_lease);
  }

  iree_hal_remote_resource_id_t resolved_id = 0;
  iree_hal_remote_file_open_response_t response;
  memset(&response, 0, sizeof(response));
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length != sizeof(response)) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "FILE_OPEN response length %" PRIhsz
                                " does not match canonical length %" PRIhsz,
                                response_payload.data_length, sizeof(response));
    } else {
      memcpy(&response, response_payload.data, sizeof(response));
    }
  }
  if (iree_status_is_ok(status)) {
    if (IREE_HAL_REMOTE_RESOURCE_ID_TYPE(response.resolved_id) !=
            IREE_HAL_REMOTE_RESOURCE_TYPE_FILE ||
        IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(response.resolved_id)) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "FILE_OPEN response has invalid file resource ID");
    } else {
      resolved_id = response.resolved_id;
    }
  }
  if (iree_status_is_ok(status) && response.reserved != 0) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "FILE_OPEN response reserved field is nonzero");
  }
  if (iree_status_is_ok(status) &&
      (response.granted_access == 0 ||
       (response.granted_access & ~allowed_access) != 0 ||
       !iree_all_bits_set(response.granted_access, access))) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "FILE_OPEN response access 0x%08" PRIx32
                              " does not satisfy requested access 0x%04" PRIx16,
                              response.granted_access, access);
  }

  iree_async_buffer_lease_release(&response_lease);

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_file_create(
        device, IREE_HAL_REMOTE_CLIENT_FILE_KIND_REMOTE_FILE,
        (iree_hal_memory_access_t)response.granted_access, response.file_size,
        /*handle=*/NULL, /*inner_file=*/NULL, iree_byte_span_empty(),
        resolved_id, /*owns_remote_resource=*/true, host_allocator, out_file);
  }
  if (iree_status_is_ok(status)) {
    resolved_id = 0;
  } else if (resolved_id != 0) {
    status = iree_status_join(
        status, iree_hal_remote_client_file_close_control(device, resolved_id));
  }

  iree_allocator_free(host_allocator, request);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_open_file(
    iree_hal_device_t* base_device, iree_string_view_t logical_name,
    iree_hal_memory_access_t access, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_remote_client_device_t* device =
      iree_hal_remote_client_device_cast(base_device);
  iree_status_t status = iree_hal_remote_client_file_open(
      device, logical_name, access, host_allocator, out_file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_file_import(
    iree_hal_remote_client_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_external_file_flags_t flags,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_EXTERNAL_FILE_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "remote file import flags must be zero");
  } else if (!iree_hal_remote_client_file_handle_allows_access(handle,
                                                               access)) {
    status = iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "remote file import requested access not allowed by the file handle");
  }

  if (iree_status_is_ok(status)) {
    switch (iree_io_file_handle_type(handle)) {
      case IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION:
        status = iree_hal_remote_client_file_import_host_allocation(
            access, handle, host_allocator, out_file);
        break;
      case IREE_IO_FILE_HANDLE_TYPE_FD:
        status = iree_hal_remote_client_file_import_file_descriptor(
            device, queue_affinity, access, handle, proactor, host_allocator,
            out_file);
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "remote file import requires a host allocation, async file, or "
            "server-side remote file handle; unsupported file handle type %d",
            (int)iree_io_file_handle_type(handle));
        break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_remote_client_file_isa(iree_hal_file_t* file) {
  return iree_hal_resource_is((const iree_hal_resource_t*)file,
                              &iree_hal_remote_client_file_vtable);
}

iree_status_t iree_hal_remote_client_file_resolve(
    iree_hal_file_t* base_file, iree_hal_remote_client_file_view_t* out_view) {
  IREE_ASSERT_ARGUMENT(base_file);
  IREE_ASSERT_ARGUMENT(out_view);
  memset(out_view, 0, sizeof(*out_view));

  if (!iree_hal_remote_client_file_isa(base_file)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file was not imported by the remote client device");
  }

  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  out_view->kind = file->kind;
  out_view->access = file->access;
  out_view->length = file->length;
  out_view->host_allocation = file->host_allocation;
  out_view->remote_file_id = file->remote_file_id;
  if (file->inner_file) {
    out_view->async_file = iree_hal_file_async_handle(file->inner_file);
  }
  return iree_ok_status();
}

void iree_hal_remote_client_file_mark_queue_referenced(
    iree_hal_file_t* base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  file->queue_referenced = true;
}

static void iree_hal_remote_client_file_destroy(
    iree_hal_file_t* IREE_RESTRICT base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  iree_allocator_t host_allocator = file->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_file_release(file->inner_file);
  iree_io_file_handle_release(file->handle);
  iree_status_t status = iree_ok_status();
  if (file->owns_remote_resource && file->remote_file_id != 0) {
    if (file->queue_referenced) {
      status = iree_hal_remote_client_device_release_resource(
          file->device, file->remote_file_id);
    } else {
      status = iree_hal_remote_client_file_close_control(file->device,
                                                         file->remote_file_id);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_remote_client_device_fail(file->device, status);
  }

  iree_allocator_free(host_allocator, file);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_memory_access_t iree_hal_remote_client_file_allowed_access(
    iree_hal_file_t* base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  return file->access;
}

static uint64_t iree_hal_remote_client_file_length(iree_hal_file_t* base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  return file->length;
}

static iree_hal_buffer_t* iree_hal_remote_client_file_storage_buffer(
    iree_hal_file_t* base_file) {
  return NULL;
}

static iree_async_file_t* iree_hal_remote_client_file_async_handle(
    iree_hal_file_t* base_file) {
  iree_hal_remote_client_file_t* file =
      iree_hal_remote_client_file_cast(base_file);
  return file->inner_file ? iree_hal_file_async_handle(file->inner_file) : NULL;
}

static bool iree_hal_remote_client_file_supports_synchronous_io(
    iree_hal_file_t* base_file) {
  return false;
}

static iree_status_t iree_hal_remote_client_file_read(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "remote client files do not support synchronous read; use queue_read "
      "with the remote bulk transfer path");
}

static iree_status_t iree_hal_remote_client_file_write(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "remote client files do not support synchronous write; use queue_write "
      "with the remote bulk transfer path");
}

static const iree_hal_file_vtable_t iree_hal_remote_client_file_vtable = {
    .destroy = iree_hal_remote_client_file_destroy,
    .allowed_access = iree_hal_remote_client_file_allowed_access,
    .length = iree_hal_remote_client_file_length,
    .storage_buffer = iree_hal_remote_client_file_storage_buffer,
    .async_handle = iree_hal_remote_client_file_async_handle,
    .supports_synchronous_io =
        iree_hal_remote_client_file_supports_synchronous_io,
    .read = iree_hal_remote_client_file_read,
    .write = iree_hal_remote_client_file_write,
};
