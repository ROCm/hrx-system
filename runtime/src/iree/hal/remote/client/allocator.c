// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/allocator.h"

#include "iree/hal/remote/client/buffer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_allocator_t
//===----------------------------------------------------------------------===//

static const iree_hal_allocator_vtable_t
    iree_hal_remote_client_allocator_vtable;

static iree_hal_remote_client_allocator_t*
iree_hal_remote_client_allocator_cast(iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_remote_client_allocator_vtable);
  return (iree_hal_remote_client_allocator_t*)base_value;
}

typedef struct iree_hal_remote_client_physical_memory_t {
  // Device used to issue status-bearing physical memory RPCs.
  iree_hal_remote_client_device_t* device;
  // Server resource ID for the physical memory handle.
  iree_hal_remote_resource_id_t resource_id;
  // Host allocator used to free this proxy.
  iree_allocator_t host_allocator;
} iree_hal_remote_client_physical_memory_t;

static iree_hal_remote_client_physical_memory_t*
iree_hal_remote_client_physical_memory_cast(
    iree_hal_physical_memory_t* physical_memory) {
  return (iree_hal_remote_client_physical_memory_t*)physical_memory;
}

static iree_hal_buffer_params_t iree_hal_remote_client_wire_params_to_hal(
    iree_hal_remote_buffer_params_t wire_params) {
  return (iree_hal_buffer_params_t){
      .usage = (iree_hal_buffer_usage_t)wire_params.usage,
      .access = (iree_hal_memory_access_t)wire_params.access,
      .type = (iree_hal_memory_type_t)wire_params.type,
      .queue_affinity = (iree_hal_queue_affinity_t)wire_params.queue_affinity,
      .min_alignment = (iree_device_size_t)wire_params.min_alignment,
  };
}

static iree_hal_remote_buffer_params_t
iree_hal_remote_client_hal_params_to_wire(iree_hal_buffer_params_t params) {
  return (iree_hal_remote_buffer_params_t){
      .usage = params.usage,
      .access = (uint16_t)params.access,
      .type = params.type,
      .queue_affinity = params.queue_affinity,
      .min_alignment = (uint64_t)params.min_alignment,
  };
}

iree_status_t iree_hal_remote_client_allocator_create(
    iree_hal_remote_client_device_t* device, iree_string_view_t identifier,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  iree_host_size_t total_size = 0;
  iree_hal_remote_client_allocator_t* allocator = NULL;
  iree_status_t status =
      IREE_STRUCT_LAYOUT(sizeof(*allocator), &total_size,
                         IREE_STRUCT_FIELD_FAM(identifier.size, char));
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&allocator);
  }
  if (iree_status_is_ok(status)) {
    memset(allocator, 0, total_size);
    iree_hal_resource_initialize(&iree_hal_remote_client_allocator_vtable,
                                 &allocator->resource);
    allocator->host_allocator = host_allocator;
    allocator->device = device;

    iree_string_view_append_to_buffer(identifier, &allocator->identifier,
                                      allocator->identifier_storage);

    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_remote_client_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = allocator->host_allocator;

  if (allocator->heaps) {
    iree_allocator_free(host_allocator, allocator->heaps);
  }
  iree_allocator_free(host_allocator, allocator);
  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_remote_client_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_remote_client_allocator_t* allocator =
      (iree_hal_remote_client_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_remote_client_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_remote_client_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
}

static iree_status_t iree_hal_remote_client_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);

  // If we don't have cached heaps, fetch from the server.
  if (!allocator->heaps) {
    // Build BUFFER_QUERY_HEAPS request.
    struct {
      iree_hal_remote_control_envelope_t envelope;
      iree_hal_remote_buffer_query_heaps_request_t request;
    } request_message;
    memset(&request_message, 0, sizeof(request_message));
    request_message.envelope.message_type =
        IREE_HAL_REMOTE_CONTROL_BUFFER_QUERY_HEAPS;

    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_control_rpc(
        allocator->device,
        iree_make_const_byte_span(&request_message, sizeof(request_message)),
        &response_payload, &response_lease));

    // Parse response.
    iree_status_t status = iree_ok_status();
    if (response_payload.data_length <
        sizeof(iree_hal_remote_buffer_query_heaps_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "BUFFER_QUERY_HEAPS response too small: %" PRIhsz " bytes",
          response_payload.data_length);
    }

    iree_host_size_t heap_count = 0;
    if (iree_status_is_ok(status)) {
      const iree_hal_remote_buffer_query_heaps_response_t* response =
          (const iree_hal_remote_buffer_query_heaps_response_t*)
              response_payload.data;
      heap_count = response->heap_count;

      iree_host_size_t expected_size = 0;
      status = IREE_STRUCT_LAYOUT(
          sizeof(iree_hal_remote_buffer_query_heaps_response_t), &expected_size,
          IREE_STRUCT_FIELD(heap_count, iree_hal_remote_memory_heap_t, NULL));
      if (iree_status_is_ok(status) &&
          response_payload.data_length < expected_size) {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "BUFFER_QUERY_HEAPS response truncated: %" PRIhsz
                             " bytes, expected %" PRIhsz,
                             response_payload.data_length, expected_size);
      }
    }

    // Allocate and populate the cached heap array.
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(
          allocator->host_allocator, heap_count,
          sizeof(iree_hal_allocator_memory_heap_t), (void**)&allocator->heaps);
    }

    if (iree_status_is_ok(status)) {
      const iree_hal_remote_memory_heap_t* wire_heaps =
          (const iree_hal_remote_memory_heap_t*)(response_payload.data +
                                                 sizeof(
                                                     iree_hal_remote_buffer_query_heaps_response_t));
      for (iree_host_size_t i = 0; i < heap_count; ++i) {
        allocator->heaps[i].type = (iree_hal_memory_type_t)wire_heaps[i].type;
        allocator->heaps[i].allowed_usage =
            (iree_hal_buffer_usage_t)wire_heaps[i].allowed_usage;
        allocator->heaps[i].max_allocation_size =
            (iree_device_size_t)wire_heaps[i].max_allocation_size;
        allocator->heaps[i].min_alignment =
            (iree_device_size_t)wire_heaps[i].min_alignment;
      }
      allocator->heap_count = heap_count;
    }

    iree_async_buffer_lease_release(&response_lease);
    IREE_RETURN_IF_ERROR(status);
  }

  // Return cached heaps. Follow the HAL contract: always set out_count, then
  // return OUT_OF_RANGE if capacity < count (the standard pre-sizing pattern).
  if (out_count) *out_count = allocator->heap_count;
  if (capacity < allocator->heap_count) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  if (heaps) {
    memcpy(heaps, allocator->heaps,
           capacity * sizeof(iree_hal_allocator_memory_heap_t));
  }
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_remote_client_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);

  // If heaps haven't been fetched yet, report as allocatable (optimistic).
  // The actual allocation will fail if the server doesn't support it.
  // We don't resolve params->type (e.g. clearing OPTIMAL) here because we
  // don't know what the server's heaps support. The server will resolve during
  // BUFFER_ALLOC.
  if (!allocator->heaps) {
    return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
           IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
           IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
  }

  // Check against cached heaps.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  for (iree_host_size_t i = 0; i < allocator->heap_count; ++i) {
    if ((allocator->heaps[i].type & params->type) == params->type &&
        (allocator->heaps[i].allowed_usage & params->usage) == params->usage) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
                       IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER |
                       IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
      break;
    }
  }

  return compatibility;
}

static iree_status_t iree_hal_remote_client_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  // Build BUFFER_ALLOC request.
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_alloc_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type = IREE_HAL_REMOTE_CONTROL_BUFFER_ALLOC;
  request_message.request.provisional_id =
      IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
          IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER, 0);
  request_message.request.params.usage = params->usage;
  request_message.request.params.access = (uint16_t)params->access;
  request_message.request.params.type = params->type;
  request_message.request.params.queue_affinity = params->queue_affinity;
  request_message.request.params.min_alignment =
      (uint64_t)params->min_alignment;
  request_message.request.allocation_size = (uint64_t)allocation_size;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);

  // Parse response.
  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_buffer_alloc_response_t)) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "BUFFER_ALLOC response too small: %" PRIhsz " bytes",
                           response_payload.data_length);
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_buffer_alloc_response_t* response =
        (const iree_hal_remote_buffer_alloc_response_t*)response_payload.data;
    resolved_id = response->resolved_id;
  }

  iree_async_buffer_lease_release(&response_lease);

  // Create the local buffer proxy.
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_buffer_create(
        allocator->device, resolved_id, params, allocation_size,
        IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE, allocator->host_allocator,
        out_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_remote_client_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer) {
  // The buffer's destroy callback handles release notification.
  // This is the default path when the HAL framework calls deallocate.
  iree_hal_buffer_destroy(buffer);
}

static iree_status_t iree_hal_remote_client_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "remote buffer import is not available");
}

static iree_status_t iree_hal_remote_client_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "remote buffer export not yet implemented");
}

static bool iree_hal_remote_client_allocator_supports_virtual_memory(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  const iree_hal_remote_client_allocator_flags_t queried_flag =
      IREE_HAL_REMOTE_CLIENT_ALLOCATOR_FLAG_VIRTUAL_MEMORY_QUERIED;
  const iree_hal_remote_client_allocator_flags_t supported_flag =
      IREE_HAL_REMOTE_CLIENT_ALLOCATOR_FLAG_VIRTUAL_MEMORY_SUPPORTED;
  if (iree_all_bits_set(allocator->flags, queried_flag)) {
    return iree_all_bits_set(allocator->flags, supported_flag);
  }

  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_virtual_query_capabilities_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_QUERY_CAPABILITIES;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);

  if (iree_status_is_ok(status) &&
      response_payload.data_length <
          sizeof(
              iree_hal_remote_buffer_virtual_query_capabilities_response_t)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "BUFFER_VIRTUAL_QUERY_CAPABILITIES response too small: %" PRIhsz
        " bytes",
        response_payload.data_length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_buffer_virtual_query_capabilities_response_t*
        response =
            (const iree_hal_remote_buffer_virtual_query_capabilities_response_t*)
                response_payload.data;
    allocator->flags |= queried_flag;
    if (response->supports_virtual_memory != 0) {
      allocator->flags |= supported_flag;
    } else {
      allocator->flags &= ~supported_flag;
    }
  }
  iree_async_buffer_lease_release(&response_lease);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }
  return iree_all_bits_set(allocator->flags, queried_flag | supported_flag);
}

static iree_status_t
iree_hal_remote_client_allocator_virtual_memory_query_granularity(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params,
    iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
    iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;
  if (!iree_hal_remote_client_allocator_supports_virtual_memory(
          base_allocator)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "remote allocator does not support virtual memory");
  }

  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_virtual_query_granularity_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_QUERY_GRANULARITY;
  request_message.request.params =
      iree_hal_remote_client_hal_params_to_wire(params);

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);
  if (iree_status_is_ok(status) &&
      response_payload.data_length <
          sizeof(iree_hal_remote_buffer_virtual_query_granularity_response_t)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "BUFFER_VIRTUAL_QUERY_GRANULARITY response too small: %" PRIhsz
        " bytes",
        response_payload.data_length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_buffer_virtual_query_granularity_response_t*
        response =
            (const iree_hal_remote_buffer_virtual_query_granularity_response_t*)
                response_payload.data;
    *out_minimum_page_size = (iree_device_size_t)response->minimum_page_size;
    *out_recommended_page_size =
        (iree_device_size_t)response->recommended_page_size;
  }
  iree_async_buffer_lease_release(&response_lease);
  return status;
}

static iree_status_t iree_hal_remote_client_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  *out_virtual_buffer = NULL;

  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_virtual_reserve_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_RESERVE;
  request_message.request.queue_affinity = queue_affinity;
  request_message.request.size = size;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);

  iree_hal_remote_buffer_virtual_reserve_response_t response;
  memset(&response, 0, sizeof(response));
  if (iree_status_is_ok(status) &&
      response_payload.data_length <
          sizeof(iree_hal_remote_buffer_virtual_reserve_response_t)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "BUFFER_VIRTUAL_RESERVE response too small: %" PRIhsz " bytes",
        response_payload.data_length);
  }
  if (iree_status_is_ok(status)) {
    memcpy(&response, response_payload.data, sizeof(response));
  }
  iree_async_buffer_lease_release(&response_lease);

  if (iree_status_is_ok(status)) {
    iree_hal_buffer_params_t params =
        iree_hal_remote_client_wire_params_to_hal(response.params);
    status = iree_hal_remote_client_buffer_create(
        allocator->device, response.resolved_id, &params,
        (iree_device_size_t)response.allocation_size,
        (iree_hal_buffer_placement_flags_t)response.placement_flags,
        allocator->host_allocator, out_virtual_buffer);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_allocator_virtual_memory_release(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_virtual_release_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_RELEASE;
  request_message.request.buffer_id =
      iree_hal_remote_client_buffer_resource_id(virtual_buffer);

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);
  iree_async_buffer_lease_release(&response_lease);
  if (iree_status_is_ok(status)) {
    iree_hal_remote_client_buffer_disown_remote_resource(virtual_buffer);
    iree_hal_buffer_destroy(virtual_buffer);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_allocator_physical_memory_allocate(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t params, iree_device_size_t size,
    iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  *out_physical_memory = NULL;

  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_physical_alloc_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_PHYSICAL_ALLOC;
  request_message.request.params =
      iree_hal_remote_client_hal_params_to_wire(params);
  request_message.request.size = size;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);

  iree_hal_remote_resource_id_t resolved_id = 0;
  if (iree_status_is_ok(status) &&
      response_payload.data_length <
          sizeof(iree_hal_remote_buffer_physical_alloc_response_t)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "BUFFER_PHYSICAL_ALLOC response too small: %" PRIhsz " bytes",
        response_payload.data_length);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_remote_buffer_physical_alloc_response_t* response =
        (const iree_hal_remote_buffer_physical_alloc_response_t*)
            response_payload.data;
    resolved_id = response->resolved_id;
  }
  iree_async_buffer_lease_release(&response_lease);

  iree_hal_remote_client_physical_memory_t* physical_memory = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*physical_memory),
                                   (void**)&physical_memory);
  }
  if (iree_status_is_ok(status)) {
    physical_memory->device = allocator->device;
    physical_memory->resource_id = resolved_id;
    physical_memory->host_allocator = host_allocator;
    *out_physical_memory = (iree_hal_physical_memory_t*)physical_memory;
  } else if (resolved_id != 0) {
    iree_status_ignore(iree_hal_remote_client_device_release_resource(
        allocator->device, resolved_id));
  }
  return status;
}

static iree_status_t iree_hal_remote_client_allocator_physical_memory_free(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  iree_hal_remote_client_physical_memory_t* remote_physical_memory =
      iree_hal_remote_client_physical_memory_cast(physical_memory);
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_physical_free_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_PHYSICAL_FREE;
  request_message.request.physical_memory_id =
      remote_physical_memory->resource_id;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      remote_physical_memory->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);
  iree_async_buffer_lease_release(&response_lease);
  if (iree_status_is_ok(status)) {
    iree_allocator_t host_allocator = remote_physical_memory->host_allocator;
    iree_allocator_free(host_allocator, remote_physical_memory);
  }
  return status;
}

static iree_status_t iree_hal_remote_client_allocator_virtual_memory_map(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  iree_hal_remote_client_physical_memory_t* remote_physical_memory =
      iree_hal_remote_client_physical_memory_cast(physical_memory);
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_virtual_map_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_MAP;
  request_message.request.buffer_id =
      iree_hal_remote_client_buffer_resource_id(virtual_buffer);
  request_message.request.physical_memory_id =
      remote_physical_memory->resource_id;
  request_message.request.virtual_offset = virtual_offset;
  request_message.request.physical_offset = physical_offset;
  request_message.request.size = size;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);
  iree_async_buffer_lease_release(&response_lease);
  return status;
}

static iree_status_t iree_hal_remote_client_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_virtual_unmap_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_UNMAP;
  request_message.request.buffer_id =
      iree_hal_remote_client_buffer_resource_id(virtual_buffer);
  request_message.request.virtual_offset = virtual_offset;
  request_message.request.size = size;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);
  iree_async_buffer_lease_release(&response_lease);
  return status;
}

static iree_status_t iree_hal_remote_client_allocator_virtual_memory_protect(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_virtual_protect_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_PROTECT;
  request_message.request.buffer_id =
      iree_hal_remote_client_buffer_resource_id(virtual_buffer);
  request_message.request.virtual_offset = virtual_offset;
  request_message.request.size = size;
  request_message.request.queue_affinity = queue_affinity;
  request_message.request.protection = protection;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);
  iree_async_buffer_lease_release(&response_lease);
  return status;
}

static iree_status_t iree_hal_remote_client_allocator_virtual_memory_advise(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  iree_hal_remote_client_allocator_t* allocator =
      iree_hal_remote_client_allocator_cast(base_allocator);
  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_buffer_virtual_advise_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_BUFFER_VIRTUAL_ADVISE;
  request_message.request.buffer_id =
      iree_hal_remote_client_buffer_resource_id(virtual_buffer);
  request_message.request.virtual_offset = virtual_offset;
  request_message.request.size = size;
  request_message.request.queue_affinity = queue_affinity;
  request_message.request.advice = advice;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      allocator->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);
  iree_async_buffer_lease_release(&response_lease);
  return status;
}

static const iree_hal_allocator_vtable_t
    iree_hal_remote_client_allocator_vtable = {
        .destroy = iree_hal_remote_client_allocator_destroy,
        .host_allocator = iree_hal_remote_client_allocator_host_allocator,
        .trim = iree_hal_remote_client_allocator_trim,
        .query_statistics = iree_hal_remote_client_allocator_query_statistics,
        .query_memory_heaps =
            iree_hal_remote_client_allocator_query_memory_heaps,
        .query_buffer_compatibility =
            iree_hal_remote_client_allocator_query_buffer_compatibility,
        .allocate_buffer = iree_hal_remote_client_allocator_allocate_buffer,
        .deallocate_buffer = iree_hal_remote_client_allocator_deallocate_buffer,
        .import_buffer = iree_hal_remote_client_allocator_import_buffer,
        .export_buffer = iree_hal_remote_client_allocator_export_buffer,
        .supports_virtual_memory =
            iree_hal_remote_client_allocator_supports_virtual_memory,
        .virtual_memory_query_granularity =
            iree_hal_remote_client_allocator_virtual_memory_query_granularity,
        .virtual_memory_reserve =
            iree_hal_remote_client_allocator_virtual_memory_reserve,
        .virtual_memory_release =
            iree_hal_remote_client_allocator_virtual_memory_release,
        .physical_memory_allocate =
            iree_hal_remote_client_allocator_physical_memory_allocate,
        .physical_memory_free =
            iree_hal_remote_client_allocator_physical_memory_free,
        .virtual_memory_map =
            iree_hal_remote_client_allocator_virtual_memory_map,
        .virtual_memory_unmap =
            iree_hal_remote_client_allocator_virtual_memory_unmap,
        .virtual_memory_protect =
            iree_hal_remote_client_allocator_virtual_memory_protect,
        .virtual_memory_advise =
            iree_hal_remote_client_allocator_virtual_memory_advise,
};
