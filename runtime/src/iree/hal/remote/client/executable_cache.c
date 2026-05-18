// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/executable_cache.h"

#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/client/executable.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/hal/remote/protocol/control.h"

static const iree_hal_executable_cache_vtable_t
    iree_hal_remote_client_executable_cache_vtable;

typedef struct iree_hal_remote_client_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_remote_client_device_t* device;
} iree_hal_remote_client_executable_cache_t;

static void iree_hal_remote_client_executable_cache_destroy(
    iree_hal_executable_cache_t* base_cache) {
  iree_hal_remote_client_executable_cache_t* cache =
      (iree_hal_remote_client_executable_cache_t*)base_cache;
  iree_allocator_free(cache->host_allocator, cache);
}

static iree_status_t iree_hal_remote_client_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                          "remote client does not infer formats locally");
}

static bool iree_hal_remote_client_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  iree_hal_remote_client_executable_cache_t* cache =
      (iree_hal_remote_client_executable_cache_t*)base_cache;
  if (executable_format.size > UINT16_MAX) return false;

  const iree_host_size_t header_size =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_executable_cache_query_format_request_t);
  iree_host_size_t format_offset = 0;
  iree_host_size_t message_length = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      header_size, &message_length,
      IREE_STRUCT_FIELD(executable_format.size, char, &format_offset));

  uint8_t* message = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(cache->host_allocator, message_length,
                                   (void**)&message);
  }
  if (iree_status_is_ok(status)) {
    memset(message, 0, message_length);
    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)message;
    envelope->message_type =
        IREE_HAL_REMOTE_CONTROL_EXECUTABLE_CACHE_QUERY_FORMAT;
    iree_hal_remote_executable_cache_query_format_request_t* request =
        (iree_hal_remote_executable_cache_query_format_request_t*)(envelope +
                                                                   1);
    request->caching_mode = (uint32_t)caching_mode;
    request->format_length = (uint16_t)executable_format.size;
    if (executable_format.size > 0) {
      memcpy(message + format_offset, executable_format.data,
             executable_format.size);
    }
  }

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_control_rpc(
        cache->device, iree_make_const_byte_span(message, message_length),
        &response_payload, &response_lease);
  }

  bool can_prepare = false;
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_executable_cache_query_format_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EXECUTABLE_CACHE_QUERY_FORMAT response too short: "
          "%" PRIhsz " < %" PRIhsz,
          response_payload.data_length,
          sizeof(iree_hal_remote_executable_cache_query_format_response_t));
    } else {
      const iree_hal_remote_executable_cache_query_format_response_t* response =
          (const iree_hal_remote_executable_cache_query_format_response_t*)
              response_payload.data;
      can_prepare = response->can_prepare != 0;
    }
  }

  iree_status_ignore(status);
  iree_async_buffer_lease_release(&response_lease);
  iree_allocator_free(cache->host_allocator, message);
  return can_prepare;
}

static iree_status_t iree_hal_remote_client_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_remote_client_executable_cache_t* cache =
      (iree_hal_remote_client_executable_cache_t*)base_cache;
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Build the EXECUTABLE_UPLOAD request.
  iree_host_size_t format_length = executable_params->executable_format.size;
  iree_host_size_t constant_count = executable_params->constant_count;
  iree_host_size_t data_length = executable_params->executable_data.data_length;
  iree_status_t status = iree_ok_status();
  if (format_length > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable format length %" PRIhsz
                              " exceeds wire limit %" PRIu16,
                              format_length, UINT16_MAX);
  }
  if (iree_status_is_ok(status) && constant_count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable constant count %" PRIhsz
                              " exceeds wire limit %" PRIu16,
                              constant_count, UINT16_MAX);
  }

  const iree_host_size_t header_size =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_executable_upload_request_t);
  iree_host_size_t format_offset = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t data_offset = 0;
  iree_host_size_t message_length = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        header_size, &message_length,
        IREE_STRUCT_FIELD(format_length, uint8_t, &format_offset),
        IREE_STRUCT_FIELD_ALIGNED(constant_count, uint32_t, 8,
                                  &constants_offset),
        IREE_STRUCT_FIELD_ALIGNED(data_length, uint8_t, 8, &data_offset));
  }

  // Heap-allocate for the variable-length message (executable binaries can
  // be large).
  uint8_t* message_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(cache->host_allocator, message_length,
                                   (void**)&message_buffer);
  }
  if (iree_status_is_ok(status)) {
    memset(message_buffer, 0, message_length);

    // Envelope.
    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)message_buffer;
    envelope->message_type = IREE_HAL_REMOTE_CONTROL_EXECUTABLE_UPLOAD;

    // Request header.
    iree_hal_remote_executable_upload_request_t* request =
        (iree_hal_remote_executable_upload_request_t*)(envelope + 1);
    request->provisional_id = IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
        IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE, 0);
    request->format_length = (uint16_t)format_length;
    request->constant_count = (uint16_t)constant_count;
    request->upload_flags = IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA;
    request->data_length = data_length;

    // Variable-length payload.
    if (format_length > 0) {
      memcpy(message_buffer + format_offset,
             executable_params->executable_format.data, format_length);
    }
    if (constant_count > 0) {
      memcpy(message_buffer + constants_offset, executable_params->constants,
             constant_count * sizeof(uint32_t));
    }
    if (data_length > 0) {
      memcpy(message_buffer + data_offset,
             executable_params->executable_data.data, data_length);
    }
  }

  // Send RPC.
  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_control_rpc(
        cache->device,
        iree_make_const_byte_span(message_buffer, message_length),
        &response_payload, &response_lease);
  }

  iree_allocator_free(cache->host_allocator, message_buffer);

  // Parse response.
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_executable_upload_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EXECUTABLE_UPLOAD response too short: "
          "%" PRIhsz " < %" PRIhsz,
          response_payload.data_length,
          sizeof(iree_hal_remote_executable_upload_response_t));
    }
  }

  if (iree_status_is_ok(status)) {
    const iree_hal_remote_executable_upload_response_t* response =
        (const iree_hal_remote_executable_upload_response_t*)
            response_payload.data;
    status = iree_hal_remote_client_executable_create(
        cache->device, response->resolved_id,
        (iree_host_size_t)response->export_count, cache->host_allocator,
        out_executable);
  }

  iree_async_buffer_lease_release(&response_lease);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_remote_client_executable_cache_create(
    iree_hal_remote_client_device_t* device, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;

  iree_hal_remote_client_executable_cache_t* cache = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*cache), (void**)&cache);
  if (iree_status_is_ok(status)) {
    memset(cache, 0, sizeof(*cache));
    iree_hal_resource_initialize(
        &iree_hal_remote_client_executable_cache_vtable, &cache->resource);
    cache->host_allocator = host_allocator;
    cache->device = device;

    *out_executable_cache = (iree_hal_executable_cache_t*)cache;
  }
  return status;
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_remote_client_executable_cache_vtable = {
        .destroy = iree_hal_remote_client_executable_cache_destroy,
        .infer_format = iree_hal_remote_client_executable_cache_infer_format,
        .can_prepare_format =
            iree_hal_remote_client_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_remote_client_executable_cache_prepare_executable,
};
