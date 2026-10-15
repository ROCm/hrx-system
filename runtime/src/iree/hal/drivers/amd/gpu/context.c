// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/context.h"

#include "iree/hal/drivers/amd/status.h"

void iree_hal_amd_gpu_context_report_status(
    const iree_hal_amd_gpu_context_t* context, iree_hal_device_t* device,
    const iree_status_t status) {
  char message[1024] = {0};
  iree_host_size_t message_length = 0;
  iree_status_format(status, sizeof(message), message, &message_length);
  const iree_hal_device_driver_failure_event_t payload = {
      .record_length = sizeof(payload),
      .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
      .status_code = iree_status_code(status),
      .message = iree_make_string_view(
          message, iree_min(message_length, sizeof(message) - 1)),
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.device = device;
  event.source.driver_id = IREE_SV("amd");
  event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  iree_hal_device_event_sink_publish(context->event_sink, &event);
}

void iree_hal_amd_gpu_context_report(const iree_hal_amd_gpu_context_t* context,
                                     amdf_status_t status,
                                     const char* operation) {
  iree_hal_amd_report_failure(context->event_sink, IREE_SV("amd"), status,
                              operation);
}

void iree_hal_amd_gpu_context_free(iree_hal_amd_gpu_context_t* context) {
  if (!context || context->cleanup_blocked) {
    return;
  }
  if (context->event) {
    amdf_status_t status = context->gpu->event_destroy(context->event);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(context, status, "gpu.event_destroy");
      return;
    }
    context->event = NULL;
  }
  if (context->device) {
    amdf_status_t status = context->api->device_destroy(context->device);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(context, status, "device_destroy");
      return;
    }
    context->device = NULL;
  }
  if (context->endpoint) {
    amdf_status_t status = context->api->endpoint_close(context->endpoint);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_gpu_context_report(context, status, "endpoint_close");
      return;
    }
    context->endpoint = NULL;
  }
  iree_allocator_free(context->host_allocator, context->memory_scopes);
  iree_allocator_free(context->host_allocator, context->families);
  iree_hal_amd_fabric_release(context->fabric);
  iree_allocator_free(context->host_allocator, context);
}

static iree_status_t iree_hal_amd_gpu_context_open(
    iree_hal_amd_gpu_context_t* context,
    const amdf_endpoint_id_t* endpoint_id) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->endpoint_open(context->fabric->instance, endpoint_id,
                                  &context->endpoint),
      "endpoint_open"));
  context->endpoint_info = (amdf_endpoint_info_t){
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(context->endpoint_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->endpoint_query_info(context->endpoint,
                                        &context->endpoint_info),
      "endpoint_query_info"));
  context->target_info = (amdf_gpu_endpoint_info_t){
      .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
      .structure_size = sizeof(context->target_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->endpoint_query_info(context->endpoint,
                                        &context->target_info),
      "gpu.endpoint_query_info"));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
      context->host_allocator, context->endpoint_info.queue_family_count,
      sizeof(*context->families), (void**)&context->families));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < context->endpoint_info.queue_family_count &&
                       iree_status_is_ok(status);
       ++i) {
    context->families[i] = (amdf_queue_family_info_t){
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(context->families[i]),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->endpoint_query_queue_family_info(context->endpoint, i,
                                                       &context->families[i]),
        "endpoint_query_queue_family_info");
  }
  IREE_RETURN_IF_ERROR(status);
  const amdf_gpu_device_create_info_t device_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
      .structure_size = sizeof(device_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->device_create(context->endpoint, &device_info,
                                  &context->device),
      "gpu.device_create"));
  context->device_info = (amdf_gpu_device_info_t){
      .type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      .structure_size = sizeof(context->device_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->device_query_info(context->device, &context->device_info),
      "gpu.device_query_info"));

  // Count-only enumeration may report BUFFER_TOO_SMALL for a nonempty set.
  amdf_status_t count_status = context->api->device_enumerate_memory_scopes(
      context->device, 0, NULL, &context->memory_scope_count);
  if (count_status != amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        count_status, "device_enumerate_memory_scopes"));
  }
  if (context->memory_scope_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        context->host_allocator, context->memory_scope_count,
        sizeof(*context->memory_scopes), (void**)&context->memory_scopes));
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->device_enumerate_memory_scopes(
            context->device, context->memory_scope_count,
            context->memory_scopes, &context->memory_scope_count),
        "device_enumerate_memory_scopes"));
  }

  const amdf_gpu_event_create_info_t event_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
      .structure_size = sizeof(event_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->event_create(context->device, &event_info, &context->event),
      "gpu.event_create"));
  context->event_info = (amdf_gpu_event_info_t){
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(context->event_info),
  };
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->event_query_info(context->event, &context->event_info),
      "gpu.event_query_info");
}

iree_status_t iree_hal_amd_gpu_context_allocate(
    iree_hal_amd_fabric_t* fabric, const amdf_endpoint_id_t* endpoint_id,
    iree_hal_device_event_sink_t event_sink, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_context = NULL;
  const void* extension = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              fabric->api->query_extension(
                  AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_3,
                  AMDF_GPU_EXTENSION_VERSION_LATEST, &extension),
              "query_extension(GPU)"));
  iree_hal_amd_gpu_context_t* context = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*context),
                                (void**)&context));
  context->host_allocator = host_allocator;
  context->fabric = fabric;
  iree_hal_amd_fabric_retain(fabric);
  context->api = fabric->api;
  context->gpu = extension;
  context->event_sink = event_sink;
  iree_status_t status = iree_hal_amd_gpu_context_open(context, endpoint_id);
  if (iree_status_is_ok(status)) {
    *out_context = context;
  } else {
    iree_hal_amd_gpu_context_free(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
