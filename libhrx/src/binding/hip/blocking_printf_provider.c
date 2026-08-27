// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/blocking_printf_provider.h"

#include <stdio.h>

#include "binding/hip/blocking_printf_protocol.h"

// Thread-confined HIP state for one physical-device packet pool.
typedef struct iree_hip_blocking_printf_provider_context_t {
  // Process-lifetime provider configuration.
  const iree_hip_blocking_printf_provider_t* provider;

  // Physical-device ordinal used for event source attribution.
  uint32_t physical_device_ordinal;

  // Initialized view of the HAL-owned shared packet pool.
  iree_hip_blocking_printf_protocol_t protocol;

  // Blocking printf protocol service owned by the HAL listener thread.
  iree_hip_blocking_printf_service_t service;
} iree_hip_blocking_printf_provider_context_t;

static void iree_hip_blocking_printf_provider_publish_output(
    void* user_data, iree_hip_blocking_printf_stream_t stream,
    iree_string_view_t text) {
  iree_hip_blocking_printf_provider_context_t* context =
      (iree_hip_blocking_printf_provider_context_t*)user_data;
  if (!iree_hal_device_event_sink_is_valid(context->provider->event_sink)) {
    FILE* file =
        stream == IREE_HIP_BLOCKING_PRINTF_STREAM_STDERR ? stderr : stdout;
    (void)fwrite(text.data, 1, text.size, file);
    return;
  }

  const iree_hal_device_printf_event_t printf_event = {
      .record_length = sizeof(printf_event),
      .abi_version = IREE_HAL_DEVICE_PRINTF_EVENT_ABI_VERSION_0,
      .stream = stream == IREE_HIP_BLOCKING_PRINTF_STREAM_STDERR
                    ? IREE_HAL_DEVICE_PRINTF_STREAM_STDERR
                    : IREE_HAL_DEVICE_PRINTF_STREAM_STDOUT,
      .flags = IREE_HAL_DEVICE_PRINTF_FLAG_NONE,
      .format_id = 0,
      .text = text,
      .arguments = iree_const_byte_span_empty(),
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_PRINTF;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_INFO;
  event.source.driver_id = IREE_SV("hip");
  event.source.physical_device_ordinal = context->physical_device_ordinal;
  event.payload =
      iree_make_const_byte_span(&printf_event, sizeof(printf_event));
  iree_hal_device_event_sink_publish(context->provider->event_sink, &event);
}

static iree_status_t iree_hip_blocking_printf_provider_query_requirements(
    void* user_data,
    const iree_hal_hostcall_provider_device_info_t* device_info,
    iree_hal_hostcall_provider_requirements_t* out_requirements) {
  (void)user_data;
  iree_hip_blocking_printf_protocol_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_hip_blocking_printf_protocol_calculate_layout(
      device_info->execution_unit_count,
      device_info->maximum_resident_subgroup_count, &layout));
  *out_requirements = (iree_hal_hostcall_provider_requirements_t){
      .allocation_size = layout.allocation_size,
      .allocation_alignment =
          iree_max(iree_alignof(iree_hip_hostcall_buffer_header_t),
                   iree_max(iree_alignof(iree_hip_hostcall_packet_header_t),
                            iree_alignof(iree_hip_hostcall_packet_payload_t))),
      .notification_type = IREE_HAL_HOSTCALL_NOTIFICATION_TYPE_HSA_SIGNAL,
  };
  return iree_ok_status();
}

static iree_status_t iree_hip_blocking_printf_provider_context_initialize(
    void* user_data,
    const iree_hal_hostcall_provider_device_info_t* device_info,
    iree_byte_span_t shared_memory, uint64_t device_address,
    iree_hal_hostcall_notification_t notification,
    iree_hal_hostcall_error_callback_t error_callback, void** out_context) {
  iree_hip_blocking_printf_provider_t* provider =
      (iree_hip_blocking_printf_provider_t*)user_data;
  *out_context = NULL;
  if (IREE_UNLIKELY(notification.type !=
                        IREE_HAL_HOSTCALL_NOTIFICATION_TYPE_HSA_SIGNAL ||
                    notification.reserved != 0 || notification.token == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HIP blocking printf requires a valid HSA signal notification");
  }

  iree_hip_blocking_printf_protocol_layout_t layout;
  IREE_RETURN_IF_ERROR(iree_hip_blocking_printf_protocol_calculate_layout(
      device_info->execution_unit_count,
      device_info->maximum_resident_subgroup_count, &layout));

  iree_hip_blocking_printf_provider_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      provider->host_allocator, sizeof(*context), (void**)&context));
  memset(context, 0, sizeof(*context));
  context->provider = provider;
  context->physical_device_ordinal =
      (uint32_t)device_info->physical_device_ordinal;

  iree_hip_blocking_printf_protocol_initialize(
      shared_memory.data, device_address, notification.token, &layout,
      &context->protocol);
  const iree_hip_blocking_printf_output_sink_t output_sink = {
      .fn = iree_hip_blocking_printf_provider_publish_output,
      .user_data = context,
  };
  iree_hip_blocking_printf_service_initialize(
      &context->protocol, output_sink, error_callback, provider->host_allocator,
      &context->service);
  *out_context = context;
  return iree_ok_status();
}

static void iree_hip_blocking_printf_provider_service(void* context_ptr) {
  iree_hip_blocking_printf_provider_context_t* context =
      (iree_hip_blocking_printf_provider_context_t*)context_ptr;
  iree_hip_blocking_printf_service_process_ready(&context->service);
}

static void iree_hip_blocking_printf_provider_context_deinitialize(
    void* context_ptr) {
  iree_hip_blocking_printf_provider_context_t* context =
      (iree_hip_blocking_printf_provider_context_t*)context_ptr;
  const iree_allocator_t host_allocator = context->provider->host_allocator;
  iree_hip_blocking_printf_service_deinitialize(&context->service);
  iree_allocator_free(host_allocator, context);
}

void iree_hip_blocking_printf_provider_initialize(
    iree_hal_device_event_sink_t event_sink, iree_allocator_t host_allocator,
    iree_hip_blocking_printf_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  memset(out_provider, 0, sizeof(*out_provider));
  out_provider->host_allocator = host_allocator;
  out_provider->event_sink = event_sink;
  out_provider->device_extension.base.type =
      IREE_HAL_DEVICE_CREATE_PARAMS_EXTENSION_TYPE_HOSTCALL_PROVIDER;
  out_provider->device_extension.provider.user_data = out_provider;
  out_provider->device_extension.provider.query_requirements =
      iree_hip_blocking_printf_provider_query_requirements;
  out_provider->device_extension.provider.initialize =
      iree_hip_blocking_printf_provider_context_initialize;
  out_provider->device_extension.provider.service =
      iree_hip_blocking_printf_provider_service;
  out_provider->device_extension.provider.deinitialize =
      iree_hip_blocking_printf_provider_context_deinitialize;
}

const iree_hal_device_create_params_extension_t*
iree_hip_blocking_printf_provider_device_extension(
    const iree_hip_blocking_printf_provider_t* provider) {
  return &provider->device_extension.base;
}
