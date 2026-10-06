// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/context.h"

#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"

void iree_hal_amd_xdna_context_report(
    const iree_hal_amd_xdna_context_t* context, amdf_status_t status,
    const char* operation) {
  const iree_hal_device_driver_failure_event_t payload = {
      .record_length = sizeof(payload),
      .abi_version = IREE_HAL_DEVICE_DRIVER_FAILURE_EVENT_ABI_VERSION_0,
      .status_code = IREE_STATUS_UNKNOWN,
      .backend_result_code = status,
      .message = iree_make_cstring_view(operation),
  };
  iree_hal_device_event_t event = iree_hal_device_event_default();
  event.type = IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE;
  event.severity = IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
  event.source.driver_id = IREE_SV("xdna");
  event.payload = iree_make_const_byte_span(&payload, sizeof(payload));
  iree_hal_device_event_sink_publish(context->event_sink, &event);
}

void iree_hal_amd_xdna_context_destroy(iree_hal_amd_xdna_context_t* context) {
  if (!context) {
    return;
  }
  if (context->handle) {
    amdf_status_t status = context->xdna->context_destroy(context->handle);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_xdna_context_report(context, status, "context_destroy");
      return;
    }
  }
  if (context->device) {
    amdf_status_t status = context->api->device_destroy(context->device);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_xdna_context_report(context, status, "device_destroy");
      return;
    }
  }
  if (context->endpoint) {
    amdf_status_t status = context->api->endpoint_close(context->endpoint);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_xdna_context_report(context, status, "endpoint_close");
      return;
    }
  }
  iree_hal_driver_release(context->driver);
  iree_allocator_free(context->host_allocator, context);
}

static iree_status_t iree_hal_amd_xdna_context_select_memory_source(
    iree_hal_amd_xdna_context_t* context, amdf_memory_scope_t* scope,
    amdf_memory_address_kind_t address_kind, amdf_memory_access_t access,
    iree_hal_amd_xdna_memory_source_t* out_source) {
  const amdf_memory_device_access_t device_access = {
      .device = context->device,
      .requirements =
          {
              .access = access,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << address_kind,
          },
  };
  amdf_memory_scope_info_t scope_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_scope_query_info(scope, &scope_info),
      "memory_scope_query_info"));
  for (uint32_t i = 0; i < scope_info.memory_profile_count; ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    amdf_status_t status = context->api->memory_scope_query_device_profile(
        scope, i, 1, &device_access, &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        status, "memory_scope_query_device_profile"));
    if (iree_all_bits_set(profile.roles,
                          AMDF_MEMORY_PROFILE_ROLE_CREATE |
                              AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        iree_all_bits_set(profile.supported_flags,
                          AMDF_MEMORY_FLAG_HOST_VISIBLE)) {
      *out_source = (iree_hal_amd_xdna_memory_source_t){
          .scope = scope,
          .profile = profile,
          .access = device_access,
          .capabilities = capabilities,
          .address_kind = address_kind,
      };
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "no host-mappable XDNA allocation contract");
}

static iree_status_t iree_hal_amd_xdna_context_open(
    iree_hal_amd_xdna_context_t* context, amdf_instance_t* instance,
    amdf_memory_scope_t* system_scope, const amdf_endpoint_id_t* endpoint_id,
    uint32_t column_count) {
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->endpoint_open(instance, endpoint_id, &context->endpoint),
      "endpoint_open"));
  context->endpoint_info = (amdf_xdna_endpoint_info_t){
      .type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO,
      .structure_size = sizeof(context->endpoint_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->xdna->endpoint_query_info(context->endpoint,
                                         &context->endpoint_info),
      "xdna.endpoint_query_info"));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      iree_make_cstring_view(context->endpoint_info.target_id), column_count,
      &context->target));
  const amdf_xdna_device_create_info_t device_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO,
      .structure_size = sizeof(device_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->xdna->device_create(context->endpoint, &device_info,
                                   &context->device),
      "xdna.device_create"));
  context->device_info = (amdf_xdna_device_info_t){
      .type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
      .structure_size = sizeof(context->device_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->xdna->device_query_info(context->device, &context->device_info),
      "xdna.device_query_info"));
  context->target.instruction_alignment =
      context->device_info.instruction.address_alignment;
  amdf_endpoint_info_t endpoint_info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->endpoint_query_info(context->endpoint, &endpoint_info),
      "endpoint_query_info"));
  context->queue_family_ordinal = UINT32_MAX;
  for (uint32_t i = 0; i < endpoint_info.queue_family_count; ++i) {
    amdf_queue_family_info_t family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family),
    };
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context->api->endpoint_query_queue_family_info(context->endpoint, i,
                                                       &family),
        "endpoint_query_queue_family_info"));
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
        iree_any_bit_set(family.publication_modes,
                         AMDF_QUEUE_PUBLICATION_MODE_KERNEL)) {
      context->queue_family_ordinal = i;
      break;
    }
  }
  if (context->queue_family_ordinal == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no native XDNA kernel queue family");
  }
  const amdf_xdna_context_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .logical_column_count = column_count,
      .physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY,
      .acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->xdna->context_create(context->device, &create_info,
                                    &context->handle),
      "xdna.context_create"));
  amdf_memory_scope_t* command_scope = NULL;
  uint32_t scope_count = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->xdna->context_enumerate_memory_scopes(
          context->handle, 1, &command_scope, &scope_count),
      "xdna.context_enumerate_memory_scopes"));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_context_select_memory_source(
      context, system_scope, AMDF_MEMORY_ADDRESS_XDNA_DMA,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      &context->data_source));
  return iree_hal_amd_xdna_context_select_memory_source(
      context, command_scope, AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
          AMDF_MEMORY_ACCESS_EXECUTE,
      &context->command_source);
}

iree_status_t iree_hal_amd_xdna_context_create(
    iree_hal_driver_t* driver, const amdf_api_t* api,
    const amdf_xdna_api_t* xdna, amdf_instance_t* instance,
    amdf_memory_scope_t* system_scope, const amdf_endpoint_id_t* endpoint_id,
    uint32_t column_count, iree_hal_device_event_sink_t event_sink,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_context_t** out_context) {
  *out_context = NULL;
  if (column_count == 0 || column_count > 8) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "XDNA context columns must be in [1, 8]");
  }
  iree_hal_amd_xdna_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*context),
                                             (void**)&context));
  context->host_allocator = host_allocator;
  context->driver = driver;
  iree_hal_driver_retain(driver);
  context->api = api;
  context->xdna = xdna;
  context->event_sink = event_sink;
  iree_status_t status = iree_hal_amd_xdna_context_open(
      context, instance, system_scope, endpoint_id, column_count);
  if (iree_status_is_ok(status)) {
    *out_context = context;
  } else {
    iree_hal_amd_xdna_context_destroy(context);
  }
  return status;
}
