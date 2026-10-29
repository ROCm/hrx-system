// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/context.h"

#include "iree/hal/drivers/amd/fabric/event.h"
#include "iree/hal/drivers/amd/status.h"

iree_status_t iree_hal_amd_gpu_context_prepare_memory(
    iree_hal_amd_gpu_context_t* context, uint32_t node_id) {
  iree_hal_amd_fabric_allocation_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_acquire_allocation_queue(
      context->fabric, &context->endpoint_info.id, node_id, &queue));
  iree_slim_mutex_initialize(&context->memory.mutex);
  iree_notification_initialize(&context->memory.notification);
  context->memory.queue = queue;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_context_memory_begin(
    iree_hal_amd_gpu_context_t* context) {
  iree_slim_mutex_lock(&context->memory.mutex);
  ++context->memory.pending_count;
  iree_slim_mutex_unlock(&context->memory.mutex);
}

static void iree_hal_amd_gpu_context_memory_end(
    iree_hal_amd_gpu_context_t* context) {
  iree_slim_mutex_lock(&context->memory.mutex);
  if (--context->memory.pending_count == 0) {
    iree_notification_post(&context->memory.notification, IREE_ALL_WAITERS);
  }
  // The join predicate takes this same lock. Publishing zero alone would let
  // teardown free the notification before post's final access to its storage.
  iree_slim_mutex_unlock(&context->memory.mutex);
}

static void iree_hal_amd_gpu_context_memory_acquired(void* user_data,
                                                     iree_status_t status) {
  iree_hal_amd_gpu_memory_acquire_t* operation = user_data;
  iree_hal_amd_gpu_context_t* context = operation->context;
  const iree_hal_amd_fabric_memory_callback_t completion =
      operation->completion;
  completion.fn(completion.user_data, status);
  iree_hal_amd_gpu_context_memory_end(context);
}

static void iree_hal_amd_gpu_context_memory_released(void* user_data) {
  iree_hal_amd_gpu_memory_release_t* operation = user_data;
  iree_hal_amd_gpu_context_t* context = operation->context;
  const iree_hal_amd_fabric_memory_release_callback_t completion =
      operation->completion;
  completion.fn(completion.user_data);
  iree_hal_amd_gpu_context_memory_end(context);
}

void iree_hal_amd_gpu_context_acquire_memory(
    iree_hal_amd_gpu_context_t* context, iree_host_size_t count,
    const iree_hal_amd_fabric_memory_request_t* requests,
    iree_hal_amd_fabric_memory_t* memories,
    iree_hal_amd_fabric_memory_callback_t completion,
    iree_hal_amd_gpu_memory_acquire_t* operation) {
  *operation = (iree_hal_amd_gpu_memory_acquire_t){
      .request =
          {
              .operation = {.api = context->api,
                            .event_sink = context->event_sink},
              .count = count,
              .requests = requests,
              .memories = memories,
              .completion = {.fn = iree_hal_amd_gpu_context_memory_acquired,
                             .user_data = operation},
          },
      .context = context,
      .completion = completion,
  };
  iree_hal_amd_gpu_context_memory_begin(context);
  iree_hal_amd_fabric_allocation_queue_acquire_memory(context->memory.queue,
                                                      &operation->request);
}

void iree_hal_amd_gpu_context_release_memory(
    iree_hal_amd_gpu_context_t* context, iree_host_size_t count,
    iree_hal_amd_fabric_memory_t* memories,
    iree_hal_amd_fabric_memory_release_callback_t completion,
    iree_hal_amd_gpu_memory_release_t* operation) {
  *operation = (iree_hal_amd_gpu_memory_release_t){
      .request =
          {
              .operation = {.api = context->api,
                            .event_sink = context->event_sink},
              .count = count,
              .memories = memories,
              .completion = {.fn = iree_hal_amd_gpu_context_memory_released,
                             .user_data = operation},
          },
      .context = context,
      .completion = completion,
  };
  iree_hal_amd_gpu_context_memory_begin(context);
  iree_hal_amd_fabric_allocation_queue_release_memory(context->memory.queue,
                                                      &operation->request);
}

static bool iree_hal_amd_gpu_context_memory_is_idle(void* user_data) {
  iree_hal_amd_gpu_context_t* context = user_data;
  iree_slim_mutex_lock(&context->memory.mutex);
  bool idle = context->memory.pending_count == 0;
  iree_slim_mutex_unlock(&context->memory.mutex);
  return idle;
}

static void iree_hal_amd_gpu_context_memory_mapped(void* user_data,
                                                   iree_status_t status) {
  iree_hal_amd_gpu_memory_map_t* operation = user_data;
  iree_hal_amd_gpu_context_t* context = operation->context;
  const iree_hal_amd_fabric_memory_callback_t completion =
      operation->completion;
  completion.fn(completion.user_data, status);
  iree_hal_amd_gpu_context_memory_end(context);
}

void iree_hal_amd_gpu_context_map_memory(
    iree_hal_amd_gpu_context_t* context, iree_hal_memory_mapping_batch_t* batch,
    iree_hal_amd_fabric_memory_callback_t completion,
    iree_hal_amd_gpu_memory_map_t* operation) {
  *operation = (iree_hal_amd_gpu_memory_map_t){
      .request = {.batch = batch,
                  .completion = {.fn = iree_hal_amd_gpu_context_memory_mapped,
                                 .user_data = operation}},
      .context = context,
      .completion = completion,
  };
  iree_hal_amd_gpu_context_memory_begin(context);
  iree_hal_amd_fabric_allocation_queue_map_memory(context->memory.queue,
                                                  &operation->request);
}

void iree_hal_amd_gpu_context_join_memory(iree_hal_amd_gpu_context_t* context) {
  if (!context->memory.queue) {
    return;
  }
  iree_notification_await(&context->memory.notification,
                          iree_hal_amd_gpu_context_memory_is_idle, context,
                          iree_infinite_timeout());
}

void iree_hal_amd_gpu_context_deinitialize_memory(
    iree_hal_amd_gpu_context_t* context) {
  if (!context->memory.queue) {
    return;
  }
  iree_hal_amd_gpu_context_join_memory(context);
  iree_hal_amd_fabric_allocation_queue_drain(context->memory.queue);
  iree_hal_amd_fabric_allocation_queue_release(context->memory.queue);
  context->memory.queue = NULL;
  iree_notification_deinitialize(&context->memory.notification);
  iree_slim_mutex_deinitialize(&context->memory.mutex);
}

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

void iree_hal_amd_gpu_context_fail_execution(
    const iree_hal_amd_gpu_context_t* context, amdf_status_t status,
    const char* operation) {
  if (context->execution_failure.notify) {
    context->execution_failure.notify(context->execution_failure.user_data,
                                      status, operation);
  } else {
    iree_hal_amd_gpu_context_report(context, status, operation);
  }
}

bool iree_hal_amd_gpu_context_wake_failure(
    const iree_hal_amd_gpu_context_t* context) {
  return context->execution_failure.wake &&
         context->execution_failure.wake(context->execution_failure.user_data);
}

amdf_status_t iree_hal_amd_gpu_context_deinitialize_completion(
    iree_hal_amd_gpu_context_t* context) {
  if (context->event) {
    amdf_status_t status = context->gpu->event_destroy(context->event);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    context->event = NULL;
  }
  context->event_info = (amdf_gpu_event_info_t){0};
  iree_async_event_native_deinitialize(&context->kernel_event);
  context->kernel_notification = (amdf_native_event_t){0};
  return AMDF_STATUS_OK;
}

void iree_hal_amd_gpu_context_free(iree_hal_amd_gpu_context_t* context) {
  if (!context || context->cleanup_blocked) {
    return;
  }
  iree_hal_amd_gpu_context_deinitialize_memory(context);
  amdf_status_t event_status =
      iree_hal_amd_gpu_context_deinitialize_completion(context);
  if (!amdf_status_is_ok(event_status)) {
    iree_hal_amd_gpu_context_report(context, event_status, "gpu.event_destroy");
    return;
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
  if (context->endpoint_info.queue_family_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
        context->host_allocator, context->endpoint_info.queue_family_count,
        sizeof(*context->families), (void**)&context->families));
  }
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

  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_context_prepare_completion(
    iree_hal_amd_gpu_context_t* context,
    amdf_queue_publication_modes_t publication_mode) {
  if (publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
    IREE_RETURN_IF_ERROR(
        iree_async_event_native_initialize(&context->kernel_event));
    context->kernel_notification =
        iree_hal_amd_fabric_event_native_handle(&context->kernel_event);
    return iree_ok_status();
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

amdf_status_t iree_hal_amd_gpu_context_signal(
    const iree_hal_amd_gpu_context_t* context) {
  if (context->event) {
    return context->gpu->event_signal(context->event);
  }
  iree_async_event_native_set(&context->kernel_event);
  return AMDF_STATUS_OK;
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
                  AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_6,
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

static iree_status_t iree_hal_amd_gpu_context_select_program_memory(
    iree_hal_amd_gpu_context_t* context,
    const amdf_memory_device_access_t* requirements, amdf_memory_scope_t* scope,
    uint32_t native_family_ordinal, bool upload, bool* out_found,
    iree_hal_amd_fabric_memory_options_t* out_options) {
  *out_found = false;
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_scope_query_info(scope, &info),
      "query resident program scope"));
  for (uint32_t i = 0; i < info.memory_profile_count; ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t access = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(access),
    };
    amdf_status_t status = context->api->memory_scope_query_device_profile(
        scope, i, 1, requirements, &profile, &access);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        status, "qualify resident program profile"));
    if (!iree_all_bits_set(profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE) ||
        (!upload && !iree_all_bits_set(profile.supported_flags,
                                       AMDF_MEMORY_FLAG_DEVICE_LOCAL))) {
      continue;
    }
    bool mapped =
        iree_all_bits_set(profile.roles, AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        iree_all_bits_set(profile.supported_flags,
                          AMDF_MEMORY_FLAG_HOST_VISIBLE) &&
        iree_all_bits_set(profile.host_mapping.supported_access,
                          AMDF_MEMORY_MAP_FLAG_WRITE);
    iree_hal_amd_fabric_memory_options_t options = {
        .scope = scope,
        .profile = profile,
        .access_count = 1,
        .accesses = requirements,
        .required_flags = (mapped ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0) |
                          (upload ? 0 : AMDF_MEMORY_FLAG_DEVICE_LOCAL),
        .host_access = mapped ? AMDF_MEMORY_MAP_FLAG_WRITE : 0,
    };
    if (mapped) {
      // A CPU aperture alone does not qualify publication to command fetch.
      // Keep local placement when that edge is unavailable and initialize the
      // resident bytes through the ordinary device-side upload instead.
      IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_query_host_publication(
          context->api, &options, 0, native_family_ordinal, &mapped));
      if (!mapped) {
        options.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
        options.host_access = 0;
      }
    }
    if (upload && !mapped) {
      continue;
    }
    *out_options = options;
    *out_found = true;
    break;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_context_query_program_memory(
    iree_hal_amd_gpu_context_t* context, uint32_t native_family_ordinal,
    const amdf_memory_device_access_t* access,
    iree_hal_amd_fabric_memory_options_t* out_upload,
    iree_hal_amd_fabric_memory_options_t* out_resident) {
  bool found = false;
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_select_program_memory(
      context, access, context->fabric->system_scope, native_family_ordinal,
      true, &found, out_upload));
  if (!found) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "GPU has no host-visible cached program profile");
  }
  found = false;
  for (uint32_t i = 0; i < context->memory_scope_count && !found; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_select_program_memory(
        context, access, context->memory_scopes[i], native_family_ordinal,
        false, &found, out_resident));
  }
  if (!found) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_context_select_program_memory(
        context, access, context->fabric->system_scope, native_family_ordinal,
        false, &found, out_resident));
  }
  if (!found) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "GPU has no device-local program profile");
  }
  return iree_ok_status();
}
