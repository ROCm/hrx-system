// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/tsan.h"

#include <string.h>

#include "iree/hal/drivers/amd/status.h"

void iree_hal_amd_gpu_tsan_options_initialize(
    iree_hal_amd_gpu_tsan_options_t* out_options) {
  *out_options = (iree_hal_amd_gpu_tsan_options_t){
      .shadow_size = UINT64_C(512) * 1024 * 1024,
  };
}

iree_status_t iree_hal_amd_gpu_tsan_calculate_config(
    const iree_hal_amd_gpu_tsan_options_t* options,
    iree_hal_amd_gpu_tsan_config_t* out_config) {
  *out_config = (iree_hal_amd_gpu_tsan_config_t){0};
  if (!options->shadow_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "GPU TSAN shadow size must be nonzero");
  }
  if (options->shadow_size > (UINT64_C(1) << 32)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "GPU TSAN shadow exceeds the 4 GiB address range");
  }
  *out_config = (iree_hal_amd_gpu_tsan_config_t){
      .record_length = sizeof(*out_config),
      .flags = 3,
      .shadow_size = options->shadow_size,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_gpu_tsan_select_scope(
    const iree_hal_amd_gpu_context_t* context, uint32_t queue_family_ordinal,
    amdf_memory_scope_t* scope, amdf_memory_device_access_t* access,
    iree_hal_amd_fabric_memory_options_t* out_options, bool* out_found) {
  *out_found = false;
  *access = (amdf_memory_device_access_t){
      .device = context->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
          },
  };
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_scope_query_info(scope, &info),
      "TSAN memory scope"));
  iree_status_t status = iree_ok_status();
  bool found = false;
  for (uint32_t i = 0;
       i < info.memory_profile_count && !found && iree_status_is_ok(status);
       ++i) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t native_status =
        context->api->memory_scope_query_device_profile(
            scope, i, 1, access, &profile, &capabilities);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    status =
        IREE_HAL_AMD_STATUS_FROM_AMDF(native_status, "TSAN memory profile");
    if (iree_status_is_ok(status) &&
        iree_all_bits_set(profile.roles,
                          AMDF_MEMORY_PROFILE_ROLE_CREATE |
                              AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) &&
        iree_all_bits_set(
            profile.host_mapping.supported_access,
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE)) {
      const iree_hal_amd_fabric_memory_options_t memory = {
          .scope = scope,
          .profile = profile,
          .access_count = 1,
          .accesses = access,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
          .host_access = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      };
      status = iree_hal_amd_fabric_memory_query_host_publication(
          context->api, &memory, 0, queue_family_ordinal, &found);
      if (iree_status_is_ok(status) && found) {
        *out_options = memory;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_found = found;
  }
  return status;
}

iree_status_t iree_hal_amd_gpu_tsan_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t queue_family_ordinal,
    const iree_hal_amd_gpu_tsan_options_t* options,
    iree_hal_amd_gpu_tsan_t* out_tsan) {
  *out_tsan = (iree_hal_amd_gpu_tsan_t){0};
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_tsan_calculate_config(options, &out_tsan->config));
  amdf_memory_device_access_t access;
  iree_hal_amd_fabric_memory_options_t memory_options;
  bool found = false;
  // Prefer device-local storage only when its CPU writes can be published.
  // The system scope also covers devices without a separate local heap.
  for (uint32_t i = 0; i <= context->memory_scope_count && !found; ++i) {
    amdf_memory_scope_t* scope = i < context->memory_scope_count
                                     ? context->memory_scopes[i]
                                     : context->fabric->system_scope;
    IREE_RETURN_IF_ERROR(
        iree_hal_amd_gpu_tsan_select_scope(context, queue_family_ordinal, scope,
                                           &access, &memory_options, &found));
  }
  if (!found) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "GPU TSAN shadow has no host-publishable backing profile");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_memory_allocate(
      context->api, &memory_options, out_tsan->config.shadow_size, 64,
      context->event_sink, &out_tsan->memory, NULL));
  uint64_t address = 0;
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(out_tsan->memory.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
      "TSAN shadow address"));
  out_tsan->config.shadow_base = address;
  memset(out_tsan->memory.contents.data, 0,
         out_tsan->memory.contents.data_length);
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->host_mapping_cache_control(
          out_tsan->memory.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
          out_tsan->memory.contents.data_length),
      "TSAN shadow publication");
}

void iree_hal_amd_gpu_tsan_deinitialize(
    const iree_hal_amd_gpu_context_t* context, iree_hal_amd_gpu_tsan_t* tsan) {
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &tsan->memory);
  *tsan = (iree_hal_amd_gpu_tsan_t){0};
}

void iree_hal_amd_gpu_tsan_registry_initialize(
    const iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    uint32_t provisioned_count, const iree_hal_amd_gpu_tsan_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_tsan_registry_t* registry) {
  *registry = (iree_hal_amd_gpu_tsan_registry_t){
      .context = context,
      .host_allocator = host_allocator,
      .options = *options,
      .family_ordinal = family_ordinal,
      .provisioned_count = provisioned_count,
  };
  iree_slim_mutex_initialize(&registry->mutex);
}

void iree_hal_amd_gpu_tsan_registry_deinitialize(
    iree_hal_amd_gpu_tsan_registry_t* registry) {
  if (!registry) {
    return;
  }
  IREE_ASSERT(!registry->clients,
              "instrumented images must be released before their device");
  for (uint32_t i = 0; i < registry->count; ++i) {
    iree_hal_amd_gpu_tsan_registry_remove_queue(registry, i);
  }
  iree_slim_mutex_deinitialize(&registry->mutex);
}

iree_status_t iree_hal_amd_gpu_tsan_registry_add_queue(
    iree_hal_amd_gpu_tsan_registry_t* registry, uint32_t ordinal) {
  if (!registry) {
    return iree_ok_status();
  }
  iree_slim_mutex_lock(&registry->mutex);
  iree_hal_amd_gpu_tsan_t* shadow = NULL;
  iree_status_t status = iree_allocator_malloc(
      registry->host_allocator, sizeof(*shadow), (void**)&shadow);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_gpu_tsan_initialize(registry->context,
                                              registry->family_ordinal,
                                              &registry->options, shadow);
  }
  for (iree_hal_amd_gpu_tsan_client_t* client = registry->clients;
       client && iree_status_is_ok(status); client = client->next) {
    status = client->prepare(client->user_data, ordinal, &shadow->config);
  }
  if (iree_status_is_ok(status)) {
    registry->queues[ordinal] = shadow;
    registry->count = iree_max(registry->count, ordinal + 1);
  } else {
    for (iree_hal_amd_gpu_tsan_client_t* client = registry->clients; client;
         client = client->next) {
      client->release(client->user_data, ordinal);
    }
    if (shadow) {
      iree_hal_amd_gpu_tsan_deinitialize(registry->context, shadow);
    }
    iree_allocator_free(registry->host_allocator, shadow);
  }
  iree_slim_mutex_unlock(&registry->mutex);
  return status;
}

void iree_hal_amd_gpu_tsan_registry_remove_queue(
    iree_hal_amd_gpu_tsan_registry_t* registry, uint32_t ordinal) {
  if (!registry) {
    return;
  }
  iree_slim_mutex_lock(&registry->mutex);
  iree_hal_amd_gpu_tsan_t* shadow = registry->queues[ordinal];
  if (shadow) {
    for (iree_hal_amd_gpu_tsan_client_t* client = registry->clients; client;
         client = client->next) {
      client->release(client->user_data, ordinal);
    }
    registry->queues[ordinal] = NULL;
    iree_hal_amd_gpu_tsan_deinitialize(registry->context, shadow);
    iree_allocator_free(registry->host_allocator, shadow);
  }
  iree_slim_mutex_unlock(&registry->mutex);
}

iree_status_t iree_hal_amd_gpu_tsan_registry_register(
    iree_hal_amd_gpu_tsan_registry_t* registry,
    iree_hal_amd_gpu_tsan_client_t* client) {
  iree_slim_mutex_lock(&registry->mutex);
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < registry->count && iree_status_is_ok(status); ++i) {
    if (registry->queues[i]) {
      status =
          client->prepare(client->user_data, i, &registry->queues[i]->config);
    }
  }
  if (iree_status_is_ok(status)) {
    client->registry = registry;
    client->next = registry->clients;
    if (client->next) {
      client->next->previous = client;
    }
    registry->clients = client;
  } else {
    for (uint32_t i = 0; i < registry->count; ++i) {
      client->release(client->user_data, i);
    }
  }
  iree_slim_mutex_unlock(&registry->mutex);
  return status;
}

void iree_hal_amd_gpu_tsan_registry_unregister(
    iree_hal_amd_gpu_tsan_client_t* client) {
  iree_hal_amd_gpu_tsan_registry_t* registry = client->registry;
  if (!registry) {
    return;
  }
  iree_slim_mutex_lock(&registry->mutex);
  if (client->previous) {
    client->previous->next = client->next;
  } else {
    registry->clients = client->next;
  }
  if (client->next) {
    client->next->previous = client->previous;
  }
  client->registry = NULL;
  iree_slim_mutex_unlock(&registry->mutex);
}
