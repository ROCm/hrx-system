// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/device.h"

#include <stddef.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/gpu/memory.h"
#include "libamdf/src/gpu/memory_reservation.h"
#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/gpu/umd/instance.h"
#include "libamdf/src/gpu/umd/native.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/platform/instance.h"
#include "libamdf/src/structure.h"

typedef struct amdf_gpu_device_t {
  // Generic device state shared by every engine implementation.
  amdf_device_t base;
  // Exact native execution and address-domain state.
  amdf_gpu_umd_device_t* umd;
  // Immutable operations selected from the endpoint before native creation.
  const amdf_gpu_umd_api_t* umd_api;
  // Immutable identity, reset state, and achieved features from the provider.
  amdf_gpu_device_info_t info;
  // Immutable placement facts established before publishing this device.
  amdf_queue_family_placement_info_t
      queue_placements[AMDF_GPU_QUEUE_FAMILY_CAPACITY];
} amdf_gpu_device_t;

_Static_assert(offsetof(amdf_gpu_device_t, base) == 0,
               "GPU device base must be the first field");

static amdf_status_t amdf_gpu_device_destroy_native(
    amdf_device_t* base_device) {
  amdf_gpu_device_t* device = (amdf_gpu_device_t*)base_device;
  const amdf_status_t status = device->umd_api->device_destroy(device->umd);
  if (amdf_status_is_ok(status)) {
    device->umd = NULL;
  }
  return status;
}

static amdf_queue_family_placement_info_t
amdf_gpu_device_query_queue_family_placement(const amdf_device_t* base_device,
                                             uint32_t queue_family_ordinal) {
  const amdf_gpu_device_t* device = (const amdf_gpu_device_t*)base_device;
  return device->queue_placements[queue_family_ordinal];
}

static amdf_status_t amdf_gpu_scope_sample_observation(
    const amdf_memory_scope_t* scope,
    amdf_memory_scope_observation_t* out_observation) {
  amdf_platform_endpoint_t* platform =
      amdf_endpoint_get_platform(scope->owner.endpoint);
  return amdf_gpu_umd_api_for_endpoint(platform)->sample_memory_scope(
      platform, scope->flags, out_observation);
}

static const amdf_device_vtable_t amdf_gpu_device_vtable = {
    .query_queue_family_placement =
        amdf_gpu_device_query_queue_family_placement,
    .query_memory_reservation_profile =
        amdf_gpu_device_query_memory_reservation_profile,
    .query_memory_profile = amdf_gpu_device_query_memory_profile,
    .memory_prepare = amdf_gpu_memory_prepare,
    .memory_prepare_import = amdf_gpu_memory_prepare_import,
    .destroy_native = amdf_gpu_device_destroy_native,
};

amdf_status_t AMDF_CALL amdf_gpu_device_create(
    amdf_endpoint_t* endpoint, const amdf_gpu_device_create_info_t* create_info,
    amdf_device_t** out_device) {
  if (out_device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (endpoint == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t validation_status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO,
      (uint32_t)sizeof(amdf_gpu_device_create_info_t));
  if (!amdf_status_is_ok(validation_status)) {
    return validation_status;
  }
  if (create_info->reserved != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  if (amdf_endpoint_get_cached_info(endpoint)->engine_kind !=
      AMDF_ENGINE_KIND_GPU) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_instance_t* instance = amdf_endpoint_get_instance(endpoint);
  const amdf_native_lifetime_t native_lifetime =
      amdf_instance_native_lifetime(instance);

  const amdf_allocator_t host_allocator =
      amdf_endpoint_host_allocator(endpoint);
  amdf_gpu_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*device),
                  amdf_alignof(amdf_gpu_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_device_initialize(&device->base, &amdf_gpu_device_vtable,
                                  endpoint, AMDF_ENGINE_KIND_GPU);
  device->umd_api =
      amdf_gpu_umd_api_for_endpoint(amdf_endpoint_get_platform(endpoint));
  amdf_gpu_umd_device_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    amdf_platform_instance_lock_native(instance->platform);
    if (instance->gpu_api == NULL) {
      instance->gpu_api = device->umd_api;
    }
    amdf_assert(instance->gpu_api == device->umd_api &&
                "one instance must use one GPU implementation");
    status = device->umd_api->instance_prepare(&instance->gpu, native_lifetime,
                                               host_allocator);
    if (amdf_status_is_ok(status)) {
      status = device->umd_api->device_create(
          instance->gpu, amdf_endpoint_get_platform(endpoint), host_allocator,
          native_lifetime, &device->umd, &result);
    }
    if (amdf_status_is_ok(status)) {
      const amdf_memory_scope_t scope = {
          .flags = result.memory.flags,
          .capacity_byte_length =
              (result.memory.flags & AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED)
                  ? instance->system_memory_scope.capacity_byte_length
                  : result.memory.capacity_byte_length,
          .sample_observation = amdf_gpu_scope_sample_observation,
      };
      amdf_endpoint_publish_local_memory_scope(endpoint, &scope);
    }
    amdf_platform_instance_unlock_native(instance->platform);
  }
  if (amdf_status_is_ok(status)) {
    device->info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
    device->info.structure_size = sizeof(device->info);
    device->info.id = result.id;
    device->info.reset_epoch = result.reset_epoch;
    device->info.features = result.features;
    memcpy(device->queue_placements, result.queue_placements,
           sizeof(device->queue_placements));
    *out_device = &device->base;
  } else {
    if (device->base.endpoint != NULL) {
      amdf_device_deinitialize(&device->base);
    }
    amdf_free(host_allocator, device);
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_gpu_device_query_info(
    amdf_device_t* device, amdf_gpu_device_info_t* out_info) {
  if (!amdf_device_is_engine(device, AMDF_ENGINE_KIND_GPU)) {
    return device == NULL
               ? amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT)
               : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO,
      (uint32_t)sizeof(amdf_gpu_device_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  const amdf_gpu_device_t* gpu_device = (const amdf_gpu_device_t*)device;
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = gpu_device->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_gpu_umd_device_t* amdf_gpu_device_get_umd(amdf_device_t* device) {
  return ((amdf_gpu_device_t*)device)->umd;
}

const amdf_gpu_umd_api_t* amdf_gpu_device_get_umd_api(
    const amdf_device_t* device) {
  return ((const amdf_gpu_device_t*)device)->umd_api;
}

const amdf_gpu_device_info_t* amdf_gpu_device_get_info(
    const amdf_device_t* device) {
  return &((const amdf_gpu_device_t*)device)->info;
}

uint64_t amdf_gpu_device_query_reset_epoch(const amdf_device_t* device) {
  return ((const amdf_gpu_device_t*)device)->info.reset_epoch;
}
