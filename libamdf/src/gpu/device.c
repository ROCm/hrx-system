// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/device.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/gpu/endpoint_profile.h"
#include "libamdf/src/gpu/memory.h"
#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/gpu/umd/instance.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/platform/instance.h"
#include "libamdf/src/structure.h"

typedef struct amdf_gpu_device_t {
  // Generic device state shared by every engine implementation.
  amdf_device_t base;
  // Exact native execution and address-domain state.
  amdf_gpu_umd_device_t* umd;
  // Immutable identity, reset state, and achieved features from the provider.
  amdf_gpu_device_info_t info;
} amdf_gpu_device_t;

_Static_assert(offsetof(amdf_gpu_device_t, base) == 0,
               "GPU device base must be the first field");

static amdf_status_t amdf_gpu_device_destroy_native(
    amdf_device_t* base_device) {
  amdf_gpu_device_t* device = (amdf_gpu_device_t*)base_device;
  const amdf_status_t status = amdf_gpu_umd_device_destroy(device->umd);
  if (amdf_status_is_ok(status)) {
    device->umd = NULL;
  }
  return status;
}

static const amdf_device_vtable_t amdf_gpu_device_vtable = {
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

  const void* untyped_profile = NULL;
  const amdf_status_t profile_status = amdf_endpoint_query_engine_profile(
      endpoint, AMDF_ENGINE_KIND_GPU, &untyped_profile);
  if (!amdf_status_is_ok(profile_status)) {
    return profile_status;
  }
  amdf_instance_t* instance = amdf_endpoint_get_instance(endpoint);
  const amdf_native_lifetime_t native_lifetime =
      amdf_instance_native_lifetime(instance);
  const amdf_gpu_endpoint_profile_t* profile = untyped_profile;
  if (!profile->native_lifetimes[native_lifetime].supported) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const amdf_allocator_t host_allocator =
      amdf_endpoint_host_allocator(endpoint);
  amdf_gpu_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*device),
                  amdf_alignof(amdf_gpu_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_device_initialize(&device->base, &amdf_gpu_device_vtable,
                                  endpoint, AMDF_ENGINE_KIND_GPU);
  amdf_gpu_umd_device_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    amdf_platform_instance_lock_native(instance->platform);
    status = amdf_gpu_umd_instance_prepare(&instance->gpu, native_lifetime,
                                           host_allocator);
    if (amdf_status_is_ok(status)) {
      status = amdf_gpu_umd_device_create(
          instance->gpu, amdf_endpoint_get_platform(endpoint), host_allocator,
          native_lifetime, &device->umd, &result);
    }
    amdf_platform_instance_unlock_native(instance->platform);
  }
  if (amdf_status_is_ok(status)) {
    device->info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO;
    device->info.structure_size = sizeof(device->info);
    device->info.id = result.id;
    device->info.reset_epoch = result.reset_epoch;
    device->info.features = result.features;
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

const amdf_gpu_device_info_t* amdf_gpu_device_get_info(
    const amdf_device_t* device) {
  return &((const amdf_gpu_device_t*)device)->info;
}

uint64_t amdf_gpu_device_query_reset_epoch(const amdf_device_t* device) {
  return ((const amdf_gpu_device_t*)device)->info.reset_epoch;
}
