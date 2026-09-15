// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/device.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/structure.h"
#include "libamdf/src/xdna/endpoint_profile.h"
#include "libamdf/src/xdna/memory.h"
#include "libamdf/src/xdna/umd/device.h"

typedef struct amdf_xdna_device_t {
  // Generic device state shared by every engine implementation.
  amdf_device_t base;
  // Immutable endpoint profile selected before native device creation.
  const amdf_xdna_endpoint_profile_t* profile;
  // Exact native ordinary-address-domain state.
  amdf_xdna_umd_device_t* umd;
  // Immutable identity and reset epoch returned by the provider.
  amdf_xdna_device_info_t info;
} amdf_xdna_device_t;

_Static_assert(offsetof(amdf_xdna_device_t, base) == 0,
               "XDNA device base must be the first field");

static amdf_status_t amdf_xdna_device_validate_create_info(
    const amdf_xdna_device_create_info_t* create_info) {
  return amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO,
      (uint32_t)sizeof(amdf_xdna_device_create_info_t));
}

static amdf_status_t amdf_xdna_device_destroy_native(
    amdf_device_t* base_device) {
  amdf_xdna_device_t* device = (amdf_xdna_device_t*)base_device;
  const amdf_status_t status = amdf_xdna_umd_device_destroy(device->umd);
  if (amdf_status_is_ok(status)) {
    device->umd = NULL;
  }
  return status;
}

static const amdf_device_vtable_t amdf_xdna_device_vtable = {
    .query_memory_profile = amdf_xdna_device_query_memory_profile,
    .memory_prepare = amdf_xdna_memory_prepare,
    .memory_prepare_import = amdf_xdna_memory_prepare_import,
    .destroy_native = amdf_xdna_device_destroy_native,
};

amdf_status_t AMDF_CALL
amdf_xdna_device_create(amdf_endpoint_t* endpoint,
                        const amdf_xdna_device_create_info_t* create_info,
                        amdf_device_t** out_device) {
  if (out_device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (endpoint == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  const amdf_xdna_endpoint_profile_t* profile =
      amdf_xdna_endpoint_profile_select(
          amdf_endpoint_get_cached_info(endpoint));
  if (profile == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_status_t status = amdf_xdna_device_validate_create_info(create_info);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  const amdf_allocator_t host_allocator =
      amdf_endpoint_host_allocator(endpoint);
  amdf_xdna_device_t* device = NULL;
  status = amdf_calloc(host_allocator, sizeof(*device),
                       amdf_alignof(amdf_xdna_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_device_initialize(&device->base, &amdf_xdna_device_vtable,
                                  endpoint, AMDF_ENGINE_KIND_XDNA);
  device->profile = profile;

  amdf_xdna_umd_device_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    status = amdf_xdna_umd_device_create(amdf_endpoint_get_platform(endpoint),
                                         profile, host_allocator, &device->umd,
                                         &result);
  }
  if (amdf_status_is_ok(status)) {
    device->info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
    device->info.structure_size = sizeof(device->info);
    device->info.id = result.id;
    device->info.reset_epoch = result.reset_epoch;
    device->info.placement_modes = result.placement_modes;
    *out_device = &device->base;
  } else {
    if (device->base.endpoint != NULL) {
      amdf_device_deinitialize(&device->base);
    }
    amdf_free(host_allocator, device);
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_xdna_device_query_info(
    amdf_device_t* device, amdf_xdna_device_info_t* out_info) {
  if (!amdf_device_is_engine(device, AMDF_ENGINE_KIND_XDNA)) {
    return device == NULL
               ? amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT)
               : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO,
      (uint32_t)sizeof(amdf_xdna_device_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  const amdf_xdna_device_t* xdna_device = (const amdf_xdna_device_t*)device;
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = xdna_device->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_xdna_umd_device_t* amdf_xdna_device_get_umd(amdf_device_t* device) {
  return ((amdf_xdna_device_t*)device)->umd;
}

const amdf_xdna_device_info_t* amdf_xdna_device_get_info(
    const amdf_device_t* device) {
  return &((const amdf_xdna_device_t*)device)->info;
}

const amdf_xdna_endpoint_profile_t* amdf_xdna_device_get_profile(
    const amdf_device_t* device) {
  return ((const amdf_xdna_device_t*)device)->profile;
}

uint64_t amdf_xdna_device_query_reset_epoch(const amdf_device_t* device) {
  return ((const amdf_xdna_device_t*)device)->info.reset_epoch;
}
