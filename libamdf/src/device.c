// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/device.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/platform/instance.h"
#include "libamdf/src/structure.h"

amdf_status_t amdf_device_initialize(amdf_device_t* device,
                                     const amdf_device_vtable_t* vtable,
                                     amdf_endpoint_t* endpoint,
                                     amdf_engine_kind_t engine_kind) {
  const amdf_status_t status = amdf_endpoint_register_device(endpoint);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->host_allocator = amdf_endpoint_host_allocator(endpoint);
  device->vtable = vtable;
  device->endpoint = endpoint;
  device->endpoint_id = amdf_endpoint_get_cached_info(endpoint)->id;
  device->provider_instance = amdf_endpoint_get_instance(endpoint);
  device->engine_kind = engine_kind;
  amdf_child_tracker_initialize(&device->children);
  return AMDF_STATUS_OK;
}

void amdf_device_deinitialize(amdf_device_t* device) {
  amdf_endpoint_unregister_device(device->endpoint);
  device->endpoint = NULL;
  device->provider_instance = NULL;
}

bool amdf_device_is_engine(const amdf_device_t* device,
                           amdf_engine_kind_t expected_engine_kind) {
  return device != NULL && device->engine_kind == expected_engine_kind;
}

bool amdf_device_shares_provider_instance(const amdf_device_t* lhs,
                                          const amdf_device_t* rhs) {
  return lhs != NULL && rhs != NULL && lhs->provider_instance != NULL &&
         lhs->provider_instance == rhs->provider_instance;
}

amdf_allocator_t amdf_device_host_allocator(const amdf_device_t* device) {
  return device->host_allocator;
}

amdf_status_t amdf_device_register_child(amdf_device_t* device) {
  return amdf_child_tracker_register(&device->children);
}

void amdf_device_unregister_child(amdf_device_t* device) {
  amdf_child_tracker_unregister(&device->children);
}

amdf_status_t AMDF_CALL amdf_device_query_queue_family_placement_info(
    amdf_device_t* device, uint32_t queue_family_ordinal,
    amdf_queue_family_placement_info_t* out_info) {
  if (device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_PLACEMENT_INFO,
      (uint32_t)sizeof(amdf_queue_family_placement_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (queue_family_ordinal >=
      amdf_endpoint_get_cached_info(device->endpoint)->queue_family_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_queue_family_placement_info_t info =
      device->vtable->query_queue_family_placement(device,
                                                   queue_family_ordinal);
  info.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_PLACEMENT_INFO;
  info.structure_size = out_info->structure_size;
  info.next = out_info->next;
  info.ordinal = queue_family_ordinal;
  *out_info = info;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_device_destroy(amdf_device_t* device) {
  if (device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_child_tracker_count(&device->children) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_platform_instance_t* platform = device->provider_instance->platform;
  amdf_platform_instance_lock_native(platform);
  const amdf_status_t status = device->vtable->destroy_native(device);
  amdf_platform_instance_unlock_native(platform);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = device->host_allocator;
    amdf_device_deinitialize(device);
    amdf_free(host_allocator, device);
  }
  return status;
}
