// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/context.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/device.h"
#include "libamdf/src/structure.h"
#include "libamdf/src/xdna/device.h"
#include "libamdf/src/xdna/endpoint_profile.h"
#include "libamdf/src/xdna/memory.h"

struct amdf_xdna_context_t {
  // Ordinary-address-domain device borrowed for the context lifetime.
  amdf_device_t* device;
  // Native scheduling, placement, completion, and private execution state.
  amdf_xdna_umd_context_t* umd;
  // Borrowed descriptor for caller-owned context-qualified instruction storage.
  amdf_memory_scope_t memory_scope;
  // Immutable identity and logical admission returned by the provider.
  amdf_xdna_context_info_t info;
  // Fixed backing, exposed only when the device supports a placement mode.
  amdf_xdna_context_placement_info_t placement_info;
  // Number of live queues borrowing this context.
  amdf_child_tracker_t children;
};

static amdf_status_t amdf_xdna_context_validate_create_info(
    amdf_device_t* device, const amdf_xdna_context_create_info_t* create_info) {
  if (!amdf_device_is_engine(device, AMDF_ENGINE_KIND_XDNA)) {
    return device == NULL
               ? amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT)
               : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO,
      (uint32_t)sizeof(amdf_xdna_context_create_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  const amdf_xdna_endpoint_profile_t* profile =
      amdf_xdna_device_get_profile(device);
  const amdf_xdna_endpoint_info_t* endpoint_info =
      amdf_xdna_endpoint_profile_get_info(profile);
  const amdf_xdna_scheduling_modes_t known_scheduling_modes =
      AMDF_XDNA_SCHEDULING_MODE_EXCLUSIVE | AMDF_XDNA_SCHEDULING_MODE_SPATIAL |
      AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
  if (create_info->acceptable_scheduling_modes == 0 ||
      (create_info->acceptable_scheduling_modes & ~known_scheduling_modes) !=
          0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_xdna_umd_context_capabilities_t context_capabilities =
      amdf_xdna_umd_query_context_capabilities(profile);
  if ((create_info->acceptable_scheduling_modes &
       endpoint_info->context.scheduling_modes &
       context_capabilities.scheduling_modes) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (create_info->logical_column_count <
          endpoint_info->context.minimum_column_count ||
      create_info->logical_column_count >
          endpoint_info->context.maximum_column_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint32_t count_offset = create_info->logical_column_count -
                                endpoint_info->context.minimum_column_count;
  if (count_offset % endpoint_info->context.column_count_granularity != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (create_info->physical_column_origin !=
      AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY) {
    if (create_info->physical_column_origin <
        endpoint_info->array.column_origin) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    const uint32_t physical_offset = create_info->physical_column_origin -
                                     endpoint_info->array.column_origin;
    if (physical_offset > endpoint_info->array.column_count ||
        create_info->logical_column_count >
            endpoint_info->array.column_count - physical_offset) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_xdna_context_create(
    amdf_device_t* device, const amdf_xdna_context_create_info_t* create_info,
    amdf_xdna_context_t** out_context) {
  if (out_context == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status =
      amdf_xdna_context_validate_create_info(device, create_info);
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  const amdf_allocator_t host_allocator = amdf_device_host_allocator(device);
  amdf_xdna_context_t* context = NULL;
  status = amdf_calloc(host_allocator, sizeof(*context),
                       amdf_alignof(amdf_xdna_context_t), (void**)&context);
  if (!amdf_status_is_ok(status)) return status;
  status = amdf_device_register_child(device);
  if (amdf_status_is_ok(status)) {
    context->device = device;
    amdf_child_tracker_initialize(&context->children);
  }

  amdf_xdna_umd_context_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    status = amdf_xdna_umd_context_create(amdf_xdna_device_get_umd(device),
                                          create_info, &context->umd, &result);
  }
  if (amdf_status_is_ok(status)) {
    const amdf_xdna_endpoint_info_t* endpoint_info =
        amdf_xdna_endpoint_profile_get_info(
            amdf_xdna_device_get_profile(device));
    context->info.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_INFO;
    context->info.structure_size = sizeof(context->info);
    context->info.id = result.id;
    context->info.device_id = amdf_xdna_device_get_info(device)->id;
    context->info.reset_epoch = amdf_xdna_device_query_reset_epoch(device);
    context->info.scheduling_mode = result.scheduling_mode;
    context->info.logical_column_count = create_info->logical_column_count;
    context->info.row_count = endpoint_info->array.row_count;
    context->placement_info.type =
        AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_PLACEMENT_INFO;
    context->placement_info.structure_size = sizeof(context->placement_info);
    context->placement_info.column_origin = result.physical_column_origin;
    context->placement_info.column_count = result.physical_column_count;
    amdf_xdna_memory_scope_initialize(device, context->umd,
                                      &context->memory_scope);
    *out_context = context;
  } else {
    if (context->device != NULL) {
      amdf_device_unregister_child(context->device);
    }
    amdf_free(host_allocator, context);
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_xdna_context_query_info(
    amdf_xdna_context_t* context, amdf_xdna_context_info_t* out_info) {
  if (context == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_INFO,
      (uint32_t)sizeof(amdf_xdna_context_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = context->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_xdna_context_query_placement_info(
    amdf_xdna_context_t* context,
    amdf_xdna_context_placement_info_t* out_info) {
  if (context == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_PLACEMENT_INFO,
      (uint32_t)sizeof(amdf_xdna_context_placement_info_t));
  if (!amdf_status_is_ok(status)) return status;
  if (amdf_xdna_device_get_info(context->device)->placement_modes == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = context->placement_info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_device_t* amdf_xdna_context_get_device(amdf_xdna_context_t* context) {
  return context->device;
}

amdf_memory_scope_t* amdf_xdna_context_get_memory_scope(
    amdf_xdna_context_t* context) {
  return &context->memory_scope;
}

amdf_status_t AMDF_CALL amdf_xdna_context_enumerate_memory_scopes(
    amdf_xdna_context_t* context, uint32_t capacity,
    amdf_memory_scope_t** scopes, uint32_t* out_count) {
  if (context == NULL || out_count == NULL ||
      (capacity != 0 && scopes == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (capacity != 0) scopes[0] = &context->memory_scope;
  *out_count = 1;
  return capacity == 0 ? amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)
                       : AMDF_STATUS_OK;
}

const amdf_xdna_context_info_t* amdf_xdna_context_get_info(
    const amdf_xdna_context_t* context) {
  return &context->info;
}

amdf_xdna_umd_context_t* amdf_xdna_context_get_umd(
    amdf_xdna_context_t* context) {
  return context->umd;
}

amdf_status_t amdf_xdna_context_register_child(amdf_xdna_context_t* context) {
  return amdf_child_tracker_register(&context->children);
}

void amdf_xdna_context_unregister_child(amdf_xdna_context_t* context) {
  amdf_child_tracker_unregister(&context->children);
}

amdf_status_t AMDF_CALL
amdf_xdna_context_destroy(amdf_xdna_context_t* context) {
  if (context == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_child_tracker_count(&context->children) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = amdf_xdna_umd_context_destroy(context->umd);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator =
        amdf_device_host_allocator(context->device);
    amdf_device_unregister_child(context->device);
    amdf_free(host_allocator, context);
  }
  return status;
}
