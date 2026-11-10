// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "queue.h"

#include "hrx_internal.h"

static_assert(HRX_QUEUE_ROLE_DISPATCH ==
                  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
              "queue role mismatch");
static_assert(HRX_QUEUE_ROLE_TRANSFER ==
                  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER,
              "queue role mismatch");
static_assert(HRX_QUEUE_ROLE_HOST_CALL ==
                  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL,
              "queue role mismatch");
static_assert(HRX_QUEUE_ROLE_PROFILING ==
                  IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING,
              "queue role mismatch");
static_assert(HRX_QUEUE_ROLE_ATOMIC == IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC,
              "queue role mismatch");
static_assert((int)HRX_QUEUE_PLACEMENT_DEFAULT ==
                      (int)IREE_HAL_QUEUE_PLACEMENT_DEFAULT &&
                  (int)HRX_QUEUE_PLACEMENT_HOST_LOCAL ==
                      (int)IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL &&
                  (int)HRX_QUEUE_PLACEMENT_DEVICE_LOCAL ==
                      (int)IREE_HAL_QUEUE_PLACEMENT_DEVICE_LOCAL,
              "queue placement mismatch");
static_assert(HRX_QUEUE_FEATURE_COOPERATIVE_DISPATCH ==
                  IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH,
              "queue feature mismatch");

iree_status_t hrx_device_initialize_queues(hrx_device_t device) {
  const iree_hal_device_queue_spec_t* spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device->hal_device));
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < spec->family_count; ++i) {
    if (!iree_host_size_checked_add(
            count, spec->families[i].provisioned_queue_count, &count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "provisioned queue count overflows");
    }
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      iree_allocator_system(), count, sizeof(*device->queues),
      (void**)&device->queues));
  iree_host_size_t position = 0;
  for (iree_host_size_t i = 0; i < spec->family_count; ++i) {
    for (uint32_t j = 0; j < spec->families[i].provisioned_queue_count; ++j) {
      hrx_queue_t queue = &device->queues[position++];
      iree_atomic_ref_count_init(&queue->ref_count);
      queue->device = device;
      queue->hal_queue =
          iree_hal_device_queue(device->hal_device, (uint32_t)i, j);
    }
  }
  return iree_ok_status();
}

void hrx_device_deinitialize_queues(hrx_device_t device) {
  iree_allocator_free(iree_allocator_system(), device->queues);
  device->queues = NULL;
}

size_t hrx_device_queue_family_count(hrx_device_t device) {
  return device ? iree_hal_device_spec_queues(
                      iree_hal_device_spec(device->hal_device))
                      ->family_count
                : 0;
}

static const iree_hal_queue_family_spec_t* hrx_device_queue_family_spec(
    hrx_device_t device, uint32_t family_ordinal) {
  if (!device) {
    return NULL;
  }
  const iree_hal_queue_family_t* family =
      iree_hal_device_queue_family(device->hal_device, family_ordinal);
  return family ? iree_hal_queue_family_spec(family) : NULL;
}

hrx_status_t hrx_device_query_queue_family(hrx_device_t device,
                                           uint32_t family_ordinal,
                                           hrx_queue_family_info_t* out_info) {
  const iree_hal_queue_family_spec_t* spec =
      hrx_device_queue_family_spec(device, family_ordinal);
  if (!spec || !out_info) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid queue family or NULL output");
  }
  *out_info = (hrx_queue_family_info_t){
      .provisioned_queue_count = spec->provisioned_queue_count,
      .supports_acquisition = iree_any_bit_set(
          spec->flags, IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION),
      .roles = spec->role_flags,
      .placements = spec->placement_capabilities,
      .features = spec->supported_queue_features,
      .priority_count = spec->priority_count,
      .priorities = spec->priorities,
      .execution_unit_count = spec->execution_unit_count,
      .execution_resource_count = spec->execution_resource_count,
      .execution_resource_group_count = spec->execution_resource_group_count,
  };
  return hrx_ok_status();
}

hrx_status_t hrx_device_query_queue_execution_resource(
    hrx_device_t device, uint32_t family_ordinal, uint32_t resource_ordinal,
    hrx_queue_execution_resource_info_t* out_info) {
  const iree_hal_queue_family_spec_t* spec =
      hrx_device_queue_family_spec(device, family_ordinal);
  if (!spec || !out_info ||
      resource_ordinal >= spec->execution_resource_count) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid queue execution resource or NULL output");
  }
  const iree_hal_queue_execution_resource_spec_t* resource =
      &spec->execution_resources[resource_ordinal];
  *out_info = (hrx_queue_execution_resource_info_t){
      .group_ordinal = resource->group_ordinal,
      .first_execution_unit_ordinal = resource->first_execution_unit_ordinal,
      .execution_unit_count = resource->execution_unit_count,
  };
  return hrx_ok_status();
}

hrx_status_t hrx_device_query_queue_execution_resource_group(
    hrx_device_t device, uint32_t family_ordinal, uint32_t group_ordinal,
    hrx_queue_execution_resource_group_info_t* out_info) {
  const iree_hal_queue_family_spec_t* spec =
      hrx_device_queue_family_spec(device, family_ordinal);
  if (!spec || !out_info ||
      group_ordinal >= spec->execution_resource_group_count) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid queue resource group or NULL output");
  }
  *out_info = (hrx_queue_execution_resource_group_info_t){
      .minimum_selected_resource_count =
          spec->execution_resource_groups[group_ordinal]
              .minimum_selected_resource_count,
  };
  return hrx_ok_status();
}

hrx_queue_t hrx_device_get_queue(hrx_device_t device, uint32_t family_ordinal,
                                 uint32_t queue_ordinal) {
  const iree_hal_queue_family_spec_t* family =
      hrx_device_queue_family_spec(device, family_ordinal);
  if (!family || queue_ordinal >= family->provisioned_queue_count) {
    return NULL;
  }
  const iree_hal_device_queue_spec_t* spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device->hal_device));
  iree_host_size_t position = queue_ordinal;
  for (uint32_t i = 0; i < family_ordinal; ++i) {
    position += spec->families[i].provisioned_queue_count;
  }
  return &device->queues[position];
}

hrx_status_t hrx_queue_acquire(hrx_device_t device, uint32_t family_ordinal,
                               const hrx_queue_params_t* params,
                               hrx_queue_t* out_queue) {
  if (!out_queue) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "queue output is NULL");
  }
  *out_queue = NULL;
  const iree_hal_queue_family_t* family =
      device ? iree_hal_device_queue_family(device->hal_device, family_ordinal)
             : NULL;
  if (!family) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "invalid queue family");
  }
  iree_hal_queue_params_t hal_params;
  iree_hal_queue_params_initialize(&hal_params);
  if (params) {
    hal_params.priority = params->priority;
    hal_params.placement = (iree_hal_queue_placement_t)params->placement;
    hal_params.features = params->features;
    hal_params.execution_resources.count = params->execution_resources.count;
    hal_params.execution_resources.ordinals =
        params->execution_resources.ordinals;
  }
  hrx_queue_t queue = NULL;
  HRX_RETURN_IF_IREE_ERROR(iree_allocator_malloc(
      iree_allocator_system(), sizeof(*queue), (void**)&queue));
  iree_status_t status =
      iree_hal_queue_acquire(family, &hal_params, &queue->hal_queue);
  if (iree_status_is_ok(status)) {
    iree_atomic_ref_count_init(&queue->ref_count);
    queue->device = device;
    *out_queue = queue;
  } else {
    iree_allocator_free(iree_allocator_system(), queue);
  }
  return hrx_status_from_iree(status);
}

void hrx_queue_retain(hrx_queue_t queue) {
  if (queue) {
    iree_atomic_ref_count_inc(&queue->ref_count);
  }
}

void hrx_queue_release(hrx_queue_t queue) {
  if (queue && iree_atomic_ref_count_dec(&queue->ref_count) == 1) {
    // Provisioned queues keep their device-owned base reference. Only exact
    // acquisitions reach this path; their handle and native queue are owned.
    iree_hal_queue_release(queue->hal_queue);
    iree_allocator_free(iree_allocator_system(), queue);
  }
}

hrx_device_t hrx_queue_get_device(hrx_queue_t queue) {
  return queue ? queue->device : NULL;
}

iree_status_t hrx_queue_validate_roles(hrx_queue_t queue,
                                       hrx_queue_role_flags_t required_roles) {
  const iree_hal_queue_family_spec_t* spec =
      iree_hal_queue_family_spec(iree_hal_queue_family(queue->hal_queue));
  if (!iree_all_bits_set(spec->role_flags, required_roles)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected queue does not support this operation");
  }
  return iree_ok_status();
}

uint32_t hrx_queue_get_family_ordinal(hrx_queue_t queue) {
  return queue ? iree_hal_queue_family_ordinal(
                     iree_hal_queue_family(queue->hal_queue))
               : UINT32_MAX;
}

hrx_status_t hrx_queue_query_params(hrx_queue_t queue,
                                    hrx_queue_params_t* out_params) {
  if (!queue || !out_params) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "queue or params output is NULL");
  }
  const iree_hal_queue_execution_resource_list_t resources =
      iree_hal_queue_execution_resources(queue->hal_queue);
  *out_params = (hrx_queue_params_t){
      .priority = iree_hal_queue_priority(queue->hal_queue),
      .placement =
          (hrx_queue_placement_t)iree_hal_queue_placement(queue->hal_queue),
      .features = iree_hal_queue_features(queue->hal_queue),
      .execution_resources = {.count = resources.count,
                              .ordinals = resources.ordinals},
  };
  return hrx_ok_status();
}
