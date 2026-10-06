// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/device.h"

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/amd/xdna/executable.h"
#include "iree/hal/drivers/amd/xdna/queue.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/hal/utils/host_semaphore.h"

typedef struct iree_hal_amd_xdna_device_t {
  // HAL resource header.
  iree_hal_resource_t resource;
  // Allocator owning this wrapper and its children.
  iree_allocator_t host_allocator;
  // Native owner, transferred only after complete construction.
  iree_hal_amd_xdna_context_t* context;
  // Retained shared proactor runner entry.
  iree_async_proactor_pool_entry_t* proactor_entry;
  // Cached immutable device facts.
  iree_hal_device_spec_t* spec;
  // Device-owned canonical family identity.
  iree_hal_queue_family_t family;
  // Owned provisioned execution queue.
  iree_hal_queue_t* queue;
  // Owned facade for prepared ordinary native allocations.
  iree_hal_allocator_t* allocator;
  // Group-assigned topology; the queue retains its frontier tracker.
  iree_hal_device_topology_info_t topology;
} iree_hal_amd_xdna_device_t;

static const iree_hal_device_vtable_t iree_hal_amd_xdna_device_vtable;

static iree_status_t iree_hal_amd_xdna_device_build_spec(
    iree_hal_amd_xdna_context_t* context, iree_string_view_t display_name,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  const iree_hal_physical_device_spec_t physical = {
      .identity = {.display_name = display_name, .vendor_id = 0x1022},
      .partition_count = 1,
      .physical_device_affinity = 1,
  };
  const iree_hal_device_identity_spec_t identity = {
      .logical_device_id = IREE_SV("xdna"),
      .display_name = display_name,
      .driver_id = IREE_SV("xdna"),
      .backend_id = IREE_SV("libamdf"),
      .vendor_name = IREE_SV("AMD"),
      .vendor_id = 0x1022,
      .physical_device_count = 1,
      .physical_devices = &physical,
  };
  const iree_hal_queue_priority_t priority = IREE_HAL_QUEUE_PRIORITY_NORMAL;
  const iree_hal_queue_family_spec_t family = {
      .name = IREE_SV("xdna"),
      .provisioned_queue_count = 1,
      .priority_count = 1,
      .priorities = &priority,
      .physical_device_affinity = 1,
      .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER,
  };
  const iree_hal_device_queue_spec_t queues = {
      .family_count = 1,
      .families = &family,
  };
  const iree_hal_device_dispatch_spec_t dispatch = {
      .launch =
          {
              .maximum_workgroup_invocations = 1,
              .maximum_workgroup_size = {1, 1, 1},
              .maximum_workgroup_count = {1, 1, 1},
          },
  };
  const iree_hal_executable_target_t target = {
      .family = IREE_SV("xdna"),
      .target_key = iree_make_cstring_view(context->endpoint_info.target_id),
      .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
      .physical_device_affinity = 1,
  };
  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_hal_device_spec_builder_set_identity(&builder, &identity);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_queues(&builder, &queues);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_device_spec_builder_add_executable_target(&builder, &target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}

static void iree_hal_amd_xdna_device_destroy(iree_hal_device_t* base) {
  iree_hal_amd_xdna_device_t* device = (iree_hal_amd_xdna_device_t*)base;
  if (device->queue &&
      iree_atomic_ref_count_load(&device->queue->resource.ref_count) != 1) {
    iree_hal_amd_xdna_context_report(
        device->context, amdf_make_api_status(AMDF_STATUS_CODE_BUSY),
        "XDNA device released with live queue users; retaining native "
        "ownership");
    return;
  }
  iree_hal_queue_release(device->queue);
  iree_hal_allocator_release(device->allocator);
  iree_hal_device_spec_release(device->spec);
  iree_hal_amd_xdna_context_destroy(device->context);
  iree_async_proactor_pool_entry_release(device->proactor_entry);
  iree_allocator_free(device->host_allocator, device);
}

iree_status_t iree_hal_amd_xdna_device_create(
    iree_hal_amd_xdna_context_t* context, iree_string_view_t display_name,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  *out_device = NULL;
  if (create_params->runtime_features || create_params->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "XDNA device has no optional runtime services");
  }
  iree_hal_amd_xdna_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*device), (void**)&device));
  iree_hal_resource_initialize(&iree_hal_amd_xdna_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  iree_status_t status = iree_hal_amd_xdna_device_build_spec(
      context, display_name, host_allocator, &device->spec);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_acquire_for_node(
        create_params->proactor_pool, UINT32_MAX, &device->proactor_entry);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amd_xdna_allocator_create((iree_hal_device_t*)device, context,
                                           host_allocator, &device->allocator);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_queue_family_initialize(
        (iree_hal_device_t*)device, 0,
        &iree_hal_device_spec_queues(device->spec)->families[0],
        &device->family);
    status = iree_hal_amd_xdna_queue_create(
        &device->family, context,
        iree_async_proactor_pool_entry_proactor(device->proactor_entry),
        host_allocator, &device->queue);
  }
  if (iree_status_is_ok(status)) {
    device->context = context;
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

static iree_string_view_t iree_hal_amd_xdna_device_id(
    iree_hal_device_t* device) {
  return IREE_SV("xdna");
}

static iree_allocator_t iree_hal_amd_xdna_device_host_allocator(
    iree_hal_device_t* device) {
  return ((iree_hal_amd_xdna_device_t*)device)->host_allocator;
}

static iree_hal_allocator_t* iree_hal_amd_xdna_device_device_allocator(
    iree_hal_device_t* device) {
  return ((iree_hal_amd_xdna_device_t*)device)->allocator;
}

static void iree_hal_amd_xdna_device_replace_channel_provider(
    iree_hal_device_t* device, iree_hal_channel_provider_t* provider) {}

static iree_status_t iree_hal_amd_xdna_device_trim(iree_hal_device_t* device) {
  return iree_ok_status();
}

static const iree_hal_device_spec_t* iree_hal_amd_xdna_device_device_spec(
    iree_hal_device_t* device) {
  return ((iree_hal_amd_xdna_device_t*)device)->spec;
}

static const iree_hal_queue_family_t* iree_hal_amd_xdna_device_queue_family(
    iree_hal_device_t* device, iree_hal_queue_family_ordinal_t ordinal) {
  return ordinal == 0 ? &((iree_hal_amd_xdna_device_t*)device)->family : NULL;
}

static iree_hal_queue_t* iree_hal_amd_xdna_device_queue(
    iree_hal_device_t* device, iree_hal_queue_family_ordinal_t family,
    iree_hal_queue_ordinal_t ordinal) {
  return family == 0 && ordinal == 0
             ? ((iree_hal_amd_xdna_device_t*)device)->queue
             : NULL;
}

static const iree_hal_device_topology_info_t*
iree_hal_amd_xdna_device_topology_info(iree_hal_device_t* device) {
  return &((iree_hal_amd_xdna_device_t*)device)->topology;
}

static iree_status_t iree_hal_amd_xdna_device_refine_topology_edge(
    iree_hal_device_t* source, iree_hal_device_t* target,
    iree_hal_topology_edge_t* edge) {
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_device_assign_topology_info(
    iree_hal_device_t* base, const iree_hal_device_topology_info_t* topology) {
  iree_hal_amd_xdna_device_t* device = (iree_hal_amd_xdna_device_t*)base;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_assign_frontier(
      device->queue, topology ? topology->frontier.tracker : NULL,
      topology ? topology->frontier.base_axis : 0));
  device->topology =
      topology ? *topology : (iree_hal_device_topology_info_t){0};
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_device_load_executable(
    iree_hal_device_t* base, const iree_hal_queue_family_t* family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* params,
    iree_hal_executable_t** out_executable) {
  iree_hal_amd_xdna_device_t* device = (iree_hal_amd_xdna_device_t*)base;
  return iree_hal_amd_xdna_executable_create(
      family, device->context, params, device->host_allocator, out_executable);
}

static iree_status_t iree_hal_amd_xdna_device_create_semaphore(
    iree_hal_device_t* base, iree_hal_queue_family_affinity_t affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_amd_xdna_device_t* device = (iree_hal_amd_xdna_device_t*)base;
  return iree_hal_host_semaphore_create(
      iree_async_proactor_pool_entry_proactor(device->proactor_entry),
      initial_value, device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_amd_xdna_device_query_semaphore_compatibility(
    iree_hal_device_t* device, iree_hal_semaphore_t* semaphore) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_ALL;
}

static iree_status_t iree_hal_amd_xdna_device_acquire_queue(
    iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
    const iree_hal_queue_params_t* params, iree_hal_queue_t** out_queue) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA device does not support acquire_queue");
}

static iree_status_t iree_hal_amd_xdna_device_sample_observation(
    iree_hal_device_t* device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA device does not support sample_observation");
}

static iree_status_t iree_hal_amd_xdna_device_create_channel(
    iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA device does not support create_channel");
}

static iree_status_t iree_hal_amd_xdna_device_create_command_buffer(
    iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA device does not support create_command_buffer");
}

static iree_status_t iree_hal_amd_xdna_device_import_file(
    iree_hal_device_t* device,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA device does not support import_file");
}

static iree_status_t iree_hal_amd_xdna_device_query_queue_pool_backend(
    iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
    iree_hal_queue_pool_backend_t* out_backend) {
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "XDNA device does not support query_queue_pool_backend");
}

static iree_status_t iree_hal_amd_xdna_device_profiling_begin(
    iree_hal_device_t* device,
    const iree_hal_device_profiling_options_t* options) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA device does not support profiling_begin");
}

static iree_status_t iree_hal_amd_xdna_device_profiling_flush(
    iree_hal_device_t* device) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA device does not support profiling_flush");
}

static iree_status_t iree_hal_amd_xdna_device_profiling_end(
    iree_hal_device_t* device) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "XDNA device does not support profiling_end");
}

static const iree_hal_device_vtable_t iree_hal_amd_xdna_device_vtable = {
    .destroy = iree_hal_amd_xdna_device_destroy,
    .id = iree_hal_amd_xdna_device_id,
    .host_allocator = iree_hal_amd_xdna_device_host_allocator,
    .device_allocator = iree_hal_amd_xdna_device_device_allocator,
    .replace_channel_provider =
        iree_hal_amd_xdna_device_replace_channel_provider,
    .trim = iree_hal_amd_xdna_device_trim,
    .device_spec = iree_hal_amd_xdna_device_device_spec,
    .queue_family = iree_hal_amd_xdna_device_queue_family,
    .queue = iree_hal_amd_xdna_device_queue,
    .acquire_queue = iree_hal_amd_xdna_device_acquire_queue,
    .sample_observation = iree_hal_amd_xdna_device_sample_observation,
    .topology_info = iree_hal_amd_xdna_device_topology_info,
    .refine_topology_edge = iree_hal_amd_xdna_device_refine_topology_edge,
    .assign_topology_info = iree_hal_amd_xdna_device_assign_topology_info,
    .create_channel = iree_hal_amd_xdna_device_create_channel,
    .create_command_buffer = iree_hal_amd_xdna_device_create_command_buffer,
    .load_executable = iree_hal_amd_xdna_device_load_executable,
    .import_file = iree_hal_amd_xdna_device_import_file,
    .create_semaphore = iree_hal_amd_xdna_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_amd_xdna_device_query_semaphore_compatibility,
    .query_queue_pool_backend =
        iree_hal_amd_xdna_device_query_queue_pool_backend,
    .profiling_begin = iree_hal_amd_xdna_device_profiling_begin,
    .profiling_flush = iree_hal_amd_xdna_device_profiling_flush,
    .profiling_end = iree_hal_amd_xdna_device_profiling_end,
};
