// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/fabric.h"

#include "iree/hal/drivers/amd/status.h"

struct iree_hal_amd_fabric_allocation_entry_t {
  // Next cold allocation domain owned by the fabric.
  iree_hal_amd_fabric_allocation_entry_t* next;
  // NUMA placement, or UINT32_MAX for unspecified placement.
  uint32_t node_id;
  // Fabric-owned native allocation queue shared by all engines on this node.
  iree_hal_amd_fabric_allocation_queue_t* queue;
};

iree_status_t iree_hal_amd_fabric_acquire_allocation_queue(
    iree_hal_amd_fabric_t* fabric, uint32_t node_id,
    iree_hal_amd_fabric_allocation_queue_t** out_queue) {
  *out_queue = NULL;
  iree_slim_mutex_lock(&fabric->allocation_mutex);
  iree_hal_amd_fabric_allocation_entry_t* entry = fabric->allocations;
  while (entry && entry->node_id != node_id) {
    entry = entry->next;
  }
  iree_status_t status = iree_ok_status();
  if (!entry) {
    status = iree_allocator_malloc(fabric->host_allocator, sizeof(*entry),
                                   (void**)&entry);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_fabric_allocation_queue_create(
          node_id, fabric->host_allocator, &entry->queue);
      if (iree_status_is_ok(status)) {
        entry->node_id = node_id;
        entry->next = fabric->allocations;
        fabric->allocations = entry;
      } else {
        iree_allocator_free(fabric->host_allocator, entry);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amd_fabric_allocation_queue_retain(entry->queue);
    *out_queue = entry->queue;
  }
  iree_slim_mutex_unlock(&fabric->allocation_mutex);
  return status;
}

void iree_hal_amd_fabric_retain(iree_hal_amd_fabric_t* fabric) {
  if (fabric) {
    iree_atomic_ref_count_inc(&fabric->ref_count);
  }
}

void iree_hal_amd_fabric_release(iree_hal_amd_fabric_t* fabric) {
  if (!fabric) {
    return;
  }
  if (iree_atomic_ref_count_dec(&fabric->ref_count) != 1) {
    return;
  }
  while (fabric->allocations) {
    iree_hal_amd_fabric_allocation_entry_t* entry = fabric->allocations;
    fabric->allocations = entry->next;
    iree_hal_amd_fabric_allocation_queue_release(entry->queue);
    iree_allocator_free(fabric->host_allocator, entry);
  }
  if (fabric->instance) {
    const amdf_status_t status =
        fabric->api->instance_destroy(fabric->instance);
    if (!amdf_status_is_ok(status)) {
      iree_hal_amd_report_failure(iree_hal_device_event_sink_stderr(),
                                  IREE_SV("amd"), status, "instance_destroy");
      return;
    }
  }
  iree_slim_mutex_deinitialize(&fabric->allocation_mutex);
  iree_allocator_free(fabric->host_allocator, fabric->endpoints);
  iree_allocator_free(fabric->host_allocator, fabric);
}

static iree_status_t iree_hal_amd_fabric_discover(
    iree_hal_amd_fabric_t* fabric) {
  uint32_t count = 0;
  const amdf_status_t count_status =
      fabric->api->instance_enumerate_memory_scopes(fabric->instance, 0, NULL,
                                                    &count);
  if (count_status != amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)) {
    IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
        count_status, "instance_enumerate_memory_scopes(count)"));
  }
  iree_host_size_t size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size, IREE_STRUCT_FIELD_FAM(count, amdf_memory_scope_t*)));
  amdf_memory_scope_t** scopes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(fabric->host_allocator, size, (void**)&scopes));
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      fabric->api->instance_enumerate_memory_scopes(fabric->instance, count,
                                                    scopes, &count),
      "instance_enumerate_memory_scopes");
  for (uint32_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    amdf_memory_scope_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(info),
    };
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        fabric->api->memory_scope_query_info(scopes[i], &info),
        "memory_scope_query_info");
    if (iree_status_is_ok(status) &&
        info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
      fabric->system_scope = scopes[i];
      break;
    }
  }
  iree_allocator_free(fabric->host_allocator, scopes);
  IREE_RETURN_IF_ERROR(status);
  if (!fabric->system_scope) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "no system memory scope");
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      fabric->api->endpoint_enumerate(fabric->instance, 0, NULL, &count),
      "endpoint_enumerate(count)"));
  if (!count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &size, IREE_STRUCT_FIELD_FAM(count, amdf_endpoint_summary_t)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(fabric->host_allocator, size,
                                             (void**)&fabric->endpoints));
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      fabric->api->endpoint_enumerate(fabric->instance, count,
                                      fabric->endpoints, &count),
      "endpoint_enumerate"));
  fabric->endpoint_count = count;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_fabric_create(const amdf_api_t* api,
                                         amdf_native_lifetime_t native_lifetime,
                                         iree_allocator_t host_allocator,
                                         iree_hal_amd_fabric_t** out_fabric) {
  *out_fabric = NULL;
  iree_hal_amd_fabric_t* fabric = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*fabric), (void**)&fabric));
  iree_atomic_ref_count_init(&fabric->ref_count);
  fabric->host_allocator = host_allocator;
  fabric->api = api;
  iree_slim_mutex_initialize(&fabric->allocation_mutex);
  const amdf_instance_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .native_lifetime = native_lifetime,
  };
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->instance_create(&create_info, &fabric->instance), "instance_create");
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_discover(fabric);
  }
  if (iree_status_is_ok(status)) {
    *out_fabric = fabric;
  } else {
    iree_hal_amd_fabric_release(fabric);
  }
  return status;
}
