// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/fabric.h"

#include "iree/hal/driver/amd/status.h"

struct iree_hal_amd_fabric_allocation_entry_t {
  // Next cold allocation domain owned by the fabric.
  iree_hal_amd_fabric_allocation_entry_t* next;
  // Physical endpoint whose native requests share this progress domain.
  amdf_endpoint_id_t endpoint_id;
  // NUMA placement, or UINT32_MAX for unspecified placement.
  uint32_t node_id;
  // Live device acquisitions, protected by the fabric allocation mutex.
  iree_host_size_t acquisition_count;
  // Fabric-owned queue shared by logical contexts of this endpoint and node.
  iree_hal_amd_fabric_allocation_queue_t* queue;
};

iree_status_t iree_hal_amd_fabric_query_physical_device_identity(
    const amdf_api_t* api, amdf_endpoint_t* endpoint,
    iree_string_view_t display_name,
    iree_hal_physical_device_identity_t* out_identity) {
  amdf_endpoint_pci_address_t address = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_PCI_ADDRESS,
      .structure_size = sizeof(address),
  };
  amdf_endpoint_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(info),
      .next = &address,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      api->endpoint_query_info(endpoint, &info), "endpoint_query_info"));
  *out_identity = (iree_hal_physical_device_identity_t){
      .display_name = display_name,
      .vendor_id = info.pci.vendor_id,
      .device_id = info.pci.device_id,
      .revision_id = info.pci.revision_id,
  };
  if (address.available) {
    out_identity->pci = (iree_hal_pci_address_t){
        .domain = address.domain,
        .bus = address.bus,
        .device = address.device,
        .function = address.function,
    };
    out_identity->flags |= IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_PCI_ADDRESS;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amd_fabric_acquire_allocation_queue(
    iree_hal_amd_fabric_t* fabric, const amdf_endpoint_id_t* endpoint_id,
    uint32_t node_id, iree_hal_amd_fabric_allocation_queue_t** out_queue) {
  *out_queue = NULL;
  iree_slim_mutex_lock(&fabric->allocation_mutex);
  iree_hal_amd_fabric_allocation_entry_t* entry = fabric->allocations;
  while (entry &&
         (entry->node_id != node_id ||
          !amdf_endpoint_id_is_equal(&entry->endpoint_id, endpoint_id))) {
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
        entry->endpoint_id = *endpoint_id;
        entry->node_id = node_id;
        entry->acquisition_count = 0;
        entry->next = fabric->allocations;
        fabric->allocations = entry;
      } else {
        iree_allocator_free(fabric->host_allocator, entry);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    ++entry->acquisition_count;
    iree_hal_amd_fabric_allocation_queue_retain(entry->queue);
    *out_queue = entry->queue;
  }
  iree_slim_mutex_unlock(&fabric->allocation_mutex);
  return status;
}

void iree_hal_amd_fabric_release_allocation_queue(
    iree_hal_amd_fabric_t* fabric,
    iree_hal_amd_fabric_allocation_queue_t* queue) {
  if (!queue) {
    return;
  }
  iree_hal_amd_fabric_allocation_entry_t* retired = NULL;
  iree_slim_mutex_lock(&fabric->allocation_mutex);
  iree_hal_amd_fabric_allocation_entry_t** link = &fabric->allocations;
  while ((*link)->queue != queue) {
    link = &(*link)->next;
  }
  iree_hal_amd_fabric_allocation_entry_t* entry = *link;
  if (--entry->acquisition_count == 0) {
    *link = entry->next;
    retired = entry;
  }
  iree_slim_mutex_unlock(&fabric->allocation_mutex);
  // Joining the final worker must not hold the cache lock and serialize new
  // device acquisitions behind its thread exit.
  if (retired) {
    iree_hal_amd_fabric_allocation_queue_release(retired->queue);
    iree_allocator_free(fabric->host_allocator, retired);
  }
  iree_hal_amd_fabric_allocation_queue_release(queue);
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
  IREE_ASSERT(!fabric->allocations);
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

iree_status_t iree_hal_amd_fabric_create(
    const amdf_api_t* api, amdf_native_lifetime_t native_lifetime,
    uint32_t rocjitsu_machine_count,
    const amdf_rocjitsu_machine_create_info_t* rocjitsu_machines,
    iree_allocator_t host_allocator, iree_hal_amd_fabric_t** out_fabric) {
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
  if (iree_status_is_ok(status) && rocjitsu_machine_count != 0) {
    const void* extension = NULL;
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        api->query_extension(
            AMDF_EXTENSION_ROCJITSU, AMDF_ROCJITSU_EXTENSION_VERSION_1,
            AMDF_ROCJITSU_EXTENSION_VERSION_LATEST, &extension),
        "query_extension(ROCJITSU)");
    const amdf_rocjitsu_api_t* rocjitsu = extension;
    for (uint32_t i = 0;
         iree_status_is_ok(status) && i < rocjitsu_machine_count; ++i) {
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          rocjitsu->instance_add_machine(fabric->instance,
                                         &rocjitsu_machines[i]),
          "rocjitsu.instance_add_machine");
    }
  }
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
