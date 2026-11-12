// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Device operations. Generic across accelerator types once you have a handle.

#include "device.h"

#include <stdint.h>
#include <string.h>

#include "device_group.h"
#include "hrx_internal.h"

IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_SITE_HOST, IREE_HAL_MEMORY_SITE_HOST,
                           "memory sites must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_SITE_QUEUE, IREE_HAL_MEMORY_SITE_QUEUE,
                           "memory sites must match HAL");
IREE_STATIC_ASSERT_ENUM_EQ(HRX_MEMORY_SITE_PROGRAM,
                           IREE_HAL_MEMORY_SITE_PROGRAM,
                           "memory sites must match HAL");

iree_status_t hrx_device_initialize(hrx_device_group_t group, size_t index,
                                    hrx_accelerator_type_t type,
                                    hrx_device_t device) {
  device->group = group;
  device->type = type;
  device->ordinal = (int)index;
  device->hal_device_group = group->hal_group;
  device->hal_device = iree_hal_device_group_device_at(group->hal_group, index);
  device->allocator.device = device;
  device->allocator.hal_allocator =
      iree_hal_device_allocator(device->hal_device);
  iree_hal_allocator_retain(device->allocator.hal_allocator);
  hrx_mem_pool_backing_initialize(&device->mem_pool_backing);
  hrx_event_timestamp_pool_initialize(NULL, group->runtime->host_allocator,
                                      &device->event_timestamps);
  hrx_buffer_table_initialize(&device->buffer_table);
  iree_arena_block_pool_initialize(32 * 1024, group->runtime->host_allocator,
                                   &device->block_pool);

  const iree_hal_device_spec_t* spec = iree_hal_device_spec(device->hal_device);
  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(spec);
  iree_string_view_t name = identity->display_name;
  if (iree_string_view_is_empty(name)) {
    name = identity->logical_device_id;
  }
  const size_t name_length = iree_min(name.size, sizeof(device->name) - 1);
  memcpy(device->name, name.data, name_length);
  device->name[name_length] = 0;

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(spec);
  for (uint32_t i = 0; i < queues->family_count; ++i) {
    if (queues->families[i].provisioned_queue_count &&
        iree_any_bit_set(queues->families[i].role_flags,
                         IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH)) {
      device->dispatch_queue = iree_hal_device_queue(device->hal_device, i, 0);
      break;
    }
  }
  if (!device->dispatch_queue) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "device has no provisioned dispatch queue");
  }
  if (type == HRX_ACCELERATOR_CPU) {
    memcpy(device->architecture, "host", sizeof("host"));
  } else {
    const uint32_t family = iree_hal_queue_family_ordinal(
        iree_hal_queue_family(device->dispatch_queue));
    const iree_hal_executable_target_selection_t selection = {
        .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
        .physical_device_affinity =
            queues->families[family].physical_device_affinity,
    };
    const iree_hal_executable_target_selection_result_t selected =
        iree_hal_device_spec_select_executable_target(spec, &selection);
    if (selected.outcome !=
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "device has no unique exact executable target");
    }
    const iree_string_view_t key = selected.target->target_key;
    if (key.size >= sizeof(device->architecture)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable target key exceeds device storage");
    }
    memcpy(device->architecture, key.data, key.size);
    device->architecture[key.size] = 0;
  }
  return hrx_device_initialize_queues(device);
}

void hrx_device_deinitialize(hrx_device_t device) {
  hrx_device_report_failure(device, hrx_device_profile_end(device));
  hrx_device_deinitialize_queues(device);
  hrx_buffer_table_deinitialize(&device->buffer_table);
  iree_arena_block_pool_deinitialize(&device->block_pool);
  hrx_event_timestamp_pool_deinitialize(&device->event_timestamps);
  hrx_mem_pool_backing_deinitialize(&device->mem_pool_backing);
  iree_hal_allocator_release(device->allocator.hal_allocator);
}

void hrx_device_report_failure(hrx_device_t device, iree_status_t status) {
  hrx_runtime_report_failure(device->group->runtime, device->hal_device,
                             status);
}

iree_hal_device_t* hrx_device_hal(hrx_device_t device) {
  return device ? device->hal_device : NULL;
}

iree_hal_command_buffer_mode_t hrx_device_profile_command_buffer_mode(
    hrx_device_t device) {
  return device->profiling_active
             ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
             : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
}

hrx_status_t hrx_device_query_total_memory_from_spec(
    hrx_device_t device, bool* out_known, iree_device_size_t* out_total) {
  if (!device || !out_known || !out_total) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_known = false;
  *out_total = 0;

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(iree_hal_device_spec(device->hal_device));
  const iree_hal_memory_type_t locality =
      device->type == HRX_ACCELERATOR_CPU ? IREE_HAL_MEMORY_TYPE_HOST_LOCAL
                                          : IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  iree_device_size_t total = 0;
  bool known = false;
  for (iree_host_size_t h = 0; h < memory->heap_count; ++h) {
    const iree_hal_memory_heap_spec_t* heap = &memory->heaps[h];
    if (iree_any_bit_set(heap->flags,
                         IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN)) {
      continue;
    }
    // Allocation profiles can expose several types backed by the same heap.
    // HRX reports local physical capacity once, excluding staging-only heaps.
    bool local = false;
    for (iree_host_size_t t = 0; t < memory->memory_type_count && !local; ++t) {
      local = memory->memory_types[t].heap_index == h &&
              iree_all_bits_set(memory->memory_types[t].memory_type, locality);
    }
    if (!local) {
      continue;
    }
    if (heap->capacity_bytes > IREE_DEVICE_SIZE_MAX ||
        !iree_device_size_checked_add(total, heap->capacity_bytes, &total)) {
      return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                             "local memory capacity exceeds device size");
    }
    known = true;
  }
  *out_known = known;
  *out_total = total;
  return hrx_ok_status();
}

static hrx_status_t hrx_device_sample_memory(
    hrx_device_t device, iree_device_size_t* out_total,
    iree_device_size_t* out_available) {
  iree_hal_device_observation_t observation;
  iree_status_t status = iree_hal_device_sample_observation(
      device->hal_device, IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY,
      &observation);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  if (!iree_all_bits_set(observation.provided_flags,
                         IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY)) {
    return hrx_make_status(HRX_STATUS_UNAVAILABLE,
                           "HAL device did not provide a memory observation");
  }
  if (!iree_all_bits_set(observation.memory.flags,
                         IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES)) {
    return hrx_make_status(
        HRX_STATUS_UNAVAILABLE,
        "HAL device did not provide total memory in its observation");
  }
  if (out_available &&
      !iree_all_bits_set(
          observation.memory.flags,
          IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_AVAILABLE_BYTES)) {
    return hrx_make_status(
        HRX_STATUS_UNAVAILABLE,
        "HAL device did not provide available memory in its observation");
  }
  *out_total = observation.memory.total_bytes;
  if (out_available) {
    *out_available = observation.memory.available_bytes;
  }
  return hrx_ok_status();
}

hrx_status_t hrx_device_get_property(hrx_device_t device,
                                     hrx_device_property_t prop, void* value,
                                     size_t value_size) {
  if (!device || !value) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or value is NULL");
  }
  switch (prop) {
    case HRX_DEVICE_PROPERTY_NAME: {
      size_t len = strlen(device->name);
      if (value_size < len + 1) {
        return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                               "buffer too small for device name");
      }
      memcpy(value, device->name, len + 1);
      return hrx_ok_status();
    }
    case HRX_DEVICE_PROPERTY_ARCHITECTURE: {
      size_t len = strlen(device->architecture);
      if (value_size < len + 1) {
        return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                               "buffer too small for architecture string");
      }
      memcpy(value, device->architecture, len + 1);
      return hrx_ok_status();
    }
    case HRX_DEVICE_PROPERTY_TOTAL_MEMORY: {
      if (value_size < sizeof(uint64_t)) {
        return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                               "buffer too small for uint64_t");
      }
      iree_device_size_t total_bytes = 0;
      bool total_memory_known = false;
      hrx_status_t status = hrx_device_query_total_memory_from_spec(
          device, &total_memory_known, &total_bytes);
      if (!hrx_status_is_ok(status)) {
        return status;
      }
      if (!total_memory_known) {
        return hrx_make_status(
            HRX_STATUS_UNAVAILABLE,
            "HAL device spec did not provide a known total memory capacity");
      }
      *(uint64_t*)value = (uint64_t)total_bytes;
      return status;
    }
    case HRX_DEVICE_PROPERTY_COMPUTE_UNITS:
    case HRX_DEVICE_PROPERTY_MAX_WORKGROUP_SIZE: {
      if (value_size < sizeof(uint32_t)) {
        return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                               "buffer too small for uint32_t");
      }
      const iree_hal_device_dispatch_spec_t* dispatch =
          iree_hal_device_spec_dispatch(
              iree_hal_device_spec(device->hal_device));
      *(uint32_t*)value = prop == HRX_DEVICE_PROPERTY_COMPUTE_UNITS
                              ? dispatch->execution.unit_count
                              : dispatch->launch.maximum_workgroup_invocations;
      return hrx_ok_status();
    }
    default:
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "unknown device property");
  }
}

hrx_status_t hrx_device_get_type(hrx_device_t device,
                                 hrx_accelerator_type_t* type) {
  if (!device || !type) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or type is NULL");
  }
  *type = device->type;
  return hrx_ok_status();
}

hrx_device_group_t hrx_device_get_device_group(hrx_device_t device) {
  return device ? device->group : NULL;
}

void hrx_device_retain(hrx_device_t device) {
  if (!device) {
    return;
  }
  hrx_device_group_retain(device->group);
}

void hrx_device_release(hrx_device_t device) {
  if (!device) {
    return;
  }
  hrx_device_group_release(device->group);
}

hrx_status_t hrx_device_memory_info(hrx_device_t device, size_t* free_bytes,
                                    size_t* total_bytes) {
  if (!device || !free_bytes || !total_bytes) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  iree_device_size_t total = 0;
  iree_device_size_t available = 0;
  hrx_status_t status = hrx_device_sample_memory(device, &total, &available);
  if (!hrx_status_is_ok(status)) {
    return status;
  }
  if (total > SIZE_MAX || available > SIZE_MAX) {
    return hrx_make_status(
        HRX_STATUS_OUT_OF_RANGE,
        "HAL memory observation exceeds the representable size_t range");
  }
  *total_bytes = (size_t)total;
  *free_bytes = (size_t)available;
  return status;
}

hrx_status_t hrx_device_can_access_peer(hrx_device_t device_a,
                                        hrx_device_t device_b,
                                        bool* can_access) {
  if (!device_a || !device_b || !can_access) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  if (device_a == device_b) {
    *can_access = true;
    return hrx_ok_status();
  }
  *can_access = false;
  if (device_a->type != HRX_ACCELERATOR_GPU ||
      device_b->type != HRX_ACCELERATOR_GPU) {
    return hrx_ok_status();
  }

  const iree_hal_device_topology_info_t* topology_info_a =
      iree_hal_device_topology_info(device_a->hal_device);
  const iree_hal_device_topology_info_t* topology_info_b =
      iree_hal_device_topology_info(device_b->hal_device);
  if (!topology_info_a->topology ||
      topology_info_a->topology != topology_info_b->topology) {
    return hrx_ok_status();
  }
  const iree_hal_topology_edge_t edge =
      iree_hal_device_topology_query_edge(topology_info_a, topology_info_b);
  *can_access = hrx_topology_edge_supports_peer_access(edge);
  return hrx_ok_status();
}

hrx_status_t hrx_device_query_host_memory_scope(hrx_device_t device,
                                                hrx_memory_scope_t* out_scope) {
  if (!device || !out_scope) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "memory scope device or output is NULL");
  }
  *out_scope = (hrx_memory_scope_t){0};
  iree_hal_memory_scope_t hal_scope;
  iree_status_t status = iree_hal_device_group_resolve_memory_scope(
      device->hal_device_group,
      (iree_hal_memory_site_t){.kind = IREE_HAL_MEMORY_SITE_HOST}, &hal_scope);
  if (iree_status_is_ok(status)) {
    *out_scope = (hrx_memory_scope_t){hal_scope.domain, hal_scope.id};
  }
  return hrx_status_from_iree(status);
}

hrx_status_t hrx_device_query_queue_family_memory_scope(
    hrx_device_t device, uint32_t family_ordinal, hrx_memory_site_t site,
    hrx_memory_scope_t* out_scope) {
  if (!device || !out_scope ||
      (site != HRX_MEMORY_SITE_QUEUE && site != HRX_MEMORY_SITE_PROGRAM)) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid memory scope device, site or output");
  }
  *out_scope = (hrx_memory_scope_t){0};
  const iree_hal_queue_family_t* family =
      iree_hal_device_queue_family(device->hal_device, family_ordinal);
  if (!family) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "invalid memory scope queue family");
  }
  const iree_hal_memory_site_t hal_site = {
      .kind = (iree_hal_memory_site_kind_t)site,
      .family = family,
  };
  iree_hal_memory_scope_t hal_scope;
  iree_status_t status = iree_hal_device_group_resolve_memory_scope(
      device->hal_device_group, hal_site, &hal_scope);
  if (iree_status_is_ok(status)) {
    *out_scope = (hrx_memory_scope_t){hal_scope.domain, hal_scope.id};
  }
  return hrx_status_from_iree(status);
}
