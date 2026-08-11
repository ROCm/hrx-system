// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/system_event.h"

#include "iree/base/threading/api.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"

typedef struct iree_hal_amdgpu_system_event_device_t {
  // Next live logical device registered for process-wide event delivery.
  struct iree_hal_amdgpu_system_event_device_t* next;
  // Logical device receiving events. Borrowed until explicit unregistration.
  iree_hal_amdgpu_logical_device_t* logical_device;
  // Allocator owning this registry entry.
  iree_allocator_t host_allocator;
  // Number of GPU agents represented by |agents|.
  iree_host_size_t agent_count;
  // True after callbacks must stop dereferencing |logical_device|.
  bool is_tearing_down;
  // Stable HSA agent handles retained through hardware queue teardown.
  hsa_agent_t agents[IREE_HAL_AMDGPU_MAX_GPU_AGENT];
} iree_hal_amdgpu_system_event_device_t;

typedef struct iree_hal_amdgpu_system_event_registry_t {
  // Serializes registration, removal, and callback traversal.
  iree_mutex_t mutex;
  // Head of the live logical device list.
  iree_hal_amdgpu_system_event_device_t* device_list;
  // Whether the process-wide callback has been registered with HSA.
  bool is_hsa_handler_registered;
} iree_hal_amdgpu_system_event_registry_t;

static iree_once_flag iree_hal_amdgpu_system_event_once = IREE_ONCE_FLAG_INIT;
static iree_hal_amdgpu_system_event_registry_t
    iree_hal_amdgpu_system_event_registry;

static void iree_hal_amdgpu_system_event_initialize(void) {
  iree_mutex_initialize(&iree_hal_amdgpu_system_event_registry.mutex);
}

static hsa_status_t iree_hal_amdgpu_system_event_callback(
    const hsa_amd_event_t* event, void* user_data) {
  (void)user_data;
  if (event->event_type != HSA_AMD_GPU_MEMORY_FAULT_EVENT) {
    return HSA_STATUS_ERROR;
  }

  const hsa_amd_gpu_memory_fault_info_t* fault = &event->memory_fault;
  bool handled = false;
  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  for (iree_hal_amdgpu_system_event_device_t* entry =
           iree_hal_amdgpu_system_event_registry.device_list;
       entry != NULL; entry = entry->next) {
    bool agent_matches = false;
    for (iree_host_size_t i = 0; i < entry->agent_count; ++i) {
      if (entry->agents[i].handle == fault->agent.handle) {
        agent_matches = true;
        break;
      }
    }
    if (!agent_matches) continue;

    handled = true;
    if (entry->is_tearing_down) continue;

    iree_hal_amdgpu_logical_device_t* logical_device = entry->logical_device;
    bool logical_device_faulted = false;
    for (iree_host_size_t i = 0; i < logical_device->physical_device_count;
         ++i) {
      iree_hal_amdgpu_physical_device_t* physical_device =
          logical_device->physical_devices[i];
      if (physical_device->device_agent.handle != fault->agent.handle) continue;

      logical_device_faulted = true;
      for (iree_host_size_t j = 0; j < physical_device->host_queue_count; ++j) {
        iree_hal_amdgpu_host_queue_fail(
            &physical_device->host_queues[j],
            iree_make_status(
                IREE_STATUS_ABORTED,
                "AMDGPU memory access fault at device address 0x%016" PRIx64
                " (reason mask 0x%08" PRIx32 ")",
                fault->virtual_address, fault->fault_reason_mask));
      }
    }
    if (logical_device_faulted) {
      iree_hal_amdgpu_logical_device_error_handler(
          logical_device,
          iree_make_status(
              IREE_STATUS_ABORTED,
              "AMDGPU memory access fault at device address 0x%016" PRIx64
              " (reason mask 0x%08" PRIx32 ")",
              fault->virtual_address, fault->fault_reason_mask));
    }
  }
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);
  return handled ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

iree_status_t iree_hal_amdgpu_system_event_register_device(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(logical_device);
  iree_call_once(&iree_hal_amdgpu_system_event_once,
                 iree_hal_amdgpu_system_event_initialize);

  iree_hal_amdgpu_system_event_device_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*entry), (void**)&entry));
  entry->next = NULL;
  entry->logical_device = logical_device;
  entry->host_allocator = host_allocator;
  entry->agent_count = logical_device->physical_device_count;
  entry->is_tearing_down = false;
  for (iree_host_size_t i = 0; i < entry->agent_count; ++i) {
    entry->agents[i] = logical_device->physical_devices[i]->device_agent;
  }

  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  iree_status_t status = iree_ok_status();
  if (!iree_hal_amdgpu_system_event_registry.is_hsa_handler_registered) {
    status = iree_hsa_amd_register_system_event_handler(
        IREE_LIBHSA(libhsa), iree_hal_amdgpu_system_event_callback,
        /*data=*/NULL);
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_system_event_registry.is_hsa_handler_registered = true;
    }
  }
  if (iree_status_is_ok(status)) {
    entry->next = iree_hal_amdgpu_system_event_registry.device_list;
    iree_hal_amdgpu_system_event_registry.device_list = entry;
  }
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, entry);
  }
  return status;
}

void iree_hal_amdgpu_system_event_begin_device_teardown(
    iree_hal_amdgpu_logical_device_t* logical_device) {
  IREE_ASSERT_ARGUMENT(logical_device);
  iree_call_once(&iree_hal_amdgpu_system_event_once,
                 iree_hal_amdgpu_system_event_initialize);

  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  for (iree_hal_amdgpu_system_event_device_t* entry =
           iree_hal_amdgpu_system_event_registry.device_list;
       entry != NULL; entry = entry->next) {
    if (entry->logical_device == logical_device) {
      entry->is_tearing_down = true;
      break;
    }
  }
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);
}

void iree_hal_amdgpu_system_event_unregister_device(
    iree_hal_amdgpu_logical_device_t* logical_device) {
  IREE_ASSERT_ARGUMENT(logical_device);
  iree_call_once(&iree_hal_amdgpu_system_event_once,
                 iree_hal_amdgpu_system_event_initialize);

  iree_hal_amdgpu_system_event_device_t* removed_entry = NULL;
  iree_mutex_lock(&iree_hal_amdgpu_system_event_registry.mutex);
  iree_hal_amdgpu_system_event_device_t** entry_ptr =
      &iree_hal_amdgpu_system_event_registry.device_list;
  while (*entry_ptr != NULL) {
    if ((*entry_ptr)->logical_device == logical_device) {
      removed_entry = *entry_ptr;
      *entry_ptr = removed_entry->next;
      break;
    }
    entry_ptr = &(*entry_ptr)->next;
  }
  iree_mutex_unlock(&iree_hal_amdgpu_system_event_registry.mutex);

  if (removed_entry) {
    iree_allocator_free(removed_entry->host_allocator, removed_entry);
  }
}
