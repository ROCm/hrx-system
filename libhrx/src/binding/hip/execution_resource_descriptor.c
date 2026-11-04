// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/execution_resource_descriptor.h"

#include "binding/hip/execution_resource.h"
#include "binding/hip/handle_registry.h"
#include "common/device.h"
#include "iree/base/threading/call_once.h"

static iree_once_flag iree_hip_execution_resource_descriptor_registry_once =
    IREE_ONCE_FLAG_INIT;
static iree_hip_handle_registry_t
    iree_hip_execution_resource_descriptor_registry;

static void iree_hip_execution_resource_descriptor_registry_initialize(void) {
  iree_hip_handle_registry_initialize(
      &iree_hip_execution_resource_descriptor_registry);
}

void iree_hip_execution_resource_descriptor_trim(void) {
  iree_call_once(&iree_hip_execution_resource_descriptor_registry_once,
                 iree_hip_execution_resource_descriptor_registry_initialize);
  iree_hip_handle_registry_trim(
      &iree_hip_execution_resource_descriptor_registry);
}

static hipError_t iree_hip_execution_resource_descriptor_consume_status(
    iree_status_t status) {
  const iree_status_code_t status_code = iree_status_code(status);
  iree_status_free(status);
  switch (status_code) {
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidResourceConfiguration;
    default:
      return hipErrorUnknown;
  }
}

static void iree_hip_execution_resource_descriptor_handle_retain(
    uintptr_t handle) {
  iree_hip_execution_resource_descriptor_t* descriptor =
      (iree_hip_execution_resource_descriptor_t*)handle;
  iree_atomic_ref_count_inc(&descriptor->ref_count);
}

hipError_t iree_hip_execution_resource_descriptor_create(
    iree_hal_streaming_device_t* device, const hipDevResource* resources,
    iree_host_size_t resource_count, hipDevResourceDesc_t* out_descriptor) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(resources);
  IREE_ASSERT_ARGUMENT(resource_count);
  IREE_ASSERT_ARGUMENT(out_descriptor);

  const iree_hal_streaming_execution_resource_set_t* first_set = NULL;
  hipError_t result = iree_hip_execution_resource_resolve_sm_for_device(
      &resources[0], device, &first_set);
  if (result != hipSuccess) {
    return result;
  }

  const iree_hal_queue_family_t* queue_family = iree_hal_device_queue_family(
      device->hal_device, first_set->queue_family_ordinal);
  if (!queue_family) {
    return hipErrorInvalidResourceConfiguration;
  }
  const iree_hal_queue_family_spec_t* family_spec =
      iree_hal_queue_family_spec(queue_family);
  // Descriptors can outlive the device-table incarnation they reference. Use
  // the process allocator so stale descriptors remain safely destructible.
  const iree_allocator_t host_allocator = iree_allocator_system();

  uint8_t* selected_resources = NULL;
  iree_hal_queue_execution_resource_ordinal_t* union_ordinals = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, family_spec->execution_resource_count,
      sizeof(*selected_resources), (void**)&selected_resources);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, family_spec->execution_resource_count,
        sizeof(*union_ordinals), (void**)&union_ordinals);
  }
  if (!iree_status_is_ok(status)) {
    result = iree_hip_execution_resource_descriptor_consume_status(status);
  }
  iree_host_size_t union_ordinal_count = 0;
  for (iree_host_size_t resource_index = 0;
       resource_index < resource_count && result == hipSuccess;
       ++resource_index) {
    const iree_hal_streaming_execution_resource_set_t* set =
        resource_index == 0 ? first_set : NULL;
    if (!set) {
      result = iree_hip_execution_resource_resolve_sm_for_device(
          &resources[resource_index], device, &set);
    }
    if (result != hipSuccess) {
      break;
    }
    if (set->queue_family_ordinal != first_set->queue_family_ordinal) {
      result = hipErrorInvalidResourceConfiguration;
      break;
    }

    for (iree_host_size_t ordinal_index = 0;
         ordinal_index < set->resources.count; ++ordinal_index) {
      const iree_hal_queue_execution_resource_ordinal_t ordinal =
          set->resources.ordinals[ordinal_index];
      if (selected_resources[ordinal]) {
        result = hipErrorInvalidResourceConfiguration;
        break;
      }
      selected_resources[ordinal] = 1;
      ++union_ordinal_count;
    }
  }

  if (result == hipSuccess) {
    iree_host_size_t ordinal_index = 0;
    for (iree_host_size_t i = 0; i < family_spec->execution_resource_count;
         ++i) {
      if (selected_resources[i]) {
        union_ordinals[ordinal_index++] =
            (iree_hal_queue_execution_resource_ordinal_t)i;
      }
    }
    if (ordinal_index != union_ordinal_count) {
      result = hipErrorInvalidResourceConfiguration;
    }
  }

  iree_hal_streaming_execution_resource_set_id_t union_set_id =
      IREE_HAL_STREAMING_EXECUTION_RESOURCE_SET_ID_INVALID;
  if (result == hipSuccess) {
    status = iree_hal_streaming_execution_resource_table_intern(
        &device->execution_resource_table, queue_family,
        (iree_hal_queue_execution_resource_list_t){
            .count = union_ordinal_count,
            .ordinals = union_ordinals,
        },
        &union_set_id);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_resource_descriptor_consume_status(status);
    }
  }

  iree_hip_execution_resource_descriptor_t* descriptor = NULL;
  if (result == hipSuccess) {
    status = iree_allocator_malloc_struct_array(
        host_allocator, sizeof(*descriptor), resource_count,
        sizeof(*descriptor->resources), (void**)&descriptor);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_resource_descriptor_consume_status(status);
    }
  }
  if (result == hipSuccess) {
    iree_atomic_ref_count_init(&descriptor->ref_count);
    descriptor->host_allocator = host_allocator;
    descriptor->device_ordinal = device->ordinal;
    descriptor->table_incarnation =
        iree_hal_streaming_execution_resource_table_incarnation(
            &device->execution_resource_table);
    descriptor->queue_family_ordinal = first_set->queue_family_ordinal;
    descriptor->sm_resource_set_id = union_set_id;
    descriptor->resource_count = resource_count;
    for (iree_host_size_t i = 0; i < resource_count; ++i) {
      descriptor->resources[i] = resources[i];
      descriptor->resources[i].nextResource = NULL;
    }

    iree_call_once(&iree_hip_execution_resource_descriptor_registry_once,
                   iree_hip_execution_resource_descriptor_registry_initialize);
    status = iree_hip_handle_registry_insert(
        &iree_hip_execution_resource_descriptor_registry,
        (uintptr_t)descriptor);
    if (!iree_status_is_ok(status)) {
      result = iree_hip_execution_resource_descriptor_consume_status(status);
    }
  }

  if (result == hipSuccess) {
    *out_descriptor = (hipDevResourceDesc_t)descriptor;
    descriptor = NULL;
  }
  iree_hip_execution_resource_descriptor_release(descriptor);
  iree_allocator_free(host_allocator, union_ordinals);
  iree_allocator_free(host_allocator, selected_resources);
  return result;
}

bool iree_hip_execution_resource_descriptor_lookup_retain(
    hipDevResourceDesc_t handle,
    iree_hip_execution_resource_descriptor_t** out_descriptor) {
  IREE_ASSERT_ARGUMENT(out_descriptor);
  if (!handle) {
    return false;
  }

  iree_call_once(&iree_hip_execution_resource_descriptor_registry_once,
                 iree_hip_execution_resource_descriptor_registry_initialize);
  if (!iree_hip_handle_registry_lookup_retain(
          &iree_hip_execution_resource_descriptor_registry, (uintptr_t)handle,
          iree_hip_execution_resource_descriptor_handle_retain)) {
    return false;
  }
  *out_descriptor = (iree_hip_execution_resource_descriptor_t*)handle;
  return true;
}

bool iree_hip_execution_resource_descriptor_take(
    hipDevResourceDesc_t handle,
    iree_hip_execution_resource_descriptor_t** out_descriptor) {
  IREE_ASSERT_ARGUMENT(out_descriptor);
  if (!handle) {
    return false;
  }

  iree_call_once(&iree_hip_execution_resource_descriptor_registry_once,
                 iree_hip_execution_resource_descriptor_registry_initialize);
  if (!iree_hip_handle_registry_remove(
          &iree_hip_execution_resource_descriptor_registry,
          (uintptr_t)handle)) {
    return false;
  }
  *out_descriptor = (iree_hip_execution_resource_descriptor_t*)handle;
  return true;
}

void iree_hip_execution_resource_descriptor_release(
    iree_hip_execution_resource_descriptor_t* descriptor) {
  if (!descriptor) {
    return;
  }
  if (iree_atomic_ref_count_dec(&descriptor->ref_count) != 1) {
    return;
  }
  iree_allocator_free(descriptor->host_allocator, descriptor);
}
