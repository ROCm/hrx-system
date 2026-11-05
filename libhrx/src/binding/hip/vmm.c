// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/vmm.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/context.h"
#include "common/device.h"
#include "common/memory.h"
#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"

// HIP owns the installed-alias ledger and the backing references it implies.
// Native devices use ordinary backing and virtual buffers; the explicit legacy
// provider retains its scalar allocator contract.
typedef struct iree_hip_vmm_allocation_t iree_hip_vmm_allocation_t;

typedef struct iree_hip_vmm_mapping_t {
  // Offset within the virtual reservation in bytes.
  size_t virtual_offset;
  // Offset within the physical allocation in bytes.
  size_t physical_offset;
  // Length of this mapping in bytes.
  size_t size;
  // Physical allocation retained by the mapping registry.
  iree_hip_vmm_allocation_t* allocation;
} iree_hip_vmm_mapping_t;

typedef struct iree_hip_vmm_access_range_t {
  // HIP location class whose permissions this record describes.
  hipMemLocationType location_type;
  // HIP location ordinal within |location_type|.
  int location_id;
  // Offset within the virtual reservation in bytes.
  size_t offset;
  // Length of the permission range in bytes.
  size_t size;
  // Permissions last applied to this range.
  hipMemAccessFlags flags;
} iree_hip_vmm_access_range_t;

typedef struct iree_hip_vmm_reservation_t {
  // Registry and in-flight operation references.
  iree_atomic_ref_count_t ref_count;
  // Serializes mapping and access-state mutation for this reservation.
  iree_slim_mutex_t mutex;
  // Device retained while allocator-owned reservation state is live.
  hrx_device_t device;
  // Borrowed allocator owned by |device|.
  hrx_allocator_t allocator;
  // HRX virtual address reservation.
  hrx_buffer_t virtual_buffer;
  // Owned mapping timeline; NULL selects the legacy scalar implementation.
  hrx_semaphore_t mapping_completion;
  // Last accepted mapping completion point, serialized by |mutex|.
  uint64_t mapping_completion_value;
  // Terminal accepted mapping error; backing remains owned until full detach.
  hipError_t mapping_failure;
  // Streaming wrapper publishing the reservation to pointer lookup.
  iree_hal_streaming_buffer_t* streaming_buffer;
  // Base device address returned to HIP callers.
  uintptr_t base_address;
  // Total reservation length in bytes.
  size_t size;
  // Minimum page size for mapping and protection operations.
  size_t granularity;
  // Device ordinal owning the virtual address space.
  int device_ordinal;
  // True after removal from the process registry begins.
  bool retiring;
  // Sorted, non-overlapping physical mappings.
  iree_hip_vmm_mapping_t* mappings;
  // Number of live entries in |mappings|.
  size_t mapping_count;
  // Allocated entry capacity of |mappings|.
  size_t mapping_capacity;
  // Sorted, normalized access ranges.
  iree_hip_vmm_access_range_t* access_ranges;
  // Number of live entries in |access_ranges|.
  size_t access_range_count;
} iree_hip_vmm_reservation_t;

struct iree_hip_vmm_allocation_t {
  // Registry and in-flight operation references.
  iree_atomic_ref_count_t ref_count;
  // Serializes public and mapping reference transitions.
  iree_slim_mutex_t mutex;
  // Device retained while physical memory is live.
  hrx_device_t device;
  // Device ordinal whose allocator owns the physical handle.
  int device_ordinal;
  // Borrowed allocator owned by |device|.
  hrx_allocator_t allocator;
  // Allocator-owned physical memory handle.
  hrx_physical_memory_t physical_memory;
  // Ordinary prepared backing; NULL selects the legacy physical handle.
  hrx_buffer_t backing;
  // Properties reported through the HIP handle query API.
  hipMemAllocationProp properties;
  // Physical allocation length in bytes.
  size_t size;
  // Monotonic opaque key exposed as the HIP handle value.
  uintptr_t handle_key;
  // Number of public HIP handle references.
  uint64_t public_reference_count;
  // Number of mapping records retaining this allocation.
  uint64_t mapping_reference_count;
  // True while the final native free is in progress.
  bool retiring;
};

typedef struct iree_hip_vmm_registry_t {
  // Guards registry vectors and handle generation only.
  iree_slim_mutex_t mutex;
  // Reservations sorted by base device address.
  iree_hip_vmm_reservation_t** reservations;
  // Number of entries in |reservations|.
  size_t reservation_count;
  // Allocated entry capacity of |reservations|.
  size_t reservation_capacity;
  // Allocations sorted by monotonic handle key.
  iree_hip_vmm_allocation_t** allocations;
  // Number of entries in |allocations|.
  size_t allocation_count;
  // Allocated entry capacity of |allocations|.
  size_t allocation_capacity;
  // Next never-reused opaque handle key.
  uintptr_t next_handle_key;
} iree_hip_vmm_registry_t;

static iree_once_flag iree_hip_vmm_registry_once = IREE_ONCE_FLAG_INIT;
static iree_hip_vmm_registry_t iree_hip_vmm_registry;

static void iree_hip_vmm_registry_initialize(void) {
  memset(&iree_hip_vmm_registry, 0, sizeof(iree_hip_vmm_registry));
  iree_slim_mutex_initialize(&iree_hip_vmm_registry.mutex);
  iree_hip_vmm_registry.next_handle_key = 1;
}

static iree_hip_vmm_registry_t* iree_hip_vmm_registry_lock(void) {
  iree_call_once(&iree_hip_vmm_registry_once, iree_hip_vmm_registry_initialize);
  iree_slim_mutex_lock(&iree_hip_vmm_registry.mutex);
  return &iree_hip_vmm_registry;
}

static void iree_hip_vmm_registry_unlock(void) {
  iree_slim_mutex_unlock(&iree_hip_vmm_registry.mutex);
}

static hipError_t iree_hip_vmm_from_hrx_status(hrx_status_t status) {
  if (hrx_status_is_ok(status)) {
    return hipSuccess;
  }
  const hrx_status_code_t code = hrx_status_code(status);
  hrx_status_ignore(status);
  switch (code) {
    case HRX_STATUS_INVALID_ARGUMENT:
    case HRX_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case HRX_STATUS_OUT_OF_MEMORY:
      return hipErrorOutOfMemory;
    case HRX_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case HRX_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case HRX_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case HRX_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    default:
      return hipErrorUnknown;
  }
}

static hipError_t iree_hip_vmm_from_iree_status(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return hipSuccess;
  }
  const iree_status_code_t code = iree_status_code(status);
  iree_status_ignore(status);
  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
    case IREE_STATUS_OUT_OF_RANGE:
      return hipErrorInvalidValue;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return hipErrorOutOfMemory;
    case IREE_STATUS_NOT_FOUND:
      return hipErrorNotFound;
    case IREE_STATUS_PERMISSION_DENIED:
      return hipErrorInvalidContext;
    case IREE_STATUS_UNIMPLEMENTED:
      return hipErrorNotSupported;
    case IREE_STATUS_UNAVAILABLE:
      return hipErrorNotReady;
    case IREE_STATUS_FAILED_PRECONDITION:
      return hipErrorNotInitialized;
    default:
      return hipErrorUnknown;
  }
}

static hipError_t iree_hip_vmm_grow_pointer_array(void** values,
                                                  size_t* capacity,
                                                  size_t minimum_capacity) {
  if (*capacity >= minimum_capacity) {
    return hipSuccess;
  }
  size_t new_capacity = *capacity ? *capacity : 16;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > SIZE_MAX / 2) {
      return hipErrorOutOfMemory;
    }
    new_capacity *= 2;
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(new_capacity, sizeof(void*),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_status_t status =
      iree_allocator_realloc(iree_allocator_system(), allocation_size, values);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  *capacity = new_capacity;
  return hipSuccess;
}

static void iree_hip_vmm_reservation_retain(
    iree_hip_vmm_reservation_t* reservation) {
  if (!reservation) {
    return;
  }
  iree_atomic_ref_count_inc(&reservation->ref_count);
}

static void iree_hip_vmm_reservation_release(
    iree_hip_vmm_reservation_t* reservation) {
  if (!reservation) {
    return;
  }
  if (iree_atomic_ref_count_dec(&reservation->ref_count) != 1) {
    return;
  }
  iree_allocator_free(iree_allocator_system(), reservation->access_ranges);
  iree_allocator_free(iree_allocator_system(), reservation->mappings);
  hrx_semaphore_release(reservation->mapping_completion);
  iree_slim_mutex_deinitialize(&reservation->mutex);
  hrx_device_release(reservation->device);
  iree_allocator_free(iree_allocator_system(), reservation);
}

static void iree_hip_vmm_allocation_retain(
    iree_hip_vmm_allocation_t* allocation) {
  if (!allocation) {
    return;
  }
  iree_atomic_ref_count_inc(&allocation->ref_count);
}

static void iree_hip_vmm_allocation_release(
    iree_hip_vmm_allocation_t* allocation) {
  if (!allocation) {
    return;
  }
  if (iree_atomic_ref_count_dec(&allocation->ref_count) != 1) {
    return;
  }
  iree_slim_mutex_deinitialize(&allocation->mutex);
  hrx_device_release(allocation->device);
  iree_allocator_free(iree_allocator_system(), allocation);
}

static size_t iree_hip_vmm_reservation_lower_bound(
    const iree_hip_vmm_registry_t* registry, uintptr_t base_address) {
  size_t begin = 0;
  size_t end = registry->reservation_count;
  while (begin < end) {
    const size_t middle = begin + (end - begin) / 2;
    if (registry->reservations[middle]->base_address < base_address) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static hipError_t iree_hip_vmm_registry_insert_reservation_locked(
    iree_hip_vmm_registry_t* registry,
    iree_hip_vmm_reservation_t* reservation) {
  hipError_t result = iree_hip_vmm_grow_pointer_array(
      (void**)&registry->reservations, &registry->reservation_capacity,
      registry->reservation_count + 1);
  if (result != hipSuccess) {
    return result;
  }
  const size_t position =
      iree_hip_vmm_reservation_lower_bound(registry, reservation->base_address);
  if (position < registry->reservation_count &&
      registry->reservations[position]->base_address ==
          reservation->base_address) {
    return hipErrorInvalidValue;
  }
  memmove(&registry->reservations[position + 1],
          &registry->reservations[position],
          (registry->reservation_count - position) *
              sizeof(registry->reservations[0]));
  registry->reservations[position] = reservation;
  ++registry->reservation_count;
  return hipSuccess;
}

static bool iree_hip_vmm_registry_remove_reservation_locked(
    iree_hip_vmm_registry_t* registry,
    iree_hip_vmm_reservation_t* reservation) {
  const size_t position =
      iree_hip_vmm_reservation_lower_bound(registry, reservation->base_address);
  if (position >= registry->reservation_count ||
      registry->reservations[position] != reservation) {
    return false;
  }
  memmove(&registry->reservations[position],
          &registry->reservations[position + 1],
          (registry->reservation_count - position - 1) *
              sizeof(registry->reservations[0]));
  --registry->reservation_count;
  if (!registry->reservation_count) {
    iree_allocator_free(iree_allocator_system(), registry->reservations);
    registry->reservations = NULL;
    registry->reservation_capacity = 0;
  }
  return true;
}

static iree_hip_vmm_reservation_t* iree_hip_vmm_lookup_reservation(
    uintptr_t address, bool require_base) {
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  const size_t position =
      iree_hip_vmm_reservation_lower_bound(registry, address);
  iree_hip_vmm_reservation_t* reservation = NULL;
  if (position < registry->reservation_count &&
      registry->reservations[position]->base_address == address) {
    reservation = registry->reservations[position];
  } else if (!require_base && position > 0) {
    iree_hip_vmm_reservation_t* candidate =
        registry->reservations[position - 1];
    if (address >= candidate->base_address &&
        address - candidate->base_address < candidate->size) {
      reservation = candidate;
    }
  }
  if (reservation) {
    iree_hip_vmm_reservation_retain(reservation);
  }
  iree_hip_vmm_registry_unlock();
  return reservation;
}

static size_t iree_hip_vmm_allocation_lower_bound(
    const iree_hip_vmm_registry_t* registry, uintptr_t handle_key) {
  size_t begin = 0;
  size_t end = registry->allocation_count;
  while (begin < end) {
    const size_t middle = begin + (end - begin) / 2;
    if (registry->allocations[middle]->handle_key < handle_key) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static hipError_t iree_hip_vmm_registry_insert_allocation_locked(
    iree_hip_vmm_registry_t* registry, iree_hip_vmm_allocation_t* allocation) {
  hipError_t result = iree_hip_vmm_grow_pointer_array(
      (void**)&registry->allocations, &registry->allocation_capacity,
      registry->allocation_count + 1);
  if (result != hipSuccess) {
    return result;
  }
  const size_t position =
      iree_hip_vmm_allocation_lower_bound(registry, allocation->handle_key);
  memmove(&registry->allocations[position + 1],
          &registry->allocations[position],
          (registry->allocation_count - position) *
              sizeof(registry->allocations[0]));
  registry->allocations[position] = allocation;
  ++registry->allocation_count;
  return hipSuccess;
}

static bool iree_hip_vmm_registry_remove_allocation_locked(
    iree_hip_vmm_registry_t* registry, iree_hip_vmm_allocation_t* allocation) {
  const size_t position =
      iree_hip_vmm_allocation_lower_bound(registry, allocation->handle_key);
  if (position >= registry->allocation_count ||
      registry->allocations[position] != allocation) {
    return false;
  }
  memmove(&registry->allocations[position],
          &registry->allocations[position + 1],
          (registry->allocation_count - position - 1) *
              sizeof(registry->allocations[0]));
  --registry->allocation_count;
  if (!registry->allocation_count) {
    iree_allocator_free(iree_allocator_system(), registry->allocations);
    registry->allocations = NULL;
    registry->allocation_capacity = 0;
  }
  return true;
}

static iree_hip_vmm_allocation_t* iree_hip_vmm_lookup_allocation(
    hipMemGenericAllocationHandle_t handle) {
  const uintptr_t handle_key = (uintptr_t)handle;
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  const size_t position =
      iree_hip_vmm_allocation_lower_bound(registry, handle_key);
  iree_hip_vmm_allocation_t* allocation = NULL;
  if (position < registry->allocation_count &&
      registry->allocations[position]->handle_key == handle_key) {
    allocation = registry->allocations[position];
    iree_hip_vmm_allocation_retain(allocation);
  }
  iree_hip_vmm_registry_unlock();
  return allocation;
}

static bool iree_hip_vmm_is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

static bool iree_hip_vmm_range_is_valid(size_t total_size, size_t offset,
                                        size_t size, size_t granularity) {
  return size != 0 && granularity != 0 && size % granularity == 0 &&
         offset % granularity == 0 && offset <= total_size &&
         size <= total_size - offset;
}

static hipError_t iree_hip_vmm_validate_properties(
    const hipMemAllocationProp* properties) {
  if (!properties || (properties->type != hipMemAllocationTypePinned &&
                      properties->type != hipMemAllocationTypeUncached)) {
    return hipErrorInvalidValue;
  }
  switch (properties->requestedHandleType) {
    case hipMemHandleTypeNone:
    case hipMemHandleTypePosixFileDescriptor:
    case hipMemHandleTypeWin32:
    case hipMemHandleTypeWin32Kmt:
      break;
    default:
      return hipErrorInvalidValue;
  }
  int device_count = 0;
  hipError_t result = hipGetDeviceCount(&device_count);
  if (result != hipSuccess) {
    return result;
  }
  if (properties->location.type == hipMemLocationTypeHost) {
    return hipSuccess;
  }
  if (properties->location.type != hipMemLocationTypeDevice) {
    return hipErrorInvalidValue;
  }
  return properties->location.id >= 0 && properties->location.id < device_count
             ? hipSuccess
             : hipErrorInvalidDevice;
}

static hrx_memory_type_t iree_hip_vmm_memory_type(
    const hipMemAllocationProp* properties) {
  const hrx_memory_type_t cache_type =
      properties->type == hipMemAllocationTypeUncached
          ? HRX_MEMORY_TYPE_DEVICE_UNCACHED
          : HRX_MEMORY_TYPE_NONE;
  if (properties->location.type == hipMemLocationTypeHost) {
    // The location selects the physical backing owner, not merely which agent
    // may access the eventual mapping. Host backing remains device-visible so
    // access can be granted independently to host and device agents.
    return HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT |
           HRX_MEMORY_TYPE_DEVICE_VISIBLE | cache_type;
  }
  return HRX_MEMORY_TYPE_DEVICE_LOCAL | cache_type;
}

static hipError_t iree_hip_vmm_get_device_for_properties(
    const hipMemAllocationProp* properties, int* out_device_ordinal,
    hrx_device_t* out_device) {
  int device_ordinal = properties->location.id;
  if (properties->location.type == hipMemLocationTypeHost) {
    hipError_t result = hipGetDevice(&device_ordinal);
    if (result != hipSuccess) {
      return result;
    }
  }
  hrx_device_t device = NULL;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) {
    return result;
  }
  *out_device_ordinal = device_ordinal;
  *out_device = device;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_prepare_scope(
    hrx_device_t device, hrx_device_t* devices,
    hrx_virtual_buffer_source_t* sources, hrx_virtual_buffer_scope_t* out_scope,
    hrx_virtual_buffer_sources_t* out_sources) {
  int count = 0;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_count(&count));
  size_t index = 1;
  devices[0] = device;
  for (int i = 0; i < count && result == hipSuccess; ++i) {
    hrx_device_t participant = NULL;
    result = iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(i, &participant));
    if (result == hipSuccess && participant != device) {
      devices[index++] = participant;
    }
  }
  *out_scope = (hrx_virtual_buffer_scope_t){
      .device_count = index,
      .devices = devices,
      .host_access = HRX_MEMORY_ACCESS_ALL,
  };
  // hipMemAddressReserve precedes physical allocation. Its immutable contract
  // must admit ordinary device backing for GPUs and coherent shared backing
  // for CPU/GPU aliases, independently of the permissions later granted by
  // hipMemSetAccess. Device backing does not imply remappable CPU access.
  for (size_t i = 0; i < index; ++i) {
    sources[2 * i] = (hrx_virtual_buffer_source_t){
        .device = devices[i],
        .memory_type = HRX_MEMORY_TYPE_DEVICE_LOCAL,
    };
    sources[2 * i + 1] = (hrx_virtual_buffer_source_t){
        .device = devices[i],
        .memory_type = HRX_MEMORY_TYPE_HOST_LOCAL,
    };
  }
  *out_sources = (hrx_virtual_buffer_sources_t){
      .source_count = 2 * index,
      .sources = sources,
  };
  return result;
}

static hipError_t iree_hip_vmm_query_granularity(hrx_device_t device,
                                                 hrx_memory_type_t memory_type,
                                                 size_t* out_minimum,
                                                 size_t* out_recommended,
                                                 bool* out_modern) {
  *out_modern = false;
  hrx_device_t devices[IREE_HAL_STREAMING_MAX_DEVICES];
  hrx_virtual_buffer_source_t sources[2 * IREE_HAL_STREAMING_MAX_DEVICES];
  hrx_virtual_buffer_scope_t scope;
  hrx_virtual_buffer_sources_t accepted_sources;
  hipError_t result = iree_hip_vmm_prepare_scope(device, devices, sources,
                                                 &scope, &accepted_sources);
  hrx_virtual_buffer_capabilities_t capabilities = {0};
  if (result == hipSuccess) {
    result = iree_hip_vmm_from_hrx_status(hrx_virtual_buffer_query_capabilities(
        scope, accepted_sources, &capabilities));
  }
  if (result != hipSuccess) {
    return result;
  }
  if (capabilities.max_byte_length) {
    if (!iree_all_bits_set(capabilities.supported_flags,
                           HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS)) {
      return hipErrorNotSupported;
    }
    const uint32_t required = (1u << HRX_MEMORY_MAPPING_OPERATION_BIND) |
                              (1u << HRX_MEMORY_MAPPING_OPERATION_UNBIND) |
                              (1u << HRX_MEMORY_MAPPING_OPERATION_PROTECT);
    if (!iree_all_bits_set(capabilities.operations, required)) {
      return hipErrorNotSupported;
    }
    hrx_memory_mapping_requirements_t requirements;
    result =
        iree_hip_vmm_from_hrx_status(hrx_allocator_query_mapping_requirements(
            hrx_device_allocator(device), memory_type, &requirements));
    if (result == hipSuccess) {
      *out_modern = true;
      *out_minimum = iree_max(requirements.minimum_granularity,
                              iree_max(capabilities.mapping_granularity,
                                       capabilities.reservation_granularity));
      *out_recommended =
          iree_max(*out_minimum, requirements.recommended_granularity);
    }
    return result;
  }
  bool supported = false;
  result = iree_hip_vmm_from_hrx_status(hrx_allocator_query_virtual_memory(
      hrx_device_allocator(device), memory_type, &supported, out_minimum,
      out_recommended));
  if (result != hipSuccess) {
    return result;
  }
  return supported && *out_minimum != 0 ? hipSuccess : hipErrorNotSupported;
}

static hipError_t iree_hip_vmm_update_mappings(
    iree_hip_vmm_reservation_t* reservation, size_t count,
    const hrx_memory_mapping_t* mappings) {
  uint64_t value = ++reservation->mapping_completion_value;
  const hrx_semaphore_list_t signals = {
      .semaphores = &reservation->mapping_completion,
      .values = &value,
      .count = 1,
  };
  hipError_t result = iree_hip_vmm_from_hrx_status(
      hrx_queue_map(reservation->device, 0, NULL, &signals, count, mappings,
                    HRX_QUEUE_MAP_FLAG_NONE));
  if (result == hipSuccess) {
    result = iree_hip_vmm_from_hrx_status(
        hrx_semaphore_wait(reservation->mapping_completion, value, UINT64_MAX));
    if (result != hipSuccess) {
      // Accepted native failure can leave a prefix installed. Existing ledger
      // references remain authoritative until checked full removal succeeds.
      reservation->mapping_failure = result;
    }
  }
  return result;
}

static size_t iree_hip_vmm_mapping_lower_bound(
    const iree_hip_vmm_reservation_t* reservation, size_t offset) {
  size_t begin = 0;
  size_t end = reservation->mapping_count;
  while (begin < end) {
    const size_t middle = begin + (end - begin) / 2;
    if (reservation->mappings[middle].virtual_offset < offset) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static size_t iree_hip_vmm_mapping_containing(
    const iree_hip_vmm_reservation_t* reservation, size_t offset) {
  const size_t position = iree_hip_vmm_mapping_lower_bound(reservation, offset);
  if (position < reservation->mapping_count &&
      reservation->mappings[position].virtual_offset == offset) {
    return position;
  }
  if (position == 0) {
    return SIZE_MAX;
  }
  const iree_hip_vmm_mapping_t* candidate =
      &reservation->mappings[position - 1];
  return offset - candidate->virtual_offset < candidate->size ? position - 1
                                                              : SIZE_MAX;
}

static bool iree_hip_vmm_mapped_range_is_complete(
    const iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size) {
  size_t position = iree_hip_vmm_mapping_containing(reservation, offset);
  if (position == SIZE_MAX) {
    return false;
  }
  size_t cursor = offset;
  const size_t end = offset + size;
  while (position < reservation->mapping_count && cursor < end) {
    const iree_hip_vmm_mapping_t* mapping = &reservation->mappings[position];
    if (cursor < mapping->virtual_offset ||
        cursor - mapping->virtual_offset >= mapping->size) {
      return false;
    }
    const size_t mapping_end = mapping->virtual_offset + mapping->size;
    cursor = mapping_end < end ? mapping_end : end;
    ++position;
  }
  return cursor == end;
}

static bool iree_hip_vmm_mapping_range_overlaps(
    const iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size) {
  const size_t position = iree_hip_vmm_mapping_lower_bound(reservation, offset);
  if (position < reservation->mapping_count &&
      reservation->mappings[position].virtual_offset < offset + size) {
    return true;
  }
  if (position == 0) {
    return false;
  }
  const iree_hip_vmm_mapping_t* previous = &reservation->mappings[position - 1];
  return offset < previous->virtual_offset + previous->size;
}

static hipError_t iree_hip_vmm_reserve_mapping_capacity(
    iree_hip_vmm_reservation_t* reservation, size_t minimum_capacity) {
  if (reservation->mapping_capacity >= minimum_capacity) {
    return hipSuccess;
  }
  size_t new_capacity =
      reservation->mapping_capacity ? reservation->mapping_capacity : 8;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > SIZE_MAX / 2) {
      return hipErrorOutOfMemory;
    }
    new_capacity *= 2;
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(
          new_capacity, sizeof(reservation->mappings[0]), &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_status_t status = iree_allocator_realloc(
      iree_allocator_system(), allocation_size, (void**)&reservation->mappings);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }
  reservation->mapping_capacity = new_capacity;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_retire_allocation(
    iree_hip_vmm_allocation_t* allocation, bool restore_public_on_failure) {
  hipError_t result = hipSuccess;
  if (allocation->backing) {
    hrx_buffer_release(allocation->backing);
    allocation->backing = NULL;
  } else {
    result = iree_hip_vmm_from_hrx_status(hrx_allocator_physical_memory_free(
        allocation->allocator, allocation->physical_memory));
  }
  if (result == hipSuccess) {
    allocation->physical_memory = NULL;
    iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
    const bool removed =
        iree_hip_vmm_registry_remove_allocation_locked(registry, allocation);
    iree_hip_vmm_registry_unlock();
    // The retiring entry remains registered until its unique native free
    // succeeds; no other operation can remove it during that call.
    IREE_ASSERT(removed);
    iree_hip_vmm_allocation_release(allocation);  // Registry reference.
    return hipSuccess;
  }

  iree_slim_mutex_lock(&allocation->mutex);
  allocation->retiring = false;
  if (restore_public_on_failure) {
    ++allocation->public_reference_count;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  return result;
}

static hipError_t iree_hip_vmm_mapping_reference_add(
    iree_hip_vmm_allocation_t* allocation, bool require_public_reference) {
  iree_slim_mutex_lock(&allocation->mutex);
  hipError_t result = hipSuccess;
  if (allocation->retiring ||
      (require_public_reference && allocation->public_reference_count == 0) ||
      allocation->mapping_reference_count == UINT64_MAX) {
    result = hipErrorInvalidValue;
  } else {
    ++allocation->mapping_reference_count;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  return result;
}

static hipError_t iree_hip_vmm_mapping_reference_remove(
    iree_hip_vmm_allocation_t* allocation) {
  bool retire = false;
  iree_slim_mutex_lock(&allocation->mutex);
  if (allocation->mapping_reference_count == 0) {
    iree_slim_mutex_unlock(&allocation->mutex);
    return hipErrorInvalidValue;
  }
  --allocation->mapping_reference_count;
  if (allocation->mapping_reference_count == 0 &&
      allocation->public_reference_count == 0 && !allocation->retiring) {
    allocation->retiring = true;
    retire = true;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  return retire ? iree_hip_vmm_retire_allocation(
                      allocation, /*restore_public_on_failure=*/false)
                : hipSuccess;
}

static int iree_hip_vmm_compare_access_ranges(const void* lhs,
                                              const void* rhs) {
  const iree_hip_vmm_access_range_t* a =
      (const iree_hip_vmm_access_range_t*)lhs;
  const iree_hip_vmm_access_range_t* b =
      (const iree_hip_vmm_access_range_t*)rhs;
  if (a->location_type != b->location_type) {
    return a->location_type < b->location_type ? -1 : 1;
  }
  if (a->location_id != b->location_id) {
    return a->location_id < b->location_id ? -1 : 1;
  }
  if (a->offset == b->offset) {
    return 0;
  }
  return a->offset < b->offset ? -1 : 1;
}

static bool iree_hip_vmm_access_location_matches(
    const iree_hip_vmm_access_range_t* range, const hipMemLocation* location) {
  const int location_id =
      location->type == hipMemLocationTypeHost ? 0 : location->id;
  return range->location_type == location->type &&
         range->location_id == location_id;
}

static hipError_t iree_hip_vmm_build_access_update(
    const iree_hip_vmm_access_range_t* previous, size_t previous_count,
    const hipMemLocation* location, size_t offset, size_t size,
    hipMemAccessFlags flags, bool insert_update,
    iree_hip_vmm_access_range_t** out_ranges, size_t* out_count) {
  *out_ranges = NULL;
  *out_count = 0;
  size_t maximum_count = 0;
  if (!iree_host_size_checked_mul(previous_count, 2, &maximum_count) ||
      !iree_host_size_checked_add(maximum_count, insert_update ? 1 : 0,
                                  &maximum_count)) {
    return hipErrorOutOfMemory;
  }
  if (maximum_count == 0) {
    return hipSuccess;
  }

  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(maximum_count,
                                  sizeof(iree_hip_vmm_access_range_t),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_hip_vmm_access_range_t* ranges = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), allocation_size, (void**)&ranges);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }

  const size_t update_end = offset + size;
  size_t count = 0;
  for (size_t i = 0; i < previous_count; ++i) {
    const iree_hip_vmm_access_range_t* old = &previous[i];
    const size_t old_end = old->offset + old->size;
    if ((location && !iree_hip_vmm_access_location_matches(old, location)) ||
        old_end <= offset || old->offset >= update_end) {
      ranges[count++] = *old;
      continue;
    }
    if (old->offset < offset) {
      ranges[count++] = (iree_hip_vmm_access_range_t){
          .location_type = old->location_type,
          .location_id = old->location_id,
          .offset = old->offset,
          .size = offset - old->offset,
          .flags = old->flags,
      };
    }
    if (old_end > update_end) {
      ranges[count++] = (iree_hip_vmm_access_range_t){
          .location_type = old->location_type,
          .location_id = old->location_id,
          .offset = update_end,
          .size = old_end - update_end,
          .flags = old->flags,
      };
    }
  }
  if (insert_update && flags != hipMemAccessFlagsProtNone) {
    IREE_ASSERT_ARGUMENT(location);
    ranges[count++] = (iree_hip_vmm_access_range_t){
        .location_type = location->type,
        .location_id =
            location->type == hipMemLocationTypeHost ? 0 : location->id,
        .offset = offset,
        .size = size,
        .flags = flags,
    };
  }
  qsort(ranges, count, sizeof(ranges[0]), iree_hip_vmm_compare_access_ranges);

  size_t merged_count = 0;
  for (size_t i = 0; i < count; ++i) {
    if (merged_count > 0) {
      iree_hip_vmm_access_range_t* previous = &ranges[merged_count - 1];
      if (previous->location_type == ranges[i].location_type &&
          previous->location_id == ranges[i].location_id &&
          previous->flags == ranges[i].flags &&
          previous->offset + previous->size == ranges[i].offset) {
        previous->size += ranges[i].size;
        continue;
      }
    }
    ranges[merged_count++] = ranges[i];
  }
  *out_ranges = ranges;
  *out_count = merged_count;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_validate_access_location(
    const hipMemLocation* location) {
  if (!location) {
    return hipErrorInvalidValue;
  }
  if (location->type == hipMemLocationTypeHost) {
    return hipSuccess;
  }
  if (location->type != hipMemLocationTypeDevice) {
    return hipErrorInvalidValue;
  }
  int device_count = 0;
  hipError_t result = hipGetDeviceCount(&device_count);
  if (result != hipSuccess) {
    return result;
  }
  return location->id >= 0 && location->id < device_count
             ? hipSuccess
             : hipErrorInvalidValue;
}

static hipError_t iree_hip_vmm_validate_access_descriptor(
    const hipMemAccessDesc* descriptor) {
  hipError_t result =
      iree_hip_vmm_validate_access_location(&descriptor->location);
  if (result != hipSuccess) {
    return result;
  }
  switch (descriptor->flags) {
    case hipMemAccessFlagsProtNone:
    case hipMemAccessFlagsProtRead:
    case hipMemAccessFlagsProtReadWrite:
      return hipSuccess;
    default:
      return hipErrorInvalidValue;
  }
}

static hrx_memory_protection_t iree_hip_vmm_protection(
    hipMemAccessFlags flags) {
  switch (flags) {
    case hipMemAccessFlagsProtRead:
      return HRX_MEMORY_PROTECTION_READ;
    case hipMemAccessFlagsProtReadWrite:
      return HRX_MEMORY_PROTECTION_READ_WRITE;
    default:
      return HRX_MEMORY_PROTECTION_NONE;
  }
}

hipError_t iree_hip_vmm_is_supported(int device_ordinal, bool* out_supported) {
  if (!out_supported) {
    return hipErrorInvalidValue;
  }
  *out_supported = false;
  hrx_device_t device = NULL;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) {
    return result;
  }
  size_t minimum = 0;
  size_t recommended = 0;
  bool modern = false;
  result = iree_hip_vmm_query_granularity(device, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                          &minimum, &recommended, &modern);
  if (result == hipErrorNotSupported) {
    return hipSuccess;
  }
  *out_supported = result == hipSuccess;
  return result;
}

hipError_t iree_hip_vmm_address_reserve(void** ptr, size_t size,
                                        size_t alignment, void* address,
                                        unsigned long long flags) {
  if (!ptr || flags != 0 || address) {
    return hipErrorInvalidValue;
  }
  *ptr = NULL;
  if (alignment != 0 && !iree_hip_vmm_is_power_of_two(alignment)) {
    return hipErrorInvalidValue;
  }

  int device_ordinal = 0;
  hipError_t result = hipGetDevice(&device_ordinal);
  if (result != hipSuccess) {
    return result;
  }
  hrx_device_t device = NULL;
  result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) {
    return result;
  }
  size_t granularity = 0;
  size_t recommended = 0;
  bool modern = false;
  result = iree_hip_vmm_query_granularity(device, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                          &granularity, &recommended, &modern);
  if (result != hipSuccess ||
      !iree_hip_vmm_range_is_valid(size, 0, size, granularity)) {
    return result == hipSuccess ? hipErrorInvalidValue : result;
  }

  hrx_allocator_t allocator = hrx_device_allocator(device);
  hrx_buffer_t virtual_buffer = NULL;
  if (modern) {
    hrx_device_t devices[IREE_HAL_STREAMING_MAX_DEVICES];
    hrx_virtual_buffer_source_t sources[2 * IREE_HAL_STREAMING_MAX_DEVICES];
    hrx_virtual_buffer_scope_t scope;
    hrx_virtual_buffer_sources_t accepted_sources;
    result = iree_hip_vmm_prepare_scope(device, devices, sources, &scope,
                                        &accepted_sources);
    const hrx_virtual_buffer_options_t options = {
        .byte_length = size,
        .min_alignment = alignment,
        .mapping_granularity = granularity,
        .flags = HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
    };
    if (result == hipSuccess) {
      result = iree_hip_vmm_from_hrx_status(hrx_virtual_buffer_create(
          scope, accepted_sources, &options, &virtual_buffer));
    }
  } else {
    result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_reserve(
        allocator, /*affinity=*/0, size, &virtual_buffer));
  }
  if (result != hipSuccess) {
    return result;
  }
  void* device_ptr = NULL;
  result = iree_hip_vmm_from_hrx_status(
      hrx_buffer_get_device_ptr(virtual_buffer, &device_ptr));
  if (result == hipSuccess && alignment != 0 &&
      (uintptr_t)device_ptr % alignment != 0) {
    result = hipErrorNotSupported;
  }

  iree_hal_streaming_buffer_t* streaming_buffer = NULL;
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (result == hipSuccess && !context) {
    result = hipErrorInvalidContext;
  }
  if (result == hipSuccess) {
    iree_status_t status = iree_hal_streaming_memory_wrap_virtual_reservation(
        context, virtual_buffer, &streaming_buffer);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      result = hipErrorOutOfMemory;
    }
  }

  iree_hip_vmm_reservation_t* reservation = NULL;
  if (result == hipSuccess) {
    iree_status_t status = iree_allocator_malloc(
        iree_allocator_system(), sizeof(*reservation), (void**)&reservation);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      result = hipErrorOutOfMemory;
    }
  }
  if (result == hipSuccess) {
    memset(reservation, 0, sizeof(*reservation));
    iree_atomic_ref_count_init(&reservation->ref_count);
    iree_slim_mutex_initialize(&reservation->mutex);
    reservation->device = device;
    hrx_device_retain(device);
    reservation->allocator = allocator;
    reservation->virtual_buffer = virtual_buffer;
    reservation->streaming_buffer = streaming_buffer;
    reservation->base_address = (uintptr_t)device_ptr;
    reservation->size = size;
    reservation->granularity = granularity;
    reservation->device_ordinal = device_ordinal;
    if (modern) {
      result = iree_hip_vmm_from_hrx_status(
          hrx_semaphore_create(device, 0, &reservation->mapping_completion));
    }
  }
  if (result == hipSuccess) {
    iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
    result =
        iree_hip_vmm_registry_insert_reservation_locked(registry, reservation);
    iree_hip_vmm_registry_unlock();
  }
  if (result != hipSuccess) {
    iree_hal_streaming_memory_release_wrapped_buffer(streaming_buffer);
    if (virtual_buffer) {
      if (modern) {
        hrx_buffer_release(virtual_buffer);
      } else {
        (void)iree_hip_vmm_from_hrx_status(
            hrx_allocator_virtual_memory_release(allocator, virtual_buffer));
      }
    }
    if (reservation) {
      reservation->virtual_buffer = NULL;
      reservation->streaming_buffer = NULL;
      iree_hip_vmm_reservation_release(reservation);
    }
    return result;
  }
  *ptr = device_ptr;
  return hipSuccess;
}

hipError_t iree_hip_vmm_address_free(void* device_ptr, size_t size) {
  if (!device_ptr || size == 0) {
    return hipErrorInvalidValue;
  }
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)device_ptr,
                                      /*require_base=*/true);
  if (!reservation) {
    return hipErrorInvalidValue;
  }

  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->retiring || reservation->size != size ||
      reservation->mapping_count != 0) {
    iree_slim_mutex_unlock(&reservation->mutex);
    iree_hip_vmm_reservation_release(reservation);
    return hipErrorInvalidValue;
  }
  reservation->retiring = true;

  hipError_t result = hipSuccess;
  if (reservation->mapping_completion) {
    result = iree_hip_vmm_from_hrx_status(hrx_virtual_buffer_detach(
        reservation->virtual_buffer, (hrx_memory_scope_t){0}));
    if (result == hipSuccess) {
      hrx_buffer_release(reservation->virtual_buffer);
    }
  } else {
    iree_hal_streaming_memory_prepare_virtual_reservation_release(
        reservation->streaming_buffer);
    result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_release(
        reservation->allocator, reservation->virtual_buffer));
  }
  if (result == hipSuccess) {
    reservation->virtual_buffer = NULL;
    iree_hal_streaming_memory_release_wrapped_buffer(
        reservation->streaming_buffer);
    reservation->streaming_buffer = NULL;
    iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
    const bool removed =
        iree_hip_vmm_registry_remove_reservation_locked(registry, reservation);
    iree_hip_vmm_registry_unlock();
    // The reservation mutex excludes another terminal release, and retiring
    // entries remain registered until native release succeeds.
    IREE_ASSERT(removed);
  } else {
    if (!reservation->mapping_completion) {
      iree_hal_streaming_memory_restore_virtual_reservation(
          reservation->streaming_buffer, reservation->virtual_buffer);
    }
    reservation->retiring = false;
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  if (result == hipSuccess) {
    iree_hip_vmm_reservation_release(reservation);  // Registry reference.
  }
  iree_hip_vmm_reservation_release(reservation);  // Operation reference.
  return result;
}

hipError_t iree_hip_vmm_create(hipMemGenericAllocationHandle_t* handle,
                               size_t size,
                               const hipMemAllocationProp* properties,
                               unsigned long long flags) {
  if (!handle || size == 0 || flags != 0) {
    return hipErrorInvalidValue;
  }
  *handle = NULL;
  hipError_t result = iree_hip_vmm_validate_properties(properties);
  if (result != hipSuccess) {
    return result;
  }
  if (properties->requestedHandleType != hipMemHandleTypeNone) {
    return hipErrorNotSupported;
  }

  int device_ordinal = 0;
  hrx_device_t device = NULL;
  result = iree_hip_vmm_get_device_for_properties(properties, &device_ordinal,
                                                  &device);
  if (result != hipSuccess) {
    return result;
  }
  const hrx_memory_type_t memory_type = iree_hip_vmm_memory_type(properties);
  size_t granularity = 0;
  size_t recommended = 0;
  bool modern = false;
  result = iree_hip_vmm_query_granularity(device, memory_type, &granularity,
                                          &recommended, &modern);
  if (result != hipSuccess ||
      !iree_hip_vmm_range_is_valid(size, 0, size, granularity)) {
    return result == hipSuccess ? hipErrorInvalidValue : result;
  }

  hrx_allocator_t allocator = hrx_device_allocator(device);
  hrx_physical_memory_t physical_memory = NULL;
  hrx_buffer_t backing = NULL;
  if (modern) {
    const hrx_buffer_params_t params = {
        .type = memory_type,
        .access = HRX_MEMORY_ACCESS_ALL,
        .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SOURCE,
    };
    result = iree_hip_vmm_from_hrx_status(
        hrx_allocator_allocate_buffer(allocator, params, size, &backing));
  } else {
    result =
        iree_hip_vmm_from_hrx_status(hrx_allocator_physical_memory_allocate(
            allocator, memory_type, size, &physical_memory));
  }
  if (result != hipSuccess) {
    return result;
  }

  iree_hip_vmm_allocation_t* allocation = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), sizeof(*allocation), (void**)&allocation);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    result = hipErrorOutOfMemory;
  }
  if (result == hipSuccess) {
    memset(allocation, 0, sizeof(*allocation));
    iree_atomic_ref_count_init(&allocation->ref_count);
    iree_slim_mutex_initialize(&allocation->mutex);
    allocation->device = device;
    hrx_device_retain(device);
    allocation->device_ordinal = device_ordinal;
    allocation->allocator = allocator;
    allocation->physical_memory = physical_memory;
    allocation->backing = backing;
    allocation->properties = *properties;
    allocation->size = size;
    allocation->public_reference_count = 1;

    iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
    if (registry->next_handle_key == UINTPTR_MAX) {
      result = hipErrorOutOfMemory;
    } else {
      allocation->handle_key = registry->next_handle_key;
      ++registry->next_handle_key;
      result =
          iree_hip_vmm_registry_insert_allocation_locked(registry, allocation);
    }
    iree_hip_vmm_registry_unlock();
  }
  if (result != hipSuccess) {
    if (backing) {
      hrx_buffer_release(backing);
    } else {
      (void)iree_hip_vmm_from_hrx_status(
          hrx_allocator_physical_memory_free(allocator, physical_memory));
    }
    if (allocation) {
      allocation->physical_memory = NULL;
      allocation->backing = NULL;
      iree_hip_vmm_allocation_release(allocation);
    }
    return result;
  }
  *handle = (hipMemGenericAllocationHandle_t)allocation->handle_key;
  return hipSuccess;
}

hipError_t iree_hip_vmm_release(hipMemGenericAllocationHandle_t handle) {
  if (!handle) {
    return hipErrorInvalidValue;
  }
  iree_hip_vmm_allocation_t* allocation =
      iree_hip_vmm_lookup_allocation(handle);
  if (!allocation) {
    return hipErrorInvalidValue;
  }

  bool retire = false;
  iree_slim_mutex_lock(&allocation->mutex);
  if (allocation->retiring || allocation->public_reference_count == 0) {
    iree_slim_mutex_unlock(&allocation->mutex);
    iree_hip_vmm_allocation_release(allocation);
    return hipErrorInvalidValue;
  }
  --allocation->public_reference_count;
  if (allocation->public_reference_count == 0 &&
      allocation->mapping_reference_count == 0) {
    allocation->retiring = true;
    retire = true;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  hipError_t result = retire
                          ? iree_hip_vmm_retire_allocation(
                                allocation, /*restore_public_on_failure=*/true)
                          : hipSuccess;
  iree_hip_vmm_allocation_release(allocation);
  return result;
}

hipError_t iree_hip_vmm_map(void* ptr, size_t size, size_t offset,
                            hipMemGenericAllocationHandle_t handle,
                            unsigned long long flags) {
  if (!ptr || !handle || size == 0 || offset != 0 || flags != 0) {
    return hipErrorInvalidValue;
  }
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  iree_hip_vmm_allocation_t* allocation =
      iree_hip_vmm_lookup_allocation(handle);
  if (!reservation || !allocation) {
    iree_hip_vmm_reservation_release(reservation);
    iree_hip_vmm_allocation_release(allocation);
    return hipErrorInvalidValue;
  }
  const size_t virtual_offset = (uintptr_t)ptr - reservation->base_address;

  iree_slim_mutex_lock(&reservation->mutex);
  hipError_t result = hipSuccess;
  if (reservation->mapping_failure != hipSuccess) {
    result = reservation->mapping_failure;
  } else if (reservation->retiring ||
             !iree_hip_vmm_range_is_valid(reservation->size, virtual_offset,
                                          size, reservation->granularity) ||
             offset > allocation->size || size > allocation->size - offset ||
             iree_hip_vmm_mapping_range_overlaps(reservation, virtual_offset,
                                                 size)) {
    result = hipErrorInvalidValue;
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_reserve_mapping_capacity(
        reservation, reservation->mapping_count + 1);
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_mapping_reference_add(
        allocation, /*require_public_reference=*/true);
  }
  if (result == hipSuccess) {
    if (!reservation->mapping_completion) {
      result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_map(
          reservation->allocator, reservation->virtual_buffer, virtual_offset,
          allocation->physical_memory, offset, size));
    }
    // Native NONE bindings remain detached until SetAccess supplies their
    // ordinary backing. The HIP ledger owns this inaccessible logical alias.
    if (result == hipSuccess) {
      const size_t position =
          iree_hip_vmm_mapping_lower_bound(reservation, virtual_offset);
      memmove(&reservation->mappings[position + 1],
              &reservation->mappings[position],
              (reservation->mapping_count - position) *
                  sizeof(reservation->mappings[0]));
      reservation->mappings[position] = (iree_hip_vmm_mapping_t){
          .virtual_offset = virtual_offset,
          .physical_offset = offset,
          .size = size,
          .allocation = allocation,
      };
      ++reservation->mapping_count;
    } else {
      (void)iree_hip_vmm_mapping_reference_remove(allocation);
    }
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_allocation_release(allocation);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

static hipError_t iree_hip_vmm_build_mapping_unmap(
    const iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size,
    iree_hip_vmm_mapping_t** out_mappings, size_t* out_count) {
  *out_mappings = NULL;
  *out_count = 0;
  size_t maximum_count = 0;
  if (!iree_host_size_checked_mul(reservation->mapping_count, 2,
                                  &maximum_count)) {
    return hipErrorOutOfMemory;
  }
  size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(maximum_count, sizeof(iree_hip_vmm_mapping_t),
                                  &allocation_size)) {
    return hipErrorOutOfMemory;
  }
  iree_hip_vmm_mapping_t* mappings = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), allocation_size, (void**)&mappings);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return hipErrorOutOfMemory;
  }

  const size_t unmap_end = offset + size;
  size_t count = 0;
  for (size_t i = 0; i < reservation->mapping_count; ++i) {
    const iree_hip_vmm_mapping_t* old = &reservation->mappings[i];
    const size_t old_end = old->virtual_offset + old->size;
    if (old_end <= offset || old->virtual_offset >= unmap_end) {
      mappings[count++] = *old;
      continue;
    }
    if (old->virtual_offset < offset) {
      mappings[count++] = (iree_hip_vmm_mapping_t){
          .virtual_offset = old->virtual_offset,
          .physical_offset = old->physical_offset,
          .size = offset - old->virtual_offset,
          .allocation = old->allocation,
      };
    }
    if (old_end > unmap_end) {
      mappings[count++] = (iree_hip_vmm_mapping_t){
          .virtual_offset = unmap_end,
          .physical_offset =
              old->physical_offset + unmap_end - old->virtual_offset,
          .size = old_end - unmap_end,
          .allocation = old->allocation,
      };
    }
  }
  *out_mappings = mappings;
  *out_count = count;
  return hipSuccess;
}

// Order temporary edits by canonical scope and address so locations sharing
// one native projection contribute only one removal for each backing range.
static int iree_hip_vmm_compare_mapping_edits(const void* lhs,
                                              const void* rhs) {
  const hrx_memory_mapping_t* a = lhs;
  const hrx_memory_mapping_t* b = rhs;
  if (a->scope.id != b->scope.id) {
    return a->scope.id < b->scope.id ? -1 : 1;
  }
  return (a->target_offset > b->target_offset) -
         (a->target_offset < b->target_offset);
}

// HIP owns both backing boundaries and per-location permissions. A pointer-only
// unmap intersects those existing records; untouched or inaccessible locations
// have no native binding to remove. Temporary descriptors die with this call.
static hipError_t iree_hip_vmm_remove_mappings(
    iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size) {
  const size_t end = offset + size;
  size_t capacity = 0;
  for (size_t i = 0; i < reservation->access_range_count; ++i) {
    const iree_hip_vmm_access_range_t* access = &reservation->access_ranges[i];
    const size_t begin = iree_max(offset, access->offset);
    const size_t limit = iree_min(end, access->offset + access->size);
    if (begin >= limit) {
      continue;
    }
    const size_t count = iree_hip_vmm_mapping_lower_bound(reservation, limit) -
                         iree_hip_vmm_mapping_containing(reservation, begin);
    if (!iree_host_size_checked_add(capacity, count, &capacity)) {
      return hipErrorOutOfMemory;
    }
  }
  hrx_memory_mapping_t inline_mappings[8];
  hrx_memory_mapping_t* mappings = inline_mappings;
  hipError_t result = hipSuccess;
  if (capacity > IREE_ARRAYSIZE(inline_mappings)) {
    result = iree_hip_vmm_from_iree_status(
        iree_allocator_malloc_array(iree_allocator_system(), capacity,
                                    sizeof(*mappings), (void**)&mappings));
  }
  hrx_memory_scope_t scopes[IREE_HAL_STREAMING_MAX_DEVICES + 1];
  uint32_t projections[IREE_HAL_STREAMING_MAX_DEVICES + 1];
  size_t scope_count = 0;
  size_t count = 0;
  for (size_t i = 0;
       i < reservation->access_range_count && result == hipSuccess; ++i) {
    const iree_hip_vmm_access_range_t* access = &reservation->access_ranges[i];
    const size_t begin = iree_max(offset, access->offset);
    const size_t limit = iree_min(end, access->offset + access->size);
    if (begin >= limit) {
      continue;
    }
    hrx_device_t device = reservation->device;
    const bool host = access->location_type == hipMemLocationTypeHost;
    if (!host) {
      result = iree_hip_vmm_from_hrx_status(
          hrx_gpu_device_get(access->location_id, &device));
    }
    hrx_memory_scope_t scope = {0};
    uint32_t projection = 0;
    if (result == hipSuccess) {
      result = iree_hip_vmm_from_hrx_status(hrx_device_query_memory_scope(
          device, 0, host ? HRX_MEMORY_SITE_HOST : HRX_MEMORY_SITE_PROGRAM,
          &scope));
    }
    if (result == hipSuccess) {
      result =
          iree_hip_vmm_from_hrx_status(hrx_virtual_buffer_resolve_projection(
              reservation->virtual_buffer, scope, &projection));
    }
    if (result != hipSuccess) {
      continue;
    }
    size_t scope_index = 0;
    while (scope_index < scope_count &&
           projections[scope_index] != projection) {
      ++scope_index;
    }
    if (scope_index == scope_count) {
      scopes[scope_count] = scope;
      projections[scope_count++] = projection;
    }
    size_t position = iree_hip_vmm_mapping_containing(reservation, begin);
    while (position < reservation->mapping_count &&
           reservation->mappings[position].virtual_offset < limit) {
      const iree_hip_vmm_mapping_t* binding =
          &reservation->mappings[position++];
      const size_t start = iree_max(begin, binding->virtual_offset);
      mappings[count++] = (hrx_memory_mapping_t){
          .target = reservation->virtual_buffer,
          .scope = scopes[scope_index],
          .target_offset = start,
          .byte_length =
              iree_min(limit, binding->virtual_offset + binding->size) - start,
          .operation = HRX_MEMORY_MAPPING_OPERATION_UNBIND,
          .previous_source = binding->allocation->backing,
          .previous_source_offset =
              binding->physical_offset + start - binding->virtual_offset,
      };
    }
  }
  if (result == hipSuccess) {
    qsort(mappings, count, sizeof(*mappings),
          iree_hip_vmm_compare_mapping_edits);
    size_t normalized_count = 0;
    for (size_t i = 0; i < count; ++i) {
      const hrx_memory_mapping_t current = mappings[i];
      hrx_memory_mapping_t* previous =
          normalized_count ? &mappings[normalized_count - 1] : NULL;
      if (previous && previous->scope.id == current.scope.id &&
          previous->previous_source == current.previous_source &&
          current.target_offset <=
              previous->target_offset + previous->byte_length &&
          current.previous_source_offset - previous->previous_source_offset ==
              current.target_offset - previous->target_offset) {
        previous->byte_length =
            iree_max(previous->target_offset + previous->byte_length,
                     current.target_offset + current.byte_length) -
            previous->target_offset;
      } else {
        mappings[normalized_count++] = current;
      }
    }
    result =
        iree_hip_vmm_update_mappings(reservation, normalized_count, mappings);
  }
  if (mappings != inline_mappings) {
    iree_allocator_free(iree_allocator_system(), mappings);
  }
  return result;
}

static hipError_t iree_hip_vmm_detach_failed(
    iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size) {
  if (offset != 0 || size != reservation->size) {
    return reservation->mapping_failure;
  }
  hipError_t result = iree_hip_vmm_from_hrx_status(hrx_virtual_buffer_detach(
      reservation->virtual_buffer, (hrx_memory_scope_t){0}));
  while (reservation->mapping_count && result == hipSuccess) {
    result = iree_hip_vmm_mapping_reference_remove(
        reservation->mappings[reservation->mapping_count - 1].allocation);
    if (result == hipSuccess) {
      --reservation->mapping_count;
    }
  }
  if (result == hipSuccess) {
    iree_allocator_free(iree_allocator_system(), reservation->access_ranges);
    reservation->access_ranges = NULL;
    reservation->access_range_count = 0;
  }
  return result;
}

hipError_t iree_hip_vmm_unmap(void* ptr, size_t size) {
  if (!ptr || size == 0) {
    return hipErrorInvalidValue;
  }
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) {
    return hipErrorInvalidValue;
  }
  const size_t offset = (uintptr_t)ptr - reservation->base_address;

  iree_slim_mutex_lock(&reservation->mutex);
  hipError_t result = hipSuccess;
  if (reservation->mapping_failure != hipSuccess && !reservation->retiring) {
    result = iree_hip_vmm_detach_failed(reservation, offset, size);
    iree_slim_mutex_unlock(&reservation->mutex);
    iree_hip_vmm_reservation_release(reservation);
    return result;
  }
  if (reservation->retiring ||
      !iree_hip_vmm_range_is_valid(reservation->size, offset, size,
                                   reservation->granularity) ||
      !iree_hip_vmm_mapped_range_is_complete(reservation, offset, size)) {
    result = hipErrorInvalidValue;
  }
  iree_hip_vmm_mapping_t* updated_mappings = NULL;
  size_t updated_mapping_count = 0;
  if (result == hipSuccess) {
    result = iree_hip_vmm_build_mapping_unmap(
        reservation, offset, size, &updated_mappings, &updated_mapping_count);
  }
  iree_hip_vmm_access_range_t* updated_access_ranges = NULL;
  size_t updated_access_range_count = 0;
  if (result == hipSuccess) {
    // A NULL location removes this interval from every location's access
    // ledger. Allocate the complete replacement before changing native state.
    result = iree_hip_vmm_build_access_update(
        reservation->access_ranges, reservation->access_range_count,
        /*location=*/NULL, offset, size, hipMemAccessFlagsProtNone,
        /*insert_update=*/false, &updated_access_ranges,
        &updated_access_range_count);
  }
  size_t retained_mapping_count = 0;
  for (size_t i = 0; i < updated_mapping_count && result == hipSuccess; ++i) {
    result =
        iree_hip_vmm_mapping_reference_add(updated_mappings[i].allocation,
                                           /*require_public_reference=*/false);
    if (result == hipSuccess) {
      ++retained_mapping_count;
    }
  }
  if (result == hipSuccess && !reservation->mapping_completion) {
    // Kernel arguments and device memory can contain pointers into this
    // reservation without exposing them to host-side resource tracking. Drain
    // every active stream before revoking the mapping those pointers name.
    result = iree_hip_vmm_from_iree_status(
        iree_hal_streaming_context_synchronize_all());
  }
  if (result == hipSuccess) {
    if (reservation->mapping_completion) {
      result = iree_hip_vmm_remove_mappings(reservation, offset, size);
    } else {
      result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_unmap(
          reservation->allocator, reservation->virtual_buffer, offset, size));
    }
  }
  if (result != hipSuccess) {
    for (size_t i = 0; i < retained_mapping_count; ++i) {
      const hipError_t remove_result =
          iree_hip_vmm_mapping_reference_remove(updated_mappings[i].allocation);
      if (remove_result != hipSuccess) {
        result = remove_result;
      }
    }
  } else {
    iree_hip_vmm_mapping_t* old_mappings = reservation->mappings;
    const size_t old_mapping_count = reservation->mapping_count;
    reservation->mappings = updated_mappings;
    reservation->mapping_count = updated_mapping_count;
    reservation->mapping_capacity = updated_mapping_count;
    updated_mappings = NULL;

    iree_allocator_free(iree_allocator_system(), reservation->access_ranges);
    reservation->access_ranges = updated_access_ranges;
    reservation->access_range_count = updated_access_range_count;
    updated_access_ranges = NULL;

    for (size_t i = 0; i < old_mapping_count; ++i) {
      hipError_t remove_result =
          iree_hip_vmm_mapping_reference_remove(old_mappings[i].allocation);
      if (result == hipSuccess) {
        result = remove_result;
      }
    }
    iree_allocator_free(iree_allocator_system(), old_mappings);
  }
  iree_allocator_free(iree_allocator_system(), updated_access_ranges);
  iree_allocator_free(iree_allocator_system(), updated_mappings);
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

typedef struct iree_hip_vmm_access_projection_t {
  // Concrete location in the sealed group's memory namespace.
  hrx_memory_scope_t scope;
  // Native page-table identity shared by logically distinct access locations.
  uint32_t projection;
  // Final requested permissions for this location.
  hrx_memory_access_t access;
} iree_hip_vmm_access_projection_t;

static hipError_t iree_hip_vmm_set_access_modern(
    iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size,
    const hipMemAccessDesc* descriptors, size_t count) {
  iree_hip_vmm_access_projection_t plans[IREE_HAL_STREAMING_MAX_DEVICES + 1];
  size_t plan_count = 0;
  iree_hip_vmm_access_range_t* updated = NULL;
  size_t updated_count = 0;
  const iree_hip_vmm_access_range_t* previous = reservation->access_ranges;
  size_t previous_count = reservation->access_range_count;
  hipError_t result = hipSuccess;
  for (size_t i = 0; i < count && result == hipSuccess; ++i) {
    const hipMemAccessDesc* descriptor = &descriptors[i];
    iree_hip_vmm_access_range_t* next = NULL;
    size_t next_count = 0;
    result = iree_hip_vmm_build_access_update(
        previous, previous_count, &descriptor->location, offset, size,
        descriptor->flags, true, &next, &next_count);
    if (result == hipSuccess) {
      iree_allocator_free(iree_allocator_system(), updated);
      updated = next;
      updated_count = next_count;
      previous = updated;
      previous_count = updated_count;
    }
    hrx_device_t device = reservation->device;
    const bool host = descriptor->location.type == hipMemLocationTypeHost;
    if (result == hipSuccess && !host) {
      result = iree_hip_vmm_from_hrx_status(
          hrx_gpu_device_get(descriptor->location.id, &device));
    }
    hrx_memory_scope_t scope = {0};
    if (result == hipSuccess) {
      result = iree_hip_vmm_from_hrx_status(hrx_device_query_memory_scope(
          device, 0, host ? HRX_MEMORY_SITE_HOST : HRX_MEMORY_SITE_PROGRAM,
          &scope));
    }
    uint32_t projection = 0;
    if (result == hipSuccess) {
      result =
          iree_hip_vmm_from_hrx_status(hrx_virtual_buffer_resolve_projection(
              reservation->virtual_buffer, scope, &projection));
    }
    if (result == hipSuccess) {
      const iree_hal_buffer_memory_view_t memory =
          iree_hal_buffer_memory_view(reservation->streaming_buffer->buffer);
      const iree_hal_buffer_interface_t type =
          host ? IREE_HAL_BUFFER_INTERFACE_HOST
               : IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
      const iree_hal_buffer_native_binding_slot_t slot = {
          .index = memory.contract->scopes[scope.id].bindings[type],
          .type = type,
      };
      const iree_hal_buffer_native_binding_t binding =
          iree_hal_buffer_memory_native_binding(&memory, slot);
      const uintptr_t address = host ? (uintptr_t)binding.host_pointer
                                     : (uintptr_t)binding.device_address;
      // HIP exposes one pointer, including values embedded in user bytes.
      // A valid HAL projection at another address cannot satisfy that ABI.
      if (descriptor->flags != hipMemAccessFlagsProtNone &&
          address != reservation->base_address) {
        result = hipErrorNotSupported;
      }
    }
    if (result == hipSuccess) {
      size_t position = 0;
      while (position < plan_count && plans[position].scope.id != scope.id) {
        ++position;
      }
      if (position == plan_count) {
        ++plan_count;
      }
      plans[position] = (iree_hip_vmm_access_projection_t){
          .scope = scope,
          .projection = projection,
          .access = descriptor->flags == hipMemAccessFlagsProtReadWrite
                        ? HRX_MEMORY_ACCESS_ALL
                    : descriptor->flags == hipMemAccessFlagsProtRead
                        ? HRX_MEMORY_ACCESS_READ
                        : HRX_MEMORY_ACCESS_NONE,
      };
    }
  }
  size_t capacity = 0;
  if (result == hipSuccess &&
      !iree_host_size_checked_mul(plan_count, reservation->mapping_count,
                                  &capacity)) {
    result = hipErrorOutOfMemory;
  }
  hrx_memory_mapping_t inline_mappings[8];
  hrx_memory_mapping_t* mappings = inline_mappings;
  if (result == hipSuccess && capacity > IREE_ARRAYSIZE(inline_mappings)) {
    result = iree_hip_vmm_from_iree_status(
        iree_allocator_malloc_array(iree_allocator_system(), capacity,
                                    sizeof(*mappings), (void**)&mappings));
  }
  size_t mapping_count = 0;
  for (size_t i = 0; i < plan_count && result == hipSuccess; ++i) {
    bool shared = false;
    for (size_t j = 0; j < i; ++j) {
      if (plans[j].projection == plans[i].projection) {
        shared = true;
        if (plans[j].access != plans[i].access) {
          result = hipErrorNotSupported;
        }
      }
    }
    if (shared || result != hipSuccess) {
      continue;
    }
    const size_t end = offset + size;
    size_t position = iree_hip_vmm_mapping_containing(reservation, offset);
    while (position < reservation->mapping_count &&
           reservation->mappings[position].virtual_offset < end) {
      const iree_hip_vmm_mapping_t* binding =
          &reservation->mappings[position++];
      const size_t begin = iree_max(offset, binding->virtual_offset);
      const size_t length =
          iree_min(end, binding->virtual_offset + binding->size) - begin;
      mappings[mapping_count++] = (hrx_memory_mapping_t){
          .target = reservation->virtual_buffer,
          .scope = plans[i].scope,
          .target_offset = begin,
          .byte_length = length,
          .source = binding->allocation->backing,
          .source_offset =
              binding->physical_offset + begin - binding->virtual_offset,
          .access = plans[i].access,
          .operation = HRX_MEMORY_MAPPING_OPERATION_PROTECT,
      };
    }
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_update_mappings(reservation, mapping_count, mappings);
  }
  if (result == hipSuccess) {
    iree_allocator_free(iree_allocator_system(), reservation->access_ranges);
    reservation->access_ranges = updated;
    reservation->access_range_count = updated_count;
    updated = NULL;
  }
  if (mappings != inline_mappings) {
    iree_allocator_free(iree_allocator_system(), mappings);
  }
  iree_allocator_free(iree_allocator_system(), updated);
  return result;
}

hipError_t iree_hip_vmm_set_access(void* ptr, size_t size,
                                   const hipMemAccessDesc* descriptors,
                                   size_t count) {
  if (!ptr || size == 0 || !descriptors || count == 0) {
    return hipErrorInvalidValue;
  }
  for (size_t i = 0; i < count; ++i) {
    hipError_t result =
        iree_hip_vmm_validate_access_descriptor(&descriptors[i]);
    if (result != hipSuccess) {
      return result;
    }
  }

  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) {
    return hipErrorInvalidValue;
  }
  const size_t offset = (uintptr_t)ptr - reservation->base_address;
  iree_slim_mutex_lock(&reservation->mutex);
  hipError_t result = hipSuccess;
  if (reservation->mapping_failure != hipSuccess) {
    result = reservation->mapping_failure;
  } else if (reservation->retiring ||
             !iree_hip_vmm_range_is_valid(reservation->size, offset, size,
                                          reservation->granularity) ||
             !iree_hip_vmm_mapped_range_is_complete(reservation, offset,
                                                    size)) {
    result = hipErrorInvalidValue;
  }
  if (result == hipSuccess && reservation->mapping_completion) {
    result = iree_hip_vmm_set_access_modern(reservation, offset, size,
                                            descriptors, count);
  }
  for (size_t i = 0;
       i < count && result == hipSuccess && !reservation->mapping_completion;
       ++i) {
    iree_hip_vmm_access_range_t* updated_ranges = NULL;
    size_t updated_count = 0;
    result = iree_hip_vmm_build_access_update(
        reservation->access_ranges, reservation->access_range_count,
        &descriptors[i].location, offset, size, descriptors[i].flags,
        /*insert_update=*/true, &updated_ranges, &updated_count);
    if (result != hipSuccess) {
      break;
    }
    const hrx_virtual_memory_access_scope_t scope =
        descriptors[i].location.type == hipMemLocationTypeHost
            ? HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_HOST
            : HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE;
    hrx_device_t access_device = reservation->device;
    if (scope == HRX_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE) {
      result = iree_hip_vmm_from_hrx_status(
          hrx_gpu_device_get(descriptors[i].location.id, &access_device));
      if (result != hipSuccess) {
        iree_allocator_free(iree_allocator_system(), updated_ranges);
        break;
      }
    }
    result = iree_hip_vmm_from_hrx_status(
        hrx_allocator_virtual_memory_protect_scoped(
            hrx_device_allocator(access_device), reservation->virtual_buffer,
            offset, size, scope,
            iree_hip_vmm_protection(descriptors[i].flags)));
    if (result == hipSuccess) {
      iree_allocator_free(iree_allocator_system(), reservation->access_ranges);
      reservation->access_ranges = updated_ranges;
      reservation->access_range_count = updated_count;
    } else {
      iree_allocator_free(iree_allocator_system(), updated_ranges);
    }
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

hipError_t iree_hip_vmm_get_access(unsigned long long* flags,
                                   const hipMemLocation* location, void* ptr) {
  if (!flags || !ptr) {
    return hipErrorInvalidValue;
  }
  hipError_t result = iree_hip_vmm_validate_access_location(location);
  if (result != hipSuccess) {
    return result;
  }
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) {
    return hipErrorInvalidValue;
  }
  const size_t offset = (uintptr_t)ptr - reservation->base_address;

  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->mapping_failure != hipSuccess) {
    result = reservation->mapping_failure;
  } else if (reservation->retiring ||
             iree_hip_vmm_mapping_containing(reservation, offset) == SIZE_MAX) {
    result = hipErrorInvalidValue;
  } else {
    *flags = hipMemAccessFlagsProtNone;
    for (size_t i = 0; i < reservation->access_range_count; ++i) {
      const iree_hip_vmm_access_range_t* range = &reservation->access_ranges[i];
      if (iree_hip_vmm_access_location_matches(range, location) &&
          offset >= range->offset && offset - range->offset < range->size) {
        *flags = range->flags;
        break;
      }
    }
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}

hipError_t iree_hip_vmm_get_allocation_granularity(
    size_t* granularity, const hipMemAllocationProp* properties,
    hipMemAllocationGranularity_flags option) {
  if (!granularity) {
    return hipErrorInvalidValue;
  }
  hipError_t result = iree_hip_vmm_validate_properties(properties);
  if (result == hipErrorInvalidDevice) {
    result = hipErrorInvalidValue;
  }
  if (result != hipSuccess) {
    return result;
  }
  if (option != hipMemAllocationGranularityMinimum &&
      option != hipMemAllocationGranularityRecommended) {
    return hipErrorInvalidValue;
  }
  int device_ordinal = 0;
  hrx_device_t device = NULL;
  result = iree_hip_vmm_get_device_for_properties(properties, &device_ordinal,
                                                  &device);
  if (result != hipSuccess) {
    return result;
  }
  size_t minimum = 0;
  size_t recommended = 0;
  bool modern = false;
  result = iree_hip_vmm_query_granularity(device,
                                          iree_hip_vmm_memory_type(properties),
                                          &minimum, &recommended, &modern);
  if (result == hipSuccess) {
    *granularity =
        option == hipMemAllocationGranularityMinimum ? minimum : recommended;
  }
  return result;
}

hipError_t iree_hip_vmm_get_allocation_properties(
    hipMemAllocationProp* properties, hipMemGenericAllocationHandle_t handle) {
  if (!properties || !handle) {
    return hipErrorInvalidValue;
  }
  iree_hip_vmm_allocation_t* allocation =
      iree_hip_vmm_lookup_allocation(handle);
  if (!allocation) {
    return hipErrorInvalidValue;
  }
  iree_slim_mutex_lock(&allocation->mutex);
  hipError_t result = hipSuccess;
  if (allocation->retiring || allocation->public_reference_count == 0) {
    result = hipErrorInvalidValue;
  } else {
    *properties = allocation->properties;
  }
  iree_slim_mutex_unlock(&allocation->mutex);
  iree_hip_vmm_allocation_release(allocation);
  return result;
}

hipError_t iree_hip_vmm_retain_allocation_handle(
    hipMemGenericAllocationHandle_t* handle, void* address) {
  if (!handle || !address) {
    return hipErrorInvalidValue;
  }
  *handle = NULL;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)address,
                                      /*require_base=*/false);
  if (!reservation) {
    return hipErrorInvalidValue;
  }
  const size_t offset = (uintptr_t)address - reservation->base_address;
  iree_slim_mutex_lock(&reservation->mutex);
  const size_t mapping_position =
      iree_hip_vmm_mapping_containing(reservation, offset);
  hipError_t result = hipSuccess;
  if (reservation->retiring || mapping_position == SIZE_MAX) {
    result = hipErrorInvalidValue;
  } else {
    iree_hip_vmm_allocation_t* allocation =
        reservation->mappings[mapping_position].allocation;
    iree_slim_mutex_lock(&allocation->mutex);
    if (allocation->retiring ||
        allocation->public_reference_count == UINT64_MAX) {
      result = hipErrorInvalidValue;
    } else {
      ++allocation->public_reference_count;
      *handle = (hipMemGenericAllocationHandle_t)allocation->handle_key;
    }
    iree_slim_mutex_unlock(&allocation->mutex);
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
  return result;
}
