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
#include "common/memory.h"
#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"

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
  if (hrx_status_is_ok(status)) return hipSuccess;
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

static hipError_t iree_hip_vmm_grow_pointer_array(void** values,
                                                  size_t* capacity,
                                                  size_t minimum_capacity) {
  if (*capacity >= minimum_capacity) return hipSuccess;
  size_t new_capacity = *capacity ? *capacity : 16;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > SIZE_MAX / 2) return hipErrorOutOfMemory;
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
  if (!reservation) return;
  iree_atomic_ref_count_inc(&reservation->ref_count);
}

static void iree_hip_vmm_reservation_release(
    iree_hip_vmm_reservation_t* reservation) {
  if (!reservation) return;
  if (iree_atomic_ref_count_dec(&reservation->ref_count) != 1) return;
  iree_allocator_free(iree_allocator_system(), reservation->access_ranges);
  iree_allocator_free(iree_allocator_system(), reservation->mappings);
  iree_slim_mutex_deinitialize(&reservation->mutex);
  hrx_device_release(reservation->device);
  iree_allocator_free(iree_allocator_system(), reservation);
}

static void iree_hip_vmm_allocation_retain(
    iree_hip_vmm_allocation_t* allocation) {
  if (!allocation) return;
  iree_atomic_ref_count_inc(&allocation->ref_count);
}

static void iree_hip_vmm_allocation_release(
    iree_hip_vmm_allocation_t* allocation) {
  if (!allocation) return;
  if (iree_atomic_ref_count_dec(&allocation->ref_count) != 1) return;
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
  if (result != hipSuccess) return result;
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
  if (reservation) iree_hip_vmm_reservation_retain(reservation);
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
  if (result != hipSuccess) return result;
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
  if (result != hipSuccess) return result;
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
    return HRX_MEMORY_TYPE_DEVICE_LOCAL | HRX_MEMORY_TYPE_HOST_VISIBLE |
           HRX_MEMORY_TYPE_HOST_COHERENT | cache_type;
  }
  return HRX_MEMORY_TYPE_DEVICE_LOCAL | cache_type;
}

static hipError_t iree_hip_vmm_get_device_for_properties(
    const hipMemAllocationProp* properties, int* out_device_ordinal,
    hrx_device_t* out_device) {
  int device_ordinal = properties->location.id;
  if (properties->location.type == hipMemLocationTypeHost) {
    hipError_t result = hipGetDevice(&device_ordinal);
    if (result != hipSuccess) return result;
  }
  hrx_device_t device = NULL;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) return result;
  *out_device_ordinal = device_ordinal;
  *out_device = device;
  return hipSuccess;
}

static hipError_t iree_hip_vmm_query_granularity(hrx_device_t device,
                                                 hrx_memory_type_t memory_type,
                                                 size_t* out_minimum,
                                                 size_t* out_recommended) {
  bool supported = false;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_allocator_query_virtual_memory(
          hrx_device_allocator(device), memory_type, &supported, out_minimum,
          out_recommended));
  if (result != hipSuccess) return result;
  return supported && *out_minimum != 0 ? hipSuccess : hipErrorNotSupported;
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
  if (position == 0) return SIZE_MAX;
  const iree_hip_vmm_mapping_t* candidate =
      &reservation->mappings[position - 1];
  return offset - candidate->virtual_offset < candidate->size ? position - 1
                                                              : SIZE_MAX;
}

static bool iree_hip_vmm_mapped_range_is_complete(
    const iree_hip_vmm_reservation_t* reservation, size_t offset, size_t size) {
  size_t position = iree_hip_vmm_mapping_containing(reservation, offset);
  if (position == SIZE_MAX) return false;
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
  if (position == 0) return false;
  const iree_hip_vmm_mapping_t* previous = &reservation->mappings[position - 1];
  return offset < previous->virtual_offset + previous->size;
}

static hipError_t iree_hip_vmm_reserve_mapping_capacity(
    iree_hip_vmm_reservation_t* reservation, size_t minimum_capacity) {
  if (reservation->mapping_capacity >= minimum_capacity) return hipSuccess;
  size_t new_capacity =
      reservation->mapping_capacity ? reservation->mapping_capacity : 8;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > SIZE_MAX / 2) return hipErrorOutOfMemory;
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
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  const bool removed =
      iree_hip_vmm_registry_remove_allocation_locked(registry, allocation);
  iree_hip_vmm_registry_unlock();
  if (!removed) return hipErrorInvalidValue;

  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_allocator_physical_memory_free(
          allocation->allocator, allocation->physical_memory));
  if (result == hipSuccess) {
    allocation->physical_memory = NULL;
    iree_hip_vmm_allocation_release(allocation);  // Registry reference.
    return hipSuccess;
  }

  registry = iree_hip_vmm_registry_lock();
  const hipError_t insert_result =
      iree_hip_vmm_registry_insert_allocation_locked(registry, allocation);
  iree_hip_vmm_registry_unlock();
  iree_slim_mutex_lock(&allocation->mutex);
  allocation->retiring = false;
  if (restore_public_on_failure) ++allocation->public_reference_count;
  iree_slim_mutex_unlock(&allocation->mutex);
  return insert_result == hipSuccess ? result : insert_result;
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
  if (a->offset == b->offset) return 0;
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
    const iree_hip_vmm_reservation_t* reservation,
    const hipMemLocation* location, size_t offset, size_t size,
    hipMemAccessFlags flags, bool insert_update,
    iree_hip_vmm_access_range_t** out_ranges, size_t* out_count) {
  *out_ranges = NULL;
  *out_count = 0;
  size_t maximum_count = 0;
  if (!iree_host_size_checked_mul(reservation->access_range_count, 2,
                                  &maximum_count) ||
      !iree_host_size_checked_add(maximum_count, insert_update ? 1 : 0,
                                  &maximum_count)) {
    return hipErrorOutOfMemory;
  }
  if (maximum_count == 0) return hipSuccess;

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
  for (size_t i = 0; i < reservation->access_range_count; ++i) {
    const iree_hip_vmm_access_range_t* old = &reservation->access_ranges[i];
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
  if (!location) return hipErrorInvalidValue;
  if (location->type == hipMemLocationTypeHost) {
    return hipSuccess;
  }
  if (location->type != hipMemLocationTypeDevice) {
    return hipErrorInvalidValue;
  }
  int device_count = 0;
  hipError_t result = hipGetDeviceCount(&device_count);
  if (result != hipSuccess) return result;
  return location->id >= 0 && location->id < device_count
             ? hipSuccess
             : hipErrorInvalidValue;
}

static hipError_t iree_hip_vmm_validate_access_descriptor(
    const hipMemAccessDesc* descriptor) {
  hipError_t result =
      iree_hip_vmm_validate_access_location(&descriptor->location);
  if (result != hipSuccess) return result;
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
  if (!out_supported) return hipErrorInvalidValue;
  *out_supported = false;
  hrx_device_t device = NULL;
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) return result;
  return iree_hip_vmm_from_hrx_status(hrx_allocator_query_virtual_memory(
      hrx_device_allocator(device), HRX_MEMORY_TYPE_DEVICE_LOCAL, out_supported,
      /*min_page_size=*/NULL,
      /*recommended_page_size=*/NULL));
}

hipError_t iree_hip_vmm_address_reserve(void** ptr, size_t size,
                                        size_t alignment, void* address,
                                        unsigned long long flags) {
  if (!ptr || flags != 0 || address) return hipErrorInvalidValue;
  *ptr = NULL;
  if (alignment != 0 && !iree_hip_vmm_is_power_of_two(alignment)) {
    return hipErrorInvalidValue;
  }

  int device_ordinal = 0;
  hipError_t result = hipGetDevice(&device_ordinal);
  if (result != hipSuccess) return result;
  hrx_device_t device = NULL;
  result =
      iree_hip_vmm_from_hrx_status(hrx_gpu_device_get(device_ordinal, &device));
  if (result != hipSuccess) return result;
  size_t granularity = 0;
  size_t recommended = 0;
  result = iree_hip_vmm_query_granularity(device, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                          &granularity, &recommended);
  if (result != hipSuccess ||
      !iree_hip_vmm_range_is_valid(size, 0, size, granularity)) {
    return result == hipSuccess ? hipErrorInvalidValue : result;
  }

  hrx_allocator_t allocator = hrx_device_allocator(device);
  hrx_buffer_t virtual_buffer = NULL;
  result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_reserve(
      allocator, /*affinity=*/0, size, &virtual_buffer));
  if (result != hipSuccess) return result;
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

    iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
    result =
        iree_hip_vmm_registry_insert_reservation_locked(registry, reservation);
    iree_hip_vmm_registry_unlock();
  }
  if (result != hipSuccess) {
    iree_hal_streaming_memory_release_wrapped_buffer(streaming_buffer);
    if (virtual_buffer) {
      (void)iree_hip_vmm_from_hrx_status(
          hrx_allocator_virtual_memory_release(allocator, virtual_buffer));
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
  if (!device_ptr || size == 0) return hipErrorInvalidValue;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)device_ptr,
                                      /*require_base=*/true);
  if (!reservation) return hipErrorInvalidValue;

  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->retiring || reservation->size != size ||
      reservation->mapping_count != 0) {
    iree_slim_mutex_unlock(&reservation->mutex);
    iree_hip_vmm_reservation_release(reservation);
    return hipErrorInvalidValue;
  }
  reservation->retiring = true;
  iree_hip_vmm_registry_t* registry = iree_hip_vmm_registry_lock();
  const bool removed =
      iree_hip_vmm_registry_remove_reservation_locked(registry, reservation);
  iree_hip_vmm_registry_unlock();
  if (!removed) {
    reservation->retiring = false;
    iree_slim_mutex_unlock(&reservation->mutex);
    iree_hip_vmm_reservation_release(reservation);
    return hipErrorInvalidValue;
  }

  iree_hal_streaming_memory_prepare_virtual_reservation_release(
      reservation->streaming_buffer);
  hipError_t result =
      iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_release(
          reservation->allocator, reservation->virtual_buffer));
  if (result == hipSuccess) {
    reservation->virtual_buffer = NULL;
    iree_hal_streaming_memory_release_wrapped_buffer(
        reservation->streaming_buffer);
    reservation->streaming_buffer = NULL;
  } else {
    iree_hal_streaming_memory_restore_virtual_reservation(
        reservation->streaming_buffer, reservation->virtual_buffer);
    registry = iree_hip_vmm_registry_lock();
    const hipError_t insert_result =
        iree_hip_vmm_registry_insert_reservation_locked(registry, reservation);
    iree_hip_vmm_registry_unlock();
    reservation->retiring = false;
    if (insert_result != hipSuccess) result = insert_result;
  }
  iree_slim_mutex_unlock(&reservation->mutex);
  if (removed && result == hipSuccess) {
    iree_hip_vmm_reservation_release(reservation);  // Registry reference.
  }
  iree_hip_vmm_reservation_release(reservation);  // Operation reference.
  return result;
}

hipError_t iree_hip_vmm_create(hipMemGenericAllocationHandle_t* handle,
                               size_t size,
                               const hipMemAllocationProp* properties,
                               unsigned long long flags) {
  if (!handle || size == 0 || flags != 0) return hipErrorInvalidValue;
  *handle = NULL;
  hipError_t result = iree_hip_vmm_validate_properties(properties);
  if (result != hipSuccess) return result;
  if (properties->requestedHandleType != hipMemHandleTypeNone) {
    return hipErrorNotSupported;
  }

  int device_ordinal = 0;
  hrx_device_t device = NULL;
  result = iree_hip_vmm_get_device_for_properties(properties, &device_ordinal,
                                                  &device);
  if (result != hipSuccess) return result;
  const hrx_memory_type_t memory_type = iree_hip_vmm_memory_type(properties);
  size_t granularity = 0;
  size_t recommended = 0;
  result = iree_hip_vmm_query_granularity(device, memory_type, &granularity,
                                          &recommended);
  if (result != hipSuccess ||
      !iree_hip_vmm_range_is_valid(size, 0, size, granularity)) {
    return result == hipSuccess ? hipErrorInvalidValue : result;
  }

  hrx_allocator_t allocator = hrx_device_allocator(device);
  hrx_physical_memory_t physical_memory = NULL;
  result = iree_hip_vmm_from_hrx_status(hrx_allocator_physical_memory_allocate(
      allocator, memory_type, size, &physical_memory));
  if (result != hipSuccess) return result;

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
    (void)iree_hip_vmm_from_hrx_status(
        hrx_allocator_physical_memory_free(allocator, physical_memory));
    if (allocation) {
      allocation->physical_memory = NULL;
      iree_hip_vmm_allocation_release(allocation);
    }
    return result;
  }
  *handle = (hipMemGenericAllocationHandle_t)allocation->handle_key;
  return hipSuccess;
}

hipError_t iree_hip_vmm_release(hipMemGenericAllocationHandle_t handle) {
  if (!handle) return hipErrorInvalidValue;
  iree_hip_vmm_allocation_t* allocation =
      iree_hip_vmm_lookup_allocation(handle);
  if (!allocation) return hipErrorInvalidValue;

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
  if (reservation->retiring ||
      !iree_hip_vmm_range_is_valid(reservation->size, virtual_offset, size,
                                   reservation->granularity) ||
      offset > allocation->size || size > allocation->size - offset ||
      reservation->device_ordinal != allocation->device_ordinal ||
      iree_hip_vmm_mapping_range_overlaps(reservation, virtual_offset, size)) {
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
    result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_map(
        reservation->allocator, reservation->virtual_buffer, virtual_offset,
        allocation->physical_memory, offset, size));
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

hipError_t iree_hip_vmm_unmap(void* ptr, size_t size) {
  if (!ptr || size == 0) return hipErrorInvalidValue;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;
  const size_t offset = (uintptr_t)ptr - reservation->base_address;

  iree_slim_mutex_lock(&reservation->mutex);
  hipError_t result = hipSuccess;
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
        reservation, /*location=*/NULL, offset, size, hipMemAccessFlagsProtNone,
        /*insert_update=*/false, &updated_access_ranges,
        &updated_access_range_count);
  }
  size_t retained_mapping_count = 0;
  for (size_t i = 0; i < updated_mapping_count && result == hipSuccess; ++i) {
    result =
        iree_hip_vmm_mapping_reference_add(updated_mappings[i].allocation,
                                           /*require_public_reference=*/false);
    if (result == hipSuccess) ++retained_mapping_count;
  }
  if (result == hipSuccess) {
    result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_unmap(
        reservation->allocator, reservation->virtual_buffer, offset, size));
  }
  if (result != hipSuccess) {
    for (size_t i = 0; i < retained_mapping_count; ++i) {
      const hipError_t remove_result =
          iree_hip_vmm_mapping_reference_remove(updated_mappings[i].allocation);
      if (remove_result != hipSuccess) result = remove_result;
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
      if (result == hipSuccess) result = remove_result;
    }
    iree_allocator_free(iree_allocator_system(), old_mappings);
  }
  iree_allocator_free(iree_allocator_system(), updated_access_ranges);
  iree_allocator_free(iree_allocator_system(), updated_mappings);
  iree_slim_mutex_unlock(&reservation->mutex);
  iree_hip_vmm_reservation_release(reservation);
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
    if (result != hipSuccess) return result;
  }

  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;
  const size_t offset = (uintptr_t)ptr - reservation->base_address;
  iree_slim_mutex_lock(&reservation->mutex);
  hipError_t result = hipSuccess;
  if (reservation->retiring ||
      !iree_hip_vmm_range_is_valid(reservation->size, offset, size,
                                   reservation->granularity) ||
      !iree_hip_vmm_mapped_range_is_complete(reservation, offset, size)) {
    result = hipErrorInvalidValue;
  }
  for (size_t i = 0; i < count && result == hipSuccess; ++i) {
    iree_hip_vmm_access_range_t* updated_ranges = NULL;
    size_t updated_count = 0;
    result = iree_hip_vmm_build_access_update(
        reservation, &descriptors[i].location, offset, size,
        descriptors[i].flags, /*insert_update=*/true, &updated_ranges,
        &updated_count);
    if (result != hipSuccess) break;
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
    result = iree_hip_vmm_from_hrx_status(hrx_allocator_virtual_memory_protect(
        hrx_device_allocator(access_device), reservation->virtual_buffer,
        offset, size, scope, iree_hip_vmm_protection(descriptors[i].flags)));
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
  if (!flags || !ptr) return hipErrorInvalidValue;
  hipError_t result = iree_hip_vmm_validate_access_location(location);
  if (result != hipSuccess) return result;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)ptr,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;
  const size_t offset = (uintptr_t)ptr - reservation->base_address;

  iree_slim_mutex_lock(&reservation->mutex);
  if (reservation->retiring ||
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
  if (!granularity) return hipErrorInvalidValue;
  hipError_t result = iree_hip_vmm_validate_properties(properties);
  if (result == hipErrorInvalidDevice) result = hipErrorInvalidValue;
  if (result != hipSuccess) return result;
  if (option != hipMemAllocationGranularityMinimum &&
      option != hipMemAllocationGranularityRecommended) {
    return hipErrorInvalidValue;
  }
  int device_ordinal = 0;
  hrx_device_t device = NULL;
  result = iree_hip_vmm_get_device_for_properties(properties, &device_ordinal,
                                                  &device);
  if (result != hipSuccess) return result;
  size_t minimum = 0;
  size_t recommended = 0;
  result = iree_hip_vmm_query_granularity(
      device, iree_hip_vmm_memory_type(properties), &minimum, &recommended);
  if (result == hipSuccess) {
    *granularity =
        option == hipMemAllocationGranularityMinimum ? minimum : recommended;
  }
  return result;
}

hipError_t iree_hip_vmm_get_allocation_properties(
    hipMemAllocationProp* properties, hipMemGenericAllocationHandle_t handle) {
  if (!properties || !handle) return hipErrorInvalidValue;
  iree_hip_vmm_allocation_t* allocation =
      iree_hip_vmm_lookup_allocation(handle);
  if (!allocation) return hipErrorInvalidValue;
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
  if (!handle || !address) return hipErrorInvalidValue;
  *handle = NULL;
  iree_hip_vmm_reservation_t* reservation =
      iree_hip_vmm_lookup_reservation((uintptr_t)address,
                                      /*require_base=*/false);
  if (!reservation) return hipErrorInvalidValue;
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
