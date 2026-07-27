// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/virtual_memory.h"

#include <stdint.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdgpu/access_policy.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

// ROCr maps and protects each VMM handle as one range. The HAL VMM API can
// express subranges, but accepting them here would imply support ROCr does not
// provide. Callers that need suballocation must allocate page-sized physical
// handles and map them individually.

typedef struct iree_hal_amdgpu_virtual_memory_reservation_t {
  // Next reservation in the allocator-owned registry.
  struct iree_hal_amdgpu_virtual_memory_reservation_t* next;

  // VMM state that owns this reservation and its registry entry.
  iree_hal_amdgpu_virtual_memory_state_t* state;

  // HAL buffer exposing the reserved virtual address range.
  iree_hal_buffer_t* virtual_buffer;

  // Base address returned by ROCr for the virtual reservation.
  void* base_ptr;

  // Byte length of the reservation.
  iree_device_size_t size;

  // Required alignment for VMM offsets and mapping sizes.
  iree_device_size_t allocation_granule;

  // Queues permitted to receive access permissions for this reservation.
  iree_hal_queue_affinity_t queue_affinity;
} iree_hal_amdgpu_virtual_memory_reservation_t;

typedef struct iree_hal_amdgpu_virtual_memory_mapping_t {
  // Next mapping in the allocator-owned registry.
  struct iree_hal_amdgpu_virtual_memory_mapping_t* next;

  // Reservation whose address range contains this mapping.
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation;

  // Physical allocation mapped into the virtual address range.
  iree_hal_physical_memory_t* physical_memory;

  // Byte offset in |reservation| where the mapping begins.
  iree_device_size_t virtual_offset;

  // Byte length of the mapping and its physical allocation.
  iree_device_size_t size;
} iree_hal_amdgpu_virtual_memory_mapping_t;

struct iree_hal_physical_memory_t {
  // Host allocator used to release this physical-memory wrapper.
  iree_allocator_t host_allocator;

  // VMM state that owns synchronization for this allocation.
  iree_hal_amdgpu_virtual_memory_state_t* state;

  // HSA VMM handle representing the physical allocation.
  hsa_amd_vmem_alloc_handle_t allocation_handle;

  // Byte length of the physical allocation.
  iree_device_size_t allocation_size;

  // Required alignment for the physical allocation size.
  iree_device_size_t allocation_granule;

  // Total bytes currently mapped from this physical allocation.
  iree_device_size_t mapped_size;
};

struct iree_hal_amdgpu_virtual_memory_state_t {
  // Unowned HSA dynamic-library table used for VMM calls.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Unowned topology used to grant access to selected queue agents.
  const iree_hal_amdgpu_topology_t* topology;

  // Host allocator used for VMM bookkeeping.
  iree_allocator_t host_allocator;

  // Serializes reservation, mapping, and physical-memory ownership changes.
  iree_slim_mutex_t mutex;

  // Registry of live virtual-address reservations.
  iree_hal_amdgpu_virtual_memory_reservation_t* reservations;

  // Registry of live physical-to-virtual mappings.
  iree_hal_amdgpu_virtual_memory_mapping_t* mappings;

  // Number of physical-memory handles that have not been freed.
  iree_host_size_t physical_memory_count;
};

static bool iree_hal_amdgpu_virtual_memory_is_aligned(
    iree_device_size_t value, iree_device_size_t alignment) {
  return alignment != 0 && value % alignment == 0;
}

static bool iree_hal_amdgpu_virtual_memory_range_is_valid(
    iree_device_size_t total_size, iree_device_size_t offset,
    iree_device_size_t size, iree_device_size_t alignment) {
  return size != 0 &&
         iree_hal_amdgpu_virtual_memory_is_aligned(offset, alignment) &&
         iree_hal_amdgpu_virtual_memory_is_aligned(size, alignment) &&
         offset <= total_size && size <= total_size - offset;
}

static iree_status_t iree_hal_amdgpu_virtual_memory_validate_placement(
    iree_hal_amdgpu_virtual_memory_placement_t placement) {
  if (IREE_UNLIKELY(!placement.memory_pool.handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement has no HSA memory pool");
  }
  if (IREE_UNLIKELY(!placement.device)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement has no HAL device");
  }
  if (IREE_UNLIKELY(
          !iree_device_size_is_valid_alignment(placement.allocation_granule))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement has invalid allocation "
                            "granule %" PRIu64,
                            (uint64_t)placement.allocation_granule);
  }
  if (IREE_UNLIKELY(placement.max_allocation_size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement has no allocation limit");
  }
  return iree_ok_status();
}

static iree_hal_amdgpu_virtual_memory_reservation_t*
iree_hal_amdgpu_virtual_memory_find_reservation_locked(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer) {
  for (iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
           state->reservations;
       reservation; reservation = reservation->next) {
    if (reservation->virtual_buffer == virtual_buffer) return reservation;
  }
  return NULL;
}

static bool iree_hal_amdgpu_virtual_memory_has_mapping_locked(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_reservation_t* reservation) {
  for (iree_hal_amdgpu_virtual_memory_mapping_t* mapping = state->mappings;
       mapping; mapping = mapping->next) {
    if (mapping->reservation == reservation) return true;
  }
  return false;
}

static iree_hal_amdgpu_virtual_memory_mapping_t*
iree_hal_amdgpu_virtual_memory_find_exact_mapping_locked(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_reservation_t* reservation,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  for (iree_hal_amdgpu_virtual_memory_mapping_t* mapping = state->mappings;
       mapping; mapping = mapping->next) {
    if (mapping->reservation == reservation &&
        mapping->virtual_offset == virtual_offset && mapping->size == size) {
      return mapping;
    }
  }
  return NULL;
}

static bool iree_hal_amdgpu_virtual_memory_ranges_overlap(
    iree_device_size_t lhs_offset, iree_device_size_t lhs_size,
    iree_device_size_t rhs_offset, iree_device_size_t rhs_size) {
  const iree_device_size_t lhs_end = lhs_offset + lhs_size;
  const iree_device_size_t rhs_end = rhs_offset + rhs_size;
  return lhs_offset < rhs_end && rhs_offset < lhs_end;
}

static bool iree_hal_amdgpu_virtual_memory_has_overlapping_mapping_locked(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_reservation_t* reservation,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  for (iree_hal_amdgpu_virtual_memory_mapping_t* mapping = state->mappings;
       mapping; mapping = mapping->next) {
    if (mapping->reservation == reservation &&
        iree_hal_amdgpu_virtual_memory_ranges_overlap(
            virtual_offset, size, mapping->virtual_offset, mapping->size)) {
      return true;
    }
  }
  return false;
}

static hsa_access_permission_t
iree_hal_amdgpu_virtual_memory_translate_protection(
    iree_hal_memory_protection_t protection, bool* out_is_valid) {
  *out_is_valid = true;
  switch (protection) {
    case IREE_HAL_MEMORY_PROTECTION_NONE:
      return HSA_ACCESS_PERMISSION_NONE;
    case IREE_HAL_MEMORY_PROTECTION_READ:
      return HSA_ACCESS_PERMISSION_RO;
    case IREE_HAL_MEMORY_PROTECTION_WRITE:
      return HSA_ACCESS_PERMISSION_WO;
    case IREE_HAL_MEMORY_PROTECTION_READ_WRITE:
      return HSA_ACCESS_PERMISSION_RW;
    default:
      *out_is_valid = false;
      return HSA_ACCESS_PERMISSION_NONE;
  }
}

static void iree_hal_amdgpu_virtual_memory_release_reservation(
    void* user_data, iree_hal_buffer_t* virtual_buffer) {
  (void)virtual_buffer;
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
      (iree_hal_amdgpu_virtual_memory_reservation_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_vmem_address_free_raw(
      reservation->state->libhsa, reservation->base_ptr, reservation->size));
  iree_allocator_free(reservation->state->host_allocator, reservation);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_virtual_memory_state_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_memory_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = NULL;

  iree_hal_amdgpu_virtual_memory_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  state->libhsa = libhsa;
  state->topology = topology;
  state->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&state->mutex);
  *out_state = state;
  return iree_ok_status();
}

void iree_hal_amdgpu_virtual_memory_state_destroy(
    iree_hal_amdgpu_virtual_memory_state_t* state) {
  if (!state) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT(state->reservations == NULL,
              "destroying AMDGPU VMM state with live reservations");
  IREE_ASSERT(state->mappings == NULL,
              "destroying AMDGPU VMM state with live mappings");
  IREE_ASSERT(state->physical_memory_count == 0,
              "destroying AMDGPU VMM state with live physical allocations");
  iree_slim_mutex_deinitialize(&state->mutex);
  iree_allocator_free(state->host_allocator, state);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_virtual_memory_reserve(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_device_size_t size, iree_hal_buffer_t** out_virtual_buffer) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_virtual_buffer);
  *out_virtual_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_memory_validate_placement(placement));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          size, /*offset=*/0, size, placement.allocation_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory reservation size %" PRIu64
        " must be a non-zero multiple of VMM granule %" PRIu64,
        (uint64_t)size, (uint64_t)placement.allocation_granule);
  }

  iree_hal_amdgpu_virtual_memory_reservation_t* reservation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator, sizeof(*reservation), (void**)&reservation));
  memset(reservation, 0, sizeof(*reservation));
  reservation->state = state;
  reservation->size = size;
  reservation->allocation_granule = placement.allocation_granule;
  reservation->queue_affinity = placement.queue_affinity;

  const iree_device_size_t reservation_alignment = placement.allocation_granule;
  iree_status_t status = iree_hsa_amd_vmem_address_reserve_align(
      IREE_LIBHSA(state->libhsa), &reservation->base_ptr, size,
      /*address=*/0, reservation_alignment, /*flags=*/0);
  if (iree_status_is_ok(status) &&
      (uintptr_t)reservation->base_ptr % reservation_alignment != 0) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "AMDGPU VMM reservation has insufficient alignment");
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t buffer_placement = {
        .device = placement.device,
        .queue_affinity = placement.queue_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_amdgpu_buffer_create(
        state->libhsa, buffer_placement, placement.memory_type,
        IREE_HAL_MEMORY_ACCESS_ALL, placement.buffer_usage, size, size,
        reservation->base_ptr,
        (iree_hal_buffer_release_callback_t){
            .fn = iree_hal_amdgpu_virtual_memory_release_reservation,
            .user_data = reservation,
        },
        state->host_allocator, &reservation->virtual_buffer);
  }

  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&state->mutex);
    reservation->next = state->reservations;
    state->reservations = reservation;
    iree_slim_mutex_unlock(&state->mutex);
    *out_virtual_buffer = reservation->virtual_buffer;
  } else {
    if (reservation->base_ptr) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(
          iree_hsa_amd_vmem_address_free_raw(state->libhsa,
                                             reservation->base_ptr, size));
    }
    iree_allocator_free(state->host_allocator, reservation);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_memory_release(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  iree_slim_mutex_lock(&state->mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_virtual_memory_reservation_t** reservation_ptr =
      &state->reservations;
  while (*reservation_ptr &&
         (*reservation_ptr)->virtual_buffer != virtual_buffer) {
    reservation_ptr = &(*reservation_ptr)->next;
  }
  if (!*reservation_ptr) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "virtual buffer does not belong to this AMDGPU "
                              "VMM state");
  } else if (iree_hal_amdgpu_virtual_memory_has_mapping_locked(
                 state, *reservation_ptr)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU virtual-memory reservation must be "
                              "fully unmapped before release");
  } else {
    *reservation_ptr = (*reservation_ptr)->next;
  }
  iree_slim_mutex_unlock(&state->mutex);

  if (iree_status_is_ok(status)) {
    iree_hal_buffer_destroy(virtual_buffer);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_physical_memory_allocate(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_physical_memory);
  *out_physical_memory = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_virtual_memory_validate_placement(placement));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          size, /*offset=*/0, size, placement.allocation_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU physical-memory allocation size %" PRIu64
        " must be a non-zero multiple of VMM granule %" PRIu64,
        (uint64_t)size, (uint64_t)placement.allocation_granule);
  }
  if (IREE_UNLIKELY(size > placement.max_allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU physical-memory allocation size %" PRIu64
                            " exceeds pool allocation limit %" PRIu64,
                            (uint64_t)size,
                            (uint64_t)placement.max_allocation_size);
  }

  // VMM physical handles must remain resident while their mappings are live.
  // The selected pool carries the cache policy, including uncached placement.
  hsa_amd_memory_type_t hsa_memory_type = MEMORY_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmem_translate_memory_type(
      IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_PINNED_HOST, &hsa_memory_type));

  iree_hal_physical_memory_t* physical_memory = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*physical_memory), (void**)&physical_memory));
  memset(physical_memory, 0, sizeof(*physical_memory));
  physical_memory->host_allocator = host_allocator;
  physical_memory->state = state;
  physical_memory->allocation_size = size;
  physical_memory->allocation_granule = placement.allocation_granule;

  iree_status_t status = iree_hsa_amd_vmem_handle_create(
      IREE_LIBHSA(state->libhsa), placement.memory_pool, size, hsa_memory_type,
      /*flags=*/0, &physical_memory->allocation_handle);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&state->mutex);
    ++state->physical_memory_count;
    iree_slim_mutex_unlock(&state->mutex);
    *out_physical_memory = physical_memory;
  } else {
    iree_allocator_free(host_allocator, physical_memory);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_physical_memory_free(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_physical_memory_t* physical_memory) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(physical_memory);
  if (physical_memory->state != state) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "physical memory belongs to a different AMDGPU "
                            "VMM state");
  }

  iree_slim_mutex_lock(&state->mutex);
  iree_status_t status = iree_ok_status();
  if (physical_memory->mapped_size != 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU physical-memory allocation still has %" PRIu64 " mapped bytes",
        (uint64_t)physical_memory->mapped_size);
  } else {
    IREE_ASSERT(state->physical_memory_count > 0);
    --state->physical_memory_count;
  }
  iree_slim_mutex_unlock(&state->mutex);

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_vmem_handle_release_raw(
            state->libhsa, physical_memory->allocation_handle));
    iree_allocator_free(physical_memory->host_allocator, physical_memory);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_memory_map(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);
  IREE_ASSERT_ARGUMENT(physical_memory);
  if (physical_memory->state != state) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "physical memory belongs to a different AMDGPU "
                            "VMM state");
  }
  if (IREE_UNLIKELY(physical_offset != 0 ||
                    size != physical_memory->allocation_size)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU VMM requires each mapping to cover one complete physical "
        "memory allocation");
  }

  iree_hal_amdgpu_virtual_memory_mapping_t* mapping = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      state->host_allocator, sizeof(*mapping), (void**)&mapping));
  memset(mapping, 0, sizeof(*mapping));

  iree_slim_mutex_lock(&state->mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
      iree_hal_amdgpu_virtual_memory_find_reservation_locked(state,
                                                             virtual_buffer);
  if (!reservation) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "virtual buffer does not belong to this AMDGPU "
                              "VMM state");
  } else if (!iree_hal_amdgpu_virtual_memory_range_is_valid(
                 reservation->size, virtual_offset, size,
                 reservation->allocation_granule)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU virtual-memory map range is not a "
                              "valid reservation-aligned range");
  } else if (!iree_hal_amdgpu_virtual_memory_is_aligned(
                 size, physical_memory->allocation_granule)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU physical-memory map size is not aligned");
  } else if (iree_hal_amdgpu_virtual_memory_has_overlapping_mapping_locked(
                 state, reservation, virtual_offset, size)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU virtual-memory map range overlaps an "
                              "existing mapping");
  }
  if (iree_status_is_ok(status)) {
    void* map_ptr = (uint8_t*)reservation->base_ptr + virtual_offset;
    status = iree_hsa_amd_vmem_map(
        IREE_LIBHSA(state->libhsa), map_ptr, size, /*in_offset=*/0,
        physical_memory->allocation_handle, /*flags=*/0);
  }
  if (iree_status_is_ok(status)) {
    mapping->reservation = reservation;
    mapping->physical_memory = physical_memory;
    mapping->virtual_offset = virtual_offset;
    mapping->size = size;
    mapping->next = state->mappings;
    state->mappings = mapping;
    IREE_ASSERT(physical_memory->mapped_size <= IREE_DEVICE_SIZE_MAX - size);
    physical_memory->mapped_size += size;
    mapping = NULL;
  }
  iree_slim_mutex_unlock(&state->mutex);
  iree_allocator_free(state->host_allocator, mapping);
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_memory_unmap(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  iree_slim_mutex_lock(&state->mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
      iree_hal_amdgpu_virtual_memory_find_reservation_locked(state,
                                                             virtual_buffer);
  iree_hal_amdgpu_virtual_memory_mapping_t* mapping = NULL;
  iree_hal_amdgpu_virtual_memory_mapping_t** mapping_ptr = NULL;
  if (!reservation) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "virtual buffer does not belong to this AMDGPU "
                              "VMM state");
  } else if (!iree_hal_amdgpu_virtual_memory_range_is_valid(
                 reservation->size, virtual_offset, size,
                 reservation->allocation_granule)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU virtual-memory unmap range is not a "
                              "valid reservation-aligned range");
  } else {
    mapping_ptr = &state->mappings;
    while (*mapping_ptr && ((*mapping_ptr)->reservation != reservation ||
                            (*mapping_ptr)->virtual_offset != virtual_offset ||
                            (*mapping_ptr)->size != size)) {
      mapping_ptr = &(*mapping_ptr)->next;
    }
    mapping = *mapping_ptr;
    if (!mapping) {
      if (iree_hal_amdgpu_virtual_memory_has_overlapping_mapping_locked(
              state, reservation, virtual_offset, size)) {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AMDGPU VMM can only unmap complete physical-memory mappings");
      } else {
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "AMDGPU virtual-memory range is not mapped");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    void* map_ptr = (uint8_t*)reservation->base_ptr + virtual_offset;
    status = iree_hsa_amd_vmem_unmap(IREE_LIBHSA(state->libhsa), map_ptr, size);
  }
  if (iree_status_is_ok(status)) {
    IREE_ASSERT(mapping->physical_memory->mapped_size >= size);
    mapping->physical_memory->mapped_size -= size;
    *mapping_ptr = mapping->next;
  }
  iree_slim_mutex_unlock(&state->mutex);
  if (iree_status_is_ok(status)) {
    iree_allocator_free(state->host_allocator, mapping);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_memory_protect(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_protection_t protection) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  bool is_valid_protection = false;
  const hsa_access_permission_t permissions =
      iree_hal_amdgpu_virtual_memory_translate_protection(protection,
                                                          &is_valid_protection);
  if (!is_valid_protection) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid AMDGPU virtual-memory protection flags");
  }

  iree_slim_mutex_lock(&state->mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
      iree_hal_amdgpu_virtual_memory_find_reservation_locked(state,
                                                             virtual_buffer);
  iree_hal_amdgpu_virtual_memory_mapping_t* mapping = NULL;
  if (!reservation) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "virtual buffer does not belong to this AMDGPU "
                              "VMM state");
  } else if (!iree_hal_amdgpu_virtual_memory_range_is_valid(
                 reservation->size, virtual_offset, size,
                 reservation->allocation_granule)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU virtual-memory protect range is not a "
                              "valid reservation-aligned range");
  } else {
    mapping = iree_hal_amdgpu_virtual_memory_find_exact_mapping_locked(
        state, reservation, virtual_offset, size);
    if (!mapping) {
      if (iree_hal_amdgpu_virtual_memory_has_overlapping_mapping_locked(
              state, reservation, virtual_offset, size)) {
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AMDGPU VMM can only protect complete physical-memory mappings");
      } else {
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "AMDGPU virtual-memory range is not mapped");
      }
    }
  }

  iree_hal_amdgpu_access_agent_list_t agent_list;
  hsa_amd_memory_access_desc_t access_descs[IREE_HAL_AMDGPU_MAX_CPU_AGENT +
                                            IREE_HAL_AMDGPU_MAX_GPU_AGENT];
  if (iree_status_is_ok(status)) {
    const iree_hal_amdgpu_queue_affinity_domain_t domain = {
        .supported_affinity = reservation->queue_affinity,
        .physical_device_count = state->topology->gpu_agent_count,
        .queue_count_per_physical_device =
            state->topology->gpu_agent_queue_count,
    };
    status = iree_hal_amdgpu_access_agent_list_resolve(
        state->topology, domain, queue_affinity, &agent_list);
  }
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < agent_list.count; ++i) {
      access_descs[i] = (hsa_amd_memory_access_desc_t){
          .permissions = permissions,
          .agent_handle = agent_list.values[i],
      };
    }
    void* map_ptr = (uint8_t*)reservation->base_ptr + virtual_offset;
    status = iree_hsa_amd_vmem_set_access(IREE_LIBHSA(state->libhsa), map_ptr,
                                          mapping->size, access_descs,
                                          agent_list.count);
  }
  iree_slim_mutex_unlock(&state->mutex);
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_memory_advise(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_advice_t advice) {
  (void)queue_affinity;
  (void)advice;
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  iree_slim_mutex_lock(&state->mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_virtual_memory_reservation_t* reservation =
      iree_hal_amdgpu_virtual_memory_find_reservation_locked(state,
                                                             virtual_buffer);
  if (!reservation) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "virtual buffer does not belong to this AMDGPU "
                              "VMM state");
  } else if (virtual_offset > reservation->size ||
             size > reservation->size - virtual_offset) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU virtual-memory advice range exceeds "
                              "the reservation");
  }
  iree_slim_mutex_unlock(&state->mutex);
  return status;
}
