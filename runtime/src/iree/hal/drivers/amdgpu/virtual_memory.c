// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/virtual_memory.h"

#include <stdint.h>

#include "iree/hal/drivers/amdgpu/access_policy.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

// ROCr is the authoritative owner of reservation, mapping, access, and handle
// state. This adapter retains only the objects needed to express HAL ownership.

typedef struct iree_hal_amdgpu_virtual_memory_range_t {
  // Base address of the ROCr virtual reservation.
  void* base_ptr;

  // Byte length of the ROCr virtual reservation.
  iree_device_size_t size;

  // HAL queues permitted to access the reservation.
  iree_hal_queue_affinity_t queue_affinity;
} iree_hal_amdgpu_virtual_memory_range_t;

struct iree_hal_physical_memory_t {
  // Host allocator used to release this physical-memory wrapper.
  iree_allocator_t host_allocator;

  // VMM context that created this allocation.
  iree_hal_amdgpu_virtual_memory_state_t* state;

  // HSA VMM handle representing the physical allocation.
  hsa_amd_vmem_alloc_handle_t allocation_handle;

  // Byte length of the physical allocation.
  iree_device_size_t allocation_size;

  // Smallest legal mapping multiple for this physical allocation.
  iree_device_size_t minimum_granule;

  // HAL memory type used for allocator statistics.
  iree_hal_memory_type_t memory_type;
};

struct iree_hal_amdgpu_virtual_memory_state_t {
  // Unowned HSA dynamic-library table used for VMM calls.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Unowned topology used to grant access to selected queue agents.
  const iree_hal_amdgpu_topology_t* topology;

  // Unowned HAL device used to validate reservation ownership.
  iree_hal_device_t* device;

  // Unowned allocator statistics updated for physical allocations.
  iree_hal_allocator_statistics_t* statistics;

  // Host allocator used to release this context.
  iree_allocator_t host_allocator;
};

static bool iree_hal_amdgpu_virtual_memory_is_aligned(
    iree_device_size_t value, iree_device_size_t alignment) {
  return alignment != 0 && value % alignment == 0;
}

static bool iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
    iree_device_size_t total_size, iree_device_size_t offset,
    iree_device_size_t size) {
  return offset <= total_size && size <= total_size - offset;
}

static bool iree_hal_amdgpu_virtual_memory_range_is_valid(
    iree_device_size_t total_size, iree_device_size_t offset,
    iree_device_size_t size, iree_device_size_t alignment) {
  return size != 0 &&
         iree_hal_amdgpu_virtual_memory_is_aligned(offset, alignment) &&
         iree_hal_amdgpu_virtual_memory_is_aligned(size, alignment) &&
         iree_hal_amdgpu_virtual_memory_range_is_in_bounds(total_size, offset,
                                                           size);
}

static iree_status_t iree_hal_amdgpu_virtual_memory_validate_placement(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_amdgpu_virtual_memory_placement_t placement) {
  if (IREE_UNLIKELY(!placement.memory_pool.handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement has no HSA memory pool");
  }
  if (IREE_UNLIKELY(placement.device != state->device)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement belongs to another device");
  }
  if (IREE_UNLIKELY(
          placement.minimum_granule == 0 ||
          !iree_device_size_is_power_of_two(placement.minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU VMM placement has invalid minimum granule %" PRIu64,
        (uint64_t)placement.minimum_granule);
  }
  if (IREE_UNLIKELY(
          placement.recommended_granule < placement.minimum_granule ||
          !iree_device_size_is_power_of_two(placement.recommended_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU VMM placement has invalid recommended granule %" PRIu64,
        (uint64_t)placement.recommended_granule);
  }
  if (IREE_UNLIKELY(placement.max_allocation_size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VMM placement has no allocation limit");
  }
  return iree_ok_status();
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
  iree_hal_amdgpu_virtual_memory_state_t* state =
      (iree_hal_amdgpu_virtual_memory_state_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_vmem_address_free_raw(
      state->libhsa, iree_hal_amdgpu_buffer_device_pointer(virtual_buffer),
      (size_t)iree_hal_buffer_allocation_size(virtual_buffer)));
  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_buffer_release_callback_t
iree_hal_amdgpu_virtual_memory_reservation_release_callback(
    iree_hal_amdgpu_virtual_memory_state_t* state) {
  return (iree_hal_buffer_release_callback_t){
      .fn = iree_hal_amdgpu_virtual_memory_release_reservation,
      .user_data = state,
  };
}

static iree_status_t iree_hal_amdgpu_virtual_memory_free_address(
    iree_hal_amdgpu_virtual_memory_state_t* state, void* base_ptr,
    iree_device_size_t size) {
  const hsa_status_t hsa_status =
      iree_hsa_amd_vmem_address_free_raw(state->libhsa, base_ptr, (size_t)size);
  if (hsa_status == HSA_STATUS_ERROR_RESOURCE_FREE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU virtual-memory reservation still has mapped ranges");
  }
  return iree_status_from_hsa_status(__FILE__, __LINE__, hsa_status,
                                     "hsa_amd_vmem_address_free", NULL);
}

static iree_status_t iree_hal_amdgpu_virtual_memory_resolve_range(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer,
    iree_hal_amdgpu_virtual_memory_range_t* out_range) {
  *out_range = (iree_hal_amdgpu_virtual_memory_range_t){0};
  const iree_hal_buffer_release_callback_t release_callback =
      iree_hal_amdgpu_virtual_memory_reservation_release_callback(state);
  if (!iree_hal_amdgpu_buffer_uses_release_callback(virtual_buffer,
                                                    release_callback)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "virtual buffer does not belong to this AMDGPU VMM "
                            "context");
  }

  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(virtual_buffer);
  if (IREE_UNLIKELY(placement.device != state->device)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual reservation belongs to another "
                            "device");
  }

  void* base_ptr = iree_hal_amdgpu_buffer_device_pointer(virtual_buffer);
  if (IREE_UNLIKELY(!base_ptr)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU virtual reservation has no device address");
  }

  *out_range = (iree_hal_amdgpu_virtual_memory_range_t){
      .base_ptr = base_ptr,
      .size = iree_hal_buffer_allocation_size(virtual_buffer),
      .queue_affinity = placement.queue_affinity,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_virtual_memory_state_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_hal_device_t* device,
    iree_hal_allocator_statistics_t* statistics,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_virtual_memory_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = NULL;

  iree_hal_amdgpu_virtual_memory_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  *state = (iree_hal_amdgpu_virtual_memory_state_t){
      .libhsa = libhsa,
      .topology = topology,
      .device = device,
      .statistics = statistics,
      .host_allocator = host_allocator,
  };
  *out_state = state;
  return iree_ok_status();
}

void iree_hal_amdgpu_virtual_memory_state_destroy(
    iree_hal_amdgpu_virtual_memory_state_t* state) {
  if (!state) return;
  IREE_TRACE_ZONE_BEGIN(z0);
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
      iree_hal_amdgpu_virtual_memory_validate_placement(state, placement));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          size, /*offset=*/0, size, placement.minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory reservation size %" PRIu64
        " must be a non-zero multiple of minimum VMM granule %" PRIu64,
        (uint64_t)size, (uint64_t)placement.minimum_granule);
  }

  const iree_device_size_t reservation_alignment =
      iree_hal_amdgpu_virtual_memory_is_aligned(size,
                                                placement.recommended_granule)
          ? placement.recommended_granule
          : placement.minimum_granule;
  void* base_ptr = NULL;
  iree_status_t status = iree_hsa_amd_vmem_address_reserve_align(
      IREE_LIBHSA(state->libhsa), &base_ptr, size, /*address=*/0,
      reservation_alignment, /*flags=*/0);
  if (iree_status_is_ok(status) &&
      (uintptr_t)base_ptr % reservation_alignment != 0) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "AMDGPU VMM reservation has insufficient alignment");
  }

  iree_hal_buffer_t* virtual_buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t buffer_placement = {
        .device = placement.device,
        .queue_affinity = placement.queue_affinity,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_amdgpu_buffer_create(
        state->libhsa, buffer_placement, placement.memory_type,
        IREE_HAL_MEMORY_ACCESS_ALL, placement.buffer_usage, size, size,
        base_ptr,
        iree_hal_amdgpu_virtual_memory_reservation_release_callback(state),
        state->host_allocator, &virtual_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_virtual_buffer = virtual_buffer;
  } else if (base_ptr) {
    status = iree_status_join(
        status, iree_hsa_amd_vmem_address_free(IREE_LIBHSA(state->libhsa),
                                               base_ptr, size));
  }
  return status;
}

iree_status_t iree_hal_amdgpu_virtual_memory_release(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_free_address(
      state, range.base_ptr, range.size));

  const iree_hal_buffer_release_callback_t release_callback =
      iree_hal_amdgpu_virtual_memory_reservation_release_callback(state);
  iree_hal_amdgpu_buffer_disarm_storage(virtual_buffer, release_callback);
  iree_hal_buffer_destroy(virtual_buffer);
  return iree_ok_status();
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
      iree_hal_amdgpu_virtual_memory_validate_placement(state, placement));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          size, /*offset=*/0, size, placement.minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU physical-memory allocation size %" PRIu64
        " must be a non-zero multiple of minimum VMM granule %" PRIu64,
        (uint64_t)size, (uint64_t)placement.minimum_granule);
  }
  if (IREE_UNLIKELY(size > placement.max_allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU physical-memory allocation size %" PRIu64
                            " exceeds pool allocation limit %" PRIu64,
                            (uint64_t)size,
                            (uint64_t)placement.max_allocation_size);
  }

  // HIP requires VMM backing allocations to remain resident and ROCr expresses
  // that contract with MEMORY_TYPE_PINNED for both host and device pools.
  hsa_amd_memory_type_t hsa_memory_type = MEMORY_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmem_translate_memory_type(
      IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_PINNED_HOST, &hsa_memory_type));

  iree_hal_physical_memory_t* physical_memory = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*physical_memory), (void**)&physical_memory));
  *physical_memory = (iree_hal_physical_memory_t){
      .host_allocator = host_allocator,
      .state = state,
      .allocation_size = size,
      .minimum_granule = placement.minimum_granule,
      .memory_type = placement.memory_type,
  };

  iree_status_t status = iree_hsa_amd_vmem_handle_create(
      IREE_LIBHSA(state->libhsa), placement.memory_pool, size, hsa_memory_type,
      /*flags=*/0, &physical_memory->allocation_handle);
  if (iree_status_is_ok(status)) {
    IREE_STATISTICS({
      if (state->statistics) {
        iree_hal_allocator_statistics_record_alloc(
            state->statistics, physical_memory->memory_type,
            physical_memory->allocation_size);
      }
    });
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
                            "VMM context");
  }

  // The HAL contract requires callers to unmap every alias before freeing.
  // ROCr retains mapped references internally, so no duplicate mapping ledger
  // is needed here to preserve native allocation lifetime.
  IREE_RETURN_IF_ERROR(iree_hsa_amd_vmem_handle_release(
      IREE_LIBHSA(state->libhsa), physical_memory->allocation_handle));
  IREE_STATISTICS({
    if (state->statistics) {
      iree_hal_allocator_statistics_record_free(
          state->statistics, physical_memory->memory_type,
          physical_memory->allocation_size);
    }
  });
  iree_allocator_free(physical_memory->host_allocator, physical_memory);
  return iree_ok_status();
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
                            "VMM context");
  }
  if (IREE_UNLIKELY(physical_offset != 0 ||
                    size != physical_memory->allocation_size)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU VMM requires each mapping to cover one complete physical "
        "memory allocation");
  }

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_valid(
          range.size, virtual_offset, size,
          physical_memory->minimum_granule))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory map range is not a valid minimum-granule "
        "physical mapping");
  }

  void* map_ptr = (uint8_t*)range.base_ptr + virtual_offset;
  if (IREE_UNLIKELY((uintptr_t)map_ptr % physical_memory->minimum_granule !=
                    0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU virtual-memory map address is not aligned to the physical "
        "allocation granule");
  }
  return iree_hsa_amd_vmem_map(IREE_LIBHSA(state->libhsa), map_ptr, size,
                               /*in_offset=*/0,
                               physical_memory->allocation_handle, /*flags=*/0);
}

iree_status_t iree_hal_amdgpu_virtual_memory_unmap(
    iree_hal_amdgpu_virtual_memory_state_t* state,
    iree_hal_buffer_t* virtual_buffer, iree_device_size_t virtual_offset,
    iree_device_size_t size) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(virtual_buffer);

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(size == 0 ||
                    !iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
                        range.size, virtual_offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual-memory unmap range exceeds the "
                            "reservation");
  }

  void* map_ptr = (uint8_t*)range.base_ptr + virtual_offset;
  return iree_hsa_amd_vmem_unmap(IREE_LIBHSA(state->libhsa), map_ptr, size);
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

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(size == 0 ||
                    !iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
                        range.size, virtual_offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual-memory protection range exceeds "
                            "the reservation");
  }

  const iree_hal_amdgpu_queue_affinity_domain_t domain = {
      .supported_affinity = range.queue_affinity,
      .physical_device_count = state->topology->gpu_agent_count,
      .queue_count_per_physical_device = state->topology->gpu_agent_queue_count,
  };
  iree_hal_amdgpu_access_agent_list_t agent_list;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_access_agent_list_resolve(
      state->topology, domain, queue_affinity, &agent_list));

  hsa_amd_memory_access_desc_t access_descs[IREE_HAL_AMDGPU_MAX_CPU_AGENT +
                                            IREE_HAL_AMDGPU_MAX_GPU_AGENT];
  for (uint32_t i = 0; i < agent_list.count; ++i) {
    access_descs[i] = (hsa_amd_memory_access_desc_t){
        .permissions = permissions,
        .agent_handle = agent_list.values[i],
    };
  }
  void* map_ptr = (uint8_t*)range.base_ptr + virtual_offset;
  return iree_hsa_amd_vmem_set_access(IREE_LIBHSA(state->libhsa), map_ptr, size,
                                      access_descs, agent_list.count);
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

  iree_hal_amdgpu_virtual_memory_range_t range;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_memory_resolve_range(
      state, virtual_buffer, &range));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_virtual_memory_range_is_in_bounds(
          range.size, virtual_offset, size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU virtual-memory advice range exceeds the "
                            "reservation");
  }
  return iree_ok_status();
}
