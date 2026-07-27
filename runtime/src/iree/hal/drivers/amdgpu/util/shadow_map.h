// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_SHADOW_MAP_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_SHADOW_MAP_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_AMDGPU_SHADOW_MAP_DEFAULT_SCALE_SHIFT 3u

typedef struct iree_hal_amdgpu_shadow_map_t iree_hal_amdgpu_shadow_map_t;

typedef enum iree_hal_amdgpu_shadow_map_mapping_mode_e {
  // Shadow slabs are mapped only when explicitly requested.
  IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_SPARSE = 0,
  // Every shadow slab is initially mapped to one shared physical slab.
  IREE_HAL_AMDGPU_SHADOW_MAP_MAPPING_MODE_PREMAPPED = 1,
} iree_hal_amdgpu_shadow_map_mapping_mode_t;

typedef struct iree_hal_amdgpu_shadow_map_range_t {
  // Application address where the mapped range begins.
  uint64_t application_address;

  // Length of the application range in bytes.
  iree_device_size_t application_length;

  // Device-visible address of the first shadow byte covering the range.
  uint64_t shadow_address;

  // Offset of |shadow_address| from the shadow reservation base.
  iree_device_size_t shadow_offset;

  // Number of shadow bytes covering the range.
  iree_device_size_t shadow_length;

  // First shadow slab index touched by the range.
  uint64_t first_slab_index;

  // Number of shadow slabs touched by the range.
  iree_host_size_t slab_count;
} iree_hal_amdgpu_shadow_map_range_t;

typedef struct iree_hal_amdgpu_shadow_map_mapper_t {
  // Mapper-specific state passed to all callbacks.
  void* user_data;

  // Reserves shadow virtual address space.
  iree_status_t (*reserve)(iree_hal_amdgpu_shadow_map_t* map,
                           iree_device_size_t reservation_size,
                           iree_device_size_t alignment,
                           IREE_AMDGPU_DEVICE_PTR void** out_base_ptr);

  // Releases shadow virtual address space reserved by |reserve|.
  void (*release_reservation)(iree_hal_amdgpu_shadow_map_t* map,
                              IREE_AMDGPU_DEVICE_PTR void* base_ptr,
                              iree_device_size_t reservation_size);

  // Creates and maps one physical shadow slab at |target_ptr|.
  iree_status_t (*map_slab)(iree_hal_amdgpu_shadow_map_t* map,
                            IREE_AMDGPU_DEVICE_PTR void* target_ptr,
                            iree_device_size_t slab_size,
                            iree_host_size_t access_desc_count,
                            const hsa_amd_memory_access_desc_t* access_descs,
                            hsa_amd_vmem_alloc_handle_t* out_allocation_handle);

  // Unmaps and releases one physical shadow slab.
  void (*unmap_slab)(iree_hal_amdgpu_shadow_map_t* map,
                     IREE_AMDGPU_DEVICE_PTR void* target_ptr,
                     iree_device_size_t slab_size,
                     hsa_amd_vmem_alloc_handle_t allocation_handle);
} iree_hal_amdgpu_shadow_map_mapper_t;

typedef struct iree_hal_amdgpu_shadow_map_params_t {
  // Host allocator used for map-owned tables.
  iree_allocator_t host_allocator;

  // Log2 application bytes represented by one shadow byte.
  uint32_t shadow_scale_shift;

  // Base application address covered by the shadow reservation.
  uint64_t application_window_base;

  // Shadow virtual address range size in bytes.
  iree_device_size_t shadow_size;

  // Physical shadow slab size in bytes.
  iree_device_size_t slab_size;

  // Shadow slab mapping policy.
  iree_hal_amdgpu_shadow_map_mapping_mode_t mapping_mode;

  // Byte value written across each newly mapped physical shadow slab.
  uint8_t initial_slab_value;

  // Number of HSA VMM access descriptors in |access_descs|.
  iree_host_size_t access_desc_count;

  // HSA VMM access descriptors applied to every mapped slab.
  const hsa_amd_memory_access_desc_t* access_descs;

  // Mapper used for virtual reservation and physical slab mapping.
  iree_hal_amdgpu_shadow_map_mapper_t mapper;
} iree_hal_amdgpu_shadow_map_params_t;

typedef struct iree_hal_amdgpu_shadow_map_hsa_params_t {
  // Host allocator used for map-owned tables.
  iree_allocator_t host_allocator;

  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // HSA memory pool used for physical shadow slabs.
  hsa_amd_memory_pool_t memory_pool;

  // HSA VMM memory type used for physical shadow slabs.
  iree_hal_amdgpu_vmem_memory_type_t memory_type;

  // Log2 application bytes represented by one shadow byte.
  uint32_t shadow_scale_shift;

  // Base application address covered by the shadow reservation.
  uint64_t application_window_base;

  // Shadow virtual address range size in bytes.
  iree_device_size_t shadow_size;

  // Requested physical shadow slab size in bytes.
  iree_device_size_t requested_slab_size;

  // Shadow slab mapping policy.
  iree_hal_amdgpu_shadow_map_mapping_mode_t mapping_mode;

  // Byte value written across each newly mapped physical shadow slab.
  uint8_t initial_slab_value;

  // Number of HSA VMM access descriptors in |access_descs|.
  iree_host_size_t access_desc_count;

  // HSA VMM access descriptors applied to every mapped slab.
  const hsa_amd_memory_access_desc_t* access_descs;
} iree_hal_amdgpu_shadow_map_hsa_params_t;

typedef struct iree_hal_amdgpu_shadow_map_slab_t {
  // Shadow slab index within the reservation.
  uint64_t index;

  // Device-visible virtual address where this slab is mapped.
  IREE_AMDGPU_DEVICE_PTR void* base_ptr;

  // Physical VMM allocation handle backing this slab.
  hsa_amd_vmem_alloc_handle_t allocation_handle;
} iree_hal_amdgpu_shadow_map_slab_t;

typedef struct iree_hal_amdgpu_shadow_map_statistics_t {
  // Number of precise physical shadow slabs mapped in |slabs|.
  iree_host_size_t mapped_slab_count;

  // Physical shadow bytes committed for precise slabs and any shared alias
  // slab.
  iree_device_size_t committed_size;
} iree_hal_amdgpu_shadow_map_statistics_t;

typedef struct iree_hal_amdgpu_shadow_map_t {
  // True when the map has been initialized and must be deinitialized.
  bool initialized;

  // Host allocator used for all map-owned allocations.
  iree_allocator_t host_allocator;

  // Log2 application bytes represented by one shadow byte.
  uint32_t shadow_scale_shift;

  // Base application address covered by the shadow reservation.
  uint64_t application_window_base;

  // Size of the application address window covered by the reservation.
  iree_device_size_t application_window_size;

  // Device-visible shadow base used by checked code.
  uint64_t shadow_base;

  // Base pointer of the reserved shadow virtual address range.
  IREE_AMDGPU_DEVICE_PTR void* reservation_base_ptr;

  // Size of the reserved shadow virtual address range in bytes.
  iree_device_size_t reservation_size;

  // Physical shadow slab size in bytes.
  iree_device_size_t slab_size;

  // Shadow slab mapping policy.
  iree_hal_amdgpu_shadow_map_mapping_mode_t mapping_mode;

  // Byte value written across each newly mapped physical shadow slab.
  uint8_t initial_slab_value;

  // Number of HSA VMM access descriptors in |access_descs|.
  iree_host_size_t access_desc_count;

  // Map-owned HSA VMM access descriptors applied to every slab.
  hsa_amd_memory_access_desc_t* access_descs;

  // Mapper used for virtual reservation and physical slab mapping.
  iree_hal_amdgpu_shadow_map_mapper_t mapper;

  // HSA-backed mapper state used by the production AMDGPU path.
  struct {
    // Borrowed HSA API table.
    const iree_hal_amdgpu_libhsa_t* libhsa;

    // HSA memory pool used for physical shadow slabs.
    hsa_amd_memory_pool_t memory_pool;

    // HSA VMM memory type used for physical shadow slabs.
    iree_hal_amdgpu_vmem_memory_type_t memory_type;

    // Translated HSA memory type for hsa_amd_vmem_handle_create.
    hsa_amd_memory_type_t hsa_memory_type;

    // Shared physical slab aliased across the reservation in PREMAPPED mode.
    hsa_amd_vmem_alloc_handle_t alias_allocation_handle;
  } hsa;

  // Guards slab table publication.
  iree_slim_mutex_t mutex;

  // Dense table of mapped slabs.
  iree_hal_amdgpu_shadow_map_slab_t* slabs;

  // Number of mapped slabs in |slabs|.
  iree_host_size_t slab_count;

  // Capacity of |slabs|.
  iree_host_size_t slab_capacity;
} iree_hal_amdgpu_shadow_map_t;

// Initializes |out_map| with a caller-provided shadow VMM mapper.
iree_status_t iree_hal_amdgpu_shadow_map_initialize(
    const iree_hal_amdgpu_shadow_map_params_t* params,
    iree_hal_amdgpu_shadow_map_t* out_map);

// Initializes |out_map| with the built-in HSA VMM mapper.
iree_status_t iree_hal_amdgpu_shadow_map_initialize_hsa(
    const iree_hal_amdgpu_shadow_map_hsa_params_t* params,
    iree_hal_amdgpu_shadow_map_t* out_map);

// Releases all mapped slabs and the shadow virtual address reservation.
void iree_hal_amdgpu_shadow_map_deinitialize(iree_hal_amdgpu_shadow_map_t* map);

// Snapshots the map's current physical shadow pressure counters.
//
// Uninitialized or NULL maps report all-zero counters. The snapshot is taken
// under the shadow-map slab-table lock and may race with later mappings.
void iree_hal_amdgpu_shadow_map_query_statistics(
    iree_hal_amdgpu_shadow_map_t* map,
    iree_hal_amdgpu_shadow_map_statistics_t* out_statistics);

// Calculates the shadow bytes and slabs covering an application byte range.
iree_status_t iree_hal_amdgpu_shadow_map_calculate_range(
    const iree_hal_amdgpu_shadow_map_t* map, uint64_t application_address,
    iree_device_size_t application_length,
    iree_hal_amdgpu_shadow_map_range_t* out_range);

// Maps all shadow slabs required to cover |application_address| and
// |application_length|.
iree_status_t iree_hal_amdgpu_shadow_map_map_range(
    iree_hal_amdgpu_shadow_map_t* map, uint64_t application_address,
    iree_device_size_t application_length,
    iree_hal_amdgpu_shadow_map_range_t* out_range);

// Maps one shadow slab by index.
iree_status_t iree_hal_amdgpu_shadow_map_map_slab(
    iree_hal_amdgpu_shadow_map_t* map, uint64_t slab_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_SHADOW_MAP_H_
