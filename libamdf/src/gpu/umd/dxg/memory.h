// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_MEMORY_H_
#define AMDF_SRC_GPU_UMD_DXG_MEMORY_H_

#include "libamdf/src/gpu/umd/dxg/adapter.h"
#include "libamdf/src/gpu/umd/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_gpu_dxg_memory_t amdf_gpu_dxg_memory_t;

// Native ownership after one allocation-destruction attempt.
typedef enum amdf_gpu_dxg_memory_ownership_e {
  // Native allocation handles remain live and may be destroyed again.
  AMDF_GPU_DXG_MEMORY_OWNERSHIP_RETAINED = 0,
  // Native allocation handles were consumed regardless of operation status.
  AMDF_GPU_DXG_MEMORY_OWNERSHIP_CONSUMED = 1,
} amdf_gpu_dxg_memory_ownership_t;

// Status and ownership progress from one native allocation destruction.
typedef struct amdf_gpu_dxg_memory_release_result_t {
  // Result of the native destruction operation.
  amdf_status_t status;
  // Resulting ownership of every supplied allocation handle and resource.
  amdf_gpu_dxg_memory_ownership_t ownership;
} amdf_gpu_dxg_memory_release_result_t;

// Cold native operations required by the shared DXG physical-memory owner.
typedef struct amdf_gpu_dxg_memory_native_api_t {
  // Reserves one unbacked GPU address interval.
  amdf_status_t (*reserve_address)(void* context, uint32_t adapter_handle,
                                   uint64_t base_address,
                                   uint64_t minimum_address,
                                   uint64_t byte_length, uint64_t* out_address);
  // Maps one physical allocation at an exact reserved GPU address.
  amdf_status_t (*map_allocation)(void* context, uint32_t paging_queue_handle,
                                  uint64_t address, uint32_t allocation_handle,
                                  uint64_t allocation_byte_offset,
                                  uint64_t byte_length,
                                  amdf_memory_access_t access,
                                  uint64_t* out_address,
                                  uint64_t* out_paging_fence);
  // Makes the complete physical allocation vector resident.
  amdf_status_t (*make_resident)(void* context, uint32_t paging_queue_handle,
                                 uint32_t allocation_count,
                                 const uint32_t* allocation_handles,
                                 uint32_t* out_resident_allocation_count,
                                 uint64_t* out_paging_fence);
  // Waits for one accepted mapping or residency point.
  amdf_status_t (*wait_for_paging)(void* context, uint32_t device_handle,
                                   uint32_t synchronization_object_handle,
                                   const volatile uint64_t* progress_fence,
                                   uint64_t target_value);
  // Destroys one resource or ungrouped allocation vector and reports whether
  // the native handles survived a failure.
  amdf_gpu_dxg_memory_release_result_t (*destroy_allocations)(
      void* context, uint32_t device_handle, uint32_t resource_handle,
      uint32_t allocation_count, const uint32_t* allocation_handles,
      bool synchronous);
  // Releases one empty GPU address reservation.
  amdf_status_t (*free_address)(void* context, uint32_t adapter_handle,
                                uint64_t address, uint64_t byte_length);
  // Publishes one host-written physical allocation range to the device.
  amdf_status_t (*invalidate_allocation)(void* context, uint32_t device_handle,
                                         uint32_t allocation_handle,
                                         uint64_t byte_offset,
                                         uint64_t byte_length);
  // Retains an independent reference to one native external-memory payload.
  amdf_status_t (*retain_external_reference)(uintptr_t reference,
                                             uintptr_t* out_reference);
  // Releases one independently retained native external-memory payload.
  amdf_status_t (*release_external_reference)(uintptr_t reference);
} amdf_gpu_dxg_memory_native_api_t;

// Immutable device state borrowed by one DXG physical-memory owner.
typedef struct amdf_gpu_dxg_memory_device_t {
  // Allocator copied into every memory owner and used for its metadata.
  amdf_allocator_t host_allocator;
  // Native operation table selected by the device provider.
  const amdf_gpu_dxg_memory_native_api_t* native_api;
  // Live provider device forwarded to every contextual native operation.
  void* native_context;
  // Private allocation provider borrowed through memory destruction.
  const amdf_gpu_dxg_adapter_t* provider_adapter;
  // Query adapter owning GPU virtual-address reservations.
  uint32_t adapter_handle;
  // DXGI adapter identity used for typed external-buffer admission.
  uint64_t adapter_luid;
  // Logical KMT device owning physical allocations.
  uint32_t device_handle;
  // Paging queue used for mapping and residency operations.
  uint32_t paging_queue_handle;
  // Paging synchronization object waited from the host.
  uint32_t paging_synchronization_object_handle;
  // CPU mapping of paging progress, or NULL when native waiting is required.
  const volatile uint64_t* paging_progress_fence;
} amdf_gpu_dxg_memory_device_t;

// One native allocation span within an already prepared ordinary resource.
typedef struct amdf_gpu_dxg_memory_span_t {
  // Borrowed allocation handle in the source consumer's logical KMT device.
  uint32_t allocation_handle;
  // Source offset within this native allocation, in bytes.
  uint64_t byte_offset;
  // Available bytes from this offset to the native allocation boundary.
  uint64_t byte_length;
} amdf_gpu_dxg_memory_span_t;

// Prepares newly allocated or registered physical memory through the shared
// DXG lifecycle. Failure publishes the partial owner for ordered destruction.
amdf_status_t amdf_gpu_dxg_memory_prepare(
    const amdf_gpu_dxg_memory_device_t* device,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_dxg_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result);

// Prepares a typed external buffer through the same address, mapping, and
// residency lifecycle. Failure publishes the partial owner for destruction.
amdf_status_t amdf_gpu_dxg_memory_prepare_import(
    const amdf_gpu_dxg_memory_device_t* device,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_dxg_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result);

// Retains and exports the memory's native external reference.
amdf_status_t amdf_gpu_dxg_memory_export(
    amdf_gpu_dxg_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value);

// Borrows the remappable host-backing source.
uintptr_t amdf_gpu_dxg_memory_host_mapping_source(
    amdf_gpu_dxg_memory_t* memory);

// Releases complete or partial native state and metadata.
amdf_status_t amdf_gpu_dxg_memory_destroy(amdf_gpu_dxg_memory_t* memory);

// Reports payload still retained by a partial owner.
uint64_t amdf_gpu_dxg_memory_remaining_backing_byte_length(
    const amdf_gpu_dxg_memory_t* memory);

// Releases only metadata after terminal cleanup failure.
void amdf_gpu_dxg_memory_abandon(amdf_gpu_dxg_memory_t* memory);

// Creates one lightweight host view over the persistent mapping.
amdf_status_t amdf_gpu_dxg_memory_map(
    amdf_gpu_dxg_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info, amdf_gpu_dxg_memory_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result);

// Performs host cache maintenance and any required native publication.
amdf_status_t amdf_gpu_dxg_memory_cache_control(
    amdf_gpu_dxg_memory_t* memory, amdf_host_cache_operation_t operation,
    uint64_t memory_byte_offset, uint64_t byte_length);

// Looks up the allocation containing one validated backing-relative offset.
amdf_gpu_dxg_memory_span_t amdf_gpu_dxg_memory_span(
    const amdf_gpu_dxg_memory_t* memory, uint64_t byte_offset);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_DXG_MEMORY_H_
