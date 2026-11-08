// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_MEMORY_RESERVATION_H_
#define AMDF_SRC_GPU_UMD_DXG_MEMORY_RESERVATION_H_

#include "libamdf/src/gpu/umd/dxg/memory.h"
#include "libamdf/src/gpu/umd/dxg/memory_profile.h"
#include "libamdf/src/gpu/umd/memory_reservation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_gpu_dxg_memory_reservation_t
    amdf_gpu_dxg_memory_reservation_t;

// Immutable native attachment copied into one virtual-memory reservation.
typedef struct amdf_gpu_dxg_memory_reservation_device_t {
  // Native operation table selected by the device provider.
  const amdf_gpu_dxg_memory_native_api_t* native_api;
  // Live provider device forwarded to contextual native operations.
  void* native_context;
  // Query adapter owning GPU virtual-address reservations.
  uint32_t adapter_handle;
  // DXGI adapter identity naming one linked-adapter domain.
  uint64_t adapter_luid;
  // Physical adapter represented by native private records.
  uint32_t physical_adapter_index;
  // Native GPU virtual-address width in bits.
  uint32_t virtual_address_bit_count;
  // Provider-supported unit for independently replaceable GPU address ranges.
  uint64_t mapping_granularity;
  // Nonzero when GPU mappings can deny device writes.
  uint32_t read_only_memory_supported;
  // Nonzero when GPU mappings can deny instruction fetches.
  uint32_t no_execute_memory_supported;
  // Logical KMT device owning physical allocations.
  uint32_t device_handle;
  // Paging queue used for mapping operations.
  uint32_t paging_queue_handle;
  // Paging synchronization object waited from the host.
  uint32_t paging_synchronization_object_handle;
  // CPU mapping of paging progress, or NULL when native waiting is required.
  const volatile uint64_t* paging_progress_fence;
} amdf_gpu_dxg_memory_reservation_device_t;

// Captures one provider device by ordinal during reservation construction.
typedef amdf_gpu_dxg_memory_reservation_device_t (
    *amdf_gpu_dxg_memory_reservation_describe_device_fn_t)(const void* context,
                                                           uint32_t ordinal);

// One provider-neutral native mapping operation after physical-span expansion.
typedef struct amdf_gpu_dxg_mapping_operation_t {
  // Exact destination GPU address.
  uint64_t target_address;
  // Nonzero native mapping length.
  uint64_t byte_length;
  // Borrowed physical allocation handle, or zero for detach.
  uint32_t allocation_handle;
  // Physical allocation-relative source offset.
  uint64_t source_byte_offset;
  // Exact installed permissions, or zero for detach.
  amdf_memory_access_t access;
} amdf_gpu_dxg_mapping_operation_t;

// Receives one flattened native operation in destination order.
typedef void (*amdf_gpu_dxg_mapping_operation_visitor_fn_t)(
    void* context, uint32_t ordinal,
    const amdf_gpu_dxg_mapping_operation_t* operation);

// True when both immutable device descriptions address one native page table.
bool amdf_gpu_dxg_devices_share_mapping_domain(
    const amdf_gpu_dxg_memory_reservation_device_t* a,
    const amdf_gpu_dxg_memory_reservation_device_t* b);

// Derives immutable public reservation limits for the selected DXG device.
amdf_status_t amdf_gpu_dxg_query_memory_reservation_capabilities(
    const amdf_gpu_dxg_memory_reservation_device_t* device,
    uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities);

// Reserves one common GPU address and copies every described device attachment.
amdf_status_t amdf_gpu_dxg_memory_reservation_create(
    amdf_allocator_t host_allocator, uint32_t device_count,
    amdf_gpu_dxg_memory_reservation_describe_device_fn_t describe_device,
    const void* describe_context, uint64_t byte_length,
    uint64_t minimum_alignment, uint64_t common_address,
    amdf_gpu_dxg_memory_reservation_t** out_reservation);

// Reads complete immutable address and alias-domain facts established by
// create.
void amdf_gpu_dxg_memory_reservation_query_access(
    const amdf_gpu_dxg_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint64_t* out_address,
    uint32_t* out_mapping_access_ordinal, uint64_t* out_mapping_granularity,
    amdf_memory_access_t* out_required_access);

// Applies trusted destination-ordered mappings through one paging timeline.
amdf_status_t amdf_gpu_dxg_memory_reservation_update(
    amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    bool* mutation_started);

// Clears all mappings through the latest publisher's paging timeline.
amdf_status_t amdf_gpu_dxg_memory_reservation_detach(
    amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t access_ordinal);

// Detaches, releases the native address, and consumes reservation metadata.
amdf_status_t amdf_gpu_dxg_memory_reservation_destroy(
    amdf_gpu_dxg_memory_reservation_t* reservation);

// Counts the exact provider-native descriptors required by trusted updates.
uint64_t amdf_gpu_dxg_memory_reservation_mapping_operation_count(
    const amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t update_count,
    const amdf_gpu_umd_mapping_update_t* updates);

// Visits the exact flattened provider-native operations in destination order.
void amdf_gpu_dxg_memory_reservation_visit_mapping_operations(
    const amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t update_count,
    const amdf_gpu_umd_mapping_update_t* updates,
    amdf_gpu_dxg_mapping_operation_visitor_fn_t visitor, void* visitor_context);

// Selects the device whose queue accepted the latest mapping mutation.
void amdf_gpu_dxg_memory_reservation_select_publisher(
    amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t access_ordinal);

// Returns the allocator copied into this reservation owner.
amdf_allocator_t amdf_gpu_dxg_memory_reservation_host_allocator(
    const amdf_gpu_dxg_memory_reservation_t* reservation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_DXG_MEMORY_RESERVATION_H_
