// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory_reservation.h"

#include "libamdf/src/gpu/umd/dxg/linux/device.h"
#include "libamdf/src/gpu/umd/dxg/linux/memory.h"
#include "libamdf/src/gpu/umd/dxg/memory_reservation.h"

static amdf_gpu_dxg_memory_reservation_device_t
amdf_gpu_linux_dxg_reservation_device(amdf_gpu_umd_device_t* device) {
  const amdf_gpu_dxg_memory_device_t memory_device =
      amdf_gpu_linux_dxg_memory_device(device);
  return (amdf_gpu_dxg_memory_reservation_device_t){
      .native_api = memory_device.native_api,
      .native_context = memory_device.native_context,
      .adapter_handle = memory_device.adapter_handle,
      .adapter_luid = memory_device.adapter_luid,
      .physical_adapter_index = device->physical_adapter_index,
      .virtual_address_bit_count =
          device->memory_capabilities.virtual_address_bit_count,
      .read_only_memory_supported =
          device->memory_capabilities.read_only_memory_supported,
      .no_execute_memory_supported =
          device->memory_capabilities.no_execute_memory_supported,
      .device_handle = memory_device.device_handle,
      .paging_queue_handle = memory_device.paging_queue_handle,
      .paging_synchronization_object_handle =
          memory_device.paging_synchronization_object_handle,
      .paging_progress_fence = memory_device.paging_progress_fence,
  };
}

static amdf_gpu_dxg_memory_reservation_device_t
amdf_gpu_linux_dxg_describe_reservation_device(const void* context,
                                               uint32_t ordinal) {
  amdf_gpu_umd_device_t* const* devices =
      (amdf_gpu_umd_device_t* const*)context;
  return amdf_gpu_linux_dxg_reservation_device(devices[ordinal]);
}

bool amdf_gpu_umd_devices_share_mapping_domain(const amdf_gpu_umd_device_t* a,
                                               const amdf_gpu_umd_device_t* b) {
  const amdf_gpu_dxg_memory_reservation_device_t a_description =
      amdf_gpu_linux_dxg_reservation_device((amdf_gpu_umd_device_t*)a);
  const amdf_gpu_dxg_memory_reservation_device_t b_description =
      amdf_gpu_linux_dxg_reservation_device((amdf_gpu_umd_device_t*)b);
  return amdf_gpu_dxg_devices_share_mapping_domain(&a_description,
                                                   &b_description);
}

amdf_status_t amdf_gpu_umd_query_memory_reservation_capabilities(
    amdf_gpu_umd_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  const amdf_gpu_dxg_memory_reservation_device_t description =
      amdf_gpu_linux_dxg_reservation_device(device);
  return amdf_gpu_dxg_query_memory_reservation_capabilities(
      &description, address_domain_ordinal, out_capabilities);
}

amdf_status_t amdf_gpu_umd_memory_reservation_create(
    amdf_allocator_t host_allocator, uint32_t device_count,
    amdf_gpu_umd_device_t* const* devices, uint64_t byte_length,
    uint64_t minimum_alignment, uint64_t common_address,
    amdf_gpu_umd_memory_reservation_t** out_reservation) {
  return amdf_gpu_dxg_memory_reservation_create(
      host_allocator, device_count,
      amdf_gpu_linux_dxg_describe_reservation_device, devices, byte_length,
      minimum_alignment, common_address,
      (amdf_gpu_dxg_memory_reservation_t**)out_reservation);
}

void amdf_gpu_umd_memory_reservation_query_access(
    const amdf_gpu_umd_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint64_t* out_address,
    uint32_t* out_mapping_access_ordinal, uint64_t* out_mapping_granularity,
    amdf_memory_access_t* out_required_access) {
  amdf_gpu_dxg_memory_reservation_query_access(
      (const amdf_gpu_dxg_memory_reservation_t*)reservation, access_ordinal,
      out_address, out_mapping_access_ordinal, out_mapping_granularity,
      out_required_access);
}

amdf_status_t amdf_gpu_umd_memory_reservation_update(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    bool* mutation_started) {
  return amdf_gpu_dxg_memory_reservation_update(
      (amdf_gpu_dxg_memory_reservation_t*)reservation, access_ordinal,
      update_count, updates, mutation_started);
}

amdf_status_t amdf_gpu_umd_memory_reservation_update_on_queue(
    amdf_gpu_umd_memory_reservation_t* reservation,
    amdf_gpu_umd_kernel_queue_t* queue, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    uint64_t submission, bool* mutation_started, bool* accepted) {
  (void)reservation;
  (void)queue;
  (void)access_ordinal;
  (void)update_count;
  (void)updates;
  (void)submission;
  (void)mutation_started;
  (void)accepted;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_gpu_umd_memory_reservation_detach(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal) {
  return amdf_gpu_dxg_memory_reservation_detach(
      (amdf_gpu_dxg_memory_reservation_t*)reservation, access_ordinal);
}

amdf_status_t amdf_gpu_umd_memory_reservation_destroy(
    amdf_gpu_umd_memory_reservation_t* reservation) {
  return amdf_gpu_dxg_memory_reservation_destroy(
      (amdf_gpu_dxg_memory_reservation_t*)reservation);
}
