// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/device.h"

#include <drm/amdxdna_accel.h>
#include <stddef.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/host_reservation.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"
#include "libamdf/src/xdna/umd/drm/device.h"

amdf_xdna_umd_context_capabilities_t amdf_xdna_umd_query_context_capabilities(
    const amdf_xdna_device_profile_t* profile) {
  amdf_xdna_umd_context_capabilities_t capabilities = {0};
  if ((profile->execution_capabilities &
       AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS) != 0) {
    capabilities.scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    capabilities.placement_modes = AMDF_XDNA_PLACEMENT_MODE_FIXED_FULL_ARRAY;
    capabilities.instruction_features = AMDF_XDNA_INSTRUCTION_FEATURE_LOAD_PDI;
  }
  return capabilities;
}

amdf_status_t amdf_xdna_umd_device_destroy(amdf_xdna_umd_device_t* device) {
  amdf_status_t status =
      amdf_linux_xdna_buffer_deinitialize(device->descriptor, &device->heap);
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_file_close(&device->descriptor);
  }
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = device->host_allocator;
    amdf_free(host_allocator, device);
  }
  return status;
}

static amdf_status_t amdf_linux_xdna_device_query_tiles(
    amdf_xdna_umd_device_t* device,
    amdf_xdna_umd_tile_metadata_t* out_metadata) {
  struct amdxdna_drm_query_aie_metadata metadata = {0};
  struct amdxdna_drm_get_info query = {
      .param = DRM_AMDXDNA_QUERY_AIE_METADATA,
      .buffer_size = sizeof(metadata),
      .buffer = (uintptr_t)&metadata,
  };
  if (ioctl(device->descriptor, DRM_IOCTL_AMDXDNA_GET_INFO, &query) != 0) {
    return amdf_linux_error(errno);
  }
  if (metadata.cols == 0 || metadata.rows == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_metadata = (amdf_xdna_umd_tile_metadata_t){
      .column_count = metadata.cols,
      .row_count = metadata.rows,
      .core_origin = metadata.core.row_start,
      .core_count = metadata.core.row_count,
      .memory_origin = metadata.mem.row_start,
      .memory_count = metadata.mem.row_count,
      .shim_origin = metadata.shim.row_start,
      .shim_count = metadata.shim.row_count,
  };
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_linux_xdna_device_prepare_execution(
    amdf_xdna_umd_device_t* device) {
  const amdf_xdna_device_profile_t* profile = device->profile;
  if (device->page_size > profile->firmware_heap_byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_status_t status = amdf_linux_xdna_buffer_create(
      device->descriptor, AMDXDNA_BO_DEV_HEAP,
      profile->firmware_heap_byte_length, &device->heap);
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_buffer_attach(
        device->descriptor, profile->firmware_heap_byte_length,
        device->page_size, NULL, &device->heap);
  }
  return status;
}

amdf_status_t amdf_xdna_umd_device_create(
    amdf_platform_endpoint_t* endpoint,
    const amdf_xdna_device_profile_t* profile, amdf_allocator_t host_allocator,
    amdf_xdna_umd_device_t** out_device,
    amdf_xdna_umd_device_result_t* out_result) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_xdna_umd_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*device),
                  amdf_alignof(amdf_xdna_umd_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  device->host_allocator = host_allocator;
  device->profile = profile;
  device->page_size = (size_t)page_size;
  device->descriptor = -1;
  status = amdf_linux_endpoint_open_file(endpoint, &device->descriptor);
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_host_cache_query_line_size(&device->cache_line_size);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_buffer_query_sva(
        device->descriptor, device->page_size, &device->uses_sva);
  }
  amdf_xdna_umd_device_result_t result = {0};
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_xdna_device_query_tiles(device, &result.tiles);
  }
  if (amdf_status_is_ok(status) &&
      (profile->execution_capabilities &
       AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS) != 0) {
    status = amdf_linux_xdna_device_prepare_execution(device);
  }
  if (amdf_status_is_ok(status)) {
    result.id.words[0] = (uintptr_t)device;
    result.id.words[1] = endpoint->info.id.words[1];
    result.reset_epoch = 1;
    result.placement_modes =
        amdf_xdna_umd_query_context_capabilities(profile).placement_modes;
    *out_result = result;
    *out_device = device;
  } else {
    // Construction has no context or accepted work. Close the local file even
    // if heap cleanup fails; neither native owner borrows device metadata.
    const amdf_status_t release_status =
        amdf_linux_xdna_buffer_deinitialize(device->descriptor, &device->heap);
    const amdf_status_t close_status =
        amdf_linux_file_close(&device->descriptor);
    amdf_free(host_allocator, device);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
  }
  return status;
}

amdf_status_t amdf_xdna_umd_device_query_memory_reservation_profile(
    amdf_xdna_umd_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_profile_t* out_profile) {
  if (address_domain_ordinal != 0 || !device->uses_sva) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint64_t bias = device->profile->dma.byte_offset;
  const uint64_t maximum =
      (UINT64_C(1) << device->profile->dma.address_bit_count) - 1;
  const uint64_t page_size = device->page_size;
  *out_profile = (amdf_memory_reservation_profile_t){
      .capabilities =
          {
              .supported_access =
                  AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
              .required_access = AMDF_MEMORY_ACCESS_READ,
              .address_domain_ordinal = 0,
              .maximum_byte_length = (maximum - bias) & ~(page_size - 1),
              .reservation_granularity = page_size,
              .maximum_alignment =
                  bias != 0 ? bias & (~bias + 1) : (SIZE_MAX / 2) + 1,
              .mapping_granularity = page_size,
              .minimum_address = page_size + bias,
              .maximum_address = maximum,
              .required_source_flags = AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE,
          },
      .vtable = &amdf_host_reservation_vtable,
      .host =
          {
              .address_byte_offset = bias,
              .address_kind = AMDF_MEMORY_ADDRESS_XDNA_DMA,
              .page_size = page_size,
              .prepare_translations =
                  amdf_linux_host_cache_prepare_translations,
          },
  };
  return AMDF_STATUS_OK;
}
