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
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"
#include "libamdf/src/xdna/umd/drm/device.h"

amdf_xdna_umd_context_capabilities_t amdf_xdna_umd_query_context_capabilities(
    const amdf_xdna_endpoint_profile_t* profile) {
  amdf_xdna_umd_context_capabilities_t capabilities = {0};
  if ((profile->execution_capabilities &
       AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS) != 0) {
    capabilities.scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    capabilities.placement_modes = AMDF_XDNA_PLACEMENT_MODE_FIXED_FULL_ARRAY;
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

static amdf_status_t amdf_linux_xdna_device_prepare_execution(
    amdf_xdna_umd_device_t* device) {
  const amdf_xdna_endpoint_profile_t* profile = device->profile;
  struct amdxdna_drm_query_aie_metadata metadata = {0};
  struct amdxdna_drm_get_info query = {
      .param = DRM_AMDXDNA_QUERY_AIE_METADATA,
      .buffer_size = sizeof(metadata),
      .buffer = (uintptr_t)&metadata,
  };
  if (ioctl(device->descriptor, DRM_IOCTL_AMDXDNA_GET_INFO, &query) != 0) {
    return amdf_linux_error(errno);
  }
  if (metadata.cols != profile->info->array.column_count ||
      metadata.rows != profile->info->array.row_count ||
      metadata.core.row_count != profile->rows.core_count ||
      metadata.core.row_start != profile->rows.core_origin ||
      metadata.mem.row_count != profile->rows.memory_count ||
      metadata.mem.row_start != profile->rows.memory_origin ||
      metadata.shim.row_count != profile->rows.shim_count ||
      metadata.shim.row_start != profile->rows.shim_origin) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
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
    const amdf_xdna_endpoint_profile_t* profile,
    amdf_allocator_t host_allocator, amdf_xdna_umd_device_t** out_device,
    amdf_xdna_umd_device_result_t* out_result) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_xdna_umd_device_t* device = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*device),
                  amdf_alignof(amdf_xdna_umd_device_t), (void**)&device);
  if (!amdf_status_is_ok(status)) return status;
  device->host_allocator = host_allocator;
  device->profile = profile;
  device->page_size = (size_t)page_size;
  device->descriptor = -1;
  amdf_linux_drm_version_t version;
  status =
      amdf_linux_endpoint_open_file(endpoint, &device->descriptor, &version);
  if (amdf_status_is_ok(status) && (version.major != 0 || version.minor < 8)) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_linux_host_cache_query_line_size(&device->cache_line_size);
  }
  if (amdf_status_is_ok(status) &&
      (profile->execution_capabilities &
       AMDF_XDNA_EXECUTION_CAPABILITY_ELF_INSTRUCTIONS) != 0) {
    status = amdf_linux_xdna_device_prepare_execution(device);
  }
  if (amdf_status_is_ok(status)) {
    amdf_xdna_umd_device_result_t result = {0};
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
    if (!amdf_status_is_ok(close_status)) status = close_status;
  }
  return status;
}
