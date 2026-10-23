// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/endpoint_profile.h"

#include <unistd.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"
#include "libamdf/src/gpu/umd/kfd/topology.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"

static amdf_status_t amdf_gpu_kfd_qualify_endpoint_profile(
    amdf_gpu_kfd_topology_t* topology,
    amdf_gpu_endpoint_profile_t* out_profile) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || page_size > UINT32_MAX ||
      (page_size & (page_size - 1)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  uint32_t cache_line_size = 0;
  amdf_status_t status =
      amdf_linux_host_cache_query_line_size(&cache_line_size);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_gpu_kfd_user_queue_plans_t queue_plans;
  amdf_gpu_kfd_target_user_queue_plans_initialize(
      topology, (size_t)page_size, cache_line_size, &queue_plans);
  for (uint32_t i = 0; i < queue_plans.count; ++i) {
    topology->properties
        .queue_families[topology->properties.queue_family_count++] =
        queue_plans.values[i].family;
  }
  amdf_gpu_endpoint_profile_t profile;
  if (!amdf_gpu_endpoint_profile_initialize(&topology->properties, &profile)) {
    return amdf_linux_error(EPROTO);
  }
  *out_profile = profile;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_create_endpoint_profile(
    amdf_platform_endpoint_t* endpoint, amdf_allocator_t host_allocator,
    amdf_gpu_endpoint_profile_t** out_profile) {
  amdf_gpu_kfd_topology_t topology = {0};
  amdf_status_t status =
      amdf_gpu_kfd_topology_initialize(endpoint, host_allocator, &topology);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  // INFO_DEV_INFO reports cached package and virtualization facts. In
  // particular, full CPU-visible VRAM does not imply that a virtual function
  // exposes the register mapping required to publish host-written rings.
  // This short-lived render file never acquires a KFD VM or execution storage.
  int descriptor = -1;
  status = amdf_linux_endpoint_open_file(endpoint, &descriptor);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_topology_query_memory(
        descriptor, endpoint->info.pci.device_id, &topology);
  }
  const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  amdf_gpu_endpoint_profile_t profile;
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_qualify_endpoint_profile(&topology, &profile);
  }
  amdf_gpu_endpoint_profile_t* owned_profile = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_malloc(host_allocator, sizeof(*owned_profile),
                         amdf_alignof(amdf_gpu_endpoint_profile_t),
                         (void**)&owned_profile);
  }
  if (amdf_status_is_ok(status)) {
    *owned_profile = profile;
    *out_profile = owned_profile;
  }
  amdf_gpu_kfd_topology_deinitialize(&topology, host_allocator);
  return status;
}
