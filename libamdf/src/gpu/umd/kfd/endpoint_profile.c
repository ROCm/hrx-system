// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/endpoint_profile.h"

#include <string.h>
#include <unistd.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/memory_profile.h"
#include "libamdf/src/gpu/umd/kfd/target/memory.h"
#include "libamdf/src/gpu/umd/kfd/target/user_queue.h"
#include "libamdf/src/gpu/umd/kfd/topology.h"
#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"

static amdf_status_t amdf_gpu_kfd_qualify_endpoint_profile(
    amdf_platform_endpoint_t* endpoint, amdf_native_lifetime_t native_lifetime,
    amdf_gpu_kfd_topology_t* topology,
    amdf_gpu_endpoint_profile_t* out_profile) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || page_size > UINT32_MAX ||
      (page_size & (page_size - 1)) != 0 ||
      !amdf_gpu_kfd_target_memory_initialize(endpoint->info.pci.device_id,
                                             (uint32_t)page_size, topology)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  // These are expected implemented policies, not installed-ABI qualification.
  // Opening /dev/kfd creates process state. Explicit device creation qualifies
  // the actual connection before selecting its requested lifetime.
  topology->properties.native_lifetimes[AMDF_NATIVE_LIFETIME_PROCESS] =
      (amdf_gpu_lifetime_properties_t){
          .supported = true,
          .features = topology->memory_features |
                      AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION |
                      AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION,
      };
  topology->properties.native_lifetimes[AMDF_NATIVE_LIFETIME_INSTANCE] =
      (amdf_gpu_lifetime_properties_t){
          .supported = true,
          .features = topology->memory_features |
                      AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION,
      };
  uint32_t cache_line_size = 0;
  amdf_status_t status =
      amdf_linux_host_cache_query_line_size(&cache_line_size);
  if (!amdf_status_is_ok(status)) return status;
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
  profile.memory.count = 1 +
                         ((topology->memory_features &
                           AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) != 0) +
                         (native_lifetime == AMDF_NATIVE_LIFETIME_PROCESS);
  for (uint32_t i = 0; i < profile.memory.count; ++i) {
    const amdf_status_t memory_status = amdf_gpu_kfd_query_memory_profile(
        topology, (size_t)page_size, native_lifetime, i,
        &profile.memory.values[i]);
    amdf_assert(amdf_status_is_ok(memory_status));
    (void)memory_status;
  }
  *out_profile = profile;
  return AMDF_STATUS_OK;
}

// One endpoint-owned allocation; all profile metadata pointers remain inside
// it.
typedef struct amdf_gpu_kfd_endpoint_profile_t {
  // Provider-neutral header consumed by the common GPU extension.
  amdf_gpu_endpoint_profile_t base;
  // Native immutable facts used for placement-relative member qualification.
  amdf_gpu_kfd_topology_t topology;
} amdf_gpu_kfd_endpoint_profile_t;

amdf_status_t amdf_gpu_umd_create_endpoint_profile(
    amdf_platform_endpoint_t* endpoint, amdf_native_lifetime_t native_lifetime,
    amdf_allocator_t host_allocator,
    amdf_gpu_endpoint_profile_t** out_profile) {
  amdf_gpu_kfd_topology_t topology = {0};
  amdf_status_t status =
      amdf_gpu_kfd_topology_initialize(endpoint, host_allocator, &topology);
  if (!amdf_status_is_ok(status)) return status;
  amdf_gpu_endpoint_profile_t profile;
  status = amdf_gpu_kfd_qualify_endpoint_profile(endpoint, native_lifetime,
                                                 &topology, &profile);
  amdf_gpu_kfd_endpoint_profile_t* owned_profile = NULL;
  const uint64_t peer_byte_length =
      (uint64_t)topology.memory_peers.count * sizeof(uint32_t);
  if (amdf_status_is_ok(status)) {
    status = amdf_malloc(
        host_allocator, sizeof(*owned_profile) + peer_byte_length,
        amdf_alignof(amdf_gpu_kfd_endpoint_profile_t), (void**)&owned_profile);
  }
  if (amdf_status_is_ok(status)) {
    owned_profile->base = profile;
    owned_profile->topology = topology;
    owned_profile->topology.memory_peers.gpu_ids =
        (uint32_t*)(owned_profile + 1);
    if (peer_byte_length != 0) {
      memcpy(owned_profile->topology.memory_peers.gpu_ids,
             topology.memory_peers.gpu_ids, (size_t)peer_byte_length);
    }
    for (uint32_t i = 0; i < owned_profile->base.memory.count; ++i) {
      owned_profile->base.memory.values[i].construction.data =
          &owned_profile->topology;
    }
    *out_profile = &owned_profile->base;
  }
  amdf_gpu_kfd_topology_deinitialize(&topology, host_allocator);
  return status;
}
