// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/endpoint_profile.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/endpoint_properties.h"
#include "libamdf/src/gpu/umd/dxg/linux/logical_device.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/loader.h"
#include "libamdf/src/host_memory.h"
#include "libamdf/src/platform/linux/dxg/endpoint.h"

static amdf_status_t amdf_gpu_linux_dxg_make_status(NTSTATUS native_status) {
  return native_status == 0 ? AMDF_STATUS_OK
                            : amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS,
                                               (uint32_t)native_status);
}

static amdf_status_t amdf_gpu_linux_dxg_query_local_capacity(
    const amdf_platform_endpoint_t* endpoint, uint64_t* out_capacity) {
  D3DKMT_SEGMENTGROUPSIZEINFO sizes = {
      .PhysicalAdapterIndex = endpoint->physical_adapter_index,
  };
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = endpoint->adapter,
      .Type = KMTQAITYPE_GETSEGMENTGROUPSIZE,
      .pPrivateDriverData = &sizes,
      .PrivateDriverDataSize = sizeof(sizes),
  };
  const amdf_status_t status = amdf_gpu_linux_dxg_make_status(
      amdf_linux_dxg_kmt_query_adapter_info(&endpoint->instance->kmt, &query));
  if (amdf_status_is_ok(status)) {
    *out_capacity = sizes.LocalMemory;
  }
  return status;
}

amdf_status_t amdf_gpu_umd_create_endpoint_profile(
    amdf_platform_endpoint_t* endpoint, amdf_allocator_t host_allocator,
    amdf_gpu_endpoint_profile_t** out_profile) {
  amdf_gpu_dxg_rocdxg_loader_t loader = {0};
  amdf_gpu_linux_dxg_logical_device_t logical_device = {0};
  amdf_dxg_bridge_gpu_properties_t provider_properties = {0};
  amdf_gpu_endpoint_profile_t profile = {0};

  amdf_status_t status =
      amdf_gpu_dxg_rocdxg_loader_initialize(host_allocator, &loader);
  const amdf_dxg_bridge_api_t* bridge_api = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_dxg_rocdxg_loader_query_api(&loader, &bridge_api);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_linux_dxg_logical_device_initialize(
        bridge_api, endpoint, host_allocator, &logical_device,
        &provider_properties);
  }
  if (amdf_status_is_ok(status)) {
    amdf_gpu_endpoint_properties_t properties = {0};
    if (!amdf_gpu_dxg_endpoint_properties_translate(&provider_properties,
                                                    &properties) ||
        !amdf_gpu_endpoint_profile_initialize(&properties, &profile)) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }

  if (logical_device.native != 0) {
    amdf_gpu_linux_dxg_logical_device_deinitialize(&logical_device);
  }
  if (loader.module != NULL) {
    amdf_gpu_dxg_rocdxg_loader_deinitialize(&loader);
  }

  if (amdf_status_is_ok(status)) {
    profile.memory.flags = AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES;
    if (provider_properties.is_discrete != 0) {
      status = amdf_gpu_linux_dxg_query_local_capacity(
          endpoint, &profile.memory.capacity_byte_length);
    } else {
      profile.memory.flags |= AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED;
    }
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
  return status;
}

amdf_status_t amdf_gpu_umd_sample_memory_scope(
    const amdf_platform_endpoint_t* endpoint, amdf_memory_scope_flags_t flags,
    amdf_memory_scope_observation_t* out_observation) {
  D3DKMT_QUERYVIDEOMEMORYINFO query = {
      // Linux DXG identifies the current process with a zero process handle.
      .hProcess = 0,
      .hAdapter = endpoint->adapter,
      .MemorySegmentGroup = D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL,
      .PhysicalAdapterIndex = endpoint->physical_adapter_index,
  };
  amdf_status_t status =
      amdf_gpu_linux_dxg_make_status(amdf_linux_dxg_kmt_query_video_memory_info(
          &endpoint->instance->kmt, &query));
  amdf_memory_scope_observation_t observation = {0};
  if (amdf_status_is_ok(status)) {
    if ((flags & AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED) != 0) {
      status = amdf_host_memory_sample_observation(&observation);
    } else {
      status = amdf_gpu_linux_dxg_query_local_capacity(
          endpoint, &observation.capacity_byte_length);
      observation.available_byte_length = observation.capacity_byte_length;
    }
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t headroom = query.CurrentUsage < query.Budget
                                  ? query.Budget - query.CurrentUsage
                                  : 0;
    observation.available_byte_length =
        observation.available_byte_length < headroom
            ? observation.available_byte_length
            : headroom;
    *out_observation = observation;
  }
  return status;
}
