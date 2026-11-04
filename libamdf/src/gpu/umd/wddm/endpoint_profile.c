// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/endpoint_profile.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/adapter.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/endpoint_properties.h"
#include "libamdf/src/host_memory.h"
#include "libamdf/src/platform/windows/endpoint.h"

amdf_status_t amdf_gpu_umd_create_endpoint_profile(
    amdf_platform_endpoint_t* platform_endpoint,
    amdf_allocator_t host_allocator,
    amdf_gpu_endpoint_profile_t** out_profile) {
  amdf_gpu_wddm_wkmi_loader_t loader = {0};
  amdf_gpu_wddm_wkmi_adapter_t adapter = {0};
  bool profile_available = false;
  amdf_gpu_endpoint_profile_t profile = {0};
  amdf_dxg_bridge_gpu_properties_t provider_properties = {0};
  amdf_status_t status =
      amdf_gpu_wddm_wkmi_loader_initialize(host_allocator, &loader);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_wddm_wkmi_adapter_initialize(
        &loader, platform_endpoint->adapter,
        platform_endpoint->physical_adapter_index, host_allocator, &adapter,
        &provider_properties);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      status = AMDF_STATUS_OK;
    }
  }

  if (amdf_status_is_ok(status) && adapter.native != NULL) {
    amdf_gpu_endpoint_properties_t properties = {0};
    if (amdf_gpu_wddm_wkmi_endpoint_properties_translate(&provider_properties,
                                                         &properties)) {
      const amdf_kmt_api_t* kmt = &platform_endpoint->instance->kmt;
      for (uint32_t i = 0; i < properties.queue_family_count; ++i) {
        amdf_gpu_queue_family_properties_t* family =
            &properties.queue_families[i];
        if ((family->kernel_queue_capabilities &
             AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY) != 0) {
          if (kmt->update_gpu_virtual_address != NULL &&
              kmt->wait_on_hardware_queue != NULL) {
            family->roles |= AMDF_QUEUE_ROLE_VIRTUAL_MEMORY;
          } else {
            family->kernel_queue_capabilities &=
                ~AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY;
          }
        }
      }
      if (amdf_gpu_endpoint_profile_initialize(&properties, &profile)) {
        profile_available = true;
      } else {
        status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
      }
    } else {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }

  if (amdf_status_is_ok(status) && profile_available) {
    const amdf_kmt_api_t* kmt = &platform_endpoint->instance->kmt;
    profile.memory.flags =
        (provider_properties.is_discrete
             ? 0
             : AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED) |
        (kmt->query_video_memory_info ? AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES
                                      : 0);
    if (provider_properties.is_discrete) {
      D3DKMT_SEGMENTGROUPSIZEINFO sizes = {
          .PhysicalAdapterIndex = platform_endpoint->physical_adapter_index,
      };
      status = amdf_kmt_query_adapter_info(kmt, platform_endpoint->adapter,
                                           KMTQAITYPE_GETSEGMENTGROUPSIZE,
                                           &sizes, sizeof(sizes));
      if (amdf_status_is_ok(status)) {
        profile.memory.capacity_byte_length = sizes.LocalMemory;
      }
    }
  }

  amdf_status_t release_status = AMDF_STATUS_OK;
  if (adapter.native != NULL) {
    release_status = amdf_gpu_wddm_wkmi_adapter_deinitialize(&adapter);
  }
  if (amdf_status_is_ok(release_status) && loader.module != NULL) {
    release_status = amdf_gpu_wddm_wkmi_loader_deinitialize(&loader);
  }
  if (!amdf_status_is_ok(release_status)) {
    status = release_status;
  }

  if (amdf_status_is_ok(status) && !profile_available) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_gpu_endpoint_profile_t* owned_profile = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_malloc(host_allocator, sizeof(profile),
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
  const amdf_kmt_api_t* kmt = &endpoint->instance->kmt;
  D3DKMT_QUERYVIDEOMEMORYINFO query = {
      .hProcess = GetCurrentProcess(),
      .hAdapter = endpoint->adapter,
      .MemorySegmentGroup = D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL,
      .PhysicalAdapterIndex = endpoint->physical_adapter_index,
  };
  amdf_status_t status =
      amdf_kmt_make_status(kmt->query_video_memory_info(&query));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_memory_scope_observation_t observation = {0};
  if (flags & AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED) {
    status = amdf_host_memory_sample_observation(&observation);
  } else {
    D3DKMT_SEGMENTGROUPSIZEINFO sizes = {
        .PhysicalAdapterIndex = endpoint->physical_adapter_index,
    };
    status = amdf_kmt_query_adapter_info(kmt, endpoint->adapter,
                                         KMTQAITYPE_GETSEGMENTGROUPSIZE, &sizes,
                                         sizeof(sizes));
    observation.capacity_byte_length = sizes.LocalMemory;
    observation.available_byte_length = sizes.LocalMemory;
  }
  if (amdf_status_is_ok(status)) {
    const uint64_t headroom = query.CurrentUsage < query.Budget
                                  ? query.Budget - query.CurrentUsage
                                  : 0;
    out_observation->capacity_byte_length = observation.capacity_byte_length;
    out_observation->available_byte_length =
        observation.available_byte_length < headroom
            ? observation.available_byte_length
            : headroom;
  }
  return status;
}
