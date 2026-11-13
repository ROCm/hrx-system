// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/adapter.h"
#include "libamdf/src/gpu/umd/dxg/endpoint_properties.h"
#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/dxg/linux/memory_profile.h"
#include "libamdf/src/gpu/umd/dxg/wsl/api.h"
#include "libamdf/src/host_memory.h"
#include "libamdf/src/platform/linux/dxg/endpoint.h"

amdf_status_t amdf_gpu_linux_dxg_create_endpoint_profile(
    amdf_platform_endpoint_t* endpoint, amdf_allocator_t host_allocator,
    amdf_gpu_endpoint_profile_t** out_profile) {
  amdf_gpu_dxg_adapter_t adapter = {0};
  amdf_dxg_provider_gpu_properties_t provider_properties = {0};
  amdf_gpu_endpoint_profile_t profile = {0};

  const amdf_dxg_provider_gpu_adapter_open_info_t open_info = {
      .adapter_handle = endpoint->adapter,
      .physical_adapter_index = endpoint->physical_adapter_index,
      .adapter_luid = endpoint->info.native_identity.value.windows_adapter.luid,
      .native_transport =
          {
              .context = (void*)&endpoint->instance->kmt,
              .invoke = amdf_linux_dxg_kmt_invoke,
          },
  };
  amdf_status_t status = amdf_gpu_dxg_adapter_initialize(
      amdf_gpu_dxg_wsl_default_api(), &open_info, host_allocator, &adapter,
      &provider_properties);
  if (amdf_status_is_ok(status)) {
    amdf_gpu_endpoint_properties_t properties = {0};
    if (!amdf_gpu_dxg_endpoint_properties_translate(&provider_properties,
                                                    &properties) ||
        !amdf_gpu_endpoint_profile_initialize(&properties, &profile)) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }

  if (adapter.native != NULL) {
    const amdf_status_t release_status =
        amdf_gpu_dxg_adapter_deinitialize(&adapter);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
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

amdf_status_t amdf_gpu_linux_dxg_sample_memory_scope(
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
      amdf_linux_dxg_kmt_make_status(amdf_linux_dxg_kmt_query_video_memory_info(
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
