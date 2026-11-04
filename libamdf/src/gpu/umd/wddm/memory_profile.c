// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/memory_profile.h"

#include "libamdf/src/platform/host_memory.h"
#include "libamdf/src/platform/windows/endpoint.h"
#include "libamdf/src/platform/windows/kmt_api.h"

amdf_status_t amdf_gpu_wddm_query_memory_capabilities(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_dxg_memory_capabilities_t* out_capabilities) {
  D3DKMT_QUERY_GPUMMU_CAPS query = {0};
  query.PhysicalAdapterIndex = endpoint->physical_adapter_index;
  const amdf_status_t status = amdf_kmt_query_adapter_info(
      &endpoint->instance->kmt, endpoint->adapter, KMTQAITYPE_QUERY_GPUMMU_CAPS,
      &query, sizeof(query));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (query.Caps.VirtualAddressBitCount == 0 ||
      query.Caps.VirtualAddressBitCount > 64) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  D3DKMT_WDDM_2_0_CAPS scheduling = {0};
  const amdf_status_t scheduling_status = amdf_kmt_query_adapter_info(
      &endpoint->instance->kmt, endpoint->adapter, KMTQAITYPE_WDDM_2_0_CAPS,
      &scheduling, sizeof(scheduling));
  if (!amdf_status_is_ok(scheduling_status)) {
    return scheduling_status;
  }

  amdf_gpu_dxg_memory_capabilities_t capabilities = {
      .virtual_address_bit_count = query.Caps.VirtualAddressBitCount,
      .host_registration_granularity = AMDF_GPU_DXG_PAGE_SIZE,
      .read_only_memory_supported = query.Caps.Flags.ReadOnlyMemorySupported,
      .no_execute_memory_supported = query.Caps.Flags.NoExecuteMemorySupported,
      .cache_coherent_memory_supported =
          query.Caps.Flags.CacheCoherentMemorySupported,
      .atomic_update_64_supported = scheduling.Support64BitAtomics,
      .d3d12_resource_supported =
          amdf_kmt_api_supports_shared_resource_import(
              &endpoint->instance->kmt) &&
          amdf_d3d12_api_supports_buffer_import(&endpoint->instance->d3d12),
  };
  const amdf_status_t host_status =
      amdf_platform_host_memory_query_cache_transition(
          &capabilities.host_cache_transition);
  if (!amdf_status_is_ok(host_status)) {
    return host_status;
  }
  *out_capabilities = capabilities;
  return AMDF_STATUS_OK;
}
