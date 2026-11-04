// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/memory_profile.h"

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

  const amdf_gpu_dxg_memory_capabilities_t capabilities = {
      .virtual_address_bit_count = query.Caps.VirtualAddressBitCount,
      .read_only_memory_supported = query.Caps.Flags.ReadOnlyMemorySupported,
      .no_execute_memory_supported = query.Caps.Flags.NoExecuteMemorySupported,
      .cache_coherent_memory_supported =
          query.Caps.Flags.CacheCoherentMemorySupported,
      .atomic_update_64_supported = scheduling.Support64BitAtomics,
      .d3d12_resource_supported = 1,
  };
  *out_capabilities = capabilities;
  return AMDF_STATUS_OK;
}
