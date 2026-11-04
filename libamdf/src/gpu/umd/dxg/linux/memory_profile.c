// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/linux/memory_profile.h"

#include "libamdf/src/platform/host_memory.h"

static amdf_status_t amdf_gpu_linux_dxg_query_adapter_info(
    const amdf_platform_endpoint_t* endpoint, KMTQUERYADAPTERINFOTYPE type,
    void* data, uint32_t data_size) {
  D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = endpoint->adapter,
      .Type = type,
      .pPrivateDriverData = data,
      .PrivateDriverDataSize = data_size,
  };
  return amdf_linux_dxg_kmt_make_status(
      amdf_linux_dxg_kmt_query_adapter_info(&endpoint->instance->kmt, &query));
}

amdf_status_t amdf_gpu_linux_dxg_query_memory_capabilities(
    const amdf_platform_endpoint_t* endpoint,
    amdf_gpu_dxg_memory_capabilities_t* out_capabilities) {
  D3DKMT_QUERY_GPUMMU_CAPS memory = {
      .PhysicalAdapterIndex = endpoint->physical_adapter_index,
  };
  amdf_status_t status = amdf_gpu_linux_dxg_query_adapter_info(
      endpoint, KMTQAITYPE_QUERY_GPUMMU_CAPS, &memory, sizeof(memory));
  if (amdf_status_is_ok(status) && (memory.Caps.VirtualAddressBitCount == 0 ||
                                    memory.Caps.VirtualAddressBitCount > 64)) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  D3DKMT_WDDM_2_0_CAPS scheduling = {0};
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_linux_dxg_query_adapter_info(
        endpoint, KMTQAITYPE_WDDM_2_0_CAPS, &scheduling, sizeof(scheduling));
  }
  if (amdf_status_is_ok(status)) {
    amdf_gpu_dxg_memory_capabilities_t capabilities = {
        .virtual_address_bit_count = memory.Caps.VirtualAddressBitCount,
        .read_only_memory_supported = memory.Caps.Flags.ReadOnlyMemorySupported,
        .no_execute_memory_supported =
            memory.Caps.Flags.NoExecuteMemorySupported,
        .cache_coherent_memory_supported =
            memory.Caps.Flags.CacheCoherentMemorySupported,
        .atomic_update_64_supported = scheduling.Support64BitAtomics,
    };
    status = amdf_platform_host_memory_query_cache_transition(
        &capabilities.host_cache_transition);
    if (amdf_status_is_ok(status)) {
      *out_capabilities = capabilities;
    }
  }
  return status;
}
