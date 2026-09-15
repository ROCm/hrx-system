// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_MEMORY_PROFILE_H_
#define AMDF_SRC_GPU_UMD_WDDM_MEMORY_PROFILE_H_

#include "libamdf/src/memory_profile.h"
#include "libamdf/src/platform/endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define AMDF_WINDOWS_GPU_PAGE_SIZE UINT64_C(4096)
#define AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY UINT64_C(65536)

// Immutable KMT memory capabilities captured for one physical GPU adapter.
typedef struct amdf_windows_gpu_memory_capabilities_t {
  // Native GPU virtual-address width in bits.
  uint32_t virtual_address_bit_count;
  // Nonzero when GPU mappings can deny device writes.
  uint32_t read_only_memory_supported;
  // Nonzero when GPU mappings can deny instruction fetches.
  uint32_t no_execute_memory_supported;
  // Nonzero when the GPU MMU supports coherent system-memory mappings.
  uint32_t cache_coherent_memory_supported;
} amdf_windows_gpu_memory_capabilities_t;

// Queries immutable GPU MMU facts through the existing endpoint adapter.
// No logical device, paging queue or allocation is created.
amdf_status_t amdf_windows_gpu_query_memory_capabilities(
    const amdf_platform_endpoint_t* endpoint,
    amdf_windows_gpu_memory_capabilities_t* out_capabilities);

// Derives complete memory capabilities from qualified adapter facts.
// This metadata query performs no native operation or device activation.
amdf_status_t amdf_gpu_wddm_query_memory_profile(
    const amdf_windows_gpu_memory_capabilities_t* capabilities,
    uint32_t memory_profile_ordinal, amdf_memory_native_profile_t* out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_WDDM_MEMORY_PROFILE_H_
