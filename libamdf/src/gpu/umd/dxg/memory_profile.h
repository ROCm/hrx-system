// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_DXG_MEMORY_PROFILE_H_
#define AMDF_SRC_GPU_UMD_DXG_MEMORY_PROFILE_H_

#include "libamdf/src/memory_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define AMDF_GPU_DXG_PAGE_SIZE UINT64_C(4096)
#define AMDF_GPU_DXG_RESERVATION_GRANULARITY UINT64_C(65536)

// Immutable DXG memory capabilities captured for one physical GPU adapter.
typedef struct amdf_gpu_dxg_memory_capabilities_t {
  // Native GPU virtual-address width in bits.
  uint32_t virtual_address_bit_count;
  // Required host-pointer, logical-length, and placement granularity for
  // native host registration.
  uint64_t host_registration_granularity;
  // Nonzero when GPU mappings can deny device writes.
  uint32_t read_only_memory_supported;
  // Nonzero when GPU mappings can deny instruction fetches.
  uint32_t no_execute_memory_supported;
  // Nonzero when the GPU MMU supports coherent system-memory mappings.
  uint32_t cache_coherent_memory_supported;
  // Nonzero when naturally aligned GPU 64-bit updates are atomic to the CPU.
  uint32_t atomic_update_64_supported;
  // Nonzero when this host can import and export D3D12 buffer resources.
  uint32_t d3d12_resource_supported;
  // GPU shader ISA generation supplied by the parsed native adapter record.
  uint32_t program_gfx_major;
  // Nonzero when the CPU-facing route supports both 32/64-bit AtomicOps.
  uint32_t program_host_rmw_supported;
  // Nonzero for an integrated CPU/GPU fabric with integer bitwise atomics.
  uint32_t program_host_is_fabric;
  // Qualified host cache-maintenance program for CPU-visible memory.
  amdf_cache_transition_t host_cache_transition;
} amdf_gpu_dxg_memory_capabilities_t;

// Derives complete memory capabilities from qualified adapter facts.
// This metadata query performs no native operation or device activation.
amdf_status_t amdf_gpu_dxg_query_memory_profile(
    const amdf_gpu_dxg_memory_capabilities_t* capabilities,
    uint32_t memory_profile_ordinal, amdf_memory_native_profile_t* out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_DXG_MEMORY_PROFILE_H_
