// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_MEMORY_PROFILE_H_
#define AMDF_SRC_GPU_UMD_KFD_MEMORY_PROFILE_H_

#include <stddef.h>

#include "libamdf/src/gpu/umd/kfd/topology.h"
#include "libamdf/src/memory_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Owned SYSTEM profiles precede optional LOCAL and registered profiles.
// Both snoop CPU caches; only control backing uses KFD's coherent GPU policy.
enum amdf_gpu_kfd_memory_profile_ordinal_e {
  AMDF_GPU_KFD_MEMORY_PROFILE_SYSTEM_CACHED = 0,
  AMDF_GPU_KFD_MEMORY_PROFILE_SYSTEM_CONTROL = 1,
  AMDF_GPU_KFD_MEMORY_PROFILE_SYSTEM_COUNT = 2,
};

// Derives complete memory capabilities from qualified topology and host facts.
// This metadata query performs no native operation or device activation.
amdf_status_t amdf_gpu_kfd_query_memory_profile(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    amdf_native_lifetime_t native_lifetime, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_KFD_MEMORY_PROFILE_H_
