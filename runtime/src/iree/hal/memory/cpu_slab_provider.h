// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_CPU_SLAB_PROVIDER_H_
#define IREE_HAL_MEMORY_CPU_SLAB_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/memory/slab_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Memory type reported by the CPU slab provider.
#define IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE                           \
  (IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | \
   IREE_HAL_MEMORY_TYPE_HOST_COHERENT | IREE_HAL_MEMORY_TYPE_HOST_CACHED)

// Buffer usage flags supported by CPU slab allocations.
#define IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE                      \
  (IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH | \
   IREE_HAL_BUFFER_USAGE_SHARING_EXPORT |                            \
   IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |                         \
   IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |                        \
   IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE |                         \
   IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |                            \
   IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |                        \
   IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |                          \
   IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |                     \
   IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE)

// Creates a slab provider that allocates host memory via |host_allocator|
// (typically the system allocator / malloc). Slabs are plain host memory
// with no special alignment, registration, or NUMA affinity.
//
// Reports IREE_HAL_CPU_SLAB_PROVIDER_MEMORY_TYPE and supports
// IREE_HAL_CPU_SLAB_PROVIDER_BUFFER_USAGE.
//
// This is the simplest slab provider; intended for CPU-only testing and as
// the backing for pass-through pools on host targets.
iree_status_t iree_hal_cpu_slab_provider_create(
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_CPU_SLAB_PROVIDER_H_
