// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/host_memory_pools.h"

// Pool and legacy-region iteration have different scopes: ROCr's legacy CPU
// region iterator includes peer NUMA nodes. Select handles as they arrive;
// neither native interface bounds the number of candidates.
typedef struct iree_hal_amdgpu_host_memory_query_t {
  // Borrowed native query interface for the duration of enumeration.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Selected handles, published only after every required query succeeds.
  iree_hal_amdgpu_host_memory_pools_t pools;
  // Whether the current fine pool is also a kernarg pool and may be replaced
  // by the first ordinary fine pool.
  bool fine_pool_is_kernarg;
} iree_hal_amdgpu_host_memory_query_t;

static hsa_status_t iree_hal_amdgpu_host_memory_query_pool(
    hsa_amd_memory_pool_t pool, void* user_data) {
  iree_hal_amdgpu_host_memory_query_t* query =
      (iree_hal_amdgpu_host_memory_query_t*)user_data;

  hsa_region_segment_t segment = 0;
  hsa_status_t status = iree_hsa_amd_memory_pool_get_info_raw(
      query->libhsa, pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  bool accessible_by_all = false;
  status = iree_hsa_amd_memory_pool_get_info_raw(
      query->libhsa, pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
      &accessible_by_all);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (!accessible_by_all) {
    return HSA_STATUS_SUCCESS;
  }

  bool alloc_allowed = false;
  status = iree_hsa_amd_memory_pool_get_info_raw(
      query->libhsa, pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
      &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  uint32_t global_flags = 0;
  status = iree_hsa_amd_memory_pool_get_info_raw(
      query->libhsa, pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
      &global_flags);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  const bool is_kernarg =
      global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  const bool is_fine =
      global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
  const bool is_uncached =
      global_flags &
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED;
  if ((global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) &&
      !query->pools.coarse_pool.handle) {
    query->pools.coarse_pool = pool;
  }
  if (is_kernarg && !query->pools.kernarg_pool.handle) {
    query->pools.kernarg_pool = pool;
  }
  if (is_fine && (!query->pools.fine_pool.handle ||
                  (query->fine_pool_is_kernarg && !is_kernarg))) {
    query->pools.fine_pool = pool;
    query->fine_pool_is_kernarg = is_kernarg;
  }
  if (is_uncached && !query->pools.uncached_pool.handle) {
    query->pools.uncached_pool = pool;
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t iree_hal_amdgpu_host_memory_query_region(
    hsa_region_t region, void* user_data) {
  iree_hal_amdgpu_host_memory_query_t* query =
      (iree_hal_amdgpu_host_memory_query_t*)user_data;

  hsa_region_segment_t segment = 0;
  hsa_status_t status = iree_hsa_region_get_info_raw(
      query->libhsa, region, HSA_REGION_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  bool alloc_allowed = false;
  status = iree_hsa_region_get_info_raw(query->libhsa, region,
                                        HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                                        &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (!alloc_allowed) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t global_flags = 0;
  status = iree_hsa_region_get_info_raw(
      query->libhsa, region, HSA_REGION_INFO_GLOBAL_FLAGS, &global_flags);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if ((global_flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) &&
      !query->pools.fine_region.handle) {
    query->pools.fine_region = region;
  }
  return HSA_STATUS_SUCCESS;
}

iree_status_t iree_hal_amdgpu_host_memory_pools_query(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t host_agent,
    iree_hal_amdgpu_host_memory_pools_t* out_pools) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_pools, 0, sizeof(*out_pools));
  iree_hal_amdgpu_host_memory_query_t query = {.libhsa = libhsa};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_agent_iterate_memory_pools(
              IREE_LIBHSA(libhsa), host_agent,
              iree_hal_amdgpu_host_memory_query_pool, &query));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_agent_iterate_regions(
              IREE_LIBHSA(libhsa), host_agent,
              iree_hal_amdgpu_host_memory_query_region, &query));

  if (!query.pools.coarse_pool.handle) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_NOT_FOUND,
                             "no accessible-by-all + coarse-grained shared "
                             "memory pool is available in the system"));
  }
  if (!query.pools.fine_pool.handle) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_NOT_FOUND,
                             "no accessible-by-all + fine-grained shared "
                             "memory pool is available in the system"));
  }
  if (!query.pools.kernarg_pool.handle) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_NOT_FOUND,
                             "no accessible-by-all + kernarg-init shared "
                             "memory pool is available in the system"));
  }
  if (!query.pools.fine_region.handle) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_NOT_FOUND,
                             "no allocatable fine-grained shared memory region "
                             "is available in the system"));
  }

  *out_pools = query.pools;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
