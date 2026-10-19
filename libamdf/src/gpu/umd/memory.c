// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory.h"

#include "amdf/gpu.h"

amdf_status_t amdf_gpu_umd_memory_describe_site(
    const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  const amdf_queue_family_info_t* family = query->queue_family_info;
  const bool sdma_format =
      family->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
      family->format_version == AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1;
  // CPU visibility does not change the engine's payload cache protocol. Host
  // aperture publication, including HDP maintenance, belongs to the separately
  // queried CPU edge and must precede a consuming queue's acquire.
  const bool scoped_sdma =
      sdma_format && (family->roles & AMDF_QUEUE_ROLE_TRANSFER) != 0 &&
      (family->format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) !=
          0;
  const bool system_cache_format =
      (family->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
       family->format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1) ||
      (family->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL &&
       family->format_version == AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1) ||
      (sdma_format && ((family->format_features &
                        AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0 ||
                       ((family->format_features &
                         AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR) != 0 &&
                        (family->publication_modes &
                         AMDF_QUEUE_PUBLICATION_MODE_KERNEL) != 0)));
  const bool system_cache_operations =
      (family->roles & AMDF_QUEUE_ROLE_CACHE_CONTROL) != 0 &&
      (family->cache_operations &
       (AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
        AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM)) ==
          (AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
           AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) &&
      (family->cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL) !=
          0;
  if (!scoped_sdma && !(system_cache_format && system_cache_operations)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_memory_site_description_t description = {0};
  if ((query->access & AMDF_MEMORY_ACCESS_READ) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((query->access & AMDF_MEMORY_ACCESS_WRITE) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  if (scoped_sdma) {
    // SYS scope belongs to each data command. It needs no separate payload
    // cache operation and makes no claim about atomic reach or completion.
    description.release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    description.acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN |
                                AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN;
  } else {
    description.release = (amdf_cache_transition_t){
        .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
        .executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
        .operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
    };
    description.acquire = (amdf_cache_transition_t){
        .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
        .executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
        .operation = AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
    };
  }
  *out_description = description;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_memory_describe_system_store_site(
    const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  amdf_memory_site_description_t description = {0};
  const amdf_status_t status =
      amdf_gpu_umd_memory_describe_site(query, &description);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_queue_family_info_t* family = query->queue_family_info;
  if ((query->flags &
       (AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_LOCAL)) ==
          AMDF_MEMORY_FLAG_HOST_COHERENT &&
      (query->access & (AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE)) ==
          (AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE) &&
      family->command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
      (family->roles & AMDF_QUEUE_ROLE_ATOMIC) != 0) {
    if ((query->atomic_operations_32 &
         family->atomic_capabilities.operations_32 &
         AMDF_ATOMIC_OPERATION_STORE) != 0) {
      description.atomic_reach.scope_32 = AMDF_ATOMIC_SCOPE_SYSTEM;
    }
    if ((query->atomic_operations_64 &
         family->atomic_capabilities.operations_64 &
         AMDF_ATOMIC_OPERATION_STORE) != 0) {
      description.atomic_reach.scope_64 = AMDF_ATOMIC_SCOPE_SYSTEM;
    }
    if (description.atomic_reach.scope_32 != AMDF_ATOMIC_SCOPE_NONE ||
        description.atomic_reach.scope_64 != AMDF_ATOMIC_SCOPE_NONE) {
      // Host-system serialization domain. Pair assembly separately proves
      // that both sites access the same backing.
      description.atomic_domain.words[0] = UINT64_C(0x43505553544f5245);
    }
  }
  *out_description = description;
  return AMDF_STATUS_OK;
}
