// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_TRANSITION_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_TRANSITION_H_

#include "amdf/memory.h"
#include "iree/hal/memory_scope.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Translates native host publication ordering at the API boundary.
iree_status_t iree_hal_amd_fabric_memory_translate_host_fence(
    amdf_host_cache_fence_t source, iree_hal_host_cache_fence_t* out_fence);

// Translates one successful libamdf pair query into the HAL's immutable pair
// vocabulary. This validates the native result at the API boundary and
// preserves unqualified UNKNOWN sides without inventing a fallback action.
iree_status_t iree_hal_amd_fabric_memory_translate_pair(
    const amdf_memory_pair_info_t* source,
    iree_hal_memory_pair_info_t* out_pair);

// Qualifies a validated pair for copy-engine handoff using global queue cache
// controls. Ranged or program-side recipes remain explicitly unsupported.
iree_hal_memory_transition_t iree_hal_amd_fabric_memory_queue_transition(
    const iree_hal_memory_pair_info_t* pair);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_MEMORY_TRANSITION_H_
