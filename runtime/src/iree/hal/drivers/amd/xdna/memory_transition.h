// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_TRANSITION_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_TRANSITION_H_

#include "amdf/memory.h"
#include "iree/hal/memory_scope.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Translates one successful libamdf pair query into the HAL's immutable pair
// vocabulary. This validates the native result at the API boundary and
// preserves unqualified UNKNOWN sides without inventing a fallback action.
iree_status_t iree_hal_amd_xdna_memory_translate_pair(
    const amdf_memory_pair_info_t* source,
    iree_hal_memory_pair_info_t* out_pair);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_MEMORY_TRANSITION_H_
