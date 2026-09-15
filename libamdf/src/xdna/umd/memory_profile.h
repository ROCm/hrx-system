// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MEMORY_PROFILE_H_
#define AMDF_SRC_XDNA_UMD_MEMORY_PROFILE_H_

#include "libamdf/src/memory_profile.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/xdna/endpoint_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Derives an expected native memory contract from cached endpoint/host facts.
// No native query, allocation, mapping or device activation occurs. Failure
// leaves caller-private result storage unchanged.
amdf_status_t amdf_xdna_umd_query_endpoint_memory_profile(
    const amdf_platform_endpoint_t* endpoint,
    const amdf_xdna_endpoint_profile_t* target, uint32_t profile_ordinal,
    amdf_memory_native_profile_t* out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_MEMORY_PROFILE_H_
