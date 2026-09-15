// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_MEMORY_PROFILE_H_
#define AMDF_SRC_XDNA_UMD_MCDM_MEMORY_PROFILE_H_

#include "libamdf/src/memory_profile.h"
#include "libamdf/src/xdna/endpoint_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Standard host backing is rounded to the Windows allocation granularity.
#define AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY UINT64_C(65536)
// KMT chooses a GPU VA on CPU-page boundaries, independently of backing size.
#define AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT UINT64_C(4096)

// Derives complete memory capabilities from qualified target facts.
// This metadata query performs no native operation or device activation.
amdf_status_t amdf_windows_xdna_query_memory_profile(
    const amdf_xdna_endpoint_profile_t* target, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_MCDM_MEMORY_PROFILE_H_
