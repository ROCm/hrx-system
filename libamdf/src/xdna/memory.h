// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_MEMORY_H_
#define AMDF_SRC_XDNA_MEMORY_H_

#include "amdf/amdf.h"
#include "libamdf/src/memory_profile.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/xdna/umd/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Copies one XDNA memory profile, or reports it unsupported.
amdf_status_t amdf_xdna_device_query_memory_profile(
    amdf_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile);

// Initializes a borrowed private descriptor in its live context owner.
void amdf_xdna_memory_scope_initialize(amdf_device_t* device,
                                       amdf_xdna_umd_context_t* context,
                                       amdf_memory_scope_t* scope);

// Prepares native state in the common memory owner, including on failure.
amdf_status_t amdf_xdna_memory_prepare(
    amdf_memory_t* memory, const amdf_memory_native_group_t* group,
    const amdf_memory_native_create_info_t* create_info,
    amdf_memory_info_t* out_info);

// Acquires an independent native backing reference without consuming input.
amdf_status_t amdf_xdna_memory_prepare_import(
    amdf_memory_t* memory, uint32_t access_ordinal,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_memory_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_MEMORY_H_
