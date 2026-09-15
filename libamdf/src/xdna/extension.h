// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_EXTENSION_H_
#define AMDF_SRC_XDNA_EXTENSION_H_

#include "amdf/amdf.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/xdna/endpoint_profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Acquires the XDNA extension table for one previously validated version range.
amdf_status_t amdf_xdna_extension_query(uint32_t minimum_version,
                                        uint32_t maximum_version,
                                        const void** out_extension_api);

// Installs metadata queries that consume the endpoint's selected target facts.
void amdf_xdna_extension_initialize_endpoint(amdf_endpoint_t* endpoint);

// Writes the XDNA queue families fully constructible for an opened endpoint.
uint32_t amdf_xdna_extension_query_endpoint_queue_families(
    const amdf_xdna_endpoint_profile_t* profile,
    const amdf_platform_endpoint_t* platform_endpoint, uint32_t capacity,
    amdf_queue_family_info_t* out_families);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_EXTENSION_H_
