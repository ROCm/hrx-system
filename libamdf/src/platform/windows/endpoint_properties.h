// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_PROPERTIES_H_
#define AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_PROPERTIES_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/platform/windows/kmt_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Queries the number of independently addressable physical adapters.
amdf_status_t amdf_windows_query_physical_adapter_count(
    const amdf_kmt_api_t* api, D3DKMT_HANDLE adapter, uint32_t* out_count);

// Queries and normalizes immutable properties of one physical adapter.
amdf_status_t amdf_windows_query_endpoint_info(const amdf_kmt_api_t* api,
                                               D3DKMT_HANDLE adapter,
                                               LUID adapter_luid,
                                               uint32_t physical_adapter_index,
                                               amdf_endpoint_info_t* out_info);

// Decodes the native location carried by an opaque Windows endpoint ID.
void amdf_windows_endpoint_id_decode(const amdf_endpoint_id_t* id,
                                     LUID* out_adapter_luid,
                                     uint32_t* out_physical_adapter_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_WINDOWS_ENDPOINT_PROPERTIES_H_
