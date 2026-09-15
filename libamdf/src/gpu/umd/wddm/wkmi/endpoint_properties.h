// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_WKMI_ENDPOINT_PROPERTIES_H_
#define AMDF_SRC_GPU_UMD_WDDM_WKMI_ENDPOINT_PROPERTIES_H_

#include <stdbool.h>

#include "libamdf/src/gpu/endpoint_profile.h"
#include "libamdf/src/gpu/umd/wddm/wkmi/bridge_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Translates one WKMI result into provider-neutral GPU endpoint properties.
bool amdf_gpu_wddm_wkmi_endpoint_properties_translate(
    const amdf_wkmi_bridge_gpu_properties_t* provider_properties,
    amdf_gpu_endpoint_properties_t* out_properties);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_WDDM_WKMI_ENDPOINT_PROPERTIES_H_
