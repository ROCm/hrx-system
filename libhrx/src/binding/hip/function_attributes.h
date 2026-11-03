// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_FUNCTION_ATTRIBUTES_H_
#define HRX_BINDING_HIP_FUNCTION_ATTRIBUTES_H_

#include "binding/hip/api.h"
#include "common/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries the cached facts of a loaded function. Missing resource facts return
// hipErrorNotSupported. AMD HIP represents unavailable constant-memory usage as
// -1. Failure leaves |out_value| unchanged.
hipError_t iree_hip_function_get_attribute(
    const iree_hal_streaming_symbol_t* symbol, hipFuncAttribute_t attribute,
    int* out_value);

// Queries all cached function attributes, including current mutable settings.
// AMD HIP represents unavailable constant-memory usage as SIZE_MAX in this
// structure. Failure leaves |out_attributes| unchanged.
hipError_t iree_hip_function_get_attributes(
    const iree_hal_streaming_symbol_t* symbol,
    hipFuncAttributes* out_attributes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // HRX_BINDING_HIP_FUNCTION_ATTRIBUTES_H_
