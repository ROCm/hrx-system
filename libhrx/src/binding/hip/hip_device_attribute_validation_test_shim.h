// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_HIP_DEVICE_ATTRIBUTE_VALIDATION_TEST_SHIM_H_
#define LIBHRX_SRC_BINDING_HIP_HIP_DEVICE_ATTRIBUTE_VALIDATION_TEST_SHIM_H_

#include "api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Calls the public C entry point with an arbitrary numeric attribute value.
// Keeping the conversion in C avoids constructing an out-of-range C++ enum
// object while still using the entry point's exact declared function type.
hipError_t hrx_test_hip_device_get_attribute(
    hipError_t (*function)(int* value, hipDeviceAttribute_t attribute,
                           int device),
    int* value, int attribute, int device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LIBHRX_SRC_BINDING_HIP_HIP_DEVICE_ATTRIBUTE_VALIDATION_TEST_SHIM_H_
