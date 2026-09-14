// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hip_device_attribute_validation_test_shim.h"

hipError_t hrx_test_hip_device_get_attribute(
    hipError_t (*function)(int* value, hipDeviceAttribute_t attribute,
                           int device),
    int* value, int attribute, int device) {
  return function(value, (hipDeviceAttribute_t)attribute, device);
}
