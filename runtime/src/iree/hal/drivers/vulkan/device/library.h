// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_DEVICE_LIBRARY_H_
#define IREE_HAL_DRIVERS_VULKAN_DEVICE_LIBRARY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the checked SPIR-V module named |file_name|.
//
// All callers use names produced by the device library build. A missing module
// is a build invariant violation rather than a recoverable runtime condition.
iree_const_byte_span_t iree_hal_vulkan_device_library_lookup(
    iree_string_view_t file_name);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_DEVICE_LIBRARY_H_
