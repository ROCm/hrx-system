// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU device projection for live Loom execution tools.

#ifndef LOOM_TOOLING_TARGET_AMDGPU_DEVICE_PROVIDER_H_
#define LOOM_TOOLING_TARGET_AMDGPU_DEVICE_PROVIDER_H_

#include "loom/tooling/execution/hal/device_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects an AMDGPU HAL device into the offline AMDGPU artifact provider.
extern const loom_device_provider_t loom_amdgpu_device_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_AMDGPU_DEVICE_PROVIDER_H_
