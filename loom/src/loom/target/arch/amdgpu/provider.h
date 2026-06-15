// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target provider.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PROVIDER_H_
#define LOOM_TARGET_ARCH_AMDGPU_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// AMDGPU target dialect, descriptor, lowering, and diagnostic capabilities.
extern const loom_target_provider_t loom_amdgpu_target_provider;

// Provider set containing only the AMDGPU target provider.
extern const loom_target_provider_set_t loom_amdgpu_target_provider_set;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PROVIDER_H_
