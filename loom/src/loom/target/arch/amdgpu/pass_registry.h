// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-owned pass registry.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PASS_REGISTRY_H_
#define LOOM_TARGET_ARCH_AMDGPU_PASS_REGISTRY_H_

#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sorted pass registry contributed by the AMDGPU target provider.
extern const loom_pass_registry_t loom_amdgpu_pass_registry;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PASS_REGISTRY_H_
