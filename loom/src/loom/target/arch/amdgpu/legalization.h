// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source legalization provider.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LEGALIZATION_H_
#define LOOM_TARGET_ARCH_AMDGPU_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns AMDGPU target-owned source legalizers.
const loom_target_legalizer_provider_t* loom_amdgpu_target_legalizer_provider(
    void);

// Storage for static provider-list construction.
extern const loom_target_legalizer_provider_t
    loom_amdgpu_target_legalizer_provider_storage;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LEGALIZATION_H_
