// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V source-to-low lowering policy.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_LOWER_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_LOWER_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the SPIR-V logical source-to-low lowering policy.
const loom_low_lower_policy_t* loom_spirv_low_lower_policy(void);

// Initializes a registry containing SPIR-V source-to-low lowering policies.
void loom_spirv_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_LOWER_H_
