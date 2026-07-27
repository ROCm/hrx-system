// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM source-to-target-low lowering policy.
//
// This package-local policy plugs the generic source-to-low lowerer into the
// ireevm.core descriptor set. Keeping the policy here prevents VM descriptor
// knowledge from leaking into the target-independent codegen/low layer.

#ifndef LOOM_TARGET_ARCH_IREEVM_LOWER_LOWER_H_
#define LOOM_TARGET_ARCH_IREEVM_LOWER_LOWER_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the package-local lowering policy for VM scalar source ops.
//
// The policy maps i1/i32/i64/f32/f64 source values to the corresponding
// ireevm register classes. Unsupported source ops are rejected through
// structured target-low diagnostics instead of producing partial low IR.
const loom_low_lower_policy_t* loom_ireevm_low_lower_policy(void);

// Initializes a target-owned registry mapping IREE VM target-contract keys to
// their source-to-low lowering policies.
void loom_ireevm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_IREEVM_LOWER_LOWER_H_
