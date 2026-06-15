// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVMIR generic source-to-target-low lowering policy.

#ifndef LOOM_TARGET_ARCH_LLVMIR_LOWER_LOWER_H_
#define LOOM_TARGET_ARCH_LLVMIR_LOWER_LOWER_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the generic LLVMIR object-function lowering policy.
//
// The policy maps object-function buffer/view arguments to pointer registers,
// maps scalar and v4 source values to typed LLVMIR virtual registers, and
// lowers source arithmetic, select, compare, and view memory operations through
// descriptor-backed LLVMIR oracle packets.
const loom_low_lower_policy_t* loom_llvmir_generic_low_lower_policy(void);

// Initializes a target-owned registry mapping LLVMIR target-contract keys to
// their source-to-low lowering policies.
void loom_llvmir_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_LLVMIR_LOWER_LOWER_H_
