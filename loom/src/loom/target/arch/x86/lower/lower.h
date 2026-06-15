// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 source-to-target-low lowering policy.
//
// The x86 arch package owns descriptor-level lowering decisions that should be
// shared by native assembly, LLVMIR, and future direct object emitters.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_LOWER_H_
#define LOOM_TARGET_ARCH_X86_LOWER_LOWER_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the x86 AVX512 register lowering policy.
//
// The policy maps object-function buffer arguments and address-domain scalar
// values to GPR64 registers, maps vector<16xi32>/vector<16xf32> values to ZMM
// registers, lowers direct vector arithmetic through table-driven AVX512
// descriptor rules, and lowers static ZMM load/store accesses through base plus
// disp32 memory packets.
const loom_low_lower_policy_t* loom_x86_avx512_low_lower_policy(void);

// Returns the x86 AVX2 register lowering policy.
//
// The policy maps object-function buffer arguments and address-domain scalar
// values to GPR64 registers, maps ordinary integer scalar values to GPR
// registers, maps scalar f32 and vector<4xi32>/vector<4xf32> values to XMM
// registers, and lowers scalar plus XMM operations through table-driven AVX2
// descriptor rules. AVX512 mask and ZMM values are intentionally not legal in
// this policy.
const loom_low_lower_policy_t* loom_x86_avx2_low_lower_policy(void);

// Returns the baseline x86 scalar register lowering policy.
//
// The policy maps object-function buffer arguments and address-domain scalar
// values to GPR64 registers, maps i32 scalar values to GPR32 registers, and
// lowers ordinary integer/address operations through scalar descriptor rules.
const loom_low_lower_policy_t* loom_x86_scalar_low_lower_policy(void);

// Returns the x86 packed-dot register lowering policy.
//
// The policy maps supported static dot vectors to XMM/YMM/ZMM registers and
// lowers target-legal vector.dot2f/vector.dot4i ops through descriptor-selected
// packed-dot packets.
const loom_low_lower_policy_t* loom_x86_packed_dot_low_lower_policy(void);

// Initializes a target-owned registry mapping x86 target-contract keys to
// their source-to-low lowering policies.
void loom_x86_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_LOWER_H_
