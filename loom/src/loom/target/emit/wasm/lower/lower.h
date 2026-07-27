// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm source-to-target-low lowering policy.
//
// This package-local policy plugs the generic source-to-low lowerer into the
// wasm.core.simd128 descriptor set. Keeping the policy here prevents Wasm
// descriptor knowledge from leaking into the target-independent codegen/low
// layer.

#ifndef LOOM_TARGET_EMIT_WASM_LOWER_LOWER_H_
#define LOOM_TARGET_EMIT_WASM_LOWER_LOWER_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the package-local lowering policy for the Wasm scalar and SIMD
// subset.
//
// The policy maps i32/index/offset source values to reg<wasm.i32>, f32 source
// values to reg<wasm.f32>, and vector<4xi32>/vector<4xi1>/vector<4xf32> source
// values to reg<wasm.v128>. It lowers the scalar and fixed-width SIMD
// arithmetic subset described by the Wasm descriptor tables.
// Unsupported source ops are rejected through structured target-low diagnostics
// instead of producing partial low IR.
const loom_low_lower_policy_t* loom_wasm_low_lower_policy(void);

// Initializes a target-owned registry mapping Wasm target-contract keys to
// their source-to-low lowering policies.
void loom_wasm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_LOWER_LOWER_H_
