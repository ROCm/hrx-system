// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm target-low verification.
//
// Generic target-low verification proves descriptor shape, register classes,
// features, and register-part use. Wasm also has structural control constraints
// that are properties of the binary target, not the byte emitter. This provider
// keeps those constraints upstream of Wasm emission.

#ifndef LOOM_TARGET_ARCH_WASM_LOW_VERIFY_H_
#define LOOM_TARGET_ARCH_WASM_LOW_VERIFY_H_

#include "loom/codegen/low/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-owned low verification provider for wasm.core.simd128 functions.
extern const loom_low_verify_provider_t loom_wasm_low_verify_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_WASM_LOW_VERIFY_H_
