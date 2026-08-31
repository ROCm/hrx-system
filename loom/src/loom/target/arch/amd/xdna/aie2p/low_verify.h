// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P target-low product-contract verification.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOW_VERIFY_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOW_VERIFY_H_

#include "loom/codegen/low/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-owned Low verifier for AIE2P array program functions.
extern const loom_low_verify_provider_t loom_aie2p_low_verify_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOW_VERIFY_H_
