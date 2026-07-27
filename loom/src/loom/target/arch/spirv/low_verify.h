// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V exact target-low verification.
//
// Generic target-low verification proves descriptor shape, register classes,
// features, and register-part use. SPIR-V also needs descriptor-local payload
// typing for `spirv.id` values and ABI materialization constraints that are not
// represented by the generic register class. This provider runs inside the
// generic low verifier's function walk so those checks stay upstream of binary
// emission without adding another module pass.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOW_VERIFY_H_
#define LOOM_TARGET_ARCH_SPIRV_LOW_VERIFY_H_

#include "loom/codegen/low/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-owned low verification provider for SPIR-V logical-core functions.
extern const loom_low_verify_provider_t loom_spirv_low_verify_provider;

// Returns the SPIR-V low verification provider list.
loom_low_verify_provider_list_t loom_spirv_low_verify_provider_list(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOW_VERIFY_H_
