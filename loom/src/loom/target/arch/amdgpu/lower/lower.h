// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-target-low lowering policy.
//
// This policy owns the architecture-specific mapping from target-legal source
// Loom ops to AMDGPU descriptor-backed low ops. It deliberately stays below
// native assembly and LLVMIR emission so the same low representation can feed
// every later AMDGPU backend path.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_LOWER_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_LOWER_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the AMDGPU scalar/VGPR lowering policy.
//
// The initial policy is intentionally narrow: it maps i32 scalar values to
// SGPRs, maps 32-bit vector lane values to VGPRs, and lowers
// scalar.constant/scalar.addi/vector.constant/vector.addi/vector.muli to
// descriptor-backed low packets. HAL buffer arguments lower to explicit
// entry low.resource imports.
// Memory operations and wider vector packing require explicit view/resource
// layout contracts and are not guessed here.
const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void);

// Returns the AMDGPU source legality provider for target-low lowering.
const loom_target_low_legality_provider_t* loom_amdgpu_low_legality_provider(
    void);

// Storage for static provider-list construction.
extern const loom_target_low_legality_provider_t
    loom_amdgpu_low_legality_provider_storage;

// Initializes a target-owned registry mapping AMDGPU target-contract keys to
// their source-to-low lowering policies.
void loom_amdgpu_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_LOWER_H_
