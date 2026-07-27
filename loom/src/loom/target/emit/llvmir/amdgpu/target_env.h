// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM AMDGPU target environment/profile presets.

#ifndef LOOM_TARGET_LLVMIR_AMDGPU_TARGET_ENV_H_
#define LOOM_TARGET_LLVMIR_AMDGPU_TARGET_ENV_H_

#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_llvmir_target_env_t* loom_llvmir_target_env_amdgcn_amd_amdhsa(void);

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_amdgpu_hal(void);

// Returns the generic target bundle represented by the AMDGPU HAL fixture.
const loom_target_bundle_t* loom_llvmir_target_bundle_amdgpu_hal(void);

// Initializes |out_profile| with a shallow copy of the built-in AMDGPU HAL
// profile. String views and |target_env| point at immutable static storage;
// callers may overwrite profile fields for one lowering invocation.
iree_status_t loom_llvmir_target_profile_initialize_amdgpu_hal(
    loom_llvmir_target_profile_t* out_profile);

// Returns the static profile provider for AMDGPU LLVMIR target presets.
const loom_llvmir_target_profile_provider_t*
loom_llvmir_amdgpu_target_profile_provider(void);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_AMDGPU_TARGET_ENV_H_
