// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU diagnostics for descriptor-backed text low assembly.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOW_ASM_DIAGNOSTICS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOW_ASM_DIAGNOSTICS_H_

#include "loom/target/low_asm_diagnostics.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_target_low_asm_diagnostic_provider_t
    loom_amdgpu_low_asm_diagnostic_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOW_ASM_DIAGNOSTICS_H_
