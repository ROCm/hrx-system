// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVMIR target profiles selected by the Loom build configuration.

#ifndef LOOM_TARGET_EMIT_LLVMIR_CONFIGURED_TARGET_PROFILES_H_
#define LOOM_TARGET_EMIT_LLVMIR_CONFIGURED_TARGET_PROFILES_H_

#include "loom/target/emit/llvmir/target_presets.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable process-lifetime registry of configured LLVMIR target profiles.
extern const loom_llvmir_target_profile_registry_t
    loom_llvmir_configured_target_profile_registry;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_LLVMIR_CONFIGURED_TARGET_PROFILES_H_
