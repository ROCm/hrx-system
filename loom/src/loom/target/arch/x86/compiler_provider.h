// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 native compiler capability composition.

#ifndef LOOM_TARGET_ARCH_X86_COMPILER_PROVIDER_H_
#define LOOM_TARGET_ARCH_X86_COMPILER_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Composes the native object emitter with x86 target identity. Source dialect,
// profiles, and lowering remain separately selectable through provider.h.
extern const loom_target_provider_t loom_x86_compiler_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_COMPILER_PROVIDER_H_
