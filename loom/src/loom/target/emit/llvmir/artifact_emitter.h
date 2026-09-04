// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVMIR target artifact emitters.
//
// This package contributes prepared-target-low emitters for LLVMIR text and
// bitcode artifacts. It does not register the LLVMIR dialect or lowering
// descriptors by itself; compose it with loom_llvmir_target_provider when a
// tool wants both LLVMIR target IR support and artifact emission.

#ifndef LOOM_TARGET_EMIT_LLVMIR_ARTIFACT_EMITTER_H_
#define LOOM_TARGET_EMIT_LLVMIR_ARTIFACT_EMITTER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Provider contribution containing LLVMIR text and bitcode artifact emitters.
extern const loom_target_provider_t loom_llvmir_artifact_emitter_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_LLVMIR_ARTIFACT_EMITTER_H_
