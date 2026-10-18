// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CONFIGURED_IR_H_
#define LOOMC_TARGET_CONFIGURED_IR_H_

#include "loomc/target.h"

/// @file
/// Build-configured target IR representation support.
///
/// Link the optional `//loom/binding/c/target/configured:ir` Bazel package or
/// `loom::binding::c::target::configured::ir` CMake target to use this factory.
/// It joins portable command IR and all enabled target representations without
/// their lowering, pass, legalization, or artifact-emission implementations.
/// The configured target package adds transformation support, and the separate
/// configured compiler package joins artifact-capable targets. Clients needing
/// one target family can continue selecting that family's package directly.

#ifdef __cplusplus
extern "C" {
#endif

/// Creates an environment supporting every configured target IR family.
///
/// The environment supplies dialects, target facts, Low descriptor registries,
/// and source diagnostics for parsing, structural verification, and printing.
/// Construction does not probe devices or load platform runtimes.
///
/// @param allocator Host allocator used for target-environment storage.
/// @param out_target_environment Receives one retained environment on success.
/// @return OK when the environment was created.
///
/// @ownership
/// The caller releases the returned reference with
/// `loomc_target_environment_release`. Contexts retain their own reference.
LOOMC_API_EXPORT loomc_status_t loomc_target_environment_create_configured_ir(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CONFIGURED_IR_H_
