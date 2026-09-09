// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_AMD_XDNA_BASE_H_
#define LOOMC_TARGET_AMD_XDNA_BASE_H_

#include "loomc/target.h"

/// @file
/// AMD XDNA compiler package identity and environment creation.
///
/// The package lowers kernel-scoped pipelines to tile and array functions in
/// one module. After the ordinary prepared-low target pipeline, generic
/// `loomc_emit_module` emits one canonical ELF32 `.xdna` byte sequence
/// containing all resident tile code, array configuration, control code, and
/// entry metadata. Compilation and emission require neither a live device nor a
/// vendor runtime.

#ifdef __cplusplus
extern "C" {
#endif

/// Canonical Loom-owned XDNA ELF32 executable format.
#define LOOMC_ARTIFACT_FORMAT_XDNA "xdna"

/// Creates a target environment containing the AIE2P XDNA compiler and emitter.
///
/// @param allocator Host allocator used for target-environment storage.
/// @param out_target_environment Receives one retained environment on success.
/// @return OK when the target environment was created.
///
/// @ownership
/// The caller releases the returned reference with
/// `loomc_target_environment_release`. The environment can be shared by
/// compiler instances and workspaces across JIT invocations.
LOOMC_API_EXPORT loomc_status_t loomc_target_environment_create_xdna(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMD_XDNA_BASE_H_
