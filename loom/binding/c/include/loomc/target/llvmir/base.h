// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_LLVMIR_BASE_H_
#define LOOMC_TARGET_LLVMIR_BASE_H_

#include "loomc/target.h"

/// @file
/// LLVMIR target package identity and environment creation.
///
/// This header is the dependency-light entry point for embedders that want to
/// link the LLVMIR oracle/debug target package. It does not require LLVM
/// libraries or command-line tools; it emits textual LLVM IR or LLVM bitcode
/// from prepared LLVMIR target-low modules.

#ifdef __cplusplus
extern "C" {
#endif

/// Textual LLVM IR artifact format.
#define LOOMC_ARTIFACT_FORMAT_LLVMIR_TEXT "llvmir-text"

/// LLVM bitcode artifact format.
#define LOOMC_ARTIFACT_FORMAT_LLVMIR_BITCODE "llvmir-bitcode"

/// Creates a target environment containing the LLVMIR target package.
///
/// @param allocator Host allocator used for target-environment storage.
/// @param out_target_environment Receives one retained target environment on
/// success.
/// @return OK when the target environment was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_target_environment_release`.
LOOMC_API_EXPORT loomc_status_t loomc_target_environment_create_llvmir(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_LLVMIR_BASE_H_
