// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_BASE_H_
#define LOOMC_TARGET_SPIRV_BASE_H_

#include "loomc/target.h"

/// @file
/// SPIR-V target package identity and environment creation.
///
/// This header is the dependency-light entry point for embedders that want to
/// link the SPIR-V target package. It does not include Vulkan, IREE HAL, or any
/// saved-profile importer dependencies. Provider-specific adapters live under
/// separate SPIR-V target leaves so an offline compiler or cross-compiler can
/// use SPIR-V support without platform headers.

#ifdef __cplusplus
extern "C" {
#endif

/// SPIR-V binary artifact format.
#define LOOMC_ARTIFACT_FORMAT_SPIRV "spirv"

/// Creates a target environment containing the SPIR-V target package.
///
/// @param allocator Host allocator used for target-environment storage.
/// @param out_target_environment Receives one retained target environment on
/// success.
/// @return OK when the target environment was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_target_environment_release`.
LOOMC_API_EXPORT loomc_status_t loomc_target_environment_create_spirv(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_SPIRV_BASE_H_
