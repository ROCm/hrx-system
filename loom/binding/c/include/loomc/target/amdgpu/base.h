// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_AMDGPU_BASE_H_
#define LOOMC_TARGET_AMDGPU_BASE_H_

#include "loomc/target.h"

/// @file
/// AMDGPU target package identity and environment creation.
///
/// This header is the dependency-light entry point for embedders that want to
/// link the AMDGPU target package. It does not include HSA, HIP, ROCm, or IREE
/// HAL headers. Live adapters should query their runtime independently,
/// normalize the selected processor name, and pass that name to the AMDGPU
/// profile API.

#ifdef __cplusplus
extern "C" {
#endif

/// AMDGPU HSA code object artifact format.
#define LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO "amdgpu-hsaco"

/// Creates a target environment containing the AMDGPU target package.
///
/// @param allocator Host allocator used for target-environment storage.
/// @param out_target_environment Receives one retained target environment on
/// success.
/// @return OK when the target environment was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_target_environment_release`.
LOOMC_API_EXPORT loomc_status_t loomc_target_environment_create_amdgpu(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMDGPU_BASE_H_
