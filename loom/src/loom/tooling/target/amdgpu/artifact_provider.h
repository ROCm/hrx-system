// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL artifact provider for Loom execution tools.

#ifndef LOOM_TOOLING_EXECUTION_HAL_AMDGPU_ARTIFACT_PROVIDER_H_
#define LOOM_TOOLING_EXECUTION_HAL_AMDGPU_ARTIFACT_PROVIDER_H_

#include "loom/tooling/execution/hal/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

// AMDGPU artifact provider for HAL device targeting and HSACO packaging.
extern const loom_run_hal_artifact_provider_t loom_amdgpu_hal_artifact_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_AMDGPU_ARTIFACT_PROVIDER_H_
