// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V/Vulkan HAL artifact provider.
//
// This provider contributes only the target-specific edges required by the
// shared HAL execution path: Vulkan device/profile projection and SPIR-V binary
// emission from prepared target-low IR. HAL loading, invocation, correctness,
// and measurement remain owned by the shared execution layer.

#ifndef LOOM_TOOLING_EXECUTION_HAL_SPIRV_ARTIFACT_PROVIDER_H_
#define LOOM_TOOLING_EXECUTION_HAL_SPIRV_ARTIFACT_PROVIDER_H_

#include "loom/tooling/execution/hal/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

// Artifact provider for Vulkan SPIR-V raw-BDA HAL kernels.
extern const loom_run_hal_artifact_provider_t
    loom_spirv_vulkan_hal_artifact_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_SPIRV_ARTIFACT_PROVIDER_H_
