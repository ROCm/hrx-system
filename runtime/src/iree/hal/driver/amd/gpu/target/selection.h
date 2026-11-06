// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_TARGET_SELECTION_H_
#define IREE_HAL_DRIVER_AMD_GPU_TARGET_SELECTION_H_

#include "iree/hal/device_spec.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"
#include "iree/hal/utils/device_spec_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Publishes the exact compiler target and its compatible code-object target
// when distinct. Artifact containers can rank exact images before generic
// images using these device-owned records without inventing temporary targets.
iree_status_t iree_hal_amd_gpu_device_spec_builder_add_executable_targets(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amd_gpu_target_identity_t* identity,
    iree_hal_physical_device_affinity_t physical_device_affinity);

// Selects a compatible native device target for an exact or generic artifact
// key. The loader's processor, physical revision, and feature-mode rules apply
// before priority and ambiguity resolution. The selected target remains the
// device's exact identity, borrowed from |device_spec| through executable load.
// Zero affinity permits any published target. Malformed artifact keys fail;
// incompatible keys return NO_MATCH without treating availability as an error.
iree_status_t iree_hal_amd_gpu_device_spec_select_executable_target(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t artifact_target_key,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_selection_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_TARGET_SELECTION_H_
