// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_TARGET_SELECTION_H_
#define IREE_HAL_DRIVERS_AMDGPU_TARGET_SELECTION_H_

#include "iree/hal/drivers/amdgpu/target/identity.h"
#include "iree/hal/utils/device_spec_builder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Adds canonical exact and compatible generic AMDGPU executable targets for
// |exact_identity| to |builder|.
//
// The generic target is omitted when the target map requires exact code
// objects. Repeated targets are merged by the device spec builder.
iree_status_t iree_hal_amdgpu_device_spec_builder_add_executable_targets(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_amdgpu_target_identity_t* exact_identity,
    iree_hal_physical_device_affinity_t physical_device_affinity);

// Selects the device target compatible with an AMDGPU executable artifact.
//
// |artifact_target_key| identifies the processor and optional feature modes
// used to compile the artifact. Exact artifacts select exact device targets
// and generic artifacts select generic device targets. Featureless artifact
// keys are compatible with any device feature mode, while explicit feature
// modes must match.
//
// A zero |physical_device_affinity| accepts any advertised device target.
// Nonzero affinities require the selected target to cover every requested
// physical device. No-match and ambiguous outcomes are returned in
// |out_result| without creating a status; malformed AMDGPU target keys fail.
iree_status_t iree_hal_amdgpu_device_spec_select_executable_target(
    const iree_hal_device_spec_t* device_spec,
    iree_string_view_t artifact_target_key,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    iree_hal_executable_target_selection_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_TARGET_SELECTION_H_
