// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target profiles parsed from canonical artifact target keys.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ARTIFACT_PROFILE_H_
#define LOOM_TARGET_ARCH_AMDGPU_ARTIFACT_PROFILE_H_

#include "iree/hal/executable/amdgpu/target_id.h"
#include "loom/target/arch/amdgpu/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parses |target_key| into a structured AMDGPU target profile.
//
// |target_key| may name an exact target, a generic target, or a target overlay
// and may carry canonical AMDHSA feature suffixes. |out_target_kind| receives
// the exact/generic class when non-NULL. The returned profile only borrows
// immutable generated target metadata and remains valid for the process
// lifetime.
iree_status_t loom_amdgpu_artifact_target_profile_parse(
    iree_string_view_t target_key, loom_amdgpu_target_profile_t* out_profile,
    iree_hal_amdgpu_target_kind_t* out_target_kind);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ARTIFACT_PROFILE_H_
