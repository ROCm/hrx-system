// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_CHECK_PROFILE_H_
#define LOOM_TARGET_ARCH_SPIRV_CHECK_PROFILE_H_

#include "loom/tools/loom-check/execute.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a public SPIR-V profile with named feature and limit refinements.
// Empty |target_specification| leaves the profile unspecified. Refinement
// diagnostics are appended to the request; a diagnosed rejection leaves the
// profile NULL. The caller releases a successful profile after compilation.
iree_status_t loom_spirv_check_profile_select(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t target_specification, iree_string_view_t features,
    iree_string_view_t limit_options, loomc_target_profile_t** out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_CHECK_PROFILE_H_
