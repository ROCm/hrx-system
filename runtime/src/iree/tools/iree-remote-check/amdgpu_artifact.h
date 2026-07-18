// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLS_IREE_REMOTE_CHECK_AMDGPU_ARTIFACT_H_
#define IREE_TOOLS_IREE_REMOTE_CHECK_AMDGPU_ARTIFACT_H_

#include "iree/tools/iree-remote-check/artifact_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Selects an embedded AMDGPU artifact, leaving |out_artifact| empty when the
// device has no compatible AMDGPU target or code object.
iree_status_t iree_remote_check_select_amdgpu_artifact(
    const iree_hal_device_spec_t* device_spec,
    iree_remote_check_artifact_t* out_artifact);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLS_IREE_REMOTE_CHECK_AMDGPU_ARTIFACT_H_
