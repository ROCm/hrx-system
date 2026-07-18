// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLS_IREE_REMOTE_CHECK_ARTIFACT_H_
#define IREE_TOOLS_IREE_REMOTE_CHECK_ARTIFACT_H_

#include "iree/tools/iree-remote-check/artifact_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Selects an embedded native artifact compatible with |device_spec|.
//
// The returned views borrow immutable device-spec or process-lifetime embedded
// storage and remain valid while |device_spec| is retained.
iree_status_t iree_remote_check_select_artifact(
    const iree_hal_device_spec_t* device_spec,
    iree_remote_check_artifact_t* out_artifact);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLS_IREE_REMOTE_CHECK_ARTIFACT_H_
