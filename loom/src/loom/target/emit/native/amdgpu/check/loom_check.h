// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native loom-check emit provider.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_LOOM_CHECK_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_LOOM_CHECK_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emit provider for AMDGPU native assembly fragments from target-low functions.
extern const loom_check_emit_provider_t
    loom_amdgpu_native_loom_check_emit_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_LOOM_CHECK_H_
